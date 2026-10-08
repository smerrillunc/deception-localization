from __future__ import annotations

import copy
import json
import logging
import os
import random
import re
import tempfile
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

try:
    from vllm import SamplingParams
except Exception:
    SamplingParams = None

from reasoning_parser import (
    extract_reasoning_trace,
    is_ministral3_family,
    strip_reasoning_trace,
)


@lru_cache(maxsize=16)
def _load_ministral_system_prompt(repo_id: str, filename: str = "SYSTEM_PROMPT.txt") -> Dict[str, Any]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=repo_id, filename=filename)
    system_prompt = Path(path).read_text(encoding="utf-8")

    begin = system_prompt.find("[THINK]")
    end = system_prompt.find("[/THINK]")
    if begin == -1 or end == -1:
        return {"role": "system", "content": system_prompt}

    return {
        "role": "system",
        "content": [
            {"type": "text", "text": system_prompt[:begin]},
            {
                "type": "thinking",
                "thinking": system_prompt[begin + len("[THINK]"): end],
                "closed": True,
            },
            {"type": "text", "text": system_prompt[end + len("[/THINK]"): ]},
        ],
    }


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                if item.strip():
                    parts.append(item)
                continue
            if not isinstance(item, dict):
                continue
            if isinstance(item.get("text"), str) and item.get("text").strip():
                parts.append(item["text"])
            elif isinstance(item.get("thinking"), str) and item.get("thinking").strip():
                parts.append(item["thinking"])
        return "\n".join(parts).strip()
    if content is None:
        return ""
    return str(content)


def _is_structured_ministral_system_content(content: Any) -> bool:
    if not isinstance(content, list):
        return False
    for item in content:
        if isinstance(item, dict) and item.get("type") == "thinking":
            return True
    return False


def _prepend_text_to_first_user_message(messages: List[Dict[str, Any]], text: str) -> List[Dict[str, Any]]:
    out = copy.deepcopy(messages)
    if not text:
        return out

    user_idx = None
    for i, msg in enumerate(out):
        if isinstance(msg, dict) and msg.get("role") == "user":
            user_idx = i
            break

    if user_idx is None:
        out.append({"role": "user", "content": text})
        return out

    user_msg = out[user_idx]
    content = user_msg.get("content")

    if isinstance(content, str):
        user_msg["content"] = f"{text}\n\n{content}" if content else text
    elif isinstance(content, list):
        user_msg["content"] = [{"type": "text", "text": f"{text}\n\n"}] + copy.deepcopy(content)
    else:
        fallback = _message_content_to_text(content)
        user_msg["content"] = f"{text}\n\n{fallback}" if fallback else text

    out[user_idx] = user_msg
    return out


def prepare_messages_for_model(messages: List[Dict[str, Any]], model_name: Optional[str] = None) -> List[Dict[str, Any]]:
    if not isinstance(messages, list):
        return messages

    if not is_ministral3_family(model_name):
        return messages

    try:
        system_msg = _load_ministral_system_prompt(str(model_name))
    except Exception as exc:
        logging.warning(
            "Failed to load Ministral system prompt for model=%s; using original messages. Error: %s",
            model_name,
            exc,
        )
        return messages

    original_system_chunks: List[str] = []
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "system":
            continue
        content = msg.get("content")
        if _is_structured_ministral_system_content(content):
            continue
        text = _message_content_to_text(content).strip()
        if text:
            original_system_chunks.append(text)

    original_system_text = "\n\n".join(original_system_chunks).strip()

    filtered = [m for m in messages if isinstance(m, dict) and m.get("role") != "system"]
    prepared = [copy.deepcopy(system_msg)] + copy.deepcopy(filtered)
    if original_system_text:
        prepared = _prepend_text_to_first_user_message(prepared, original_system_text)
    return prepared


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _json_default(obj: Any) -> Any:
    try:
        return obj.__dict__
    except Exception:
        return str(obj)


def append_jsonl(obj: Any, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    line = json.dumps(obj, default=_json_default) + "\n"

    lock_path = path + ".lock"
    lock_fd = None
    stale_after = 120.0
    poll = 0.05

    while True:
        try:
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            break
        except FileExistsError:
            try:
                age = time.time() - os.path.getmtime(lock_path)
                if age > stale_after:
                    os.remove(lock_path)
                    continue
            except FileNotFoundError:
                continue
            time.sleep(poll)

    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
    finally:
        if lock_fd is not None:
            try:
                os.close(lock_fd)
            except Exception:
                pass
            try:
                os.remove(lock_path)
            except Exception:
                pass


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _parse_json_candidate(raw_json: str) -> dict:
    cleaned_json = re.sub(r"#.*?$", "", raw_json, flags=re.MULTILINE)
    cleaned_json = re.sub(r"//.*?$", "", cleaned_json, flags=re.MULTILINE)
    cleaned_json = re.sub(r",\s*([\]}])", r"\1", cleaned_json)
    try:
        return json.loads(cleaned_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse JSON after cleaning:\n{cleaned_json}") from exc


def extract_json_with_reasoning(text: str) -> dict:
    match = re.search(r"\{[\s\S]*?\}", text)
    if not match:
        raise ValueError("No JSON object found in text.")

    raw_json = match.group(0)
    data = _parse_json_candidate(raw_json)

    reasoning = (text[:match.start()] + text[match.end():]).strip()
    reasoning = re.sub(r"(JSON\s*Response\s*:?)$", "", reasoning, flags=re.IGNORECASE | re.MULTILINE).strip()
    reasoning = re.sub(r"\n{3,}", "\n\n", reasoning)
    data["reasoning"] = reasoning
    return data


def extract_last_json_with_reasoning(text: str) -> dict:
    spans: List[tuple[int, int]] = []
    depth = 0
    start: Optional[int] = None
    in_str = False
    escaped = False

    for i, ch in enumerate(text):
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue

        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
            continue

        if ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                spans.append((start, i + 1))
                start = None

    if not spans:
        raise ValueError("No JSON object found in text.")

    last_error: Optional[Exception] = None
    for start, end in reversed(spans):
        raw_json = text[start:end]
        try:
            data = _parse_json_candidate(raw_json)
            reasoning = text[:start].strip()
            reasoning = re.sub(r"(JSON\s*Response\s*:?)$", "", reasoning, flags=re.IGNORECASE | re.MULTILINE).strip()
            reasoning = re.sub(r"\n{3,}", "\n\n", reasoning)
            data["reasoning"] = reasoning
            return data
        except ValueError as exc:
            last_error = exc

    if last_error is not None:
        raise last_error
    raise ValueError("Failed to parse any candidate JSON object.")


def get_reasoning_model_output(text: str, model_name: Optional[str] = None) -> dict:
    reasoning = extract_reasoning_trace(text, model_name=model_name)
    remaining = strip_reasoning_trace(text, model_name=model_name)

    if reasoning or remaining != ("" if text is None else str(text).strip()):
        output_json = extract_json_with_reasoning(remaining)
        output_json.update({"reasoning": reasoning})
        return output_json

    return extract_last_json_with_reasoning(text)


def atomic_write_json(path: str, data: Any) -> None:
    dirpath = os.path.dirname(path)
    os.makedirs(dirpath, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix="tmp_", dir=dirpath, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def get_model_output(
    llm: Any,
    messages: List[Dict[str, Any]],
    is_reasoning_model: bool,
    temperature: float,
    top_p: float,
    max_tokens: int,
    repetition_penalty: float,
    num_responses: int,
    max_retries: int,
    model_name: Optional[str] = None,
    seed_offset: int = 0,
):
    if SamplingParams is None:
        raise RuntimeError("vLLM is unavailable in this environment.")
    for attempt in range(max_retries):
        try:
            prepared_messages = prepare_messages_for_model(messages, model_name=model_name)
            msg_list = prepared_messages if num_responses == 1 else [prepared_messages] * num_responses
            params_list = [
                SamplingParams(
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    repetition_penalty=repetition_penalty,
                    seed=int(seed_offset) + j + attempt * (num_responses or 1),
                )
                for j in range(num_responses)
            ]
            outputs = llm.chat(msg_list, sampling_params=params_list)
            parsed_results = []

            for out in outputs:
                try:
                    raw_text = out.outputs[0].text
                except Exception:
                    raw_text = str(out)
                try:
                    if is_reasoning_model:
                        parsed = get_reasoning_model_output(raw_text, model_name=model_name)
                    else:
                        parsed = extract_json_with_reasoning(raw_text)
                    parsed["_raw_text"] = raw_text
                    parsed_results.append(parsed)
                except Exception as exc:
                    parsed_results.append(
                        {
                            "Parse_fail": True,
                            "error": str(exc),
                            "_raw_text": raw_text,
                        }
                    )

            if num_responses == 1:
                return parsed_results[0], attempt
            return parsed_results, attempt
        except Exception as exc:
            logging.exception("LLM call failed on attempt %d: %s", attempt, exc)
            continue

    fail_msg = {"Parse_fail": True, "error": "Exceeded max_retries without successful LLM call"}
    return ([fail_msg] * num_responses) if num_responses > 1 else (fail_msg, max_retries - 1)
