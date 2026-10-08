from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

from reasoning_parser import extract_reasoning_trace


LABEL_FILTER_ALL = "all"
LABEL_FILTER_DECEPTIVE_ONLY = "deceptive_only"
LABEL_FILTER_TRUTHFUL_ONLY = "truthful_only"
LABEL_FILTER_CHOICES = (
    LABEL_FILTER_ALL,
    LABEL_FILTER_DECEPTIVE_ONLY,
    LABEL_FILTER_TRUTHFUL_ONLY,
)


def _coerce_deceptive_label(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
        return None

    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "t", "yes", "y", "deceptive"}:
            return True
        if v in {"0", "false", "f", "no", "n", "truthful"}:
            return False

    return None


def keep_record_for_label_filter(record: Dict[str, Any], label_filter: str) -> bool:
    if label_filter == LABEL_FILTER_ALL:
        return True

    deceptive_label = _coerce_deceptive_label(record.get("deceptive"))
    if label_filter == LABEL_FILTER_DECEPTIVE_ONLY:
        return deceptive_label is True
    if label_filter == LABEL_FILTER_TRUTHFUL_ONLY:
        return deceptive_label is False

    raise ValueError(
        f"Unknown label_filter={label_filter!r}. "
        f"Expected one of {LABEL_FILTER_CHOICES}"
    )


def normalize_label_filter(
    label_filter: Optional[str] = None,
    *,
    only_deceptive: bool = False,
    only_truthful: bool = False,
) -> str:
    chosen = label_filter or LABEL_FILTER_ALL
    if chosen not in LABEL_FILTER_CHOICES:
        raise ValueError(
            f"Invalid label_filter={chosen!r}. "
            f"Expected one of {LABEL_FILTER_CHOICES}"
        )

    if only_deceptive and only_truthful:
        raise ValueError("Cannot set both deceptive-only and truthful-only filters.")

    if only_deceptive:
        if chosen not in {LABEL_FILTER_ALL, LABEL_FILTER_DECEPTIVE_ONLY}:
            raise ValueError("Conflicting filter flags: deceptive-only vs truthful-only.")
        return LABEL_FILTER_DECEPTIVE_ONLY

    if only_truthful:
        if chosen not in {LABEL_FILTER_ALL, LABEL_FILTER_TRUTHFUL_ONLY}:
            raise ValueError("Conflicting filter flags: truthful-only vs deceptive-only.")
        return LABEL_FILTER_TRUTHFUL_ONLY

    return chosen


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_jsonl(path: Path, strict: bool) -> Iterator[tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_num, json.loads(line)
            except json.JSONDecodeError:
                if strict:
                    raise


def _extract_run_info(samples_path: Path, root_dir: Path) -> Dict[str, Optional[str]]:
    rel = samples_path.relative_to(root_dir)
    parts = rel.parts

    run_id = str(Path(*parts[:-1])) if len(parts) > 1 else samples_path.parent.name
    gpu = next((p for p in parts if p.startswith("gpu_")), None)
    run_date = next((p for p in parts if re.fullmatch(r"\d{4}-\d{2}-\d{2}", p)), None)

    return {
        "run_id": run_id,
        "run_date": run_date,
        "gpu": gpu,
        "source_path": str(samples_path),
    }


def _extract_reasoning_from_raw_text(text: Any) -> Optional[str]:
    reasoning = extract_reasoning_trace(text)
    return reasoning or None


def _synthesize_action_raw_text(action: Dict[str, Any]) -> Optional[str]:
    if not isinstance(action, dict) or not action:
        return None
    try:
        return json.dumps(action, ensure_ascii=False, indent=2)
    except Exception:
        return None


def _sanitize_record_token(value: Any) -> Optional[str]:
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip()
    if not text:
        return None
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)


def _compose_record_id(run_id: str, line_num: int, row: Dict[str, Any]) -> str:
    parts = [run_id]

    conversation_id = _sanitize_record_token(row.get("conversation_id"))
    game_id = _sanitize_record_token(row.get("game_id"))
    turn_idx = _sanitize_record_token(row.get("turn_idx"))
    state_id = _sanitize_record_token(row.get("state_id"))
    sample_idx = _sanitize_record_token(row.get("sample_idx"))

    if conversation_id is not None:
        parts.append(f"conversation_{conversation_id}")
    elif game_id is not None:
        parts.append(f"game_{game_id}")

    if turn_idx is not None:
        parts.append(f"turn_{turn_idx}")
    if state_id is not None:
        parts.append(f"state_{state_id}")
    if sample_idx is not None:
        parts.append(f"sample_{sample_idx}")

    if len(parts) == 1:
        parts.append(f"line_{line_num}")

    return "/".join(parts)


def iter_deception_records(
    root_dir: str | Path,
    *,
    include_messages: bool = False,
    include_action: bool = False,
    flatten_action: bool = True,
    include_meta: bool = True,
    strict_json: bool = True,
    label_filter: str = LABEL_FILTER_ALL,
) -> Iterator[Dict[str, Any]]:
    """
    Iterate over all deception_samples.jsonl files under root_dir, yielding
    normalized records with run metadata attached.
    """
    label_filter = normalize_label_filter(label_filter)
    root = Path(root_dir)
    samples_files = sorted(root.rglob("deception_samples.jsonl"))

    for samples_path in samples_files:
        run_info = _extract_run_info(samples_path, root)
        meta_path = samples_path.with_name("meta.json")
        meta = _load_json(meta_path) if include_meta and meta_path.exists() else {}

        for line_num, rec in _iter_jsonl(samples_path, strict_json):
            out: Dict[str, Any] = {}

            if include_meta and meta:
                for key, value in meta.items():
                    out[f"meta_{key}"] = value

            out.update(run_info)

            for key in (
                "state_id",
                "sample_idx",
                "seed",
                "deceptive",
                "game_id",
                "conversation_id",
                "scenario_name",
                "base_scenario_name",
                "turn_idx",
                "phase",
                "current_rank",
                "active_player",
                "hand",
                "pile_size",
                "history_len",
                "prompt",
                "game_type",
                "truth_context",
            ):
                if key in rec:
                    out[key] = rec[key]

            if include_messages and "messages" in rec:
                out["messages"] = rec["messages"]

            action = rec.get("action") if isinstance(rec.get("action"), dict) else None
            if include_action and action is not None:
                out["action"] = action

            if flatten_action and action is not None:
                out["action_type"] = action.get("Action")
                out["cards_played"] = action.get("Cards_played")
                out["action_parse_fail"] = action.get("Parse_fail")
                raw_text = action.get("_raw_text")
                if not (isinstance(raw_text, str) and raw_text.strip()):
                    raw_text = _synthesize_action_raw_text(action)

                action_reasoning = action.get("reasoning")
                if not (isinstance(action_reasoning, str) and action_reasoning.strip()):
                    action_reasoning = action.get("Reasoning")
                if not (isinstance(action_reasoning, str) and action_reasoning.strip()):
                    action_reasoning = _extract_reasoning_from_raw_text(raw_text)
                if isinstance(action_reasoning, str) and action_reasoning.strip():
                    out["action_reasoning"] = action_reasoning
                if isinstance(raw_text, str) and raw_text.strip():
                    out["action_raw_text"] = raw_text
                if isinstance(action.get("Cards_played"), list):
                    out["cards_played_len"] = len(action["Cards_played"])

            out["record_id"] = _compose_record_id(run_info["run_id"], line_num, out)

            if not keep_record_for_label_filter(out, label_filter):
                continue

            yield out


def write_jsonl(records: List[Dict[str, Any]], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def count_sentences(text: Any) -> int:
    if not isinstance(text, str):
        return 0
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return 0
    parts = [p for p in _SENTENCE_SPLIT_RE.split(cleaned) if p]
    return len(parts)


def _iter_text_fields(row: Dict[str, Any], fields: Sequence[str]) -> Iterator[str]:
    for field in fields:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            yield value
