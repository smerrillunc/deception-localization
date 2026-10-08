from __future__ import annotations

import re
from typing import Any, Optional, Tuple


_GPT_OSS_FINAL_RE = re.compile(r"(?is)assistant\s*final")
_THINK_CLOSE_RE = re.compile(r"(?is)</think>")
_MINISTRAL_THINK_CLOSE_RE = re.compile(r"(?is)\[/think\]")
_LEADING_ANALYSIS_RE = re.compile(r"(?is)^\s*analysis\s*")
_FENCE_OPEN_RE = re.compile(r"^\s*```(?:json)?\s*", re.IGNORECASE)
_FENCE_CLOSE_RE = re.compile(r"\s*```\s*$")


def is_ministral3_family(model_name: Optional[str]) -> bool:
    name = (model_name or "").strip().lower()
    return "mistralai/" in name and "ministral-3" in name


def is_gpt_oss_family(model_name: Optional[str]) -> bool:
    name = (model_name or "").strip().lower()
    return "gpt-oss" in name or "gpt_oss" in name


def strip_json_fences(text: Any) -> str:
    stripped = "" if text is None else str(text).strip()
    if not stripped:
        return ""
    stripped = _FENCE_OPEN_RE.sub("", stripped)
    stripped = _FENCE_CLOSE_RE.sub("", stripped)
    return stripped.strip()


def reasoning_close_span_from_text(
    text: Any,
    model_name: Optional[str] = None,
) -> Optional[Tuple[int, int]]:
    if not isinstance(text, str):
        return None

    patterns = []
    if is_gpt_oss_family(model_name):
        patterns.extend((_GPT_OSS_FINAL_RE, _THINK_CLOSE_RE, _MINISTRAL_THINK_CLOSE_RE))
    elif is_ministral3_family(model_name):
        patterns.extend((_MINISTRAL_THINK_CLOSE_RE, _THINK_CLOSE_RE, _GPT_OSS_FINAL_RE))
    else:
        patterns.extend((_THINK_CLOSE_RE, _MINISTRAL_THINK_CLOSE_RE, _GPT_OSS_FINAL_RE))

    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return match.start(), match.end()
    return None


def extract_reasoning_trace(text: Any, model_name: Optional[str] = None) -> str:
    if not isinstance(text, str):
        return ""

    close_span = reasoning_close_span_from_text(text, model_name=model_name)
    if close_span is None:
        return ""

    close_start, close_end = close_span
    if is_gpt_oss_family(model_name) or _GPT_OSS_FINAL_RE.match(text[close_start:close_end]):
        reasoning = _LEADING_ANALYSIS_RE.sub("", text[:close_start], count=1)
        return reasoning.strip()

    return text[:close_end].strip()


def strip_reasoning_trace(text: Any, model_name: Optional[str] = None) -> str:
    stripped = "" if text is None else str(text).strip()
    if not stripped:
        return ""

    close_span = reasoning_close_span_from_text(stripped, model_name=model_name)
    if close_span is None:
        return strip_json_fences(stripped)

    _, close_end = close_span
    return strip_json_fences(stripped[close_end:])
