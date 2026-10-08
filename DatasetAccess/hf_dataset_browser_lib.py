from __future__ import annotations

import gzip
import json
import re
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

try:
    from scipy.stats import binomtest
except Exception:
    binomtest = None


DEFAULT_REPO_ID = "anonymous-neurips-2026-ED/deception-localization"
DEFAULT_REPO_TYPE = "dataset"
DEFAULT_SCAN_LIMIT = 500
DEFAULT_BOOTSTRAP_SAMPLES = 1000


def get_hf_api() -> HfApi:
    return HfApi()


def _safe_int(value: object) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: object) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _entry_path(entry: object) -> Optional[str]:
    for attr in ("path", "rfilename"):
        value = getattr(entry, attr, None)
        if isinstance(value, str) and value:
            return value
    return None


def _leaf_name(path: str) -> str:
    return str(path).rstrip("/").split("/")[-1]


def _looks_like_folder_name(name: str) -> bool:
    if not name or name.startswith("."):
        return False
    return "." not in name


def parse_repo_path(repo_path: str) -> Dict[str, str]:
    parts = str(repo_path).split("/")
    if len(parts) < 4:
        raise ValueError(f"Unexpected repo path: {repo_path}")
    return {
        "environment": parts[0],
        "model": parts[1],
        "section": parts[2],
        "filename": parts[-1],
        "repo_path": repo_path,
    }


@lru_cache(maxsize=32)
def list_environments(
    repo_id: str = DEFAULT_REPO_ID,
    repo_type: str = DEFAULT_REPO_TYPE,
    token: Optional[str] = None,
) -> List[str]:
    api = get_hf_api()
    out: List[str] = []
    for entry in api.list_repo_tree(
        repo_id=repo_id,
        path_in_repo=None,
        recursive=False,
        expand=False,
        repo_type=repo_type,
        token=token or None,
    ):
        path = _entry_path(entry)
        if not path:
            continue
        leaf = _leaf_name(path)
        if _looks_like_folder_name(leaf):
            out.append(leaf)
    return sorted(set(out))


@lru_cache(maxsize=128)
def list_models_for_environment(
    environment: str,
    repo_id: str = DEFAULT_REPO_ID,
    repo_type: str = DEFAULT_REPO_TYPE,
    token: Optional[str] = None,
) -> List[str]:
    if not environment:
        return []
    api = get_hf_api()
    out: List[str] = []
    for entry in api.list_repo_tree(
        repo_id=repo_id,
        path_in_repo=environment,
        recursive=False,
        expand=False,
        repo_type=repo_type,
        token=token or None,
    ):
        path = _entry_path(entry)
        if not path:
            continue
        leaf = _leaf_name(path)
        if _looks_like_folder_name(leaf):
            out.append(leaf)
    return sorted(set(out))


@lru_cache(maxsize=512)
def list_localization_repo_paths(
    environment: str,
    model: str,
    *,
    repo_id: str = DEFAULT_REPO_ID,
    repo_type: str = DEFAULT_REPO_TYPE,
    token: Optional[str] = None,
    limit: Optional[int] = DEFAULT_SCAN_LIMIT,
    filename_filter: str = "",
) -> List[str]:
    if not environment or not model:
        return []

    api = get_hf_api()
    path_in_repo = f"{environment}/{model}/localization"
    needle = filename_filter.strip().lower()
    out: List[str] = []

    for entry in api.list_repo_tree(
        repo_id=repo_id,
        path_in_repo=path_in_repo,
        recursive=False,
        expand=False,
        repo_type=repo_type,
        token=token or None,
    ):
        repo_path = _entry_path(entry)
        if not repo_path or not repo_path.endswith(".json.gz"):
            continue
        if needle and needle not in repo_path.lower():
            continue
        out.append(repo_path)
        if limit is not None and len(out) >= int(limit):
            break

    return sorted(out)


@lru_cache(maxsize=2048)
def load_repo_record(
    repo_path: str,
    *,
    repo_id: str = DEFAULT_REPO_ID,
    repo_type: str = DEFAULT_REPO_TYPE,
    token: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> Dict:
    local_path = hf_hub_download(
        repo_id=repo_id,
        repo_type=repo_type,
        filename=repo_path,
        token=token or None,
        cache_dir=cache_dir,
    )
    with gzip.open(local_path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def preview_value(value: object, limit: int = 120) -> object:
    if isinstance(value, str):
        compact = " ".join(value.split())
        if len(compact) > limit:
            return compact[: limit - 3] + "..."
        return compact
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return f"list[{len(value)}]"
    if isinstance(value, dict):
        return f"dict[{len(value)}]"
    return type(value).__name__


def summarize_record_schema(record: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    top_level_df = pd.DataFrame(
        [
            {
                "key": key,
                "python_type": type(value).__name__,
                "preview": preview_value(value),
            }
            for key, value in record.items()
        ]
    ).sort_values("key").reset_index(drop=True)

    history = record.get("history") or []
    history_keys = sorted({key for row in history for key in row.keys()})
    history_df = pd.DataFrame({"history_key": history_keys})

    generations = []
    for row in history:
        generations.extend(row.get("generations") or [])
    generation_keys = sorted({key for row in generations for key in row.keys()})
    generation_df = pd.DataFrame({"generation_key": generation_keys})
    return top_level_df, history_df, generation_df


def split_sentence_spans(text: str) -> List[Dict[str, object]]:
    if not text:
        return []

    spans: List[Dict[str, object]] = []
    for match in re.finditer(r".*?(?:[.!?](?:\s+|$)|\n+|$)", text, flags=re.S):
        start, end = match.span()
        if end <= start:
            continue
        segment = text[start:end]
        if not segment.strip():
            continue
        spans.append({"start": start, "end": end, "text": segment})

    if not spans:
        spans = [{"start": 0, "end": len(text), "text": text}]
    return spans


def bootstrap_deception_ci(
    generations: List[Dict],
    *,
    alpha: float = 0.05,
    n_boot: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    truth_values = [
        generation.get("is_truthful")
        for generation in generations
        if generation.get("is_truthful") is not None
    ]
    if not truth_values:
        return np.nan, np.nan

    deceptions = np.array([0.0 if truth else 1.0 for truth in truth_values], dtype=float)
    if n_boot <= 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    samples = rng.choice(deceptions, size=(n_boot, len(deceptions)), replace=True)
    rates = samples.mean(axis=1)
    lo, hi = np.quantile(rates, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def normalize_history(history: Iterable[Dict]) -> List[Dict]:
    normalized: List[Dict] = []
    for probe in history:
        out = dict(probe)
        sent_end = out.get("sentence_end_idx")
        sent_idx = out.get("sentence_idx_inclusive")
        if sent_idx is None:
            sent_idx = out.get("sentence_idx")
        if sent_end is None and sent_idx is not None:
            sent_end = int(sent_idx) + 1
        if sent_end is not None:
            sent_idx = int(sent_end) - 1 if int(sent_end) > 0 else None
        out["sentence_end_idx"] = int(sent_end) if sent_end is not None else None
        out["sentence_idx"] = int(sent_idx) if sent_idx is not None else None

        char_span = out.get("char_span")
        if isinstance(char_span, list):
            char_span = tuple(char_span)
        if isinstance(char_span, tuple) and len(char_span) == 2:
            out["char_span"] = (int(char_span[0]), int(char_span[1]))
        else:
            out["char_span"] = None
        normalized.append(out)
    return normalized


def flatten_history(history: List[Dict], raw_text: str) -> pd.DataFrame:
    rows = []
    for step_id, probe in enumerate(history):
        generations = probe.get("generations") or []
        for sample_id, generation in enumerate(generations):
            rows.append(
                {
                    "step_id": step_id,
                    "sentence_end_idx": probe.get("sentence_end_idx"),
                    "sentence_idx": probe.get("sentence_idx"),
                    "sample_id": sample_id,
                    "deception_rate_step": probe.get("deception_rate"),
                    "num_truthful_step": probe.get("num_truthful"),
                    "num_valid_step": probe.get("num_valid"),
                    "gen_text": generation.get("gen_text"),
                    "full_generation_text": generation.get("full_generation_text"),
                    "is_truthful": generation.get("is_truthful"),
                    "deceptive": generation.get("deceptive"),
                    "parse_error": generation.get("parse_error"),
                    "parsed": generation.get("parsed"),
                    "evaluation": generation.get("evaluation"),
                    "sentence_text": probe.get("sentence_text"),
                    "char_span": probe.get("char_span"),
                    "raw_text": raw_text,
                }
            )
    return pd.DataFrame(rows)


def build_stats(history: List[Dict], *, n_boot: int = DEFAULT_BOOTSTRAP_SAMPLES) -> pd.DataFrame:
    rows = []
    for step_id, probe in enumerate(history):
        sent_end = probe.get("sentence_end_idx")
        if sent_end is None:
            continue
        num_true = int(probe.get("num_truthful") or 0)
        num_valid = int(probe.get("num_valid") or 0)
        dec_rate = probe.get("deception_rate")
        if dec_rate is None and num_valid > 0:
            dec_rate = 1.0 - (num_true / num_valid)
        seed = probe.get("seed")
        if seed is None:
            seed = step_id + 1
        generations = probe.get("generations") or []
        ci_low, ci_high = bootstrap_deception_ci(
            generations,
            alpha=0.05,
            n_boot=n_boot,
            seed=int(seed),
        )
        p_value = binomtest(num_true, num_valid, p=0.5).pvalue if (binomtest and num_valid > 0) else np.nan
        rows.append(
            {
                "step_id": step_id,
                "sentence_end_idx": sent_end,
                "sentence_idx": probe.get("sentence_idx"),
                "deception_rate": dec_rate,
                "num_truthful": num_true,
                "num_valid": num_valid,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p_value": p_value,
            }
        )
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("sentence_end_idx").reset_index(drop=True)
    return df


def build_sentence_spans(raw_text: str) -> Tuple[List[Dict], Dict[int, Dict]]:
    spans: List[Dict] = []
    for idx, span in enumerate(split_sentence_spans(raw_text)):
        start = _safe_int(span.get("start"))
        end = _safe_int(span.get("end"))
        if start is None or end is None or end <= start:
            continue
        spans.append(
            {
                "sentence_idx": idx,
                "start": start,
                "end": end,
                "text": span.get("text"),
            }
        )
    spans.sort(key=lambda item: (item.get("start", 0), item.get("sentence_idx", 0)))
    span_map = {int(item["sentence_idx"]): item for item in spans}
    return spans, span_map


def resolve_sentence_span(
    sentence_idx: Optional[int],
    sentence_span_map: Dict[int, Dict],
    probe: Optional[Dict] = None,
) -> Tuple[Optional[str], Optional[int], Optional[int]]:
    if sentence_idx is None or sentence_idx < 0:
        return None, None, None

    idx = int(sentence_idx)
    span = sentence_span_map.get(idx)
    if span:
        return span.get("text"), span.get("start"), span.get("end")

    if probe is None:
        return None, None, None

    char_span = probe.get("char_span")
    if isinstance(char_span, (list, tuple)) and len(char_span) == 2:
        return probe.get("sentence_text"), int(char_span[0]), int(char_span[1])
    return probe.get("sentence_text"), None, None


def compute_deceptive_sentence_idx(
    right_sentence_end_idx: Optional[int],
    df_stats: pd.DataFrame,
) -> Optional[int]:
    if right_sentence_end_idx is not None:
        idx = int(right_sentence_end_idx) - 1
        if idx >= 0:
            return idx

    if len(df_stats) == 0:
        return None

    candidates = df_stats[
        (df_stats["deception_rate"].notna())
        & (df_stats["num_valid"] > 0)
        & (df_stats["deception_rate"] >= 0.5)
    ]
    if len(candidates) == 0:
        return None
    return int(candidates.sort_values("sentence_idx").iloc[0]["sentence_idx"])


def plot_sentence_localization(
    df_stats: pd.DataFrame,
    deceptive_sentence_idx: Optional[int] = None,
    right_sentence_end_idx: Optional[int] = None,
) -> Tuple[plt.Figure, pd.DataFrame]:
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    if len(df_stats) == 0:
        ax.set_title("No stats available")
        return fig, df_stats

    x = df_stats["sentence_idx"]
    y = df_stats["deception_rate"]
    ax.plot(x, y, color="#666666", linewidth=2, label="Deception rate")

    if deceptive_sentence_idx is not None:
        after_mask = x > deceptive_sentence_idx
        before_mask = ~after_mask
    else:
        after_mask = pd.Series([False] * len(df_stats))
        before_mask = ~after_mask

    ax.scatter(x[before_mask], y[before_mask], color="#1f77b4", s=40, label="Before/at deceptive")
    if after_mask.any():
        ax.scatter(x[after_mask], y[after_mask], color="#2ca02c", s=40, label="After deceptive")

    if df_stats["ci_low"].notna().any():
        ax.fill_between(x, df_stats["ci_low"], df_stats["ci_high"], alpha=0.2, label="95% bootstrap CI")

    ax.axhline(0.5, linestyle="--", linewidth=2, label="50% threshold")
    if right_sentence_end_idx is not None:
        ax.axvline(
            int(right_sentence_end_idx) - 1,
            linestyle="-.",
            linewidth=2,
            label=f"Earliest >= 0.5 @ {int(right_sentence_end_idx) - 1}",
        )

    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Sentence index")
    ax.set_ylabel("Deception rate")
    if len(df_stats) and df_stats["p_value"].notna().any():
        ax.set_title(f"Sentence-level deception localization\nmin p = {df_stats['p_value'].min():.1e}")
    else:
        ax.set_title("Sentence-level deception localization")
    ax.grid(alpha=0.3)
    ax.legend()
    return fig, df_stats


def _escape_html(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _escape_html_attr(text: str) -> str:
    return _escape_html(text).replace('"', "&quot;").replace("'", "&#39;")


def render_highlighted_sentences_html(
    raw_text: str,
    sentence_spans: List[Dict],
    selected_idx: Optional[int],
    deceptive_idx: Optional[int],
    selected_span: Optional[Tuple[int, int]] = None,
) -> str:
    if not raw_text:
        return "<i>No raw text available.</i>"

    if not sentence_spans:
        return (
            "<div style=\"background-color:white; padding:8px; line-height:1.5; "
            "white-space:pre-wrap; font-family:serif;\">"
            f"{_escape_html(raw_text)}"
            "</div>"
        )

    parts: List[str] = []
    last = 0
    sel_start = None
    sel_end = None
    if selected_span and len(selected_span) == 2:
        sel_start, sel_end = selected_span
        if sel_start is not None and sel_end is not None and sel_end <= sel_start:
            sel_start, sel_end = None, None

    def with_style(text: str, *, bg_color: Optional[str], tooltip: Optional[str]) -> str:
        if not text:
            return ""
        escaped = _escape_html(text)
        inner = escaped
        if bg_color:
            inner = f"<mark style='background-color:{bg_color};'>{escaped}</mark>"
        if tooltip:
            return f"<span title=\"{_escape_html_attr(tooltip)}\">{inner}</span>"
        return inner

    for idx, span in enumerate(sentence_spans):
        start = span.get("start")
        end = span.get("end")
        if start is None or end is None:
            continue
        if start > last:
            parts.append(_escape_html(raw_text[last:start]))

        text = raw_text[start:end]
        span_idx = span.get("sentence_idx")
        if span_idx is None:
            span_idx = idx
        span_idx = _safe_int(span_idx)
        if span_idx is None:
            span_idx = idx
        base_green = deceptive_idx is not None and span_idx > deceptive_idx
        tooltip = f"Sentence {span_idx}"

        if sel_start is not None and sel_end is not None and not (sel_end <= start or sel_start >= end):
            overlap_start = max(start, sel_start)
            overlap_end = min(end, sel_end)
            before = raw_text[start:overlap_start]
            middle = raw_text[overlap_start:overlap_end]
            after = raw_text[overlap_end:end]
            if base_green:
                if before:
                    parts.append(with_style(before, bg_color="#c8f7c5", tooltip=tooltip))
                parts.append(with_style(middle, bg_color="#ffe08a", tooltip=tooltip))
                if after:
                    parts.append(with_style(after, bg_color="#c8f7c5", tooltip=tooltip))
            else:
                parts.append(with_style(before, bg_color=None, tooltip=tooltip))
                parts.append(with_style(middle, bg_color="#ffe08a", tooltip=tooltip))
                parts.append(with_style(after, bg_color=None, tooltip=tooltip))
        elif selected_idx is not None and span_idx == selected_idx:
            parts.append(with_style(text, bg_color="#ffe08a", tooltip=tooltip))
        elif base_green:
            parts.append(with_style(text, bg_color="#c8f7c5", tooltip=tooltip))
        else:
            parts.append(with_style(text, bg_color=None, tooltip=tooltip))
        last = end

    if last < len(raw_text):
        parts.append(_escape_html(raw_text[last:]))

    return (
        "<div style=\"background-color:white; padding:8px; line-height:1.5; "
        "white-space:pre-wrap; font-family:serif;\">"
        + "".join(parts)
        + "</div>"
    )


def strip_code_fences(text: str) -> str:
    if not text:
        return ""
    out = str(text).strip()
    out = re.sub(r"^```[A-Za-z0-9_+-]*\n?", "", out)
    out = re.sub(r"\n?```$", "", out)
    return out.strip()


def generation_bucket(generation: Dict) -> str:
    truth = generation.get("is_truthful")
    if truth is True:
        return "truthful"
    if truth is False:
        return "deceptive"
    return "invalid"


def _append_colored_segment(
    html_parts: List[str],
    full_text: str,
    start: int,
    end: int,
    color: str,
) -> None:
    if start >= end:
        return
    html_parts.append(f"<span style='color:{color}'>{_escape_html(full_text[start:end])}</span>")


def render_prefix_generation_with_sentence_indices_html(
    prefix_text: str,
    gen_text: str,
    selected_sentence_idx: Optional[int],
    sentence_labels: Optional[Dict[int, str]] = None,
) -> str:
    full_text = (prefix_text or "") + (gen_text or "")
    if not full_text:
        return "<i>No text available for rendering.</i>"

    spans = [
        span for span in split_sentence_spans(full_text)
        if span.get("start") is not None and span.get("end") is not None
    ]
    spans.sort(key=lambda span: span.get("start", 0))
    if not spans:
        return (
            "<div style='background-color:white; padding:5px; line-height:1.5; "
            "white-space:pre-wrap; font-family:monospace'>"
            f"{_escape_html(full_text)}"
            "</div>"
        )

    if selected_sentence_idx is not None:
        selected_sentence_idx = max(0, min(int(selected_sentence_idx), len(spans) - 1))

    html_parts: List[str] = []
    last = 0

    def color_for_idx(idx: int) -> str:
        if selected_sentence_idx is None:
            return "black"
        if idx < selected_sentence_idx:
            return "blue"
        if idx > selected_sentence_idx:
            return "green"
        return "black"

    for idx, span in enumerate(spans):
        start = int(span["start"])
        end = int(span["end"])
        color = color_for_idx(idx)
        if start > last:
            _append_colored_segment(html_parts, full_text, last, start, color)
        _append_colored_segment(html_parts, full_text, start, end, color)
        label = sentence_labels.get(idx) if sentence_labels else None
        marker = f"{idx + 1}"
        if label:
            marker = f"{marker} [{label}]"
        html_parts.append(
            f"<sup style='font-size:0.7em;color:{color}'>{_escape_html(marker)}</sup> "
        )
        last = end

    if last < len(full_text):
        tail_color = color_for_idx(len(spans) - 1) if spans else "black"
        _append_colored_segment(html_parts, full_text, last, len(full_text), tail_color)

    return (
        "<div style='background-color:white; padding:5px; line-height:1.5; "
        "white-space:pre-wrap; font-family:monospace'>"
        + "".join(html_parts)
        + "</div>"
    )


def build_generation_sentence_labels(
    prefix_text: str,
    prefix_mode: str,
    resolved_sentence_idx: Optional[int],
) -> Tuple[Optional[int], Dict[int, str]]:
    prefix_spans = [
        span for span in split_sentence_spans(prefix_text or "")
        if span.get("start") is not None and span.get("end") is not None
    ]
    prefix_spans.sort(key=lambda span: span.get("start", 0))
    display_selected_idx = (len(prefix_spans) - 1) if prefix_spans else None
    sentence_labels: Dict[int, str] = {}

    if not prefix_spans:
        return display_selected_idx, sentence_labels

    if prefix_mode == "sentence" and resolved_sentence_idx is not None:
        global_offset = int(resolved_sentence_idx) - (len(prefix_spans) - 1)
    else:
        global_offset = 0

    for local_idx in range(len(prefix_spans)):
        global_idx = global_offset + local_idx
        sentence_labels[local_idx] = f"S_{global_idx + 1}"

    return display_selected_idx, sentence_labels


def sentence_selector_label(sentence_idx: int, sentence_span_map: Dict[int, Dict]) -> str:
    idx = _safe_int(sentence_idx)
    if idx is None:
        return str(sentence_idx)

    span = sentence_span_map.get(idx, {})
    sentence_text = span.get("text") if isinstance(span, dict) else ""
    if not isinstance(sentence_text, str):
        sentence_text = ""
    sentence_text = " ".join(sentence_text.split())
    if len(sentence_text) > 120:
        sentence_text = sentence_text[:117] + "..."

    if sentence_text:
        return f"{idx}: {sentence_text}"
    return str(idx)
