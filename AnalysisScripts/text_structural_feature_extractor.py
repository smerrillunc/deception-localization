#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # noqa: BLE001
    _tqdm = None

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]
CORE_DIR = REPO_ROOT / "Core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from attention_features import (
    DatasetPaths,
    ExampleValidationError,
    StreamingParquetWriter,
    add_span_match_columns,
    build_localized_sentence_df,
    infer_model_id,
    iter_localization_paths,
    maybe_raise_invalid_example,
    tokenize_and_align_localized_sentences,
)


DEFAULT_OUTPUT_NAME = "commitment_text_structural_baselines.parquet"
DEFAULT_TFIDF_CACHE_DIRNAME = "commitment_text_baseline_tfidf_cache"
DEFAULT_RECENT_WINDOW_SENTENCES = 5
DEFAULT_DELTA_THRESHOLD = 0.3
DEFAULT_WRITE_EVERY_EXAMPLES = 32
DEFAULT_PROGRESS_EVERY = 25
DEFAULT_TFIDF_TEXT_FIELDS = ("last_sentence_text", "prefix_text")
DEFAULT_TFIDF_MAX_FEATURES = 20000
DEFAULT_TFIDF_MIN_NGRAM = 1
DEFAULT_TFIDF_MAX_NGRAM = 2

WORD_RE = re.compile(r"[A-Za-z0-9']+")

OUTPUT_COLUMNS = [
    "dataset",
    "model_name",
    "model_bundle_name",
    "example_id",
    "trace_id",
    "localization_path",
    "prompt",
    "example_label",
    "example_label_source",
    "commitment_direction",
    "sentence_text",
    "last_sentence_text",
    "previous_sentence_text",
    "prefix_text",
    "full_prefix_text",
    "sentence_idx",
    "sentence_idx_1based",
    "trace_length",
    "prefix_sentence_count",
    "sentences_remaining",
    "deceptive_commitment_sentence_idx",
    "truthful_commitment_sentence_idx",
    "example_commitment_sentence_idx",
    "num_truthful",
    "num_valid",
    "raw_start",
    "raw_end",
    "full_start",
    "full_end",
    "start_token",
    "end_token",
    "token_count",
    "current_sentence_token_count",
    "context_token_count",
    "prompt_token_count",
    "raw_text_context_token_count",
    "available_token_count",
    "prior_all_token_count",
    "previous_sentence_token_count",
    "recent_token_count",
    "early_token_count",
    "current_sentence_char_count",
    "current_sentence_word_count",
    "previous_sentence_char_count",
    "previous_sentence_word_count",
    "prefix_char_count",
    "prefix_word_count",
    "prefix_token_count",
    "sentence_char_delta",
    "sentence_word_delta",
    "sentence_token_delta",
    "normalized_position",
    "reverse_normalized_position",
    "deception_rate",
    "prev_deception_rate",
    "next_deception_rate",
    "delta_deception_rate",
    "abs_delta_deception_rate",
    "target_value",
    "has_previous_sentence",
    "has_next_sentence",
    "is_usable_example",
    "has_deceptive_commitment",
    "has_truthful_commitment",
    "has_example_commitment",
    "is_deceptive_commitment_juncture",
    "is_truthful_commitment_juncture",
    "is_commitment_juncture",
]

STRING_COLUMNS = {
    "dataset",
    "model_name",
    "model_bundle_name",
    "example_id",
    "trace_id",
    "localization_path",
    "prompt",
    "example_label",
    "example_label_source",
    "commitment_direction",
    "sentence_text",
    "last_sentence_text",
    "previous_sentence_text",
    "prefix_text",
    "full_prefix_text",
}

BOOL_COLUMNS = {
    "has_previous_sentence",
    "has_next_sentence",
    "is_usable_example",
    "has_deceptive_commitment",
    "has_truthful_commitment",
    "has_example_commitment",
    "is_deceptive_commitment_juncture",
    "is_truthful_commitment_juncture",
    "is_commitment_juncture",
}

FLOAT_COLUMNS = {
    "normalized_position",
    "reverse_normalized_position",
    "deception_rate",
    "prev_deception_rate",
    "next_deception_rate",
    "delta_deception_rate",
    "abs_delta_deception_rate",
    "target_value",
}

INT_COLUMNS = set(OUTPUT_COLUMNS) - STRING_COLUMNS - BOOL_COLUMNS - FLOAT_COLUMNS


def maybe_tqdm(iterable: Sequence[Any], *, desc: str, total: Optional[int] = None, disable: bool = False):
    if disable or _tqdm is None:
        return iterable
    return _tqdm(iterable, desc=desc, total=total)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build sentence-level text-only and structural baseline features for commitment-juncture "
            "prediction. The output parquet includes the current sentence, the reasoning prefix up "
            "to that sentence, the rendered full prefix context, cheap structural features, and "
            "binary commitment labels aligned to the example's final deceptive/truthful outcome."
        )
    )
    parser.add_argument(
        "input_path",
        type=str,
        help=(
            "Dataset directory like "
            "DatasetMain/advisor_audit/DeepSeek-R1-Distill-Qwen-7B "
            "or its localization subdirectory."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=f"Parquet output path. Defaults to <dataset_dir>/{DEFAULT_OUTPUT_NAME}.",
    )
    parser.add_argument("--model-id", type=str, default=None, help="Override inferred Hugging Face model id.")
    parser.add_argument(
        "--recent-window-sentences",
        type=int,
        default=DEFAULT_RECENT_WINDOW_SENTENCES,
        help="Recent-sentence window used when deriving token-count metadata from aligned prefixes.",
    )
    parser.add_argument(
        "--delta-threshold",
        type=float,
        default=DEFAULT_DELTA_THRESHOLD,
        help="Commitment threshold on consecutive-sentence deception-rate changes.",
    )
    parser.add_argument("--max-examples", type=int, default=0, help="Optional cap on localization files to process.")
    parser.add_argument("--shard-id", type=int, default=0, help="Zero-based shard index after sorting localization files.")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards to split localization files across.")
    parser.add_argument(
        "--write-every-examples",
        type=int,
        default=DEFAULT_WRITE_EVERY_EXAMPLES,
        help="Flush buffered rows to parquet after this many examples.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=DEFAULT_PROGRESS_EVERY,
        help="Print a progress update every N processed examples.",
    )
    parser.add_argument("--disable-tqdm", action="store_true", default=False)
    parser.add_argument("--strict", action="store_true", default=False)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--trust-remote-code", action="store_true", default=False)
    parser.add_argument(
        "--compute-tfidf",
        action="store_true",
        default=False,
        help="Also compute cached TF-IDF text features after writing parquet.",
    )
    parser.add_argument(
        "--tfidf-text-fields",
        type=str,
        default=",".join(DEFAULT_TFIDF_TEXT_FIELDS),
        help="Comma-separated text columns to vectorize when --compute-tfidf is enabled.",
    )
    parser.add_argument(
        "--tfidf-cache-dir",
        type=str,
        default=None,
        help=(
            "Override the TF-IDF cache directory. Defaults to "
            f"<dataset_dir>/{DEFAULT_TFIDF_CACHE_DIRNAME}."
        ),
    )
    parser.add_argument("--tfidf-max-features", type=int, default=DEFAULT_TFIDF_MAX_FEATURES)
    parser.add_argument("--tfidf-min-ngram", type=int, default=DEFAULT_TFIDF_MIN_NGRAM)
    parser.add_argument("--tfidf-max-ngram", type=int, default=DEFAULT_TFIDF_MAX_NGRAM)
    parser.add_argument("--overwrite-tfidf-cache", action="store_true", default=False)
    return parser.parse_args(argv)


def resolve_dataset_paths(input_path: str | Path, output_path: Optional[str | Path]) -> DatasetPaths:
    root = Path(input_path).expanduser().resolve()
    if root.name == "localization":
        dataset_dir = root.parent
        localization_dir = root
    else:
        dataset_dir = root
        localization_dir = dataset_dir / "localization"

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")
    if not localization_dir.is_dir():
        raise FileNotFoundError(f"Localization directory does not exist: {localization_dir}")

    output = Path(output_path).expanduser().resolve() if output_path else dataset_dir / DEFAULT_OUTPUT_NAME
    return DatasetPaths(
        dataset_dir=dataset_dir,
        localization_dir=localization_dir,
        output_path=output,
        examples_path=dataset_dir / "examples.jsonl",
    )


def infer_dataset_name(dataset_dir: Path) -> str:
    return str(dataset_dir.parent.name)


def slugify_token(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(text)).strip("_")


def dataset_row_fingerprint(dataset_df: pd.DataFrame) -> str:
    key_df = dataset_df.loc[:, ["example_id", "sentence_idx"]].copy()
    hashed = pd.util.hash_pandas_object(key_df, index=False).to_numpy(dtype=np.uint64, copy=False)
    return hashlib.sha1(hashed.tobytes()).hexdigest()


def render_full_prefix_context(
    *,
    tokenizer: Any,
    prompt: str,
    prompt_messages: Optional[list[dict[str, Any]]],
    prefix_text: str,
) -> str:
    if not isinstance(prompt_messages, list) or not prompt_messages:
        return f"{prompt}{prefix_text}"

    if not prefix_text:
        try:
            return tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                enable_thinking=True,
                add_generation_prompt=True,
            )
        except Exception:
            return str(prompt)

    messages = list(prompt_messages) + [{"role": "assistant", "content": prefix_text}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
            enable_thinking=True,
        )
    except (TypeError, ValueError):
        try:
            base_prompt = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                enable_thinking=True,
                add_generation_prompt=True,
            )
            return f"{base_prompt}{prefix_text}"
        except Exception:
            return f"{prompt}{prefix_text}"


def locate_raw_text_span(full_context: str, raw_text: str) -> tuple[int, int]:
    if full_context.endswith(raw_text):
        start = len(full_context) - len(raw_text)
        return start, len(full_context)
    first_idx = full_context.find(raw_text)
    if first_idx >= 0:
        return first_idx, first_idx + len(raw_text)
    raise ExampleValidationError(
        "raw_text_span_not_found",
        "Could not locate raw_text inside the rendered full context.",
    )


def shift_localized_sentence_spans(sentence_df: pd.DataFrame, *, raw_text_start_char: int) -> pd.DataFrame:
    out = sentence_df.copy()
    out["full_start"] = pd.to_numeric(out["raw_start"], errors="coerce").astype(int) + int(raw_text_start_char)
    out["full_end"] = pd.to_numeric(out["raw_end"], errors="coerce").astype(int) + int(raw_text_start_char)
    return out


def add_trace_region_columns(aligned_sentence_df: pd.DataFrame, *, recent_window_sentences: int) -> pd.DataFrame:
    df = aligned_sentence_df.sort_values("sentence_idx").reset_index(drop=True).copy()
    previous_sentence_start_tokens: list[Optional[int]] = []
    previous_sentence_end_tokens: list[Optional[int]] = []
    previous_sentence_token_counts: list[int] = []
    recent_token_counts: list[int] = []
    early_token_counts: list[int] = []
    prior_all_token_counts: list[int] = []
    available_token_counts: list[int] = []

    for row_idx, row in enumerate(df.itertuples()):
        if row.start_token is None or pd.isna(row.start_token):
            previous_sentence_start_tokens.append(None)
            previous_sentence_end_tokens.append(None)
            previous_sentence_token_counts.append(0)
            recent_token_counts.append(0)
            early_token_counts.append(0)
            prior_all_token_counts.append(0)
            available_token_counts.append(0)
            continue

        start_token = int(row.start_token)
        end_token = int(row.end_token)
        prior_all_token_counts.append(start_token)
        available_token_counts.append(end_token + 1)

        if row_idx == 0:
            previous_sentence_start_tokens.append(None)
            previous_sentence_end_tokens.append(None)
            previous_sentence_token_counts.append(0)
        else:
            prev_row = df.iloc[row_idx - 1]
            prev_start = int(prev_row["start_token"])
            prev_end = int(prev_row["end_token"])
            previous_sentence_start_tokens.append(prev_start)
            previous_sentence_end_tokens.append(prev_end)
            previous_sentence_token_counts.append(prev_end - prev_start + 1)

        recent_start_sentence_idx = max(0, row_idx - int(recent_window_sentences) + 1)
        recent_start = int(df.iloc[recent_start_sentence_idx]["start_token"])
        recent_token_counts.append(end_token + 1 - recent_start)
        early_token_counts.append(recent_start)

    df["previous_sentence_start_token"] = previous_sentence_start_tokens
    df["previous_sentence_end_token"] = previous_sentence_end_tokens
    df["previous_sentence_token_count"] = previous_sentence_token_counts
    df["recent_token_count"] = recent_token_counts
    df["early_token_count"] = early_token_counts
    df["prior_all_token_count"] = prior_all_token_counts
    df["available_token_count"] = available_token_counts
    return df


def build_aligned_sentence_frame_for_full_context(
    *,
    example: dict[str, Any],
    tokenizer: Any,
    recent_window_sentences: int,
) -> tuple[pd.DataFrame, str, int, int]:
    example_id = example.get("example_id")
    if not isinstance(example_id, str) or not example_id:
        raise ExampleValidationError("missing_example_id", "Localization example is missing example_id.")

    raw_text = example.get("raw_text")
    if not isinstance(raw_text, str) or not raw_text:
        raise ExampleValidationError("missing_raw_text", f"{example_id} is missing raw_text.")

    prompt = str(example.get("prompt") or "")
    prompt_messages = example.get("prompt_messages")
    full_context = render_full_prefix_context(
        tokenizer=tokenizer,
        prompt=prompt,
        prompt_messages=prompt_messages if isinstance(prompt_messages, list) else None,
        prefix_text=raw_text,
    )
    raw_text_start, _ = locate_raw_text_span(full_context, raw_text)

    localized_sentence_df = build_localized_sentence_df(example)
    if localized_sentence_df.empty:
        raise ExampleValidationError("empty_history", f"{example_id} has no localized history entries.")

    localized_sentence_df = add_span_match_columns(raw_text, localized_sentence_df)
    if not localized_sentence_df["span_matches"].all():
        bad_count = int((~localized_sentence_df["span_matches"]).sum())
        raise ExampleValidationError(
            "span_mismatch",
            f"{example_id} has {bad_count} localized sentence spans that do not match raw_text.",
        )

    history_prefix_lookup = {
        int(hist["sentence_idx_inclusive"]): str(hist.get("prefix_text", ""))
        for hist in sorted(example.get("history", []), key=lambda item: int(item["sentence_idx_inclusive"]))
    }
    shifted_sentence_df = shift_localized_sentence_spans(localized_sentence_df, raw_text_start_char=raw_text_start)
    token_alignment = tokenize_and_align_localized_sentences(
        tokenizer=tokenizer,
        full_text=full_context,
        sentence_df=shifted_sentence_df,
        raw_text_start_char=raw_text_start,
    )
    input_ids_list = token_alignment.input_ids
    if not input_ids_list:
        raise ExampleValidationError("no_tokens", f"{example_id} tokenized to zero tokens.")

    aligned_sentence_df = token_alignment.aligned_sentence_df
    aligned_sentence_df["prefix_text"] = aligned_sentence_df["sentence_idx"].map(history_prefix_lookup)
    if not (aligned_sentence_df["token_count"] > 0).all():
        bad_count = int((aligned_sentence_df["token_count"] == 0).sum())
        raise ExampleValidationError(
            "unmapped_sentence",
            f"{example_id} has {bad_count} localized sentences that failed to map to tokens.",
        )

    aligned_sentence_df = add_trace_region_columns(
        aligned_sentence_df,
        recent_window_sentences=recent_window_sentences,
    )
    prompt_token_count = int(token_alignment.prompt_token_count)
    return aligned_sentence_df, full_context, prompt_token_count, len(input_ids_list)


def build_empty_output_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=list(OUTPUT_COLUMNS))


def coerce_output_frame_columns(feature_df: pd.DataFrame) -> pd.DataFrame:
    df = feature_df.copy().reindex(columns=list(OUTPUT_COLUMNS)).copy()
    for column in OUTPUT_COLUMNS:
        if column in STRING_COLUMNS:
            df[column] = df[column].astype("string")
        elif column in BOOL_COLUMNS:
            df[column] = df[column].astype("boolean")
        elif column in FLOAT_COLUMNS:
            df[column] = pd.to_numeric(df[column], errors="coerce").astype("float64")
        elif column in INT_COLUMNS:
            df[column] = pd.to_numeric(df[column], errors="coerce").astype("Int64")
    return df


def flush_feature_buffer(writer: StreamingParquetWriter, buffer: list[pd.DataFrame]) -> int:
    if not buffer:
        return 0
    chunk_df = pd.concat(buffer, ignore_index=True)
    buffer.clear()
    writer.write(coerce_output_frame_columns(chunk_df))
    return len(chunk_df)


def create_empty_output_file(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    coerce_output_frame_columns(build_empty_output_frame()).to_parquet(output_path, index=False)


def _coerce_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (int, np.integer)):
        return bool(int(value))
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "t", "1", "yes", "y"}:
            return True
        if normalized in {"false", "f", "0", "no", "n"}:
            return False
    return None


def _extract_example_label_from_generation_block(block: Any) -> tuple[Optional[str], Optional[str]]:
    if not isinstance(block, dict):
        return None, None

    generations = block.get("generations")
    if isinstance(generations, list) and generations:
        labels: list[bool] = []
        for generation in generations:
            if not isinstance(generation, dict):
                continue
            deceptive = _coerce_bool(generation.get("deceptive"))
            if deceptive is not None:
                labels.append(bool(deceptive))
                continue
            truthful = _coerce_bool(generation.get("is_truthful"))
            if truthful is not None:
                labels.append(not bool(truthful))
        if labels:
            deceptive_rate = float(np.mean(np.asarray(labels, dtype=float)))
            return ("deceptive" if deceptive_rate > 0.5 else "truthful"), "generations"

    deceptive = _coerce_bool(block.get("deceptive"))
    if deceptive is not None:
        return ("deceptive" if deceptive else "truthful"), "deceptive_flag"

    truthful = _coerce_bool(block.get("is_truthful"))
    if truthful is not None:
        return ("truthful" if truthful else "deceptive"), "truthful_flag"

    rate = pd.to_numeric(pd.Series([block.get("deception_rate")]), errors="coerce").iloc[0]
    if pd.notna(rate):
        return ("deceptive" if float(rate) > 0.5 else "truthful"), "deception_rate"

    return None, None


def load_examples_label_map(examples_path: Path) -> dict[str, tuple[str, str]]:
    label_map: dict[str, tuple[str, str]] = {}
    if not examples_path.exists():
        return label_map
    with examples_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            example_id = row.get("example_id")
            if example_id is None:
                continue
            deceptive = _coerce_bool(row.get("deceptive"))
            if deceptive is not None:
                label_map[str(example_id)] = ("deceptive" if deceptive else "truthful", "examples_jsonl_deceptive")
                continue
            truthful = _coerce_bool(row.get("is_truthful"))
            if truthful is not None:
                label_map[str(example_id)] = ("truthful" if truthful else "deceptive", "examples_jsonl_is_truthful")
    return label_map


def infer_example_label(
    example: dict[str, Any],
    *,
    explicit_label_map: dict[str, tuple[str, str]],
) -> tuple[Optional[str], str]:
    example_id = str(example.get("example_id") or "")
    explicit = explicit_label_map.get(example_id)
    if explicit is not None:
        return explicit

    full_score_label, full_score_source = _extract_example_label_from_generation_block(example.get("full_score"))
    if full_score_label is not None:
        return full_score_label, f"full_score_{full_score_source}"

    history = example.get("history")
    if isinstance(history, list) and history:
        last_history_label, last_history_source = _extract_example_label_from_generation_block(history[-1])
        if last_history_label is not None:
            return last_history_label, f"history_last_{last_history_source}"

    return None, "unknown"


def count_words(text: str) -> int:
    if not isinstance(text, str) or not text:
        return 0
    return len(WORD_RE.findall(text))


def resolve_prefix_text(raw_text: str, sentence_row: Any) -> str:
    prefix_text = getattr(sentence_row, "prefix_text", None)
    if isinstance(prefix_text, str) and prefix_text:
        return prefix_text
    raw_end = getattr(sentence_row, "raw_end", None)
    if raw_end is not None and not pd.isna(raw_end):
        return str(raw_text[: int(raw_end)])
    sentence_text = getattr(sentence_row, "sentence_text", "")
    return str(sentence_text) if isinstance(sentence_text, str) else ""


def summarize_commitment_targets(
    aligned_sentence_df: pd.DataFrame,
    *,
    example_label: Optional[str],
    delta_threshold: float,
) -> dict[str, Any]:
    df = aligned_sentence_df.sort_values("sentence_idx").reset_index(drop=True).copy()
    if df.empty:
        return {
            "deceptive_commitment_sentence_idx": None,
            "truthful_commitment_sentence_idx": None,
            "example_commitment_sentence_idx": None,
            "commitment_direction": None,
            "has_deceptive_commitment": False,
            "has_truthful_commitment": False,
            "has_example_commitment": None if example_label not in {"deceptive", "truthful"} else False,
        }

    deceptive_idx: Optional[int] = None
    truthful_idx: Optional[int] = None
    previous_rate = float("nan")
    for row in df.itertuples():
        current_rate = pd.to_numeric(pd.Series([row.deception_rate]), errors="coerce").iloc[0]
        if pd.notna(previous_rate) and pd.notna(current_rate):
            delta = float(current_rate) - float(previous_rate)
            if deceptive_idx is None and delta > float(delta_threshold):
                deceptive_idx = int(row.sentence_idx)
            if truthful_idx is None and delta < -float(delta_threshold):
                truthful_idx = int(row.sentence_idx)
        previous_rate = current_rate

    example_commitment_idx: Optional[int]
    commitment_direction: Optional[str]
    if example_label == "deceptive":
        example_commitment_idx = deceptive_idx
        commitment_direction = "toward_deception"
    elif example_label == "truthful":
        example_commitment_idx = truthful_idx
        commitment_direction = "toward_truthfulness"
    else:
        example_commitment_idx = None
        commitment_direction = None

    return {
        "deceptive_commitment_sentence_idx": deceptive_idx,
        "truthful_commitment_sentence_idx": truthful_idx,
        "example_commitment_sentence_idx": example_commitment_idx,
        "commitment_direction": commitment_direction,
        "has_deceptive_commitment": deceptive_idx is not None,
        "has_truthful_commitment": truthful_idx is not None,
        "has_example_commitment": (
            None if example_label not in {"deceptive", "truthful"} else example_commitment_idx is not None
        ),
    }


def build_example_rows(
    *,
    example: dict[str, Any],
    dataset_name: str,
    model_name: str,
    model_bundle_name: str,
    localization_path: Path,
    tokenizer: Any,
    recent_window_sentences: int,
    delta_threshold: float,
    explicit_label_map: dict[str, tuple[str, str]],
) -> pd.DataFrame:
    example_id = example.get("example_id")
    if not isinstance(example_id, str) or not example_id:
        raise ExampleValidationError("missing_example_id", "Localization example is missing example_id.")

    prompt = str(example.get("prompt") or "")
    prompt_messages = example.get("prompt_messages")
    raw_text = example.get("raw_text")
    if not isinstance(raw_text, str) or not raw_text:
        raise ExampleValidationError("missing_raw_text", f"{example_id} is missing raw_text.")

    aligned_sentence_df, _full_context, prompt_token_count, _total_input_tokens = build_aligned_sentence_frame_for_full_context(
        example=example,
        tokenizer=tokenizer,
        recent_window_sentences=int(recent_window_sentences),
    )
    if aligned_sentence_df.empty:
        return build_empty_output_frame()

    example_label, example_label_source = infer_example_label(example, explicit_label_map=explicit_label_map)
    commitment_summary = summarize_commitment_targets(
        aligned_sentence_df,
        example_label=example_label,
        delta_threshold=float(delta_threshold),
    )

    trace_length = int(len(aligned_sentence_df))
    rows: list[dict[str, Any]] = []
    sentence_texts = aligned_sentence_df["sentence_text"].astype(str).tolist()
    deception_rates = pd.to_numeric(aligned_sentence_df["deception_rate"], errors="coerce").astype(float).tolist()

    for row_idx, sentence_row in enumerate(aligned_sentence_df.itertuples()):
        sentence_text = str(sentence_row.sentence_text)
        previous_sentence_text = sentence_texts[row_idx - 1] if row_idx > 0 else ""
        prefix_text = resolve_prefix_text(raw_text, sentence_row)
        full_prefix_text = render_full_prefix_context(
            tokenizer=tokenizer,
            prompt=prompt,
            prompt_messages=prompt_messages if isinstance(prompt_messages, list) else None,
            prefix_text=prefix_text,
        )

        current_char_count = len(sentence_text)
        current_word_count = count_words(sentence_text)
        previous_char_count = len(previous_sentence_text) if row_idx > 0 else 0
        previous_word_count = count_words(previous_sentence_text) if row_idx > 0 else 0
        current_token_count = int(sentence_row.token_count)
        previous_token_count = int(sentence_row.previous_sentence_token_count)
        prefix_token_count = max(0, int(sentence_row.available_token_count) - int(prompt_token_count))

        current_rate = deception_rates[row_idx]
        prev_rate = deception_rates[row_idx - 1] if row_idx > 0 else float("nan")
        next_rate = deception_rates[row_idx + 1] if row_idx + 1 < trace_length else float("nan")
        delta_rate = float(current_rate - prev_rate) if row_idx > 0 and np.isfinite(prev_rate) and np.isfinite(current_rate) else float("nan")
        abs_delta_rate = abs(delta_rate) if np.isfinite(delta_rate) else float("nan")

        sentence_idx = int(sentence_row.sentence_idx)
        is_usable_example = example_label in {"deceptive", "truthful"}
        example_commitment_idx = commitment_summary["example_commitment_sentence_idx"]
        is_commitment_juncture = (
            pd.NA
            if not is_usable_example
            else bool(example_commitment_idx is not None and sentence_idx == int(example_commitment_idx))
        )
        target_value = (
            float("nan")
            if not is_usable_example
            else 1.0
            if bool(example_commitment_idx is not None and sentence_idx == int(example_commitment_idx))
            else 0.0
        )

        rows.append(
            {
                "dataset": dataset_name,
                "model_name": model_name,
                "model_bundle_name": model_bundle_name,
                "example_id": example_id,
                "trace_id": example_id,
                "localization_path": str(localization_path),
                "prompt": prompt,
                "example_label": example_label,
                "example_label_source": example_label_source,
                "commitment_direction": commitment_summary["commitment_direction"],
                "sentence_text": sentence_text,
                "last_sentence_text": sentence_text,
                "previous_sentence_text": previous_sentence_text,
                "prefix_text": prefix_text,
                "full_prefix_text": full_prefix_text,
                "sentence_idx": sentence_idx,
                "sentence_idx_1based": row_idx + 1,
                "trace_length": trace_length,
                "prefix_sentence_count": row_idx + 1,
                "sentences_remaining": trace_length - (row_idx + 1),
                "deceptive_commitment_sentence_idx": commitment_summary["deceptive_commitment_sentence_idx"],
                "truthful_commitment_sentence_idx": commitment_summary["truthful_commitment_sentence_idx"],
                "example_commitment_sentence_idx": example_commitment_idx,
                "num_truthful": sentence_row.num_truthful,
                "num_valid": sentence_row.num_valid,
                "raw_start": int(sentence_row.raw_start),
                "raw_end": int(sentence_row.raw_end),
                "full_start": int(sentence_row.full_start),
                "full_end": int(sentence_row.full_end),
                "start_token": int(sentence_row.start_token),
                "end_token": int(sentence_row.end_token),
                "token_count": current_token_count,
                "current_sentence_token_count": current_token_count,
                "context_token_count": int(sentence_row.context_token_count),
                "prompt_token_count": int(prompt_token_count),
                "raw_text_context_token_count": max(0, int(sentence_row.start_token) - int(prompt_token_count)),
                "available_token_count": int(sentence_row.available_token_count),
                "prior_all_token_count": int(sentence_row.prior_all_token_count),
                "previous_sentence_token_count": previous_token_count,
                "recent_token_count": int(sentence_row.recent_token_count),
                "early_token_count": int(sentence_row.early_token_count),
                "current_sentence_char_count": current_char_count,
                "current_sentence_word_count": current_word_count,
                "previous_sentence_char_count": previous_char_count,
                "previous_sentence_word_count": previous_word_count,
                "prefix_char_count": len(prefix_text),
                "prefix_word_count": count_words(prefix_text),
                "prefix_token_count": prefix_token_count,
                "sentence_char_delta": current_char_count - previous_char_count if row_idx > 0 else 0,
                "sentence_word_delta": current_word_count - previous_word_count if row_idx > 0 else 0,
                "sentence_token_delta": current_token_count - previous_token_count if row_idx > 0 else 0,
                "normalized_position": float((row_idx + 1) / trace_length) if trace_length > 0 else float("nan"),
                "reverse_normalized_position": float((trace_length - row_idx) / trace_length) if trace_length > 0 else float("nan"),
                "deception_rate": float(current_rate) if np.isfinite(current_rate) else float("nan"),
                "prev_deception_rate": float(prev_rate) if np.isfinite(prev_rate) else float("nan"),
                "next_deception_rate": float(next_rate) if np.isfinite(next_rate) else float("nan"),
                "delta_deception_rate": delta_rate,
                "abs_delta_deception_rate": abs_delta_rate,
                "target_value": target_value,
                "has_previous_sentence": row_idx > 0,
                "has_next_sentence": row_idx + 1 < trace_length,
                "is_usable_example": is_usable_example,
                "has_deceptive_commitment": commitment_summary["has_deceptive_commitment"],
                "has_truthful_commitment": commitment_summary["has_truthful_commitment"],
                "has_example_commitment": (
                    pd.NA if commitment_summary["has_example_commitment"] is None else bool(commitment_summary["has_example_commitment"])
                ),
                "is_deceptive_commitment_juncture": bool(
                    commitment_summary["deceptive_commitment_sentence_idx"] is not None
                    and sentence_idx == int(commitment_summary["deceptive_commitment_sentence_idx"])
                ),
                "is_truthful_commitment_juncture": bool(
                    commitment_summary["truthful_commitment_sentence_idx"] is not None
                    and sentence_idx == int(commitment_summary["truthful_commitment_sentence_idx"])
                ),
                "is_commitment_juncture": is_commitment_juncture,
            }
        )

    return pd.DataFrame(rows, columns=list(OUTPUT_COLUMNS))


def parse_tfidf_text_fields(raw_value: str) -> list[str]:
    fields = [field.strip() for field in str(raw_value).split(",") if field.strip()]
    seen: set[str] = set()
    out: list[str] = []
    for field in fields:
        if field not in seen:
            seen.add(field)
            out.append(field)
    return out


def compute_and_save_tfidf_features(
    dataset_df: pd.DataFrame,
    *,
    cache_dir: Path,
    text_fields: Sequence[str],
    max_features: int,
    ngram_range: tuple[int, int],
    overwrite: bool,
) -> list[dict[str, Path]]:
    from joblib import dump
    from scipy.sparse import csr_matrix, save_npz
    from sklearn.feature_extraction.text import TfidfVectorizer

    cache_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = dataset_row_fingerprint(dataset_df)
    config_slug = slugify_token(
        f"tfidf_maxfeat_{int(max_features)}_ngram_{int(ngram_range[0])}_{int(ngram_range[1])}"
    )
    artifact_sets: list[dict[str, Path]] = []

    for text_field in text_fields:
        if text_field not in dataset_df.columns:
            raise ValueError(f"--tfidf-text-fields requested unknown column: {text_field}")

        field_slug = slugify_token(text_field)
        artifact_stem = f"{field_slug}__{config_slug}__{fingerprint}"
        matrix_path = cache_dir / f"{artifact_stem}.npz"
        meta_path = cache_dir / f"{artifact_stem}.json"
        vectorizer_path = cache_dir / f"{artifact_stem}.joblib"
        feature_names_path = cache_dir / f"{artifact_stem}__feature_names.npy"
        artifact_paths = {
            "matrix_path": matrix_path,
            "meta_path": meta_path,
            "vectorizer_path": vectorizer_path,
            "feature_names_path": feature_names_path,
        }
        if (
            matrix_path.exists()
            and meta_path.exists()
            and vectorizer_path.exists()
            and feature_names_path.exists()
            and not overwrite
        ):
            artifact_sets.append(artifact_paths)
            continue

        vectorizer = TfidfVectorizer(
            max_features=int(max_features),
            ngram_range=(int(ngram_range[0]), int(ngram_range[1])),
            lowercase=True,
            token_pattern=r"(?u)\b[\w']+\b",
            sublinear_tf=True,
            dtype=np.float32,
        )
        texts = dataset_df[text_field].fillna("").astype(str).tolist()
        vectorizer_fitted = True
        try:
            matrix = vectorizer.fit_transform(texts)
            feature_names = np.asarray(vectorizer.get_feature_names_out(), dtype=np.str_)
        except ValueError as exc:
            if "empty vocabulary" not in str(exc).lower():
                raise
            vectorizer_fitted = False
            matrix = csr_matrix((int(len(texts)), 0), dtype=np.float32)
            feature_names = np.asarray([], dtype=np.str_)

        save_npz(matrix_path, matrix, compressed=True)
        dump(vectorizer, vectorizer_path)
        np.save(feature_names_path, feature_names, allow_pickle=False)
        meta_path.write_text(
            json.dumps(
                {
                    "feature_type": "tfidf",
                    "text_field": text_field,
                    "fingerprint": fingerprint,
                    "num_rows": int(len(dataset_df)),
                    "num_features": int(matrix.shape[1]),
                    "matrix_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
                    "nnz": int(matrix.nnz),
                    "dtype": str(matrix.dtype),
                    "vectorizer_fitted": bool(vectorizer_fitted),
                    "vectorizer_params": {
                        "max_features": int(max_features),
                        "ngram_range": [int(ngram_range[0]), int(ngram_range[1])],
                        "lowercase": True,
                        "token_pattern": r"(?u)\b[\w']+\b",
                        "sublinear_tf": True,
                        "stop_words": None,
                    },
                    "row_key_columns": ["example_id", "sentence_idx"],
                    "row_order_matches_parquet": True,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        artifact_sets.append(artifact_paths)

    return artifact_sets


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    dataset_paths = resolve_dataset_paths(args.input_path, args.output)
    model_id = infer_model_id(dataset_paths, args.model_id)
    dataset_name = infer_dataset_name(dataset_paths.dataset_dir)
    model_bundle_name = dataset_paths.dataset_dir.name
    explicit_label_map = load_examples_label_map(dataset_paths.examples_path)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("This script requires a fast tokenizer because it uses offset mappings.")

    localization_paths = iter_localization_paths(
        dataset_paths.localization_dir,
        max_examples=int(args.max_examples),
        shard_id=int(args.shard_id),
        num_shards=int(args.num_shards),
    )
    if not localization_paths:
        raise FileNotFoundError(f"No localization JSON files found in {dataset_paths.localization_dir}")

    writer = StreamingParquetWriter(dataset_paths.output_path, overwrite=args.overwrite)
    buffered_frames: list[pd.DataFrame] = []
    skip_counts: Counter[str] = Counter()
    processed = 0
    successful = 0
    write_every_examples = max(1, int(args.write_every_examples))

    print(f"Dataset dir: {dataset_paths.dataset_dir}")
    print(f"Localization dir: {dataset_paths.localization_dir}")
    print(f"Output parquet: {dataset_paths.output_path}")
    print(f"Dataset name: {dataset_name}")
    print(f"Model id: {model_id}")
    print(f"Recent window sentences: {int(args.recent_window_sentences)}")
    print(f"Commitment delta threshold: {float(args.delta_threshold):.4f}")
    print(f"Localization files to process: {len(localization_paths)}")
    print(f"Loaded explicit labels from examples.jsonl: {len(explicit_label_map):,}")

    try:
        path_iter = maybe_tqdm(
            localization_paths,
            desc="Extract commitment baselines",
            total=len(localization_paths),
            disable=bool(args.disable_tqdm),
        )
        for path in path_iter:
            processed += 1
            try:
                example = json.loads(path.read_text(encoding="utf-8"))
                example_df = build_example_rows(
                    example=example,
                    dataset_name=dataset_name,
                    model_name=model_id,
                    model_bundle_name=model_bundle_name,
                    localization_path=path,
                    tokenizer=tokenizer,
                    recent_window_sentences=int(args.recent_window_sentences),
                    delta_threshold=float(args.delta_threshold),
                    explicit_label_map=explicit_label_map,
                )
            except Exception as exc:  # noqa: BLE001
                reason = getattr(exc, "reason", exc.__class__.__name__)
                skip_counts[str(reason)] += 1
                if not hasattr(exc, "reason"):
                    print(f"[error] {path.name}: {exc.__class__.__name__}: {exc}")
                maybe_raise_invalid_example(args, path, exc)
                continue

            if not example_df.empty:
                buffered_frames.append(example_df)
                successful += 1

            if processed % write_every_examples == 0:
                written = flush_feature_buffer(writer, buffered_frames)
                if written:
                    print(f"[flush] processed={processed:,} successful={successful:,} rows_written_now={written:,}")

            if processed % max(1, int(args.progress_every)) == 0:
                print(
                    f"[progress] processed={processed:,}/{len(localization_paths):,} "
                    f"successful={successful:,} skipped={sum(skip_counts.values()):,}"
                )

        final_written = flush_feature_buffer(writer, buffered_frames)
        writer.close()
    except Exception:
        writer.abort()
        raise

    if writer.rows_written == 0:
        create_empty_output_file(dataset_paths.output_path)
        print(f"Created empty output file: {dataset_paths.output_path}")
    else:
        print(f"Wrote commitment text/structural baseline dataset to: {dataset_paths.output_path}")
    print(f"Rows written: {writer.rows_written:,} (+ final flush {final_written:,})")
    if skip_counts:
        print("Skipped examples by reason:")
        for reason, count in sorted(skip_counts.items()):
            print(f"  - {reason}: {count}")

    if args.compute_tfidf and dataset_paths.output_path.exists():
        text_fields = parse_tfidf_text_fields(args.tfidf_text_fields)
        if not text_fields:
            raise ValueError("--compute-tfidf requires at least one --tfidf-text-fields column.")
        tfidf_cache_dir = (
            Path(args.tfidf_cache_dir).expanduser().resolve()
            if args.tfidf_cache_dir
            else dataset_paths.dataset_dir / DEFAULT_TFIDF_CACHE_DIRNAME
        )
        ngram_range = (int(args.tfidf_min_ngram), int(args.tfidf_max_ngram))
        if ngram_range[0] <= 0 or ngram_range[1] <= 0 or ngram_range[0] > ngram_range[1]:
            raise ValueError("--tfidf-min-ngram/--tfidf-max-ngram must define a valid positive range.")
        print(f"TF-IDF cache dir: {tfidf_cache_dir}")
        print(f"TF-IDF text fields: {text_fields}")
        print(f"TF-IDF max features: {int(args.tfidf_max_features)}")
        print(f"TF-IDF ngram range: {ngram_range}")
        print("TF-IDF lowercase: True")
        print("TF-IDF sublinear_tf: True")
        print("TF-IDF stop_words: None")

        dataset_df = pd.read_parquet(dataset_paths.output_path)
        artifact_sets = compute_and_save_tfidf_features(
            dataset_df,
            cache_dir=tfidf_cache_dir,
            text_fields=text_fields,
            max_features=int(args.tfidf_max_features),
            ngram_range=ngram_range,
            overwrite=bool(args.overwrite_tfidf_cache),
        )
        print("TF-IDF caches:")
        for artifact_paths in artifact_sets:
            print(
                "  - "
                f"matrix={artifact_paths['matrix_path']} :: "
                f"vectorizer={artifact_paths['vectorizer_path']} :: "
                f"meta={artifact_paths['meta_path']}"
            )


if __name__ == "__main__":
    main()
