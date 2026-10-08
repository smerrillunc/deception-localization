#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


DEFAULT_ATTN_IMPLEMENTATION = "eager"
DEFAULT_OUTPUT_NAME = "attention_features.parquet"
DEFAULT_WRITE_EVERY_EXAMPLES = 32
DEFAULT_PROGRESS_EVERY = 25

KNOWN_MODEL_IDS = {
    "DeepSeek-R1-Distill-Qwen-7B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
}

METADATA_COLUMNS = [
    "example_id",
    "sentence_idx",
    "sentence_text",
    "deception_rate",
    "num_truthful",
    "num_valid",
    "raw_start",
    "raw_end",
    "full_start",
    "full_end",
    "start_token",
    "end_token",
    "token_count",
    "context_token_count",
    "prompt_token_count",
    "raw_text_context_token_count",
]

STRING_COLUMNS = [
    "example_id",
    "sentence_text",
]

FLOAT_COLUMNS = [
    "deception_rate",
]

INT_COLUMNS = [
    "sentence_idx",
    "num_truthful",
    "num_valid",
    "raw_start",
    "raw_end",
    "full_start",
    "full_end",
    "start_token",
    "end_token",
    "token_count",
    "context_token_count",
    "prompt_token_count",
    "raw_text_context_token_count",
]


@dataclass(frozen=True)
class DatasetPaths:
    dataset_dir: Path
    localization_dir: Path
    output_path: Path
    examples_path: Path


class ExampleValidationError(RuntimeError):
    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class TokenizedSentenceAlignment:
    input_ids: list[int]
    offsets: list[tuple[int, int]]
    aligned_sentence_df: pd.DataFrame
    prompt_token_count: int
    used_decoded_fallback: bool


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build reasoning-trace attention features from localization JSON files and "
            "write them as a sentence-level parquet dataset."
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
        help="Parquet output path. Defaults to <dataset_dir>/attention_features.parquet.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Hugging Face model id. Defaults to the value inferred from examples.jsonl.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run on: auto, cpu, cuda, or cuda:<idx>.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
        help="Model load dtype.",
    )
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default=DEFAULT_ATTN_IMPLEMENTATION,
        help="Attention implementation passed to from_pretrained.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=False,
        help="Pass trust_remote_code=True to the tokenizer, config, and model loaders.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=0,
        help="Optional cap on the number of localization JSON files to process.",
    )
    parser.add_argument(
        "--write-every-examples",
        type=int,
        default=DEFAULT_WRITE_EVERY_EXAMPLES,
        help="Flush buffered example feature frames to parquet after this many examples.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=DEFAULT_PROGRESS_EVERY,
        help="Print a progress update every N processed files.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Fail immediately on invalid examples instead of skipping them.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite an existing parquet output.",
    )
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


def infer_model_id(dataset_paths: DatasetPaths, override: Optional[str]) -> str:
    if override:
        return override

    if dataset_paths.examples_path.exists():
        with dataset_paths.examples_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                model_id = row.get("model_name") or row.get("meta_model_name")
                if isinstance(model_id, str) and model_id.strip():
                    return model_id.strip()
                break

    inferred = KNOWN_MODEL_IDS.get(dataset_paths.dataset_dir.name)
    if inferred:
        return inferred

    raise ValueError(
        "Could not infer the Hugging Face model id. Pass --model-id explicitly or add it to "
        f"{dataset_paths.examples_path}."
    )


def pick_best_device() -> tuple[str, pd.DataFrame]:
    if not torch.cuda.is_available():
        return "cpu", pd.DataFrame(columns=["gpu", "free_gb", "total_gb", "device_name"])

    rows = []
    for gpu_idx in range(torch.cuda.device_count()):
        free_bytes, total_bytes = torch.cuda.mem_get_info(gpu_idx)
        rows.append(
            {
                "gpu": gpu_idx,
                "free_gb": round(free_bytes / (1024 ** 3), 2),
                "total_gb": round(total_bytes / (1024 ** 3), 2),
                "device_name": torch.cuda.get_device_name(gpu_idx),
            }
        )
    gpu_df = pd.DataFrame(rows).sort_values(["free_gb", "gpu"], ascending=[False, True]).reset_index(drop=True)
    return f"cuda:{int(gpu_df.loc[0, 'gpu'])}", gpu_df


def resolve_device(device_arg: str) -> tuple[str, pd.DataFrame]:
    if device_arg == "auto":
        return pick_best_device()
    return device_arg, pd.DataFrame(columns=["gpu", "free_gb", "total_gb", "device_name"])


def resolve_dtype(dtype_arg: str, device: str) -> torch.dtype:
    if dtype_arg == "float32":
        return torch.float32
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    if device.startswith("cuda"):
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def validate_shard_args(*, shard_id: int, num_shards: int) -> None:
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if shard_id < 0:
        raise ValueError(f"shard_id must be >= 0, got {shard_id}")
    if shard_id >= num_shards:
        raise ValueError(f"shard_id must be in [0, num_shards), got shard_id={shard_id}, num_shards={num_shards}")


def iter_localization_paths(
    localization_dir: Path,
    *,
    max_examples: int,
    shard_id: int = 0,
    num_shards: int = 1,
) -> list[Path]:
    validate_shard_args(shard_id=shard_id, num_shards=num_shards)
    paths = sorted(localization_dir.glob("*.json"))
    if max_examples > 0:
        paths = paths[:max_examples]
    if num_shards == 1:
        return paths
    return paths[shard_id::num_shards]


def build_localized_sentence_df(example: dict[str, Any]) -> pd.DataFrame:
    rows = []
    history = sorted(example.get("history", []), key=lambda item: int(item["sentence_idx_inclusive"]))
    for hist in history:
        raw_start, raw_end = hist["char_span"]
        rows.append(
            {
                "sentence_idx": int(hist["sentence_idx_inclusive"]),
                "sentence_text": hist["sentence_text"],
                "deception_rate": float(hist["deception_rate"]) if hist.get("deception_rate") is not None else np.nan,
                "num_truthful": hist.get("num_truthful"),
                "num_valid": hist.get("num_valid"),
                "raw_start": int(raw_start),
                "raw_end": int(raw_end),
                "full_start": int(raw_start),
                "full_end": int(raw_end),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "sentence_idx",
                "sentence_text",
                "deception_rate",
                "num_truthful",
                "num_valid",
                "raw_start",
                "raw_end",
                "full_start",
                "full_end",
            ]
        )
    return pd.DataFrame(rows).sort_values("sentence_idx").reset_index(drop=True)


def add_span_match_columns(full_text: str, sentence_df: pd.DataFrame) -> pd.DataFrame:
    sentence_df = sentence_df.copy()
    sentence_df["span_text"] = sentence_df.apply(
        lambda row: full_text[int(row["full_start"]): int(row["full_end"])],
        axis=1,
    )
    sentence_df["span_matches"] = sentence_df["span_text"] == sentence_df["sentence_text"]
    return sentence_df


def token_indices_for_char_span(
    offsets: Sequence[Sequence[int]],
    start_char: int,
    end_char: int,
) -> list[int]:
    token_idxs: list[int] = []
    for token_idx, (tok_start, tok_end) in enumerate(offsets):
        if tok_start == tok_end:
            continue
        midpoint = (tok_start + tok_end) / 2.0
        if start_char <= midpoint < end_char:
            token_idxs.append(token_idx)
    if token_idxs:
        return token_idxs
    # Some localized sentence spans can be a single punctuation character that
    # lives inside a tokenizer piece like "..." or merged whitespace+punctuation.
    # In that case there is no token midpoint inside the span, so fall back to
    # any token that overlaps the requested character interval.
    for token_idx, (tok_start, tok_end) in enumerate(offsets):
        if tok_start == tok_end:
            continue
        if int(tok_start) < int(end_char) and int(tok_end) > int(start_char):
            token_idxs.append(token_idx)
    return token_idxs


def _align_localized_sentences_to_offsets(
    offsets: Sequence[Sequence[int]],
    sentence_df: pd.DataFrame,
) -> pd.DataFrame:
    aligned = sentence_df.copy()
    token_lists = []
    start_tokens = []
    end_tokens = []
    for row in aligned.itertuples():
        token_idxs = token_indices_for_char_span(offsets, int(row.full_start), int(row.full_end))
        token_lists.append(token_idxs)
        start_tokens.append(token_idxs[0] if token_idxs else None)
        end_tokens.append(token_idxs[-1] if token_idxs else None)

    aligned["token_indices"] = token_lists
    aligned["start_token"] = start_tokens
    aligned["end_token"] = end_tokens
    aligned["token_count"] = aligned["token_indices"].apply(len)
    aligned["context_token_count"] = aligned["start_token"].fillna(0).astype(int)
    return aligned


def align_localized_sentences_to_tokens(
    offsets: Sequence[Sequence[int]],
    sentence_df: pd.DataFrame,
) -> pd.DataFrame:
    return _align_localized_sentences_to_offsets(offsets, sentence_df)


def count_tokens_before_char_boundary(
    offset_mapping: Sequence[Sequence[int]],
    boundary_char: int,
) -> int:
    return int(sum(1 for _, end in offset_mapping if int(end) <= int(boundary_char)))


def offsets_need_decoded_fallback(
    offset_mapping: Sequence[Sequence[int]],
    full_text: str,
) -> bool:
    prev_end: Optional[int] = None
    for token_start, token_end in offset_mapping:
        start = int(token_start)
        end = int(token_end)
        if start == end:
            continue
        if end < start:
            return True
        if prev_end is not None:
            if start < prev_end:
                return True
            if start > prev_end and full_text[prev_end:start].strip():
                return True
        elif start > 0 and full_text[:start].strip():
            return True
        prev_end = end
    if prev_end is None:
        return False
    return bool(prev_end < len(full_text) and full_text[prev_end:].strip())


def _build_decoded_token_offsets(
    tokenizer: Any,
    input_ids: Sequence[int],
) -> tuple[list[tuple[int, int]], str]:
    token_pieces = [
        tokenizer.decode([int(token_id)], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        for token_id in input_ids
    ]
    decoded_text = "".join(token_pieces)
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for piece in token_pieces:
        start = cursor
        cursor += len(piece)
        offsets.append((start, cursor))
    return offsets, decoded_text


def _build_raw_to_decoded_prefix_map(
    raw_text: str,
    decoded_text: str,
) -> list[int]:
    prefix_map = [0] * (len(raw_text) + 1)
    matcher = SequenceMatcher(a=raw_text, b=decoded_text, autojunk=False)
    for tag, raw_start, raw_end, decoded_start, decoded_end in matcher.get_opcodes():
        if tag == "insert":
            prefix_map[raw_start] = max(prefix_map[raw_start], int(decoded_end))
            continue

        raw_len = int(raw_end) - int(raw_start)
        decoded_len = int(decoded_end) - int(decoded_start)

        if tag == "equal":
            for rel_idx in range(raw_len + 1):
                prefix_map[int(raw_start) + rel_idx] = int(decoded_start) + rel_idx
            continue

        if tag == "delete":
            for rel_idx in range(raw_len + 1):
                prefix_map[int(raw_start) + rel_idx] = int(decoded_start)
            continue

        for rel_idx in range(raw_len + 1):
            mapped = int(decoded_start)
            if raw_len > 0:
                mapped += (rel_idx * decoded_len) // raw_len
            else:
                mapped = int(decoded_end)
            prefix_map[int(raw_start) + rel_idx] = mapped

    prefix_map[-1] = len(decoded_text)
    running_max = 0
    for idx, value in enumerate(prefix_map):
        if value < running_max:
            prefix_map[idx] = running_max
        else:
            running_max = value
    return prefix_map


def _try_align_localized_sentences_via_decoded_text(
    *,
    tokenizer: Any,
    input_ids: Sequence[int],
    full_text: str,
    sentence_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[tuple[int, int]], str] | None:
    decoded_offsets, decoded_text = _build_decoded_token_offsets(tokenizer, input_ids)
    if len(decoded_offsets) != len(input_ids):
        return None

    raw_to_decoded_prefix = _build_raw_to_decoded_prefix_map(full_text, decoded_text)
    if not raw_to_decoded_prefix or raw_to_decoded_prefix[-1] != len(decoded_text):
        return None

    mapped_sentence_df = sentence_df.copy()
    mapped_starts = [int(raw_to_decoded_prefix[int(start)]) for start in sentence_df["full_start"].tolist()]
    mapped_ends = [int(raw_to_decoded_prefix[int(end)]) for end in sentence_df["full_end"].tolist()]
    mapped_sentence_df["full_start"] = mapped_starts
    mapped_sentence_df["full_end"] = [max(start, end) for start, end in zip(mapped_starts, mapped_ends)]

    aligned = _align_localized_sentences_to_offsets(decoded_offsets, mapped_sentence_df)
    aligned["full_start"] = sentence_df["full_start"].to_numpy()
    aligned["full_end"] = sentence_df["full_end"].to_numpy()
    return aligned, decoded_offsets, decoded_text


def tokenize_and_align_localized_sentences(
    *,
    tokenizer: Any,
    full_text: str,
    sentence_df: pd.DataFrame,
    raw_text_start_char: int = 0,
) -> TokenizedSentenceAlignment:
    tokenized = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    input_ids_list = [int(token_id) for token_id in tokenized["input_ids"]]
    raw_offsets = [(int(start), int(end)) for start, end in tokenized["offset_mapping"]]

    raw_aligned = _align_localized_sentences_to_offsets(raw_offsets, sentence_df)
    use_fallback = offsets_need_decoded_fallback(raw_offsets, full_text)
    raw_zero_count = int((raw_aligned["token_count"] == 0).sum()) if not raw_aligned.empty else 0

    if not use_fallback and raw_zero_count == 0:
        return TokenizedSentenceAlignment(
            input_ids=input_ids_list,
            offsets=raw_offsets,
            aligned_sentence_df=raw_aligned,
            prompt_token_count=count_tokens_before_char_boundary(raw_offsets, raw_text_start_char),
            used_decoded_fallback=False,
        )

    repaired = _try_align_localized_sentences_via_decoded_text(
        tokenizer=tokenizer,
        input_ids=input_ids_list,
        full_text=full_text,
        sentence_df=sentence_df,
    )
    if repaired is not None:
        repaired_aligned, repaired_offsets, _decoded_text = repaired
        repaired_zero_count = int((repaired_aligned["token_count"] == 0).sum()) if not repaired_aligned.empty else 0
        if use_fallback or repaired_zero_count < raw_zero_count:
            raw_to_decoded_prefix = _build_raw_to_decoded_prefix_map(full_text, _decoded_text)
            repaired_boundary = int(raw_to_decoded_prefix[int(raw_text_start_char)])
            return TokenizedSentenceAlignment(
                input_ids=input_ids_list,
                offsets=repaired_offsets,
                aligned_sentence_df=repaired_aligned,
                prompt_token_count=count_tokens_before_char_boundary(repaired_offsets, repaired_boundary),
                used_decoded_fallback=True,
            )

    return TokenizedSentenceAlignment(
        input_ids=input_ids_list,
        offsets=raw_offsets,
        aligned_sentence_df=raw_aligned,
        prompt_token_count=count_tokens_before_char_boundary(raw_offsets, raw_text_start_char),
        used_decoded_fallback=False,
    )


def compute_attention_features(
    attentions: Sequence[torch.Tensor],
    aligned_sentence_df: pd.DataFrame,
    *,
    prompt_token_count: int,
    example_id: str,
) -> pd.DataFrame:
    records = []
    num_heads = int(attentions[0].shape[1])

    for row in aligned_sentence_df.itertuples():
        q_idx = torch.tensor(row.token_indices, device=attentions[0].device, dtype=torch.long)
        start_token = int(row.start_token)
        end_token = int(row.end_token)

        feature_row: dict[str, Any] = {
            "example_id": example_id,
            "sentence_idx": int(row.sentence_idx),
            "sentence_text": row.sentence_text,
            "deception_rate": float(row.deception_rate),
            "num_truthful": row.num_truthful,
            "num_valid": row.num_valid,
            "raw_start": int(row.raw_start),
            "raw_end": int(row.raw_end),
            "full_start": int(row.full_start),
            "full_end": int(row.full_end),
            "start_token": start_token,
            "end_token": end_token,
            "token_count": int(row.token_count),
            "context_token_count": int(row.context_token_count),
            "prompt_token_count": int(prompt_token_count),
            "raw_text_context_token_count": max(0, start_token - int(prompt_token_count)),
        }

        for layer_idx, layer_attn in enumerate(attentions):
            layer = layer_attn[0].to(dtype=torch.float32)
            avg_attn = layer[:, q_idx, :].mean(dim=1)

            raw_context_start = min(int(prompt_token_count), start_token)
            raw_context_slice = avg_attn[:, raw_context_start:start_token]
            sentence_slice = avg_attn[:, start_token : end_token + 1]

            raw_context_mass = raw_context_slice.sum(dim=1)
            sentence_mass = sentence_slice.sum(dim=1)
            raw_context_width = float(raw_context_slice.shape[1])
            sentence_width = float(sentence_slice.shape[1])

            raw_context_mass_per_token = torch.full(
                (num_heads,),
                float("nan"),
                device=layer.device,
                dtype=torch.float32,
            )
            if raw_context_width > 0:
                raw_context_mass_per_token = raw_context_mass / raw_context_width

            sentence_mass_per_token = torch.full(
                (num_heads,),
                float("nan"),
                device=layer.device,
                dtype=torch.float32,
            )
            if sentence_width > 0:
                sentence_mass_per_token = sentence_mass / sentence_width

            lb_denom = raw_context_mass_per_token + sentence_mass_per_token
            lb = torch.full((num_heads,), float("nan"), device=layer.device, dtype=torch.float32)
            valid_lb = lb_denom > 0
            lb[valid_lb] = raw_context_mass_per_token[valid_lb] / lb_denom[valid_lb]

            entropy_norm = torch.full((num_heads,), float("nan"), device=layer.device, dtype=torch.float32)
            if raw_context_slice.shape[1] > 0:
                valid_context = raw_context_mass > 0
                if valid_context.any():
                    context_norm = raw_context_slice[valid_context] / raw_context_mass[valid_context].unsqueeze(1).clamp_min(1e-12)
                    entropy_vals = -(context_norm * context_norm.clamp_min(1e-12).log()).sum(dim=1)
                    if raw_context_width > 1.0:
                        entropy_norm_vals = torch.clamp(
                            entropy_vals / float(np.log(raw_context_width)),
                            min=0.0,
                            max=1.0,
                        )
                    else:
                        entropy_norm_vals = torch.zeros_like(entropy_vals)
                    entropy_norm[valid_context] = entropy_norm_vals

            entropy_norm_vals = entropy_norm.detach().cpu().tolist()
            lb_vals = lb.detach().cpu().tolist()
            for head_idx in range(num_heads):
                feature_row[f"entropy_norm_{layer_idx}_{head_idx}"] = entropy_norm_vals[head_idx]
                feature_row[f"lb_{layer_idx}_{head_idx}"] = lb_vals[head_idx]

        records.append(feature_row)

    return pd.DataFrame(records)


def build_feature_columns(num_layers: int, num_heads: int) -> list[str]:
    columns: list[str] = []
    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            columns.append(f"entropy_norm_{layer_idx}_{head_idx}")
            columns.append(f"lb_{layer_idx}_{head_idx}")
    return columns


def build_empty_feature_frame(num_layers: int, num_heads: int) -> pd.DataFrame:
    return pd.DataFrame(columns=METADATA_COLUMNS + build_feature_columns(num_layers, num_heads))


def coerce_feature_frame_columns(
    feature_df: pd.DataFrame,
    *,
    ordered_columns: Sequence[str],
) -> pd.DataFrame:
    df = feature_df.copy()
    for column in ordered_columns:
        if column not in df.columns:
            df[column] = np.nan
    df = df.loc[:, list(ordered_columns)]

    for column in STRING_COLUMNS:
        df[column] = df[column].astype("string")
    for column in INT_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce").astype("Int64")
    feature_columns = [column for column in df.columns if column not in STRING_COLUMNS and column not in INT_COLUMNS]
    for column in FLOAT_COLUMNS + [column for column in feature_columns if column not in FLOAT_COLUMNS]:
        df[column] = pd.to_numeric(df[column], errors="coerce").astype("float64")
    return df


def cleanup_tensors() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()


def extract_example_feature_df(
    *,
    example: dict[str, Any],
    tokenizer: Any,
    model: Any,
    device: str,
) -> pd.DataFrame:
    example_id = example.get("example_id")
    if not isinstance(example_id, str) or not example_id:
        raise ExampleValidationError("missing_example_id", "Localization example is missing example_id.")

    full_text = example.get("raw_text")
    if not isinstance(full_text, str) or not full_text:
        raise ExampleValidationError("missing_raw_text", f"{example_id} is missing raw_text.")

    localized_sentence_df = build_localized_sentence_df(example)
    if localized_sentence_df.empty:
        raise ExampleValidationError("empty_history", f"{example_id} has no localized history entries.")

    localized_sentence_df = add_span_match_columns(full_text, localized_sentence_df)
    if not localized_sentence_df["span_matches"].all():
        bad_count = int((~localized_sentence_df["span_matches"]).sum())
        raise ExampleValidationError(
            "span_mismatch",
            f"{example_id} has {bad_count} localized sentence spans that do not match raw_text.",
        )

    token_alignment = tokenize_and_align_localized_sentences(
        tokenizer=tokenizer,
        full_text=full_text,
        sentence_df=localized_sentence_df,
        raw_text_start_char=0,
    )
    input_ids_list = token_alignment.input_ids
    if not input_ids_list:
        raise ExampleValidationError("no_tokens", f"{example_id} tokenized to zero tokens.")

    aligned_sentence_df = token_alignment.aligned_sentence_df
    if not (aligned_sentence_df["token_count"] > 0).all():
        bad_count = int((aligned_sentence_df["token_count"] == 0).sum())
        raise ExampleValidationError(
            "unmapped_sentence",
            f"{example_id} has {bad_count} localized sentences that failed to map to tokens.",
        )

    modeling_sentence_df = aligned_sentence_df.loc[aligned_sentence_df["start_token"].fillna(0).astype(int) > 0].copy()
    if modeling_sentence_df.empty:
        return build_empty_feature_frame(num_layers=0, num_heads=0)

    input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=device)
    try:
        with torch.no_grad():
            outputs = model(input_ids=input_ids, output_attentions=True, use_cache=False)
        attentions = outputs.attentions
        feature_df = compute_attention_features(
            attentions,
            modeling_sentence_df,
            prompt_token_count=int(token_alignment.prompt_token_count),
            example_id=example_id,
        )
    finally:
        if "outputs" in locals():
            del outputs
        if "attentions" in locals():
            del attentions
        del input_ids
        cleanup_tensors()

    return feature_df


class StreamingParquetWriter:
    def __init__(self, output_path: Path, *, overwrite: bool) -> None:
        self.output_path = output_path
        self.temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        self.overwrite = overwrite
        self.writer = None
        self.rows_written = 0

        if self.output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Output already exists: {self.output_path}. Pass --overwrite to replace it."
            )
        if self.temp_path.exists():
            if overwrite:
                self.temp_path.unlink()
            else:
                raise FileExistsError(
                    f"Temporary output already exists: {self.temp_path}. "
                    "Pass --overwrite to replace it."
                )

    def write(self, df: pd.DataFrame) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pandas(df, preserve_index=False)
        if self.writer is None:
            self.temp_path.parent.mkdir(parents=True, exist_ok=True)
            self.writer = pq.ParquetWriter(self.temp_path, table.schema, compression="snappy")
        self.writer.write_table(table)
        self.rows_written += len(df)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
        if self.output_path.exists():
            self.output_path.unlink()
        if self.temp_path.exists():
            self.temp_path.replace(self.output_path)

    def abort(self) -> None:
        if self.writer is not None:
            self.writer.close()
        if self.temp_path.exists():
            self.temp_path.unlink()


def flush_feature_buffer(
    writer: StreamingParquetWriter,
    buffer: list[pd.DataFrame],
    *,
    ordered_columns: Sequence[str],
) -> int:
    if not buffer:
        return 0
    chunk_df = pd.concat(buffer, ignore_index=True)
    buffer.clear()
    chunk_df = coerce_feature_frame_columns(chunk_df, ordered_columns=ordered_columns)
    writer.write(chunk_df)
    return len(chunk_df)


def maybe_raise_invalid_example(
    args: argparse.Namespace,
    path: Path,
    exc: Exception,
) -> None:
    if args.strict:
        raise RuntimeError(f"Failed to process {path}: {exc}") from exc


def maybe_raise_runtime_error(
    args: argparse.Namespace,
    path: Path,
    exc: RuntimeError,
) -> None:
    if args.strict:
        raise RuntimeError(f"Runtime failure while processing {path}: {exc}") from exc


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    dataset_paths = resolve_dataset_paths(args.input_path, args.output)
    model_id = infer_model_id(dataset_paths, args.model_id)
    device, gpu_df = resolve_device(args.device)
    model_dtype = resolve_dtype(args.dtype, device)
    write_every_examples = max(1, int(args.write_every_examples))

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=args.trust_remote_code)
    num_layers = int(config.num_hidden_layers)
    num_heads = int(config.num_attention_heads)
    ordered_columns = METADATA_COLUMNS + build_feature_columns(num_layers, num_heads)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("This script requires a fast tokenizer because it uses offset mappings.")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=model_dtype,
        low_cpu_mem_usage=True,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
    )
    model.to(device)
    model.eval()

    localization_paths = iter_localization_paths(
        dataset_paths.localization_dir,
        max_examples=int(args.max_examples),
    )
    if not localization_paths:
        raise FileNotFoundError(f"No localization JSON files found in {dataset_paths.localization_dir}")

    writer = StreamingParquetWriter(dataset_paths.output_path, overwrite=args.overwrite)
    skip_counts: Counter[str] = Counter()
    buffered_frames: list[pd.DataFrame] = []
    processed = 0
    successful = 0

    print(f"Dataset dir: {dataset_paths.dataset_dir}")
    print(f"Localization dir: {dataset_paths.localization_dir}")
    print(f"Output parquet: {dataset_paths.output_path}")
    print(f"Model id: {model_id}")
    print(f"Device: {device}")
    print(f"Model dtype: {model_dtype}")
    print(f"Layers: {num_layers} | Heads: {num_heads} | Feature columns: {len(ordered_columns) - len(METADATA_COLUMNS)}")
    if not gpu_df.empty:
        print("Visible GPUs:")
        print(gpu_df.to_string(index=False))
    print(f"Localization files to process: {len(localization_paths)}")

    try:
        for path in localization_paths:
            processed += 1
            try:
                example = json.loads(path.read_text(encoding="utf-8"))
                feature_df = extract_example_feature_df(
                    example=example,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                )
            except json.JSONDecodeError as exc:
                skip_counts["invalid_json"] += 1
                maybe_raise_invalid_example(args, path, exc)
                feature_df = None
            except ExampleValidationError as exc:
                skip_counts[exc.reason] += 1
                maybe_raise_invalid_example(args, path, exc)
                feature_df = None
            except (KeyError, TypeError, ValueError, IndexError) as exc:
                skip_counts["malformed_example"] += 1
                maybe_raise_invalid_example(args, path, exc)
                feature_df = None
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    skip_counts["oom"] += 1
                    cleanup_tensors()
                    maybe_raise_runtime_error(args, path, exc)
                    feature_df = None
                else:
                    raise

            if feature_df is not None and not feature_df.empty:
                buffered_frames.append(feature_df)
                successful += 1
            elif feature_df is not None and feature_df.empty:
                skip_counts["no_prior_reasoning_context"] += 1

            if len(buffered_frames) >= write_every_examples:
                flush_feature_buffer(
                    writer,
                    buffered_frames,
                    ordered_columns=ordered_columns,
                )

            if int(args.progress_every) > 0 and processed % int(args.progress_every) == 0:
                buffered_row_count = sum(len(df) for df in buffered_frames)
                print(
                    f"Processed {processed}/{len(localization_paths)} files | "
                    f"successful={successful} | skipped={sum(skip_counts.values())} | "
                    f"rows_buffered_or_written={writer.rows_written + buffered_row_count}"
                )

        flush_feature_buffer(
            writer,
            buffered_frames,
            ordered_columns=ordered_columns,
        )

        if writer.rows_written == 0:
            writer.write(
                coerce_feature_frame_columns(
                    build_empty_feature_frame(num_layers=num_layers, num_heads=num_heads),
                    ordered_columns=ordered_columns,
                )
            )
        writer.close()
    except Exception:
        writer.abort()
        raise
    finally:
        del model
        cleanup_tensors()

    print(f"Wrote attention features to: {dataset_paths.output_path}")
    print(f"Processed files: {processed}")
    print(f"Examples with output rows: {successful}")
    print(f"Total parquet rows: {writer.rows_written}")
    if skip_counts:
        print("Skipped examples by reason:")
        for reason, count in sorted(skip_counts.items()):
            print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()
