#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Sequence

import h5py
import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import BitsAndBytesConfig
except Exception:  # noqa: BLE001
    BitsAndBytesConfig = None

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
    cleanup_tensors,
    infer_model_id,
    iter_localization_paths,
    maybe_raise_invalid_example,
    maybe_raise_runtime_error,
    resolve_device,
    resolve_dtype,
    tokenize_and_align_localized_sentences,
)

DEFAULT_ATTN_IMPLEMENTATION = "eager"
DEFAULT_FEATURE_OUTPUT_NAME = "prefix_deception_features.parquet"
DEFAULT_ACTIVATION_OUTPUT_NAME = "prefix_deception_activations.h5"
DEFAULT_WRITE_EVERY_EXAMPLES = 32
DEFAULT_PROGRESS_EVERY = 25
DEFAULT_RECENT_WINDOW_TOKENS = 64
DEFAULT_NUM_PREFIX_SENTENCES = 5
DEFAULT_COMPRESSION = "lzf"
DEFAULT_GZIP_LEVEL = 4
DEFAULT_MAX_INPUT_TOKENS = 10_000
EPS = 1e-6

ATTENTION_METRIC_NAMES = (
    "current_vs_prior",
    "current_vs_prev",
    "current_vs_prev3",
    "recent_vs_early",
    "prev_share_of_prior",
    "current_share_total",
    "entropy_prior",
    "entropy_full",
    "top1_prior",
    "top5_prior",
    "herfindahl_prior",
    "effective_support_prior",
)
HEAD_SUMMARY_NAMES = ("mean", "std", "max")
TRANSITION_PREFIXES = ("delta", "slope3", "devrun", "min_gap", "max_gap")
ACTIVATION_SCALAR_FEATURE_NAMES = (
    "act_cos_cur_prev",
    "act_cos_cur_mean3",
    "act_norm_cur",
    "act_norm_jump",
    "act_delta_vec_norm",
    "act_pairwise_recent_cohesion",
)

METADATA_COLUMNS = (
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
    "available_token_count",
    "prior_all_token_count",
    "previous_sentence_token_count",
    "previous_three_sentence_token_count",
    "recent_token_count",
    "early_token_count",
    "available_prefix_sentence_count",
    "total_input_token_count",
)
STRING_COLUMNS = ("example_id", "sentence_text")
FLOAT_COLUMNS = ("deception_rate",)
INT_COLUMNS = tuple(column for column in METADATA_COLUMNS if column not in STRING_COLUMNS and column not in FLOAT_COLUMNS)


class DualOutputs:
    def __init__(self, feature_output: Path, activation_output: Path) -> None:
        self.feature_output = feature_output
        self.activation_output = activation_output


def maybe_tqdm(iterable: Sequence[Any], *, desc: str, total: Optional[int] = None, disable: bool = False):
    if disable or _tqdm is None:
        return iterable
    return _tqdm(iterable, desc=desc, total=total)


def progress_write(progress_iter: Any, message: str) -> None:
    if _tqdm is not None and hasattr(progress_iter, "write"):
        progress_iter.write(message)
    else:
        print(message)


def format_skip_message(path: Path, reason: str, exc: Exception) -> str:
    return f"[skip:{reason}] {path.name}: {exc}"


def resolve_cuda_index(device: str) -> int:
    if device == "cuda":
        return int(torch.cuda.current_device())
    if device.startswith("cuda:"):
        return int(device.split(":", 1)[1])
    raise ValueError(f"8-bit loading requires a CUDA device, got: {device}")


def resolve_requested_device_map(device_map_arg: str) -> Optional[str]:
    value = str(device_map_arg).strip().lower()
    if value in {"", "single", "none"}:
        return None
    return str(device_map_arg).strip()


def infer_model_input_device(model: Any, fallback_device: str) -> str:
    try:
        for parameter in model.parameters():
            param_device = getattr(parameter, "device", None)
            if param_device is None or str(param_device) == "meta":
                continue
            return str(param_device)
    except Exception:  # noqa: BLE001
        pass

    hf_device_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_device_map, dict):
        for target in hf_device_map.values():
            if isinstance(target, int):
                return f"cuda:{int(target)}"
            if isinstance(target, str) and target not in {"cpu", "disk", "meta"}:
                return str(target)

    return str(fallback_device)


def summarize_hf_device_map(model: Any) -> str:
    hf_device_map = getattr(model, "hf_device_map", None)
    if not isinstance(hf_device_map, dict) or not hf_device_map:
        return "single-device"
    counts = Counter(str(target) for target in hf_device_map.values())
    summary_parts = [f"{target}={count}" for target, count in sorted(counts.items())]
    return ", ".join(summary_parts)


def build_quant_config(*, load_in_8bit: bool, device: str) -> BitsAndBytesConfig | None:
    if not load_in_8bit:
        return None
    if BitsAndBytesConfig is None:
        raise ImportError("transformers BitsAndBytesConfig is unavailable; cannot use --load-in-8bit.")
    if not torch.cuda.is_available():
        raise ValueError("--load-in-8bit requires CUDA, but torch.cuda.is_available() is false.")
    if not str(device).startswith("cuda"):
        raise ValueError(f"--load-in-8bit requires a CUDA device, got: {device}")
    return BitsAndBytesConfig(load_in_8bit=True)


def infer_repo_quant_method(model_id: str, *, trust_remote_code: bool) -> Optional[str]:
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    quant_config = getattr(config, "quantization_config", None)
    if quant_config is None:
        return None
    if isinstance(quant_config, dict):
        quant_method = quant_config.get("quant_method")
    else:
        quant_method = getattr(quant_config, "quant_method", None)
    return str(quant_method) if quant_method else None


def build_model_load_kwargs(
    *,
    args: argparse.Namespace,
    device: str,
    model_dtype: torch.dtype,
    load_in_8bit: bool,
    requested_device_map: Optional[str],
) -> dict[str, Any]:
    model_kwargs: dict[str, Any] = {
        "low_cpu_mem_usage": True,
        "attn_implementation": args.attn_implementation,
        "trust_remote_code": args.trust_remote_code,
    }
    quant_config = build_quant_config(load_in_8bit=bool(load_in_8bit), device=device)
    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config
        model_kwargs["device_map"] = requested_device_map if requested_device_map is not None else {"": resolve_cuda_index(device)}
        model_kwargs["torch_dtype"] = model_dtype
    else:
        model_kwargs["torch_dtype"] = model_dtype
        if requested_device_map is not None:
            model_kwargs["device_map"] = requested_device_map
    return model_kwargs


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build sentence-level deception prefix features from localization JSON files. "
            "This writes both (A) a parquet table of handcrafted attention / transition / activation-summary features "
            "and (B) an HDF5 file of raw final-layer sentence-end activations."
        )
    )
    parser.add_argument("input_path", type=str, help="Dataset directory or its localization subdirectory.")
    parser.add_argument(
        "--feature-output",
        type=str,
        default=None,
        help=f"Parquet feature output path. Defaults to <dataset_dir>/{DEFAULT_FEATURE_OUTPUT_NAME}.",
    )
    parser.add_argument(
        "--activation-output",
        type=str,
        default=None,
        help=f"HDF5 activation output path. Defaults to <dataset_dir>/{DEFAULT_ACTIVATION_OUTPUT_NAME}.",
    )
    parser.add_argument("--model-id", type=str, default=None, help="HF model id. Defaults to value inferred from examples.jsonl.")
    parser.add_argument("--device", type=str, default="auto", help="Device to run on: auto, cpu, cuda, or cuda:<idx>.")
    parser.add_argument(
        "--device-map",
        type=str,
        default="single",
        help=(
            "How to place model weights. Use 'single' (default) to keep the model on one device, "
            "or a Transformers device_map string such as 'auto', 'balanced', 'balanced_low_0', or 'sequential'."
        ),
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
        "--load-in-8bit",
        action="store_true",
        default=False,
        help="Load the model with bitsandbytes 8-bit quantization. Requires CUDA.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=False,
        help="Pass trust_remote_code=True to the tokenizer and model loaders.",
    )
    parser.add_argument(
        "--recent-window-tokens",
        type=int,
        default=DEFAULT_RECENT_WINDOW_TOKENS,
        help="Number of trailing prior tokens to treat as the recent context window.",
    )
    parser.add_argument(
        "--num-prefix-sentences",
        type=int,
        default=DEFAULT_NUM_PREFIX_SENTENCES,
        help="Number of sentence-end activations to save per row, counting backward from the current sentence.",
    )
    parser.add_argument("--max-examples", type=int, default=0, help="Optional cap on number of localization files to process.")
    parser.add_argument("--shard-id", type=int, default=0, help="Zero-based shard index after sorting localization files.")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards to split the sorted files across.")
    parser.add_argument(
        "--write-every-examples",
        type=int,
        default=DEFAULT_WRITE_EVERY_EXAMPLES,
        help="Flush buffered outputs after this many examples.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=DEFAULT_PROGRESS_EVERY,
        help="Print a progress update every N processed files.",
    )
    parser.add_argument(
        "--compression",
        type=str,
        choices=("lzf", "gzip", "none"),
        default=DEFAULT_COMPRESSION,
        help="HDF5 compression for activation arrays.",
    )
    parser.add_argument("--gzip-level", type=int, default=DEFAULT_GZIP_LEVEL, help="Compression level when --compression=gzip.")
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=DEFAULT_MAX_INPUT_TOKENS,
        help="Skip examples whose prompt + raw_text tokenized length exceeds this limit.",
    )
    parser.add_argument("--strict", action="store_true", default=False, help="Fail immediately on invalid examples.")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite existing outputs.")
    parser.add_argument("--disable-tqdm", action="store_true", default=False, help="Disable tqdm progress bars.")
    parser.add_argument("--quiet-skips", action="store_true", default=False, help="Do not print a line for each skipped example.")
    return parser.parse_args(argv)


def resolve_dataset_paths(
    input_path: str | Path,
    feature_output: Optional[str | Path],
    activation_output: Optional[str | Path],
) -> tuple[DatasetPaths, DualOutputs]:
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

    feature_path = Path(feature_output).expanduser().resolve() if feature_output else dataset_dir / DEFAULT_FEATURE_OUTPUT_NAME
    activation_path = (
        Path(activation_output).expanduser().resolve() if activation_output else dataset_dir / DEFAULT_ACTIVATION_OUTPUT_NAME
    )
    dataset_paths = DatasetPaths(
        dataset_dir=dataset_dir,
        localization_dir=localization_dir,
        output_path=feature_path,
        examples_path=dataset_dir / "examples.jsonl",
    )
    return dataset_paths, DualOutputs(feature_output=feature_path, activation_output=activation_path)


def build_feature_columns(num_layers: int) -> list[str]:
    columns: list[str] = []
    for layer_idx in range(int(num_layers)):
        for metric_name in ATTENTION_METRIC_NAMES:
            for agg_name in HEAD_SUMMARY_NAMES:
                columns.append(f"{metric_name}_{agg_name}_l{layer_idx}")
    columns.extend(ACTIVATION_SCALAR_FEATURE_NAMES)

    transition_targets = [
        *(f"{metric_name}_{agg_name}_l{layer_idx}" for layer_idx in range(int(num_layers)) for metric_name in ATTENTION_METRIC_NAMES for agg_name in HEAD_SUMMARY_NAMES),
        *ACTIVATION_SCALAR_FEATURE_NAMES,
    ]
    for column in transition_targets:
        for prefix in TRANSITION_PREFIXES:
            columns.append(f"{prefix}_{column}")
    return columns


def build_empty_feature_frame(num_layers: int) -> pd.DataFrame:
    return pd.DataFrame(columns=list(METADATA_COLUMNS) + build_feature_columns(num_layers))


def coerce_feature_frame_columns(feature_df: pd.DataFrame, *, ordered_columns: Sequence[str]) -> pd.DataFrame:
    df = feature_df.copy().reindex(columns=list(ordered_columns)).copy()
    for column in STRING_COLUMNS:
        df[column] = df[column].astype("string")
    for column in INT_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce").astype("Int64")
    float_like_columns = [column for column in df.columns if column not in STRING_COLUMNS and column not in INT_COLUMNS]
    for column in float_like_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce").astype("float64")
    return df


def flush_feature_buffer(writer: StreamingParquetWriter, buffer: list[pd.DataFrame], *, ordered_columns: Sequence[str]) -> int:
    if not buffer:
        return 0
    chunk_df = pd.concat(buffer, ignore_index=True)
    buffer.clear()
    chunk_df = coerce_feature_frame_columns(chunk_df, ordered_columns=ordered_columns)
    writer.write(chunk_df)
    return len(chunk_df)


class ActivationH5Writer:
    def __init__(
        self,
        path: str | Path,
        *,
        num_prefix_sentences: int,
        overwrite: bool,
        compression: str,
        gzip_level: int,
    ) -> None:
        self.path = Path(path)
        if self.path.exists() and not overwrite:
            raise FileExistsError(f"Activation output already exists: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = h5py.File(self.path, "w")
        self.num_prefix_sentences = int(num_prefix_sentences)
        self.compression = None if compression == "none" else compression
        self.compression_opts = int(gzip_level) if compression == "gzip" else None
        self.datasets: dict[str, h5py.Dataset] = {}
        self.rows_written = 0
        self.hidden_size: Optional[int] = None
        self.file.attrs["num_prefix_sentences"] = int(num_prefix_sentences)
        self.file.attrs["activation_dtype"] = "float16"
        self.file.attrs["padding_value"] = 0.0
        self.file.attrs["slot_order"] = json.dumps(
            [f"slot {i}: sentence-end token {i} sentences back from current prefix end" for i in range(int(num_prefix_sentences))]
        )

    @staticmethod
    def _string_dtype() -> np.dtype:
        return h5py.string_dtype(encoding="utf-8")

    def _create_scalar_dataset(self, name: str, dtype: Any, chunk_rows: int) -> None:
        is_string = isinstance(dtype, np.dtype) and dtype.kind == "O"
        self.datasets[name] = self.file.create_dataset(
            name,
            shape=(0,),
            maxshape=(None,),
            chunks=(chunk_rows,),
            dtype=dtype,
            compression=None if is_string else self.compression,
            compression_opts=None if is_string else self.compression_opts,
            shuffle=False if is_string else (self.compression is not None),
        )

    def _create_array_dataset(self, name: str, tail_shape: tuple[int, ...], dtype: Any, chunk_rows: int) -> None:
        self.datasets[name] = self.file.create_dataset(
            name,
            shape=(0, *tail_shape),
            maxshape=(None, *tail_shape),
            chunks=(chunk_rows, *tail_shape),
            dtype=dtype,
            compression=self.compression,
            compression_opts=self.compression_opts,
            shuffle=self.compression is not None,
        )

    def _initialize(self, batch: dict[str, Any]) -> None:
        activations = np.asarray(batch["activations"])
        if activations.ndim != 3:
            raise ValueError(f"Expected activations with shape [rows, slots, hidden], got {activations.shape}")
        hidden_size = int(activations.shape[2])
        self.hidden_size = hidden_size
        chunk_rows = max(1, min(256, int(activations.shape[0]) if activations.shape[0] > 0 else 32))

        self._create_array_dataset("activations", (self.num_prefix_sentences, hidden_size), np.float16, chunk_rows)
        self._create_array_dataset("activation_mask", (self.num_prefix_sentences,), np.bool_, chunk_rows)
        self._create_array_dataset("prefix_sentence_indices", (self.num_prefix_sentences,), np.int32, chunk_rows)
        self._create_array_dataset("prefix_end_token_indices", (self.num_prefix_sentences,), np.int32, chunk_rows)

        for name in STRING_COLUMNS:
            self._create_scalar_dataset(name, self._string_dtype(), chunk_rows)
        for name in INT_COLUMNS:
            self._create_scalar_dataset(name, np.int64, chunk_rows)
        for name in FLOAT_COLUMNS:
            self._create_scalar_dataset(name, np.float64, chunk_rows)

    def append(self, batch: dict[str, Any]) -> None:
        if not self.datasets:
            self._initialize(batch)

        activations = np.asarray(batch["activations"], dtype=np.float16)
        row_count = int(activations.shape[0])
        if row_count == 0:
            return
        new_total = self.rows_written + row_count
        for dataset in self.datasets.values():
            dataset.resize((new_total, *dataset.shape[1:]))

        self.datasets["activations"][self.rows_written:new_total] = activations
        self.datasets["activation_mask"][self.rows_written:new_total] = np.asarray(batch["activation_mask"], dtype=np.bool_)
        self.datasets["prefix_sentence_indices"][self.rows_written:new_total] = np.asarray(
            batch["prefix_sentence_indices"], dtype=np.int32
        )
        self.datasets["prefix_end_token_indices"][self.rows_written:new_total] = np.asarray(
            batch["prefix_end_token_indices"], dtype=np.int32
        )
        for name in STRING_COLUMNS:
            self.datasets[name][self.rows_written:new_total] = np.asarray(batch[name], dtype=object)
        for name in INT_COLUMNS:
            self.datasets[name][self.rows_written:new_total] = np.asarray(batch[name], dtype=np.int64)
        for name in FLOAT_COLUMNS:
            self.datasets[name][self.rows_written:new_total] = np.asarray(batch[name], dtype=np.float64)
        self.rows_written = new_total

    def close(self) -> None:
        self.file.flush()
        self.file.close()

    def abort(self) -> None:
        try:
            self.file.close()
        finally:
            if self.path.exists():
                self.path.unlink()


def flush_activation_buffer(writer: ActivationH5Writer, buffer: list[dict[str, Any]]) -> int:
    if not buffer:
        return 0
    merged: dict[str, Any] = {}
    first = buffer[0]
    for key in first.keys():
        values = [chunk[key] for chunk in buffer]
        if isinstance(first[key], np.ndarray):
            merged[key] = np.concatenate(values, axis=0)
        else:
            flat: list[Any] = []
            for value in values:
                flat.extend(value)
            merged[key] = flat
    row_count = int(np.asarray(merged["activations"]).shape[0])
    buffer.clear()
    writer.append(merged)
    return row_count


def _empty_slice(reference: torch.Tensor) -> torch.Tensor:
    return reference[:, :0]


def _normalize_slice(slice_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mass = slice_tensor.sum(dim=1)
    probs = torch.zeros_like(slice_tensor)
    valid = mass > 0
    if valid.any():
        probs[valid] = slice_tensor[valid] / mass[valid].unsqueeze(1).clamp_min(EPS)
    return probs, mass


def _normalized_entropy(slice_tensor: torch.Tensor) -> torch.Tensor:
    width = int(slice_tensor.shape[1])
    out = torch.full((slice_tensor.shape[0],), float("nan"), device=slice_tensor.device, dtype=torch.float32)
    probs, mass = _normalize_slice(slice_tensor)
    valid = mass > 0
    if not valid.any():
        return out
    if width <= 1:
        out[valid] = 0.0
        return out
    probs_valid = probs[valid]
    entropy = -(probs_valid * probs_valid.clamp_min(EPS).log()).sum(dim=1)
    out[valid] = torch.clamp(entropy / float(math.log(width)), min=0.0, max=1.0)
    return out


def _topk_mass(slice_tensor: torch.Tensor, k: int) -> torch.Tensor:
    width = int(slice_tensor.shape[1])
    out = torch.full((slice_tensor.shape[0],), float("nan"), device=slice_tensor.device, dtype=torch.float32)
    probs, mass = _normalize_slice(slice_tensor)
    valid = mass > 0
    if not valid.any():
        return out
    k = max(1, min(int(k), width))
    out[valid] = torch.clamp(torch.topk(probs[valid], k=k, dim=1).values.sum(dim=1), min=0.0, max=1.0)
    return out


def _herfindahl(slice_tensor: torch.Tensor) -> torch.Tensor:
    width = int(slice_tensor.shape[1])
    out = torch.full((slice_tensor.shape[0],), float("nan"), device=slice_tensor.device, dtype=torch.float32)
    probs, mass = _normalize_slice(slice_tensor)
    valid = mass > 0
    if not valid.any():
        return out
    if width <= 1:
        out[valid] = 1.0
        return out
    raw_herfindahl = (probs[valid] ** 2).sum(dim=1)
    uniform_baseline = 1.0 / float(width)
    out[valid] = torch.clamp(
        (raw_herfindahl - uniform_baseline) / (1.0 - uniform_baseline),
        min=0.0,
        max=1.0,
    )
    return out


def _effective_support(slice_tensor: torch.Tensor) -> torch.Tensor:
    width = int(slice_tensor.shape[1])
    out = torch.full((slice_tensor.shape[0],), float("nan"), device=slice_tensor.device, dtype=torch.float32)
    probs, mass = _normalize_slice(slice_tensor)
    valid = mass > 0
    if not valid.any():
        return out
    if width <= 1:
        out[valid] = 1.0
        return out
    raw_herfindahl = (probs[valid] ** 2).sum(dim=1)
    raw_effective_support = 1.0 / raw_herfindahl.clamp_min(EPS)
    out[valid] = torch.clamp(
        (raw_effective_support - 1.0) / (float(width) - 1.0),
        min=0.0,
        max=1.0,
    )
    return out


def _per_token_mass(region_mass: torch.Tensor, width: int) -> torch.Tensor:
    out = torch.full_like(region_mass, float("nan"), dtype=torch.float32)
    if width <= 0:
        return out
    out[:] = torch.clamp(region_mass / float(width), min=0.0, max=1.0)
    return out


def _ratio_from_per_token_mass(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    out = torch.full_like(numerator, float("nan"), dtype=torch.float32)
    total = numerator + denominator
    valid = total > 0
    out[valid] = torch.clamp(numerator[valid] / total[valid], min=0.0, max=1.0)
    return out


def _summarize_head_values(head_values: np.ndarray) -> dict[str, float]:
    values = np.asarray(head_values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {name: float("nan") for name in HEAD_SUMMARY_NAMES}
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "max": float(values.max()),
    }


def _cosine_similarity(vec_a: np.ndarray | None, vec_b: np.ndarray | None) -> float:
    if vec_a is None or vec_b is None:
        return float("nan")
    norm_a = float(np.linalg.norm(vec_a))
    norm_b = float(np.linalg.norm(vec_b))
    if norm_a <= EPS or norm_b <= EPS:
        return float("nan")
    value = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
    return float(np.clip(value, -1.0, 1.0))


def _mean_pairwise_cosine(vectors: np.ndarray) -> float:
    if vectors.ndim != 2 or vectors.shape[0] <= 1:
        return float("nan")
    norms = np.linalg.norm(vectors, axis=1)
    valid = norms > EPS
    if valid.sum() <= 1:
        return float("nan")
    unit = vectors[valid] / norms[valid, None]
    sim = np.clip(unit @ unit.T, -1.0, 1.0)
    mask = ~np.eye(sim.shape[0], dtype=bool)
    if not mask.any():
        return float("nan")
    return float(sim[mask].mean())


def _bounded_relative_change(current: pd.Series, reference: pd.Series) -> pd.Series:
    current_values = pd.to_numeric(current, errors="coerce").astype(float)
    reference_values = pd.to_numeric(reference, errors="coerce").astype(float)
    out = pd.Series(np.nan, index=current_values.index, dtype=float)
    valid = current_values.notna() & reference_values.notna()
    if valid.any():
        numerator = current_values.loc[valid] - reference_values.loc[valid]
        denominator = current_values.loc[valid].abs() + reference_values.loc[valid].abs() + EPS
        out.loc[valid] = numerator / denominator
    return out.clip(-1.0, 1.0)


def _rolling_slope3(values: pd.Series) -> pd.Series:
    arr = pd.to_numeric(values, errors="coerce").astype(float).to_numpy(dtype=float)
    out = np.full(arr.shape[0], np.nan, dtype=float)
    x = np.array([0.0, 1.0, 2.0], dtype=float)
    x_centered = x - x.mean()
    denom = float((x_centered ** 2).sum())
    for idx in range(2, arr.shape[0]):
        window = arr[idx - 2 : idx + 1]
        if not np.isfinite(window).all():
            continue
        y = window - window.mean()
        out[idx] = float((x_centered * y).sum() / denom)
    return pd.Series(out, index=values.index, dtype=float)


def add_region_columns(aligned_sentence_df: pd.DataFrame, *, recent_window_tokens: int) -> pd.DataFrame:
    df = aligned_sentence_df.sort_values("sentence_idx").reset_index(drop=True).copy()
    previous_sentence_start_tokens: list[Optional[int]] = []
    previous_sentence_end_tokens: list[Optional[int]] = []
    previous_sentence_token_counts: list[int] = []
    previous_three_sentence_start_tokens: list[Optional[int]] = []
    previous_three_sentence_end_tokens: list[Optional[int]] = []
    previous_three_sentence_token_counts: list[int] = []
    recent_token_counts: list[int] = []
    early_token_counts: list[int] = []
    prior_all_token_counts: list[int] = []
    available_token_counts: list[int] = []

    for row_idx, row in enumerate(df.itertuples()):
        if row.start_token is None or pd.isna(row.start_token):
            previous_sentence_start_tokens.append(None)
            previous_sentence_end_tokens.append(None)
            previous_sentence_token_counts.append(0)
            previous_three_sentence_start_tokens.append(None)
            previous_three_sentence_end_tokens.append(None)
            previous_three_sentence_token_counts.append(0)
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
            previous_three_sentence_start_tokens.append(None)
            previous_three_sentence_end_tokens.append(None)
            previous_three_sentence_token_counts.append(0)
        else:
            prev_row = df.iloc[row_idx - 1]
            prev_start = int(prev_row["start_token"])
            prev_end = int(prev_row["end_token"])
            previous_sentence_start_tokens.append(prev_start)
            previous_sentence_end_tokens.append(prev_end)
            previous_sentence_token_counts.append(prev_end - prev_start + 1)
            prev3_row = df.iloc[max(0, row_idx - 3)]
            prev3_start = int(prev3_row["start_token"])
            previous_three_sentence_start_tokens.append(prev3_start)
            previous_three_sentence_end_tokens.append(prev_end)
            previous_three_sentence_token_counts.append(prev_end - prev3_start + 1)
        recent_start = max(0, start_token - int(recent_window_tokens))
        recent_token_counts.append(start_token - recent_start)
        early_token_counts.append(recent_start)

    df["previous_sentence_start_token"] = previous_sentence_start_tokens
    df["previous_sentence_end_token"] = previous_sentence_end_tokens
    df["previous_sentence_token_count"] = previous_sentence_token_counts
    df["previous_three_sentence_start_token"] = previous_three_sentence_start_tokens
    df["previous_three_sentence_end_token"] = previous_three_sentence_end_tokens
    df["previous_three_sentence_token_count"] = previous_three_sentence_token_counts
    df["recent_token_count"] = recent_token_counts
    df["early_token_count"] = early_token_counts
    df["prior_all_token_count"] = prior_all_token_counts
    df["available_token_count"] = available_token_counts
    return df


def add_prefix_slot_columns(aligned_sentence_df: pd.DataFrame, *, num_prefix_sentences: int) -> pd.DataFrame:
    df = aligned_sentence_df.sort_values("sentence_idx").reset_index(drop=True).copy()
    end_tokens = df["end_token"].astype(int).tolist()
    sentence_indices = df["sentence_idx"].astype(int).tolist()
    available_counts: list[int] = []

    for offset in range(int(num_prefix_sentences)):
        df[f"prefix_end_token_{offset}"] = -1
        df[f"prefix_sentence_idx_{offset}"] = -1
        df[f"prefix_slot_valid_{offset}"] = False

    for row_idx in range(len(df)):
        available_counts.append(min(int(num_prefix_sentences), row_idx + 1))
        for offset in range(int(num_prefix_sentences)):
            source_idx = row_idx - offset
            if source_idx < 0:
                continue
            df.at[row_idx, f"prefix_end_token_{offset}"] = int(end_tokens[source_idx])
            df.at[row_idx, f"prefix_sentence_idx_{offset}"] = int(sentence_indices[source_idx])
            df.at[row_idx, f"prefix_slot_valid_{offset}"] = True

    df["available_prefix_sentence_count"] = available_counts
    return df


def compute_attention_metric_tensors(
    attentions: Sequence[torch.Tensor],
    sentence_row: Any,
    *,
    recent_window_tokens: int,
) -> dict[str, np.ndarray]:
    q_idx = int(sentence_row.end_token)
    start_token = int(sentence_row.start_token)
    end_token = int(sentence_row.end_token)
    recent_start = max(0, start_token - int(recent_window_tokens))
    prev_start = (
        None
        if sentence_row.previous_sentence_start_token is None or pd.isna(sentence_row.previous_sentence_start_token)
        else int(sentence_row.previous_sentence_start_token)
    )
    prev_end = (
        None
        if sentence_row.previous_sentence_end_token is None or pd.isna(sentence_row.previous_sentence_end_token)
        else int(sentence_row.previous_sentence_end_token)
    )
    prev3_start = (
        None
        if sentence_row.previous_three_sentence_start_token is None or pd.isna(sentence_row.previous_three_sentence_start_token)
        else int(sentence_row.previous_three_sentence_start_token)
    )
    prev3_end = (
        None
        if sentence_row.previous_three_sentence_end_token is None or pd.isna(sentence_row.previous_three_sentence_end_token)
        else int(sentence_row.previous_three_sentence_end_token)
    )

    num_layers = len(attentions)
    num_heads = int(attentions[0].shape[1])
    metric_tensors = {
        name: np.full((num_layers, num_heads), np.nan, dtype=np.float32)
        for name in ATTENTION_METRIC_NAMES
    }

    for layer_idx, layer_attn in enumerate(attentions):
        layer = layer_attn[0]
        # Only the single query row is needed for the sentence metrics, so avoid
        # upcasting the full [heads, seq, seq] attention map to float32 on GPU.
        query_attn = layer[:, q_idx, : end_token + 1].to(dtype=torch.float32)

        full_slice = query_attn
        prior_slice = query_attn[:, :start_token]
        current_slice = query_attn[:, start_token : end_token + 1]
        prev_slice = query_attn[:, prev_start : prev_end + 1] if prev_start is not None and prev_end is not None else _empty_slice(query_attn)
        prev3_slice = query_attn[:, prev3_start : prev3_end + 1] if prev3_start is not None and prev3_end is not None else _empty_slice(query_attn)
        recent_slice = query_attn[:, recent_start:start_token]
        early_slice = query_attn[:, :recent_start]

        current_mass = current_slice.sum(dim=1)
        prior_mass = prior_slice.sum(dim=1)
        prev_mass = prev_slice.sum(dim=1)
        prev3_mass = prev3_slice.sum(dim=1)
        recent_mass = recent_slice.sum(dim=1)
        early_mass = early_slice.sum(dim=1)

        current_ptm = _per_token_mass(current_mass, int(current_slice.shape[1]))
        prior_ptm = _per_token_mass(prior_mass, int(prior_slice.shape[1]))
        prev_ptm = _per_token_mass(prev_mass, int(prev_slice.shape[1]))
        recent_ptm = _per_token_mass(recent_mass, int(recent_slice.shape[1]))
        early_ptm = _per_token_mass(early_mass, int(early_slice.shape[1]))

        layer_metrics: dict[str, torch.Tensor] = {}
        layer_metrics["current_vs_prior"] = _ratio_from_per_token_mass(current_ptm, prior_ptm)
        layer_metrics["current_vs_prev"] = _ratio_from_per_token_mass(current_ptm, prev_ptm)
        current_vs_prev3 = torch.full_like(current_mass, float("nan"), dtype=torch.float32)
        total_prev3 = current_mass + prev3_mass
        valid_prev3 = total_prev3 > 0
        current_vs_prev3[valid_prev3] = torch.clamp(
            current_mass[valid_prev3] / (total_prev3[valid_prev3] + EPS),
            min=0.0,
            max=1.0,
        )
        layer_metrics["current_vs_prev3"] = current_vs_prev3
        layer_metrics["recent_vs_early"] = _ratio_from_per_token_mass(recent_ptm, early_ptm)

        prev_share = torch.full_like(prev_mass, float("nan"), dtype=torch.float32)
        valid_prior = prior_mass > 0
        prev_share[valid_prior] = torch.clamp(prev_mass[valid_prior] / (prior_mass[valid_prior] + EPS), min=0.0, max=1.0)
        layer_metrics["prev_share_of_prior"] = prev_share

        current_share_total = torch.full_like(current_mass, float("nan"), dtype=torch.float32)
        total_mass = current_mass + prior_mass
        valid_total = total_mass > 0
        current_share_total[valid_total] = torch.clamp(current_mass[valid_total] / (total_mass[valid_total] + EPS), min=0.0, max=1.0)
        layer_metrics["current_share_total"] = current_share_total

        layer_metrics["entropy_prior"] = _normalized_entropy(prior_slice)
        layer_metrics["entropy_full"] = _normalized_entropy(full_slice)
        layer_metrics["top1_prior"] = _topk_mass(prior_slice, 1)
        layer_metrics["top5_prior"] = _topk_mass(prior_slice, 5)
        layer_metrics["herfindahl_prior"] = _herfindahl(prior_slice)
        layer_metrics["effective_support_prior"] = _effective_support(prior_slice)

        for metric_name, metric_value in layer_metrics.items():
            metric_tensors[metric_name][layer_idx] = metric_value.detach().to(device="cpu", dtype=torch.float32).numpy()

    return metric_tensors


def compute_activation_scalars_for_row(last_hidden_np: np.ndarray, sentence_row: Any, *, num_prefix_sentences: int) -> dict[str, float]:
    token_columns = [f"prefix_end_token_{offset}" for offset in range(int(num_prefix_sentences))]
    valid_columns = [f"prefix_slot_valid_{offset}" for offset in range(int(num_prefix_sentences))]
    token_indices = np.asarray([int(getattr(sentence_row, column)) for column in token_columns], dtype=np.int64)
    valid_mask = np.asarray([bool(getattr(sentence_row, column)) for column in valid_columns], dtype=bool)
    valid_indices = token_indices[valid_mask]
    vectors = last_hidden_np[valid_indices] if valid_indices.size > 0 else np.zeros((0, last_hidden_np.shape[1]), dtype=np.float32)

    current_vec = vectors[0] if vectors.shape[0] >= 1 else None
    prev_vec = vectors[1] if vectors.shape[0] >= 2 else None
    prev_mean3 = vectors[1:4].mean(axis=0) if vectors.shape[0] >= 2 else None

    norm_cur = float(np.linalg.norm(current_vec)) if current_vec is not None else float("nan")
    norm_prev = float(np.linalg.norm(prev_vec)) if prev_vec is not None else float("nan")
    return {
        "act_cos_cur_prev": _cosine_similarity(current_vec, prev_vec),
        "act_cos_cur_mean3": _cosine_similarity(current_vec, prev_mean3),
        "act_norm_cur": norm_cur,
        "act_norm_jump": (norm_cur - norm_prev) if np.isfinite(norm_cur) and np.isfinite(norm_prev) else float("nan"),
        "act_delta_vec_norm": float(np.linalg.norm(current_vec - prev_vec)) if current_vec is not None and prev_vec is not None else float("nan"),
        "act_pairwise_recent_cohesion": _mean_pairwise_cosine(vectors),
    }


def build_base_feature_record(
    *,
    example_id: str,
    sentence_row: Any,
    metric_tensors: dict[str, np.ndarray],
    activation_scalars: dict[str, float],
    prompt_token_count: int,
    total_input_token_count: int,
) -> dict[str, Any]:
    feature_row: dict[str, Any] = {
        "example_id": example_id,
        "sentence_idx": int(sentence_row.sentence_idx),
        "sentence_text": sentence_row.sentence_text,
        "deception_rate": float(sentence_row.deception_rate),
        "num_truthful": int(sentence_row.num_truthful),
        "num_valid": int(sentence_row.num_valid),
        "raw_start": int(sentence_row.raw_start),
        "raw_end": int(sentence_row.raw_end),
        "full_start": int(sentence_row.full_start),
        "full_end": int(sentence_row.full_end),
        "start_token": int(sentence_row.start_token),
        "end_token": int(sentence_row.end_token),
        "token_count": int(sentence_row.token_count),
        "context_token_count": int(sentence_row.context_token_count),
        "prompt_token_count": int(prompt_token_count),
        "raw_text_context_token_count": max(0, int(sentence_row.start_token) - int(prompt_token_count)),
        "available_token_count": int(sentence_row.available_token_count),
        "prior_all_token_count": int(sentence_row.prior_all_token_count),
        "previous_sentence_token_count": int(sentence_row.previous_sentence_token_count),
        "previous_three_sentence_token_count": int(sentence_row.previous_three_sentence_token_count),
        "recent_token_count": int(sentence_row.recent_token_count),
        "early_token_count": int(sentence_row.early_token_count),
        "available_prefix_sentence_count": int(sentence_row.available_prefix_sentence_count),
        "total_input_token_count": int(total_input_token_count),
    }

    for metric_name, metric_tensor in metric_tensors.items():
        for layer_idx, layer_values in enumerate(np.asarray(metric_tensor, dtype=float)):
            summary = _summarize_head_values(layer_values)
            for agg_name, value in summary.items():
                feature_row[f"{metric_name}_{agg_name}_l{layer_idx}"] = value

    for metric_name, value in activation_scalars.items():
        feature_row[metric_name] = float(value) if np.isfinite(value) else float("nan")
    return feature_row


def add_transition_features(base_feature_df: pd.DataFrame, *, num_layers: int) -> pd.DataFrame:
    df = base_feature_df.sort_values("sentence_idx").reset_index(drop=True).copy()
    new_columns: dict[str, pd.Series] = {}
    transition_targets = [
        *(f"{metric_name}_{agg_name}_l{layer_idx}" for layer_idx in range(int(num_layers)) for metric_name in ATTENTION_METRIC_NAMES for agg_name in HEAD_SUMMARY_NAMES),
        *ACTIVATION_SCALAR_FEATURE_NAMES,
    ]
    for column in transition_targets:
        current = pd.to_numeric(df[column], errors="coerce").astype(float)
        prev = current.shift(1)
        prev_running_mean = current.expanding(min_periods=1).mean().shift(1)
        prev_running_min = current.cummin().shift(1)
        prev_running_max = current.cummax().shift(1)
        new_columns[f"delta_{column}"] = _bounded_relative_change(current, prev)
        new_columns[f"slope3_{column}"] = _rolling_slope3(current)
        new_columns[f"devrun_{column}"] = current - prev_running_mean
        new_columns[f"min_gap_{column}"] = current - prev_running_min
        new_columns[f"max_gap_{column}"] = current - prev_running_max
    return pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)


def _extract_last_hidden_state(base_outputs: Any) -> torch.Tensor:
    if hasattr(base_outputs, "last_hidden_state") and base_outputs.last_hidden_state is not None:
        return base_outputs.last_hidden_state
    if isinstance(base_outputs, (tuple, list)) and len(base_outputs) > 0 and torch.is_tensor(base_outputs[0]):
        return base_outputs[0]
    raise ValueError("Could not extract last_hidden_state from base model outputs.")


def compute_tabular_features(
    attentions: Sequence[torch.Tensor],
    last_hidden_np: np.ndarray,
    aligned_sentence_df: pd.DataFrame,
    *,
    example_id: str,
    prompt_token_count: int,
    total_input_token_count: int,
    recent_window_tokens: int,
    num_prefix_sentences: int,
) -> pd.DataFrame:
    base_records: list[dict[str, Any]] = []
    for row in aligned_sentence_df.itertuples():
        metric_tensors = compute_attention_metric_tensors(
            attentions,
            row,
            recent_window_tokens=recent_window_tokens,
        )
        activation_scalars = compute_activation_scalars_for_row(
            last_hidden_np,
            row,
            num_prefix_sentences=num_prefix_sentences,
        )
        base_records.append(
            build_base_feature_record(
                example_id=example_id,
                sentence_row=row,
                metric_tensors=metric_tensors,
                activation_scalars=activation_scalars,
                prompt_token_count=prompt_token_count,
                total_input_token_count=total_input_token_count,
            )
        )
    if not base_records:
        return build_empty_feature_frame(num_layers=len(attentions))
    feature_df = pd.DataFrame(base_records)
    return add_transition_features(feature_df, num_layers=len(attentions))


def build_activation_batch(feature_df: pd.DataFrame, slot_df: pd.DataFrame, last_hidden_np: np.ndarray) -> dict[str, Any]:
    working_df = feature_df[list(METADATA_COLUMNS)].copy()
    for column in slot_df.columns:
        working_df[column] = slot_df[column].to_numpy()

    token_index_columns = [column for column in working_df.columns if column.startswith("prefix_end_token_")]
    sentence_index_columns = [column for column in working_df.columns if column.startswith("prefix_sentence_idx_")]
    valid_columns = [column for column in working_df.columns if column.startswith("prefix_slot_valid_")]

    token_index_matrix = working_df[token_index_columns].to_numpy(dtype=np.int64)
    sentence_index_matrix = working_df[sentence_index_columns].to_numpy(dtype=np.int32)
    valid_mask = working_df[valid_columns].to_numpy(dtype=bool)

    safe_indices = token_index_matrix.copy()
    safe_indices[~valid_mask] = 0
    activations = last_hidden_np[safe_indices].astype(np.float16, copy=False)
    activations[~valid_mask] = np.float16(0.0)

    batch: dict[str, Any] = {
        "activations": np.asarray(activations, dtype=np.float16),
        "activation_mask": np.asarray(valid_mask, dtype=np.bool_),
        "prefix_sentence_indices": np.asarray(sentence_index_matrix, dtype=np.int32),
        "prefix_end_token_indices": np.asarray(token_index_matrix, dtype=np.int32),
    }
    for name in STRING_COLUMNS:
        batch[name] = working_df[name].astype(str).tolist()
    for name in INT_COLUMNS:
        batch[name] = working_df[name].astype(np.int64).tolist()
    for name in FLOAT_COLUMNS:
        batch[name] = working_df[name].astype(float).tolist()
    return batch


def extract_example_outputs(
    *,
    example: dict[str, Any],
    tokenizer: Any,
    base_model: Any,
    model_input_device: str,
    recent_window_tokens: int,
    num_prefix_sentences: int,
    max_input_tokens: int,
) -> tuple[pd.DataFrame, dict[str, Any], int]:
    example_id = example.get("example_id")
    if not isinstance(example_id, str) or not example_id:
        raise ExampleValidationError("missing_example_id", "Localization example is missing example_id.")

    raw_text = example.get("raw_text")
    if not isinstance(raw_text, str) or not raw_text:
        raise ExampleValidationError("missing_raw_text", f"{example_id} is missing raw_text.")

    prompt_text = example.get("prompt")
    if prompt_text is None:
        prompt_text = ""
    if not isinstance(prompt_text, str):
        raise ExampleValidationError("invalid_prompt", f"{example_id} has a non-string prompt field.")

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

    prompt_char_count = len(prompt_text)
    model_input_text = prompt_text + raw_text
    if not model_input_text:
        raise ExampleValidationError("empty_input", f"{example_id} has empty prompt + raw_text.")

    shifted_df = localized_sentence_df.copy()
    for column in ("raw_start", "raw_end", "full_start", "full_end"):
        shifted_df[column] = shifted_df[column].astype(int) + prompt_char_count

    token_alignment = tokenize_and_align_localized_sentences(
        tokenizer=tokenizer,
        full_text=model_input_text,
        sentence_df=shifted_df,
        raw_text_start_char=prompt_char_count,
    )
    input_ids_list = token_alignment.input_ids
    if not input_ids_list:
        raise ExampleValidationError("no_tokens", f"{example_id} tokenized to zero tokens.")
    if int(max_input_tokens) > 0 and len(input_ids_list) > int(max_input_tokens):
        raise ExampleValidationError(
            "context_too_long",
            f"{example_id} tokenized to {len(input_ids_list)} tokens, exceeding max_input_tokens={int(max_input_tokens)}.",
        )

    prompt_token_count = int(token_alignment.prompt_token_count)
    total_input_token_count = int(len(input_ids_list))

    aligned_sentence_df = token_alignment.aligned_sentence_df
    if not (aligned_sentence_df["token_count"] > 0).all():
        bad_count = int((aligned_sentence_df["token_count"] == 0).sum())
        raise ExampleValidationError(
            "unmapped_sentence",
            f"{example_id} has {bad_count} localized sentences that failed to map to tokens.",
        )

    for column in ("raw_start", "raw_end", "full_start", "full_end"):
        aligned_sentence_df[column] = aligned_sentence_df[column].astype(int) - prompt_char_count

    aligned_sentence_df = add_region_columns(aligned_sentence_df, recent_window_tokens=recent_window_tokens)
    aligned_sentence_df = add_prefix_slot_columns(aligned_sentence_df, num_prefix_sentences=num_prefix_sentences)
    modeling_sentence_df = aligned_sentence_df.loc[aligned_sentence_df["start_token"].fillna(0).astype(int) > 0].copy()
    num_layers = int(getattr(base_model.config, "num_hidden_layers", 0))

    if modeling_sentence_df.empty:
        empty_feature_df = build_empty_feature_frame(num_layers=num_layers)
        empty_batch = {
            "activations": np.zeros((0, int(num_prefix_sentences), int(getattr(base_model.config, "hidden_size", 0))), dtype=np.float16),
            "activation_mask": np.zeros((0, int(num_prefix_sentences)), dtype=np.bool_),
            "prefix_sentence_indices": np.zeros((0, int(num_prefix_sentences)), dtype=np.int32),
            "prefix_end_token_indices": np.zeros((0, int(num_prefix_sentences)), dtype=np.int32),
            **{name: [] for name in STRING_COLUMNS},
            **{name: [] for name in INT_COLUMNS},
            **{name: [] for name in FLOAT_COLUMNS},
        }
        return empty_feature_df, empty_batch, num_layers

    input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=model_input_device)
    try:
        with torch.no_grad():
            base_outputs = base_model(
                input_ids=input_ids,
                use_cache=False,
                output_attentions=True,
                return_dict=True,
            )
        attentions = base_outputs.attentions
        last_hidden = _extract_last_hidden_state(base_outputs)
        last_hidden_np = last_hidden[0].detach().to(dtype=torch.float32, device="cpu").numpy()
        feature_df = compute_tabular_features(
            attentions,
            last_hidden_np,
            modeling_sentence_df,
            example_id=example_id,
            prompt_token_count=prompt_token_count,
            total_input_token_count=total_input_token_count,
            recent_window_tokens=recent_window_tokens,
            num_prefix_sentences=num_prefix_sentences,
        )
        slot_columns = [column for column in modeling_sentence_df.columns if column.startswith("prefix_")]
        activation_batch = build_activation_batch(feature_df, modeling_sentence_df[slot_columns], last_hidden_np)
        num_layers = len(attentions)
    finally:
        if "base_outputs" in locals():
            del base_outputs
        if "attentions" in locals():
            del attentions
        if "last_hidden" in locals():
            del last_hidden
        del input_ids
        cleanup_tensors()

    return feature_df, activation_batch, num_layers


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    dataset_paths, outputs = resolve_dataset_paths(args.input_path, args.feature_output, args.activation_output)
    model_id = infer_model_id(dataset_paths, args.model_id)
    device, gpu_df = resolve_device(args.device)
    model_dtype = resolve_dtype(args.dtype, device)
    requested_device_map = resolve_requested_device_map(str(args.device_map))
    repo_quant_method = infer_repo_quant_method(model_id, trust_remote_code=args.trust_remote_code)
    effective_load_in_8bit = bool(args.load_in_8bit)
    if effective_load_in_8bit and repo_quant_method and repo_quant_method != "bitsandbytes":
        print(
            f"Requested --load-in-8bit, but model {model_id} already declares native quantization "
            f"method '{repo_quant_method}'. Ignoring --load-in-8bit and using the repo's native quantization."
        )
        effective_load_in_8bit = False
    write_every_examples = max(1, int(args.write_every_examples))
    num_prefix_sentences = max(1, int(args.num_prefix_sentences))

    all_localization_paths = iter_localization_paths(dataset_paths.localization_dir, max_examples=int(args.max_examples))
    if not all_localization_paths:
        raise FileNotFoundError(f"No localization JSON files found in {dataset_paths.localization_dir}")

    localization_paths = iter_localization_paths(
        dataset_paths.localization_dir,
        max_examples=int(args.max_examples),
        shard_id=int(args.shard_id),
        num_shards=int(args.num_shards),
    )
    if not localization_paths:
        print(f"Dataset dir: {dataset_paths.dataset_dir}")
        print(f"Localization dir: {dataset_paths.localization_dir}")
        print(f"Feature output: {outputs.feature_output}")
        print(f"Activation output: {outputs.activation_output}")
        print(f"Shard: {int(args.shard_id) + 1}/{int(args.num_shards)}")
        print(f"Localization files before sharding: {len(all_localization_paths)}")
        print("Localization files to process on this shard: 0")
        print("No localization files assigned to this shard. Exiting without writing output.")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=args.trust_remote_code)
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("This script requires a fast tokenizer because it uses offset mappings.")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        **build_model_load_kwargs(
            args=args,
            device=device,
            model_dtype=model_dtype,
            load_in_8bit=effective_load_in_8bit,
            requested_device_map=requested_device_map,
        ),
    )
    model_is_dispatched = requested_device_map is not None or effective_load_in_8bit
    if not model_is_dispatched:
        model.to(device)
    model.eval()
    model_input_device = infer_model_input_device(model, device)
    base_model = model.base_model
    num_layers = int(getattr(base_model.config, "num_hidden_layers", getattr(model.config, "num_hidden_layers", 0)))
    ordered_columns = list(METADATA_COLUMNS) + build_feature_columns(num_layers)

    feature_writer = StreamingParquetWriter(outputs.feature_output, overwrite=args.overwrite)
    activation_writer = ActivationH5Writer(
        outputs.activation_output,
        num_prefix_sentences=num_prefix_sentences,
        overwrite=args.overwrite,
        compression=args.compression,
        gzip_level=int(args.gzip_level),
    )

    feature_buffer: list[pd.DataFrame] = []
    activation_buffer: list[dict[str, Any]] = []
    skip_counts: dict[str, int] = {}
    processed = 0
    successful = 0

    attention_base_feature_count = int(num_layers) * len(ATTENTION_METRIC_NAMES) * len(HEAD_SUMMARY_NAMES)
    activation_scalar_feature_count = len(ACTIVATION_SCALAR_FEATURE_NAMES)
    transition_feature_count = len(build_feature_columns(num_layers)) - attention_base_feature_count - activation_scalar_feature_count

    print(f"Dataset dir: {dataset_paths.dataset_dir}")
    print(f"Localization dir: {dataset_paths.localization_dir}")
    print(f"Feature output: {outputs.feature_output}")
    print(f"Activation output: {outputs.activation_output}")
    print(f"Model id: {model_id}")
    print(f"Device: {device}")
    print(f"Requested device map: {requested_device_map or 'single'}")
    print(f"Resolved device map: {summarize_hf_device_map(model)}")
    print(f"Model input device: {model_input_device}")
    print(f"Model dtype: {model_dtype}")
    print(f"Repo quantization method: {repo_quant_method or 'none'}")
    print(f"8-bit quantization: {'enabled' if effective_load_in_8bit else 'disabled'}")
    print(f"Layers: {num_layers}")
    print(f"Shard: {int(args.shard_id) + 1}/{int(args.num_shards)}")
    print(
        "Feature columns: "
        f"{attention_base_feature_count} attention-base + "
        f"{activation_scalar_feature_count} activation-scalar + "
        f"{transition_feature_count} transition = {len(ordered_columns) - len(METADATA_COLUMNS)}"
    )
    print(f"Recent window tokens: {int(args.recent_window_tokens)}")
    print(f"Max input tokens: {int(args.max_input_tokens)}")
    print(f"Saved prefix sentence slots: {num_prefix_sentences}")
    print(f"Localization files before sharding: {len(all_localization_paths)}")
    print(f"Localization files to process on this shard: {len(localization_paths)}")
    if not gpu_df.empty:
        print("Visible GPUs:")
        print(gpu_df.to_string(index=False))

    feature_closed = False
    activation_closed = False
    try:
        path_iter = maybe_tqdm(
            localization_paths,
            desc="Extract prefix features",
            total=len(localization_paths),
            disable=bool(args.disable_tqdm),
        )
        for path in path_iter:
            processed += 1
            had_error = False
            try:
                example = json.loads(path.read_text(encoding="utf-8"))
                feature_df, activation_batch, _ = extract_example_outputs(
                    example=example,
                    tokenizer=tokenizer,
                    base_model=base_model,
                    model_input_device=model_input_device,
                    recent_window_tokens=int(args.recent_window_tokens),
                    num_prefix_sentences=num_prefix_sentences,
                    max_input_tokens=int(args.max_input_tokens),
                )
            except json.JSONDecodeError as exc:
                had_error = True
                reason = "invalid_json"
                skip_counts[reason] = skip_counts.get(reason, 0) + 1
                if not args.quiet_skips:
                    progress_write(path_iter, format_skip_message(path, reason, exc))
                maybe_raise_invalid_example(args, path, exc)
                feature_df = None
                activation_batch = None
            except ExampleValidationError as exc:
                had_error = True
                reason = str(exc.reason)
                skip_counts[reason] = skip_counts.get(reason, 0) + 1
                if not args.quiet_skips:
                    progress_write(path_iter, format_skip_message(path, reason, exc))
                maybe_raise_invalid_example(args, path, exc)
                feature_df = None
                activation_batch = None
            except (KeyError, TypeError, ValueError, IndexError) as exc:
                had_error = True
                reason = "malformed_example"
                skip_counts[reason] = skip_counts.get(reason, 0) + 1
                if not args.quiet_skips:
                    progress_write(path_iter, format_skip_message(path, reason, exc))
                maybe_raise_invalid_example(args, path, exc)
                feature_df = None
                activation_batch = None
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    had_error = True
                    reason = "oom"
                    skip_counts[reason] = skip_counts.get(reason, 0) + 1
                    if not args.quiet_skips:
                        progress_write(path_iter, format_skip_message(path, reason, exc))
                    cleanup_tensors()
                    maybe_raise_runtime_error(args, path, exc)
                    feature_df = None
                    activation_batch = None
                else:
                    raise

            if feature_df is not None and activation_batch is not None and not feature_df.empty:
                feature_buffer.append(feature_df)
                activation_buffer.append(activation_batch)
                successful += 1
            elif not had_error:
                skip_counts["no_rows"] = skip_counts.get("no_rows", 0) + 1

            if len(feature_buffer) >= write_every_examples:
                flush_feature_buffer(feature_writer, feature_buffer, ordered_columns=ordered_columns)
                flush_activation_buffer(activation_writer, activation_buffer)

            if int(args.progress_every) > 0 and processed % int(args.progress_every) == 0:
                buffered_rows = sum(len(df) for df in feature_buffer)
                progress_write(
                    path_iter,
                    f"Processed {processed}/{len(localization_paths)} files | "
                    f"successful={successful} | skipped={sum(skip_counts.values())} | "
                    f"rows_buffered_or_written={activation_writer.rows_written + buffered_rows}",
                )

        flush_feature_buffer(feature_writer, feature_buffer, ordered_columns=ordered_columns)
        flush_activation_buffer(activation_writer, activation_buffer)
        feature_writer.close()
        feature_closed = True
        activation_writer.close()
        activation_closed = True
    except Exception:
        if not feature_closed:
            feature_writer.abort()
        elif not activation_closed and outputs.feature_output.exists():
            outputs.feature_output.unlink()
        if not activation_closed:
            activation_writer.abort()
        raise
    finally:
        del model
        cleanup_tensors()

    print(f"Wrote handcrafted features to: {outputs.feature_output}")
    print(f"Wrote raw activations to: {outputs.activation_output}")
    print(f"Processed files: {processed}")
    print(f"Examples with output rows: {successful}")
    print(f"Total activation rows written: {activation_writer.rows_written}")
    if activation_writer.hidden_size is not None:
        print(f"Hidden size: {activation_writer.hidden_size}")
    if skip_counts:
        print("Skipped examples by reason:")
        for reason, count in sorted(skip_counts.items()):
            print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()
