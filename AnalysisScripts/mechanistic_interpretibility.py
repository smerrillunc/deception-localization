#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import math
import os
import random
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import einops
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

THIS_FILE = Path(__file__).resolve()
ANALYSIS_ROOT = THIS_FILE.parent
REPO_ROOT = THIS_FILE.parents[1]
CORE_DIR = REPO_ROOT / "Core"
INTERPRETABILITY_SUPPORT_DIR = ANALYSIS_ROOT / "interpretability_support"
DEFAULT_OUTPUT_BASE = REPO_ROOT / "Results" / "activation_patchingHeadonly"
PATCH_SCOPE_CHOICES = (
    "commitment_first",
    "commitment_half",
    "commitment_full",
)
DEFAULT_CANDIDATE_CIRCUIT_SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
DEFAULT_CROSS_CORPUS_ENVS = ("advisor_audit", "car_sales", "gridworld", "interview")
DEFAULT_MODEL_ID = "gpt-oss-20b"
DEFAULT_DTYPE_NAME = "bfloat16"
DEFAULT_TRAIN_PAIR_COUNT = 300
DEFAULT_VALIDATION_PAIR_COUNT = 50
DEFAULT_TEST_PAIR_COUNT = 50
DEFAULT_BATCH_PAIR_COUNT = 1
DEFAULT_CONTROL_CIRCUIT_COUNT = 8
DEFAULT_STEERING_ALPHA = 1.0
DEFAULT_STEERING_POSITION = "last"
DEFAULT_STEERING_BATCH_SIZE = 4
DEFAULT_STEERING_SAMPLES_PER_PROMPT = 50
DEFAULT_STEERING_MAX_NEW_TOKENS = 96
DEFAULT_STEERING_MAX_DECODE_TOKENS = 10
DEFAULT_STEERING_TEMPERATURE = 0.7
DEFAULT_STEERING_TOP_P = 0.95
DEFAULT_STEERING_EARLY_STOP_CHECK_INTERVAL = 8
DEFAULT_STEERING_EARLY_STOP_MIN_NEW_TOKENS = 16
DEFAULT_GENERATION_PROMPTS_PER_ENV = 10
DEFAULT_MEAN_SHUFFLED_DONOR_COUNT = 10
DEFAULT_HF_CACHE_ROOT = (
    Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))).expanduser()
    / "hub"
)

MODEL_CONFIGS: dict[str, dict[str, str]] = {
    "gpt-oss-20b": {
        "hf_repo": "openai/gpt-oss-20b",
        "hf_cache_dir": "models--openai--gpt-oss-20b",
    },
    "DeepSeek-R1-Distill-Llama-8B": {
        "hf_repo": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "hf_cache_dir": "models--deepseek-ai--DeepSeek-R1-Distill-Llama-8B",
    },
    "DeepSeek-R1-Distill-Qwen-14B": {
        "hf_repo": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "hf_cache_dir": "models--deepseek-ai--DeepSeek-R1-Distill-Qwen-14B",
    },
    "DeepSeek-R1-Distill-Qwen-7B": {
        "hf_repo": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "hf_cache_dir": "models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B",
    },
}
DTYPE_BY_NAME = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

for search_root in (INTERPRETABILITY_SUPPORT_DIR, CORE_DIR):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

import activation_patching as ap
import activation_steering as steering_utils
from activation_patching import encode_text_for_model, get_nested_attr, resolve_decoder_layers


@dataclass(frozen=True)
class ScopeRun:
    run_name: str
    patch_scope: str
    patch_first_n_tokens: int
    description: str


SCOPE_RUN_LIBRARY: dict[str, ScopeRun] = {
    "patch_first_1_token": ScopeRun(
        run_name="patch_first_1_token",
        patch_scope="commitment_first",
        patch_first_n_tokens=1,
        description="Patch only the first commitment token.",
    ),
    "patch_first_half_sentence": ScopeRun(
        run_name="patch_first_half_sentence",
        patch_scope="commitment_half",
        patch_first_n_tokens=0,
        description="Patch the first half of the commitment sentence position-wise.",
    ),
    "patch_full_sentence": ScopeRun(
        run_name="patch_full_sentence",
        patch_scope="commitment_full",
        patch_first_n_tokens=0,
        description="Patch the full commitment sentence position-wise.",
    ),
}


@dataclass
class HeadModelRuntime:
    model: Any
    tokenizer: Any
    layers: Any
    layer_path: str
    output_head_module: Any
    output_head_path: str | None
    n_layers: int
    n_heads: int
    head_dim: int
    model_id: str
    model_name_or_path: str
    dtype_name: str


@dataclass
class GenerationModelRuntime:
    model: Any
    tokenizer: Any
    layers: Any
    layer_path: str
    n_layers: int
    n_heads: int
    head_dim: int
    model_name_or_path: str
    dtype_name: str


@dataclass
class PreparedSplit:
    split_name: str
    pairs_df: pd.DataFrame
    prepared_pairs: list[dict[str, Any]]
    pair_chunks: list[list[dict[str, Any]]]
    total_pairs: int
    dataset_max_input_tokens: int
    pairs_overview_df: pd.DataFrame


def slugify(text: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text).strip())
    return normalized.strip("_") or "artifact"


def deterministic_seed(base_seed: int, *parts: Any) -> int:
    payload = "||".join([str(base_seed), *[str(part) for part in parts]])
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int((int(base_seed) + int(digest[:12], 16)) % (2**31 - 1))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_int_list(text: str | None) -> list[int]:
    if text is None:
        return []
    return [int(part.strip()) for part in str(text).split(",") if part.strip()]


def parse_float_list(text: str | None) -> list[float]:
    if text is None:
        return []
    return [float(part.strip()) for part in str(text).split(",") if part.strip()]


def parse_string_list(text: str | None) -> list[str]:
    if text is None:
        return []
    return [part.strip() for part in str(text).split(",") if part.strip()]


def to_json_safe(obj: Any) -> Any:
    if obj is pd.NA:
        return None
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(key): to_json_safe(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [to_json_safe(value) for value in obj]
    if isinstance(obj, tuple):
        return [to_json_safe(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_json_safe(payload), indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(to_json_safe(row), ensure_ascii=False) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    flat = df.copy()
    if isinstance(flat.columns, pd.MultiIndex):
        flat.columns = [
            "__".join(str(piece) for piece in column if str(piece) and str(piece) != "nan").strip("_")
            for column in flat.columns.to_flat_index()
        ]
    return flat


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    flat = flatten_columns(df)
    if list(flat.columns) != list(df.columns):
        flat.to_csv(path.with_name(path.stem + "__flat.csv"), index=False)


def latest_snapshot_path(root: Path) -> Path | None:
    snapshot_root = root / "snapshots"
    if not snapshot_root.exists():
        return None
    snapshots = sorted(path for path in snapshot_root.iterdir() if path.is_dir())
    return snapshots[-1] if snapshots else None


def resolve_dtype(dtype_name: str) -> torch.dtype:
    key = str(dtype_name).strip().lower()
    if key not in DTYPE_BY_NAME:
        raise ValueError(f"Unsupported dtype {dtype_name!r}. Choose from {sorted(DTYPE_BY_NAME)}.")
    return DTYPE_BY_NAME[key]


def apply_runtime_env(*, cuda_visible_devices: str, pytorch_cuda_alloc_conf: str) -> None:
    if str(cuda_visible_devices).strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices).strip()
    if str(pytorch_cuda_alloc_conf).strip():
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = str(pytorch_cuda_alloc_conf).strip()


def resolve_model_name_or_path(
    *,
    model_id: str,
    model_name_or_path: str,
    hf_cache_root: Path,
) -> str:
    if str(model_name_or_path).strip():
        return str(Path(model_name_or_path).expanduser()) if Path(model_name_or_path).expanduser().exists() else str(model_name_or_path)
    if model_id not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model_id={model_id!r}. Choose from {sorted(MODEL_CONFIGS)}.")
    model_cfg = MODEL_CONFIGS[model_id]
    cached_snapshot = latest_snapshot_path(Path(hf_cache_root).expanduser().resolve() / model_cfg["hf_cache_dir"])
    if cached_snapshot is not None:
        return str(cached_snapshot)
    return str(model_cfg["hf_repo"])


def resolve_scope_run(args: argparse.Namespace) -> ScopeRun:
    if args.scope_run and args.patch_scope:
        raise ValueError("Use either --scope-run or --patch-scope, not both.")

    if args.scope_run:
        if args.scope_run not in SCOPE_RUN_LIBRARY:
            raise ValueError(f"Unknown scope run {args.scope_run!r}. Choose from {sorted(SCOPE_RUN_LIBRARY)}.")
        preset = SCOPE_RUN_LIBRARY[args.scope_run]
        if args.run_name:
            return ScopeRun(
                run_name=slugify(args.run_name),
                patch_scope=preset.patch_scope,
                patch_first_n_tokens=preset.patch_first_n_tokens,
                description=preset.description,
            )
        return preset

    if not args.patch_scope:
        raise ValueError("Pass either --scope-run or --patch-scope.")

    patch_scope = str(args.patch_scope)
    if patch_scope not in PATCH_SCOPE_CHOICES:
        raise ValueError(f"patch_scope must be one of {sorted(PATCH_SCOPE_CHOICES)}.")
    patch_first_n_tokens = 0
    if patch_scope == "commitment_first":
        patch_first_n_tokens = 1
    run_name = slugify(args.run_name) if args.run_name else (
        "patch_first_1_token"
        if patch_scope == "commitment_first"
        else (
            "patch_first_half_sentence"
            if patch_scope == "commitment_half"
            else "patch_full_sentence"
        )
    )
    return ScopeRun(
        run_name=run_name,
        patch_scope=patch_scope,
        patch_first_n_tokens=patch_first_n_tokens,
        description=f"Custom run for patch_scope={patch_scope}.",
    )


def build_output_root(base_dir: Path, run_tag: str | None, run_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = slugify(run_tag) if run_tag else timestamp
    output_root = base_dir / tag / slugify(run_name)
    output_root.mkdir(parents=True, exist_ok=True)
    return output_root


def clear_memory(model: Any | None = None) -> None:
    if model is not None and hasattr(model, "clear_edits"):
        model.clear_edits()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def is_cuda_oom_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cuda error: out of memory" in message


def freeze_model_parameters(model: Any) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def zero_model_gradients(model: Any) -> None:
    if not hasattr(model, "zero_grad"):
        return
    try:
        model.zero_grad(set_to_none=True)
    except TypeError:
        model.zero_grad()


def saved_value(x: Any) -> Any:
    return getattr(x, "value", x)


def attn_out_input(layer: Any) -> Any:
    if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "o_proj"):
        return layer.self_attn.o_proj.input
    if hasattr(layer, "attn") and hasattr(layer.attn, "c_proj"):
        return layer.attn.c_proj.input
    raise AttributeError("Could not find the attention output projection input for this layer.")


def attn_out_proj_module(layer: Any) -> Any:
    if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "o_proj"):
        return layer.self_attn.o_proj
    if hasattr(layer, "attn") and hasattr(layer.attn, "c_proj"):
        return layer.attn.c_proj
    raise AttributeError("Could not find the attention output projection module for this layer.")


def resolve_output_head(model: Any) -> tuple[Any, str | None]:
    output_module = None
    for getter_target in (model, getattr(model, "_model", None)):
        if getter_target is None:
            continue
        try:
            output_module = getter_target.get_output_embeddings()
        except Exception:
            continue
        if output_module is not None:
            break
    if output_module is None:
        underlying = getattr(model, "_model", model)
        for dotted_name in ("lm_head", "embed_out", "output_projection"):
            try:
                output_module = get_nested_attr(underlying, dotted_name)
            except Exception:
                continue
            if output_module is not None:
                break

    output_path = None
    for dotted_name in ("lm_head", "embed_out", "output_projection"):
        try:
            get_nested_attr(model, dotted_name)
        except Exception:
            continue
        output_path = dotted_name
        break

    if output_module is None:
        raise ValueError("Could not resolve the model output head module.")
    return output_module, output_path


def output_head_input(runtime: HeadModelRuntime) -> Any:
    if runtime.output_head_path is None:
        raise ValueError("Could not resolve the model output head path for traced scoring.")
    return get_nested_attr(runtime.model, runtime.output_head_path).input


def load_head_runtime(
    *,
    model_id: str,
    model_name_or_path: str,
    dtype_name: str,
    device_map: str,
    cuda_visible_devices: str,
    pytorch_cuda_alloc_conf: str,
) -> HeadModelRuntime:
    try:
        from nnsight import LanguageModel
    except ImportError as exc:
        raise ImportError(
            "The analyze/vectors commands require `nnsight` to be installed in the active Python environment."
        ) from exc
    apply_runtime_env(
        cuda_visible_devices=cuda_visible_devices,
        pytorch_cuda_alloc_conf=pytorch_cuda_alloc_conf,
    )
    model_dtype = resolve_dtype(dtype_name)
    model = LanguageModel(
        model_name_or_path,
        device_map=device_map,
        dispatch=True,
        torch_dtype=model_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    if hasattr(model, "eval"):
        model.eval()
    for config_owner in (model, getattr(model, "_model", None), getattr(model, "model", None)):
        config = getattr(config_owner, "config", None)
        if config is not None and hasattr(config, "use_cache"):
            config.use_cache = False
    freeze_model_parameters(model)
    tokenizer = model.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    layers, layer_path = resolve_decoder_layers(model)
    output_head_module, output_head_path = resolve_output_head(model)
    n_layers = len(layers)
    n_heads = int(model.config.num_attention_heads)
    head_dim = int(getattr(model.config, "head_dim", 0) or (model.config.hidden_size // n_heads))
    return HeadModelRuntime(
        model=model,
        tokenizer=tokenizer,
        layers=layers,
        layer_path=layer_path,
        output_head_module=output_head_module,
        output_head_path=output_head_path,
        n_layers=n_layers,
        n_heads=n_heads,
        head_dim=head_dim,
        model_id=model_id,
        model_name_or_path=model_name_or_path,
        dtype_name=dtype_name,
    )


def load_generation_runtime(
    *,
    model_name_or_path: str,
    dtype_name: str,
    device_map: str,
    cuda_visible_devices: str,
    pytorch_cuda_alloc_conf: str,
) -> GenerationModelRuntime:
    apply_runtime_env(
        cuda_visible_devices=cuda_visible_devices,
        pytorch_cuda_alloc_conf=pytorch_cuda_alloc_conf,
    )
    model_dtype = resolve_dtype(dtype_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=model_dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    layers, layer_path = resolve_decoder_layers(model)
    n_layers = len(layers)
    n_heads = int(model.config.num_attention_heads)
    head_dim = int(getattr(model.config, "head_dim", 0) or (model.config.hidden_size // n_heads))
    return GenerationModelRuntime(
        model=model,
        tokenizer=tokenizer,
        layers=layers,
        layer_path=layer_path,
        n_layers=n_layers,
        n_heads=n_heads,
        head_dim=head_dim,
        model_name_or_path=model_name_or_path,
        dtype_name=dtype_name,
    )


def resolve_analysis_paths(
    *,
    dataset_root: Path,
    environment: str,
    dataset_model_id: str,
    localization_dir: str,
    pair_cache_path: str,
) -> tuple[Path, Path]:
    if str(environment).strip().lower() != "bs":
        raise ValueError("This head-only commitment script currently expects --environment bs for in-domain analysis.")
    resolved_localization_dir = (
        Path(localization_dir).expanduser().resolve()
        if str(localization_dir).strip()
        else dataset_root / environment / dataset_model_id / "localization"
    )
    resolved_pair_cache_path = (
        Path(pair_cache_path).expanduser().resolve()
        if str(pair_cache_path).strip()
        else (
            REPO_ROOT
            / "Cache"
            / "activation_patching"
            / f"{environment}_commitment_pairs_for_notebook__{dataset_model_id}.jsonl"
        )
    )
    return resolved_localization_dir, resolved_pair_cache_path


def load_commitment_pairs_for_analysis(
    *,
    localization_dir: Path,
    pair_cache_path: Path,
    pair_count: int,
    pair_search_limit: int,
    refresh_cache: bool,
    min_commitment_delta: float,
    min_commitment_deception_rate: float,
    min_donor_clarity_score: float,
    min_num_valid: int,
    min_sentence_alpha_words: int,
    exclude_multiline_sentences: bool,
    disable_tqdm: bool,
) -> pd.DataFrame:
    load_kwargs: dict[str, Any] = dict(
        localization_dir=localization_dir,
        pair_cache_path=pair_cache_path,
        pair_count=int(pair_count),
        pair_search_limit=int(pair_search_limit),
        refresh_cache=bool(refresh_cache),
        min_commitment_delta=float(min_commitment_delta),
        min_commitment_deception_rate=float(min_commitment_deception_rate),
        min_donor_clarity_score=float(min_donor_clarity_score),
        disable_tqdm=bool(disable_tqdm),
    )
    signature = inspect.signature(ap.load_commitment_pairs)
    if "min_num_valid" in signature.parameters:
        load_kwargs["min_num_valid"] = int(min_num_valid)
    if "min_sentence_alpha_words" in signature.parameters:
        load_kwargs["min_sentence_alpha_words"] = int(min_sentence_alpha_words)
    if "exclude_multiline_sentences" in signature.parameters:
        load_kwargs["exclude_multiline_sentences"] = bool(exclude_multiline_sentences)
    return ap.load_commitment_pairs(**load_kwargs).copy()


def _usable_donor_extension_chunk(text: object) -> bool:
    clean = str(text or "").strip()
    if not clean:
        return False
    lower = clean.lower()
    if clean.startswith("{") or clean.startswith("["):
        return False
    if any(marker in clean for marker in ["<|start|>", "<|end|>", "{", "}"]):
        return False
    if any(
        marker in lower
        for marker in [
            "assistantanalysis",
            "assistantfinal",
            "assistantcommentary",
            "developer#",
            "systemyou are chatgpt",
            "usergame history",
            "cards_played",
            "\"action\"",
            "action:",
            "respond in json",
        ]
    ):
        return False
    return bool(re.search(r"[A-Za-z]", clean))


def _strip_generation_prefix(
    full_generation_text: object,
    *,
    prompt_text: object,
    shared_context_text: object,
) -> str:
    full_text = str(full_generation_text or "")
    if not full_text:
        return ""
    candidates = [
        str(prompt_text or "") + str(shared_context_text or ""),
        str(shared_context_text or ""),
    ]
    for candidate in candidates:
        if candidate and full_text.startswith(candidate):
            return full_text[len(candidate) :].lstrip()
    for candidate in candidates:
        if candidate:
            idx = full_text.find(candidate)
            if idx >= 0:
                full_text = full_text[idx + len(candidate) :].lstrip()
                break
    artifact_markers = [
        "<|start|>",
        "assistantanalysis",
        "assistantfinal",
        "assistantcommentary",
        "\nsystem",
        "\ndeveloper",
        "\nuser",
    ]
    lower_text = full_text.lower()
    cut_positions = [lower_text.find(marker.lower()) for marker in artifact_markers if lower_text.find(marker.lower()) >= 0]
    if cut_positions:
        full_text = full_text[: min(cut_positions)].rstrip()
    return full_text.strip()


def _continuation_token_count(
    tokenizer: Any,
    *,
    prompt_text: str,
    shared_context_text: str,
    continuation_text: str,
    max_input_tokens: int | None,
) -> int:
    prefix_ids = encode_text_for_model(
        tokenizer,
        prompt_text + shared_context_text,
        max_input_tokens=max_input_tokens,
    )["input_ids"][0]
    full_ids = encode_text_for_model(
        tokenizer,
        prompt_text + ap.append_continuation(shared_context_text, continuation_text),
        max_input_tokens=max_input_tokens,
    )["input_ids"][0]
    prefix_len = int(prefix_ids.shape[0])
    full_len = int(full_ids.shape[0])
    if full_len < prefix_len:
        raise ValueError("Continuation tokenization produced a sequence shorter than its prefix.")
    return full_len - prefix_len


def _build_truthful_donor_patch_text(
    tokenizer: Any,
    row: dict[str, Any],
    *,
    max_input_tokens: int | None,
) -> dict[str, Any]:
    prompt_text = str(row.get("prompt", ""))
    shared_context_text = str(row.get("shared_context_text", ""))
    donor_prompt = str(row.get("donor_prompt", prompt_text))
    donor_shared_context_text = str(row.get("donor_shared_context_text", shared_context_text))
    deceptive_sentence = str(row.get("deceptive_commitment_sentence", "")).strip()
    original_truthful_sentence = str(row.get("truthful_donor_sentence", "")).strip()

    target_token_count = _continuation_token_count(
        tokenizer,
        prompt_text=prompt_text,
        shared_context_text=shared_context_text,
        continuation_text=deceptive_sentence,
        max_input_tokens=max_input_tokens,
    )

    donor_continuation = _strip_generation_prefix(
        row.get("donor_full_generation_text", ""),
        prompt_text=donor_prompt,
        shared_context_text=donor_shared_context_text,
    )
    if not donor_continuation:
        donor_continuation = original_truthful_sentence

    first_sentence, remainder_text = ap.extract_first_sentence(donor_continuation)
    patch_text = first_sentence.strip() or original_truthful_sentence
    if original_truthful_sentence:
        original_norm = ap.normalize_sentence_for_compare(original_truthful_sentence)
        first_norm = ap.normalize_sentence_for_compare(first_sentence)
        if first_norm != original_norm:
            located_idx = donor_continuation.find(original_truthful_sentence)
            if located_idx >= 0:
                remainder_text = donor_continuation[located_idx + len(original_truthful_sentence) :].lstrip()
            patch_text = original_truthful_sentence
    sentence_count = 1 if patch_text else 0
    donor_token_count = _continuation_token_count(
        tokenizer,
        prompt_text=prompt_text,
        shared_context_text=shared_context_text,
        continuation_text=patch_text,
        max_input_tokens=max_input_tokens,
    ) if patch_text else 0

    while donor_token_count < target_token_count and remainder_text:
        next_sentence, remainder_text = ap.extract_first_sentence(remainder_text)
        next_sentence = str(next_sentence).strip()
        if not _usable_donor_extension_chunk(next_sentence):
            break
        patch_text = ap.append_continuation(patch_text, next_sentence)
        sentence_count += 1
        donor_token_count = _continuation_token_count(
            tokenizer,
            prompt_text=prompt_text,
            shared_context_text=shared_context_text,
            continuation_text=patch_text,
            max_input_tokens=max_input_tokens,
        )

    donor_budget_satisfied = bool(donor_token_count >= target_token_count and patch_text.strip())
    return {
        "truthful_donor_patch_text": patch_text,
        "truthful_donor_patch_sentence_count": int(sentence_count),
        "truthful_donor_patch_token_count": int(donor_token_count),
        "deceptive_patch_token_count": int(target_token_count),
        "truthful_donor_patch_budget_satisfied": donor_budget_satisfied,
        "truthful_donor_extension_remainder_available": bool(str(remainder_text or "").strip()),
    }


def ensure_truthful_donor_token_budget(
    tokenizer: Any,
    pairs_df: pd.DataFrame,
    *,
    max_input_tokens: int | None,
    allow_empty: bool = False,
    context_label: str | None = None,
) -> pd.DataFrame:
    if pairs_df.empty:
        return pairs_df.copy()

    prefix = f"{context_label}: " if context_label else ""
    input_count = int(len(pairs_df))
    augmented_rows = []
    for row in pairs_df.to_dict(orient="records"):
        donor_patch_info = _build_truthful_donor_patch_text(
            tokenizer,
            row,
            max_input_tokens=max_input_tokens,
        )
        patch_text = str(donor_patch_info["truthful_donor_patch_text"])
        donor_prompt = str(row.get("donor_prompt", row.get("prompt", "")))
        donor_shared_context_text = str(row.get("donor_shared_context_text", row.get("shared_context_text", "")))
        augmented_rows.append(
            {
                **row,
                **donor_patch_info,
                "truthful_difference_branch_text": str(
                    row.get(
                        "truthful_difference_branch_text",
                        row.get("truthful_branch_text", ""),
                    )
                ),
                "truthful_donor_was_extended": bool(
                    ap.normalize_sentence_for_compare(patch_text)
                    != ap.normalize_sentence_for_compare(str(row.get("truthful_donor_sentence", "")))
                ),
                "truthful_branch_text": str(row.get("prompt", "")) + ap.append_continuation(
                    str(row.get("shared_context_text", "")),
                    patch_text,
                ),
            }
        )

    augmented_df = pd.DataFrame(augmented_rows)
    augmented_df = augmented_df.loc[
        augmented_df["truthful_donor_patch_budget_satisfied"].astype(bool)
    ].copy()
    if augmented_df.empty:
        message = "No commitment pairs remained after enforcing truthful donor token coverage."
        if allow_empty:
            print(f"{prefix}{message} Returning an empty pair set.")
            return augmented_df.reset_index(drop=True)
        raise ValueError(message)
    extended_count = int(augmented_df["truthful_donor_was_extended"].sum())
    dropped_count = input_count - int(len(augmented_df))
    print(
        f"{prefix}Truthful donor token-budget check: "
        f"kept {len(augmented_df)}/{input_count} pairs | "
        f"extended {extended_count} donor continuations | "
        f"dropped {dropped_count} insufficient-donor pairs"
    )
    return augmented_df.reset_index(drop=True)


def upsample_pairs_with_replacement(
    pairs_df: pd.DataFrame,
    *,
    target_count: int,
    seed: int,
    context_label: str | None = None,
) -> pd.DataFrame:
    if pairs_df.empty:
        return pairs_df.copy()

    target_count = int(target_count)
    base_df = pairs_df.copy().reset_index(drop=True)
    if "sampled_with_replacement" not in base_df.columns:
        base_df["sampled_with_replacement"] = False
    else:
        base_df["sampled_with_replacement"] = base_df["sampled_with_replacement"].fillna(False).astype(bool)
    if "replacement_source_pair_index" not in base_df.columns:
        base_df["replacement_source_pair_index"] = pd.to_numeric(
            base_df.get("pair_index", pd.Series(range(len(base_df)))),
            errors="coerce",
        ).fillna(-1).astype(int)
    if "replacement_source_pair_id" not in base_df.columns:
        base_df["replacement_source_pair_id"] = base_df.get(
            "pair_id",
            pd.Series([f"pair_{idx}" for idx in range(len(base_df))]),
        ).astype(str)

    if target_count <= 0 or len(base_df) >= target_count:
        return base_df

    prefix = f"{context_label}: " if context_label else ""
    extra_count = target_count - int(len(base_df))
    rng = np.random.default_rng(int(seed))
    sampled_positions = np.atleast_1d(rng.choice(len(base_df), size=extra_count, replace=True))
    max_pair_index = int(pd.to_numeric(base_df["pair_index"], errors="coerce").fillna(-1).max())
    cloned_rows: list[dict[str, Any]] = []
    for clone_offset, sampled_pos in enumerate(sampled_positions.tolist()):
        source_row = base_df.iloc[int(sampled_pos)].to_dict()
        source_pair_id = str(source_row.get("replacement_source_pair_id", source_row.get("pair_id", sampled_pos)))
        source_pair_index = int(source_row.get("replacement_source_pair_index", source_row.get("pair_index", sampled_pos)))
        source_row["sampled_with_replacement"] = True
        source_row["replacement_source_pair_id"] = source_pair_id
        source_row["replacement_source_pair_index"] = source_pair_index
        source_row["pair_index"] = max_pair_index + int(clone_offset) + 1
        source_row["pair_id"] = f"{source_pair_id}__resample_{int(clone_offset):03d}"
        cloned_rows.append(source_row)

    augmented_df = pd.concat([base_df, pd.DataFrame(cloned_rows)], ignore_index=True, sort=False)
    print(
        f"{prefix}Upsampled {len(base_df)} surviving pair(s) to {len(augmented_df)} with replacement "
        f"(added {extra_count} reused pair copies)."
    )
    return augmented_df.reset_index(drop=True)


def _assign_group_subset(
    groups_df: pd.DataFrame,
    *,
    target_pairs: int,
) -> tuple[set[str], pd.DataFrame]:
    if target_pairs <= 0 or groups_df.empty:
        return set(), groups_df
    chosen_group_ids: list[str] = []
    total_pairs = 0
    remaining = groups_df.copy()
    while not remaining.empty and total_pairs < int(target_pairs):
        row = remaining.iloc[0]
        group_id = str(row["split_group_id"])
        chosen_group_ids.append(group_id)
        total_pairs += int(row["n_pairs"])
        remaining = remaining.iloc[1:].reset_index(drop=True)
    return set(chosen_group_ids), remaining


def assign_pair_splits(
    pairs_df: pd.DataFrame,
    *,
    train_pair_count: int,
    validation_pair_count: int,
    test_pair_count: int,
    seed: int,
) -> pd.DataFrame:
    split_df = pairs_df.copy().reset_index(drop=True)
    split_df["split_group_id"] = split_df["example_id"].astype(str)
    split_df["split"] = "unused"
    group_sizes = (
        split_df.groupby("split_group_id", dropna=False)
        .size()
        .rename("n_pairs")
        .reset_index()
        .sort_values(["n_pairs", "split_group_id"], ascending=[False, True])
        .sample(frac=1.0, random_state=deterministic_seed(seed, "bs_headonly_split"))
        .reset_index(drop=True)
    )
    test_groups, remaining = _assign_group_subset(group_sizes, target_pairs=int(test_pair_count))
    validation_groups, remaining = _assign_group_subset(remaining, target_pairs=int(validation_pair_count))
    train_groups, _remaining = _assign_group_subset(remaining, target_pairs=int(train_pair_count))

    split_df.loc[split_df["split_group_id"].isin(train_groups), "split"] = "train"
    split_df.loc[split_df["split_group_id"].isin(validation_groups), "split"] = "validation"
    split_df.loc[split_df["split_group_id"].isin(test_groups), "split"] = "test"
    return split_df


def encode_branch(tokenizer: Any, prefix_text: str, full_text: str, *, max_input_tokens: int | None) -> dict[str, Any]:
    prefix_ids = encode_text_for_model(
        tokenizer,
        prefix_text,
        max_input_tokens=max_input_tokens,
    )["input_ids"][0]
    full_ids = encode_text_for_model(
        tokenizer,
        full_text,
        max_input_tokens=max_input_tokens,
    )["input_ids"][0]
    prefix_len = int(prefix_ids.shape[0])
    total_len = int(full_ids.shape[0])
    if total_len <= prefix_len:
        raise ValueError("Each branch needs at least one token after the shared prefix.")
    if not torch.equal(full_ids[:prefix_len], prefix_ids):
        raise ValueError("Full branch tokenization does not start with the shared prefix tokens.")
    return {
        "full_ids": full_ids,
        "prefix_len": prefix_len,
        "total_len": total_len,
        "score_start_pos": prefix_len,
        "score_stop_pos": total_len,
    }


def prepare_pair_records(
    tokenizer: Any,
    pairs_df: pd.DataFrame,
    *,
    max_input_tokens: int | None,
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for pair_row in pairs_df.to_dict(orient="records"):
        deceptive_encoded = encode_branch(
            tokenizer,
            pair_row["shared_prefix_text"],
            pair_row["deceptive_branch_text"],
            max_input_tokens=max_input_tokens,
        )
        truthful_encoded = encode_branch(
            tokenizer,
            pair_row["shared_prefix_text"],
            pair_row["truthful_branch_text"],
            max_input_tokens=max_input_tokens,
        )
        truthful_difference_branch_text = str(
            pair_row.get("truthful_difference_branch_text", pair_row["truthful_branch_text"])
        )
        truthful_difference_encoded = encode_branch(
            tokenizer,
            pair_row["shared_prefix_text"],
            truthful_difference_branch_text,
            max_input_tokens=max_input_tokens,
        )
        if int(deceptive_encoded["prefix_len"]) != int(truthful_encoded["prefix_len"]):
            raise ValueError("Prefix token lengths differ across deceptive/truthful branches.")
        if int(deceptive_encoded["prefix_len"]) != int(truthful_difference_encoded["prefix_len"]):
            raise ValueError("Prefix token lengths differ across deceptive/truthful-difference branches.")
        prepared.append(
            {
                **pair_row,
                "deceptive_encoded": deceptive_encoded,
                "truthful_encoded": truthful_encoded,
                "truthful_difference_encoded": truthful_difference_encoded,
                "prefix_token_len": int(deceptive_encoded["prefix_len"]),
                "deceptive_total_len": int(deceptive_encoded["total_len"]),
                "truthful_total_len": int(truthful_encoded["total_len"]),
                "max_total_len": int(max(deceptive_encoded["total_len"], truthful_encoded["total_len"])),
            }
        )
    return sorted(prepared, key=lambda row: (int(row["max_total_len"]), int(row["pair_index"])))


def pair_chunk_slices(prepared_pairs: list[dict[str, Any]], batch_pair_count: int) -> list[list[dict[str, Any]]]:
    if int(batch_pair_count) <= 0:
        raise ValueError("--batch-pair-count must be positive.")
    return [
        prepared_pairs[start : start + int(batch_pair_count)]
        for start in range(0, len(prepared_pairs), int(batch_pair_count))
    ]


def make_row(prepared_pair: dict[str, Any], branch_role: str) -> dict[str, Any]:
    if branch_role == "deceptive":
        encoded = prepared_pair["deceptive_encoded"]
        sentence_text = prepared_pair["deceptive_commitment_sentence"]
    elif branch_role == "truthful":
        encoded = prepared_pair["truthful_encoded"]
        sentence_text = prepared_pair.get("truthful_donor_patch_text", prepared_pair["truthful_donor_sentence"])
    elif branch_role == "truthful_difference":
        encoded = prepared_pair["truthful_difference_encoded"]
        sentence_text = prepared_pair["truthful_donor_sentence"]
    else:
        raise ValueError(f"Unsupported branch_role={branch_role!r}")
    return {
        "pair_index": int(prepared_pair["pair_index"]),
        "pair_id": str(prepared_pair.get("pair_id", prepared_pair["pair_index"])),
        "example_id": str(prepared_pair["example_id"]),
        "branch_role": branch_role,
        "sentence_text": sentence_text,
        **encoded,
    }


def pad_rows(rows: list[dict[str, Any]], tokenizer: Any, *, max_len: int | None = None) -> dict[str, Any]:
    if max_len is None:
        max_len = max(int(row["total_len"]) for row in rows)
    input_ids = torch.full((len(rows), int(max_len)), tokenizer.pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(rows), int(max_len)), dtype=torch.long)
    for row_idx, row in enumerate(rows):
        total_len = int(row["total_len"])
        input_ids[row_idx, :total_len] = row["full_ids"]
        attention_mask[row_idx, :total_len] = 1
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "rows": rows,
    }


def build_source_target_batches(
    tokenizer: Any,
    source_pairs: list[dict[str, Any]],
    target_pairs: list[dict[str, Any]],
    *,
    source_role: str = "truthful",
    target_role: str = "deceptive",
) -> tuple[dict[str, Any], dict[str, Any], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    if len(source_pairs) != len(target_pairs):
        raise ValueError("source_pairs and target_pairs must have the same batch size.")
    source_rows = [make_row(prepared_pair, source_role) for prepared_pair in source_pairs]
    target_rows = [make_row(prepared_pair, target_role) for prepared_pair in target_pairs]
    max_len = max(
        max(int(row["total_len"]) for row in source_rows),
        max(int(row["total_len"]) for row in target_rows),
    )
    source_batch = pad_rows(source_rows, tokenizer, max_len=max_len)
    target_batch = pad_rows(target_rows, tokenizer, max_len=max_len)
    source_inputs = {
        "input_ids": source_batch["input_ids"],
        "attention_mask": source_batch["attention_mask"],
        "use_cache": False,
    }
    target_inputs = {
        "input_ids": target_batch["input_ids"],
        "attention_mask": target_batch["attention_mask"],
        "use_cache": False,
    }
    return source_batch, target_batch, source_inputs, target_inputs


def prepare_split(
    tokenizer: Any,
    split_name: str,
    split_pairs_df: pd.DataFrame,
    *,
    batch_pair_count: int,
    max_input_tokens: int | None,
) -> PreparedSplit:
    if split_pairs_df.empty:
        raise ValueError(f"No pairs are available for split={split_name!r}.")
    prepared_pairs = prepare_pair_records(tokenizer, split_pairs_df, max_input_tokens=max_input_tokens)
    pair_chunks = pair_chunk_slices(prepared_pairs, batch_pair_count)
    dataset_max_input_tokens = max(int(pair["max_total_len"]) for pair in prepared_pairs)
    pairs_overview_df = pd.DataFrame(
        [
            {
                "split": split_name,
                "pair_index": int(pair["pair_index"]),
                "pair_id": str(pair.get("pair_id", pair["pair_index"])),
                "example_id": str(pair["example_id"]),
                "batch_index": int(batch_idx),
                "prefix_token_len": int(pair["prefix_token_len"]),
                "deceptive_total_len": int(pair["deceptive_total_len"]),
                "truthful_total_len": int(pair["truthful_total_len"]),
                "max_total_len": int(pair["max_total_len"]),
                "shared_context_num_valid": int(pair["shared_context_num_valid"]),
                "deceptive_prefix_num_valid": int(pair["deceptive_prefix_num_valid"]),
                "commitment_delta": float(pair["commitment_delta"]),
                "truthful_donor_was_extended": bool(pair.get("truthful_donor_was_extended", False)),
                "truthful_donor_patch_sentence_count": int(pair.get("truthful_donor_patch_sentence_count", 1)),
                "truthful_donor_patch_token_count": int(pair.get("truthful_donor_patch_token_count", 0)),
                "deceptive_patch_token_count": int(pair.get("deceptive_patch_token_count", 0)),
            }
            for batch_idx, chunk in enumerate(pair_chunks)
            for pair in chunk
        ]
    ).sort_values("pair_index").reset_index(drop=True)
    return PreparedSplit(
        split_name=split_name,
        pairs_df=split_pairs_df.copy().reset_index(drop=True),
        prepared_pairs=prepared_pairs,
        pair_chunks=pair_chunks,
        total_pairs=len(prepared_pairs),
        dataset_max_input_tokens=dataset_max_input_tokens,
        pairs_overview_df=pairs_overview_df,
    )


def scored_logits_and_targets(logits: torch.Tensor, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    scored_logits = []
    scored_targets = []
    row_lengths = []
    for row_idx, row in enumerate(batch["rows"]):
        start = int(row["score_start_pos"])
        stop = int(row["score_stop_pos"])
        row_logits = logits[row_idx, start - 1 : stop - 1, :]
        row_targets = batch["input_ids"][row_idx, start:stop].to(logits.device)
        scored_logits.append(row_logits)
        scored_targets.append(row_targets)
        row_lengths.append(int(stop - start))
    return torch.cat(scored_logits, dim=0), torch.cat(scored_targets, dim=0), row_lengths


def scored_hidden_and_targets(hidden: torch.Tensor, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    scored_hidden = []
    scored_targets = []
    row_lengths = []
    for row_idx, row in enumerate(batch["rows"]):
        start = int(row["score_start_pos"])
        stop = int(row["score_stop_pos"])
        row_hidden = hidden[row_idx, start - 1 : stop - 1, :]
        row_targets = batch["input_ids"][row_idx, start:stop].to(hidden.device)
        scored_hidden.append(row_hidden)
        scored_targets.append(row_targets)
        row_lengths.append(int(stop - start))
    return torch.cat(scored_hidden, dim=0), torch.cat(scored_targets, dim=0), row_lengths


def score_sentence_token_log_probs(row_token_log_probs: torch.Tensor, *, sentence_score_mode: str) -> torch.Tensor:
    row_token_log_probs = row_token_log_probs.float()
    if sentence_score_mode == "mean_logprob":
        return row_token_log_probs.mean()
    if sentence_score_mode == "sum_logprob":
        return row_token_log_probs.sum()
    if sentence_score_mode == "geomean_prob":
        return torch.exp(row_token_log_probs.mean())
    if sentence_score_mode == "sentence_prob":
        return torch.exp(row_token_log_probs.sum())
    raise ValueError(f"Unsupported sentence_score_mode={sentence_score_mode!r}.")


def sentence_score_by_row(
    logits: torch.Tensor,
    batch: dict[str, Any],
    *,
    sentence_score_mode: str,
) -> torch.Tensor:
    flat_logits, flat_targets, row_lengths = scored_logits_and_targets(logits, batch)
    return sentence_scores_from_flat_logits(
        flat_logits,
        flat_targets,
        row_lengths,
        sentence_score_mode=sentence_score_mode,
    )


def sentence_scores_from_flat_logits(
    flat_logits: torch.Tensor,
    flat_targets: torch.Tensor,
    row_lengths: list[int],
    *,
    sentence_score_mode: str,
) -> torch.Tensor:
    token_log_probs = -F.cross_entropy(flat_logits, flat_targets, reduction="none")
    scores = []
    offset = 0
    for row_len in row_lengths:
        row_token_log_probs = token_log_probs[offset : offset + row_len]
        scores.append(score_sentence_token_log_probs(row_token_log_probs, sentence_score_mode=sentence_score_mode))
        offset += row_len
    return torch.stack(scores)


def target_metric_from_logits(
    logits: torch.Tensor,
    target_batch: dict[str, Any],
    *,
    sentence_score_mode: str,
) -> torch.Tensor:
    target_scores = sentence_score_by_row(logits, target_batch, sentence_score_mode=sentence_score_mode)
    return -target_scores.mean()


def target_metric_from_hidden(
    hidden: torch.Tensor,
    target_batch: dict[str, Any],
    *,
    output_head_module: Any,
    sentence_score_mode: str,
) -> torch.Tensor:
    flat_hidden, flat_targets, row_lengths = scored_hidden_and_targets(hidden, target_batch)
    flat_logits = F.linear(
        flat_hidden,
        output_head_module.weight,
        getattr(output_head_module, "bias", None),
    )
    target_scores = sentence_scores_from_flat_logits(
        flat_logits,
        flat_targets,
        row_lengths,
        sentence_score_mode=sentence_score_mode,
    )
    return -target_scores.mean()


def traced_target_metric(
    runtime: HeadModelRuntime,
    target_batch: dict[str, Any],
    *,
    sentence_score_mode: str,
) -> torch.Tensor:
    if runtime.output_head_path is not None:
        hidden = output_head_input(runtime)
        return target_metric_from_hidden(
            hidden,
            target_batch,
            output_head_module=runtime.output_head_module,
            sentence_score_mode=sentence_score_mode,
        )
    logits = runtime.model.lm_head.output
    return target_metric_from_logits(logits, target_batch, sentence_score_mode=sentence_score_mode)


def example_game_id(example_id: object) -> str:
    text = str(example_id or "").strip()
    match = re.search(r"^(.*?/game_\d+)(?:/|$)", text)
    if match:
        return match.group(1)
    return re.sub(r"/sample_\d+$", "", text)


def commitment_patch_token_count(
    row: dict[str, Any],
    *,
    patch_scope: str,
    patch_first_n_tokens: int,
) -> int:
    start = int(row["score_start_pos"])
    stop = int(row["score_stop_pos"])
    full_len = max(stop - start, 0)
    if patch_scope == "commitment_first":
        return min(1, full_len)
    if patch_scope == "commitment_half":
        return min(max(int(math.ceil(full_len / 2.0)), 1), full_len) if full_len > 0 else 0
    if patch_scope == "commitment_full":
        return full_len
    raise ValueError(f"Unsupported commitment patch scope={patch_scope!r}.")


def commitment_patch_bounds(
    row: dict[str, Any],
    *,
    patch_scope: str,
    patch_first_n_tokens: int,
) -> tuple[int, int]:
    start = int(row["score_start_pos"])
    span_len = commitment_patch_token_count(
        row,
        patch_scope=patch_scope,
        patch_first_n_tokens=patch_first_n_tokens,
    )
    return start, start + span_len


def paired_commitment_patch_span(
    source_row: dict[str, Any],
    target_row: dict[str, Any],
    *,
    patch_scope: str,
    patch_first_n_tokens: int,
) -> tuple[int, int, int]:
    source_start, source_stop = commitment_patch_bounds(
        source_row,
        patch_scope=patch_scope,
        patch_first_n_tokens=patch_first_n_tokens,
    )
    target_start, target_stop = commitment_patch_bounds(
        target_row,
        patch_scope=patch_scope,
        patch_first_n_tokens=patch_first_n_tokens,
    )
    span_len = min(source_stop - source_start, target_stop - target_start)
    return source_start, target_start, max(int(span_len), 0)


def activation_delta_for_patch_scope(
    patch_scope: str,
    patch_first_n_tokens: int,
    source: torch.Tensor,
    target: torch.Tensor,
    source_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
) -> torch.Tensor:
    delta = torch.zeros_like(target)
    if patch_scope in {"commitment_first", "commitment_half", "commitment_full"}:
        for row_idx, (source_row, target_row) in enumerate(zip(source_rows, target_rows)):
            source_start, target_start, span_len = paired_commitment_patch_span(
                source_row,
                target_row,
                patch_scope=patch_scope,
                patch_first_n_tokens=patch_first_n_tokens,
            )
            if span_len <= 0:
                continue
            delta[row_idx : row_idx + 1, target_start : target_start + span_len, :] = (
                source[row_idx : row_idx + 1, source_start : source_start + span_len, :]
                - target[row_idx : row_idx + 1, target_start : target_start + span_len, :]
            )
        return delta

    raise ValueError(f"Unsupported patch_scope={patch_scope!r}.")


def apply_scoped_head_patch(
    patch_scope: str,
    patch_first_n_tokens: int,
    head_dim: int,
    current: torch.Tensor,
    source: torch.Tensor,
    source_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    heads: list[int],
) -> None:
    for head_idx in heads:
        start = int(head_idx) * head_dim
        stop = start + head_dim
        if patch_scope in {"commitment_first", "commitment_half", "commitment_full"}:
            for row_idx, (source_row, target_row) in enumerate(zip(source_rows, target_rows)):
                source_start, target_start, span_len = paired_commitment_patch_span(
                    source_row,
                    target_row,
                    patch_scope=patch_scope,
                    patch_first_n_tokens=patch_first_n_tokens,
                )
                if span_len <= 0:
                    continue
                current[row_idx : row_idx + 1, target_start : target_start + span_len, start:stop] = source[
                    row_idx : row_idx + 1,
                    source_start : source_start + span_len,
                    start:stop,
                ]
        else:
            raise ValueError(f"Unsupported patch_scope={patch_scope!r}.")


def compute_split_baseline_summary(
    runtime: HeadModelRuntime,
    prepared_split: PreparedSplit,
    *,
    sentence_score_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    summary_records = []
    source_score_total = 0.0
    target_score_total = 0.0
    target_metric_total = 0.0
    truth_minus_deceptive_total = 0.0

    for chunk_idx, chunk_pairs in enumerate(prepared_split.pair_chunks, start=1):
        source_batch, target_batch, source_inputs, target_inputs = build_source_target_batches(
            runtime.tokenizer,
            chunk_pairs,
            chunk_pairs,
        )
        with torch.inference_mode():
            source_logits = runtime.model.trace(source_inputs, trace=False).logits
        source_scores = sentence_score_by_row(
            source_logits,
            source_batch,
            sentence_score_mode=sentence_score_mode,
        ).detach().cpu()
        del source_logits
        clear_memory(runtime.model)

        with torch.inference_mode():
            target_logits = runtime.model.trace(target_inputs, trace=False).logits
        target_scores = sentence_score_by_row(
            target_logits,
            target_batch,
            sentence_score_mode=sentence_score_mode,
        ).detach().cpu()
        del target_logits
        clear_memory(runtime.model)

        chunk_pair_count = len(chunk_pairs)
        chunk_source_score = float(source_scores.mean().item())
        chunk_target_score = float(target_scores.mean().item())
        chunk_target_metric = float((-target_scores).mean().item())
        chunk_truth_minus_deceptive = float((source_scores - target_scores).mean().item())
        source_score_total += chunk_source_score * float(chunk_pair_count)
        target_score_total += chunk_target_score * float(chunk_pair_count)
        target_metric_total += chunk_target_metric * float(chunk_pair_count)
        truth_minus_deceptive_total += chunk_truth_minus_deceptive * float(chunk_pair_count)

        for local_idx, pair_row in enumerate(chunk_pairs):
            score_h = float(source_scores[local_idx].item())
            score_d = float(target_scores[local_idx].item())
            summary_records.append(
                {
                    "split": prepared_split.split_name,
                    "pair_index": int(pair_row["pair_index"]),
                    "pair_id": str(pair_row.get("pair_id", pair_row["pair_index"])),
                    "example_id": str(pair_row["example_id"]),
                    "commitment_delta": float(pair_row["commitment_delta"]),
                    "max_total_len": int(pair_row["max_total_len"]),
                    "score_H_source": score_h,
                    "score_D_target": score_d,
                    "truthful_minus_deceptive": score_h - score_d,
                    "target_metric_neg_score_D": -score_d,
                    "deceptive_commitment_sentence": str(pair_row["deceptive_commitment_sentence"]),
                    "truthful_donor_sentence": str(pair_row["truthful_donor_sentence"]),
                }
            )
        print(
            f"[{prepared_split.split_name}] Scored chunk {chunk_idx}/{len(prepared_split.pair_chunks)} "
            f"| pairs={chunk_pair_count} | max_total_len={max(int(pair['max_total_len']) for pair in chunk_pairs)}"
        )
        del source_scores, target_scores, source_batch, target_batch, source_inputs, target_inputs
        clear_memory(runtime.model)

    split_totals = {
        "source_truthful_score": source_score_total / float(prepared_split.total_pairs),
        "target_deceptive_score": target_score_total / float(prepared_split.total_pairs),
        "target_baseline": target_metric_total / float(prepared_split.total_pairs),
        "truth_minus_deceptive_baseline": truth_minus_deceptive_total / float(prepared_split.total_pairs),
    }
    summary_df = pd.DataFrame(summary_records).sort_values("pair_index").reset_index(drop=True)
    metric_sanity_df = pd.DataFrame(
        [
            {
                "split": prepared_split.split_name,
                "n_pairs": int(len(summary_df)),
                "mean_score_H_source": float(summary_df["score_H_source"].mean()),
                "mean_score_D_target": float(summary_df["score_D_target"].mean()),
                "mean_truthful_minus_deceptive": float(summary_df["truthful_minus_deceptive"].mean()),
                "median_truthful_minus_deceptive": float(summary_df["truthful_minus_deceptive"].median()),
                "target_baseline_b_prime": float(split_totals["target_baseline"]),
                "fixed_truthful_score_reference": float(split_totals["source_truthful_score"]),
                "target_deceptive_score_reference": float(split_totals["target_deceptive_score"]),
                "truth_minus_deceptive_baseline": float(split_totals["truth_minus_deceptive_baseline"]),
                "sentence_score_mode": sentence_score_mode,
            }
        ]
    )
    return summary_df, metric_sanity_df, split_totals


def compute_head_attributions(
    runtime: HeadModelRuntime,
    prepared_split: PreparedSplit,
    *,
    sentence_score_mode: str,
    patch_scope: str,
    patch_first_n_tokens: int,
    offload_attr_tensors_to_cpu: bool,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    layer_sums = [torch.zeros(runtime.n_heads, dtype=torch.float32) for _ in range(runtime.n_layers)]
    diag_records = []
    print(
        f"Attribution objective: {patch_scope} patch truthful source activations into deceptive target "
        "and maximize -score_C(s_D | p)."
    )
    for chunk_idx, chunk_pairs in enumerate(prepared_split.pair_chunks, start=1):
        source_batch, target_batch, source_inputs, target_inputs = build_source_target_batches(
            runtime.tokenizer,
            chunk_pairs,
            chunk_pairs,
        )
        chunk_pair_count = len(chunk_pairs)
        chunk_max_total_len = max(int(pair["max_total_len"]) for pair in chunk_pairs)
        print(
            f"[train] Chunk {chunk_idx}/{len(prepared_split.pair_chunks)} | pairs={chunk_pair_count} "
            f"| max_total_len={chunk_max_total_len}"
        )

        for layer_idx, layer in enumerate(runtime.layers):
            try:
                clear_memory(runtime.model)
                with torch.inference_mode():
                    with runtime.model.trace(source_inputs):
                        source_proxy = attn_out_input(layer)
                        source_out = source_proxy.save()
                source_tensor = saved_value(source_out).detach()
                if offload_attr_tensors_to_cpu:
                    source_tensor = source_tensor.to("cpu")
                del source_out, source_proxy
                clear_memory(runtime.model)

                with runtime.model.trace(target_inputs):
                    target_proxy = attn_out_input(layer)
                    target_proxy.requires_grad_()
                    target_out = target_proxy.save()
                    target_grad = target_proxy.grad.save()
                    value = traced_target_metric(runtime, target_batch, sentence_score_mode=sentence_score_mode)
                    traced_value = value.save()
                    value.backward()
                zero_model_gradients(runtime.model)
                objective_value = float(saved_value(traced_value).item())
                target_tensor = saved_value(target_out).detach()
                target_grad_tensor = saved_value(target_grad).detach()
                if offload_attr_tensors_to_cpu:
                    target_tensor = target_tensor.to("cpu")
                    target_grad_tensor = target_grad_tensor.to("cpu")
                del target_out, target_grad, traced_value, value, target_proxy
                clear_memory(runtime.model)

                patch_delta = activation_delta_for_patch_scope(
                    patch_scope,
                    patch_first_n_tokens,
                    source_tensor,
                    target_tensor,
                    list(source_batch["rows"]),
                    list(target_batch["rows"]),
                )
                layer_attr = einops.reduce(
                    target_grad_tensor * patch_delta,
                    "batch pos (head d_head) -> head",
                    "sum",
                    head=runtime.n_heads,
                    d_head=runtime.head_dim,
                )
                grad_norm = float(target_grad_tensor.float().norm().item())
                attr_abs_sum = float(layer_attr.float().abs().sum().item())
                attr_max_abs = float(layer_attr.float().abs().max().item())

                layer_sums[layer_idx] += layer_attr.detach().float().cpu()
                diag_records.append(
                    {
                        "split": prepared_split.split_name,
                        "chunk_index": int(chunk_idx - 1),
                        "layer": int(layer_idx),
                        "chunk_pairs": int(chunk_pair_count),
                        "objective_value": objective_value,
                        "grad_norm": grad_norm,
                        "attr_abs_sum": attr_abs_sum,
                        "attr_max_abs": attr_max_abs,
                    }
                )
                print(
                    f"  Layer {layer_idx:02d} | obj={objective_value:.4f} | "
                    f"grad_norm={grad_norm:.4f} | attr_abs_sum={attr_abs_sum:.4f}"
                )
                del source_tensor, target_tensor, target_grad_tensor, patch_delta, layer_attr
                clear_memory(runtime.model)
            except RuntimeError as exc:
                clear_memory(runtime.model)
                if not is_cuda_oom_error(exc):
                    raise
                raise RuntimeError(
                    "CUDA OOM during head attribution "
                    f"(split={prepared_split.split_name}, chunk={chunk_idx}/{len(prepared_split.pair_chunks)}, "
                    f"layer={layer_idx}, pairs={chunk_pair_count}, max_total_len={chunk_max_total_len}). "
                    "This pass is already running with pairs=1, so reducing --batch-pair-count or the total "
                    "number of train pairs will not lower peak memory for this example. "
                    "If you disabled --offload-attr-tensors-to-cpu, try re-enabling it. "
                    "Otherwise rerun with a smaller --max-input-tokens value, for example 896, 768, or 640."
                ) from exc

        del source_batch, target_batch, source_inputs, target_inputs
        clear_memory(runtime.model)

    patching_results = torch.stack(
        [layer_sum / float(prepared_split.total_pairs) for layer_sum in layer_sums]
    ).float().numpy()
    diagnostics_df = pd.DataFrame(diag_records)
    layer_summary_df = (
        diagnostics_df.groupby("layer", as_index=False)[["grad_norm", "attr_abs_sum", "attr_max_abs"]]
        .mean()
        .sort_values("layer")
        .reset_index(drop=True)
    )
    return patching_results, diagnostics_df, layer_summary_df


def build_ranked_sites(
    patching_results: np.ndarray,
    *,
    n_layers: int,
    n_heads: int,
    circuit_select: str,
    max_circuit_size_raw: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[tuple[int, int]], list[tuple[int, int]]]:
    site_df = pd.DataFrame(
        [
            {
                "layer": int(layer_idx),
                "head": int(head_idx),
                "attribution": float(patching_results[layer_idx, head_idx]),
                "abs_attribution": abs(float(patching_results[layer_idx, head_idx])),
            }
            for layer_idx in range(n_layers)
            for head_idx in range(n_heads)
        ]
    )
    if circuit_select == "positive" and (site_df["attribution"] > 0).any():
        ranked_site_df = (
            site_df[site_df["attribution"] > 0]
            .sort_values("attribution", ascending=False)
            .reset_index(drop=True)
        )
    elif circuit_select in {"positive", "abs"}:
        ranked_site_df = site_df.sort_values("abs_attribution", ascending=False).reset_index(drop=True)
    else:
        raise ValueError("--circuit-select must be 'positive' or 'abs'.")

    if max_circuit_size_raw in {"", "auto", "all", "none"}:
        max_circuit_size = len(ranked_site_df)
    else:
        max_circuit_size = min(int(max_circuit_size_raw), len(ranked_site_df))
    if max_circuit_size <= 0:
        raise ValueError("No ranked circuit sites are available.")

    ranked_site_df = ranked_site_df.head(max_circuit_size).reset_index(drop=True)
    ranked_sites = [(int(row.layer), int(row.head)) for row in ranked_site_df.itertuples(index=False)]
    all_sites = [(layer_idx, head_idx) for layer_idx in range(n_layers) for head_idx in range(n_heads)]
    return site_df, ranked_site_df, ranked_sites, all_sites


def normalize_sites(selected_sites: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    return [(int(layer_idx), int(head_idx)) for layer_idx, head_idx in selected_sites]


def group_sites_by_layer(selected_sites: Iterable[tuple[int, int]]) -> dict[int, list[int]]:
    grouped: dict[int, set[int]] = {}
    for layer_idx, head_idx in normalize_sites(selected_sites):
        grouped.setdefault(layer_idx, set()).add(head_idx)
    return {layer_idx: sorted(heads) for layer_idx, heads in grouped.items()}


def subset_chunks(chunks: list[list[dict[str, Any]]], max_chunks: int) -> list[list[dict[str, Any]]]:
    if int(max_chunks) > 0:
        return chunks[: int(max_chunks)]
    return chunks


def percent_probability_reduction_from_delta(delta: float) -> float:
    return 100.0 * (1.0 - math.exp(-float(delta)))


def score_unpatched_chunks(
    runtime: HeadModelRuntime,
    chunks: list[list[dict[str, Any]]],
    *,
    sentence_score_mode: str,
    target_role: str = "deceptive",
) -> list[float]:
    scores = []
    for chunk_pairs in chunks:
        _, target_batch, _, target_inputs = build_source_target_batches(
            runtime.tokenizer,
            chunk_pairs,
            chunk_pairs,
            source_role="truthful",
            target_role=target_role,
        )
        with torch.inference_mode():
            logits = runtime.model.trace(target_inputs, trace=False).logits
        metric = target_metric_from_logits(logits, target_batch, sentence_score_mode=sentence_score_mode)
        scores.append(float(metric.item()))
        del logits, metric, target_batch, target_inputs
        clear_memory(runtime.model)
    return scores


def weighted_average_chunk_scores(chunks: list[list[dict[str, Any]]], scores: list[float]) -> float:
    weighted_total = 0.0
    weight_total = 0
    for chunk_pairs, score in zip(chunks, scores):
        weight = len(chunk_pairs)
        weighted_total += float(score) * weight
        weight_total += weight
    return weighted_total / float(weight_total)


def score_source_target_baselines(
    runtime: HeadModelRuntime,
    chunks: list[list[dict[str, Any]]],
    *,
    sentence_score_mode: str,
    return_chunk_records: bool = False,
) -> tuple[float, float, float, float] | tuple[float, float, float, float, list[dict[str, Any]]]:
    source_score_total = 0.0
    target_score_total = 0.0
    target_metric_total = 0.0
    truth_minus_total = 0.0
    weight_total = 0
    chunk_records = []
    for chunk_idx, chunk_pairs in enumerate(chunks):
        source_batch, target_batch, source_inputs, target_inputs = build_source_target_batches(
            runtime.tokenizer,
            chunk_pairs,
            chunk_pairs,
        )
        weight = len(chunk_pairs)
        with torch.inference_mode():
            source_logits = runtime.model.trace(source_inputs, trace=False).logits
        source_scores = sentence_score_by_row(
            source_logits,
            source_batch,
            sentence_score_mode=sentence_score_mode,
        ).detach().cpu()
        source_score = float(source_scores.mean().item())
        del source_logits
        clear_memory(runtime.model)

        with torch.inference_mode():
            target_logits = runtime.model.trace(target_inputs, trace=False).logits
        target_scores = sentence_score_by_row(
            target_logits,
            target_batch,
            sentence_score_mode=sentence_score_mode,
        ).detach().cpu()
        target_score = float(target_scores.mean().item())
        target_metric = float((-target_scores).mean().item())
        truth_minus = float((source_scores - target_scores).mean().item())
        del target_logits
        clear_memory(runtime.model)

        source_score_total += source_score * weight
        target_score_total += target_score * weight
        target_metric_total += target_metric * weight
        truth_minus_total += truth_minus * weight
        weight_total += weight
        chunk_records.append(
            {
                "chunk_index": int(chunk_idx),
                "n_pairs": int(weight),
                "source_truthful_score": source_score,
                "target_deceptive_score": target_score,
                "target_metric": target_metric,
                "truth_minus_deceptive_metric": truth_minus,
            }
        )
        del source_scores, target_scores, source_batch, target_batch, source_inputs, target_inputs
        clear_memory(runtime.model)

    source_truthful_score = source_score_total / weight_total
    target_deceptive_score = target_score_total / weight_total
    target_baseline = target_metric_total / weight_total
    truth_minus_deceptive = truth_minus_total / weight_total
    if return_chunk_records:
        return source_truthful_score, target_deceptive_score, target_baseline, truth_minus_deceptive, chunk_records
    return source_truthful_score, target_deceptive_score, target_baseline, truth_minus_deceptive


def eligible_donor_pairs_for_target(
    target_pair: dict[str, Any],
    donor_pool: list[dict[str, Any]],
    *,
    require_different_game: bool,
) -> list[dict[str, Any]]:
    target_pair_index = int(target_pair["pair_index"])
    target_game_id = example_game_id(target_pair.get("example_id"))
    eligible = []
    for donor_pair in donor_pool:
        if int(donor_pair["pair_index"]) == target_pair_index:
            continue
        if require_different_game and example_game_id(donor_pair.get("example_id")) == target_game_id:
            continue
        eligible.append(donor_pair)
    return eligible if eligible else list(donor_pool)


def sample_donor_groups_for_chunk(
    chunk_pairs: list[dict[str, Any]],
    donor_pool: list[dict[str, Any]],
    *,
    rng: np.random.Generator,
    donor_count: int,
    require_different_game: bool,
) -> list[list[dict[str, Any]]]:
    donor_pool = list(donor_pool)
    donor_groups: list[list[dict[str, Any]]] = []
    for target_pair in chunk_pairs:
        eligible = eligible_donor_pairs_for_target(
            target_pair,
            donor_pool,
            require_different_game=require_different_game,
        )
        if not eligible:
            eligible = donor_pool
        replace = len(eligible) < int(donor_count)
        sampled_indices = np.atleast_1d(
            rng.choice(len(eligible), size=int(donor_count), replace=replace)
        )
        donor_groups.append([eligible[int(idx)] for idx in sampled_indices.tolist()])
    return donor_groups


def shuffled_donor_pairs_for_chunk(
    chunk_pairs: list[dict[str, Any]],
    donor_pool: list[dict[str, Any]],
    *,
    rng: np.random.Generator,
    require_different_game: bool = False,
) -> list[dict[str, Any]]:
    donor_groups = sample_donor_groups_for_chunk(
        chunk_pairs,
        donor_pool,
        rng=rng,
        donor_count=1,
        require_different_game=require_different_game,
    )
    return [group[0] for group in donor_groups]


def mean_shuffled_donor_groups_for_chunk(
    chunk_pairs: list[dict[str, Any]],
    donor_pool: list[dict[str, Any]],
    *,
    rng: np.random.Generator,
    donor_count: int,
    require_different_game: bool = False,
) -> list[list[dict[str, Any]]]:
    return sample_donor_groups_for_chunk(
        chunk_pairs,
        donor_pool,
        rng=rng,
        donor_count=donor_count,
        require_different_game=require_different_game,
    )


def averaged_shuffled_difference_donor_groups_for_chunk(
    chunk_pairs: list[dict[str, Any]],
    donor_pool: list[dict[str, Any]],
    *,
    rng: np.random.Generator,
    donor_count: int,
    require_different_game: bool = False,
) -> list[list[dict[str, Any]]]:
    return sample_donor_groups_for_chunk(
        chunk_pairs,
        donor_pool,
        rng=rng,
        donor_count=donor_count,
        require_different_game=require_different_game,
    )


def apply_mean_scoped_head_patch(
    patch_scope: str,
    patch_first_n_tokens: int,
    head_dim: int,
    current: torch.Tensor,
    source: torch.Tensor,
    source_rows: list[dict[str, Any]],
    source_group_slices: list[tuple[int, int]],
    target_rows: list[dict[str, Any]],
    heads: list[int],
) -> None:
    for head_idx in heads:
        start = int(head_idx) * head_dim
        stop = start + head_dim
        for target_row_idx, target_row in enumerate(target_rows):
            donor_start_idx, donor_stop_idx = source_group_slices[target_row_idx]
            donor_vectors = []
            for donor_row_idx in range(int(donor_start_idx), int(donor_stop_idx)):
                donor_row = source_rows[donor_row_idx]
                donor_start, donor_stop = commitment_patch_bounds(
                    donor_row,
                    patch_scope=patch_scope,
                    patch_first_n_tokens=patch_first_n_tokens,
                )
                if donor_stop <= donor_start:
                    continue
                donor_vectors.append(source[donor_row_idx, donor_start:donor_stop, start:stop].mean(dim=0))
            if not donor_vectors:
                continue
            mean_vec = torch.stack(donor_vectors, dim=0).mean(dim=0).view(1, 1, head_dim)
            target_start, target_stop = commitment_patch_bounds(
                target_row,
                patch_scope=patch_scope,
                patch_first_n_tokens=patch_first_n_tokens,
            )
            if target_stop <= target_start:
                continue
            current[target_row_idx : target_row_idx + 1, target_start:target_stop, start:stop] = mean_vec.expand(
                1,
                target_stop - target_start,
                head_dim,
            )


def apply_difference_scoped_head_patch(
    patch_scope: str,
    patch_first_n_tokens: int,
    head_dim: int,
    current: torch.Tensor,
    donor_deceptive: torch.Tensor,
    donor_truthful: torch.Tensor,
    donor_deceptive_rows: list[dict[str, Any]],
    donor_truthful_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    heads: list[int],
) -> None:
    for head_idx in heads:
        start = int(head_idx) * head_dim
        stop = start + head_dim
        for row_idx, target_row in enumerate(target_rows):
            donor_start, donor_truth_start, donor_span_len = paired_commitment_patch_span(
                donor_deceptive_rows[row_idx],
                donor_truthful_rows[row_idx],
                patch_scope=patch_scope,
                patch_first_n_tokens=patch_first_n_tokens,
            )
            if donor_span_len <= 0:
                continue
            target_start, target_stop = commitment_patch_bounds(
                target_row,
                patch_scope=patch_scope,
                patch_first_n_tokens=patch_first_n_tokens,
            )
            target_span_len = max(int(target_stop - target_start), 0)
            span_len = min(int(donor_span_len), int(target_span_len))
            if span_len <= 0:
                continue
            current[row_idx : row_idx + 1, target_start : target_start + span_len, start:stop] = (
                current[row_idx : row_idx + 1, target_start : target_start + span_len, start:stop]
                + donor_deceptive[
                    row_idx : row_idx + 1,
                    donor_start : donor_start + span_len,
                    start:stop,
                ]
                - donor_truthful[
                    row_idx : row_idx + 1,
                    donor_truth_start : donor_truth_start + span_len,
                    start:stop,
                ]
            )


def apply_mean_difference_scoped_head_patch(
    patch_scope: str,
    patch_first_n_tokens: int,
    head_dim: int,
    current: torch.Tensor,
    donor_deceptive: torch.Tensor,
    donor_truthful: torch.Tensor,
    donor_deceptive_rows: list[dict[str, Any]],
    donor_truthful_rows: list[dict[str, Any]],
    donor_group_slices: list[tuple[int, int]],
    target_rows: list[dict[str, Any]],
    heads: list[int],
) -> None:
    for head_idx in heads:
        start = int(head_idx) * head_dim
        stop = start + head_dim
        for target_row_idx, target_row in enumerate(target_rows):
            target_start, target_stop = commitment_patch_bounds(
                target_row,
                patch_scope=patch_scope,
                patch_first_n_tokens=patch_first_n_tokens,
            )
            target_span_len = max(int(target_stop - target_start), 0)
            if target_span_len <= 0:
                continue
            donor_start_idx, donor_stop_idx = donor_group_slices[target_row_idx]
            for token_offset in range(target_span_len):
                donor_vectors = []
                for donor_row_idx in range(int(donor_start_idx), int(donor_stop_idx)):
                    donor_start, donor_truth_start, donor_span_len = paired_commitment_patch_span(
                        donor_deceptive_rows[donor_row_idx],
                        donor_truthful_rows[donor_row_idx],
                        patch_scope=patch_scope,
                        patch_first_n_tokens=patch_first_n_tokens,
                    )
                    if token_offset >= int(donor_span_len):
                        continue
                    donor_vectors.append(
                        donor_deceptive[
                            donor_row_idx,
                            donor_start + token_offset,
                            start:stop,
                        ]
                        - donor_truthful[
                            donor_row_idx,
                            donor_truth_start + token_offset,
                            start:stop,
                        ]
                    )
                if not donor_vectors:
                    continue
                mean_diff = torch.stack(donor_vectors, dim=0).mean(dim=0)
                target_pos = target_start + token_offset
                current[target_row_idx, target_pos, start:stop] = (
                    current[target_row_idx, target_pos, start:stop] + mean_diff
                )


def patch_circuit_chunk(
    runtime: HeadModelRuntime,
    chunk_pairs: list[dict[str, Any]],
    selected_sites: list[tuple[int, int]],
    *,
    sentence_score_mode: str,
    patch_scope: str,
    patch_first_n_tokens: int,
    activation_device: torch.device,
    source_pairs: list[dict[str, Any]] | None = None,
    source_pair_groups: list[list[dict[str, Any]]] | None = None,
    difference_source_pairs: list[dict[str, Any]] | None = None,
    difference_source_pair_groups: list[list[dict[str, Any]]] | None = None,
    source_role: str = "truthful",
    target_role: str = "deceptive",
) -> float:
    grouped = group_sites_by_layer(selected_sites)
    if not grouped:
        raise ValueError("No circuit sites were selected.")
    mode_count = sum(
        int(option is not None)
        for option in (
            source_pairs,
            source_pair_groups,
            difference_source_pairs,
            difference_source_pair_groups,
        )
    )
    if mode_count > 1:
        raise ValueError(
            "Pass at most one of source_pairs, source_pair_groups, difference_source_pairs, "
            "or difference_source_pair_groups."
        )
    if (
        source_pairs is None
        and source_pair_groups is None
        and difference_source_pairs is None
        and difference_source_pair_groups is None
    ):
        source_pairs = chunk_pairs

    source_group_slices: list[tuple[int, int]] | None = None
    difference_group_slices: list[tuple[int, int]] | None = None
    donor_deceptive_rows: list[dict[str, Any]] | None = None
    donor_truthful_rows: list[dict[str, Any]] | None = None
    donor_truth_inputs: dict[str, torch.Tensor] | None = None
    if difference_source_pairs is not None:
        donor_deceptive_batch, donor_truthful_batch, source_inputs, donor_truth_inputs = build_source_target_batches(
            runtime.tokenizer,
            difference_source_pairs,
            difference_source_pairs,
            source_role="deceptive",
            target_role="truthful_difference",
        )
        donor_deceptive_rows = list(donor_deceptive_batch["rows"])
        donor_truthful_rows = list(donor_truthful_batch["rows"])
        target_batch = pad_rows([make_row(prepared_pair, target_role) for prepared_pair in chunk_pairs], runtime.tokenizer)
        target_rows = list(target_batch["rows"])
        target_inputs = {
            "input_ids": target_batch["input_ids"],
            "attention_mask": target_batch["attention_mask"],
            "use_cache": False,
        }
        del donor_truthful_batch
    elif difference_source_pair_groups is not None:
        if len(difference_source_pair_groups) != len(chunk_pairs):
            raise ValueError("difference_source_pair_groups must align 1:1 with chunk_pairs.")
        flat_difference_pairs: list[dict[str, Any]] = []
        difference_group_slices = []
        for donor_group in difference_source_pair_groups:
            start_idx = len(flat_difference_pairs)
            flat_difference_pairs.extend(donor_group)
            difference_group_slices.append((start_idx, len(flat_difference_pairs)))
        donor_deceptive_rows = [make_row(prepared_pair, "deceptive") for prepared_pair in flat_difference_pairs]
        donor_truthful_rows = [make_row(prepared_pair, "truthful_difference") for prepared_pair in flat_difference_pairs]
        target_rows = [make_row(prepared_pair, target_role) for prepared_pair in chunk_pairs]
        donor_deceptive_batch = pad_rows(donor_deceptive_rows, runtime.tokenizer)
        donor_truthful_batch = pad_rows(donor_truthful_rows, runtime.tokenizer)
        target_batch = pad_rows(target_rows, runtime.tokenizer)
        source_inputs = {
            "input_ids": donor_deceptive_batch["input_ids"],
            "attention_mask": donor_deceptive_batch["attention_mask"],
            "use_cache": False,
        }
        donor_truth_inputs = {
            "input_ids": donor_truthful_batch["input_ids"],
            "attention_mask": donor_truthful_batch["attention_mask"],
            "use_cache": False,
        }
        target_inputs = {
            "input_ids": target_batch["input_ids"],
            "attention_mask": target_batch["attention_mask"],
            "use_cache": False,
        }
        del donor_truthful_batch
    elif source_pair_groups is None:
        source_batch, target_batch, source_inputs, target_inputs = build_source_target_batches(
            runtime.tokenizer,
            source_pairs,
            chunk_pairs,
            source_role=source_role,
            target_role=target_role,
        )
        source_rows = list(source_batch["rows"])
        target_rows = list(target_batch["rows"])
    else:
        if len(source_pair_groups) != len(chunk_pairs):
            raise ValueError("source_pair_groups must align 1:1 with chunk_pairs.")
        flat_source_pairs: list[dict[str, Any]] = []
        source_group_slices = []
        for donor_group in source_pair_groups:
            start_idx = len(flat_source_pairs)
            flat_source_pairs.extend(donor_group)
            source_group_slices.append((start_idx, len(flat_source_pairs)))
        source_rows = [make_row(prepared_pair, source_role) for prepared_pair in flat_source_pairs]
        target_rows = [make_row(prepared_pair, target_role) for prepared_pair in chunk_pairs]
        source_batch = pad_rows(source_rows, runtime.tokenizer)
        target_batch = pad_rows(target_rows, runtime.tokenizer)
        source_inputs = {
            "input_ids": source_batch["input_ids"],
            "attention_mask": source_batch["attention_mask"],
            "use_cache": False,
        }
        target_inputs = {
            "input_ids": target_batch["input_ids"],
            "attention_mask": target_batch["attention_mask"],
            "use_cache": False,
        }
    source_proxies = {}
    with torch.inference_mode():
        with runtime.model.trace(source_inputs):
            for layer_idx in grouped:
                source_proxies[layer_idx] = attn_out_input(runtime.layers[layer_idx]).save()

    source_tensors = {
        layer_idx: saved_value(proxy).detach().to("cpu")
        for layer_idx, proxy in source_proxies.items()
    }
    donor_truth_proxies = {}
    donor_truth_tensors: dict[int, torch.Tensor] | None = None
    if donor_truth_inputs is not None:
        with torch.inference_mode():
            with runtime.model.trace(donor_truth_inputs):
                for layer_idx in grouped:
                    donor_truth_proxies[layer_idx] = attn_out_input(runtime.layers[layer_idx]).save()
        donor_truth_tensors = {
            layer_idx: saved_value(proxy).detach().to("cpu")
            for layer_idx, proxy in donor_truth_proxies.items()
        }
        del donor_truth_proxies, donor_truth_inputs, donor_deceptive_batch
    else:
        del donor_truth_proxies
    del source_proxies, source_inputs
    if 'source_batch' in locals():
        del source_batch
    clear_memory(runtime.model)

    device_sources = {
        layer_idx: tensor.to(activation_device)
        for layer_idx, tensor in source_tensors.items()
    }
    device_donor_truth: dict[int, torch.Tensor] | None = None
    if donor_truth_tensors is not None:
        device_donor_truth = {
            layer_idx: tensor.to(activation_device)
            for layer_idx, tensor in donor_truth_tensors.items()
        }
    with torch.inference_mode():
        with runtime.model.trace(target_inputs):
            for layer_idx, heads in grouped.items():
                current = attn_out_input(runtime.layers[layer_idx])
                source = device_sources[layer_idx]
                if difference_source_pairs is not None:
                    if donor_deceptive_rows is None or donor_truthful_rows is None or device_donor_truth is None:
                        raise ValueError("Missing donor deceptive/truthful rows for difference patch.")
                    apply_difference_scoped_head_patch(
                        patch_scope,
                        patch_first_n_tokens,
                        runtime.head_dim,
                        current,
                        source,
                        device_donor_truth[layer_idx],
                        donor_deceptive_rows,
                        donor_truthful_rows,
                        target_rows,
                        heads,
                    )
                elif difference_source_pair_groups is not None:
                    if (
                        donor_deceptive_rows is None
                        or donor_truthful_rows is None
                        or device_donor_truth is None
                        or difference_group_slices is None
                    ):
                        raise ValueError("Missing donor deceptive/truthful rows for averaged difference patch.")
                    apply_mean_difference_scoped_head_patch(
                        patch_scope,
                        patch_first_n_tokens,
                        runtime.head_dim,
                        current,
                        source,
                        device_donor_truth[layer_idx],
                        donor_deceptive_rows,
                        donor_truthful_rows,
                        difference_group_slices,
                        target_rows,
                        heads,
                    )
                elif source_group_slices is None:
                    apply_scoped_head_patch(
                        patch_scope,
                        patch_first_n_tokens,
                        runtime.head_dim,
                        current,
                        source,
                        source_rows,
                        target_rows,
                        heads,
                    )
                else:
                    apply_mean_scoped_head_patch(
                        patch_scope,
                        patch_first_n_tokens,
                        runtime.head_dim,
                        current,
                        source,
                        source_rows,
                        source_group_slices,
                        target_rows,
                        heads,
                    )
            metric = traced_target_metric(runtime, target_batch, sentence_score_mode=sentence_score_mode).save()

    value = float(saved_value(metric).item())
    del metric, target_batch, target_inputs, source_tensors, device_sources
    if donor_truth_tensors is not None:
        del donor_truth_tensors
    if device_donor_truth is not None:
        del device_donor_truth
    clear_memory(runtime.model)
    return value


def score_circuit_on_chunks(
    runtime: HeadModelRuntime,
    selected_sites: list[tuple[int, int]],
    chunks: list[list[dict[str, Any]]],
    *,
    sentence_score_mode: str,
    patch_scope: str,
    patch_first_n_tokens: int,
    activation_device: torch.device,
    target_baseline: float,
    unpatched_scores: list[float] | None = None,
    source_pair_fn: Any = None,
    source_pair_group_fn: Any = None,
    difference_source_pair_fn: Any = None,
    difference_source_pair_group_fn: Any = None,
    source_role: str = "truthful",
    target_role: str = "deceptive",
    label: str = "circuit",
) -> dict[str, Any]:
    if unpatched_scores is None:
        unpatched_scores = score_unpatched_chunks(
            runtime,
            chunks,
            sentence_score_mode=sentence_score_mode,
            target_role=target_role,
        )
    patched_total = 0.0
    unpatched_total = 0.0
    weight_total = 0
    for chunk_idx, chunk_pairs in enumerate(chunks):
        source_pairs = None if source_pair_fn is None else source_pair_fn(chunk_pairs)
        source_pair_groups = None if source_pair_group_fn is None else source_pair_group_fn(chunk_pairs)
        difference_source_pairs = None if difference_source_pair_fn is None else difference_source_pair_fn(chunk_pairs)
        difference_source_pair_groups = (
            None if difference_source_pair_group_fn is None else difference_source_pair_group_fn(chunk_pairs)
        )
        patched = patch_circuit_chunk(
            runtime,
            chunk_pairs,
            selected_sites,
            sentence_score_mode=sentence_score_mode,
            patch_scope=patch_scope,
            patch_first_n_tokens=patch_first_n_tokens,
            activation_device=activation_device,
            source_pairs=source_pairs,
            source_pair_groups=source_pair_groups,
            difference_source_pairs=difference_source_pairs,
            difference_source_pair_groups=difference_source_pair_groups,
            source_role=source_role,
            target_role=target_role,
        )
        unpatched = float(unpatched_scores[chunk_idx])
        weight = len(chunk_pairs)
        patched_total += patched * weight
        unpatched_total += unpatched * weight
        weight_total += weight
        print(f"{label} | chunk {chunk_idx + 1}/{len(chunks)} | b_prime={unpatched:.4f} | m={patched:.4f}")
    patched_metric = patched_total / weight_total
    unpatched_metric = unpatched_total / weight_total
    delta = patched_metric - float(target_baseline)
    percent_reduction = percent_probability_reduction_from_delta(delta)
    return {
        "unpatched_metric": unpatched_metric,
        "patched_metric": patched_metric,
        "source_role": source_role,
        "target_role": target_role,
        "target_baseline_metric": float(target_baseline),
        "delta": delta,
        "percent_probability_reduction": percent_reduction,
        "unpatched_target_sentence_score": -unpatched_metric,
        "patched_target_sentence_score": -patched_metric,
        "target_sentence_score_delta": (-patched_metric) - (-unpatched_metric),
        "unpatched_target_deceptive_score": -unpatched_metric,
        "patched_target_deceptive_score": -patched_metric,
        "target_deceptive_score_delta": (-patched_metric) - (-unpatched_metric),
        "n_chunks": len(chunks),
        "n_pairs": int(sum(len(chunk) for chunk in chunks)),
        "circuit_size": len(normalize_sites(selected_sites)),
    }


def top_ranked_sites(ranked_sites: list[tuple[int, int]], edge_count: int) -> list[tuple[int, int]]:
    edge_count = int(edge_count)
    if edge_count <= 0:
        raise ValueError("edge_count must be positive.")
    if edge_count > len(ranked_sites):
        raise ValueError(f"edge_count={edge_count} exceeds {len(ranked_sites)} ranked sites.")
    return ranked_sites[:edge_count]


def random_sites_for_size(all_sites: list[tuple[int, int]], edge_count: int, *, rng: np.random.Generator) -> list[tuple[int, int]]:
    random_site_indices = rng.choice(len(all_sites), size=int(edge_count), replace=False)
    return [all_sites[int(idx)] for idx in random_site_indices]


def layer_matched_random_sites(
    selected_sites: list[tuple[int, int]],
    *,
    n_heads: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    selected_sites = normalize_sites(selected_sites)
    selected_set = set(selected_sites)
    sampled_sites = []
    for layer_idx, heads in group_sites_by_layer(selected_sites).items():
        layer_candidates = [
            (int(layer_idx), int(head_idx))
            for head_idx in range(n_heads)
            if (int(layer_idx), int(head_idx)) not in selected_set
        ]
        if len(layer_candidates) < len(heads):
            layer_candidates = [(int(layer_idx), int(head_idx)) for head_idx in range(n_heads)]
        sampled_indices = rng.choice(len(layer_candidates), size=len(heads), replace=False)
        sampled_sites.extend(layer_candidates[int(idx)] for idx in sampled_indices)
    return sampled_sites


def candidate_circuit_sizes(ranked_site_count: int, *, explicit_sizes: list[int] | None = None) -> list[int]:
    sizes = explicit_sizes if explicit_sizes else list(DEFAULT_CANDIDATE_CIRCUIT_SIZES)
    return [int(k) for k in sizes if 0 < int(k) <= int(ranked_site_count)]


def evaluate_top_k_for_percent_reduction(
    runtime: HeadModelRuntime,
    ranked_sites: list[tuple[int, int]],
    edge_count: int,
    chunks: list[list[dict[str, Any]]],
    *,
    sentence_score_mode: str,
    patch_scope: str,
    patch_first_n_tokens: int,
    activation_device: torch.device,
    target_baseline: float,
    unpatched_scores: list[float],
    percent_reduction_threshold: float,
) -> dict[str, Any]:
    edge_count = int(edge_count)
    record = score_circuit_on_chunks(
        runtime,
        top_ranked_sites(ranked_sites, edge_count),
        chunks,
        sentence_score_mode=sentence_score_mode,
        patch_scope=patch_scope,
        patch_first_n_tokens=patch_first_n_tokens,
        activation_device=activation_device,
        target_baseline=target_baseline,
        unpatched_scores=unpatched_scores,
        label=f"top_{edge_count}",
    )
    record = {
        "candidate_edge_count": edge_count,
        "meets_threshold": bool(
            record["percent_probability_reduction"] >= float(percent_reduction_threshold)
            and record["delta"] > 0.0
        ),
        **record,
    }
    print(
        f"top_{edge_count} aggregate | delta={record['delta']:.4f} | "
        f"percent_reduction={record['percent_probability_reduction']:.2f}%"
    )
    return record


def choose_top_k_from_sweep(
    sweep_df: pd.DataFrame,
    *,
    flattening_percent_point_tol: float,
) -> tuple[int, dict[str, Any], str]:
    meeting_df = sweep_df[sweep_df["meets_threshold"]].copy()
    if not meeting_df.empty:
        best_row = meeting_df.sort_values("candidate_edge_count").iloc[0]
        return int(best_row["candidate_edge_count"]), best_row.to_dict(), "threshold_reached"

    ordered = sweep_df.sort_values("candidate_edge_count").reset_index(drop=True)
    for idx in range(len(ordered) - 1):
        current = float(ordered.loc[idx, "percent_probability_reduction"])
        nxt = float(ordered.loc[idx + 1, "percent_probability_reduction"])
        if current > 0.0 and (nxt - current) <= float(flattening_percent_point_tol):
            row = ordered.loc[idx]
            return int(row["candidate_edge_count"]), row.to_dict(), "threshold_not_reached_flattened"

    best_row = ordered.loc[ordered["percent_probability_reduction"].idxmax()]
    return int(best_row["candidate_edge_count"]), best_row.to_dict(), "threshold_not_reached_best_percent"


def sweep_top_k_percent_reduction_circuits(
    runtime: HeadModelRuntime,
    ranked_sites: list[tuple[int, int]],
    chunks: list[list[dict[str, Any]]],
    *,
    sentence_score_mode: str,
    patch_scope: str,
    patch_first_n_tokens: int,
    activation_device: torch.device,
    target_baseline: float,
    unpatched_scores: list[float],
    percent_reduction_threshold: float,
    candidate_sizes: list[int],
    flattening_percent_point_tol: float,
) -> tuple[int, dict[str, Any], pd.DataFrame, str]:
    records = []
    for edge_count in candidate_sizes:
        records.append(
            evaluate_top_k_for_percent_reduction(
                runtime,
                ranked_sites,
                edge_count,
                chunks,
                sentence_score_mode=sentence_score_mode,
                patch_scope=patch_scope,
                patch_first_n_tokens=patch_first_n_tokens,
                activation_device=activation_device,
                target_baseline=target_baseline,
                unpatched_scores=unpatched_scores,
                percent_reduction_threshold=percent_reduction_threshold,
            )
        )
    sweep_df = pd.DataFrame(records).reset_index(drop=True)
    discovered_edge_count, discovered_record, status = choose_top_k_from_sweep(
        sweep_df,
        flattening_percent_point_tol=flattening_percent_point_tol,
    )
    return discovered_edge_count, discovered_record, sweep_df, status


def score_selected_circuit_controls(
    runtime: HeadModelRuntime,
    selected_sites: list[tuple[int, int]],
    chunks: list[list[dict[str, Any]]],
    *,
    sentence_score_mode: str,
    patch_scope: str,
    patch_first_n_tokens: int,
    activation_device: torch.device,
    target_baseline: float,
    unpatched_scores: list[float],
    rng: np.random.Generator,
    donor_pool: list[dict[str, Any]],
    all_sites: list[tuple[int, int]],
    control_count: int,
    mean_shuffled_donor_count: int,
    label_prefix: str,
    n_heads: int,
) -> pd.DataFrame:
    records = []
    selected_sites = normalize_sites(selected_sites)
    control_specs = [
        "random",
        "layer_matched_random",
        "shuffled_truthful_donor",
        "shuffled_deceptive_donor",
        "averaged_shuffled_deceptive_donor",
        "shuffled_difference_donor",
        "averaged_shuffled_difference_donor",
    ]
    for control_kind in control_specs:
        for control_idx in range(int(control_count)):
            control_target_baseline = target_baseline
            control_unpatched_scores = unpatched_scores
            source_role = "truthful"
            target_role = "deceptive"
            source_pair_fn = None
            source_pair_group_fn = None
            difference_source_pair_fn = None
            difference_source_pair_group_fn = None

            if control_kind == "random":
                control_sites = random_sites_for_size(all_sites, len(selected_sites), rng=rng)
            elif control_kind == "layer_matched_random":
                control_sites = layer_matched_random_sites(selected_sites, n_heads=n_heads, rng=rng)
            elif control_kind == "shuffled_truthful_donor":
                control_sites = selected_sites
                source_pair_fn = lambda chunk_pairs, rng=rng: shuffled_donor_pairs_for_chunk(
                    chunk_pairs,
                    donor_pool,
                    rng=rng,
                )
            elif control_kind == "shuffled_deceptive_donor":
                control_sites = selected_sites
                source_role = "deceptive"
                source_pair_fn = lambda chunk_pairs, rng=rng: shuffled_donor_pairs_for_chunk(
                    chunk_pairs,
                    donor_pool,
                    rng=rng,
                    require_different_game=True,
                )
            elif control_kind == "averaged_shuffled_deceptive_donor":
                control_sites = selected_sites
                source_role = "deceptive"
                source_pair_group_fn = lambda chunk_pairs, rng=rng: mean_shuffled_donor_groups_for_chunk(
                    chunk_pairs,
                    donor_pool,
                    rng=rng,
                    donor_count=int(mean_shuffled_donor_count),
                    require_different_game=True,
                )
            elif control_kind == "shuffled_difference_donor":
                control_sites = selected_sites
                difference_source_pair_fn = lambda chunk_pairs, rng=rng: shuffled_donor_pairs_for_chunk(
                    chunk_pairs,
                    donor_pool,
                    rng=rng,
                    require_different_game=True,
                )
            elif control_kind == "averaged_shuffled_difference_donor":
                control_sites = selected_sites
                difference_source_pair_group_fn = (
                    lambda chunk_pairs, rng=rng: averaged_shuffled_difference_donor_groups_for_chunk(
                        chunk_pairs,
                        donor_pool,
                        rng=rng,
                        donor_count=int(mean_shuffled_donor_count),
                        require_different_game=True,
                    )
                )
            else:
                raise ValueError(f"Unsupported control_kind={control_kind!r}.")

            record = score_circuit_on_chunks(
                runtime,
                control_sites,
                chunks,
                sentence_score_mode=sentence_score_mode,
                patch_scope=patch_scope,
                patch_first_n_tokens=patch_first_n_tokens,
                activation_device=activation_device,
                target_baseline=control_target_baseline,
                unpatched_scores=control_unpatched_scores,
                source_pair_fn=source_pair_fn,
                source_pair_group_fn=source_pair_group_fn,
                difference_source_pair_fn=difference_source_pair_fn,
                difference_source_pair_group_fn=difference_source_pair_group_fn,
                source_role=source_role,
                target_role=target_role,
                label=f"{label_prefix}_{control_kind}_{control_idx:02d}",
            )
            records.append(
                {
                    "circuit_kind": control_kind,
                    "circuit_id": f"{label_prefix}_{control_kind}_{control_idx:02d}",
                    "control_index": int(control_idx),
                    "candidate_edge_count": len(selected_sites),
                    **record,
                }
            )
    return pd.DataFrame(records)


def count_alpha_words(text: Any) -> int:
    return len(re.findall(r"[A-Za-z]+", str(text)))


def usable_sentence(text: Any, *, min_sentence_alpha_words: int, exclude_multiline_sentences: bool) -> bool:
    clean = str(text).strip()
    if not clean:
        return False
    if exclude_multiline_sentences and "\n" in clean:
        return False
    if int(min_sentence_alpha_words) > 0 and count_alpha_words(clean) < int(min_sentence_alpha_words):
        return False
    return True


def choose_generic_truthful_donor(
    shared_entry: dict[str, Any],
    target_sentence: str,
    *,
    min_sentence_alpha_words: int,
    exclude_multiline_sentences: bool,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    target_norm = ap.normalize_sentence_for_compare(target_sentence)
    candidates = []
    for gen_idx, generation in enumerate(shared_entry.get("generations") or []):
        if generation.get("is_truthful") is not True:
            continue
        first_sentence, _ = ap.extract_first_sentence(generation.get("gen_text", ""))
        first_sentence = str(first_sentence).strip()
        if not usable_sentence(
            first_sentence,
            min_sentence_alpha_words=min_sentence_alpha_words,
            exclude_multiline_sentences=exclude_multiline_sentences,
        ):
            continue
        if ap.normalize_sentence_for_compare(first_sentence) == target_norm:
            continue
        candidates.append(
            {
                "gen_idx": int(gen_idx),
                "first_sentence": first_sentence,
                "prompt": generation.get("prompt", shared_entry.get("prompt", "")),
                "prefix_text": generation.get("prefix_text", shared_entry.get("prefix_text", "")),
                "full_generation_text": generation.get("full_generation_text", ""),
                "is_truthful": generation.get("is_truthful"),
                "deceptive": generation.get("deceptive"),
                "parse_error": generation.get("parse_error"),
                "evaluation": generation.get("evaluation"),
                "word_count": count_alpha_words(first_sentence),
            }
        )
    if not candidates:
        return None, []
    candidates = sorted(candidates, key=lambda row: (abs(row["word_count"] - 12), len(row["first_sentence"])))
    return candidates[0], candidates


def load_generic_commitment_pairs_for_env(
    *,
    dataset_root: Path,
    environment: str,
    dataset_model_id: str,
    pair_count: int,
    search_limit: int,
    min_commitment_delta: float,
    min_commitment_deception_rate: float,
    min_num_valid: int,
    min_sentence_alpha_words: int,
    exclude_multiline_sentences: bool,
) -> pd.DataFrame:
    localization_dir = dataset_root / environment / dataset_model_id / "localization"
    if not localization_dir.exists():
        print(f"Skipping {environment}: missing {localization_dir}")
        return pd.DataFrame()

    rows = []
    paths = sorted(localization_dir.glob("sentence_localization_*.json"))
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        history = payload.get("history") or []
        if len(history) < 2:
            continue
        example_id = str(payload.get("example_id", path.stem))
        for right_pos in range(1, len(history)):
            left_pos = right_pos - 1
            shared_entry = history[left_pos]
            target_entry = history[right_pos]
            try:
                shared_rate = float(shared_entry.get("deception_rate", float("nan")))
                target_rate = float(target_entry.get("deception_rate", float("nan")))
                commitment_delta = target_rate - shared_rate
            except Exception:
                continue
            if not math.isfinite(commitment_delta) or commitment_delta <= float(min_commitment_delta):
                continue
            if target_rate < float(min_commitment_deception_rate):
                continue
            shared_num_valid = int(shared_entry.get("num_valid") or 0)
            target_num_valid = int(target_entry.get("num_valid") or 0)
            if int(min_num_valid) > 0 and (
                shared_num_valid < int(min_num_valid) or target_num_valid < int(min_num_valid)
            ):
                continue
            deceptive_sentence = str(target_entry.get("sentence_text", "")).strip()
            if not usable_sentence(
                deceptive_sentence,
                min_sentence_alpha_words=min_sentence_alpha_words,
                exclude_multiline_sentences=exclude_multiline_sentences,
            ):
                continue
            donor_row, donor_candidates = choose_generic_truthful_donor(
                shared_entry,
                deceptive_sentence,
                min_sentence_alpha_words=min_sentence_alpha_words,
                exclude_multiline_sentences=exclude_multiline_sentences,
            )
            if donor_row is None:
                continue

            prompt = str(target_entry.get("prompt", payload.get("prompt", "")))
            shared_context_text = str(shared_entry.get("prefix_text", ""))
            truthful_sentence = str(donor_row["first_sentence"]).strip()
            if not prompt or not shared_context_text:
                continue

            rows.append(
                {
                    "pair_id": f"{environment}__{slugify(example_id)}__sent_{int(right_pos)}__donor_{int(donor_row['gen_idx'])}",
                    "localization_path": str(path),
                    "example_id": example_id,
                    "shared_context_sentence_pos": int(left_pos),
                    "commitment_sentence_pos": int(right_pos),
                    "shared_context_sentence_text": str(shared_entry.get("sentence_text", "")),
                    "deceptive_commitment_sentence": deceptive_sentence,
                    "truthful_donor_sentence": truthful_sentence,
                    "prompt": prompt,
                    "shared_context_text": shared_context_text,
                    "shared_context_deception_rate": shared_rate,
                    "deceptive_prefix_deception_rate": target_rate,
                    "commitment_deception_rate": target_rate,
                    "commitment_delta": commitment_delta,
                    "shared_context_num_valid": shared_num_valid,
                    "shared_context_num_truthful": shared_entry.get("num_truthful"),
                    "deceptive_prefix_num_valid": target_num_valid,
                    "deceptive_prefix_num_truthful": target_entry.get("num_truthful"),
                    "donor_generation_idx": int(donor_row["gen_idx"]),
                    "donor_full_generation_text": donor_row.get("full_generation_text", ""),
                    "donor_is_truthful": donor_row.get("is_truthful") is True,
                    "donor_deceptive": donor_row.get("deceptive"),
                    "donor_parse_error": donor_row.get("parse_error"),
                    "donor_evaluation": ap.to_json_safe(donor_row.get("evaluation")),
                    "donor_clarity_score": float(donor_row["word_count"]),
                    "n_truthful_donors": int(len(donor_candidates)),
                    "shared_prefix_text": prompt + shared_context_text,
                    "deceptive_branch_text": prompt + ap.append_continuation(shared_context_text, deceptive_sentence),
                    "truthful_branch_text": prompt + ap.append_continuation(shared_context_text, truthful_sentence),
                }
            )
            if len(rows) > max(int(search_limit) * 4, int(search_limit) + 100):
                rows = sorted(
                    rows,
                    key=lambda row: (
                        row["commitment_delta"],
                        row["deceptive_prefix_deception_rate"],
                        row["n_truthful_donors"],
                        row["donor_clarity_score"],
                    ),
                    reverse=True,
                )[: int(search_limit)]
    if not rows:
        return pd.DataFrame()
    rows = sorted(
        rows,
        key=lambda row: (
            row["commitment_delta"],
            row["deceptive_prefix_deception_rate"],
            row["n_truthful_donors"],
            row["donor_clarity_score"],
        ),
        reverse=True,
    )[: int(pair_count)]
    env_df = pd.DataFrame(rows).reset_index(drop=True)
    env_df.insert(0, "pair_index", range(len(env_df)))
    return env_df


def compute_head_steering_vectors(
    runtime: HeadModelRuntime,
    chunks: list[list[dict[str, Any]]],
    selected_sites: list[tuple[int, int]],
) -> tuple[dict[tuple[int, int], torch.Tensor], pd.DataFrame]:
    grouped = group_sites_by_layer(selected_sites)
    sums = {
        (layer_idx, head_idx): torch.zeros(runtime.head_dim, dtype=torch.float32)
        for layer_idx, heads in grouped.items()
        for head_idx in heads
    }
    counts = {site: 0 for site in sums}

    for layer_idx, heads in grouped.items():
        print(f"Computing steering directions for layer {layer_idx:02d} ({len(heads)} head(s))")
        for chunk_pairs in chunks:
            source_batch, target_batch, source_inputs, target_inputs = build_source_target_batches(
                runtime.tokenizer,
                chunk_pairs,
                chunk_pairs,
            )
            with torch.inference_mode():
                with runtime.model.trace(source_inputs):
                    source_proxy = attn_out_input(runtime.layers[layer_idx]).save()
            source_acts = saved_value(source_proxy).detach().float().cpu()
            del source_proxy
            clear_memory(runtime.model)

            with torch.inference_mode():
                with runtime.model.trace(target_inputs):
                    target_proxy = attn_out_input(runtime.layers[layer_idx]).save()
            target_acts = saved_value(target_proxy).detach().float().cpu()

            for local_pair_idx, _pair in enumerate(chunk_pairs):
                source_row = source_batch["rows"][local_pair_idx]
                target_row = target_batch["rows"][local_pair_idx]
                source_start, target_start, span_len = paired_commitment_patch_span(
                    source_row,
                    target_row,
                    patch_scope="commitment_full",
                    patch_first_n_tokens=0,
                )
                if span_len <= 0:
                    continue
                source_slice = slice(int(source_start), int(source_start + span_len))
                target_slice = slice(int(target_start), int(target_start + span_len))
                for head_idx in heads:
                    start = int(head_idx) * runtime.head_dim
                    stop = start + runtime.head_dim
                    source_vec = source_acts[local_pair_idx, source_slice, start:stop].mean(dim=0)
                    target_vec = target_acts[local_pair_idx, target_slice, start:stop].mean(dim=0)
                    site = (int(layer_idx), int(head_idx))
                    sums[site] += source_vec - target_vec
                    counts[site] += 1

            del target_proxy, source_acts, target_acts, source_batch, target_batch, source_inputs, target_inputs
            clear_memory(runtime.model)

    vectors = {site: sums[site] / max(int(counts[site]), 1) for site in sums}
    vector_df = pd.DataFrame(
        [
            {
                "layer": int(layer_idx),
                "head": int(head_idx),
                "n_pairs": int(counts[(layer_idx, head_idx)]),
                "direction_norm": float(vector.norm().item()),
                "direction_mean_abs": float(vector.abs().mean().item()),
                "direction_max_abs": float(vector.abs().max().item()) if int(vector.numel()) > 0 else 0.0,
                "vector_length": int(vector.numel()),
                "all_finite": bool(torch.isfinite(vector).all().item()),
                "nan_count": int(torch.isnan(vector).sum().item()),
                "inf_count": int(torch.isinf(vector).sum().item()),
            }
            for (layer_idx, head_idx), vector in vectors.items()
        ]
    ).sort_values(["layer", "head"]).reset_index(drop=True)
    return vectors, vector_df


def validate_steering_vectors(
    vectors: dict[tuple[int, int], torch.Tensor],
    *,
    expected_sites: list[tuple[int, int]] | None = None,
    expected_head_dim: int | None = None,
    context: str,
) -> pd.DataFrame:
    if not vectors:
        raise ValueError(f"{context}: no steering vectors were provided.")

    rows = []
    for (layer_idx, head_idx), vector in sorted(vectors.items()):
        tensor = vector.detach().float().cpu().reshape(-1)
        rows.append(
            {
                "layer": int(layer_idx),
                "head": int(head_idx),
                "vector_length": int(tensor.numel()),
                "direction_norm": float(tensor.norm().item()) if int(tensor.numel()) > 0 else 0.0,
                "direction_mean_abs": float(tensor.abs().mean().item()) if int(tensor.numel()) > 0 else 0.0,
                "direction_max_abs": float(tensor.abs().max().item()) if int(tensor.numel()) > 0 else 0.0,
                "all_finite": bool(torch.isfinite(tensor).all().item()),
                "nan_count": int(torch.isnan(tensor).sum().item()),
                "inf_count": int(torch.isinf(tensor).sum().item()),
            }
        )
    diagnostics_df = pd.DataFrame(rows)

    vector_sites = set((int(layer_idx), int(head_idx)) for layer_idx, head_idx in vectors)
    if expected_sites is not None:
        expected_site_set = {
            (int(layer_idx), int(head_idx))
            for layer_idx, head_idx in expected_sites
        }
        missing_sites = sorted(expected_site_set - vector_sites)
        extra_sites = sorted(vector_sites - expected_site_set)
        if missing_sites or extra_sites:
            raise ValueError(
                f"{context}: steering-vector sites do not match expected discovered sites. "
                f"missing={missing_sites[:8]} extra={extra_sites[:8]}"
            )

    if expected_head_dim is not None and int(expected_head_dim) > 0:
        bad_shape_df = diagnostics_df.loc[
            pd.to_numeric(diagnostics_df["vector_length"], errors="coerce").ne(int(expected_head_dim))
        ].copy()
        if not bad_shape_df.empty:
            raise ValueError(
                f"{context}: some steering vectors do not match head_dim={int(expected_head_dim)}. "
                f"bad_sites={bad_shape_df[['layer', 'head', 'vector_length']].to_dict(orient='records')[:8]}"
            )

    nonfinite_df = diagnostics_df.loc[~diagnostics_df["all_finite"].astype(bool)].copy()
    if not nonfinite_df.empty:
        raise ValueError(
            f"{context}: found non-finite steering vectors. "
            f"bad_sites={nonfinite_df[['layer', 'head', 'nan_count', 'inf_count']].to_dict(orient='records')[:8]}"
        )

    zero_norm_df = diagnostics_df.loc[
        pd.to_numeric(diagnostics_df["direction_norm"], errors="coerce") <= 0.0
    ].copy()
    if not zero_norm_df.empty:
        raise ValueError(
            f"{context}: found zero-norm steering vectors. "
            f"bad_sites={zero_norm_df[['layer', 'head', 'direction_norm']].to_dict(orient='records')[:8]}"
        )

    return diagnostics_df.sort_values(["layer", "head"]).reset_index(drop=True)


def load_analysis_bundle(analysis_dir: Path) -> dict[str, Any]:
    analysis_dir = Path(analysis_dir).expanduser().resolve()
    metadata = read_json(analysis_dir / "metadata.json")
    config = read_json(analysis_dir / "runner_config.json")
    split_pair_rows = ap.read_jsonl_rows(analysis_dir / "tables" / "split_pairs.jsonl")
    split_pairs_df = pd.DataFrame(split_pair_rows)
    discovered_sites = [tuple(site) for site in metadata.get("discovered_sites", [])]
    default_vector_path = analysis_dir / "steering_vectors.pt"
    vector_metadata_path = analysis_dir / "vector_metadata.json"
    return {
        "analysis_dir": analysis_dir,
        "metadata": metadata,
        "config": config,
        "split_pair_rows": split_pair_rows,
        "split_pairs_df": split_pairs_df,
        "discovered_sites": discovered_sites,
        "default_vector_path": default_vector_path,
        "vector_metadata": read_json(vector_metadata_path) if vector_metadata_path.exists() else {},
    }


def filter_rows_by_split(
    rows: list[dict[str, Any]],
    *,
    split_names: list[str],
) -> list[dict[str, Any]]:
    if not split_names or "all" in split_names:
        return list(rows)
    wanted = {str(name) for name in split_names}
    return [row for row in rows if str(row.get("split")) in wanted]


def save_vector_bundle(
    output_dir: Path,
    *,
    analysis_dir: Path,
    selected_splits: list[str],
    model_id: str,
    model_name_or_path: str,
    dtype_name: str,
    dataset_model_id: str | None,
    environment: str | None,
    discovered_sites: list[tuple[int, int]],
    prepared_split: PreparedSplit,
    vectors: dict[tuple[int, int], torch.Tensor],
    steering_vector_df: pd.DataFrame,
    batch_pair_count: int,
    max_input_tokens: int | None,
    extra_table_path: Path | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_df = validate_steering_vectors(
        vectors,
        expected_sites=discovered_sites,
        context=f"steering vector bundle save for {output_dir}",
    )
    vector_payload = {
        "metadata": {
            "analysis_dir": str(Path(analysis_dir).expanduser().resolve()),
            "vector_source_splits": list(selected_splits),
            "model_id": model_id,
            "model_name_or_path": model_name_or_path,
            "dtype_name": dtype_name,
            "dataset_model_id": dataset_model_id,
            "environment": environment,
            "discovered_sites": discovered_sites,
            "n_pairs": int(prepared_split.total_pairs),
        },
        "vectors": {
            f"layer_{int(layer_idx):02d}_head_{int(head_idx):02d}": value.detach().cpu()
            for (layer_idx, head_idx), value in vectors.items()
        },
    }
    torch.save(vector_payload, output_dir / "steering_vectors.pt")
    save_dataframe(steering_vector_df, output_dir / "steering_vector_df.csv")
    if extra_table_path is not None:
        save_dataframe(steering_vector_df, extra_table_path)
    vector_metadata = {
        "analysis_dir": str(Path(analysis_dir).expanduser().resolve()),
        "vector_output_dir": str(output_dir),
        "vector_path": str(output_dir / "steering_vectors.pt"),
        "vector_source_splits": list(selected_splits),
        "model_id": model_id,
        "model_name_or_path": model_name_or_path,
        "dtype_name": dtype_name,
        "dataset_model_id": dataset_model_id,
        "environment": environment,
        "discovered_sites": discovered_sites,
        "n_pairs": int(prepared_split.total_pairs),
        "batch_pair_count": int(batch_pair_count),
        "max_input_tokens": max_input_tokens,
        "site_count": int(len(vectors)),
        "vector_norm_summary": {
            "min": float(pd.to_numeric(diagnostics_df["direction_norm"], errors="coerce").min()),
            "max": float(pd.to_numeric(diagnostics_df["direction_norm"], errors="coerce").max()),
            "mean": float(pd.to_numeric(diagnostics_df["direction_norm"], errors="coerce").mean()),
            "median": float(pd.to_numeric(diagnostics_df["direction_norm"], errors="coerce").median()),
        },
        "all_finite": bool(diagnostics_df["all_finite"].astype(bool).all()),
        "zero_norm_count": int(
            pd.to_numeric(diagnostics_df["direction_norm"], errors="coerce").le(0.0).sum()
        ),
    }
    write_json(output_dir / "vector_metadata.json", vector_metadata)
    return vector_metadata


def build_vector_output_dir(analysis_dir: Path, output_dir: str, tag: str) -> Path:
    if str(output_dir).strip():
        out_dir = Path(output_dir).expanduser().resolve()
    else:
        out_dir = analysis_dir / "steering_vectors" / slugify(tag or datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def build_steering_output_dir(
    vector_path: Path,
    output_dir: str,
    tag: str,
    *,
    num_shards: int,
    shard_index: int,
) -> Path:
    if str(output_dir).strip():
        base_out_dir = Path(output_dir).expanduser().resolve()
    else:
        base_out_dir = vector_path.parent / "steering" / slugify(tag or datetime.now().strftime("%Y%m%d_%H%M%S"))
    if int(num_shards) > 1:
        out_dir = base_out_dir / f"shard_{int(shard_index):03d}_of_{int(num_shards):03d}"
    else:
        out_dir = base_out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def build_generation_output_dir(vector_path: Path, output_dir: str, tag: str) -> Path:
    return build_steering_output_dir(
        vector_path,
        output_dir,
        tag,
        num_shards=1,
        shard_index=0,
    )


def load_vector_bundle(path: Path) -> tuple[dict[tuple[int, int], torch.Tensor], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "vectors" in payload:
        metadata = dict(payload.get("metadata") or {})
        raw_vectors = payload["vectors"]
    else:
        metadata = {}
        raw_vectors = payload
    vectors: dict[tuple[int, int], torch.Tensor] = {}
    for key, value in dict(raw_vectors).items():
        if isinstance(key, tuple) and len(key) == 2:
            site = (int(key[0]), int(key[1]))
        else:
            match = re.match(r"layer_(\d+)_head_(\d+)", str(key))
            if not match:
                raise ValueError(f"Unrecognized steering vector key: {key!r}")
            site = (int(match.group(1)), int(match.group(2)))
        vectors[site] = value.detach().float().cpu()
    return vectors, metadata


def generation_environment_for_job(job: dict[str, Any]) -> str:
    return steering_utils.canonicalize_environment_name(job.get("environment", "bs"))


def generation_eval_context_for_job(job: dict[str, Any]) -> dict[str, Any]:
    raw_context = job.get("eval_context")
    if isinstance(raw_context, str) and raw_context.strip():
        try:
            raw_context = json.loads(raw_context)
        except Exception:
            raw_context = {}
    context = dict(raw_context or {})
    if generation_environment_for_job(job) == "bs" and context.get("truthful_rank") is None:
        required_rank = job.get("required_rank")
        if required_rank is not None:
            context["truthful_rank"] = int(required_rank)
    return context


def evaluate_generation_for_job(
    output_text: str,
    *,
    job: dict[str, Any],
    model_name_or_path: str | None,
) -> dict[str, Any]:
    return steering_utils.evaluate_generation_generic(
        output_text,
        game=generation_environment_for_job(job),
        eval_context=generation_eval_context_for_job(job),
        model_name_or_path=model_name_or_path,
    )


def _sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
    generator: torch.Generator,
) -> torch.Tensor:
    if float(temperature) <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    logits = logits / float(temperature)
    if 0 < float(top_p) < 1:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        remove_mask = cumulative_probs > float(top_p)
        remove_mask[..., 1:] = remove_mask[..., :-1].clone()
        remove_mask[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove_mask, float("-inf"))
        filtered_logits = torch.full_like(logits, float("-inf"))
        logits = filtered_logits.scatter(dim=-1, index=sorted_indices, src=sorted_logits)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)


def _make_finished_decode_token(tokenizer: Any) -> int:
    if tokenizer.eos_token_id is not None:
        return int(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        return int(tokenizer.pad_token_id)
    return 0


def _build_generation_jobs(
    prompt_rows: list[dict[str, Any]],
    *,
    samples_per_prompt: int,
    base_seed: int,
) -> list[dict[str, Any]]:
    jobs = []
    for row_idx, prompt_row in enumerate(prompt_rows):
        row_id = str(prompt_row.get("pair_id") or prompt_row.get("example_id") or prompt_row.get("prompt_id") or row_idx)
        for sample_idx in range(int(samples_per_prompt)):
            jobs.append(
                {
                    **prompt_row,
                    "sample_idx": int(sample_idx),
                    "seed": deterministic_seed(base_seed, row_id, sample_idx),
                }
            )
    return jobs


def _prepare_prompt_rows_from_analysis(
    analysis_bundle: dict[str, Any],
    *,
    prompt_splits: list[str],
    pair_indices: list[int],
    max_prompts: int,
) -> list[dict[str, Any]]:
    source_environment = str(analysis_bundle["metadata"].get("environment", "bs"))
    rows = filter_rows_by_split(analysis_bundle["split_pair_rows"], split_names=prompt_splits)
    if pair_indices:
        wanted = {int(value) for value in pair_indices}
        rows = [row for row in rows if int(row.get("pair_index", -1)) in wanted]
    rows = [
        {
            "pair_index": int(row["pair_index"]),
            "pair_id": str(row.get("pair_id", row["pair_index"])),
            "example_id": str(row.get("example_id", "")),
            "environment": source_environment,
            "split": str(row.get("split", "")),
            "localization_path": str(row.get("localization_path", "")),
            "prompt_text": str(row.get("shared_prefix_text", "")),
            "required_rank": int(row["required_rank"]) if row.get("required_rank") is not None else None,
            "eval_context": (
                {"truthful_rank": int(row["required_rank"])}
                if row.get("required_rank") is not None
                else {}
            ),
            "shared_context_sentence_pos": (
                int(row["shared_context_sentence_pos"])
                if row.get("shared_context_sentence_pos") is not None
                else None
            ),
            "commitment_sentence_pos": (
                int(row["commitment_sentence_pos"])
                if row.get("commitment_sentence_pos") is not None
                else None
            ),
            "deceptive_commitment_sentence": str(row.get("deceptive_commitment_sentence", "")),
            "truthful_donor_sentence": str(row.get("truthful_donor_sentence", "")),
            "shared_context_deception_rate": (
                float(row["shared_context_deception_rate"])
                if row.get("shared_context_deception_rate") is not None
                else float("nan")
            ),
            "commitment_deception_rate": (
                float(row["commitment_deception_rate"])
                if row.get("commitment_deception_rate") is not None
                else float("nan")
            ),
            "commitment_delta": (
                float(row["commitment_delta"])
                if row.get("commitment_delta") is not None
                else float("nan")
            ),
        }
        for row in rows
        if str(row.get("shared_prefix_text", "")).strip()
    ]
    rows = sorted(
        rows,
        key=lambda row: (
            float(row.get("shared_context_deception_rate", float("-inf"))),
            float(row.get("commitment_delta", float("-inf"))),
            -int(row["pair_index"]),
        ),
        reverse=True,
    )
    if int(max_prompts) > 0:
        rows = rows[: int(max_prompts)]
    return rows


def _prepare_prompt_rows_from_environments(
    analysis_bundle: dict[str, Any] | None,
    *,
    prompt_environments: list[str],
    prompts_per_environment: int,
    prompt_splits: list[str],
    dataset_root: Path,
    dataset_model_id: str,
    pair_cache_path: Path,
    refresh_pair_cache: bool,
    min_commitment_delta: float,
    disable_tqdm: bool,
) -> list[dict[str, Any]]:
    prompts_per_environment = int(prompts_per_environment)
    if prompts_per_environment <= 0:
        raise ValueError("--prompts-per-environment must be positive.")

    source_environment = (
        str(analysis_bundle["metadata"].get("environment", "bs"))
        if analysis_bundle is not None
        else "bs"
    )
    requested_envs = [
        steering_utils.canonicalize_environment_name(environment)
        for environment in prompt_environments
    ]
    rows: list[dict[str, Any]] = []

    if source_environment in requested_envs:
        if analysis_bundle is None:
            raise ValueError("--prompt-environments including the source environment requires --analysis-dir.")
        source_rows = _prepare_prompt_rows_from_analysis(
            analysis_bundle,
            prompt_splits=prompt_splits or ["test"],
            pair_indices=[],
            max_prompts=0,
        )
        rows.extend(source_rows[:prompts_per_environment])

    ood_envs = [environment for environment in requested_envs if environment != source_environment]
    if ood_envs:
        pair_df = steering_utils.load_or_build_bidirectional_pair_cache(
            dataset_root=dataset_root,
            environments=ood_envs,
            model_tail=dataset_model_id,
            pair_cache_path=pair_cache_path,
            refresh_cache=refresh_pair_cache,
            min_commitment_delta=float(min_commitment_delta),
            disable_tqdm=disable_tqdm,
        )
        for environment in ood_envs:
            env_df = pair_df.loc[pair_df["environment"].astype(str) == str(environment)].copy()
            if env_df.empty:
                continue
            env_df = env_df.sort_values(
                ["prefix_deception_rate", "delta_d", "pair_id"],
                ascending=[False, False, True],
            ).head(prompts_per_environment)
            for row in env_df.to_dict(orient="records"):
                eval_context = row.get("eval_context") if isinstance(row.get("eval_context"), dict) else {}
                rows.append(
                    {
                        "pair_index": None,
                        "pair_id": str(row.get("pair_id", "")),
                        "example_id": str(row.get("example_id", "")),
                        "environment": str(environment),
                        "split": "ood_test",
                        "localization_path": str(row.get("localization_path", "")),
                        "prompt_text": str(row.get("prompt", "")) + str(row.get("prefix_text", "")),
                        "required_rank": (
                            int(eval_context["truthful_rank"])
                            if steering_utils.canonicalize_environment_name(environment) == "bs"
                            and eval_context.get("truthful_rank") is not None
                            else None
                        ),
                        "eval_context": eval_context,
                        "shared_context_sentence_pos": (
                            int(row["prefix_sentence_idx_inclusive"])
                            if row.get("prefix_sentence_idx_inclusive") is not None
                            else None
                        ),
                        "commitment_sentence_pos": (
                            int(row["sentence_idx_inclusive"])
                            if row.get("sentence_idx_inclusive") is not None
                            else None
                        ),
                        "deceptive_commitment_sentence": str(row.get("deceptive_sentence", "")),
                        "truthful_donor_sentence": str(row.get("truthful_sentence", "")),
                        "shared_context_deception_rate": float(row.get("prefix_deception_rate", float("nan"))),
                        "commitment_deception_rate": float(row.get("deceptive_deception_rate", float("nan"))),
                        "commitment_delta": float(row.get("delta_d", float("nan"))),
                    }
                )

    return rows


def _prepare_prompt_rows_from_jsonl(prompt_jsonl: Path, *, max_prompts: int) -> list[dict[str, Any]]:
    rows = ap.read_jsonl_rows(prompt_jsonl)
    prompt_rows = []
    for idx, row in enumerate(rows):
        prompt_text = str(row.get("prompt_text", "")).strip()
        if not prompt_text:
            continue
        prompt_rows.append(
            {
                "prompt_id": str(row.get("prompt_id", idx)),
                "pair_id": str(row.get("pair_id", row.get("prompt_id", idx))),
                "example_id": str(row.get("example_id", row.get("prompt_id", idx))),
                "environment": str(row.get("environment", "bs")),
                "split": str(row.get("split", "")),
                "localization_path": str(row.get("localization_path", "")),
                "prompt_text": prompt_text,
                "required_rank": int(row["required_rank"]) if row.get("required_rank") is not None else None,
                "eval_context": row.get("eval_context") if isinstance(row.get("eval_context"), dict) else {},
                "shared_context_deception_rate": (
                    float(row["shared_context_deception_rate"])
                    if row.get("shared_context_deception_rate") is not None
                    else float("nan")
                ),
                "commitment_deception_rate": (
                    float(row["commitment_deception_rate"])
                    if row.get("commitment_deception_rate") is not None
                    else float("nan")
                ),
                "commitment_delta": (
                    float(row["commitment_delta"])
                    if row.get("commitment_delta") is not None
                    else float("nan")
                ),
            }
        )
    if int(max_prompts) > 0:
        prompt_rows = prompt_rows[: int(max_prompts)]
    return prompt_rows


def _prepare_single_prompt_row(prompt_text: str, *, required_rank: int | None) -> list[dict[str, Any]]:
    if not str(prompt_text).strip():
        raise ValueError("--prompt-text must be non-empty.")
    return [
        {
            "prompt_id": "single_prompt",
            "pair_id": "single_prompt",
            "example_id": "single_prompt",
            "environment": "bs",
            "split": "",
            "prompt_text": str(prompt_text),
            "required_rank": None if required_rank is None else int(required_rank),
            "eval_context": (
                {"truthful_rank": int(required_rank)}
                if required_rank is not None
                else {}
            ),
        }
    ]


def default_prompt_environments_from_analysis(
    analysis_bundle: dict[str, Any] | None,
    vector_metadata: dict[str, Any],
) -> list[str]:
    ordered: list[str] = []
    for value in [
        analysis_bundle["metadata"].get("environment") if analysis_bundle is not None else None,
        vector_metadata.get("environment"),
    ]:
        if str(value or "").strip():
            ordered.append(steering_utils.canonicalize_environment_name(str(value)))
    source_bundle = analysis_bundle["metadata"] if analysis_bundle is not None else {}
    for value in source_bundle.get("cross_corpus_envs", []) or []:
        if str(value or "").strip():
            ordered.append(steering_utils.canonicalize_environment_name(str(value)))
    deduped: list[str] = []
    seen: set[str] = set()
    for environment in ordered:
        if environment not in seen:
            deduped.append(environment)
            seen.add(environment)
    return deduped


def filter_prompt_rows_by_prefix_deception_rate(
    prompt_rows: list[dict[str, Any]],
    *,
    min_prefix_deception_rate: float,
) -> list[dict[str, Any]]:
    if float(min_prefix_deception_rate) <= 0.0:
        return list(prompt_rows)
    kept = []
    for row in prompt_rows:
        try:
            rate = float(row.get("shared_context_deception_rate", float("nan")))
        except Exception:
            rate = float("nan")
        if math.isfinite(rate) and rate >= float(min_prefix_deception_rate):
            kept.append(row)
    return kept


def shard_rows(rows: list[dict[str, Any]], *, num_shards: int, shard_index: int) -> list[dict[str, Any]]:
    if int(num_shards) <= 1:
        return rows
    if not (0 <= int(shard_index) < int(num_shards)):
        raise ValueError("--shard-index must satisfy 0 <= shard_index < num_shards.")
    return [row for idx, row in enumerate(rows) if idx % int(num_shards) == int(shard_index)]


def generate_batch_with_head_steering(
    runtime: GenerationModelRuntime,
    jobs: list[dict[str, Any]],
    *,
    steering_vectors: dict[tuple[int, int], torch.Tensor],
    alpha: float,
    steering_position: str,
    steering_max_decode_tokens: int,
    max_model_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    early_stop_on_valid_json: bool,
    early_stop_check_interval: int,
    early_stop_min_new_tokens: int,
) -> list[dict[str, Any]]:
    if not jobs:
        return []
    if steering_position not in {"last", "all"}:
        raise ValueError("--steering-position must be 'last' or 'all'.")

    device = ap.resolve_model_device(runtime.model)
    prompt_texts = [str(job["prompt_text"]) for job in jobs]
    encoded = runtime.tokenizer(
        prompt_texts,
        add_special_tokens=False,
        padding=True,
        truncation=bool(int(max_model_length) > 0),
        max_length=int(max_model_length) if int(max_model_length) > 0 else None,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    batch_size = len(jobs)
    attention_mask = encoded["attention_mask"]
    prompt_input_ids_by_row = [encoded["input_ids"][row_idx, attention_mask[row_idx].bool()].detach().cpu() for row_idx in range(batch_size)]
    grouped_vectors = group_sites_by_layer(steering_vectors.keys())
    hook_state: dict[str, Any] = {
        "phase": "prefill",
        "attention_mask": attention_mask,
        "active_mask": torch.ones(batch_size, dtype=torch.bool, device=device),
        # This budget is intended to count decode-time steering applications only.
        # Prefill steering over the prompt should not consume it.
        "steering_steps_remaining": (
            float("inf") if int(steering_max_decode_tokens) <= 0 else int(steering_max_decode_tokens)
        ),
    }
    handles = []

    for layer_idx, heads in grouped_vectors.items():
        module = attn_out_proj_module(runtime.layers[layer_idx])
        site_vectors = {(int(layer_idx), int(head_idx)): steering_vectors[(int(layer_idx), int(head_idx))] for head_idx in heads}

        def pre_hook(module: Any, args: tuple[Any, ...], site_vectors: dict[tuple[int, int], torch.Tensor] = site_vectors):
            if not args or not torch.is_tensor(args[0]):
                return None
            hidden = args[0]
            patched = hidden.clone()
            active_rows = hook_state["active_mask"].nonzero(as_tuple=False).flatten().tolist()
            if not active_rows:
                return None
            if float(hook_state["steering_steps_remaining"]) <= 0:
                return None

            if hook_state["phase"] == "prefill":
                if steering_position == "last":
                    for site, vector in site_vectors.items():
                        head_idx = int(site[1])
                        start = head_idx * runtime.head_dim
                        stop = start + runtime.head_dim
                        delta = (float(alpha) * vector).to(device=hidden.device, dtype=hidden.dtype).view(1, 1, runtime.head_dim)
                        patched[active_rows, -1:, start:stop] = patched[active_rows, -1:, start:stop] + delta
                else:
                    mask = hook_state["attention_mask"].to(device=hidden.device)
                    for row_idx in active_rows:
                        pos_mask = mask[row_idx].bool()
                        if not bool(pos_mask.any()):
                            continue
                        for site, vector in site_vectors.items():
                            head_idx = int(site[1])
                            start = head_idx * runtime.head_dim
                            stop = start + runtime.head_dim
                            delta = (float(alpha) * vector).to(device=hidden.device, dtype=hidden.dtype)
                            patched[row_idx, pos_mask, start:stop] = patched[row_idx, pos_mask, start:stop] + delta
            elif hook_state["phase"] == "decode" and int(hidden.shape[1]) == 1:
                for site, vector in site_vectors.items():
                    head_idx = int(site[1])
                    start = head_idx * runtime.head_dim
                    stop = start + runtime.head_dim
                    delta = (float(alpha) * vector).to(device=hidden.device, dtype=hidden.dtype).view(1, 1, runtime.head_dim)
                    patched[active_rows, -1:, start:stop] = patched[active_rows, -1:, start:stop] + delta
            else:
                return None

            new_args = list(args)
            new_args[0] = patched
            return tuple(new_args)

        handles.append(module.register_forward_pre_hook(pre_hook))

    with torch.no_grad():
        outputs = runtime.model(**encoded, use_cache=True, return_dict=True)
    hook_state["phase"] = "decode"

    past_key_values = outputs.past_key_values
    generator_device = device if device.type != "cpu" else torch.device("cpu")
    generators = [torch.Generator(device=generator_device).manual_seed(int(job["seed"])) for job in jobs]
    finished_token_id = _make_finished_decode_token(runtime.tokenizer)
    generated_token_ids_by_row: list[list[int]] = [[] for _ in range(batch_size)]
    ended_with_eos_by_row = [False for _ in range(batch_size)]
    valid_stopped_by_row = [False for _ in range(batch_size)]

    def sample_next_tokens(logits: torch.Tensor) -> torch.Tensor:
        next_tokens: list[torch.Tensor] = []
        for row_idx in range(batch_size):
            if (
                ended_with_eos_by_row[row_idx]
                or valid_stopped_by_row[row_idx]
                or len(generated_token_ids_by_row[row_idx]) >= int(max_new_tokens)
            ):
                token = torch.tensor([[finished_token_id]], dtype=encoded["input_ids"].dtype, device=device)
                next_tokens.append(token)
                continue

            token = _sample_next_token(
                logits[row_idx : row_idx + 1],
                temperature=float(temperature),
                top_p=float(top_p),
                generator=generators[row_idx],
            ).to(device=device)
            token_id = int(token.item())
            generated_token_ids_by_row[row_idx].append(token_id)
            ended_with_eos_by_row[row_idx] = (
                runtime.tokenizer.eos_token_id is not None and token_id == int(runtime.tokenizer.eos_token_id)
            )
            if early_stop_on_valid_json and not ended_with_eos_by_row[row_idx]:
                required_rank = jobs[row_idx].get("required_rank")
                n_tokens = len(generated_token_ids_by_row[row_idx])
                interval = max(int(early_stop_check_interval), 1)
                if (
                    n_tokens >= int(early_stop_min_new_tokens)
                    and (interval <= 1 or n_tokens % interval == 0)
                ):
                    evaluation = evaluate_generation_for_job(
                        runtime.tokenizer.decode(generated_token_ids_by_row[row_idx], skip_special_tokens=True),
                        job=jobs[row_idx],
                        model_name_or_path=runtime.model_name_or_path,
                    )
                    valid_stopped_by_row[row_idx] = bool(evaluation.get("is_valid") is True)
            next_tokens.append(token)
        hook_state["active_mask"] = torch.tensor(
            [
                not ended_with_eos_by_row[row_idx]
                and not valid_stopped_by_row[row_idx]
                and len(generated_token_ids_by_row[row_idx]) < int(max_new_tokens)
                for row_idx in range(batch_size)
            ],
            dtype=torch.bool,
            device=device,
        )
        return torch.cat(next_tokens, dim=0)

    next_input_ids = sample_next_tokens(outputs.logits[:, -1, :])
    try:
        while not all(
            ended_with_eos_by_row[row_idx]
            or valid_stopped_by_row[row_idx]
            or len(generated_token_ids_by_row[row_idx]) >= int(max_new_tokens)
            for row_idx in range(batch_size)
        ):
            with torch.no_grad():
                step_outputs = runtime.model(
                    input_ids=next_input_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
            if np.isfinite(float(hook_state["steering_steps_remaining"])) and hook_state["steering_steps_remaining"] > 0:
                hook_state["steering_steps_remaining"] -= 1
            past_key_values = step_outputs.past_key_values
            next_input_ids = sample_next_tokens(step_outputs.logits[:, -1, :])
    finally:
        for handle in handles:
            handle.remove()

    rows = []
    for row_idx, job in enumerate(jobs):
        prompt_ids = prompt_input_ids_by_row[row_idx]
        ids = generated_token_ids_by_row[row_idx]
        new_ids = torch.tensor(ids, dtype=prompt_ids.dtype)
        full_ids = torch.cat([prompt_ids, new_ids], dim=0) if len(ids) > 0 else prompt_ids.clone()
        generated_text = runtime.tokenizer.decode(ids, skip_special_tokens=True)
        evaluation = evaluate_generation_for_job(
            generated_text,
            job=job,
            model_name_or_path=runtime.model_name_or_path,
        )
        rows.append(
            {
                **{key: value for key, value in job.items() if key != "prompt_text"},
                "prompt_text": str(job["prompt_text"]),
                "generated_text": generated_text,
                "full_text": runtime.tokenizer.decode(full_ids, skip_special_tokens=True),
                "prompt_token_count": int(prompt_ids.shape[0]),
                "n_new_tokens": int(len(ids)),
                "ended_with_eos": bool(ended_with_eos_by_row[row_idx]),
                "early_stopped_on_valid_json": bool(valid_stopped_by_row[row_idx]),
                "hit_token_cap": bool(len(ids) >= int(max_new_tokens)),
                "likely_truncated": bool(
                    len(ids) >= int(max_new_tokens) and not ended_with_eos_by_row[row_idx] and not valid_stopped_by_row[row_idx]
                ),
                "is_valid": evaluation.get("is_valid"),
                "deceptive": evaluation.get("deceptive"),
                "is_truthful": None if evaluation.get("deceptive") is None else (not bool(evaluation["deceptive"])),
                "error": evaluation.get("error"),
                "parsed": evaluation.get("parsed"),
                "cards_played": evaluation.get("cards_played"),
                "action": evaluation.get("action"),
            }
        )
    return rows


def _group_generation_rows_by_prompt(
    generation_rows: list[dict[str, Any]],
) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in generation_rows:
        key = (
            str(row.get("environment", "")),
            str(row.get("pair_id", row.get("example_id", row.get("prompt_id", "")))),
            str(row.get("split", "")),
        )
        grouped.setdefault(key, []).append(row)
    return grouped


def summarize_generation_rows(
    generation_rows: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prompt_summary_rows: list[dict[str, Any]] = []
    for grouped_rows in _group_generation_rows_by_prompt(generation_rows).values():
        condition_groups: dict[str, list[dict[str, Any]]] = {}
        for row in grouped_rows:
            condition_groups.setdefault(str(row.get("condition", "steered")), []).append(row)
        for condition_name, rows in sorted(condition_groups.items()):
            rows = sorted(rows, key=lambda row: int(row.get("sample_idx", 0)))
            representative = rows[0]
            valid_rows = [row for row in rows if row.get("is_valid") is True]
            deceptive_rows = [row for row in rows if row.get("deceptive") is True]
            truthful_rows = [row for row in rows if row.get("deceptive") is False or row.get("is_truthful") is True]
            prompt_summary_rows.append(
                {
                    "environment": str(representative.get("environment", "")),
                    "split": str(representative.get("split", "")),
                    "pair_id": str(representative.get("pair_id", "")),
                    "example_id": str(representative.get("example_id", "")),
                    "condition": condition_name,
                    "n_samples": int(len(rows)),
                    "valid_generations": int(len(valid_rows)),
                    "valid_rate": float(len(valid_rows)) / float(len(rows)) if rows else float("nan"),
                    "deceptive_generations": int(len(deceptive_rows)),
                    "truthful_generations": int(len(truthful_rows)),
                    "deceptive_rate_overall": (
                        float(len(deceptive_rows)) / float(len(rows))
                        if rows
                        else float("nan")
                    ),
                    "truthful_rate_overall": (
                        float(len(truthful_rows)) / float(len(rows))
                        if rows
                        else float("nan")
                    ),
                    "invalid_rate": (
                        float(len(rows) - len(valid_rows)) / float(len(rows))
                        if rows
                        else float("nan")
                    ),
                    "mean_shared_context_deception_rate": (
                        float(representative["shared_context_deception_rate"])
                        if representative.get("shared_context_deception_rate") is not None
                        else float("nan")
                    ),
                    "mean_commitment_deception_rate": (
                        float(representative["commitment_deception_rate"])
                        if representative.get("commitment_deception_rate") is not None
                        else float("nan")
                    ),
                    "mean_commitment_delta": (
                        float(representative["commitment_delta"])
                        if representative.get("commitment_delta") is not None
                        else float("nan")
                    ),
                    "mean_new_tokens": float(np.mean([float(row.get("n_new_tokens", 0.0)) for row in rows])) if rows else float("nan"),
                    "hit_token_cap_rate": float(np.mean([bool(row.get("hit_token_cap")) for row in rows])) if rows else float("nan"),
                    "parse_error_rate": float(np.mean([row.get("error") is not None for row in rows])) if rows else float("nan"),
                }
            )
    prompt_summary_df = pd.DataFrame(prompt_summary_rows)
    if prompt_summary_df.empty:
        return prompt_summary_df, pd.DataFrame()
    environment_summary_df = (
        prompt_summary_df.groupby(["environment", "split", "condition"], dropna=False)
        .agg(
            n_prompts=("pair_id", "nunique"),
            n_samples=("n_samples", "sum"),
            valid_generations=("valid_generations", "sum"),
            deceptive_generations=("deceptive_generations", "sum"),
            truthful_generations=("truthful_generations", "sum"),
            valid_rate=("valid_rate", "mean"),
            mean_deceptive_rate_overall=("deceptive_rate_overall", "mean"),
            mean_truthful_rate_overall=("truthful_rate_overall", "mean"),
            mean_invalid_rate=("invalid_rate", "mean"),
            mean_shared_context_deception_rate=("mean_shared_context_deception_rate", "mean"),
            mean_commitment_deception_rate=("mean_commitment_deception_rate", "mean"),
            mean_commitment_delta=("mean_commitment_delta", "mean"),
            mean_new_tokens=("mean_new_tokens", "mean"),
            hit_token_cap_rate=("hit_token_cap_rate", "mean"),
            parse_error_rate=("parse_error_rate", "mean"),
        )
        .reset_index()
        .sort_values(["environment", "split", "condition"])
        .reset_index(drop=True)
    )
    environment_summary_df["weighted_deceptive_rate_overall"] = [
        float(deceptive_generations) / float(n_samples) if int(n_samples) > 0 else float("nan")
        for deceptive_generations, n_samples in zip(
            environment_summary_df["deceptive_generations"],
            environment_summary_df["n_samples"],
        )
    ]
    environment_summary_df["weighted_truthful_rate_overall"] = [
        float(truthful_generations) / float(n_samples) if int(n_samples) > 0 else float("nan")
        for truthful_generations, n_samples in zip(
            environment_summary_df["truthful_generations"],
            environment_summary_df["n_samples"],
        )
    ]
    environment_summary_df["weighted_invalid_rate"] = [
        1.0 - float(valid_generations) / float(n_samples) if int(n_samples) > 0 else float("nan")
        for valid_generations, n_samples in zip(
            environment_summary_df["valid_generations"],
            environment_summary_df["n_samples"],
        )
    ]
    return prompt_summary_df, environment_summary_df


def write_steering_prompt_payloads(
    output_dir: Path,
    generation_rows: list[dict[str, Any]],
    *,
    run_metadata: dict[str, Any],
) -> None:
    examples_dir = Path(output_dir) / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    for grouped_rows in _group_generation_rows_by_prompt(generation_rows).values():
        representative = grouped_rows[0]
        condition_payloads: dict[str, Any] = {}
        condition_groups: dict[str, list[dict[str, Any]]] = {}
        for row in grouped_rows:
            condition_groups.setdefault(str(row.get("condition", "steered")), []).append(row)
        for condition_name, rows in sorted(condition_groups.items()):
            rows = sorted(rows, key=lambda row: int(row.get("sample_idx", 0)))
            valid_rows = [row for row in rows if row.get("is_valid") is True]
            deceptive_rows = [row for row in rows if row.get("deceptive") is True]
            truthful_rows = [row for row in rows if row.get("deceptive") is False or row.get("is_truthful") is True]
            condition_payloads[condition_name] = {
                "summary": {
                    "n_samples": int(len(rows)),
                    "valid_generations": int(len(valid_rows)),
                    "valid_rate": float(len(valid_rows)) / float(len(rows)) if rows else float("nan"),
                    "deceptive_generations": int(len(deceptive_rows)),
                    "truthful_generations": int(len(truthful_rows)),
                    "deceptive_rate_overall": (
                        float(len(deceptive_rows)) / float(len(rows))
                        if rows
                        else float("nan")
                    ),
                    "truthful_rate_overall": (
                        float(len(truthful_rows)) / float(len(rows))
                        if rows
                        else float("nan")
                    ),
                    "invalid_rate": (
                        float(len(rows) - len(valid_rows)) / float(len(rows))
                        if rows
                        else float("nan")
                    ),
                },
                "generations": [
                    {
                        "sample_idx": int(row.get("sample_idx", 0)),
                        "seed": int(row.get("seed", 0)),
                        "generated_text": str(row.get("generated_text", "")),
                        "full_text": str(row.get("full_text", "")),
                        "n_new_tokens": int(row.get("n_new_tokens", 0)),
                        "ended_with_eos": bool(row.get("ended_with_eos")),
                        "early_stopped_on_valid_json": bool(row.get("early_stopped_on_valid_json")),
                        "hit_token_cap": bool(row.get("hit_token_cap")),
                        "likely_truncated": bool(row.get("likely_truncated")),
                        "evaluation": {
                            "is_valid": row.get("is_valid"),
                            "deceptive": row.get("deceptive"),
                            "is_truthful": row.get("is_truthful"),
                            "error": row.get("error"),
                            "action": row.get("action"),
                            "cards_played": row.get("cards_played"),
                            "parsed": row.get("parsed"),
                        },
                    }
                    for row in rows
                ],
            }
        payload = {
            "pair_id": str(representative.get("pair_id", "")),
            "example_id": str(representative.get("example_id", "")),
            "environment": str(representative.get("environment", "")),
            "split": str(representative.get("split", "")),
            "localization_path": str(representative.get("localization_path", "")),
            "prompt_text": str(representative.get("prompt_text", "")),
            "eval_context": representative.get("eval_context") if isinstance(representative.get("eval_context"), dict) else {},
            "required_rank": representative.get("required_rank"),
            "shared_context_sentence_pos": representative.get("shared_context_sentence_pos"),
            "commitment_sentence_pos": representative.get("commitment_sentence_pos"),
            "shared_context_deception_rate": representative.get("shared_context_deception_rate"),
            "commitment_deception_rate": representative.get("commitment_deception_rate"),
            "commitment_delta": representative.get("commitment_delta"),
            "deceptive_commitment_sentence": representative.get("deceptive_commitment_sentence"),
            "truthful_donor_sentence": representative.get("truthful_donor_sentence"),
            "run_metadata": run_metadata,
            "conditions": condition_payloads,
        }
        filename = (
            f"steering_{slugify(str(representative.get('environment', 'env')))}__"
            f"{slugify(str(representative.get('pair_id', representative.get('example_id', 'prompt'))))}.json"
        )
        write_json(examples_dir / filename, payload)


def run_analysis(args: argparse.Namespace) -> None:
    scope = resolve_scope_run(args)
    required_pair_count = int(args.pair_count) if int(args.pair_count) > 0 else (
        int(args.train_pair_count) + int(args.validation_pair_count) + int(args.test_pair_count)
    )
    if required_pair_count < (int(args.train_pair_count) + int(args.validation_pair_count) + int(args.test_pair_count)):
        raise ValueError("--pair-count must be at least train + validation + test pair counts.")

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    model_id = str(args.model_id)
    dataset_model_id = str(args.dataset_model_id).strip() or model_id
    model_name_or_path = resolve_model_name_or_path(
        model_id=model_id,
        model_name_or_path=str(args.model_name_or_path),
        hf_cache_root=Path(args.hf_cache_root),
    )
    output_root = build_output_root(Path(args.output_base).expanduser().resolve(), args.run_tag or None, scope.run_name)
    localization_dir, pair_cache_path = resolve_analysis_paths(
        dataset_root=dataset_root,
        environment=str(args.environment).strip().lower(),
        dataset_model_id=dataset_model_id,
        localization_dir=str(args.localization_dir),
        pair_cache_path=str(args.pair_cache_path),
    )
    runtime = load_head_runtime(
        model_id=model_id,
        model_name_or_path=model_name_or_path,
        dtype_name=str(args.dtype),
        device_map=str(args.device_map),
        cuda_visible_devices=str(args.cuda_visible_devices),
        pytorch_cuda_alloc_conf=str(args.pytorch_cuda_alloc_conf),
    )
    activation_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    write_json(
        output_root / "runner_config.json",
        {
            "command": "analyze",
            "argv": sys.argv,
            "scope": asdict(scope),
            "model_id": model_id,
            "dataset_model_id": dataset_model_id,
            "model_name_or_path": model_name_or_path,
            "localization_dir": str(localization_dir),
            "pair_cache_path": str(pair_cache_path),
            "output_root": str(output_root),
            "args": vars(args),
        },
    )

    max_input_tokens = None if str(args.max_input_tokens).strip().lower() in {"", "auto", "none"} else int(args.max_input_tokens)
    pairs_df = load_commitment_pairs_for_analysis(
        localization_dir=localization_dir,
        pair_cache_path=pair_cache_path,
        pair_count=required_pair_count,
        pair_search_limit=int(args.pair_search_limit),
        refresh_cache=bool(args.refresh_pair_cache),
        min_commitment_delta=float(args.min_commitment_delta),
        min_commitment_deception_rate=float(args.min_commitment_deception_rate),
        min_donor_clarity_score=float(args.min_donor_clarity_score),
        min_num_valid=int(args.min_num_valid),
        min_sentence_alpha_words=int(args.min_sentence_alpha_words),
        exclude_multiline_sentences=bool(args.exclude_multiline_sentences),
        disable_tqdm=bool(args.disable_tqdm),
    )
    pairs_df = ensure_truthful_donor_token_budget(
        runtime.tokenizer,
        pairs_df,
        max_input_tokens=max_input_tokens,
    )
    split_df = assign_pair_splits(
        pairs_df,
        train_pair_count=int(args.train_pair_count),
        validation_pair_count=int(args.validation_pair_count),
        test_pair_count=int(args.test_pair_count),
        seed=int(args.split_seed),
    )
    split_counts = (
        split_df.groupby("split", dropna=False)
        .size()
        .rename("n_pairs")
        .reset_index()
        .sort_values("split")
        .reset_index(drop=True)
    )
    print("Split counts:")
    if not split_counts.empty:
        print(split_counts.to_string(index=False))

    train_df = split_df.loc[split_df["split"].eq("train")].copy().reset_index(drop=True)
    validation_df = split_df.loc[split_df["split"].eq("validation")].copy().reset_index(drop=True)
    test_df = split_df.loc[split_df["split"].eq("test")].copy().reset_index(drop=True)
    if train_df.empty:
        raise ValueError("No train pairs are available after splitting.")
    if validation_df.empty:
        raise ValueError("No validation pairs are available after splitting.")
    if test_df.empty:
        raise ValueError("No test pairs are available after splitting.")

    train_split = prepare_split(
        runtime.tokenizer,
        "train",
        train_df,
        batch_pair_count=int(args.batch_pair_count),
        max_input_tokens=max_input_tokens,
    )
    validation_split = prepare_split(
        runtime.tokenizer,
        "validation",
        validation_df,
        batch_pair_count=int(args.batch_pair_count),
        max_input_tokens=max_input_tokens,
    )
    test_split = prepare_split(
        runtime.tokenizer,
        "test",
        test_df,
        batch_pair_count=int(args.batch_pair_count),
        max_input_tokens=max_input_tokens,
    )
    pairs_overview_df = pd.concat(
        [train_split.pairs_overview_df, validation_split.pairs_overview_df, test_split.pairs_overview_df],
        ignore_index=True,
    )

    train_summary_df, train_metric_sanity_df, train_totals = compute_split_baseline_summary(
        runtime,
        train_split,
        sentence_score_mode=str(args.sentence_score_mode),
    )
    patching_results, diagnostics_df, layer_summary_df = compute_head_attributions(
        runtime,
        train_split,
        sentence_score_mode=str(args.sentence_score_mode),
        patch_scope=scope.patch_scope,
        patch_first_n_tokens=scope.patch_first_n_tokens,
        offload_attr_tensors_to_cpu=bool(args.offload_attr_tensors_to_cpu),
    )
    site_df, ranked_site_df, ranked_sites, all_sites = build_ranked_sites(
        patching_results,
        n_layers=runtime.n_layers,
        n_heads=runtime.n_heads,
        circuit_select=str(args.circuit_select),
        max_circuit_size_raw=str(args.max_circuit_size),
    )
    print(
        f"Ranked {len(ranked_sites)} candidate head-site(s) using circuit_select={args.circuit_select!r}; "
        f"patch_scope={scope.patch_scope!r}."
    )

    validation_chunks = subset_chunks(validation_split.pair_chunks, int(args.validation_max_chunks))
    (
        validation_source_truthful_score,
        validation_target_deceptive_score,
        validation_target_baseline,
        validation_truth_minus_deceptive_baseline,
        validation_baseline_chunk_records,
    ) = score_source_target_baselines(
        runtime,
        validation_chunks,
        sentence_score_mode=str(args.sentence_score_mode),
        return_chunk_records=True,
    )
    baseline_unpatched_scores = [float(record["target_metric"]) for record in validation_baseline_chunk_records]
    validation_baseline_df = pd.DataFrame(validation_baseline_chunk_records)
    candidate_sizes = candidate_circuit_sizes(
        len(ranked_sites),
        explicit_sizes=parse_int_list(args.candidate_circuit_sizes),
    )
    if not candidate_sizes:
        raise ValueError("No candidate circuit sizes are available.")
    print(f"Sweeping candidate circuit sizes on validation: {candidate_sizes}")
    discovered_edge_count, selection_record, circuit_search_df, circuit_search_status = sweep_top_k_percent_reduction_circuits(
        runtime,
        ranked_sites,
        validation_chunks,
        sentence_score_mode=str(args.sentence_score_mode),
        patch_scope=scope.patch_scope,
        patch_first_n_tokens=scope.patch_first_n_tokens,
        activation_device=activation_device,
        target_baseline=validation_target_baseline,
        unpatched_scores=baseline_unpatched_scores,
        percent_reduction_threshold=float(args.percent_reduction_threshold),
        candidate_sizes=candidate_sizes,
        flattening_percent_point_tol=float(args.flattening_percent_point_tol),
    )
    discovered_sites = top_ranked_sites(ranked_sites, discovered_edge_count)
    discovered_circuit_df = ranked_site_df.head(discovered_edge_count).reset_index(drop=True)

    test_chunks = subset_chunks(test_split.pair_chunks, int(args.test_max_chunks))
    (
        test_source_truthful_score,
        test_target_deceptive_score,
        test_target_baseline,
        test_truth_minus_deceptive_baseline,
        test_baseline_chunk_records,
    ) = score_source_target_baselines(
        runtime,
        test_chunks,
        sentence_score_mode=str(args.sentence_score_mode),
        return_chunk_records=True,
    )
    test_unpatched_scores = [float(record["target_metric"]) for record in test_baseline_chunk_records]
    test_baseline_df = pd.DataFrame(test_baseline_chunk_records)
    test_discovered_record = score_circuit_on_chunks(
        runtime,
        discovered_sites,
        test_chunks,
        sentence_score_mode=str(args.sentence_score_mode),
        patch_scope=scope.patch_scope,
        patch_first_n_tokens=scope.patch_first_n_tokens,
        activation_device=activation_device,
        target_baseline=test_target_baseline,
        unpatched_scores=test_unpatched_scores,
        label="test_discovered",
    )
    rng = np.random.default_rng(int(args.control_seed))
    selected_controls_df = score_selected_circuit_controls(
        runtime,
        discovered_sites,
        test_chunks,
        sentence_score_mode=str(args.sentence_score_mode),
        patch_scope=scope.patch_scope,
        patch_first_n_tokens=scope.patch_first_n_tokens,
        activation_device=activation_device,
        target_baseline=test_target_baseline,
        unpatched_scores=test_unpatched_scores,
        rng=rng,
        donor_pool=test_split.prepared_pairs,
        all_sites=all_sites,
        control_count=int(args.control_circuit_count),
        mean_shuffled_donor_count=int(args.mean_shuffled_donor_count),
        label_prefix=f"test_top_{discovered_edge_count}",
        n_heads=runtime.n_heads,
    )
    comparison_records = [
        {
            "evaluation_split": "test",
            "circuit_kind": "target_reference",
            "circuit_id": "target_unpatched_b_prime",
            "search_status": "reference",
            "selection_status": circuit_search_status,
            "selected_edge_count": int(discovered_edge_count),
            "percent_reduction_threshold": float(args.percent_reduction_threshold),
            "patch_scope": scope.patch_scope,
            "candidate_edge_count": 0,
            "meets_threshold": False,
            "ranked_candidate_count": len(ranked_sites),
            "total_attention_sites": len(all_sites),
            "source_truthful_score": test_source_truthful_score,
            "truth_minus_deceptive_reference": test_truth_minus_deceptive_baseline,
            "unpatched_metric": test_target_baseline,
            "patched_metric": test_target_baseline,
            "source_role": "truthful",
            "target_role": "deceptive",
            "target_baseline_metric": test_target_baseline,
            "delta": 0.0,
            "percent_probability_reduction": 0.0,
            "unpatched_target_sentence_score": test_target_deceptive_score,
            "patched_target_sentence_score": test_target_deceptive_score,
            "target_sentence_score_delta": 0.0,
            "unpatched_target_deceptive_score": test_target_deceptive_score,
            "patched_target_deceptive_score": test_target_deceptive_score,
            "target_deceptive_score_delta": 0.0,
            "n_chunks": len(test_chunks),
            "n_pairs": int(sum(len(chunk) for chunk in test_chunks)),
            "circuit_size": 0,
        },
        {
            "evaluation_split": "test",
            "circuit_kind": "discovered",
            "circuit_id": f"top_{discovered_edge_count}",
            "search_status": "selected_on_validation",
            "selection_status": circuit_search_status,
            "selected_edge_count": int(discovered_edge_count),
            "selection_delta": selection_record["delta"],
            "selection_percent_probability_reduction": selection_record["percent_probability_reduction"],
            "percent_reduction_threshold": float(args.percent_reduction_threshold),
            "patch_scope": scope.patch_scope,
            "ranked_candidate_count": len(ranked_sites),
            "total_attention_sites": len(all_sites),
            "source_truthful_score": test_source_truthful_score,
            "truth_minus_deceptive_reference": test_truth_minus_deceptive_baseline,
            **test_discovered_record,
        },
    ]
    for row in selected_controls_df.to_dict(orient="records"):
        comparison_records.append(
            {
                "evaluation_split": "test",
                "search_status": "selected_size_control",
                "selection_status": circuit_search_status,
                "selected_edge_count": int(discovered_edge_count),
                "percent_reduction_threshold": float(args.percent_reduction_threshold),
                "patch_scope": scope.patch_scope,
                "ranked_candidate_count": len(ranked_sites),
                "total_attention_sites": len(all_sites),
                "source_truthful_score": test_source_truthful_score,
                "truth_minus_deceptive_reference": test_truth_minus_deceptive_baseline,
                "meets_threshold": False,
                **row,
            }
        )
    control_comparison_df = pd.DataFrame(comparison_records)
    reported_control_kinds = [
        "discovered",
        "random",
        "layer_matched_random",
        "shuffled_truthful_donor",
        "shuffled_deceptive_donor",
        "averaged_shuffled_deceptive_donor",
        "shuffled_difference_donor",
        "averaged_shuffled_difference_donor",
    ]
    summary_comparison_df = control_comparison_df[
        control_comparison_df["circuit_kind"].isin(reported_control_kinds)
    ].copy()

    cross_records = []
    for env_name in parse_string_list(args.cross_corpus_envs) or list(DEFAULT_CROSS_CORPUS_ENVS):
        requested_env_pair_count = int(args.cross_corpus_pair_count)
        env_pairs_df = load_generic_commitment_pairs_for_env(
            dataset_root=dataset_root,
            environment=env_name,
            dataset_model_id=dataset_model_id,
            pair_count=requested_env_pair_count,
            search_limit=int(args.cross_corpus_search_limit),
            min_commitment_delta=float(args.min_commitment_delta),
            min_commitment_deception_rate=float(args.min_commitment_deception_rate),
            min_num_valid=int(args.min_num_valid),
            min_sentence_alpha_words=int(args.min_sentence_alpha_words),
            exclude_multiline_sentences=bool(args.exclude_multiline_sentences),
        )
        if env_pairs_df.empty:
            cross_records.append(
                {
                    "environment": env_name,
                    "evaluation_split": "ood_test",
                    "source_environment": str(args.environment),
                    "status": "no_pairs",
                    "n_pairs_requested": requested_env_pair_count,
                    "n_pairs_loaded": 0,
                    "n_pairs_after_donor_budget": 0,
                    "n_pairs_after_resample": 0,
                    "pairs_sampled_with_replacement": 0,
                    "n_pairs": 0,
                }
            )
            continue
        loaded_env_pair_count = int(len(env_pairs_df))
        env_pairs_df = ensure_truthful_donor_token_budget(
            runtime.tokenizer,
            env_pairs_df,
            max_input_tokens=max_input_tokens,
            allow_empty=True,
            context_label=f"OOD {env_name}",
        )
        if env_pairs_df.empty:
            cross_records.append(
                {
                    "environment": env_name,
                    "evaluation_split": "ood_test",
                    "source_environment": str(args.environment),
                    "status": "no_pairs_after_donor_budget",
                    "n_pairs_requested": requested_env_pair_count,
                    "n_pairs_loaded": loaded_env_pair_count,
                    "n_pairs_after_donor_budget": 0,
                    "n_pairs_after_resample": 0,
                    "pairs_sampled_with_replacement": 0,
                    "n_pairs": 0,
                }
            )
            continue
        post_budget_pair_count = int(len(env_pairs_df))
        env_pairs_df = upsample_pairs_with_replacement(
            env_pairs_df,
            target_count=requested_env_pair_count,
            seed=deterministic_seed(int(args.control_seed), "ood_pair_reuse", env_name, requested_env_pair_count),
            context_label=f"OOD {env_name}",
        )
        resampled_env_pair_count = int(len(env_pairs_df))
        sampled_with_replacement_count = int(
            env_pairs_df.get("sampled_with_replacement", pd.Series(False, index=env_pairs_df.index))
            .fillna(False)
            .astype(bool)
            .sum()
        )
        env_split = prepare_split(
            runtime.tokenizer,
            env_name,
            env_pairs_df,
            batch_pair_count=int(args.batch_pair_count),
            max_input_tokens=max_input_tokens,
        )
        env_chunks = subset_chunks(env_split.pair_chunks, int(args.cross_corpus_max_chunks))
        (
            env_source_truthful_score,
            env_target_deceptive_score,
            env_target_baseline,
            env_truth_minus_deceptive,
            env_baseline_chunk_records,
        ) = score_source_target_baselines(
            runtime,
            env_chunks,
            sentence_score_mode=str(args.sentence_score_mode),
            return_chunk_records=True,
        )
        env_unpatched_scores = [float(record["target_metric"]) for record in env_baseline_chunk_records]
        env_record = score_circuit_on_chunks(
            runtime,
            discovered_sites,
            env_chunks,
            sentence_score_mode=str(args.sentence_score_mode),
            patch_scope=scope.patch_scope,
            patch_first_n_tokens=scope.patch_first_n_tokens,
            activation_device=activation_device,
            target_baseline=env_target_baseline,
            unpatched_scores=env_unpatched_scores,
            label=f"{env_name}_discovered",
        )
        cross_records.append(
            {
                "environment": env_name,
                "evaluation_split": "ood_test",
                "source_environment": str(args.environment),
                "status": "ok",
                "circuit_kind": "discovered",
                "circuit_id": f"{env_name}_discovered",
                "source_truthful_score": env_source_truthful_score,
                "target_deceptive_score": env_target_deceptive_score,
                "target_baseline": env_target_baseline,
                "truth_minus_deceptive_reference": env_truth_minus_deceptive,
                "max_total_len": max(int(pair["max_total_len"]) for pair in env_split.prepared_pairs),
                "n_pairs_requested": requested_env_pair_count,
                "n_pairs_loaded": loaded_env_pair_count,
                "n_pairs_after_donor_budget": post_budget_pair_count,
                "n_pairs_after_resample": resampled_env_pair_count,
                "pairs_sampled_with_replacement": sampled_with_replacement_count,
                **env_record,
            }
        )
        env_controls_df = score_selected_circuit_controls(
            runtime,
            discovered_sites,
            env_chunks,
            sentence_score_mode=str(args.sentence_score_mode),
            patch_scope=scope.patch_scope,
            patch_first_n_tokens=scope.patch_first_n_tokens,
            activation_device=activation_device,
            target_baseline=env_target_baseline,
            unpatched_scores=env_unpatched_scores,
            rng=rng,
            donor_pool=env_split.prepared_pairs,
            all_sites=all_sites,
            control_count=int(args.control_circuit_count),
            mean_shuffled_donor_count=int(args.mean_shuffled_donor_count),
            label_prefix=env_name,
            n_heads=runtime.n_heads,
        )
        for control_row in env_controls_df.to_dict(orient="records"):
            cross_records.append(
                {
                    "environment": env_name,
                    "evaluation_split": "ood_test",
                    "source_environment": str(args.environment),
                    "status": "ok",
                    "source_truthful_score": env_source_truthful_score,
                    "target_deceptive_score": env_target_deceptive_score,
                    "target_baseline": env_target_baseline,
                    "truth_minus_deceptive_reference": env_truth_minus_deceptive,
                    "max_total_len": max(int(pair["max_total_len"]) for pair in env_split.prepared_pairs),
                    "n_pairs_requested": requested_env_pair_count,
                    "n_pairs_loaded": loaded_env_pair_count,
                    "n_pairs_after_donor_budget": post_budget_pair_count,
                    "n_pairs_after_resample": resampled_env_pair_count,
                    "pairs_sampled_with_replacement": sampled_with_replacement_count,
                    **control_row,
                }
            )
        clear_memory(runtime.model)
    cross_corpus_df = pd.DataFrame(cross_records)
    shuffled_difference_donor_df = control_comparison_df.loc[
        control_comparison_df["circuit_kind"].eq("shuffled_difference_donor")
    ].copy()
    averaged_shuffled_difference_donor_df = control_comparison_df.loc[
        control_comparison_df["circuit_kind"].eq("averaged_shuffled_difference_donor")
    ].copy()
    if "circuit_kind" in cross_corpus_df.columns:
        cross_corpus_shuffled_difference_donor_df = cross_corpus_df.loc[
            cross_corpus_df["circuit_kind"].astype(str).eq("shuffled_difference_donor")
        ].copy()
        cross_corpus_averaged_shuffled_difference_donor_df = cross_corpus_df.loc[
            cross_corpus_df["circuit_kind"].astype(str).eq("averaged_shuffled_difference_donor")
        ].copy()
    else:
        cross_corpus_shuffled_difference_donor_df = pd.DataFrame()
        cross_corpus_averaged_shuffled_difference_donor_df = pd.DataFrame()

    steering_vectors, steering_vector_df = compute_head_steering_vectors(
        runtime,
        train_split.pair_chunks,
        discovered_sites,
    )
    vector_metadata = save_vector_bundle(
        output_root,
        analysis_dir=output_root,
        selected_splits=["train"],
        model_id=model_id,
        model_name_or_path=model_name_or_path,
        dtype_name=str(args.dtype),
        dataset_model_id=dataset_model_id,
        environment=str(args.environment),
        discovered_sites=discovered_sites,
        prepared_split=train_split,
        vectors=steering_vectors,
        steering_vector_df=steering_vector_df,
        batch_pair_count=int(args.batch_pair_count),
        max_input_tokens=max_input_tokens,
        extra_table_path=output_root / "tables" / "steering_vector_df.csv",
    )

    tables_dir = output_root / "tables"
    arrays_dir = output_root / "arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    save_dataframe(split_df, tables_dir / "split_pairs_df.csv")
    write_jsonl(tables_dir / "split_pairs.jsonl", split_df.to_dict(orient="records"))
    save_dataframe(pairs_df, tables_dir / "pairs_df.csv")
    save_dataframe(pairs_overview_df, tables_dir / "pairs_overview_df.csv")
    save_dataframe(train_summary_df, tables_dir / "train_summary_df.csv")
    save_dataframe(train_metric_sanity_df, tables_dir / "train_metric_sanity_df.csv")
    save_dataframe(diagnostics_df, tables_dir / "diagnostics_df.csv")
    save_dataframe(layer_summary_df, tables_dir / "layer_summary_df.csv")
    save_dataframe(site_df, tables_dir / "site_df.csv")
    save_dataframe(ranked_site_df, tables_dir / "ranked_site_df.csv")
    save_dataframe(validation_baseline_df, tables_dir / "validation_baseline_df.csv")
    save_dataframe(circuit_search_df, tables_dir / "circuit_search_df.csv")
    save_dataframe(discovered_circuit_df, tables_dir / "discovered_circuit_df.csv")
    save_dataframe(test_baseline_df, tables_dir / "test_baseline_df.csv")
    save_dataframe(selected_controls_df, tables_dir / "selected_controls_df.csv")
    save_dataframe(control_comparison_df, tables_dir / "control_comparison_df.csv")
    save_dataframe(summary_comparison_df, tables_dir / "summary_comparison_df.csv")
    save_dataframe(shuffled_difference_donor_df, tables_dir / "shuffled_difference_donor_df.csv")
    save_dataframe(
        averaged_shuffled_difference_donor_df,
        tables_dir / "averaged_shuffled_difference_donor_df.csv",
    )
    save_dataframe(cross_corpus_df, tables_dir / "cross_corpus_df.csv")
    save_dataframe(
        cross_corpus_shuffled_difference_donor_df,
        tables_dir / "cross_corpus_shuffled_difference_donor_df.csv",
    )
    save_dataframe(
        cross_corpus_averaged_shuffled_difference_donor_df,
        tables_dir / "cross_corpus_averaged_shuffled_difference_donor_df.csv",
    )
    np.save(arrays_dir / "patching_results.npy", patching_results)
    write_json(output_root / "discovered_sites.json", {"sites": discovered_sites})
    metadata = {
        "scope": asdict(scope),
        "environment": str(args.environment),
        "dataset_root": str(dataset_root),
        "model_id": model_id,
        "dataset_model_id": dataset_model_id,
        "model_name_or_path": model_name_or_path,
        "dtype_name": str(args.dtype),
        "localization_dir": str(localization_dir),
        "pair_cache_path": str(pair_cache_path),
        "cross_corpus_envs": parse_string_list(args.cross_corpus_envs) or list(DEFAULT_CROSS_CORPUS_ENVS),
        "reported_control_kinds": list(reported_control_kinds),
        "mean_shuffled_donor_count": int(args.mean_shuffled_donor_count),
        "averaged_shuffled_deceptive_donor_count": int(args.mean_shuffled_donor_count),
        "averaged_shuffled_difference_donor_count": int(args.mean_shuffled_donor_count),
        "split_counts": split_counts.to_dict(orient="records"),
        "train_dataset_max_input_tokens": int(train_split.dataset_max_input_tokens),
        "validation_dataset_max_input_tokens": int(validation_split.dataset_max_input_tokens),
        "test_dataset_max_input_tokens": int(test_split.dataset_max_input_tokens),
        "patching_results_shape": list(patching_results.shape),
        "ranked_site_count": len(ranked_sites),
        "all_site_count": len(all_sites),
        "discovered_edge_count": int(discovered_edge_count),
        "discovered_sites": discovered_sites,
        "steering_vector_path": str(output_root / "steering_vectors.pt"),
        "steering_vector_source_splits": ["train"],
        "steering_vector_n_pairs": int(train_split.total_pairs),
        "vector_metadata": vector_metadata,
        "selection_record": selection_record,
        "selection_status": circuit_search_status,
        "test_record": test_discovered_record,
        "train_totals": train_totals,
        "validation_totals": {
            "source_truthful_score": validation_source_truthful_score,
            "target_deceptive_score": validation_target_deceptive_score,
            "target_baseline": validation_target_baseline,
            "truth_minus_deceptive_baseline": validation_truth_minus_deceptive_baseline,
        },
        "test_totals": {
            "source_truthful_score": test_source_truthful_score,
            "target_deceptive_score": test_target_deceptive_score,
            "target_baseline": test_target_baseline,
            "truth_minus_deceptive_baseline": test_truth_minus_deceptive_baseline,
        },
    }
    write_json(output_root / "metadata.json", metadata)
    clear_memory(runtime.model)
    print(f"Saved analysis outputs to {output_root}")


def run_vector_export(args: argparse.Namespace) -> None:
    analysis_bundle = load_analysis_bundle(Path(args.analysis_dir))
    metadata = analysis_bundle["metadata"]
    config = analysis_bundle["config"]
    selected_splits = parse_string_list(args.vector_source_splits) or ["train"]
    selected_rows = filter_rows_by_split(analysis_bundle["split_pair_rows"], split_names=selected_splits)
    if not selected_rows:
        raise ValueError(f"No rows matched vector_source_splits={selected_splits}.")
    vector_df_input = pd.DataFrame(selected_rows).reset_index(drop=True)
    model_id = str(args.model_id).strip() or str(metadata.get("model_id", DEFAULT_MODEL_ID))
    model_name_or_path = str(args.model_name_or_path).strip() or str(metadata.get("model_name_or_path", ""))
    if not model_name_or_path:
        raise ValueError("Could not resolve a model_name_or_path for vector export.")
    dtype_name = str(args.dtype).strip() or str(metadata.get("dtype_name", DEFAULT_DTYPE_NAME))
    runtime = load_head_runtime(
        model_id=model_id,
        model_name_or_path=model_name_or_path,
        dtype_name=dtype_name,
        device_map=str(args.device_map),
        cuda_visible_devices=str(args.cuda_visible_devices),
        pytorch_cuda_alloc_conf=str(args.pytorch_cuda_alloc_conf),
    )
    max_input_tokens_raw = args.max_input_tokens if args.max_input_tokens else config.get("args", {}).get("max_input_tokens", "auto")
    max_input_tokens = None if str(max_input_tokens_raw).strip().lower() in {"", "auto", "none"} else int(max_input_tokens_raw)
    batch_pair_count = int(args.batch_pair_count) if int(args.batch_pair_count) > 0 else int(config.get("args", {}).get("batch_pair_count", DEFAULT_BATCH_PAIR_COUNT))
    prepared_split = prepare_split(
        runtime.tokenizer,
        "_".join(selected_splits),
        vector_df_input,
        batch_pair_count=batch_pair_count,
        max_input_tokens=max_input_tokens,
    )
    discovered_sites = analysis_bundle["discovered_sites"]
    if not discovered_sites:
        raise ValueError("No discovered_sites were found in the analysis bundle.")
    vectors, steering_vector_df = compute_head_steering_vectors(runtime, prepared_split.pair_chunks, discovered_sites)
    output_dir = build_vector_output_dir(
        Path(args.analysis_dir).expanduser().resolve(),
        str(args.output_dir),
        args.vector_tag or "_".join(selected_splits),
    )
    save_vector_bundle(
        output_dir,
        analysis_dir=Path(args.analysis_dir).expanduser().resolve(),
        selected_splits=selected_splits,
        model_id=model_id,
        model_name_or_path=model_name_or_path,
        dtype_name=dtype_name,
        dataset_model_id=metadata.get("dataset_model_id"),
        environment=metadata.get("environment"),
        discovered_sites=discovered_sites,
        prepared_split=prepared_split,
        vectors=vectors,
        steering_vector_df=steering_vector_df,
        batch_pair_count=int(batch_pair_count),
        max_input_tokens=max_input_tokens,
    )
    clear_memory(runtime.model)
    print(f"Saved steering vectors to {output_dir}")


def run_steering(args: argparse.Namespace) -> None:
    analysis_bundle = load_analysis_bundle(Path(args.analysis_dir)) if str(args.analysis_dir).strip() else None
    if str(args.vector_path).strip():
        vector_path = Path(args.vector_path).expanduser().resolve()
    elif analysis_bundle is not None:
        vector_path = Path(analysis_bundle["default_vector_path"]).expanduser().resolve()
    else:
        raise ValueError("Provide --vector-path or --analysis-dir with a saved steering_vectors.pt bundle.")
    steering_vectors, vector_metadata = load_vector_bundle(vector_path)
    if not steering_vectors:
        raise ValueError(f"No steering vectors were found in {vector_path}.")

    if str(args.prompt_text).strip():
        prompt_rows = _prepare_single_prompt_row(
            str(args.prompt_text),
            required_rank=None if args.required_rank is None else int(args.required_rank),
        )
    elif str(args.prompt_jsonl).strip():
        prompt_rows = _prepare_prompt_rows_from_jsonl(
            Path(args.prompt_jsonl).expanduser().resolve(),
            max_prompts=int(args.max_prompts),
        )
    else:
        prompt_envs = parse_string_list(args.prompt_environments)
        if not prompt_envs and analysis_bundle is not None:
            prompt_envs = default_prompt_environments_from_analysis(analysis_bundle, vector_metadata)
        if prompt_envs:
            dataset_root = Path(
                str(args.prompt_dataset_root).strip()
                or (analysis_bundle["metadata"].get("dataset_root", "") if analysis_bundle is not None else "")
                or str(REPO_ROOT / "DatasetMain")
            ).expanduser().resolve()
            dataset_model_id = (
                str(args.prompt_dataset_model_id).strip()
                or str(vector_metadata.get("dataset_model_id", ""))
                or (str(analysis_bundle["metadata"].get("dataset_model_id", "")) if analysis_bundle is not None else "")
            )
            if not dataset_model_id:
                raise ValueError(
                    "Could not resolve prompt_dataset_model_id. Pass --prompt-dataset-model-id or provide --analysis-dir with metadata."
                )
            env_tag = "_".join(prompt_envs)
            pair_cache_path = (
                Path(args.prompt_pair_cache_path).expanduser().resolve()
                if str(args.prompt_pair_cache_path).strip()
                else (
                    REPO_ROOT
                    / "Cache"
                    / "activation_steering"
                    / f"{dataset_model_id}__{env_tag}__bidirectional_pairs.jsonl"
                )
            )
            prompt_rows = _prepare_prompt_rows_from_environments(
                analysis_bundle,
                prompt_environments=prompt_envs,
                prompts_per_environment=int(args.prompts_per_environment),
                prompt_splits=parse_string_list(args.prompt_splits) or ["test"],
                dataset_root=dataset_root,
                dataset_model_id=dataset_model_id,
                pair_cache_path=pair_cache_path,
                refresh_pair_cache=bool(args.refresh_prompt_pair_cache),
                min_commitment_delta=float(args.prompt_min_commitment_delta),
                disable_tqdm=bool(args.disable_tqdm),
            )
        elif analysis_bundle is not None:
            prompt_rows = _prepare_prompt_rows_from_analysis(
                analysis_bundle,
                prompt_splits=parse_string_list(args.prompt_splits) or ["test"],
                pair_indices=parse_int_list(args.pair_indices),
                max_prompts=int(args.max_prompts),
            )
        else:
            raise ValueError("Provide one of --prompt-text, --prompt-jsonl, --prompt-environments, or --analysis-dir.")

    prompt_rows = filter_prompt_rows_by_prefix_deception_rate(
        prompt_rows,
        min_prefix_deception_rate=float(args.prompt_min_prefix_deception_rate),
    )
    prompt_rows = shard_rows(
        prompt_rows,
        num_shards=int(args.num_shards),
        shard_index=int(args.shard_index),
    )
    if not prompt_rows:
        raise ValueError("No prompt rows remain after filtering and sharding.")

    model_name_or_path = str(args.model_name_or_path).strip() or str(vector_metadata.get("model_name_or_path", ""))
    if not model_name_or_path and analysis_bundle is not None:
        model_name_or_path = str(analysis_bundle["metadata"].get("model_name_or_path", ""))
    if not model_name_or_path:
        raise ValueError("Could not resolve a model_name_or_path for steering.")
    dtype_name = (
        str(args.dtype).strip()
        or str(vector_metadata.get("dtype_name", ""))
        or (str(analysis_bundle["metadata"].get("dtype_name", "")) if analysis_bundle is not None else "")
        or DEFAULT_DTYPE_NAME
    )
    runtime = load_generation_runtime(
        model_name_or_path=model_name_or_path,
        dtype_name=dtype_name,
        device_map=str(args.device_map),
        cuda_visible_devices=str(args.cuda_visible_devices),
        pytorch_cuda_alloc_conf=str(args.pytorch_cuda_alloc_conf),
    )
    if runtime.head_dim <= 0:
        raise ValueError("Could not resolve a positive head_dim for steering runtime.")

    expected_sites = None
    if analysis_bundle is not None and analysis_bundle.get("discovered_sites"):
        expected_sites = analysis_bundle["discovered_sites"]
    elif vector_metadata.get("discovered_sites"):
        expected_sites = [tuple(site) for site in vector_metadata.get("discovered_sites", [])]
    vector_diagnostics_df = validate_steering_vectors(
        steering_vectors,
        expected_sites=expected_sites,
        expected_head_dim=int(runtime.head_dim),
        context=f"loaded steering bundle {vector_path}",
    )
    print(
        "Loaded steering vectors:",
        f"{len(vector_diagnostics_df)} site(s) | ",
        f"norm range=[{float(vector_diagnostics_df['direction_norm'].min()):.4f}, "
        f"{float(vector_diagnostics_df['direction_norm'].max()):.4f}]",
    )

    jobs = _build_generation_jobs(
        prompt_rows,
        samples_per_prompt=int(args.samples_per_prompt),
        base_seed=int(args.base_seed),
    )
    output_dir = build_steering_output_dir(
        vector_path,
        str(args.output_dir),
        str(args.run_tag),
        num_shards=int(args.num_shards),
        shard_index=int(args.shard_index),
    )
    write_jsonl(output_dir / "prompt_rows.jsonl", prompt_rows)
    save_dataframe(pd.DataFrame(prompt_rows), output_dir / "prompt_rows.csv")

    generation_rows: list[dict[str, Any]] = []
    conditions = [("steered", True, float(args.alpha))]
    if bool(args.include_baseline):
        conditions.insert(0, ("baseline", False, 0.0))

    for condition_name, do_steer, alpha in conditions:
        print(f"Running condition={condition_name} over {len(jobs)} prompt/sample jobs")
        for start_idx in range(0, len(jobs), int(args.batch_size)):
            batch_jobs = jobs[start_idx : start_idx + int(args.batch_size)]
            batch_rows = generate_batch_with_head_steering(
                runtime,
                batch_jobs,
                steering_vectors=steering_vectors if do_steer else {},
                alpha=alpha,
                steering_position=str(args.steering_position),
                steering_max_decode_tokens=int(args.steering_max_decode_tokens),
                max_model_length=int(args.max_model_length),
                max_new_tokens=int(args.max_new_tokens),
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                early_stop_on_valid_json=bool(args.early_stop_on_valid_json),
                early_stop_check_interval=int(args.early_stop_check_interval),
                early_stop_min_new_tokens=int(args.early_stop_min_new_tokens),
            )
            for row in batch_rows:
                generation_rows.append(
                    {
                        **row,
                        "condition": condition_name,
                        "alpha": float(alpha),
                        "vector_path": str(vector_path),
                    }
                )
            print(
                f"  condition={condition_name} | batch {1 + start_idx // int(args.batch_size)}/"
                f"{math.ceil(len(jobs) / int(args.batch_size))}"
            )

    generation_df = pd.DataFrame(generation_rows)
    prompt_summary_df, environment_summary_df = summarize_generation_rows(generation_rows)
    run_metadata = {
        "analysis_dir": str(Path(args.analysis_dir).expanduser().resolve()) if str(args.analysis_dir).strip() else "",
        "vector_path": str(vector_path),
        "vector_metadata": vector_metadata,
        "output_dir": str(output_dir),
        "n_prompt_rows": len(prompt_rows),
        "n_jobs": len(jobs),
        "conditions": [condition_name for condition_name, _, _alpha in conditions],
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "model_name_or_path": model_name_or_path,
        "dtype_name": dtype_name,
        "args": vars(args),
    }
    save_dataframe(generation_df, output_dir / "steering_generations.csv")
    write_jsonl(output_dir / "steering_generations.jsonl", generation_rows)
    save_dataframe(prompt_summary_df, output_dir / "steering_prompt_summary.csv")
    save_dataframe(environment_summary_df, output_dir / "steering_environment_summary.csv")
    write_json(output_dir / "steering_metadata.json", run_metadata)
    write_json(output_dir / "generation_metadata.json", run_metadata)
    write_steering_prompt_payloads(output_dir, generation_rows, run_metadata=run_metadata)
    clear_memory(runtime.model)
    print(f"Saved steering outputs to {output_dir}")


def add_common_model_args(
    parser: argparse.ArgumentParser,
    *,
    model_id_default: str = DEFAULT_MODEL_ID,
    model_name_or_path_default: str = "",
    dtype_default: str = DEFAULT_DTYPE_NAME,
) -> None:
    dtype_choices = sorted(DTYPE_BY_NAME) if str(dtype_default).strip() else None
    parser.add_argument("--model-id", type=str, default=model_id_default)
    parser.add_argument("--model-name-or-path", type=str, default=model_name_or_path_default)
    parser.add_argument("--hf-cache-root", type=str, default=str(DEFAULT_HF_CACHE_ROOT))
    parser.add_argument("--dtype", type=str, default=dtype_default, choices=dtype_choices)
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--cuda-visible-devices", type=str, default="")
    parser.add_argument("--pytorch-cuda-alloc-conf", type=str, default="expandable_segments:True")


def add_scope_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--scope-run", type=str, default="patch_first_1_token")
    parser.add_argument("--patch-scope", type=str, default="")
    parser.add_argument("--run-name", type=str, default="")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone head-only activation patching pipeline. "
            "Supports split-aware circuit discovery, steering-vector export, and batched steering generation."
        )
    )
    subparsers = parser.add_subparsers(dest="command")

    analyze = subparsers.add_parser(
        "analyze",
        help=(
            "Rank heads on train, choose circuit size on validation, report final results on test/OOD, "
            "and save the default train-split steering vector bundle."
        ),
    )
    add_common_model_args(analyze)
    add_scope_args(analyze)
    analyze.add_argument("--dataset-root", type=str, default=str(REPO_ROOT / "DatasetMain"))
    analyze.add_argument("--environment", type=str, default="bs")
    analyze.add_argument("--dataset-model-id", type=str, default="")
    analyze.add_argument("--localization-dir", type=str, default="")
    analyze.add_argument("--pair-cache-path", type=str, default="")
    analyze.add_argument("--refresh-pair-cache", action="store_true", default=False)
    analyze.add_argument("--pair-count", type=int, default=0)
    analyze.add_argument("--train-pair-count", type=int, default=DEFAULT_TRAIN_PAIR_COUNT)
    analyze.add_argument("--validation-pair-count", type=int, default=DEFAULT_VALIDATION_PAIR_COUNT)
    analyze.add_argument("--test-pair-count", type=int, default=DEFAULT_TEST_PAIR_COUNT)
    analyze.add_argument("--pair-search-limit", type=int, default=5000)
    analyze.add_argument("--batch-pair-count", type=int, default=DEFAULT_BATCH_PAIR_COUNT)
    analyze.add_argument("--split-seed", type=int, default=17)
    analyze.add_argument("--control-seed", type=int, default=17)
    analyze.add_argument("--sentence-score-mode", type=str, default="mean_logprob", choices=["mean_logprob", "sum_logprob", "geomean_prob", "sentence_prob"])
    analyze.add_argument("--max-input-tokens", type=str, default="auto")
    analyze.add_argument(
        "--offload-attr-tensors-to-cpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Move saved attribution activations/gradients to CPU between the trace and reduction step.",
    )
    analyze.add_argument("--min-commitment-delta", type=float, default=0.3)
    analyze.add_argument("--min-commitment-deception-rate", type=float, default=0.0)
    analyze.add_argument("--min-donor-clarity-score", type=float, default=0.0)
    analyze.add_argument("--min-num-valid", type=int, default=11)
    analyze.add_argument("--min-sentence-alpha-words", type=int, default=4)
    analyze.add_argument("--exclude-multiline-sentences", action="store_true", default=True)
    analyze.add_argument("--allow-multiline-sentences", action="store_true", default=False)
    analyze.add_argument("--circuit-select", type=str, default="positive", choices=["positive", "abs"])
    analyze.add_argument("--percent-reduction-threshold", type=float, default=50.0)
    analyze.add_argument("--max-circuit-size", type=str, default="auto")
    analyze.add_argument("--candidate-circuit-sizes", type=str, default="")
    analyze.add_argument("--flattening-percent-point-tol", type=float, default=1.0)
    analyze.add_argument("--control-circuit-count", type=int, default=DEFAULT_CONTROL_CIRCUIT_COUNT)
    analyze.add_argument(
        "--averaged-shuffled-deceptive-donor-count",
        dest="mean_shuffled_donor_count",
        type=int,
        default=DEFAULT_MEAN_SHUFFLED_DONOR_COUNT,
        help=(
            "Number of donor examples to average for the averaged shuffled deceptive donor "
            "and averaged shuffled difference donor controls."
        ),
    )
    analyze.add_argument(
        "--mean-shuffled-donor-count",
        dest="mean_shuffled_donor_count",
        type=int,
        default=DEFAULT_MEAN_SHUFFLED_DONOR_COUNT,
        help=argparse.SUPPRESS,
    )
    analyze.add_argument("--validation-max-chunks", type=int, default=0)
    analyze.add_argument("--test-max-chunks", type=int, default=0)
    analyze.add_argument("--cross-corpus-envs", type=str, default=",".join(DEFAULT_CROSS_CORPUS_ENVS))
    analyze.add_argument("--cross-corpus-pair-count", type=int, default=3)
    analyze.add_argument("--cross-corpus-search-limit", type=int, default=128)
    analyze.add_argument("--cross-corpus-max-chunks", type=int, default=0)
    analyze.add_argument("--output-base", type=str, default=str(DEFAULT_OUTPUT_BASE))
    analyze.add_argument("--run-tag", type=str, default="")
    analyze.add_argument("--disable-tqdm", action="store_true", default=False)

    vectors = subparsers.add_parser(
        "vectors",
        help="Load a saved analysis run and export an alternate steering vector bundle for chosen split(s).",
    )
    add_common_model_args(vectors, model_id_default="", dtype_default="")
    vectors.add_argument("--analysis-dir", type=str, required=True)
    vectors.add_argument("--vector-source-splits", type=str, default="train")
    vectors.add_argument("--batch-pair-count", type=int, default=0)
    vectors.add_argument("--max-input-tokens", type=str, default="")
    vectors.add_argument("--output-dir", type=str, default="")
    vectors.add_argument("--vector-tag", type=str, default="")

    steering = subparsers.add_parser(
        "steering",
        aliases=["generate"],
        help="Load saved steering vectors and run shardable counterfactual steering generations.",
    )
    add_common_model_args(steering, model_id_default="", dtype_default="")
    steering.add_argument("--analysis-dir", type=str, default="")
    steering.add_argument("--vector-path", type=str, default="")
    steering.add_argument("--prompt-environments", type=str, default="")
    steering.add_argument("--prompt-splits", type=str, default="test")
    steering.add_argument("--prompts-per-environment", type=int, default=DEFAULT_GENERATION_PROMPTS_PER_ENV)
    steering.add_argument("--prompt-dataset-root", type=str, default="")
    steering.add_argument("--prompt-dataset-model-id", type=str, default="")
    steering.add_argument("--prompt-pair-cache-path", type=str, default="")
    steering.add_argument("--refresh-prompt-pair-cache", action="store_true", default=False)
    steering.add_argument("--prompt-min-commitment-delta", type=float, default=0.3)
    steering.add_argument("--prompt-min-prefix-deception-rate", type=float, default=0.0)
    steering.add_argument("--pair-indices", type=str, default="")
    steering.add_argument("--prompt-jsonl", type=str, default="")
    steering.add_argument("--prompt-text", type=str, default="")
    steering.add_argument("--required-rank", type=int, default=None)
    steering.add_argument("--max-prompts", type=int, default=0)
    steering.add_argument("--batch-size", type=int, default=DEFAULT_STEERING_BATCH_SIZE)
    steering.add_argument("--samples-per-prompt", type=int, default=DEFAULT_STEERING_SAMPLES_PER_PROMPT)
    steering.add_argument("--alpha", type=float, default=DEFAULT_STEERING_ALPHA)
    steering.add_argument("--steering-position", type=str, default=DEFAULT_STEERING_POSITION, choices=["last", "all"])
    steering.add_argument(
        "--steering-max-decode-tokens",
        type=int,
        default=DEFAULT_STEERING_MAX_DECODE_TOKENS,
        help="Apply steering over this many generated-token steps after the prefix (default: 10).",
    )
    steering.add_argument("--max-model-length", type=int, default=10000)
    steering.add_argument("--max-new-tokens", type=int, default=DEFAULT_STEERING_MAX_NEW_TOKENS)
    steering.add_argument("--temperature", type=float, default=DEFAULT_STEERING_TEMPERATURE)
    steering.add_argument("--top-p", type=float, default=DEFAULT_STEERING_TOP_P)
    steering.add_argument("--base-seed", type=int, default=23)
    steering.add_argument("--include-baseline", action="store_true", default=False)
    steering.add_argument("--early-stop-on-valid-json", action="store_true", default=False)
    steering.add_argument("--early-stop-check-interval", type=int, default=DEFAULT_STEERING_EARLY_STOP_CHECK_INTERVAL)
    steering.add_argument("--early-stop-min-new-tokens", type=int, default=DEFAULT_STEERING_EARLY_STOP_MIN_NEW_TOKENS)
    steering.add_argument("--num-shards", type=int, default=1)
    steering.add_argument("--shard-index", type=int, default=0)
    steering.add_argument("--output-dir", type=str, default="")
    steering.add_argument("--run-tag", type=str, default="")
    steering.add_argument("--disable-tqdm", action="store_true", default=False)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in {"analyze", "vectors", "steering", "generate"}:
        argv = ["analyze", *argv]
    args = parser.parse_args(argv)

    if getattr(args, "allow_multiline_sentences", False):
        args.exclude_multiline_sentences = False

    if args.command == "analyze":
        run_analysis(args)
    elif args.command == "vectors":
        run_vector_export(args)
    elif args.command in {"steering", "generate"}:
        run_steering(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
