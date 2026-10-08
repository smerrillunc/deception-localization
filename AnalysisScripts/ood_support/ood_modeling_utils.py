from __future__ import annotations

import gc
import json
import math
import re
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


AGGREGATE_STAT_NAMES = (
    "mean",
    "std",
    "min",
    "max",
    "median",
    "first",
    "last",
    "range",
    "last_minus_first",
    "slope",
    "upper_minus_lower_mean",
)

LAYER_SUFFIX_RE = re.compile(r"_l(\d+)$")
BLOCK_SUFFIX_RE = re.compile(r"_b(\d+)$")
AGGREGATE_FEATURE_RE = re.compile(
    r"__(layer_(?:"
    + "|".join(re.escape(stat_name) for stat_name in AGGREGATE_STAT_NAMES)
    + r"))$"
)
CHANGE_PREFIXES = ("delta_", "devrun_", "logratio_prev_", "slope3_", "min_gap_", "max_gap_")
NORMALIZED_PREFIXES = ("z_", "pct_")
CONCENTRATION_PREFIXES = (
    "entropy_",
    "top1_",
    "top5_",
    "top10_",
    "herfindahl_",
    "effective_support_",
)

BASE_METADATA_COLUMNS = ["dataset", "example_id", "sentence_idx", "deception_rate"]
DERIVED_TARGET_COLUMNS = ["label_binary", "prev_deception_rate", "delta_deception_rate", "delta_label"]
DELTA_SPIKE_THRESHOLDS = (0.2, 0.3)
COMMITMENT_NON_FEATURE_COLUMNS = {
    "example_id",
    "sentence_idx",
    "sentence_text",
    "deception_rate",
    # These directly determine deception_rate and would leak the target.
    "num_truthful",
    "num_valid",
    # Char offsets are bookkeeping rather than model features.
    "raw_start",
    "raw_end",
    "full_start",
    "full_end",
}


@dataclass(frozen=True)
class ScenarioConfig:
    key: str
    title: str
    task: str
    target_col: str
    event_col: str
    decision_threshold: float
    decision_label: str
    score_label: str
    objective_label: str
    notes: str


@dataclass(frozen=True)
class PreparedScenario:
    config: ScenarioConfig
    env_meta: OrderedDict[str, pd.DataFrame]
    env_targets: OrderedDict[str, np.ndarray]
    env_events: OrderedDict[str, np.ndarray]


def notebook_display(value: Any) -> None:
    try:
        from IPython.display import display

        display(value)
    except Exception:
        print(value)


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


def feature_root_without_layer(feature_name: str) -> str:
    return re.sub(r"_l\d+$", "", str(feature_name))


def split_aggregate_feature(feature_name: str) -> tuple[str, str]:
    match = AGGREGATE_FEATURE_RE.search(str(feature_name))
    if match is None:
        block_match = BLOCK_SUFFIX_RE.search(str(feature_name))
        if block_match is None:
            return str(feature_name), ""
        return str(feature_name)[: block_match.start()], f"block_b{block_match.group(1)}"
    return str(feature_name)[: match.start()], match.group(1)


def classify_feature_family(feature_name: str) -> str:
    root_name, _ = split_aggregate_feature(feature_name)
    if root_name.startswith(CHANGE_PREFIXES):
        return "change"
    if root_name.startswith(NORMALIZED_PREFIXES):
        return "normalized"
    if root_name.startswith("geom_"):
        return "geometry"
    if root_name.startswith("dyn_"):
        return "dynamics"
    if root_name.endswith("_token_count") or root_name in {
        "token_count",
        "context_token_count",
        "prompt_token_count",
        "raw_text_context_token_count",
        "available_token_count",
        "prior_all_token_count",
        "previous_sentence_token_count",
        "recent_token_count",
        "early_token_count",
    }:
        return "token_count"
    if root_name in {"start_token", "end_token"}:
        return "token_position"
    if root_name.startswith("act_") or "_act_" in root_name:
        return "activation"
    if root_name.startswith("g_"):
        return "grounding"
    if root_name.startswith(CONCENTRATION_PREFIXES):
        return "concentration"
    return "other"


def safe_float(value: Any) -> float:
    if value is None or pd.isna(value):
        return float("nan")
    return float(value)


def safe_metric_mean(values: pd.Series | np.ndarray | list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    if np.all(~np.isfinite(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def safe_metric_min(values: pd.Series | np.ndarray | list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.min(finite))


def safe_metric_std(values: pd.Series | np.ndarray | list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.std(finite))


def rankable_metric(value: Any) -> float:
    if value is None or pd.isna(value):
        return float("-inf")
    return float(value)


def threshold_suffix(value: float) -> str:
    return str(value).replace("-", "neg_").replace(".", "_")


def delta_abs_spike_col(threshold: float) -> str:
    return f"delta_abs_gt_{threshold_suffix(threshold)}"


def delta_sign_spike_col(threshold: float) -> str:
    return f"delta_sign_pos_if_abs_gt_{threshold_suffix(threshold)}"


def annotate_delta_spike_targets(
    df: pd.DataFrame,
    *,
    thresholds: tuple[float, ...] = DELTA_SPIKE_THRESHOLDS,
) -> pd.DataFrame:
    delta = pd.to_numeric(df["delta_deception_rate"], errors="coerce")
    abs_delta = delta.abs()
    valid_delta = delta.notna()

    for threshold in thresholds:
        spike_mask = valid_delta & (abs_delta > float(threshold))
        df[delta_abs_spike_col(threshold)] = np.where(valid_delta, spike_mask.astype(np.int8), np.nan)
        df[delta_sign_spike_col(threshold)] = np.where(
            spike_mask,
            (delta > 0.0).astype(np.int8),
            np.nan,
        )
    return df


def choose_decision_threshold(
    y_event: np.ndarray,
    score: np.ndarray,
    *,
    default_threshold: float,
    mode: str = "fixed",
) -> float:
    y_event = np.asarray(y_event, dtype=np.int8)
    score = np.asarray(score, dtype=np.float32)
    valid = np.isfinite(score)
    y_valid = y_event[valid]
    score_valid = score[valid]

    if score_valid.size == 0:
        return float(default_threshold)

    if mode == "fixed":
        return float(default_threshold)

    if mode == "train_prevalence_match":
        positive_rate = float(np.mean(y_valid))
        if positive_rate <= 0.0:
            return float(np.inf)
        if positive_rate >= 1.0:
            return float(-np.inf)
        return float(np.quantile(score_valid, 1.0 - positive_rate))

    if mode == "train_balanced_accuracy":
        candidate_thresholds = np.unique(score_valid)
        best_threshold = float(default_threshold)
        best_score = float("-inf")
        for threshold in candidate_thresholds:
            predicted = (score_valid >= float(threshold)).astype(np.int8)
            if np.unique(y_valid).size < 2:
                metric_value = float(np.mean(predicted == y_valid))
            else:
                metric_value = float(balanced_accuracy_score(y_valid, predicted))
            if metric_value > best_score:
                best_score = metric_value
                best_threshold = float(threshold)
        return best_threshold

    raise ValueError(f"Unsupported decision-threshold mode: {mode}")


def build_scenario_catalog() -> OrderedDict[str, ScenarioConfig]:
    scenarios = [
            (
                "raw_rate",
                ScenarioConfig(
                    key="raw_rate",
                    title="Raw Deception Rate Regression",
                    task="regression",
                    target_col="deception_rate",
                    event_col="label_binary",
                    decision_threshold=0.5,
                    decision_label="Predicted deception_rate > 0.5",
                    score_label="Predicted deception_rate",
                    objective_label="Min held-out AUROC for deception_rate > 0.5",
                    notes=(
                        "Train a regression model on raw deception_rate, then evaluate AUROC against "
                        "the binary event deception_rate > 0.5, ranking feature subsets by worst-case "
                        "held-out AUROC."
                    ),
                ),
            ),
            (
                "binary_rate",
                ScenarioConfig(
                    key="binary_rate",
                    title="Binary Deception Label Classification",
                    task="classification",
                    target_col="label_binary",
                    event_col="label_binary",
                    decision_threshold=0.5,
                    decision_label="Predicted label for deception_rate > 0.5",
                    score_label="Predicted probability of deception_rate > 0.5",
                    objective_label="Min held-out AUROC for deception_rate > 0.5",
                    notes=(
                        "Train a logistic classifier directly on the binary label deception_rate > 0.5, "
                        "ranking feature subsets by worst-case held-out AUROC."
                    ),
                ),
            ),
            (
                "delta_rate",
                ScenarioConfig(
                    key="delta_rate",
                    title="Consecutive-Sentence Delta Regression",
                    task="regression",
                    target_col="delta_deception_rate",
                    event_col="delta_label",
                    decision_threshold=0.0,
                    decision_label="Predicted positive change in deception_rate",
                    score_label="Predicted delta_deception_rate",
                    objective_label="Min held-out AUROC for delta_deception_rate > 0",
                    notes=(
                        "Train a regression model on the consecutive-sentence change in deception_rate, "
                        "then evaluate AUROC against the event delta_deception_rate > 0, ranking feature "
                        "subsets by worst-case held-out AUROC."
                    ),
                ),
            ),
        ]

    for threshold in DELTA_SPIKE_THRESHOLDS:
        abs_col = delta_abs_spike_col(threshold)
        sign_col = delta_sign_spike_col(threshold)
        suffix = threshold_suffix(threshold)
        scenarios.extend(
            [
                (
                    f"delta_spike_abs_{suffix}",
                    ScenarioConfig(
                        key=f"delta_spike_abs_{suffix}",
                        title=f"Large Delta Spike Detection |delta| > {threshold:.1f}",
                        task="classification",
                        target_col=abs_col,
                        event_col=abs_col,
                        decision_threshold=0.5,
                        decision_label=f"Predicted |delta_deception_rate| > {threshold:.1f}",
                        score_label=f"Predicted probability of |delta_deception_rate| > {threshold:.1f}",
                        objective_label=f"Min held-out AUROC for |delta_deception_rate| > {threshold:.1f}",
                        notes=(
                            "Train a logistic classifier directly on the large-spike event "
                            f"|delta_deception_rate| > {threshold:.1f}, ranking feature subsets by "
                            "worst-case held-out AUROC."
                        ),
                    ),
                ),
                (
                    f"delta_spike_sign_{suffix}",
                    ScenarioConfig(
                        key=f"delta_spike_sign_{suffix}",
                        title=f"Large Delta Spike Direction |delta| > {threshold:.1f}",
                        task="classification",
                        target_col=sign_col,
                        event_col=sign_col,
                        decision_threshold=0.5,
                        decision_label=f"Predicted positive direction among |delta_deception_rate| > {threshold:.1f}",
                        score_label=(
                            "Predicted probability that delta_deception_rate > 0 "
                            f"among rows with |delta_deception_rate| > {threshold:.1f}"
                        ),
                        objective_label=(
                            "Min held-out AUROC for delta_deception_rate > 0 "
                            f"among rows with |delta_deception_rate| > {threshold:.1f}"
                        ),
                        notes=(
                            "Restrict to rows with a large consecutive-sentence spike, then train a logistic "
                            f"classifier to predict the direction of the spike when |delta_deception_rate| > {threshold:.1f}. "
                            "Feature subsets are ranked by worst-case held-out AUROC."
                        ),
                    ),
                ),
            ]
        )

    return OrderedDict(scenarios)


def ordered_feature_roots_for_path(path: Path) -> OrderedDict[str, list[str]]:
    parquet_file = pq.ParquetFile(path)
    ordered: OrderedDict[str, list[str]] = OrderedDict()
    for column_name in parquet_file.schema_arrow.names:
        if LAYER_SUFFIX_RE.search(column_name) is None:
            continue
        feature_root = feature_root_without_layer(column_name)
        ordered.setdefault(feature_root, []).append(column_name)
    for feature_root, columns in ordered.items():
        ordered[feature_root] = sorted(
            columns,
            key=lambda column_name: int(LAYER_SUFFIX_RE.search(column_name).group(1)),
        )
    return ordered


def build_common_layer_roots(
    feature_paths: OrderedDict[str, Path],
) -> tuple[OrderedDict[str, list[str]], pd.DataFrame]:
    per_env_roots: OrderedDict[str, OrderedDict[str, list[str]]] = OrderedDict()
    common_roots: set[str] | None = None

    for env_name, feature_path in feature_paths.items():
        root_map = ordered_feature_roots_for_path(feature_path)
        per_env_roots[env_name] = root_map
        if common_roots is None:
            common_roots = set(root_map)
        else:
            common_roots &= set(root_map)

    if not common_roots:
        raise ValueError("No shared layer-wise feature roots were found across the requested environments.")

    first_env_name = next(iter(per_env_roots))
    ordered_common_roots = OrderedDict(
        (feature_root, per_env_roots[first_env_name][feature_root])
        for feature_root in per_env_roots[first_env_name]
        if feature_root in common_roots
    )

    aggregate_lookup_rows: list[dict[str, Any]] = []
    for feature_root, columns in ordered_common_roots.items():
        aggregate_lookup_rows.append(
            {
                "feature_root": feature_root,
                "family": classify_feature_family(feature_root),
                "layer_count": int(len(columns)),
                "first_layer_col": columns[0],
                "last_layer_col": columns[-1],
            }
        )
    aggregate_lookup_df = pd.DataFrame(aggregate_lookup_rows)
    return ordered_common_roots, aggregate_lookup_df


def ordered_commitment_feature_columns_for_path(
    path: Path,
    *,
    excluded_columns: set[str] | None = None,
) -> list[str]:
    excluded = set(COMMITMENT_NON_FEATURE_COLUMNS)
    if excluded_columns:
        excluded |= set(excluded_columns)

    parquet_file = pq.ParquetFile(path)
    ordered: list[str] = []
    for field in parquet_file.schema_arrow:
        if field.name in excluded:
            continue
        if pd.api.types.is_numeric_dtype(field.type.to_pandas_dtype()):
            ordered.append(str(field.name))
    return ordered


def build_common_commitment_feature_catalog(
    feature_paths: OrderedDict[str, Path],
    *,
    excluded_columns: set[str] | None = None,
) -> tuple[list[str], pd.DataFrame]:
    per_env_features: OrderedDict[str, list[str]] = OrderedDict()
    common_features: set[str] | None = None

    for env_name, feature_path in feature_paths.items():
        ordered_features = ordered_commitment_feature_columns_for_path(
            feature_path,
            excluded_columns=excluded_columns,
        )
        per_env_features[env_name] = ordered_features
        if common_features is None:
            common_features = set(ordered_features)
        else:
            common_features &= set(ordered_features)

    if not common_features:
        raise ValueError("No shared commitment-prefix feature columns were found across the requested environments.")

    first_env_name = next(iter(per_env_features))
    ordered_common_features = [
        feature_name
        for feature_name in per_env_features[first_env_name]
        if feature_name in common_features
    ]

    lookup_rows: list[dict[str, Any]] = []
    for feature_name in ordered_common_features:
        feature_root, feature_variant = split_aggregate_feature(feature_name)
        block_match = BLOCK_SUFFIX_RE.search(feature_name)
        lookup_rows.append(
            {
                "feature": feature_name,
                "feature_root": feature_root,
                "feature_variant": feature_variant,
                "family": classify_feature_family(feature_name),
                "block_index": (
                    int(block_match.group(1))
                    if block_match is not None
                    else pd.NA
                ),
            }
        )
    lookup_df = pd.DataFrame(lookup_rows)
    return ordered_common_features, lookup_df


def _numeric_chunk_to_array(chunk_df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    numeric_chunk = chunk_df.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    numeric_chunk = numeric_chunk.replace([np.inf, -np.inf], np.nan)
    return numeric_chunk.to_numpy(dtype=np.float32, copy=False)


def _compute_layer_stat_map(feature_array: np.ndarray) -> dict[str, np.ndarray]:
    feature_array = np.asarray(feature_array, dtype=np.float32)
    n_rows, n_layers = feature_array.shape

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        stat_map = {
            "mean": np.nanmean(feature_array, axis=1, dtype=np.float32),
            "std": np.nanstd(feature_array, axis=1, dtype=np.float32),
            "min": np.nanmin(feature_array, axis=1),
            "max": np.nanmax(feature_array, axis=1),
            "median": np.nanmedian(feature_array, axis=1),
        }

    stat_map["first"] = feature_array[:, 0].astype(np.float32, copy=False)
    stat_map["last"] = feature_array[:, -1].astype(np.float32, copy=False)
    stat_map["range"] = np.asarray(stat_map["max"] - stat_map["min"], dtype=np.float32)
    stat_map["last_minus_first"] = np.asarray(
        stat_map["last"] - stat_map["first"],
        dtype=np.float32,
    )

    if n_layers < 2:
        stat_map["slope"] = np.full(n_rows, np.nan, dtype=np.float32)
        stat_map["upper_minus_lower_mean"] = np.full(n_rows, np.nan, dtype=np.float32)
        return stat_map

    layer_positions = np.linspace(0.0, 1.0, n_layers, dtype=np.float32)
    finite_mask = np.isfinite(feature_array)
    count = finite_mask.sum(axis=1).astype(np.float32)
    sum_x = np.sum(finite_mask * layer_positions[None, :], axis=1, dtype=np.float32)
    sum_x2 = np.sum(finite_mask * (layer_positions[None, :] ** 2), axis=1, dtype=np.float32)
    sum_y = np.nansum(feature_array, axis=1, dtype=np.float32)
    sum_xy = np.nansum(feature_array * layer_positions[None, :], axis=1, dtype=np.float32)
    slope_denominator = count * sum_x2 - (sum_x**2)
    slope = np.full(n_rows, np.nan, dtype=np.float32)
    valid_slope = (count >= 2.0) & np.isfinite(slope_denominator) & (np.abs(slope_denominator) > 1e-8)
    slope[valid_slope] = (
        (count[valid_slope] * sum_xy[valid_slope]) - (sum_x[valid_slope] * sum_y[valid_slope])
    ) / slope_denominator[valid_slope]
    stat_map["slope"] = slope

    split_idx = n_layers // 2
    if split_idx == 0 or split_idx == n_layers:
        stat_map["upper_minus_lower_mean"] = np.full(n_rows, np.nan, dtype=np.float32)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            lower_mean = np.nanmean(feature_array[:, :split_idx], axis=1, dtype=np.float32)
            upper_mean = np.nanmean(feature_array[:, split_idx:], axis=1, dtype=np.float32)
        stat_map["upper_minus_lower_mean"] = np.asarray(upper_mean - lower_mean, dtype=np.float32)

    return stat_map


def build_aggregate_frame(
    feature_path: Path,
    env_name: str,
    layer_root_map: OrderedDict[str, list[str]],
    *,
    batch_root_count: int,
    rate_threshold: float,
    aggregate_specs: OrderedDict[str, str],
) -> pd.DataFrame:
    metadata_df = pd.read_parquet(feature_path, columns=["example_id", "sentence_idx", "deception_rate"]).copy()
    metadata_df["__row_id__"] = np.arange(len(metadata_df), dtype=np.int64)
    metadata_df["example_id"] = metadata_df["example_id"].astype(str)
    metadata_df["sentence_idx"] = pd.to_numeric(metadata_df["sentence_idx"], errors="coerce")
    metadata_df["deception_rate"] = pd.to_numeric(metadata_df["deception_rate"], errors="coerce")
    metadata_df = metadata_df.replace([np.inf, -np.inf], np.nan)
    metadata_df = metadata_df.dropna(subset=["example_id", "sentence_idx", "deception_rate"]).copy()
    metadata_df["sentence_idx"] = metadata_df["sentence_idx"].astype(int)
    metadata_df["deception_rate"] = metadata_df["deception_rate"].astype(np.float32)
    metadata_df = metadata_df.sort_values(["example_id", "sentence_idx", "__row_id__"]).reset_index(drop=True)

    row_ids = metadata_df["__row_id__"].to_numpy(dtype=np.int64, copy=False)
    ordered_roots = list(layer_root_map)
    aggregate_arrays: dict[str, np.ndarray] = {}

    for start_idx in range(0, len(ordered_roots), batch_root_count):
        batch_roots = ordered_roots[start_idx : start_idx + batch_root_count]
        batch_columns = [column_name for feature_root in batch_roots for column_name in layer_root_map[feature_root]]
        batch_df = pd.read_parquet(feature_path, columns=batch_columns)
        batch_df = batch_df.iloc[row_ids].reset_index(drop=True)

        for feature_root in batch_roots:
            feature_columns = layer_root_map[feature_root]
            feature_array = _numeric_chunk_to_array(batch_df, feature_columns)
            stats_map = _compute_layer_stat_map(feature_array)
            for aggregate_name in aggregate_specs:
                if aggregate_name not in stats_map:
                    raise KeyError(
                        f"Unsupported aggregate_name={aggregate_name!r}. "
                        f"Available stats: {sorted(stats_map)}"
                    )
                aggregate_arrays[f"{feature_root}__layer_{aggregate_name}"] = np.asarray(
                    stats_map[aggregate_name], dtype=np.float32
                )

        del batch_df
        gc.collect()

    aggregate_df = pd.DataFrame(aggregate_arrays)
    out = pd.concat(
        [
            pd.DataFrame(
                {
                    "dataset": env_name,
                    "example_id": metadata_df["example_id"].to_numpy(copy=False),
                    "sentence_idx": metadata_df["sentence_idx"].to_numpy(copy=False),
                    "deception_rate": metadata_df["deception_rate"].to_numpy(copy=False),
                }
            ),
            aggregate_df,
        ],
        axis=1,
    )

    out["label_binary"] = (out["deception_rate"] > float(rate_threshold)).astype(np.int8)
    out["prev_deception_rate"] = (
        out.groupby("example_id", sort=False)["deception_rate"].shift(1).astype(np.float32)
    )
    out["delta_deception_rate"] = (out["deception_rate"] - out["prev_deception_rate"]).astype(np.float32)
    out["delta_label"] = np.where(
        out["delta_deception_rate"].notna(),
        (out["delta_deception_rate"] > 0.0).astype(np.int8),
        np.nan,
    )
    return out


def summarize_aggregate_frame(df: pd.DataFrame) -> dict[str, Any]:
    feature_names = [column_name for column_name in df.columns if AGGREGATE_FEATURE_RE.search(column_name)]
    delta_valid = df["delta_deception_rate"].notna()
    return {
        "environment": str(df["dataset"].iloc[0]),
        "rows": int(len(df)),
        "examples": int(df["example_id"].nunique()),
        "aggregate_feature_count": int(len(feature_names)),
        "binary_positive_rate": safe_float(df["label_binary"].mean()),
        "delta_rows": int(delta_valid.sum()),
        "delta_positive_rate": safe_float(df.loc[delta_valid, "delta_label"].mean()),
    }


def build_commitment_feature_frame(
    feature_path: Path,
    env_name: str,
    feature_names: list[str],
    *,
    rate_threshold: float,
) -> pd.DataFrame:
    requested_columns = ["example_id", "sentence_idx", "deception_rate", *feature_names]
    df = pd.read_parquet(feature_path, columns=requested_columns).copy()
    df["__row_id__"] = np.arange(len(df), dtype=np.int64)
    df["example_id"] = df["example_id"].astype(str)
    df["sentence_idx"] = pd.to_numeric(df["sentence_idx"], errors="coerce")
    df["deception_rate"] = pd.to_numeric(df["deception_rate"], errors="coerce")

    for feature_name in feature_names:
        df[feature_name] = pd.to_numeric(df[feature_name], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["example_id", "sentence_idx", "deception_rate"]).copy()
    df["sentence_idx"] = df["sentence_idx"].astype(int)
    df["deception_rate"] = df["deception_rate"].astype(np.float32)
    df = df.sort_values(["example_id", "sentence_idx", "__row_id__"]).reset_index(drop=True)

    out = pd.concat(
        [
            pd.DataFrame(
                {
                    "dataset": env_name,
                    "example_id": df["example_id"].to_numpy(copy=False),
                    "sentence_idx": df["sentence_idx"].to_numpy(copy=False),
                    "deception_rate": df["deception_rate"].to_numpy(copy=False),
                }
            ),
            df.loc[:, feature_names].reset_index(drop=True),
        ],
        axis=1,
    )

    out["label_binary"] = (out["deception_rate"] > float(rate_threshold)).astype(np.int8)
    out["prev_deception_rate"] = (
        out.groupby("example_id", sort=False)["deception_rate"].shift(1).astype(np.float32)
    )
    out["delta_deception_rate"] = (out["deception_rate"] - out["prev_deception_rate"]).astype(np.float32)
    out["delta_label"] = np.where(
        out["delta_deception_rate"].notna(),
        (out["delta_deception_rate"] > 0.0).astype(np.int8),
        np.nan,
    )
    return out


def summarize_commitment_feature_frame(df: pd.DataFrame, feature_names: list[str]) -> dict[str, Any]:
    delta_valid = df["delta_deception_rate"].notna()
    return {
        "environment": str(df["dataset"].iloc[0]),
        "rows": int(len(df)),
        "examples": int(df["example_id"].nunique()),
        "feature_count": int(len(feature_names)),
        "binary_positive_rate": safe_float(df["label_binary"].mean()),
        "delta_rows": int(delta_valid.sum()),
        "delta_positive_rate": safe_float(df.loc[delta_valid, "delta_label"].mean()),
    }


def prepare_aggregate_envs(
    feature_paths: OrderedDict[str, Path],
    *,
    cache_dir: Path,
    force_rebuild: bool = False,
    batch_root_count: int = 24,
    rate_threshold: float = 0.5,
    aggregate_specs: OrderedDict[str, str] | None = None,
) -> tuple[OrderedDict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, list[str]]:
    if aggregate_specs is None:
        aggregate_specs = OrderedDict(
            [
                ("mean", "mean"),
                ("std", "std"),
                ("min", "min"),
                ("max", "max"),
                ("median", "median"),
            ]
        )

    missing_paths = {env_name: path for env_name, path in feature_paths.items() if not path.exists()}
    if missing_paths:
        missing_text = "\n".join(f"- {env_name}: {path}" for env_name, path in missing_paths.items())
        raise FileNotFoundError(f"Missing attention_features2 parquet files:\n{missing_text}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    common_root_map, aggregate_lookup_df = build_common_layer_roots(feature_paths)
    aggregate_envs: OrderedDict[str, pd.DataFrame] = OrderedDict()
    summary_rows: list[dict[str, Any]] = []

    for env_name, feature_path in feature_paths.items():
        cache_path = cache_dir / f"{slugify(env_name)}__aggregated.parquet"
        if cache_path.exists() and not force_rebuild:
            env_df = pd.read_parquet(cache_path)
        else:
            env_df = build_aggregate_frame(
                feature_path,
                env_name,
                common_root_map,
                batch_root_count=int(batch_root_count),
                rate_threshold=float(rate_threshold),
                aggregate_specs=aggregate_specs,
            )
            env_df.to_parquet(cache_path, index=False)
        env_df["dataset"] = env_df["dataset"].astype(str)
        env_df["example_id"] = env_df["example_id"].astype(str)
        env_df["sentence_idx"] = pd.to_numeric(env_df["sentence_idx"], errors="coerce").astype(int)
        env_df["deception_rate"] = pd.to_numeric(env_df["deception_rate"], errors="coerce").astype(np.float32)
        env_df["label_binary"] = pd.to_numeric(env_df["label_binary"], errors="coerce").astype(np.int8)
        env_df = annotate_delta_spike_targets(env_df)
        aggregate_envs[env_name] = env_df
        summary_rows.append(summarize_aggregate_frame(env_df))

    common_feature_names = sorted(
        set.intersection(
            *(set(column_name for column_name in df.columns if AGGREGATE_FEATURE_RE.search(column_name)) for df in aggregate_envs.values())
        )
    )

    dataset_summary_df = pd.DataFrame(summary_rows).sort_values("environment").reset_index(drop=True)
    usable_feature_names: list[str] = []
    dropped_all_nan_features: list[str] = []
    for feature_name in common_feature_names:
        if any(aggregate_envs[env_name][feature_name].isna().all() for env_name in aggregate_envs):
            dropped_all_nan_features.append(feature_name)
        else:
            usable_feature_names.append(feature_name)

    if dropped_all_nan_features:
        warnings.warn(
            "Dropping aggregate features that are all-NaN in at least one environment: "
            f"{len(dropped_all_nan_features)} removed, {len(usable_feature_names)} remain.",
            stacklevel=2,
        )

    common_feature_names = usable_feature_names
    return aggregate_envs, dataset_summary_df, aggregate_lookup_df, common_feature_names


def prepare_commitment_feature_envs(
    feature_paths: OrderedDict[str, Path],
    *,
    cache_dir: Path,
    force_rebuild: bool = False,
    rate_threshold: float = 0.5,
    excluded_columns: set[str] | None = None,
) -> tuple[OrderedDict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, list[str]]:
    missing_paths = {env_name: path for env_name, path in feature_paths.items() if not path.exists()}
    if missing_paths:
        missing_text = "\n".join(f"- {env_name}: {path}" for env_name, path in missing_paths.items())
        raise FileNotFoundError(f"Missing commitment_prefix_features parquet files:\n{missing_text}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    common_feature_names, feature_lookup_df = build_common_commitment_feature_catalog(
        feature_paths,
        excluded_columns=excluded_columns,
    )

    feature_envs: OrderedDict[str, pd.DataFrame] = OrderedDict()
    summary_rows: list[dict[str, Any]] = []

    for env_name, feature_path in feature_paths.items():
        cache_path = cache_dir / f"{slugify(env_name)}__commitment_features.parquet"
        if cache_path.exists() and not force_rebuild:
            env_df = pd.read_parquet(cache_path)
        else:
            env_df = build_commitment_feature_frame(
                feature_path,
                env_name,
                common_feature_names,
                rate_threshold=float(rate_threshold),
            )
            env_df.to_parquet(cache_path, index=False)

        env_df["dataset"] = env_df["dataset"].astype(str)
        env_df["example_id"] = env_df["example_id"].astype(str)
        env_df["sentence_idx"] = pd.to_numeric(env_df["sentence_idx"], errors="coerce").astype(int)
        env_df["deception_rate"] = pd.to_numeric(env_df["deception_rate"], errors="coerce").astype(np.float32)
        env_df["label_binary"] = pd.to_numeric(env_df["label_binary"], errors="coerce").astype(np.int8)
        env_df = annotate_delta_spike_targets(env_df)
        feature_envs[env_name] = env_df
        summary_rows.append(summarize_commitment_feature_frame(env_df, common_feature_names))

    usable_feature_names: list[str] = []
    dropped_all_nan_features: list[str] = []
    for feature_name in common_feature_names:
        if any(feature_envs[env_name][feature_name].isna().all() for env_name in feature_envs):
            dropped_all_nan_features.append(feature_name)
        else:
            usable_feature_names.append(feature_name)

    if dropped_all_nan_features:
        warnings.warn(
            "Dropping commitment features that are all-NaN in at least one environment: "
            f"{len(dropped_all_nan_features)} removed, {len(usable_feature_names)} remain.",
            stacklevel=2,
        )
        feature_lookup_df = feature_lookup_df.loc[
            feature_lookup_df["feature"].isin(usable_feature_names)
        ].reset_index(drop=True)

    dataset_summary_df = pd.DataFrame(summary_rows).sort_values("environment").reset_index(drop=True)
    return feature_envs, dataset_summary_df, feature_lookup_df, usable_feature_names


def build_commitment_feature_spaces(feature_names: list[str]) -> OrderedDict[str, list[str]]:
    ordered = list(dict.fromkeys(feature_names))
    feature_spaces = OrderedDict(
        [
            ("all_features", ordered),
            ("drop_normalized", [name for name in ordered if not name.startswith(NORMALIZED_PREFIXES)]),
            (
                "base_only",
                [
                    name
                    for name in ordered
                    if not name.startswith(("delta_", *NORMALIZED_PREFIXES))
                ],
            ),
            ("delta_only", [name for name in ordered if name.startswith("delta_")]),
        ]
    )
    return OrderedDict((space_name, names) for space_name, names in feature_spaces.items() if names)


def prepare_scenario_data(
    aggregate_envs: OrderedDict[str, pd.DataFrame],
    scenario: ScenarioConfig,
) -> PreparedScenario:
    env_meta: OrderedDict[str, pd.DataFrame] = OrderedDict()
    env_targets: OrderedDict[str, np.ndarray] = OrderedDict()
    env_events: OrderedDict[str, np.ndarray] = OrderedDict()

    for env_name, df in aggregate_envs.items():
        missing_cols = [column_name for column_name in (scenario.target_col, scenario.event_col) if column_name not in df.columns]
        if missing_cols:
            raise KeyError(
                f"Scenario {scenario.key} requires missing columns in {env_name}: {', '.join(missing_cols)}"
            )

        mask = df[scenario.target_col].notna() & df[scenario.event_col].notna()

        meta_cols = BASE_METADATA_COLUMNS + DERIVED_TARGET_COLUMNS
        env_meta[env_name] = df.loc[mask, meta_cols].copy()
        env_targets[env_name] = df.loc[mask, scenario.target_col].to_numpy(dtype=np.float32, copy=True)
        env_events[env_name] = df.loc[mask, scenario.event_col].to_numpy(dtype=np.int8, copy=True)

    return PreparedScenario(
        config=scenario,
        env_meta=env_meta,
        env_targets=env_targets,
        env_events=env_events,
    )


def compute_binary_label_effect_consistency(
    aggregate_envs: OrderedDict[str, pd.DataFrame],
    feature_names: list[str],
    binary_label_name: str = 'label_binary'
) -> pd.DataFrame:
    if not feature_names:
        return pd.DataFrame(
            columns=[
                "feature",
                "feature_root",
                "family",
                "aggregate_name",
                "same_sign_all",
                "sign_direction",
                "min_abs_effect",
                "mean_abs_effect",
                "max_abs_effect",
                "std_effect",
            ]
        )

    env_order = list(aggregate_envs.keys())
    effect_by_env: OrderedDict[str, np.ndarray] = OrderedDict()

    for env_name, env_df in aggregate_envs.items():
        mask = env_df[binary_label_name].notna()
        y_event = env_df.loc[mask, binary_label_name].to_numpy(dtype=np.int8, copy=True)
        if np.unique(y_event).size < 2:
            raise ValueError(
                f"Environment {env_name} does not contain both label_binary classes; "
                "cannot compute cross-dataset consistency."
            )

        x_df = env_df.loc[mask, feature_names].apply(pd.to_numeric, errors="coerce")
        x_df = x_df.replace([np.inf, -np.inf], np.nan)
        x_array = x_df.to_numpy(dtype=np.float32, copy=False)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            feature_mean = np.nanmean(x_array, axis=0, dtype=np.float32)
            feature_std = np.nanstd(x_array, axis=0, dtype=np.float32)

        feature_std = np.asarray(feature_std, dtype=np.float32)
        feature_std[np.abs(feature_std) < 1e-8] = np.nan
        z_array = (x_array - feature_mean) / feature_std

        positive_mask = y_event == 1
        negative_mask = y_event == 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            positive_mean = np.nanmean(z_array[positive_mask], axis=0, dtype=np.float32)
            negative_mean = np.nanmean(z_array[negative_mask], axis=0, dtype=np.float32)

        effect_by_env[env_name] = np.asarray(positive_mean - negative_mean, dtype=np.float32)

    out = pd.DataFrame({"feature": list(feature_names)})
    out["feature_root"] = out["feature"].map(lambda feature_name: split_aggregate_feature(feature_name)[0])
    out["family"] = out["feature"].map(classify_feature_family)
    out["aggregate_name"] = out["feature"].map(lambda feature_name: split_aggregate_feature(feature_name)[1])

    effect_cols: list[str] = []
    abs_effect_cols: list[str] = []
    for env_name in env_order:
        env_slug = slugify(env_name)
        effect_col = f"{env_slug}_effect"
        abs_effect_col = f"{env_slug}_abs_effect"
        out[effect_col] = effect_by_env[env_name]
        out[abs_effect_col] = out[effect_col].abs()
        effect_cols.append(effect_col)
        abs_effect_cols.append(abs_effect_col)

    effect_array = out[effect_cols].to_numpy(dtype=np.float32, copy=False)
    abs_effect_array = out[abs_effect_cols].to_numpy(dtype=np.float32, copy=False)

    positive_all = np.all(effect_array > 0.0, axis=1)
    negative_all = np.all(effect_array < 0.0, axis=1)
    out["same_sign_all"] = positive_all | negative_all
    out["sign_direction"] = np.select(
        [positive_all, negative_all],
        ["positive", "negative"],
        default="mixed",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        out["min_abs_effect"] = np.nanmin(abs_effect_array, axis=1)
        out["mean_abs_effect"] = np.nanmean(abs_effect_array, axis=1)
        out["max_abs_effect"] = np.nanmax(abs_effect_array, axis=1)
        out["std_effect"] = np.nanstd(effect_array, axis=1)

    if "CarSales" in env_order:
        carsales_slug = slugify("CarSales")
        carsales_effect_col = f"{carsales_slug}_effect"
        carsales_abs_effect_col = f"{carsales_slug}_abs_effect"
        other_effect_cols = [column_name for column_name in effect_cols if column_name != carsales_effect_col]
        other_abs_effect_cols = [column_name for column_name in abs_effect_cols if column_name != carsales_abs_effect_col]

        out["carsales_effect"] = out[carsales_effect_col]
        out["carsales_abs_effect"] = out[carsales_abs_effect_col]
        out["carsales_vs_other_abs_ratio"] = out["carsales_abs_effect"] / out[other_abs_effect_cols].mean(axis=1)

        other_positive = np.all(out[other_effect_cols].to_numpy(dtype=np.float32, copy=False) > 0.0, axis=1)
        other_negative = np.all(out[other_effect_cols].to_numpy(dtype=np.float32, copy=False) < 0.0, axis=1)
        other_same_sign = other_positive | other_negative
        out["other_envs_same_sign"] = other_same_sign
        out["carsales_matches_other_sign"] = np.where(
            other_same_sign,
            np.sign(out["carsales_effect"].to_numpy(dtype=np.float32, copy=False))
            == np.where(other_positive, 1.0, -1.0),
            np.nan,
        )

    return out.sort_values(
        ["same_sign_all", "min_abs_effect", "carsales_abs_effect", "mean_abs_effect", "std_effect"],
        ascending=[False, False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)


def select_rate_consistent_features(
    consistency_df: pd.DataFrame,
    *,
    min_carsales_abs_effect: float = 0.05,
    min_min_abs_effect: float = 0.05,
    require_same_sign: bool = True,
    aggregate_name_whitelist: tuple[str, ...] | None = None,
    excluded_feature_prefixes: tuple[str, ...] | None = None,
    top_k: int | None = None,
) -> pd.DataFrame:
    if consistency_df.empty:
        return consistency_df.copy()

    selected_df = consistency_df.copy()
    if excluded_feature_prefixes:
        excluded_prefixes = tuple(str(prefix) for prefix in excluded_feature_prefixes)
        selected_df = selected_df.loc[~selected_df["feature"].str.startswith(excluded_prefixes)]
    if aggregate_name_whitelist:
        allowed_names = set(aggregate_name_whitelist)
        selected_df = selected_df.loc[selected_df["aggregate_name"].isin(allowed_names)]
    if require_same_sign:
        selected_df = selected_df.loc[selected_df["same_sign_all"]]
    if "carsales_abs_effect" in selected_df.columns:
        selected_df = selected_df.loc[selected_df["carsales_abs_effect"] >= float(min_carsales_abs_effect)]
    selected_df = selected_df.loc[selected_df["min_abs_effect"] >= float(min_min_abs_effect)]

    sort_columns = ["min_abs_effect", "mean_abs_effect", "std_effect"]
    ascending = [False, False, True]
    if "carsales_abs_effect" in selected_df.columns:
        sort_columns.insert(1, "carsales_abs_effect")
        ascending.insert(1, False)

    selected_df = selected_df.sort_values(
        sort_columns,
        ascending=ascending,
        na_position="last",
    ).reset_index(drop=True)
    if top_k is not None:
        selected_df = selected_df.head(int(top_k)).copy()
    selected_df["selected_rank"] = np.arange(1, len(selected_df) + 1, dtype=int)
    return selected_df


def run_rate_consistency_screen(
    *,
    aggregate_envs: OrderedDict[str, pd.DataFrame],
    feature_names: list[str],
    output_root: Path,
    min_carsales_abs_effect: float = 0.05,
    min_min_abs_effect: float = 0.05,
    require_same_sign: bool = True,
    aggregate_name_whitelist: tuple[str, ...] | None = None,
    excluded_feature_prefixes: tuple[str, ...] | None = None,
    top_k: int | None = 64,
) -> dict[str, Any]:
    consistency_df = compute_binary_label_effect_consistency(aggregate_envs, feature_names)
    selected_feature_df = select_rate_consistent_features(
        consistency_df,
        min_carsales_abs_effect=float(min_carsales_abs_effect),
        min_min_abs_effect=float(min_min_abs_effect),
        require_same_sign=bool(require_same_sign),
        aggregate_name_whitelist=aggregate_name_whitelist,
        excluded_feature_prefixes=excluded_feature_prefixes,
        top_k=top_k,
    )
    if selected_feature_df.empty:
        raise ValueError(
            "Rate consistency screen selected zero features. "
            "Try lowering min_carsales_abs_effect / min_min_abs_effect or relaxing require_same_sign."
        )

    output_dir = output_root / "rate_consistency"
    output_dir.mkdir(parents=True, exist_ok=True)
    config_payload = {
        "min_carsales_abs_effect": float(min_carsales_abs_effect),
        "min_min_abs_effect": float(min_min_abs_effect),
        "require_same_sign": bool(require_same_sign),
        "aggregate_name_whitelist": list(aggregate_name_whitelist) if aggregate_name_whitelist else None,
        "excluded_feature_prefixes": list(excluded_feature_prefixes) if excluded_feature_prefixes else None,
        "top_k": None if top_k is None else int(top_k),
        "selected_feature_count": int(len(selected_feature_df)),
        "train_environments": list(aggregate_envs.keys()),
    }
    (output_dir / "config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
    consistency_df.to_csv(output_dir / "consistency_full.csv", index=False)
    selected_feature_df.to_csv(output_dir / "selected_features.csv", index=False)

    return {
        "consistency_df": consistency_df,
        "selected_feature_df": selected_feature_df,
        "selected_features": selected_feature_df["feature"].tolist(),
        "output_dir": output_dir,
    }


def build_estimator(scenario: ScenarioConfig, *, use_standard_scaler: bool = True) -> Pipeline:
    if scenario.task == "classification":
        model = LogisticRegression(
            solver="liblinear",
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
        )
    elif scenario.task == "regression":
        model = Ridge(alpha=1.0)
    else:
        raise ValueError(f"Unsupported task: {scenario.task}")

    scaler_step: StandardScaler | str
    if use_standard_scaler:
        scaler_step = StandardScaler()
    else:
        scaler_step = "passthrough"

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", scaler_step),
            ("model", model),
        ]
    )


def score_from_estimator(estimator: Pipeline, X: pd.DataFrame, scenario: ScenarioConfig) -> np.ndarray:
    if scenario.task == "classification":
        return estimator.predict_proba(X)[:, 1].astype(np.float32)
    return estimator.predict(X).astype(np.float32)


def summarize_score_metrics(
    y_event: np.ndarray,
    score: np.ndarray,
    *,
    decision_threshold: float,
) -> dict[str, Any]:
    y_event = np.asarray(y_event, dtype=np.int8)
    score = np.asarray(score, dtype=np.float32)
    valid = np.isfinite(score)
    y_valid = y_event[valid]
    score_valid = score[valid]

    if y_valid.size == 0:
        cm = np.zeros((2, 2), dtype=int)
        return {
            "auroc": float("nan"),
            "average_precision": float("nan"),
            "accuracy": float("nan"),
            "balanced_accuracy": float("nan"),
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
            "n_rows": 0,
            "positive_rate": float("nan"),
        }

    if np.unique(y_valid).size < 2:
        auroc = float("nan")
        average_precision = float("nan")
    else:
        auroc = float(roc_auc_score(y_valid, score_valid))
        average_precision = float(average_precision_score(y_valid, score_valid))

    predicted_label = (score_valid >= float(decision_threshold)).astype(np.int8)
    cm = confusion_matrix(y_valid, predicted_label, labels=[0, 1])

    if np.unique(y_valid).size < 2:
        balanced_accuracy = float("nan")
    else:
        balanced_accuracy = float(balanced_accuracy_score(y_valid, predicted_label))

    return {
        "auroc": auroc,
        "average_precision": average_precision,
        "accuracy": float(accuracy_score(y_valid, predicted_label)),
        "balanced_accuracy": balanced_accuracy,
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
        "n_rows": int(y_valid.size),
        "positive_rate": float(np.mean(y_valid)),
    }


def extract_coefficient_df(
    estimator: Pipeline,
    feature_names: list[str],
    *,
    train_env_name: str,
) -> pd.DataFrame:
    model = estimator.named_steps["model"]
    if not hasattr(model, "coef_"):
        return pd.DataFrame(columns=["train_env_name", "feature", "coefficient", "abs_coefficient"])

    coef = np.asarray(model.coef_, dtype=np.float32).reshape(-1)
    out = pd.DataFrame(
        {
            "train_env_name": train_env_name,
            "feature": feature_names,
            "coefficient": coef,
        }
    )
    out["abs_coefficient"] = out["coefficient"].abs()
    return out


def evaluate_feature_subset(
    aggregate_envs: OrderedDict[str, pd.DataFrame],
    prepared_scenario: PreparedScenario,
    feature_names: list[str],
    *,
    decision_threshold_mode: str = "fixed",
    use_standard_scaler: bool = True,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    if not feature_names:
        raise ValueError("evaluate_feature_subset received an empty feature list.")

    pair_rows: list[dict[str, Any]] = []
    coefficient_frames: list[pd.DataFrame] = []

    for train_env_name, train_meta_df in prepared_scenario.env_meta.items():
        estimator = build_estimator(
            prepared_scenario.config,
            use_standard_scaler=use_standard_scaler,
        )
        X_train = aggregate_envs[train_env_name].loc[train_meta_df.index, feature_names]
        if X_train.shape[1] == 0:
            raise ValueError(
                f"No feature columns were selected for train_env_name={train_env_name}. "
                f"Requested features: {feature_names!r}"
            )
        y_train = prepared_scenario.env_targets[train_env_name]
        estimator.fit(X_train, y_train)

        train_scores = score_from_estimator(estimator, X_train, prepared_scenario.config)
        decision_threshold = choose_decision_threshold(
            prepared_scenario.env_events[train_env_name],
            train_scores,
            default_threshold=prepared_scenario.config.decision_threshold,
            mode=decision_threshold_mode,
        )
        train_metrics = summarize_score_metrics(
            prepared_scenario.env_events[train_env_name],
            train_scores,
            decision_threshold=decision_threshold,
        )
        pair_rows.append(
            {
                "train_env_name": train_env_name,
                "eval_env_name": train_env_name,
                "eval_role": "train",
                "decision_threshold": float(decision_threshold),
                "decision_threshold_mode": decision_threshold_mode,
                **train_metrics,
            }
        )
        coefficient_frames.append(
            extract_coefficient_df(estimator, feature_names, train_env_name=train_env_name)
        )

        for eval_env_name, eval_meta_df in prepared_scenario.env_meta.items():
            if eval_env_name == train_env_name:
                continue
            X_eval = aggregate_envs[eval_env_name].loc[eval_meta_df.index, feature_names]
            eval_scores = score_from_estimator(estimator, X_eval, prepared_scenario.config)
            eval_metrics = summarize_score_metrics(
                prepared_scenario.env_events[eval_env_name],
                eval_scores,
                decision_threshold=decision_threshold,
            )
            pair_rows.append(
                {
                    "train_env_name": train_env_name,
                    "eval_env_name": eval_env_name,
                    "eval_role": "ood",
                    "decision_threshold": float(decision_threshold),
                    "decision_threshold_mode": decision_threshold_mode,
                    **eval_metrics,
                }
            )

    pair_metrics_df = pd.DataFrame(pair_rows)
    coefficient_df = (
        pd.concat(coefficient_frames, ignore_index=True)
        if coefficient_frames
        else pd.DataFrame(columns=["train_env_name", "feature", "coefficient", "abs_coefficient"])
    )

    ood_df = pair_metrics_df.loc[pair_metrics_df["eval_role"] == "ood"].copy()
    train_df = pair_metrics_df.loc[pair_metrics_df["eval_role"] == "train"].copy()

    summary = {
        "feature_count": int(len(feature_names)),
        "selected_features_json": json.dumps(list(feature_names)),
        "mean_ood_auroc": safe_metric_mean(ood_df["auroc"]),
        "min_ood_auroc": safe_metric_min(ood_df["auroc"]),
        "std_ood_auroc": safe_metric_std(ood_df["auroc"]),
        "mean_ood_average_precision": safe_metric_mean(ood_df["average_precision"]),
        "mean_ood_accuracy": safe_metric_mean(ood_df["accuracy"]),
        "mean_ood_balanced_accuracy": safe_metric_mean(ood_df["balanced_accuracy"]),
        "mean_train_auroc": safe_metric_mean(train_df["auroc"]),
    }
    return summary, pair_metrics_df, coefficient_df


def _summary_is_better(candidate: dict[str, Any], incumbent: dict[str, Any] | None) -> bool:
    if incumbent is None:
        return True
    candidate_key = (
        rankable_metric(candidate.get("min_ood_auroc")),
        rankable_metric(candidate.get("mean_ood_auroc")),
        rankable_metric(candidate.get("mean_ood_average_precision")),
        -int(candidate.get("feature_count", 0)),
    )
    incumbent_key = (
        rankable_metric(incumbent.get("min_ood_auroc")),
        rankable_metric(incumbent.get("mean_ood_auroc")),
        rankable_metric(incumbent.get("mean_ood_average_precision")),
        -int(incumbent.get("feature_count", 0)),
    )
    return candidate_key > incumbent_key


def summarize_feature_importance(coefficient_df: pd.DataFrame) -> pd.DataFrame:
    if coefficient_df.empty:
        return pd.DataFrame(columns=["feature", "mean_abs_coefficient", "mean_coefficient", "family"])

    summary_df = (
        coefficient_df.groupby("feature", as_index=False)
        .agg(
            mean_abs_coefficient=("abs_coefficient", "mean"),
            mean_coefficient=("coefficient", "mean"),
            std_coefficient=("coefficient", "std"),
        )
        .sort_values(["mean_abs_coefficient", "mean_coefficient"], ascending=[False, False])
        .reset_index(drop=True)
    )
    summary_df["family"] = summary_df["feature"].map(classify_feature_family)
    return summary_df


def build_feature_shortlist(
    univariate_df: pd.DataFrame,
    *,
    per_family_limit: int,
    overall_limit: int,
) -> pd.DataFrame:
    if univariate_df.empty:
        return univariate_df.copy()

    sorted_df = univariate_df.sort_values(
        ["min_ood_auroc", "mean_ood_auroc", "mean_ood_average_precision"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)

    chunks: list[pd.DataFrame] = []
    for family_name in sorted(sorted_df["family"].dropna().unique()):
        family_df = sorted_df.loc[sorted_df["family"] == family_name].head(int(per_family_limit))
        if not family_df.empty:
            chunks.append(family_df)

    shortlist_df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=sorted_df.columns)
    shortlist_df = shortlist_df.drop_duplicates(subset=["feature"])

    if len(shortlist_df) < int(overall_limit):
        remaining_df = sorted_df.loc[~sorted_df["feature"].isin(shortlist_df["feature"])]
        shortlist_df = pd.concat(
            [shortlist_df, remaining_df.head(int(overall_limit) - len(shortlist_df))],
            ignore_index=True,
        )

    shortlist_df = shortlist_df.sort_values(
        ["min_ood_auroc", "mean_ood_auroc", "mean_ood_average_precision"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    return shortlist_df.head(int(overall_limit)).copy()


def run_scenario_search(
    *,
    aggregate_envs: OrderedDict[str, pd.DataFrame],
    scenario: ScenarioConfig,
    feature_names: list[str],
    output_root: Path,
    shortlist_size: int = 48,
    per_family_shortlist: int = 12,
    seed_feature_count: int = 8,
    max_selected_features: int = 10,
    min_forward_improvement: float = 0.001,
    decision_threshold_mode: str = "fixed",
    use_standard_scaler: bool = True,
) -> dict[str, Any]:
    prepared_scenario = prepare_scenario_data(aggregate_envs, scenario)
    subset_cache: dict[tuple[str, ...], dict[str, Any]] = {}

    def evaluate_cached(selected_features: list[str]) -> dict[str, Any]:
        cache_key = tuple(selected_features)
        cached = subset_cache.get(cache_key)
        if cached is not None:
            return cached
        summary, pair_df, coefficient_df = evaluate_feature_subset(
            aggregate_envs,
            prepared_scenario,
            list(selected_features),
            decision_threshold_mode=decision_threshold_mode,
            use_standard_scaler=use_standard_scaler,
        )
        cached = {
            "features": list(selected_features),
            "summary": summary,
            "pair_metrics_df": pair_df,
            "coefficient_df": coefficient_df,
        }
        subset_cache[cache_key] = cached
        return cached

    univariate_rows: list[dict[str, Any]] = []
    for feature_name in feature_names:
        feature_result = evaluate_cached([feature_name])
        univariate_rows.append(
            {
                "feature": feature_name,
                "feature_root": split_aggregate_feature(feature_name)[0],
                "family": classify_feature_family(feature_name),
                **feature_result["summary"],
            }
        )

    univariate_df = pd.DataFrame(univariate_rows).sort_values(
        ["min_ood_auroc", "mean_ood_auroc", "mean_ood_average_precision"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)

    shortlist_df = build_feature_shortlist(
        univariate_df,
        per_family_limit=int(per_family_shortlist),
        overall_limit=int(shortlist_size),
    )
    seed_features = shortlist_df["feature"].head(int(seed_feature_count)).tolist()

    trace_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    best_result: dict[str, Any] | None = None

    for seed_rank, seed_feature in enumerate(seed_features, start=1):
        selected_features = [seed_feature]
        current_result = evaluate_cached(selected_features)
        trace_rows.append(
            {
                "seed_rank": int(seed_rank),
                "seed_feature": seed_feature,
                "step_idx": 1,
                "candidate_feature": seed_feature,
                "accepted": True,
                "selected_features_json": json.dumps(selected_features),
                **current_result["summary"],
            }
        )

        while len(selected_features) < int(max_selected_features):
            best_candidate_feature: str | None = None
            best_candidate_result: dict[str, Any] | None = None

            for candidate_feature in shortlist_df["feature"]:
                if candidate_feature in selected_features:
                    continue
                candidate_features = selected_features + [candidate_feature]
                candidate_result = evaluate_cached(candidate_features)
                trace_rows.append(
                    {
                        "seed_rank": int(seed_rank),
                        "seed_feature": seed_feature,
                        "step_idx": int(len(candidate_features)),
                        "candidate_feature": candidate_feature,
                        "accepted": False,
                        "selected_features_json": json.dumps(candidate_features),
                        **candidate_result["summary"],
                    }
                )
                if _summary_is_better(
                    candidate_result["summary"],
                    best_candidate_result["summary"] if best_candidate_result is not None else None,
                ):
                    best_candidate_feature = candidate_feature
                    best_candidate_result = candidate_result

            if best_candidate_feature is None or best_candidate_result is None:
                break

            improvement = (
                rankable_metric(best_candidate_result["summary"]["min_ood_auroc"])
                - rankable_metric(current_result["summary"]["min_ood_auroc"])
            )
            if improvement < float(min_forward_improvement):
                break

            selected_features = best_candidate_result["features"]
            current_result = best_candidate_result
            trace_rows.append(
                {
                    "seed_rank": int(seed_rank),
                    "seed_feature": seed_feature,
                    "step_idx": int(len(selected_features)),
                    "candidate_feature": best_candidate_feature,
                    "accepted": True,
                    "selected_features_json": json.dumps(selected_features),
                    **current_result["summary"],
                }
            )

        seed_rows.append(
            {
                "seed_rank": int(seed_rank),
                "seed_feature": seed_feature,
                **current_result["summary"],
            }
        )
        if _summary_is_better(
            current_result["summary"],
            best_result["summary"] if best_result is not None else None,
        ):
            best_result = current_result

    if best_result is None:
        raise RuntimeError(f"No feature subset could be evaluated for scenario {scenario.key}.")

    trace_df = pd.DataFrame(trace_rows)
    seed_summary_df = pd.DataFrame(seed_rows).sort_values(
        ["min_ood_auroc", "mean_ood_auroc", "feature_count"],
        ascending=[False, False, True],
        na_position="last",
    ).reset_index(drop=True)

    best_summary_df = pd.DataFrame([best_result["summary"]])
    importance_df = summarize_feature_importance(best_result["coefficient_df"])
    selected_feature_df = pd.DataFrame(
        {
            "selected_order": np.arange(1, len(best_result["features"]) + 1),
            "feature": best_result["features"],
        }
    )
    selected_feature_df["family"] = selected_feature_df["feature"].map(classify_feature_family)
    selected_feature_df = selected_feature_df.merge(
        univariate_df[
            [
                "feature",
                "mean_ood_auroc",
                "min_ood_auroc",
                "std_ood_auroc",
                "mean_ood_average_precision",
            ]
        ],
        on="feature",
        how="left",
    ).merge(
        importance_df[
            [
                "feature",
                "mean_abs_coefficient",
                "mean_coefficient",
                "std_coefficient",
            ]
        ],
        on="feature",
        how="left",
    )

    scenario_output_dir = output_root / scenario.key
    scenario_output_dir.mkdir(parents=True, exist_ok=True)
    config_payload = {
        "scenario": scenario.key,
        "title": scenario.title,
        "notes": scenario.notes,
        "feature_count_total": int(len(feature_names)),
        "shortlist_size": int(shortlist_size),
        "per_family_shortlist": int(per_family_shortlist),
        "seed_feature_count": int(seed_feature_count),
        "max_selected_features": int(max_selected_features),
        "min_forward_improvement": float(min_forward_improvement),
        "decision_threshold_mode": str(decision_threshold_mode),
        "use_standard_scaler": bool(use_standard_scaler),
        "train_environments": list(aggregate_envs.keys()),
    }
    (scenario_output_dir / "config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
    univariate_df.to_csv(scenario_output_dir / "univariate_screen.csv", index=False)
    shortlist_df.to_csv(scenario_output_dir / "shortlist.csv", index=False)
    trace_df.to_csv(scenario_output_dir / "greedy_trace.csv", index=False)
    seed_summary_df.to_csv(scenario_output_dir / "seed_summary.csv", index=False)
    best_summary_df.to_csv(scenario_output_dir / "best_summary.csv", index=False)
    best_result["pair_metrics_df"].to_csv(scenario_output_dir / "best_pair_metrics.csv", index=False)
    best_result["coefficient_df"].to_csv(scenario_output_dir / "best_coefficients_by_train_env.csv", index=False)
    importance_df.to_csv(scenario_output_dir / "feature_importance_summary.csv", index=False)
    selected_feature_df.to_csv(scenario_output_dir / "selected_features.csv", index=False)

    return {
        "scenario": scenario,
        "prepared_scenario": prepared_scenario,
        "univariate_df": univariate_df,
        "shortlist_df": shortlist_df,
        "trace_df": trace_df,
        "seed_summary_df": seed_summary_df,
        "best_summary_df": best_summary_df,
        "pair_metrics_df": best_result["pair_metrics_df"],
        "coefficient_df": best_result["coefficient_df"],
        "importance_df": importance_df,
        "selected_feature_df": selected_feature_df,
        "selected_features": best_result["features"],
        "output_dir": scenario_output_dir,
        "env_order": list(aggregate_envs.keys()),
    }


def plot_feature_importance(importance_df: pd.DataFrame, *, title: str, top_k: int = 10) -> None:
    plot_df = importance_df.head(int(top_k)).iloc[::-1].copy()
    fig, ax = plt.subplots(figsize=(10, 4 + 0.35 * len(plot_df)))
    ax.barh(plot_df["feature"], plot_df["mean_abs_coefficient"], color="#1f77b4")
    ax.set_xlabel("Mean feature importance across train environments")
    ax.set_ylabel("Feature")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_ood_auroc_heatmap(
    pair_metrics_df: pd.DataFrame,
    *,
    env_order: list[str],
    title: str,
) -> None:
    ood_df = pair_metrics_df.loc[pair_metrics_df["eval_role"] == "ood"].copy()
    heatmap_df = (
        ood_df.pivot(index="train_env_name", columns="eval_env_name", values="auroc")
        .reindex(index=env_order, columns=env_order)
    )

    fig, ax = plt.subplots(figsize=(1.9 * len(env_order), 1.6 * len(env_order)))
    data = heatmap_df.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(data)
    image = ax.imshow(masked, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(env_order)))
    ax.set_xticklabels(env_order, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(env_order)))
    ax.set_yticklabels(env_order)
    ax.set_xlabel("Evaluation environment")
    ax.set_ylabel("Training environment")
    ax.set_title(title)

    for row_idx in range(len(env_order)):
        for col_idx in range(len(env_order)):
            value = heatmap_df.iloc[row_idx, col_idx]
            if pd.isna(value):
                label = "-"
            else:
                label = f"{value:.3f}"
            ax.text(col_idx, row_idx, label, ha="center", va="center", color="white", fontsize=9)

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="AUROC")
    plt.tight_layout()
    plt.show()


def plot_env_accuracy_heatmap(
    pair_metrics_df: pd.DataFrame,
    *,
    env_order: list[str],
    title: str,
) -> None:
    train_df = pair_metrics_df.loc[pair_metrics_df["eval_role"] == "train"].copy()
    ood_df = pair_metrics_df.loc[pair_metrics_df["eval_role"] == "ood"].copy()

    heatmap_df = pd.DataFrame(np.nan, index=env_order, columns=env_order, dtype=float)

    if not train_df.empty:
        train_matrix = (
            train_df.pivot(index="train_env_name", columns="eval_env_name", values="accuracy")
            .reindex(index=env_order, columns=env_order)
        )
        for env_name in env_order:
            if env_name in train_matrix.index and env_name in train_matrix.columns:
                heatmap_df.loc[env_name, env_name] = train_matrix.loc[env_name, env_name]

    if not ood_df.empty:
        ood_matrix = (
            ood_df.pivot(index="train_env_name", columns="eval_env_name", values="accuracy")
            .reindex(index=env_order, columns=env_order)
        )
        for train_env_name in env_order:
            for eval_env_name in env_order:
                if train_env_name == eval_env_name:
                    continue
                value = ood_matrix.loc[train_env_name, eval_env_name]
                if pd.notna(value):
                    heatmap_df.loc[train_env_name, eval_env_name] = value

    fig, ax = plt.subplots(figsize=(1.9 * len(env_order), 1.6 * len(env_order)))
    data = heatmap_df.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(data)
    image = ax.imshow(masked, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(env_order)))
    ax.set_xticklabels(env_order, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(env_order)))
    ax.set_yticklabels(env_order)
    ax.set_xlabel("Evaluation environment")
    ax.set_ylabel("Training environment")
    ax.set_title(title)

    for row_idx in range(len(env_order)):
        for col_idx in range(len(env_order)):
            value = heatmap_df.iloc[row_idx, col_idx]
            if pd.isna(value):
                label = "-"
            else:
                label = f"{value:.3f}"
                if row_idx == col_idx:
                    label = f"train\n{value:.3f}"
            ax.text(col_idx, row_idx, label, ha="center", va="center", color="white", fontsize=9)

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Accuracy")
    plt.tight_layout()
    plt.show()


def plot_confusion_grid(
    pair_metrics_df: pd.DataFrame,
    *,
    title: str,
) -> None:
    ood_df = pair_metrics_df.loc[pair_metrics_df["eval_role"] == "ood"].copy().reset_index(drop=True)
    if ood_df.empty:
        return

    n_panels = len(ood_df)
    n_cols = min(4, n_panels)
    n_rows = int(math.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 3.5 * n_rows))
    axes = np.atleast_1d(axes).reshape(n_rows, n_cols)

    vmax = max(int(ood_df[["tn", "fp", "fn", "tp"]].to_numpy().max()), 1)
    for axis, (_, row) in zip(axes.flat, ood_df.iterrows()):
        confusion = np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]], dtype=float)
        image = axis.imshow(confusion, cmap="Blues", vmin=0.0, vmax=vmax)
        axis.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
        axis.set_yticks([0, 1], labels=["True 0", "True 1"])
        axis.set_title(
            f"{row['train_env_name']} -> {row['eval_env_name']}\nAUROC={safe_float(row['auroc']):.3f}",
            fontsize=10,
        )
        for i in range(2):
            for j in range(2):
                axis.text(j, i, f"{int(confusion[i, j])}", ha="center", va="center", color="black", fontsize=10)

    for axis in axes.flat[n_panels:]:
        axis.axis("off")

    fig.suptitle(title, fontsize=14)
    fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02, label="Count")
    plt.tight_layout()
    plt.show()


def show_scenario_result(result: dict[str, Any]) -> None:
    scenario: ScenarioConfig = result["scenario"]
    print(scenario.title)
    print(scenario.notes)
    print(f"Outputs saved to: {result['output_dir']}")
    print()

    notebook_display(result["best_summary_df"])
    notebook_display(result["selected_feature_df"])
    notebook_display(
        result["pair_metrics_df"]
        .loc[result["pair_metrics_df"]["eval_role"].isin(["train", "ood"])]
        .assign(eval_role_order=lambda df: df["eval_role"].map({"train": 0, "ood": 1}).fillna(99).astype(int))
        .sort_values(["train_env_name", "eval_role_order", "eval_env_name"])
        .drop(columns=["eval_role_order"])
        .reset_index(drop=True)
    )

    plot_feature_importance(
        result["importance_df"],
        title=f"{scenario.title}: selected-feature importance",
        top_k=min(10, len(result["importance_df"])),
    )
    plot_env_accuracy_heatmap(
        result["pair_metrics_df"],
        env_order=result["env_order"],
        title=f"{scenario.title}: validation and OOD accuracy",
    )
    plot_confusion_grid(
        result["pair_metrics_df"],
        title=f"{scenario.title}: OOD confusion matrices",
    )


def list_output_files(output_root: Path) -> pd.DataFrame:
    if not output_root.exists():
        return pd.DataFrame(columns=["scenario", "path", "bytes"])

    rows: list[dict[str, Any]] = []
    for output_path in sorted(path for path in output_root.rglob("*") if path.is_file()):
        relative_parts = output_path.relative_to(output_root).parts
        scenario_name = relative_parts[0] if relative_parts else output_path.name
        rows.append(
            {
                "scenario": scenario_name,
                "path": str(output_path),
                "bytes": int(output_path.stat().st_size),
            }
        )
    return pd.DataFrame(rows)
