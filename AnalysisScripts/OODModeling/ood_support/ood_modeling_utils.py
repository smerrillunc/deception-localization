"""Shared helpers for the out-of-distribution modeling pipeline.

Scenario catalogs, feature-space definitions, and the frame builders that turn
per-environment feature tables into the aligned matrices `ood_modeling_lib.py`
fits on.
"""
from __future__ import annotations

import json
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


