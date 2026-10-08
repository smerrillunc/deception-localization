from __future__ import annotations

import gc
import importlib
import json
import math
import re
import sys
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    from IPython.display import display
except Exception:
    def display(value: Any) -> None:
        print(value)

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:
    _tqdm = None

RUNTIME_CONFIG: dict[str, Any] = dict(globals().get("RUNTIME_CONFIG", {}) or {})
_MISSING = object()


def config_value(name: str, default: Any = _MISSING) -> Any:
    value = RUNTIME_CONFIG.get(name, _MISSING)
    if value is _MISSING or value is None:
        if default is _MISSING:
            raise ValueError(f"Missing required OOD modeling config value: {name}")
        return default
    return value


def config_int(name: str, default: int | None = None) -> int | None:
    raw = config_value(name, default)
    if raw is None:
        return None
    return int(raw)


def config_float(name: str, default: float | None = None) -> float | None:
    raw = config_value(name, default)
    if raw is None:
        return None
    return float(raw)


def config_bool(name: str, default: bool = False) -> bool:
    raw = config_value(name, default)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        normalized = raw.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return bool(raw)


def config_float_tuple(name: str, default: tuple[float, ...]) -> tuple[float, ...]:
    raw = config_value(name, default)
    if isinstance(raw, (list, tuple)):
        return tuple(float(value) for value in raw)
    parts = [part.strip() for part in str(raw).split(",") if part.strip()]
    return tuple(float(part) for part in parts)


def config_int_tuple(name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    raw = config_value(name, default)
    if isinstance(raw, (list, tuple)):
        return tuple(int(value) for value in raw)
    parts = [part.strip() for part in str(raw).split(",") if part.strip()]
    return tuple(int(part) for part in parts)


def config_str_tuple(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = config_value(name, default)
    if isinstance(raw, (list, tuple)):
        return tuple(str(value).strip() for value in raw if str(value).strip())
    return tuple(part.strip() for part in str(raw).split(",") if part.strip())


SOURCE_PATH = Path(
    config_value(
        "source_path",
        __file__ if "__file__" in globals() else str(Path.cwd() / "ood_modeling_lib.py"),
    )
).resolve()
ANALYSIS_ROOT = Path(config_value("analysis_root", str(SOURCE_PATH.parent.parent))).resolve()
ROOT_DIR = Path(config_value("repo_root", str(ANALYSIS_ROOT.parent))).resolve()
CORE_DIR = ROOT_DIR / "Core"
OOD_SUPPORT_DIR = SOURCE_PATH.parent
for search_root in (ANALYSIS_ROOT, OOD_SUPPORT_DIR, CORE_DIR):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

import ood_modeling_utils as ood_utils
import attention_activation_feature_extractor as extractor
import text_structural_feature_extractor as baseline_extractor

ood_utils = importlib.reload(ood_utils)
slugify = ood_utils.slugify
choose_decision_threshold = ood_utils.choose_decision_threshold
summarize_score_metrics = ood_utils.summarize_score_metrics
safe_metric_mean = ood_utils.safe_metric_mean
safe_metric_min = ood_utils.safe_metric_min
safe_metric_std = ood_utils.safe_metric_std
build_common_layer_roots = ood_utils.build_common_layer_roots
COMMITMENT_NON_FEATURE_COLUMNS = set(ood_utils.COMMITMENT_NON_FEATURE_COLUMNS)
classify_feature_family = ood_utils.classify_feature_family

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.max_columns", 200)
pd.set_option("display.max_rows", 200)
pd.set_option("display.width", 240)


def maybe_tqdm(iterable: Iterable[Any], *, desc: str, total: int | None = None, disable: bool = False):
    if disable or _tqdm is None:
        return iterable
    return _tqdm(iterable, desc=desc, total=total)


def normalize_model_family(raw_value: str | None) -> str:
    value = str(raw_value or "logreg").strip().lower()
    if value in {"logreg", "logistic", "logistic_regression", "lr"}:
        return "logreg"
    if value in {"xgboost", "xgb"}:
        return "xgboost"
    raise ValueError(
        f"Unsupported model family {raw_value!r}. "
        "Expected one of ['logreg', 'xgboost']."
    )


def normalize_scenario_key(raw_value: str | None) -> str:
    value = slugify(str(raw_value or "single_source_ood"))
    aliases = {
        "single_source_ood": "single_source_ood",
        "train_one_eval_all": "single_source_ood",
        "single_env_ood": "single_source_ood",
        "one_to_all": "single_source_ood",
        "holdout_env_ood": "holdout_env_ood",
        "train_four_holdout_one": "holdout_env_ood",
        "leave_one_env_out": "holdout_env_ood",
        "four_to_one": "holdout_env_ood",
    }
    if value not in aliases:
        raise ValueError(
            f"Unsupported scenario key {raw_value!r}. "
            "Expected one of ['single_source_ood', 'holdout_env_ood']."
        )
    return aliases[value]


# %%
DATASET_ROOT = Path(
    config_value(
        "dataset_root",
        str(ROOT_DIR / "DatasetMain"),
    )
).resolve()
MODEL_DIRNAME = str(config_value("model_dirname"))

ENV_SPECS = OrderedDict(
    [
        ("AdvisorAudit", "advisor_audit"),
        ("BS", "bs"),
        ("CarSales", "car_sales"),
        ("Gridworld", "gridworld"),
        ("Interview", "interview"),
    ]
)
ENV_ORDER = list(ENV_SPECS.keys())

FEATURE_FILENAME_CANDIDATES = (
    "prefix_deception_features.parquet",
    "prefix_deception_features.parquet.tmp",
)
ACTIVATION_FILENAME = "prefix_deception_activations.h5"
STRUCTURAL_BASELINE_FILENAME = str(config_value(
    "structural_baseline_filename",
    getattr(baseline_extractor, "DEFAULT_OUTPUT_NAME", "commitment_text_structural_baselines.parquet"),
))
TFIDF_CACHE_DIRNAME = str(config_value(
    "tfidf_cache_dirname",
    getattr(baseline_extractor, "DEFAULT_TFIDF_CACHE_DIRNAME", "commitment_text_baseline_tfidf_cache"),
))
TFIDF_TEXT_FIELDS = tuple(
    config_str_tuple(
        "tfidf_text_fields",
        tuple(getattr(baseline_extractor, "DEFAULT_TFIDF_TEXT_FIELDS", ("last_sentence_text", "prefix_text"))),
    )
)
RANDOM_SEED = int(config_value("seed"))
VAL_SIZE = float(config_value("val_size"))
DELTA_THRESHOLD = float(config_value("delta_threshold"))
MIN_NUM_VALID = int(config_value("min_num_valid", 11))
MIN_SENTENCE_ALPHA_WORDS = int(config_value("min_sentence_alpha_words", 4))
EXCLUDE_MULTILINE_SENTENCES = config_bool("exclude_multiline_sentences", True)
FEATURE_SIZE_GRID = config_int_tuple("feature_sizes", ())
if not FEATURE_SIZE_GRID:
    raise ValueError("feature_sizes must contain at least one integer.")
SCENARIO_KEYS = tuple(
    normalize_scenario_key(raw_key)
    for raw_key in config_str_tuple("scenarios", ())
)
if not SCENARIO_KEYS:
    raise ValueError("scenarios must contain at least one scenario key.")
MODEL_FAMILY = normalize_model_family(str(config_value("model_family")))
MODEL_FAMILY_TITLE = {
    "logreg": "Logistic regression",
    "xgboost": "XGBoost",
}[MODEL_FAMILY]
MODEL_WEIGHT_KIND = {
    "logreg": "coefficient",
    "xgboost": "gain_importance",
}[MODEL_FAMILY]
LOGREG_C = config_float("logreg_c", 0.1)
XGB_MAX_DEPTH = config_int("xgb_max_depth", 5)
XGB_N_ESTIMATORS = int(config_value("xgb_n_estimators", 300))
XGB_LEARNING_RATE = float(config_value("xgb_learning_rate", 0.05))
XGB_SUBSAMPLE = float(config_value("xgb_subsample", 0.8))
XGB_COLSAMPLE_BYTREE = float(config_value("xgb_colsample_bytree", 0.8))
XGB_REG_LAMBDA = float(config_value("xgb_reg_lambda", 1.0))
XGB_MIN_CHILD_WEIGHT = float(config_value("xgb_min_child_weight", 1.0))
XGB_GAMMA = float(config_value("xgb_gamma", 0.0))
XGB_N_JOBS = int(config_value("xgb_n_jobs", 1))
XGB_IMPORTANCE_TYPE = str(config_value("xgb_importance_type", "gain"))
ROOT_BATCH_SIZE = int(config_value("root_batch_size"))
ACTIVATION_PCA_DIM = int(max(FEATURE_SIZE_GRID))
DECISION_THRESHOLD_MODE = str(config_value("decision_threshold_mode"))
MODEL_SELECTION_OBJECTIVE = str(config_value("model_selection_objective"))
FORCE_REBUILD_REDUCTIONS = config_bool("force_rebuild", False)
DISABLE_TQDM = config_bool("disable_tqdm", False)
TOP_FEATURES_TO_SHOW = int(config_value("top_features_to_show"))
FEATURE_SPACE_MODE = str(config_value("feature_space_mode", "all")).strip().lower()

STRUCTURAL_BASELINE_EXCLUDED_COLUMNS = {
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
    "deceptive_commitment_sentence_idx",
    "truthful_commitment_sentence_idx",
    "example_commitment_sentence_idx",
    "num_truthful",
    "num_valid",
    "deception_rate",
    "prev_deception_rate",
    "next_deception_rate",
    "delta_deception_rate",
    "abs_delta_deception_rate",
    "target_value",
    "has_deceptive_commitment",
    "has_truthful_commitment",
    "has_example_commitment",
    "is_deceptive_commitment_juncture",
    "is_truthful_commitment_juncture",
    "is_commitment_juncture",
}
STRUCTURAL_BASELINE_FEATURE_COLUMNS = tuple(
    column_name
    for column_name in baseline_extractor.OUTPUT_COLUMNS
    if column_name not in STRUCTURAL_BASELINE_EXCLUDED_COLUMNS
    and (
        column_name in baseline_extractor.INT_COLUMNS
        or column_name in baseline_extractor.FLOAT_COLUMNS
        or column_name in baseline_extractor.BOOL_COLUMNS
    )
)

SCENARIO_TITLES = OrderedDict(
    [
        ("single_source_ood", "Train on 1 environment; evaluate OOD on the other 4"),
        ("holdout_env_ood", "Train on 4 environments; evaluate OOD on the held-out environment"),
    ]
)
unknown_scenarios = sorted(set(SCENARIO_KEYS) - set(SCENARIO_TITLES))
if unknown_scenarios:
    raise ValueError(f"Unsupported scenarios requested: {', '.join(unknown_scenarios)}")
SELECTED_SCENARIO_TITLES = OrderedDict((scenario_key, SCENARIO_TITLES[scenario_key]) for scenario_key in SCENARIO_KEYS)
SCENARIO_SLUG = "__".join(slugify(scenario_key) for scenario_key in SCENARIO_KEYS)

if MODEL_FAMILY == "xgboost" and XGBClassifier is None:
    raise ImportError("xgboost is not installed, but MODEL_FAMILY='xgboost' was requested.")

OUTPUT_ROOT = Path(
    config_value(
        "output_root",
        str(
            ANALYSIS_ROOT
            / f"ood_modeling_outputs__{slugify(MODEL_DIRNAME)}__{slugify(MODEL_FAMILY)}__{SCENARIO_SLUG}"
        ),
    )
)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
CACHE_DIR = OUTPUT_ROOT / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_EVERY_MODEL_SELECTIONS = max(1, int(config_value("checkpoint_every", 5)))

ATTENTION_METRIC_NAMES = tuple(extractor.ATTENTION_METRIC_NAMES)
HEAD_SUMMARY_NAMES = tuple(extractor.HEAD_SUMMARY_NAMES)
ATTENTION_TRANSITION_PREFIXES = tuple(extractor.TRANSITION_PREFIXES)
BAND_NAMES = ("early", "mid", "late")
BAND_STAT_NAMES = ("mean", "min", "max", "std")

GROUNDING_ATTENTION_METRIC_NAMES = (
    "current_vs_prior",
    "current_vs_prev",
    "current_vs_prev3",
    "recent_vs_early",
    "prev_share_of_prior",
    "current_share_total",
)
CONCENTRATION_ATTENTION_METRIC_NAMES = (
    "entropy_prior",
    "entropy_full",
    "top1_prior",
    "top5_prior",
    "herfindahl_prior",
    "effective_support_prior",
)
ATTENTION_METRIC_GROUP_BY_NAME = {
    **{metric_name: "grounding" for metric_name in GROUNDING_ATTENTION_METRIC_NAMES},
    **{metric_name: "concentration" for metric_name in CONCENTRATION_ATTENTION_METRIC_NAMES},
}
unknown_attention_metrics = sorted(set(ATTENTION_METRIC_NAMES) - set(ATTENTION_METRIC_GROUP_BY_NAME))
if unknown_attention_metrics:
    raise ValueError(
        "Attention metrics are missing grounding/concentration assignments: "
        + ", ".join(unknown_attention_metrics)
    )

config_df = pd.DataFrame(
    [
        {"setting": "source_path", "value": str(SOURCE_PATH)},
        {"setting": "analysis_root", "value": str(ANALYSIS_ROOT)},
        {"setting": "repo_root", "value": str(ROOT_DIR)},
        {"setting": "model_dirname", "value": MODEL_DIRNAME},
        {"setting": "dataset_root", "value": str(DATASET_ROOT)},
        {"setting": "output_root", "value": str(OUTPUT_ROOT)},
        {"setting": "env_order", "value": ", ".join(ENV_ORDER)},
        {"setting": "scenario_keys", "value": ", ".join(SCENARIO_KEYS)},
        {"setting": "val_size", "value": VAL_SIZE},
        {"setting": "delta_threshold", "value": DELTA_THRESHOLD},
        {"setting": "min_num_valid", "value": MIN_NUM_VALID},
        {"setting": "min_sentence_alpha_words", "value": MIN_SENTENCE_ALPHA_WORDS},
        {"setting": "exclude_multiline_sentences", "value": EXCLUDE_MULTILINE_SENTENCES},
        {"setting": "model_family", "value": MODEL_FAMILY},
        {"setting": "model_family_title", "value": MODEL_FAMILY_TITLE},
        {"setting": "model_weight_kind", "value": MODEL_WEIGHT_KIND},
        {"setting": "attention_pool_mode", "value": "all_ranked_attention_features"},
        {"setting": "feature_size_grid", "value": ", ".join(str(value) for value in FEATURE_SIZE_GRID)},
        {"setting": "model_search_mode", "value": "fixed_hyperparameters"},
        {"setting": "logreg_c", "value": LOGREG_C},
        {"setting": "xgb_max_depth", "value": XGB_MAX_DEPTH},
        {"setting": "xgb_n_estimators", "value": XGB_N_ESTIMATORS},
        {"setting": "xgb_learning_rate", "value": XGB_LEARNING_RATE},
        {"setting": "xgb_subsample", "value": XGB_SUBSAMPLE},
        {"setting": "xgb_colsample_bytree", "value": XGB_COLSAMPLE_BYTREE},
        {"setting": "xgb_reg_lambda", "value": XGB_REG_LAMBDA},
        {"setting": "xgb_min_child_weight", "value": XGB_MIN_CHILD_WEIGHT},
        {"setting": "xgb_gamma", "value": XGB_GAMMA},
        {"setting": "xgb_n_jobs", "value": XGB_N_JOBS},
        {"setting": "xgb_importance_type", "value": XGB_IMPORTANCE_TYPE},
        {"setting": "root_batch_size", "value": ROOT_BATCH_SIZE},
        {"setting": "activation_pca_dim", "value": ACTIVATION_PCA_DIM},
        {"setting": "model_selection_objective", "value": MODEL_SELECTION_OBJECTIVE},
        {"setting": "checkpoint_every_model_selections", "value": CHECKPOINT_EVERY_MODEL_SELECTIONS},
        {"setting": "force_rebuild_reductions", "value": FORCE_REBUILD_REDUCTIONS},
        {"setting": "disable_tqdm", "value": DISABLE_TQDM},
        {"setting": "top_features_to_show", "value": TOP_FEATURES_TO_SHOW},
    ]
)
display(config_df)
config_df.to_csv(OUTPUT_ROOT / "config.csv", index=False)


TARGET_SPECS = OrderedDict(
    [
        (
            "delta_pos_gt_0_3",
            {
                "title": "delta_deception_rate > 0.3",
                "label_fn": lambda delta: delta > DELTA_THRESHOLD,
                "negative_label": f"<= {DELTA_THRESHOLD:.1f}",
                "positive_label": f"> {DELTA_THRESHOLD:.1f}",
            },
        ),
        (
            "delta_neg_lt_neg_0_3",
            {
                "title": "delta_deception_rate < -0.3",
                "label_fn": lambda delta: delta < -DELTA_THRESHOLD,
                "negative_label": f">= -{DELTA_THRESHOLD:.1f}",
                "positive_label": f"< -{DELTA_THRESHOLD:.1f}",
            },
        ),
    ]
)


@dataclass(frozen=True)
class AttentionSubsetSpec:
    key: str
    title: str
    metric_names: tuple[str, ...]
    transition_mode: str


@dataclass(frozen=True)
class FeatureSpaceSpec:
    name: str
    title: str
    family_title: str
    uses_attention: bool
    attention_subset_key: str | None
    activation_variant: str | None
    activation_use_pca: bool
    baseline_variant: str | None = None
    baseline_text_field: str | None = None


ATTENTION_SUBSETS = OrderedDict(
    [
        (
            "all_attention",
            AttentionSubsetSpec(
                key="all_attention",
                title="All attention features",
                metric_names=ATTENTION_METRIC_NAMES,
                transition_mode="all",
            ),
        ),
        (
            "grounding_only",
            AttentionSubsetSpec(
                key="grounding_only",
                title="Grounding only",
                metric_names=GROUNDING_ATTENTION_METRIC_NAMES,
                transition_mode="base_only",
            ),
        ),
        (
            "concentration_only",
            AttentionSubsetSpec(
                key="concentration_only",
                title="Concentration only",
                metric_names=CONCENTRATION_ATTENTION_METRIC_NAMES,
                transition_mode="base_only",
            ),
        ),
        (
            "grounding_transition_only",
            AttentionSubsetSpec(
                key="grounding_transition_only",
                title="Grounding transitions only",
                metric_names=GROUNDING_ATTENTION_METRIC_NAMES,
                transition_mode="transition_only",
            ),
        ),
        (
            "concentration_transition_only",
            AttentionSubsetSpec(
                key="concentration_transition_only",
                title="Concentration transitions only",
                metric_names=CONCENTRATION_ATTENTION_METRIC_NAMES,
                transition_mode="transition_only",
            ),
        ),
    ]
)


BASE_FEATURE_SPACES = OrderedDict(
    [
        (
            "attention_only",
            FeatureSpaceSpec(
                name="attention_only",
                title="Attention only",
                family_title="attention_only",
                uses_attention=True,
                attention_subset_key="all_attention",
                activation_variant=None,
                activation_use_pca=False,
            ),
        ),
        (
            "attention_grounding_only",
            FeatureSpaceSpec(
                name="attention_grounding_only",
                title="Attention only: grounding",
                family_title="attention_only",
                uses_attention=True,
                attention_subset_key="grounding_only",
                activation_variant=None,
                activation_use_pca=False,
            ),
        ),
        (
            "attention_concentration_only",
            FeatureSpaceSpec(
                name="attention_concentration_only",
                title="Attention only: concentration",
                family_title="attention_only",
                uses_attention=True,
                attention_subset_key="concentration_only",
                activation_variant=None,
                activation_use_pca=False,
            ),
        ),
        (
            "attention_grounding_transition_only",
            FeatureSpaceSpec(
                name="attention_grounding_transition_only",
                title="Attention only: grounding transition",
                family_title="attention_only",
                uses_attention=True,
                attention_subset_key="grounding_transition_only",
                activation_variant=None,
                activation_use_pca=False,
            ),
        ),
        (
            "attention_concentration_transition_only",
            FeatureSpaceSpec(
                name="attention_concentration_transition_only",
                title="Attention only: concentration transition",
                family_title="attention_only",
                uses_attention=True,
                attention_subset_key="concentration_transition_only",
                activation_variant=None,
                activation_use_pca=False,
            ),
        ),
        (
            "activation_pca_final",
            FeatureSpaceSpec(
                name="activation_pca_final",
                title="Activation only: PCA final",
                family_title="activation_only",
                uses_attention=False,
                attention_subset_key=None,
                activation_variant="final_sentence",
                activation_use_pca=True,
            ),
        ),
        (
            "activation_pca_delta_last2",
            FeatureSpaceSpec(
                name="activation_pca_delta_last2",
                title="Activation only: PCA final - previous",
                family_title="activation_only",
                uses_attention=False,
                attention_subset_key=None,
                activation_variant="delta_last2",
                activation_use_pca=True,
            ),
        ),
        (
            "activation_pca_delta_prev4mean",
            FeatureSpaceSpec(
                name="activation_pca_delta_prev4mean",
                title="Activation only: PCA final - mean(prev 4)",
                family_title="activation_only",
                uses_attention=False,
                attention_subset_key=None,
                activation_variant="delta_prev4mean",
                activation_use_pca=True,
            ),
        ),
        (
            "baseline_raw_final",
            FeatureSpaceSpec(
                name="baseline_raw_final",
                title="Baseline: raw final activation",
                family_title="baseline",
                uses_attention=False,
                attention_subset_key=None,
                activation_variant="final_sentence",
                activation_use_pca=False,
            ),
        ),
        (
            "attention_plus_activation_pca_final",
            FeatureSpaceSpec(
                name="attention_plus_activation_pca_final",
                title="Attention + PCA final",
                family_title="attention_plus_activation",
                uses_attention=True,
                attention_subset_key="all_attention",
                activation_variant="final_sentence",
                activation_use_pca=True,
            ),
        ),
        (
            "attention_plus_activation_pca_delta_last2",
            FeatureSpaceSpec(
                name="attention_plus_activation_pca_delta_last2",
                title="Attention + PCA final - previous",
                family_title="attention_plus_activation",
                uses_attention=True,
                attention_subset_key="all_attention",
                activation_variant="delta_last2",
                activation_use_pca=True,
            ),
        ),
        (
            "attention_plus_activation_pca_delta_prev4mean",
            FeatureSpaceSpec(
                name="attention_plus_activation_pca_delta_prev4mean",
                title="Attention + PCA final - mean(prev 4)",
                family_title="attention_plus_activation",
                uses_attention=True,
                attention_subset_key="all_attention",
                activation_variant="delta_prev4mean",
                activation_use_pca=True,
            ),
        ),
    ]
)

FAMILY_PANEL_ORDER = (
    "attention_only",
    "activation_only",
    "attention_plus_activation",
    "baseline",
)

FAMILY_DISPLAY_TITLES = {
    "attention_only": "Attention only",
    "activation_only": "Activation only",
    "attention_plus_activation": "Attention + activation",
    "baseline": "Baseline",
}


def validate_parquet_file(path: Path) -> None:
    pq.ParquetFile(path)


def resolve_feature_path_optional(env_name: str, env_dir: str, model_dirname: str) -> Path | None:
    candidate_paths = [
        (DATASET_ROOT / env_dir / model_dirname / filename).resolve()
        for filename in FEATURE_FILENAME_CANDIDATES
    ]
    existing_paths = [path for path in candidate_paths if path.exists()]
    if not existing_paths:
        return None

    for path in existing_paths:
        try:
            validate_parquet_file(path)
            return path
        except Exception:
            continue
    return None


def resolve_feature_path(env_name: str, env_dir: str, model_dirname: str) -> Path:
    candidate_paths = [
        (DATASET_ROOT / env_dir / model_dirname / filename).resolve()
        for filename in FEATURE_FILENAME_CANDIDATES
    ]
    resolved_path = resolve_feature_path_optional(env_name, env_dir, model_dirname)
    if resolved_path is not None:
        return resolved_path

    existing_paths = [path for path in candidate_paths if path.exists()]
    if not existing_paths:
        raise FileNotFoundError(
            "Missing feature parquet for "
            f"{env_name}: checked {', '.join(str(path) for path in candidate_paths)}"
        )

    validation_errors: list[str] = []
    for path in existing_paths:
        try:
            validate_parquet_file(path)
            return path
        except Exception as exc:
            validation_errors.append(f"{path}: {type(exc).__name__}: {exc}")

    raise ValueError(
        "No readable feature parquet for "
        f"{env_name} / {model_dirname}. Checked: " + " | ".join(validation_errors)
    )


def feature_mode_requires_attention(mode: str) -> bool:
    normalized_mode = str(mode or "all").strip().lower()
    return normalized_mode not in {
        "activation_only",
        "baseline_only",
        "no_attention",
        "non_attention",
        "only_tfidf",
    }


def build_dataset_file_map(model_dirname: str, *, require_feature_parquet: bool) -> OrderedDict[str, dict[str, Path | None]]:
    file_map: OrderedDict[str, dict[str, Path | None]] = OrderedDict()
    for env_name, env_dir in ENV_SPECS.items():
        feature_path = (
            resolve_feature_path(env_name, env_dir, model_dirname)
            if require_feature_parquet
            else resolve_feature_path_optional(env_name, env_dir, model_dirname)
        )
        activation_path = DATASET_ROOT / env_dir / model_dirname / ACTIVATION_FILENAME
        if not activation_path.exists():
            raise FileNotFoundError(f"Missing activation h5 for {env_name}: {activation_path}")
        file_map[env_name] = {
            "feature_path": feature_path,
            "activation_path": activation_path,
            "structural_baseline_path": DATASET_ROOT / env_dir / model_dirname / STRUCTURAL_BASELINE_FILENAME,
            "tfidf_cache_dir": DATASET_ROOT / env_dir / model_dirname / TFIDF_CACHE_DIRNAME,
        }
    return file_map


DATASET_FILE_MAP = build_dataset_file_map(
    MODEL_DIRNAME,
    require_feature_parquet=feature_mode_requires_attention(FEATURE_SPACE_MODE),
)


def parse_tfidf_artifact_meta(meta_path: Path) -> dict[str, Any]:
    return json.loads(meta_path.read_text(encoding="utf-8"))


def locate_tfidf_artifact(cache_dir: Path, text_field: str) -> dict[str, Path] | None:
    if not cache_dir.is_dir():
        return None

    candidate_rows: list[dict[str, Any]] = []
    for meta_path in sorted(cache_dir.glob("*.json")):
        try:
            meta = parse_tfidf_artifact_meta(meta_path)
        except Exception:
            continue
        if str(meta.get("feature_type", "")).strip().lower() != "tfidf":
            continue
        if str(meta.get("text_field", "")).strip() != str(text_field):
            continue
        matrix_path = meta_path.with_suffix(".npz")
        vectorizer_path = meta_path.with_suffix(".joblib")
        feature_names_path = meta_path.with_name(f"{meta_path.stem}__feature_names.npy")
        if not matrix_path.exists() or not vectorizer_path.exists() or not feature_names_path.exists():
            continue
        vectorizer_params = meta.get("vectorizer_params", {})
        candidate_rows.append(
            {
                "meta_path": meta_path,
                "matrix_path": matrix_path,
                "vectorizer_path": vectorizer_path,
                "feature_names_path": feature_names_path,
                "meta": meta,
                "is_default_config": (
                    int(vectorizer_params.get("max_features", -1)) == int(getattr(baseline_extractor, "DEFAULT_TFIDF_MAX_FEATURES", 20000))
                    and tuple(vectorizer_params.get("ngram_range", [])) == (
                        int(getattr(baseline_extractor, "DEFAULT_TFIDF_MIN_NGRAM", 1)),
                        int(getattr(baseline_extractor, "DEFAULT_TFIDF_MAX_NGRAM", 2)),
                    )
                    and bool(vectorizer_params.get("lowercase", True))
                    and bool(vectorizer_params.get("sublinear_tf", True))
                    and vectorizer_params.get("stop_words", None) is None
                ),
                "mtime_ns": meta_path.stat().st_mtime_ns,
            }
        )

    if not candidate_rows:
        return None

    candidate_rows.sort(
        key=lambda row: (bool(row["is_default_config"]), int(row["mtime_ns"])),
        reverse=True,
    )
    selected = candidate_rows[0]
    return {
        "meta_path": selected["meta_path"],
        "matrix_path": selected["matrix_path"],
        "vectorizer_path": selected["vectorizer_path"],
        "feature_names_path": selected["feature_names_path"],
    }


def discover_optional_feature_spaces() -> tuple[OrderedDict[str, FeatureSpaceSpec], pd.DataFrame]:
    availability_rows: list[dict[str, Any]] = []
    active_feature_spaces: OrderedDict[str, FeatureSpaceSpec] = OrderedDict(BASE_FEATURE_SPACES)

    structural_ready = all(env_paths["structural_baseline_path"].exists() for env_paths in DATASET_FILE_MAP.values())

    known_tfidf_titles = {
        "last_sentence_text": "Baseline: TF-IDF last sentence",
        "prefix_text": "Baseline: TF-IDF prefix",
        "full_prefix_text": "Baseline: TF-IDF full prefix",
    }
    tfidf_space_specs: OrderedDict[str, dict[str, str]] = OrderedDict()
    for text_field in TFIDF_TEXT_FIELDS:
        feature_space_name = f"baseline_tfidf_{slugify(str(text_field))}"
        tfidf_space_specs[feature_space_name] = {
            "title": known_tfidf_titles.get(str(text_field), f"Baseline: TF-IDF {text_field}"),
            "text_field": str(text_field),
        }
    for feature_space_name, spec in tfidf_space_specs.items():
        text_field = str(spec["text_field"])
        artifacts_present = structural_ready and all(
            locate_tfidf_artifact(env_paths["tfidf_cache_dir"], text_field) is not None
            for env_paths in DATASET_FILE_MAP.values()
        )
        availability_rows.append(
            {
                "feature_space": feature_space_name,
                "feature_space_title": spec["title"],
                "artifact_kind": "tfidf_cache",
                "is_available": bool(artifacts_present),
                "detail": text_field,
            }
        )
        if not artifacts_present:
            continue
        active_feature_spaces[feature_space_name] = FeatureSpaceSpec(
            name=feature_space_name,
            title=str(spec["title"]),
            family_title="baseline",
            uses_attention=False,
            attention_subset_key=None,
            activation_variant=None,
            activation_use_pca=False,
            baseline_variant="tfidf",
            baseline_text_field=text_field,
        )

    availability_df = pd.DataFrame(availability_rows)
    return active_feature_spaces, availability_df


def filter_feature_spaces_for_mode(
    feature_spaces: OrderedDict[str, FeatureSpaceSpec],
    *,
    mode: str,
) -> OrderedDict[str, FeatureSpaceSpec]:
    normalized_mode = str(mode or "all").strip().lower()
    if normalized_mode in {"all", ""}:
        return feature_spaces
    if normalized_mode == "only_tfidf":
        filtered = OrderedDict(
            (feature_space_name, feature_space)
            for feature_space_name, feature_space in feature_spaces.items()
            if feature_space.baseline_variant == "tfidf"
        )
        if not filtered:
            raise ValueError(
                "FEATURE_SPACE_MODE='only_tfidf' was requested, but no TF-IDF feature spaces were discovered. "
                "Check that the TF-IDF cache exists for every environment."
            )
        return filtered
    if normalized_mode == "activation_only":
        filtered = OrderedDict(
            (feature_space_name, feature_space)
            for feature_space_name, feature_space in feature_spaces.items()
            if feature_space.family_title == "activation_only"
        )
        if not filtered:
            raise ValueError("FEATURE_SPACE_MODE='activation_only' found no activation-only feature spaces.")
        return filtered
    if normalized_mode == "baseline_only":
        filtered = OrderedDict(
            (feature_space_name, feature_space)
            for feature_space_name, feature_space in feature_spaces.items()
            if feature_space.family_title == "baseline"
        )
        if not filtered:
            raise ValueError("FEATURE_SPACE_MODE='baseline_only' found no baseline feature spaces.")
        return filtered
    if normalized_mode in {"no_attention", "non_attention"}:
        filtered = OrderedDict(
            (feature_space_name, feature_space)
            for feature_space_name, feature_space in feature_spaces.items()
            if not feature_space.uses_attention
        )
        if not filtered:
            raise ValueError(
                f"FEATURE_SPACE_MODE={mode!r} found no non-attention feature spaces."
            )
        return filtered
    raise ValueError(
        f"Unsupported FEATURE_SPACE_MODE={mode!r}. "
        "Expected one of ['all', 'only_tfidf', 'activation_only', 'baseline_only', 'no_attention', 'non_attention']."
    )


_discovered_feature_spaces, optional_feature_space_availability_df = discover_optional_feature_spaces()
FEATURE_SPACES = filter_feature_spaces_for_mode(
    _discovered_feature_spaces,
    mode=FEATURE_SPACE_MODE,
)
USES_TFIDF_BASELINES = any(
    feature_space.baseline_variant == "tfidf"
    for feature_space in FEATURE_SPACES.values()
)
USES_ATTENTION_FEATURES = any(feature_space.uses_attention for feature_space in FEATURE_SPACES.values())
if USES_ATTENTION_FEATURES:
    missing_feature_envs = [
        env_name
        for env_name, env_paths in DATASET_FILE_MAP.items()
        if env_paths["feature_path"] is None
    ]
    if missing_feature_envs:
        raise ValueError(
            "The selected feature spaces require readable feature parquet files, but none were found for: "
            + ", ".join(sorted(missing_feature_envs))
        )
optional_feature_space_availability_df["selected_by_mode"] = optional_feature_space_availability_df["feature_space"].astype(str).isin(
    set(FEATURE_SPACES.keys())
)
optional_feature_space_availability_df.to_csv(
    OUTPUT_ROOT / "optional_feature_space_availability.csv",
    index=False,
)
display(optional_feature_space_availability_df)

attention_subset_catalog_df = pd.DataFrame(
    [
        {
            "attention_subset_key": subset_spec.key,
            "attention_subset_title": subset_spec.title,
            "transition_mode": subset_spec.transition_mode,
            "metric_names": ", ".join(subset_spec.metric_names),
        }
        for subset_spec in ATTENTION_SUBSETS.values()
    ]
)
attention_subset_catalog_df.to_csv(OUTPUT_ROOT / "attention_subset_catalog.csv", index=False)

feature_space_catalog_df = pd.DataFrame(
    [
        {
            "feature_space": feature_space.name,
            "feature_space_title": feature_space.title,
            "feature_family_group": feature_space.family_title,
            "uses_attention": feature_space.uses_attention,
            "attention_subset_key": feature_space.attention_subset_key or "",
            "attention_subset_title": (
                ATTENTION_SUBSETS[str(feature_space.attention_subset_key)].title
                if feature_space.attention_subset_key is not None
                else ""
            ),
            "activation_variant": feature_space.activation_variant or "",
            "activation_use_pca": feature_space.activation_use_pca,
            "baseline_variant": feature_space.baseline_variant or "",
            "baseline_text_field": feature_space.baseline_text_field or "",
        }
        for feature_space in FEATURE_SPACES.values()
    ]
)
feature_space_catalog_df.to_csv(OUTPUT_ROOT / "feature_space_catalog.csv", index=False)
config_df = pd.concat(
    [
        config_df,
        pd.DataFrame(
            [
                {"setting": "structural_baseline_filename", "value": STRUCTURAL_BASELINE_FILENAME},
                {"setting": "feature_filename_candidates", "value": ", ".join(FEATURE_FILENAME_CANDIDATES)},
                {"setting": "tfidf_cache_dirname", "value": TFIDF_CACHE_DIRNAME},
                {"setting": "tfidf_text_fields", "value": ", ".join(TFIDF_TEXT_FIELDS)},
                {"setting": "feature_space_mode", "value": FEATURE_SPACE_MODE},
                {"setting": "active_feature_spaces", "value": ", ".join(FEATURE_SPACES.keys())},
            ]
        ),
    ],
    ignore_index=True,
)
config_df.to_csv(OUTPUT_ROOT / "config.csv", index=False)


# %%
def count_alpha_words(text: Any) -> int:
    return len(re.findall(r"[A-Za-z]+", str(text or "")))


def load_parquet_with_optional_columns(
    path: Path,
    *,
    required_columns: list[str],
    optional_columns: list[str],
) -> pd.DataFrame:
    requested_columns = list(required_columns) + list(optional_columns)
    try:
        return pd.read_parquet(path, columns=requested_columns).copy()
    except Exception:
        df = pd.read_parquet(path, columns=required_columns).copy()
        for column_name in optional_columns:
            try:
                optional_df = pd.read_parquet(path, columns=[column_name]).copy()
            except Exception:
                continue
            if column_name in optional_df.columns:
                df[column_name] = optional_df[column_name]
        return df


def annotate_prefix_metadata(df: pd.DataFrame, *, env_name: str) -> pd.DataFrame:
    out = df.copy()
    out["env_name"] = env_name
    out["example_id"] = out["example_id"].astype(str)
    out["sentence_idx"] = pd.to_numeric(out["sentence_idx"], errors="coerce")
    out["deception_rate"] = pd.to_numeric(out["deception_rate"], errors="coerce")
    has_num_valid = "num_valid" in out.columns
    has_sentence_text = "sentence_text" in out.columns
    if has_num_valid:
        out["num_valid"] = pd.to_numeric(out["num_valid"], errors="coerce")
    else:
        out["num_valid"] = np.nan
    if has_sentence_text:
        out["sentence_text"] = out["sentence_text"].fillna("").astype(str)
    else:
        out["sentence_text"] = ""
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["example_id", "sentence_idx", "deception_rate"]).copy()
    out["sentence_idx"] = out["sentence_idx"].astype(int)
    out["deception_rate"] = out["deception_rate"].astype(np.float32)
    out = out.sort_values(["example_id", "sentence_idx", "row_idx"], kind="mergesort").reset_index(drop=True)
    out["prev_deception_rate"] = (
        out.groupby("example_id", sort=False)["deception_rate"].shift(1).astype(np.float32)
    )
    out["prev_num_valid"] = out.groupby("example_id", sort=False)["num_valid"].shift(1)
    out["delta_deception_rate"] = (out["deception_rate"] - out["prev_deception_rate"]).astype(np.float32)
    if has_sentence_text:
        out["sentence_alpha_word_count"] = out["sentence_text"].map(count_alpha_words).astype(np.int32)
        usable_sentence_mask = out["sentence_text"].astype(str).str.strip().ne("").to_numpy(dtype=bool, copy=False)
        if EXCLUDE_MULTILINE_SENTENCES:
            usable_sentence_mask &= ~out["sentence_text"].astype(str).str.contains("\n", regex=False).to_numpy(
                dtype=bool,
                copy=False,
            )
        if int(MIN_SENTENCE_ALPHA_WORDS) > 0:
            usable_sentence_mask &= out["sentence_alpha_word_count"].ge(int(MIN_SENTENCE_ALPHA_WORDS)).to_numpy(
                dtype=bool,
                copy=False,
            )
    else:
        out["sentence_alpha_word_count"] = pd.Series(np.nan, index=out.index, dtype=np.float32)
        usable_sentence_mask = np.ones(len(out), dtype=bool)

    if has_num_valid and int(MIN_NUM_VALID) > 0:
        enough_num_valid_mask = (
            out["num_valid"].ge(int(MIN_NUM_VALID)) & out["prev_num_valid"].ge(int(MIN_NUM_VALID))
        ).fillna(False).to_numpy(dtype=bool, copy=False)
    else:
        enough_num_valid_mask = np.ones(len(out), dtype=bool)

    out["passes_commitment_pair_filters"] = usable_sentence_mask & enough_num_valid_mask
    valid_delta = out["delta_deception_rate"].notna()
    for target_name, target_spec in TARGET_SPECS.items():
        out[f"label__{target_name}"] = np.where(
            valid_delta,
            np.asarray(
                target_spec["label_fn"](out["delta_deception_rate"].to_numpy(dtype=np.float32, copy=False)),
                dtype=np.int8,
            ),
            np.nan,
        )
    return out


def build_example_split_map(example_ids: pd.Series, *, seed: int, val_size: float) -> dict[str, str]:
    unique_examples = pd.Series(example_ids.astype(str).unique(), dtype="string")
    if unique_examples.shape[0] < 2:
        raise ValueError("Need at least 2 unique example_id values to build train / val splits.")
    rng = np.random.RandomState(int(seed))
    order = rng.permutation(unique_examples.shape[0])
    val_count = int(round(float(val_size) * unique_examples.shape[0]))
    val_count = max(1, min(unique_examples.shape[0] - 1, val_count))
    val_examples = set(unique_examples.iloc[order[:val_count]].astype(str).tolist())
    return {
        str(example_id): ("val" if str(example_id) in val_examples else "train")
        for example_id in unique_examples.astype(str).tolist()
    }


def split_summary_row(df: pd.DataFrame, *, env_name: str) -> dict[str, Any]:
    train_df = df.loc[df["split"].eq("train")].copy()
    val_df = df.loc[df["split"].eq("val")].copy()
    train_modeled_df = train_df.loc[
        train_df["delta_deception_rate"].notna() & train_df["passes_commitment_pair_filters"].astype(bool)
    ].copy()
    val_modeled_df = val_df.loc[
        val_df["delta_deception_rate"].notna() & val_df["passes_commitment_pair_filters"].astype(bool)
    ].copy()
    row: dict[str, Any] = {
        "env_name": env_name,
        "rows": int(len(df)),
        "examples": int(df["example_id"].nunique()),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "train_delta_rows": int(train_df["delta_deception_rate"].notna().sum()),
        "val_delta_rows": int(val_df["delta_deception_rate"].notna().sum()),
        "train_modeled_rows": int(len(train_modeled_df)),
        "val_modeled_rows": int(len(val_modeled_df)),
        "train_modeled_examples": int(train_modeled_df["example_id"].nunique()),
        "val_modeled_examples": int(val_modeled_df["example_id"].nunique()),
    }
    for target_name in TARGET_SPECS:
        row[f"train_pos_rate__{target_name}"] = float(
            pd.to_numeric(train_modeled_df[f"label__{target_name}"], errors="coerce").mean()
        )
        row[f"val_pos_rate__{target_name}"] = float(
            pd.to_numeric(val_modeled_df[f"label__{target_name}"], errors="coerce").mean()
        )
    return row


def load_feature_metadata(feature_path: Path, env_name: str) -> pd.DataFrame:
    df = load_parquet_with_optional_columns(
        feature_path,
        required_columns=["example_id", "sentence_idx", "deception_rate"],
        optional_columns=["sentence_text", "num_valid"],
    )
    missing_optional_columns = [column_name for column_name in ("sentence_text", "num_valid") if column_name not in df.columns]
    if "num_valid" in missing_optional_columns and int(MIN_NUM_VALID) > 0:
        print(
            f"[warn] {env_name}: feature parquet is missing `num_valid`; "
            "skipping the MIN_NUM_VALID-style filter for this environment."
        )
    if "sentence_text" in missing_optional_columns and (
        bool(EXCLUDE_MULTILINE_SENTENCES) or int(MIN_SENTENCE_ALPHA_WORDS) > 0
    ):
        print(
            f"[warn] {env_name}: feature parquet is missing `sentence_text`; "
            "skipping the sentence-text usability filters for this environment."
        )
    df["row_idx"] = np.arange(len(df), dtype=np.int64)
    return annotate_prefix_metadata(df, env_name=env_name)


def load_structural_baseline_metadata(structural_path: Path, env_name: str) -> pd.DataFrame:
    df = pd.read_parquet(structural_path, columns=["example_id", "sentence_idx"]).copy()
    df["env_name"] = env_name
    df["example_id"] = df["example_id"].astype(str)
    df["sentence_idx"] = pd.to_numeric(df["sentence_idx"], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["example_id", "sentence_idx"]).copy()
    df["sentence_idx"] = df["sentence_idx"].astype(int)
    df["row_idx"] = np.arange(len(df), dtype=np.int64)
    return df.reset_index(drop=True)


def load_activation_metadata(activation_path: Path, env_name: str) -> pd.DataFrame:
    with h5py.File(activation_path, "r") as f:
        example_ids = pd.Series(f["example_id"].asstr()[:], dtype="string")
        sentence_idx = pd.Series(np.asarray(f["sentence_idx"][:], dtype=np.int64))
        deception_rate = pd.Series(np.asarray(f["deception_rate"][:], dtype=np.float32))
    df = pd.DataFrame(
        {
            "example_id": example_ids,
            "sentence_idx": sentence_idx,
            "deception_rate": deception_rate,
            "row_idx": np.arange(len(example_ids), dtype=np.int64),
        }
    )
    return annotate_prefix_metadata(df, env_name=env_name)


@dataclass
class ActivationEnvStore:
    env_name: str
    activation_path: Path
    metadata_df: pd.DataFrame

    def load_matrix(self, row_indices: np.ndarray, *, variant: str) -> np.ndarray:
        row_indices = np.asarray(row_indices, dtype=np.int64)
        if row_indices.size == 0:
            return np.zeros((0, 0), dtype=np.float32)

        sort_order = np.argsort(row_indices, kind="mergesort")
        sorted_indices = row_indices[sort_order]
        unsort = np.empty_like(sort_order)
        unsort[sort_order] = np.arange(sort_order.shape[0], dtype=np.int64)

        with h5py.File(self.activation_path, "r") as f:
            activations = np.asarray(f["activations"][sorted_indices], dtype=np.float32)
            activation_mask = np.asarray(f["activation_mask"][sorted_indices], dtype=bool)

        current = activations[:, 0, :]
        if variant == "final_sentence":
            feature_matrix = current
        elif variant == "delta_last2":
            prev = activations[:, 1, :]
            prev_valid = activation_mask[:, 1]
            feature_matrix = current - np.where(prev_valid[:, None], prev, 0.0)
        elif variant == "delta_prev4mean":
            prev_vectors = activations[:, 1:5, :]
            prev_mask = activation_mask[:, 1:5]
            prev_count = prev_mask.sum(axis=1, keepdims=True)
            prev_sum = (prev_vectors * prev_mask[:, :, None]).sum(axis=1)
            prev_mean = np.divide(
                prev_sum,
                np.maximum(prev_count, 1),
                out=np.zeros_like(current),
                where=prev_count > 0,
            )
            feature_matrix = current - prev_mean
        else:
            raise ValueError(f"Unsupported activation variant: {variant!r}")

        return np.asarray(feature_matrix[unsort], dtype=np.float32)


feature_metadata_by_env: OrderedDict[str, pd.DataFrame] = OrderedDict()
activation_metadata_by_env: OrderedDict[str, pd.DataFrame] = OrderedDict()
structural_metadata_by_env: OrderedDict[str, pd.DataFrame] = OrderedDict()
activation_stores: OrderedDict[str, ActivationEnvStore] = OrderedDict()
split_summary_rows: list[dict[str, Any]] = []

for env_name, env_paths in maybe_tqdm(
    DATASET_FILE_MAP.items(),
    desc="Load metadata",
    total=len(DATASET_FILE_MAP),
    disable=DISABLE_TQDM,
):
    feature_metadata_df = load_feature_metadata(env_paths["feature_path"], env_name)
    split_map = build_example_split_map(feature_metadata_df["example_id"], seed=RANDOM_SEED, val_size=VAL_SIZE)
    feature_metadata_df["split"] = feature_metadata_df["example_id"].map(split_map).astype("string")
    feature_metadata_by_env[env_name] = feature_metadata_df

    activation_metadata_df = load_activation_metadata(env_paths["activation_path"], env_name)
    activation_metadata_df["split"] = activation_metadata_df["example_id"].map(split_map).astype("string")
    if activation_metadata_df["split"].isna().any():
        raise ValueError(f"{env_name} activation metadata contains example IDs absent from the feature split map.")
    activation_metadata_by_env[env_name] = activation_metadata_df
    activation_stores[env_name] = ActivationEnvStore(
        env_name=env_name,
        activation_path=env_paths["activation_path"],
        metadata_df=activation_metadata_df,
    )

    if USES_TFIDF_BASELINES and env_paths["structural_baseline_path"].exists():
        structural_metadata_df = load_structural_baseline_metadata(env_paths["structural_baseline_path"], env_name)
        structural_metadata_df["split"] = structural_metadata_df["example_id"].map(split_map).astype("string")
        unmatched_structural_examples = sorted(
            {
                str(example_id)
                for example_id in structural_metadata_df.loc[structural_metadata_df["split"].isna(), "example_id"].astype(str).tolist()
            }
        )
        if unmatched_structural_examples:
            print(
                f"[warn] {env_name}: ignoring {len(unmatched_structural_examples)} companion structural example IDs "
                "absent from the feature split map. "
                "This usually means those examples produced TF-IDF companion rows but no modeling rows "
                "in prefix_deception_features.parquet.tmp."
            )
        structural_metadata_by_env[env_name] = structural_metadata_df

    split_summary_rows.append(split_summary_row(feature_metadata_df, env_name=env_name))

split_summary_df = pd.DataFrame(split_summary_rows).sort_values("env_name").reset_index(drop=True)
display(split_summary_df)
split_summary_df.to_csv(OUTPUT_ROOT / "split_summary.csv", index=False)

sample_activation_store = next(iter(activation_stores.values()))
sample_activation_row_idx = sample_activation_store.metadata_df["row_idx"].iloc[:1].to_numpy(dtype=np.int64, copy=False)
ACTIVATION_HIDDEN_DIM = int(
    sample_activation_store.load_matrix(sample_activation_row_idx, variant="final_sentence").shape[1]
)


split_cache_by_env: dict[str, dict[str, Any]] = {}
for env_name, metadata_df in feature_metadata_by_env.items():
    valid_delta = metadata_df["delta_deception_rate"].notna().to_numpy(dtype=bool, copy=False)
    passes_commitment_pair_filters = metadata_df["passes_commitment_pair_filters"].to_numpy(dtype=bool, copy=False)
    train_mask = metadata_df["split"].eq("train").to_numpy(dtype=bool, copy=False) & valid_delta & passes_commitment_pair_filters
    val_mask = metadata_df["split"].eq("val").to_numpy(dtype=bool, copy=False) & valid_delta & passes_commitment_pair_filters
    bundle: dict[str, Any] = {
        "train_mask": train_mask,
        "val_mask": val_mask,
        "train_row_idx": metadata_df.loc[train_mask, "row_idx"].to_numpy(dtype=np.int64, copy=False),
        "val_row_idx": metadata_df.loc[val_mask, "row_idx"].to_numpy(dtype=np.int64, copy=False),
    }
    for target_name in TARGET_SPECS:
        bundle[f"y_train__{target_name}"] = metadata_df.loc[train_mask, f"label__{target_name}"].to_numpy(dtype=np.int8, copy=False)
        bundle[f"y_val__{target_name}"] = metadata_df.loc[val_mask, f"label__{target_name}"].to_numpy(dtype=np.int8, copy=False)

    structural_metadata_df = structural_metadata_by_env.get(env_name)
    if structural_metadata_df is not None:
        feature_keys_df = metadata_df.loc[:, ["example_id", "sentence_idx"]].copy()
        feature_keys_df["feature_row_idx"] = metadata_df["row_idx"].to_numpy(dtype=np.int64, copy=False)
        structural_keys_df = structural_metadata_df.loc[:, ["example_id", "sentence_idx"]].copy()
        structural_keys_df["structural_row_idx"] = structural_metadata_df["row_idx"].to_numpy(dtype=np.int64, copy=False)
        aligned_key_df = feature_keys_df.merge(
            structural_keys_df,
            on=["example_id", "sentence_idx"],
            how="left",
            validate="one_to_one",
        )
        missing_structural_mask = aligned_key_df["structural_row_idx"].isna().to_numpy(dtype=bool, copy=False)
        if missing_structural_mask.any():
            missing_key_df = aligned_key_df.loc[
                missing_structural_mask,
                ["example_id", "sentence_idx"],
            ].head(5)
            print(
                f"[warn] {env_name}: companion structural / TF-IDF rows are missing for "
                f"{int(missing_structural_mask.sum())} modeling rows; zero-filling those baseline features. "
                f"feature_rows={len(metadata_df)}, structural_rows={len(structural_metadata_df)}, "
                f"train_missing={int(np.count_nonzero(missing_structural_mask & train_mask))}, "
                f"val_missing={int(np.count_nonzero(missing_structural_mask & val_mask))}, "
                f"sample_missing={missing_key_df.to_dict(orient='records')}"
            )
        aligned_key_df = aligned_key_df.sort_values("feature_row_idx", kind="mergesort").reset_index(drop=True)
        structural_row_idx = aligned_key_df["structural_row_idx"].fillna(-1).to_numpy(dtype=np.int64, copy=False)
        bundle["train_structural_row_idx"] = structural_row_idx[train_mask]
        bundle["val_structural_row_idx"] = structural_row_idx[val_mask]

    split_cache_by_env[env_name] = bundle


# %%
ATTN_ROOT_RE = re.compile(
    r"^(?:(?P<transition_prefix>"
    + "|".join(re.escape(prefix) for prefix in ATTENTION_TRANSITION_PREFIXES)
    + r")_)?(?P<metric>"
    + "|".join(re.escape(metric) for metric in ATTENTION_METRIC_NAMES)
    + r")_(?P<head_summary>"
    + "|".join(re.escape(name) for name in HEAD_SUMMARY_NAMES)
    + r")$"
)


def split_layer_positions(layer_count: int) -> dict[str, np.ndarray]:
    if layer_count < 1:
        raise ValueError(f"Expected at least one layer, got {layer_count}.")
    parts = np.array_split(np.arange(layer_count, dtype=np.int64), len(BAND_NAMES))
    return {band_name: indices for band_name, indices in zip(BAND_NAMES, parts, strict=True)}


def reduce_layer_block(feature_array: np.ndarray) -> dict[str, np.ndarray]:
    feature_array = np.asarray(feature_array, dtype=np.float32)
    band_indices = split_layer_positions(int(feature_array.shape[1]))
    out: dict[str, np.ndarray] = {}
    for band_name, indices in band_indices.items():
        band_values = feature_array[:, indices]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            out[f"{band_name}__mean"] = np.nanmean(band_values, axis=1, dtype=np.float32)
            out[f"{band_name}__min"] = np.nanmin(band_values, axis=1)
            out[f"{band_name}__max"] = np.nanmax(band_values, axis=1)
            out[f"{band_name}__std"] = np.nanstd(band_values, axis=1, dtype=np.float32)
    return out


layer_root_map, _ = build_common_layer_roots(
    OrderedDict((env_name, env_paths["feature_path"]) for env_name, env_paths in DATASET_FILE_MAP.items())
)
attention_layer_roots = OrderedDict(
    (root_name, columns)
    for root_name, columns in layer_root_map.items()
    if ATTN_ROOT_RE.fullmatch(root_name) is not None
)
if not attention_layer_roots:
    raise ValueError("No shared attention layer roots found across environments.")


def build_attention_reduction_lookup() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for root_name, columns in attention_layer_roots.items():
        match = ATTN_ROOT_RE.fullmatch(root_name)
        if match is None:
            continue
        layer_count = len(columns)
        metric_name = str(match.group("metric"))
        transition_prefix = str(match.group("transition_prefix") or "")
        is_transition = transition_prefix != ""
        metric_group = ATTENTION_METRIC_GROUP_BY_NAME[metric_name]
        attention_feature_group = f"{metric_group}_transition" if is_transition else metric_group
        for band_name in BAND_NAMES:
            for stat_name in BAND_STAT_NAMES:
                rows.append(
                    {
                        "feature": f"{root_name}__band_{band_name}__{stat_name}",
                        "feature_root": root_name,
                        "transition_prefix": transition_prefix,
                        "is_transition": bool(is_transition),
                        "metric_name": metric_name,
                        "metric_group": metric_group,
                        "attention_feature_group": attention_feature_group,
                        "head_summary": match.group("head_summary"),
                        "band": band_name,
                        "band_stat": stat_name,
                        "layer_count": int(layer_count),
                        "family": attention_feature_group,
                    }
                )
    return pd.DataFrame(rows)


attention_reduction_lookup_df = build_attention_reduction_lookup()
display(attention_reduction_lookup_df.head(20))
attention_reduction_lookup_df.to_csv(OUTPUT_ROOT / "attention_reduction_lookup.csv", index=False)
expected_attention_reduced_columns = set(attention_reduction_lookup_df["feature"].astype(str).tolist())


def build_attention_subset_lookup(
    feature_lookup_df: pd.DataFrame,
    *,
    subset_spec: AttentionSubsetSpec,
) -> pd.DataFrame:
    subset_mask = feature_lookup_df["metric_name"].isin(subset_spec.metric_names)
    is_transition = feature_lookup_df["is_transition"].to_numpy(dtype=bool, copy=False)
    if subset_spec.transition_mode == "base_only":
        subset_mask &= ~is_transition
    elif subset_spec.transition_mode == "transition_only":
        subset_mask &= is_transition
    elif subset_spec.transition_mode != "all":
        raise ValueError(f"Unsupported attention transition mode: {subset_spec.transition_mode!r}")

    subset_df = feature_lookup_df.loc[subset_mask].copy().reset_index(drop=True)
    if subset_df.empty:
        raise ValueError(f"Attention subset {subset_spec.key!r} did not match any reduced features.")
    return subset_df


def build_attention_reduced_env_frame(
    env_name: str,
    *,
    feature_path: Path,
    metadata_df: pd.DataFrame,
    root_batch_size: int,
) -> pd.DataFrame:
    row_ids = metadata_df["row_idx"].to_numpy(dtype=np.int64, copy=False)
    arrays: dict[str, np.ndarray] = {}
    ordered_roots = list(attention_layer_roots.items())
    iterator = maybe_tqdm(
        range(0, len(ordered_roots), int(root_batch_size)),
        desc=f"Reduce attention:{env_name}",
        total=int(math.ceil(len(ordered_roots) / int(root_batch_size))),
        disable=DISABLE_TQDM,
    )
    for start_idx in iterator:
        batch_items = ordered_roots[start_idx : start_idx + int(root_batch_size)]
        batch_columns = [column_name for _, columns in batch_items for column_name in columns]
        batch_df = pd.read_parquet(feature_path, columns=batch_columns)
        batch_df = batch_df.iloc[row_ids].reset_index(drop=True)
        for root_name, columns in batch_items:
            feature_array = (
                batch_df.loc[:, columns]
                .apply(pd.to_numeric, errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .to_numpy(dtype=np.float32, copy=False)
            )
            reduced = reduce_layer_block(feature_array)
            for key, values in reduced.items():
                arrays[f"{root_name}__band_{key}"] = np.asarray(values, dtype=np.float32)
        del batch_df
        gc.collect()

    return pd.concat([metadata_df.reset_index(drop=True), pd.DataFrame(arrays)], axis=1)


def feature_columns_for_env(env_df: pd.DataFrame) -> list[str]:
    reserved = {
        "env_name",
        "example_id",
        "sentence_idx",
        "sentence_text",
        "sentence_alpha_word_count",
        "deception_rate",
        "num_valid",
        "prev_num_valid",
        "row_idx",
        "prev_deception_rate",
        "delta_deception_rate",
        "passes_commitment_pair_filters",
        "split",
        *(f"label__{target_name}" for target_name in TARGET_SPECS),
    }
    return [column for column in env_df.columns if column not in reserved]


attention_envs: OrderedDict[str, pd.DataFrame] = OrderedDict()
for env_name, env_paths in maybe_tqdm(
    DATASET_FILE_MAP.items(),
    desc="Build env frames",
    total=len(DATASET_FILE_MAP),
    disable=DISABLE_TQDM,
):
    attention_cache_path = CACHE_DIR / f"{slugify(env_name)}__attention_reduced.parquet"
    should_rebuild_cache = bool(FORCE_REBUILD_REDUCTIONS)
    if attention_cache_path.exists() and not should_rebuild_cache:
        attention_df = pd.read_parquet(attention_cache_path)
        missing_reduced_columns = sorted(expected_attention_reduced_columns - set(attention_df.columns.astype(str).tolist()))
        if missing_reduced_columns:
            print(
                f"Rebuilding stale attention cache for {env_name}: "
                f"{len(missing_reduced_columns)} reduced columns missing."
            )
            should_rebuild_cache = True
    if not attention_cache_path.exists() or should_rebuild_cache:
        attention_df = build_attention_reduced_env_frame(
            env_name,
            feature_path=env_paths["feature_path"],
            metadata_df=feature_metadata_by_env[env_name],
            root_batch_size=int(ROOT_BATCH_SIZE),
        )
        attention_df.to_parquet(attention_cache_path, index=False)
    attention_envs[env_name] = attention_df

attention_feature_names = feature_columns_for_env(next(iter(attention_envs.values())))
attention_lookup = attention_reduction_lookup_df.loc[
    attention_reduction_lookup_df["feature"].isin(attention_feature_names)
].copy()
attention_lookup_by_subset = OrderedDict(
    (
        subset_key,
        build_attention_subset_lookup(attention_lookup, subset_spec=subset_spec),
    )
    for subset_key, subset_spec in ATTENTION_SUBSETS.items()
)
attention_subset_feature_counts_df = pd.DataFrame(
    [
        {
            "attention_subset_key": subset_key,
            "attention_subset_title": ATTENTION_SUBSETS[subset_key].title,
            "reduced_feature_count": len(subset_lookup_df),
        }
        for subset_key, subset_lookup_df in attention_lookup_by_subset.items()
    ]
)
attention_subset_feature_counts_df.to_csv(OUTPUT_ROOT / "attention_subset_feature_counts.csv", index=False)
feature_space_summary_df = pd.DataFrame(
    [
        {"setting": "attention_layer_root_count", "value": len(attention_layer_roots)},
        {"setting": "attention_reduced_feature_count", "value": len(attention_feature_names)},
        {"setting": "attention_subset_count", "value": len(ATTENTION_SUBSETS)},
        {"setting": "activation_hidden_dim", "value": ACTIVATION_HIDDEN_DIM},
        {"setting": "activation_pca_dim", "value": ACTIVATION_PCA_DIM},
        {"setting": "feature_size_grid", "value": ",".join(str(value) for value in FEATURE_SIZE_GRID)},
    ]
)
display(feature_space_summary_df)
feature_space_summary_df.to_csv(OUTPUT_ROOT / "feature_space_summary.csv", index=False)


# %%
def compute_standardized_effects(x_array: np.ndarray, y_array: np.ndarray) -> np.ndarray:
    x_array = np.asarray(x_array, dtype=np.float32)
    y_array = np.asarray(y_array, dtype=np.int8)
    pos_mask = y_array == 1
    neg_mask = y_array == 0
    if not pos_mask.any() or not neg_mask.any():
        return np.full(x_array.shape[1], np.nan, dtype=np.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(x_array, axis=0, dtype=np.float32)
        std = np.nanstd(x_array, axis=0, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    std[~np.isfinite(std) | (np.abs(std) < 1e-8)] = np.nan

    z_array = (x_array - mean[None, :]) / std[None, :]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        pos_mean = np.nanmean(z_array[pos_mask], axis=0, dtype=np.float32)
        neg_mean = np.nanmean(z_array[neg_mask], axis=0, dtype=np.float32)
    return np.asarray(pos_mean - neg_mean, dtype=np.float32)


def build_consistency_ranking(
    env_frames: OrderedDict[str, pd.DataFrame],
    *,
    feature_names: list[str],
    feature_lookup_df: pd.DataFrame,
    target_name: str,
) -> pd.DataFrame:
    effect_rows: list[pd.Series] = []
    target_col = f"label__{target_name}"

    for env_name, env_df in env_frames.items():
        train_df = env_df.loc[env_df["split"].eq("train") & env_df[target_col].notna()].copy()
        y_train = train_df[target_col].to_numpy(dtype=np.int8, copy=False)
        if np.unique(y_train).size < 2:
            raise ValueError(f"{env_name} train split does not contain both classes for {target_name}.")
        x_array = (
            train_df.loc[:, feature_names]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .to_numpy(dtype=np.float32, copy=False)
        )
        effect_rows.append(pd.Series(compute_standardized_effects(x_array, y_train), index=feature_names, name=env_name, dtype=np.float32))

    effects_df = pd.DataFrame(effect_rows).T.reset_index().rename(columns={"index": "feature"})
    effects_df = feature_lookup_df.merge(effects_df, on="feature", how="left", validate="one_to_one")
    effects_df = effects_df.rename(columns={env_name: f"{slugify(env_name)}_effect" for env_name in ENV_ORDER})

    effect_cols = [f"{slugify(env_name)}_effect" for env_name in ENV_ORDER]
    for effect_col in effect_cols:
        effects_df[f"{effect_col}_abs"] = effects_df[effect_col].abs()

    effect_array = effects_df[effect_cols].to_numpy(dtype=np.float32, copy=False)
    abs_array = effects_df[[f"{col}_abs" for col in effect_cols]].to_numpy(dtype=np.float32, copy=False)
    positive_all = np.all(effect_array > 0.0, axis=1)
    negative_all = np.all(effect_array < 0.0, axis=1)
    effects_df["same_sign_all"] = positive_all | negative_all
    effects_df["sign_direction"] = np.select([positive_all, negative_all], ["positive", "negative"], default="mixed")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        effects_df["min_abs_effect"] = np.nanmin(abs_array, axis=1)
        effects_df["mean_abs_effect"] = np.nanmean(abs_array, axis=1)
        effects_df["median_abs_effect"] = np.nanmedian(abs_array, axis=1)
        effects_df["std_effect"] = np.nanstd(effect_array, axis=1)

    effects_df["consistency_score"] = (
        effects_df["min_abs_effect"]
        + (0.35 * effects_df["mean_abs_effect"])
        - (0.25 * effects_df["std_effect"])
    )
    effects_df = effects_df.replace([np.inf, -np.inf], np.nan)
    effects_df = effects_df.dropna(subset=["min_abs_effect", "mean_abs_effect", "std_effect"]).copy()
    effects_df = effects_df.sort_values(
        ["consistency_score", "min_abs_effect", "mean_abs_effect", "std_effect"],
        ascending=[False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    effects_df["global_rank"] = np.arange(1, len(effects_df) + 1, dtype=int)
    effects_df["target_name"] = target_name
    return effects_df


def select_ranked_attention_pool(ranking_df: pd.DataFrame) -> list[str]:
    ordered_df = ranking_df.copy().reset_index(drop=True)
    if ordered_df.empty:
        raise ValueError("Attention ranking dataframe is empty.")
    return ordered_df["feature"].astype(str).tolist()


def make_activation_lookup(
    *,
    space_name: str,
    variant: str,
    use_pca: bool,
    hidden_dim: int,
    pca_dim: int,
) -> pd.DataFrame:
    if use_pca:
        feature_names = [f"{space_name}__pc_{idx + 1:03d}" for idx in range(int(pca_dim))]
        metric_name = f"{variant}__pca"
    else:
        feature_names = [f"{space_name}__dim_{idx:04d}" for idx in range(int(hidden_dim))]
        metric_name = f"{variant}__raw"
    return pd.DataFrame(
        [
            {
                "feature": feature_name,
                "feature_root": space_name,
                "transition_prefix": "",
                "is_transition": False,
                "metric_name": metric_name,
                "metric_group": "",
                "attention_feature_group": "",
                "head_summary": "",
                "band": "",
                "band_stat": "",
                "layer_count": pd.NA,
                "family": "activation",
            }
            for feature_name in feature_names
        ]
    )


def make_generic_feature_lookup(
    *,
    space_name: str,
    feature_names: list[str],
    feature_root: str,
    metric_name: str,
    family_resolver: Any,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature_name in feature_names:
        rows.append(
            {
                "feature": str(feature_name),
                "feature_root": str(feature_root),
                "transition_prefix": "",
                "is_transition": False,
                "metric_name": str(metric_name),
                "metric_group": "",
                "attention_feature_group": "",
                "head_summary": "",
                "band": "",
                "band_stat": "",
                "layer_count": pd.NA,
                "family": str(family_resolver(str(feature_name))),
            }
        )
    return pd.DataFrame(rows)


def make_activation_ranking_df(
    *,
    target_name: str,
    feature_lookup_df: pd.DataFrame,
    start_rank: int = 1,
) -> pd.DataFrame:
    out = feature_lookup_df.copy().reset_index(drop=True)
    for env_name in ENV_ORDER:
        effect_col = f"{slugify(env_name)}_effect"
        out[effect_col] = np.nan
        out[f"{effect_col}_abs"] = np.nan
    out["same_sign_all"] = pd.NA
    out["sign_direction"] = "not_ranked"
    out["min_abs_effect"] = np.nan
    out["mean_abs_effect"] = np.nan
    out["median_abs_effect"] = np.nan
    out["std_effect"] = np.nan
    out["consistency_score"] = np.nan
    out["global_rank"] = np.arange(int(start_rank), int(start_rank) + len(out), dtype=int)
    out["target_name"] = target_name
    return out


def make_placeholder_ranking_df(
    *,
    target_name: str,
    feature_lookup_df: pd.DataFrame,
    start_rank: int = 1,
) -> pd.DataFrame:
    return make_activation_ranking_df(
        target_name=target_name,
        feature_lookup_df=feature_lookup_df,
        start_rank=start_rank,
    )


def safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int8)
    y_score = np.asarray(y_score, dtype=np.float32)
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def safe_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int8)
    y_score = np.asarray(y_score, dtype=np.float32)
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def finite_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float32)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite, dtype=np.float32))


def finite_min(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float32)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float("nan")
    return float(np.min(finite))


def finite_std(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float32)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float("nan")
    return float(np.std(finite, dtype=np.float32))


def serialize_candidate_params(candidate_params: dict[str, Any]) -> str:
    return json.dumps(candidate_params, sort_keys=True)


def build_model_candidate_specs(y_train: np.ndarray) -> list[dict[str, Any]]:
    y_train = np.asarray(y_train, dtype=np.int8)
    if MODEL_FAMILY == "logreg":
        return [
            {
                "candidate_key": f"c={float(LOGREG_C):g}",
                "candidate_label": f"C={float(LOGREG_C):g}",
                "candidate_complexity": float(LOGREG_C),
                "chosen_c": float(LOGREG_C),
                "candidate_max_depth": np.nan,
                "candidate_params": {"C": float(LOGREG_C)},
            }
        ]

    if MODEL_FAMILY == "xgboost":
        pos_count = int((y_train == 1).sum())
        neg_count = int((y_train == 0).sum())
        scale_pos_weight = float(neg_count / max(pos_count, 1))
        return [
            {
                "candidate_key": f"max_depth={int(XGB_MAX_DEPTH)}",
                "candidate_label": f"max_depth={int(XGB_MAX_DEPTH)}",
                "candidate_complexity": float(XGB_MAX_DEPTH),
                "chosen_c": np.nan,
                "candidate_max_depth": int(XGB_MAX_DEPTH),
                "candidate_params": {
                    "objective": "binary:logistic",
                    "n_estimators": int(XGB_N_ESTIMATORS),
                    "max_depth": int(XGB_MAX_DEPTH),
                    "learning_rate": float(XGB_LEARNING_RATE),
                    "subsample": float(XGB_SUBSAMPLE),
                    "colsample_bytree": float(XGB_COLSAMPLE_BYTREE),
                    "reg_lambda": float(XGB_REG_LAMBDA),
                    "min_child_weight": float(XGB_MIN_CHILD_WEIGHT),
                    "gamma": float(XGB_GAMMA),
                    "scale_pos_weight": scale_pos_weight,
                    "tree_method": "hist",
                    "eval_metric": "logloss",
                    "importance_type": XGB_IMPORTANCE_TYPE,
                    "random_state": int(RANDOM_SEED),
                    "n_jobs": int(XGB_N_JOBS),
                    "verbosity": 0,
                },
            }
        ]

    raise ValueError(f"Unsupported MODEL_FAMILY={MODEL_FAMILY!r}")


@dataclass
class FittedBinaryModel:
    estimator: Pipeline
    model_family: str
    model_family_title: str
    model_weight_kind: str
    candidate_key: str
    candidate_label: str
    candidate_complexity: float
    candidate_params: dict[str, Any]
    chosen_c: float | None
    candidate_max_depth: int | None
    decision_threshold: float
    validation_metrics: dict[str, Any]


def build_estimator(*, candidate_params: dict[str, Any], input_is_sparse: bool = False) -> Pipeline:
    if MODEL_FAMILY == "logreg":
        if input_is_sparse:
            return Pipeline(
                [
                    (
                        "model",
                        LogisticRegression(
                            C=float(candidate_params["C"]),
                            class_weight="balanced",
                            max_iter=4000,
                            solver="liblinear",
                            random_state=int(RANDOM_SEED),
                        ),
                    ),
                ]
            )
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=float(candidate_params["C"]),
                        class_weight="balanced",
                        max_iter=4000,
                        solver="liblinear",
                        random_state=int(RANDOM_SEED),
                    ),
                ),
            ]
        )

    if MODEL_FAMILY == "xgboost":
        if XGBClassifier is None:
            raise ImportError("xgboost is unavailable, but MODEL_FAMILY='xgboost' was requested.")
        if input_is_sparse:
            return Pipeline(
                [
                    ("model", XGBClassifier(**candidate_params)),
                ]
            )
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", XGBClassifier(**candidate_params)),
            ]
        )

    raise ValueError(f"Unsupported MODEL_FAMILY={MODEL_FAMILY!r}")


def fit_candidate_classifiers(
    x_train: Any,
    y_train: np.ndarray,
    x_val: Any,
    y_val: np.ndarray,
) -> tuple[list[FittedBinaryModel], pd.DataFrame]:
    candidate_rows: list[dict[str, Any]] = []
    fitted_models: list[FittedBinaryModel] = []
    input_is_sparse = bool(sp.issparse(x_train) or sp.issparse(x_val))

    for candidate_spec in build_model_candidate_specs(y_train):
        estimator = build_estimator(
            candidate_params=dict(candidate_spec["candidate_params"]),
            input_is_sparse=input_is_sparse,
        )
        estimator.fit(x_train, y_train)
        val_scores = estimator.predict_proba(x_val)[:, 1].astype(np.float32)
        decision_threshold = choose_decision_threshold(
            y_val,
            val_scores,
            default_threshold=0.5,
            mode=DECISION_THRESHOLD_MODE,
        )
        metrics = summarize_score_metrics(y_val, val_scores, decision_threshold=decision_threshold)
        candidate_rows.append(
            {
                "model_family": MODEL_FAMILY,
                "model_family_title": MODEL_FAMILY_TITLE,
                "model_weight_kind": MODEL_WEIGHT_KIND,
                "candidate_key": str(candidate_spec["candidate_key"]),
                "candidate_label": str(candidate_spec["candidate_label"]),
                "candidate_complexity": float(candidate_spec["candidate_complexity"]),
                "candidate_params_json": serialize_candidate_params(candidate_spec["candidate_params"]),
                "candidate_c": (
                    float(candidate_spec["chosen_c"])
                    if pd.notna(candidate_spec["chosen_c"])
                    else np.nan
                ),
                "candidate_max_depth": (
                    int(candidate_spec["candidate_max_depth"])
                    if pd.notna(candidate_spec["candidate_max_depth"])
                    else np.nan
                ),
                "decision_threshold": float(decision_threshold),
                **metrics,
            }
        )
        fitted_models.append(
            FittedBinaryModel(
                estimator=estimator,
                model_family=MODEL_FAMILY,
                model_family_title=MODEL_FAMILY_TITLE,
                model_weight_kind=MODEL_WEIGHT_KIND,
                candidate_key=str(candidate_spec["candidate_key"]),
                candidate_label=str(candidate_spec["candidate_label"]),
                candidate_complexity=float(candidate_spec["candidate_complexity"]),
                candidate_params=dict(candidate_spec["candidate_params"]),
                chosen_c=(
                    float(candidate_spec["chosen_c"])
                    if pd.notna(candidate_spec["chosen_c"])
                    else None
                ),
                candidate_max_depth=(
                    int(candidate_spec["candidate_max_depth"])
                    if pd.notna(candidate_spec["candidate_max_depth"])
                    else None
                ),
                decision_threshold=float(decision_threshold),
                validation_metrics=metrics,
            )
        )

    if not fitted_models:
        raise RuntimeError("fit_candidate_classifiers failed to produce any models.")
    return fitted_models, pd.DataFrame(candidate_rows)


def extract_feature_weights(
    fitted_model: FittedBinaryModel,
    *,
    feature_names: list[str],
    target_name: str,
    feature_space: str,
    train_env: str,
) -> pd.DataFrame:
    model = fitted_model.estimator.named_steps["model"]
    if fitted_model.model_family == "logreg":
        feature_weight = np.asarray(model.coef_, dtype=np.float32).reshape(-1)
    elif fitted_model.model_family == "xgboost":
        feature_weight = np.asarray(model.feature_importances_, dtype=np.float32).reshape(-1)
    else:
        raise ValueError(f"Unsupported model family: {fitted_model.model_family!r}")
    if feature_weight.shape[0] != len(feature_names):
        raise ValueError(
            "Feature-weight length mismatch: "
            f"expected {len(feature_names)}, got {feature_weight.shape[0]}"
        )
    out = pd.DataFrame(
        {
            "target_name": target_name,
            "feature_space": feature_space,
            "train_env": train_env,
            "feature": feature_names,
            "model_family": fitted_model.model_family,
            "model_family_title": fitted_model.model_family_title,
            "feature_weight_kind": fitted_model.model_weight_kind,
            "feature_weight": feature_weight,
        }
    )
    out["abs_feature_weight"] = out["feature_weight"].abs()
    out["coefficient"] = out["feature_weight"]
    out["abs_coefficient"] = out["abs_feature_weight"]
    return out


def build_model_selection_key(
    *,
    objective: str,
    oracle_mean_ood_auroc: float,
    oracle_mean_ood_average_precision: float,
    source_val_auroc: float,
    source_val_average_precision: float,
    source_val_balanced_accuracy: float,
    feature_count: int,
    candidate_complexity: float,
) -> tuple[float, ...]:
    if objective == "mean_ood_auroc_oracle":
        return (
            oracle_mean_ood_auroc if np.isfinite(oracle_mean_ood_auroc) else float("-inf"),
            oracle_mean_ood_average_precision if np.isfinite(oracle_mean_ood_average_precision) else float("-inf"),
            source_val_auroc if np.isfinite(source_val_auroc) else float("-inf"),
            source_val_average_precision if np.isfinite(source_val_average_precision) else float("-inf"),
            -int(feature_count),
            -float(candidate_complexity),
        )
    if objective == "source_val_auroc":
        return (
            source_val_auroc if np.isfinite(source_val_auroc) else float("-inf"),
            source_val_average_precision if np.isfinite(source_val_average_precision) else float("-inf"),
            source_val_balanced_accuracy if np.isfinite(source_val_balanced_accuracy) else float("-inf"),
            -int(feature_count),
            -float(candidate_complexity),
        )
    raise ValueError(
        "Unsupported MODEL_SELECTION_OBJECTIVE="
        f"{objective!r}. Expected one of ['mean_ood_auroc_oracle', 'source_val_auroc']."
    )


@dataclass
class ActivationMatrixBundle:
    matrices_by_env: dict[str, dict[str, np.ndarray]]
    feature_names: list[str]
    feature_lookup_df: pd.DataFrame
    effective_pca_dim: int | None


@dataclass
class BaselineMatrixBundle:
    matrices_by_env: dict[str, dict[str, Any]]
    feature_names: list[str]
    feature_lookup_df: pd.DataFrame


def build_attention_matrix_cache(selected_features: list[str]) -> dict[str, dict[str, np.ndarray]]:
    env_cache: dict[str, dict[str, np.ndarray]] = {}
    for env_name in ENV_ORDER:
        env_df = attention_envs[env_name]
        split_bundle = split_cache_by_env[env_name]
        pooled = (
            env_df.loc[:, selected_features]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .to_numpy(dtype=np.float32, copy=False)
        )
        env_cache[env_name] = {
            "train": np.asarray(pooled[split_bundle["train_mask"]], dtype=np.float32),
            "val": np.asarray(pooled[split_bundle["val_mask"]], dtype=np.float32),
        }
        del pooled
        gc.collect()
    return env_cache


def structural_feature_family(feature_name: str) -> str:
    return str(classify_feature_family(str(feature_name)))


def tfidf_feature_family(_feature_name: str) -> str:
    return "tfidf"


def align_matrix_to_feature_order(
    matrix: sp.spmatrix,
    env_feature_names: list[str],
    target_feature_names: list[str],
) -> sp.csr_matrix:
    matrix = matrix.tocsr()
    env_feature_to_idx = {str(feature_name): idx for idx, feature_name in enumerate(env_feature_names)}
    source_columns: list[int] = []
    target_columns: list[int] = []
    for target_idx, feature_name in enumerate(target_feature_names):
        source_idx = env_feature_to_idx.get(str(feature_name))
        if source_idx is None:
            continue
        source_columns.append(int(source_idx))
        target_columns.append(int(target_idx))

    if not target_feature_names:
        return sp.csr_matrix((matrix.shape[0], 0), dtype=np.float32)
    if not source_columns:
        return sp.csr_matrix((matrix.shape[0], len(target_feature_names)), dtype=np.float32)

    subset = matrix[:, np.asarray(source_columns, dtype=np.int64)]
    if len(source_columns) == len(target_feature_names) and all(
        source_col == target_col for source_col, target_col in zip(range(len(target_feature_names)), target_columns, strict=True)
    ):
        return subset.tocsr().astype(np.float32)

    subset_coo = subset.tocoo()
    remapped_columns = np.asarray(target_columns, dtype=np.int64)[subset_coo.col]
    return sp.csr_matrix(
        (subset_coo.data.astype(np.float32, copy=False), (subset_coo.row, remapped_columns)),
        shape=(matrix.shape[0], len(target_feature_names)),
        dtype=np.float32,
    )


def gather_dense_rows_with_missing(matrix: np.ndarray, row_idx: np.ndarray) -> np.ndarray:
    row_idx = np.asarray(row_idx, dtype=np.int64)
    if row_idx.size == 0:
        return np.zeros((0, int(matrix.shape[1])), dtype=np.float32)
    out = np.zeros((row_idx.shape[0], int(matrix.shape[1])), dtype=np.float32)
    valid_mask = row_idx >= 0
    if valid_mask.any():
        out[valid_mask] = np.asarray(matrix[row_idx[valid_mask]], dtype=np.float32)
    return out


def gather_sparse_rows_with_missing(matrix: sp.spmatrix, row_idx: np.ndarray) -> sp.csr_matrix:
    row_idx = np.asarray(row_idx, dtype=np.int64)
    matrix = matrix.tocsr().astype(np.float32)
    if row_idx.size == 0:
        return sp.csr_matrix((0, matrix.shape[1]), dtype=np.float32)

    valid_positions = np.flatnonzero(row_idx >= 0)
    if valid_positions.size == row_idx.size:
        return matrix[row_idx].tocsr()
    if valid_positions.size == 0:
        return sp.csr_matrix((row_idx.size, matrix.shape[1]), dtype=np.float32)

    valid_rows = matrix[row_idx[valid_positions]].tocoo()
    remapped_row = valid_positions[valid_rows.row]
    return sp.csr_matrix(
        (valid_rows.data.astype(np.float32, copy=False), (remapped_row, valid_rows.col)),
        shape=(row_idx.size, matrix.shape[1]),
        dtype=np.float32,
    )


def choose_cross_env_tfidf_feature_names(feature_names_by_env: OrderedDict[str, list[str]]) -> tuple[list[str], str]:
    if not feature_names_by_env:
        return [], "empty"

    ordered_env_names = list(feature_names_by_env.keys())
    first_env = ordered_env_names[0]
    common_names = set(feature_names_by_env[first_env])
    for env_name in ordered_env_names[1:]:
        common_names &= set(feature_names_by_env[env_name])

    if common_names:
        ordered_common = [
            str(feature_name)
            for feature_name in feature_names_by_env[first_env]
            if str(feature_name) in common_names
        ]
        return ordered_common, "intersection"

    seen: set[str] = set()
    ordered_union: list[str] = []
    for env_name in ordered_env_names:
        for feature_name in feature_names_by_env[env_name]:
            feature_name = str(feature_name)
            if feature_name in seen:
                continue
            seen.add(feature_name)
            ordered_union.append(feature_name)
    return ordered_union, "union_fallback"


def build_structural_matrix_bundle() -> BaselineMatrixBundle:
    matrices_by_env: dict[str, dict[str, np.ndarray]] = {}
    feature_lookup_df = make_generic_feature_lookup(
        space_name="baseline_structural",
        feature_names=list(STRUCTURAL_BASELINE_FEATURE_COLUMNS),
        feature_root="baseline_structural",
        metric_name="sentence_structure",
        family_resolver=structural_feature_family,
    )

    for env_name in ENV_ORDER:
        env_paths = DATASET_FILE_MAP[env_name]
        split_bundle = split_cache_by_env[env_name]
        raw_df = pd.read_parquet(
            env_paths["structural_baseline_path"],
            columns=list(STRUCTURAL_BASELINE_FEATURE_COLUMNS),
        ).copy()
        raw_df = raw_df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        raw_matrix = raw_df.to_numpy(dtype=np.float32, copy=False)
        matrices_by_env[env_name] = {
            "train": gather_dense_rows_with_missing(raw_matrix, split_bundle["train_structural_row_idx"]),
            "val": gather_dense_rows_with_missing(raw_matrix, split_bundle["val_structural_row_idx"]),
        }

    return BaselineMatrixBundle(
        matrices_by_env=matrices_by_env,
        feature_names=list(STRUCTURAL_BASELINE_FEATURE_COLUMNS),
        feature_lookup_df=feature_lookup_df,
    )


def build_tfidf_matrix_bundle(*, text_field: str, space_name: str) -> BaselineMatrixBundle:
    artifact_paths_by_env: OrderedDict[str, dict[str, Path]] = OrderedDict()
    feature_names_by_env: OrderedDict[str, list[str]] = OrderedDict()

    for env_name in ENV_ORDER:
        env_paths = DATASET_FILE_MAP[env_name]
        artifact_paths = locate_tfidf_artifact(env_paths["tfidf_cache_dir"], text_field)
        if artifact_paths is None:
            raise FileNotFoundError(
                f"Could not find TF-IDF cache artifact for {env_name} / {text_field} under {env_paths['tfidf_cache_dir']}"
            )
        meta = parse_tfidf_artifact_meta(artifact_paths["meta_path"])
        structural_metadata_df = structural_metadata_by_env.get(env_name)
        if structural_metadata_df is None:
            raise FileNotFoundError(f"{env_name} is missing companion structural metadata needed to align TF-IDF rows.")
        structural_key_df = pd.read_parquet(
            env_paths["structural_baseline_path"],
            columns=["example_id", "sentence_idx"],
        ).copy()
        expected_fingerprint = baseline_extractor.dataset_row_fingerprint(structural_key_df)
        if str(meta.get("fingerprint", "")) != str(expected_fingerprint):
            raise ValueError(
                f"{env_name} TF-IDF fingerprint mismatch for {text_field}. "
                f"expected={expected_fingerprint}, found={meta.get('fingerprint', '')}"
            )
        if int(meta.get("num_rows", len(structural_metadata_df))) != int(len(structural_metadata_df)):
            raise ValueError(
                f"{env_name} TF-IDF row-count mismatch for {text_field}. "
                f"expected={len(structural_metadata_df)}, found={meta.get('num_rows')}"
            )
        feature_names = np.load(artifact_paths["feature_names_path"], allow_pickle=False).astype(str).tolist()
        artifact_paths_by_env[env_name] = artifact_paths
        feature_names_by_env[env_name] = feature_names

    shared_feature_names, shared_mode = choose_cross_env_tfidf_feature_names(feature_names_by_env)
    if not shared_feature_names:
        raise ValueError(f"No shared TF-IDF features available for text field {text_field!r} across environments.")

    feature_lookup_df = make_generic_feature_lookup(
        space_name=space_name,
        feature_names=shared_feature_names,
        feature_root=space_name,
        metric_name=f"tfidf__{text_field}__{shared_mode}",
        family_resolver=tfidf_feature_family,
    )
    matrices_by_env: dict[str, dict[str, sp.csr_matrix]] = {}
    for env_name in ENV_ORDER:
        split_bundle = split_cache_by_env[env_name]
        artifact_paths = artifact_paths_by_env[env_name]
        env_feature_names = feature_names_by_env[env_name]
        matrix = sp.load_npz(artifact_paths["matrix_path"]).tocsr().astype(np.float32)
        aligned_matrix = align_matrix_to_feature_order(matrix, env_feature_names, shared_feature_names)
        matrices_by_env[env_name] = {
            "train": gather_sparse_rows_with_missing(aligned_matrix, split_bundle["train_structural_row_idx"]),
            "val": gather_sparse_rows_with_missing(aligned_matrix, split_bundle["val_structural_row_idx"]),
        }

    return BaselineMatrixBundle(
        matrices_by_env=matrices_by_env,
        feature_names=shared_feature_names,
        feature_lookup_df=feature_lookup_df,
    )


def build_activation_matrix_bundle(
    *,
    source_envs: tuple[str, ...],
    train_env_label: str,
    feature_space_name: str,
    feature_space: FeatureSpaceSpec,
) -> ActivationMatrixBundle:
    if feature_space.activation_variant is None:
        raise ValueError(f"{feature_space_name} does not define an activation variant.")

    raw_train_parts: list[np.ndarray] = []
    for env_name in source_envs:
        env_train = activation_stores[env_name].load_matrix(
            split_cache_by_env[env_name]["train_row_idx"],
            variant=str(feature_space.activation_variant),
        )
        if env_train.size == 0:
            raise ValueError(f"{env_name} has no activation rows for {feature_space_name}.")
        raw_train_parts.append(np.asarray(env_train, dtype=np.float32))
    raw_train = np.concatenate(raw_train_parts, axis=0)

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_imputed = imputer.fit_transform(raw_train)
    train_scaled = scaler.fit_transform(train_imputed)

    pca: PCA | None = None
    effective_pca_dim: int | None = None
    if feature_space.activation_use_pca:
        effective_pca_dim = int(min(int(ACTIVATION_PCA_DIM), train_scaled.shape[0] - 1, train_scaled.shape[1]))
        if effective_pca_dim < 1:
            raise ValueError(
                f"Effective PCA dim < 1 for {feature_space_name} / {train_env_label}. "
                f"train_rows={train_scaled.shape[0]}, hidden_dim={train_scaled.shape[1]}"
            )
        pca = PCA(
            n_components=int(effective_pca_dim),
            random_state=int(RANDOM_SEED),
            svd_solver="randomized",
        )
        pca.fit(train_scaled)

    matrices_by_env: dict[str, dict[str, np.ndarray]] = {}
    for env_name in ENV_ORDER:
        matrices_by_env[env_name] = {}
        for split_name in ("train", "val"):
            row_idx = split_cache_by_env[env_name][f"{split_name}_row_idx"]
            raw_matrix = activation_stores[env_name].load_matrix(
                row_idx,
                variant=str(feature_space.activation_variant),
            )
            transformed = scaler.transform(imputer.transform(raw_matrix))
            if pca is not None:
                transformed = pca.transform(transformed)
            matrices_by_env[env_name][split_name] = np.asarray(transformed, dtype=np.float32)

    feature_lookup_df = make_activation_lookup(
        space_name=feature_space_name,
        variant=str(feature_space.activation_variant),
        use_pca=bool(feature_space.activation_use_pca),
        hidden_dim=int(train_scaled.shape[1]),
        pca_dim=int(effective_pca_dim if effective_pca_dim is not None else ACTIVATION_PCA_DIM),
    )
    feature_names = feature_lookup_df["feature"].astype(str).tolist()
    return ActivationMatrixBundle(
        matrices_by_env=matrices_by_env,
        feature_names=feature_names,
        feature_lookup_df=feature_lookup_df,
        effective_pca_dim=effective_pca_dim,
    )


activation_lookup_by_space: dict[str, pd.DataFrame] = {}
for feature_space_name, feature_space in FEATURE_SPACES.items():
    if feature_space.activation_variant is None:
        continue
    activation_lookup_by_space[feature_space_name] = make_activation_lookup(
        space_name=feature_space_name,
        variant=str(feature_space.activation_variant),
        use_pca=bool(feature_space.activation_use_pca),
        hidden_dim=int(ACTIVATION_HIDDEN_DIM),
        pca_dim=int(ACTIVATION_PCA_DIM),
    )

baseline_bundle_by_space: dict[str, BaselineMatrixBundle] = {}
baseline_lookup_by_space: dict[str, pd.DataFrame] = {}
for feature_space_name, feature_space in FEATURE_SPACES.items():
    if feature_space.baseline_variant == "tfidf":
        if feature_space.baseline_text_field is None:
            raise ValueError(f"{feature_space_name} baseline_variant='tfidf' is missing baseline_text_field.")
        baseline_bundle = build_tfidf_matrix_bundle(
            text_field=str(feature_space.baseline_text_field),
            space_name=feature_space_name,
        )
    else:
        continue
    baseline_bundle_by_space[feature_space_name] = baseline_bundle
    baseline_lookup_by_space[feature_space_name] = baseline_bundle.feature_lookup_df.copy()

ranking_frames: list[pd.DataFrame] = []
ranking_pools: dict[tuple[str, str], list[str]] = {}
ranking_tables_by_key: dict[tuple[str, str], pd.DataFrame] = {}
attention_pools_by_target_subset: dict[tuple[str, str], list[str]] = {}
used_attention_subset_keys = [
    subset_key
    for subset_key in ATTENTION_SUBSETS
    if any(
        feature_space.uses_attention and feature_space.attention_subset_key == subset_key
        for feature_space in FEATURE_SPACES.values()
    )
]

for target_name in maybe_tqdm(list(TARGET_SPECS.keys()), desc="Rank targets", total=len(TARGET_SPECS), disable=DISABLE_TQDM):
    attention_rankings_by_subset: dict[str, pd.DataFrame] = {}
    for subset_key in used_attention_subset_keys:
        subset_lookup_df = attention_lookup_by_subset[subset_key]
        attention_ranking_df = build_consistency_ranking(
            attention_envs,
            feature_names=subset_lookup_df["feature"].astype(str).tolist(),
            feature_lookup_df=subset_lookup_df,
            target_name=target_name,
        )
        attention_rankings_by_subset[subset_key] = attention_ranking_df
        attention_pools_by_target_subset[(target_name, subset_key)] = select_ranked_attention_pool(
            attention_ranking_df,
        )

    for feature_space_name, feature_space in FEATURE_SPACES.items():
        feature_space_attention_subset_key = str(feature_space.attention_subset_key or "")
        feature_space_attention_subset_title = (
            ATTENTION_SUBSETS[str(feature_space.attention_subset_key)].title
            if feature_space.attention_subset_key is not None
            else ""
        )
        if feature_space.uses_attention and feature_space.activation_variant is None:
            if feature_space.attention_subset_key is None:
                raise ValueError(f"{feature_space_name} uses attention but has no attention subset key.")
            ranking_df = attention_rankings_by_subset[str(feature_space.attention_subset_key)].copy()
            feature_pool = list(attention_pools_by_target_subset[(target_name, str(feature_space.attention_subset_key))])
        elif (not feature_space.uses_attention) and feature_space.activation_variant is not None:
            activation_lookup_df = activation_lookup_by_space[feature_space_name].copy()
            ranking_df = make_activation_ranking_df(
                target_name=target_name,
                feature_lookup_df=activation_lookup_df,
            )
            feature_pool = activation_lookup_df["feature"].astype(str).tolist()
        elif feature_space.baseline_variant is not None:
            baseline_lookup_df = baseline_lookup_by_space[feature_space_name].copy()
            ranking_df = make_placeholder_ranking_df(
                target_name=target_name,
                feature_lookup_df=baseline_lookup_df,
            )
            feature_pool = baseline_lookup_df["feature"].astype(str).tolist()
        else:
            if feature_space.attention_subset_key is None:
                raise ValueError(f"{feature_space_name} uses attention but has no attention subset key.")
            activation_lookup_df = activation_lookup_by_space[feature_space_name].copy()
            attention_part = attention_rankings_by_subset[str(feature_space.attention_subset_key)].copy()
            activation_part = make_activation_ranking_df(
                target_name=target_name,
                feature_lookup_df=activation_lookup_df,
                start_rank=int(attention_part["global_rank"].max()) + 1,
            )
            ranking_df = pd.concat([attention_part, activation_part], ignore_index=True)
            feature_pool = list(attention_pools_by_target_subset[(target_name, str(feature_space.attention_subset_key))]) + activation_lookup_df["feature"].astype(str).tolist()

        ranking_df["feature_space_attention_subset_key"] = feature_space_attention_subset_key
        ranking_df["feature_space_attention_subset_title"] = feature_space_attention_subset_title
        ranking_tables_by_key[(target_name, feature_space_name)] = ranking_df.copy()
        ranking_pools[(target_name, feature_space_name)] = feature_pool
        ranking_frame = ranking_df.copy()
        ranking_frame["feature_space"] = feature_space_name
        ranking_frame["feature_space_title"] = feature_space.title
        ranking_frame["feature_family_group"] = feature_space.family_title
        ranking_frames.append(ranking_frame)

ranking_df_all = pd.concat(ranking_frames, ignore_index=True)
ranking_df_all.to_csv(OUTPUT_ROOT / "global_consistency_rankings.csv", index=False)
display(ranking_df_all.head(20))

for (target_name, feature_space_name), feature_pool in ranking_pools.items():
    rank_dir = OUTPUT_ROOT / "rankings" / target_name
    rank_dir.mkdir(parents=True, exist_ok=True)
    ranking_meta_df = ranking_tables_by_key[(target_name, feature_space_name)].drop_duplicates(subset=["feature"], keep="first")
    pool_df = pd.DataFrame(
        {
            "feature": feature_pool,
            "pool_rank": np.arange(1, len(feature_pool) + 1, dtype=int),
        }
    )
    pool_df = pool_df.merge(
        ranking_meta_df,
        on="feature",
        how="left",
        validate="one_to_one",
    )
    pool_df.to_csv(rank_dir / f"{feature_space_name}__feature_pool.csv", index=False)

attention_matrix_cache_by_target_subset: dict[tuple[str, str], dict[str, dict[str, np.ndarray]]] = {}
for target_name in maybe_tqdm(
    list(TARGET_SPECS.keys()),
    desc="Cache attention",
    total=len(TARGET_SPECS),
    disable=DISABLE_TQDM,
):
    for subset_key in used_attention_subset_keys:
        attention_matrix_cache_by_target_subset[(target_name, subset_key)] = build_attention_matrix_cache(
            attention_pools_by_target_subset[(target_name, subset_key)]
        )


# %%
def feature_size_to_label(feature_size: int) -> str:
    return f"k{int(feature_size):03d}"


def selected_feature_size_label(feature_space: FeatureSpaceSpec, feature_size: int | None) -> str:
    if feature_space.family_title == "baseline":
        if feature_space.baseline_variant == "tfidf":
            return f"tfidf_{slugify(str(feature_space.baseline_text_field or 'text'))}"
        if feature_space.activation_variant == "final_sentence" and not feature_space.activation_use_pca:
            return "raw_final"
        return slugify(feature_space.name)
    if feature_space.family_title == "attention_only":
        return "all_attention"
    if feature_size is None:
        return "all_features"
    return feature_size_to_label(int(feature_size))


def feature_size_options_for_space(feature_space: FeatureSpaceSpec) -> tuple[int | None, ...]:
    if feature_space.family_title in {"baseline", "attention_only"}:
        return (None,)
    return tuple(int(value) for value in FEATURE_SIZE_GRID)


@dataclass(frozen=True)
class ExperimentRunSpec:
    scenario_name: str
    scenario_title: str
    train_env_label: str
    source_envs: tuple[str, ...]
    source_envs_label: str
    ood_envs: tuple[str, ...]
    heldout_env: str | None


def build_experiment_run_specs() -> tuple[list[ExperimentRunSpec], dict[str, list[str]]]:
    run_specs: list[ExperimentRunSpec] = []
    train_axis_labels_by_scenario: dict[str, list[str]] = {}
    for scenario_name, scenario_title in SELECTED_SCENARIO_TITLES.items():
        scenario_labels: list[str] = []
        if scenario_name == "single_source_ood":
            for env_name in ENV_ORDER:
                scenario_labels.append(env_name)
                run_specs.append(
                    ExperimentRunSpec(
                        scenario_name=scenario_name,
                        scenario_title=scenario_title,
                        train_env_label=env_name,
                        source_envs=(env_name,),
                        source_envs_label=env_name,
                        ood_envs=tuple(other_env for other_env in ENV_ORDER if other_env != env_name),
                        heldout_env=None,
                    )
                )
            train_axis_labels_by_scenario[scenario_name] = scenario_labels
            continue
        if scenario_name == "holdout_env_ood":
            for heldout_env in ENV_ORDER:
                train_env_label = f"All except {heldout_env}"
                source_envs = tuple(env_name for env_name in ENV_ORDER if env_name != heldout_env)
                scenario_labels.append(train_env_label)
                run_specs.append(
                    ExperimentRunSpec(
                        scenario_name=scenario_name,
                        scenario_title=scenario_title,
                        train_env_label=train_env_label,
                        source_envs=source_envs,
                        source_envs_label=", ".join(source_envs),
                        ood_envs=(heldout_env,),
                        heldout_env=heldout_env,
                    )
                )
            train_axis_labels_by_scenario[scenario_name] = scenario_labels
            continue
        raise ValueError(f"Unsupported scenario_name={scenario_name!r}")
    return run_specs, train_axis_labels_by_scenario


EXPERIMENT_RUN_SPECS, TRAIN_AXIS_LABELS_BY_SCENARIO = build_experiment_run_specs()


def assemble_split_matrix(
    env_cache: dict[str, Any],
    *,
    split_name: str,
    feature_space: FeatureSpaceSpec,
    attention_dim: int,
    activation_dim: int,
) -> Any:
    parts: list[Any] = []
    if feature_space.uses_attention:
        parts.append(np.asarray(env_cache[f"{split_name}_attention_pool"][:, :attention_dim], dtype=np.float32))
    if feature_space.activation_variant is not None:
        activation_pool = np.asarray(env_cache[f"{split_name}_activation_pool"], dtype=np.float32)
        if feature_space.activation_use_pca:
            parts.append(np.asarray(activation_pool[:, :activation_dim], dtype=np.float32))
        else:
            parts.append(activation_pool)
    if feature_space.baseline_variant is not None:
        baseline_pool = env_cache[f"{split_name}_baseline_pool"]
        if sp.issparse(baseline_pool):
            parts.append(baseline_pool.tocsr())
        else:
            parts.append(np.asarray(baseline_pool, dtype=np.float32))
    if not parts:
        raise ValueError("assemble_split_matrix received no feature parts.")
    if len(parts) == 1:
        return parts[0]
    if any(sp.issparse(part) for part in parts):
        sparse_parts = [
            part.tocsr() if sp.issparse(part) else sp.csr_matrix(np.asarray(part, dtype=np.float32))
            for part in parts
        ]
        return sp.hstack(sparse_parts, format="csr")
    return np.concatenate(parts, axis=1)


all_transfer_metric_rows: list[dict[str, Any]] = []
all_model_selection_rows: list[dict[str, Any]] = []
all_coefficient_frames: list[pd.DataFrame] = []
activation_bundle_cache: dict[tuple[str, str, str], ActivationMatrixBundle] = {}
completed_model_selection_count = 0


def persist_core_artifacts(
    *,
    transfer_rows: list[dict[str, Any]],
    model_selection_rows: list[dict[str, Any]],
    coefficient_frames: list[pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    transfer_df = pd.DataFrame(transfer_rows)
    model_selection_df = pd.DataFrame(model_selection_rows)
    coefficients_df = pd.concat(coefficient_frames, ignore_index=True) if coefficient_frames else pd.DataFrame()

    if not transfer_df.empty:
        transfer_df.to_csv(OUTPUT_ROOT / "all_transfer_metrics.csv", index=False)
    if not model_selection_df.empty:
        model_selection_df.to_csv(OUTPUT_ROOT / "all_model_selection.csv", index=False)
    if not coefficients_df.empty:
        coefficients_df.to_csv(OUTPUT_ROOT / "all_coefficients.csv", index=False)

    return transfer_df, model_selection_df, coefficients_df


def concatenate_source_split_matrices(
    env_pool_cache: dict[str, dict[str, Any]],
    *,
    source_envs: tuple[str, ...],
    split_name: str,
    feature_space: FeatureSpaceSpec,
    attention_dim: int,
    activation_dim: int,
) -> tuple[Any, np.ndarray]:
    x_parts: list[Any] = []
    y_parts: list[np.ndarray] = []
    for env_name in source_envs:
        env_cache = env_pool_cache[env_name]
        x_parts.append(
            assemble_split_matrix(
                env_cache,
                split_name=split_name,
                feature_space=feature_space,
                attention_dim=attention_dim,
                activation_dim=activation_dim,
            )
        )
        y_parts.append(np.asarray(env_cache[f"y_{split_name}"], dtype=np.int8))
    x_out = (
        sp.vstack([part.tocsr() if sp.issparse(part) else sp.csr_matrix(np.asarray(part, dtype=np.float32)) for part in x_parts], format="csr")
        if any(sp.issparse(part) for part in x_parts)
        else np.concatenate(x_parts, axis=0)
    )
    return x_out, np.concatenate(y_parts, axis=0)


try:
    for target_name in maybe_tqdm(list(TARGET_SPECS.keys()), desc="Model targets", total=len(TARGET_SPECS), disable=DISABLE_TQDM):
        target_title = str(TARGET_SPECS[target_name]["title"])
        for feature_space_name, feature_space in maybe_tqdm(
            FEATURE_SPACES.items(),
            desc=f"Model spaces:{target_name}",
            total=len(FEATURE_SPACES),
            disable=DISABLE_TQDM,
        ):
            feature_space_attention_subset_key = str(feature_space.attention_subset_key or "")
            feature_space_attention_subset_title = (
                ATTENTION_SUBSETS[str(feature_space.attention_subset_key)].title
                if feature_space.attention_subset_key is not None
                else ""
            )
            attention_feature_pool: list[str] = []
            if feature_space.uses_attention:
                if feature_space.attention_subset_key is None:
                    raise ValueError(f"{feature_space_name} uses attention but has no attention subset key.")
                attention_feature_pool = list(
                    attention_pools_by_target_subset[(target_name, str(feature_space.attention_subset_key))]
                )
            ranking_df = ranking_tables_by_key[(target_name, feature_space_name)].copy()
            ranking_meta_df = ranking_df.loc[:, [column for column in ranking_df.columns if column != "feature_space"]].drop_duplicates(
                subset=["feature"],
                keep="first",
            )
            size_options = feature_size_options_for_space(feature_space)

            for run_spec in maybe_tqdm(
                EXPERIMENT_RUN_SPECS,
                desc=f"Runs:{target_name}:{feature_space_name}",
                total=len(EXPERIMENT_RUN_SPECS),
                disable=DISABLE_TQDM,
            ):
                activation_bundle: ActivationMatrixBundle | None = None
                baseline_bundle: BaselineMatrixBundle | None = None
                if feature_space.activation_variant is not None:
                    bundle_key = (run_spec.scenario_name, run_spec.train_env_label, feature_space_name)
                    activation_bundle = activation_bundle_cache.get(bundle_key)
                    if activation_bundle is None:
                        activation_bundle = build_activation_matrix_bundle(
                            source_envs=run_spec.source_envs,
                            train_env_label=run_spec.train_env_label,
                            feature_space_name=feature_space_name,
                            feature_space=feature_space,
                        )
                        activation_bundle_cache[bundle_key] = activation_bundle
                if feature_space.baseline_variant is not None:
                    baseline_bundle = baseline_bundle_by_space.get(feature_space_name)
                    if baseline_bundle is None:
                        raise ValueError(f"Missing baseline bundle for {feature_space_name}.")

                env_pool_cache: dict[str, dict[str, Any]] = {}
                for env_name in ENV_ORDER:
                    y_train = np.asarray(split_cache_by_env[env_name][f"y_train__{target_name}"], dtype=np.int8)
                    y_val = np.asarray(split_cache_by_env[env_name][f"y_val__{target_name}"], dtype=np.int8)
                    if np.unique(y_train).size < 2:
                        raise ValueError(f"{env_name} train split does not contain both classes for {target_name}.")
                    if np.unique(y_val).size < 2:
                        raise ValueError(f"{env_name} val split does not contain both classes for {target_name}.")

                    env_entry: dict[str, np.ndarray] = {
                        "y_train": y_train,
                        "y_val": y_val,
                    }
                    if feature_space.uses_attention:
                        env_entry["train_attention_pool"] = attention_matrix_cache_by_target_subset[
                            (target_name, str(feature_space.attention_subset_key))
                        ][env_name]["train"]
                        env_entry["val_attention_pool"] = attention_matrix_cache_by_target_subset[
                            (target_name, str(feature_space.attention_subset_key))
                        ][env_name]["val"]
                    if activation_bundle is not None:
                        env_entry["train_activation_pool"] = activation_bundle.matrices_by_env[env_name]["train"]
                        env_entry["val_activation_pool"] = activation_bundle.matrices_by_env[env_name]["val"]
                    if baseline_bundle is not None:
                        env_entry["train_baseline_pool"] = baseline_bundle.matrices_by_env[env_name]["train"]
                        env_entry["val_baseline_pool"] = baseline_bundle.matrices_by_env[env_name]["val"]
                    env_pool_cache[env_name] = env_entry

                for feature_size in size_options:
                    size_label = selected_feature_size_label(feature_space, feature_size)
                    attention_dim = 0
                    activation_dim = 0
                    current_feature_names: list[str] = []

                    if feature_space.uses_attention:
                        attention_dim = int(len(attention_feature_pool))
                        current_feature_names.extend(attention_feature_pool)

                    if activation_bundle is not None:
                        if feature_space.activation_use_pca:
                            requested_activation = len(activation_bundle.feature_names) if feature_size is None else int(feature_size)
                            activation_dim = int(min(len(activation_bundle.feature_names), requested_activation))
                            current_feature_names.extend(activation_bundle.feature_names[:activation_dim])
                        else:
                            activation_dim = int(len(activation_bundle.feature_names))
                            current_feature_names.extend(activation_bundle.feature_names)
                    if baseline_bundle is not None:
                        current_feature_names.extend(baseline_bundle.feature_names)

                    if not current_feature_names:
                        raise ValueError(
                            f"No features constructed for {target_name} / {feature_space_name} / {run_spec.train_env_label} / {size_label}."
                        )

                    x_train, y_train = concatenate_source_split_matrices(
                        env_pool_cache,
                        source_envs=run_spec.source_envs,
                        split_name="train",
                        feature_space=feature_space,
                        attention_dim=attention_dim,
                        activation_dim=activation_dim,
                    )
                    x_val, y_val = concatenate_source_split_matrices(
                        env_pool_cache,
                        source_envs=run_spec.source_envs,
                        split_name="val",
                        feature_space=feature_space,
                        attention_dim=attention_dim,
                        activation_dim=activation_dim,
                    )
                    fitted_models, candidate_df = fit_candidate_classifiers(
                        x_train,
                        y_train,
                        x_val,
                        y_val,
                    )

                    best_model: FittedBinaryModel | None = None
                    best_key: tuple[float, ...] | None = None
                    best_candidate_oracle_metrics: dict[str, float] | None = None
                    oracle_rows: list[dict[str, Any]] = []

                    for fitted_model in fitted_models:
                        oracle_aurocs: list[float] = []
                        oracle_average_precisions: list[float] = []
                        for test_env in run_spec.ood_envs:
                            eval_cache = env_pool_cache[test_env]
                            x_eval = assemble_split_matrix(
                                eval_cache,
                                split_name="val",
                                feature_space=feature_space,
                                attention_dim=attention_dim,
                                activation_dim=activation_dim,
                            )
                            eval_scores = fitted_model.estimator.predict_proba(x_eval)[:, 1].astype(np.float32)
                            oracle_aurocs.append(safe_auroc(eval_cache["y_val"], eval_scores))
                            oracle_average_precisions.append(safe_average_precision(eval_cache["y_val"], eval_scores))
                        oracle_metrics = {
                            "oracle_mean_ood_auroc": finite_mean(oracle_aurocs),
                            "oracle_min_ood_auroc": finite_min(oracle_aurocs),
                            "oracle_std_ood_auroc": finite_std(oracle_aurocs),
                            "oracle_mean_ood_average_precision": finite_mean(oracle_average_precisions),
                        }
                        oracle_rows.append(
                            {
                                "candidate_key": str(fitted_model.candidate_key),
                                **oracle_metrics,
                            }
                        )
                        candidate_key = build_model_selection_key(
                            objective=MODEL_SELECTION_OBJECTIVE,
                            oracle_mean_ood_auroc=oracle_metrics["oracle_mean_ood_auroc"],
                            oracle_mean_ood_average_precision=oracle_metrics["oracle_mean_ood_average_precision"],
                            source_val_auroc=float(fitted_model.validation_metrics["auroc"]),
                            source_val_average_precision=float(fitted_model.validation_metrics["average_precision"]),
                            source_val_balanced_accuracy=float(fitted_model.validation_metrics["balanced_accuracy"]),
                            feature_count=len(current_feature_names),
                            candidate_complexity=float(fitted_model.candidate_complexity),
                        )
                        if best_key is None or candidate_key > best_key:
                            best_key = candidate_key
                            best_model = fitted_model
                            best_candidate_oracle_metrics = oracle_metrics

                    if best_model is None or best_candidate_oracle_metrics is None:
                        raise RuntimeError(
                            f"Failed to select model for {target_name} / {feature_space_name} / {run_spec.train_env_label} / {size_label}."
                        )

                    selection_dir = OUTPUT_ROOT / "model_selection" / run_spec.scenario_name / target_name / feature_space_name / size_label
                    selection_dir.mkdir(parents=True, exist_ok=True)
                    candidate_path = selection_dir / f"{slugify(run_spec.train_env_label)}__candidates.csv"
                    candidate_df = candidate_df.merge(
                        pd.DataFrame(oracle_rows),
                        on="candidate_key",
                        how="left",
                        validate="one_to_one",
                    )
                    candidate_df["scenario_name"] = run_spec.scenario_name
                    candidate_df["scenario_title"] = run_spec.scenario_title
                    candidate_df["target_name"] = target_name
                    candidate_df["feature_space"] = feature_space_name
                    candidate_df["feature_space_title"] = feature_space.title
                    candidate_df["feature_family_group"] = feature_space.family_title
                    candidate_df["feature_space_attention_subset_key"] = feature_space_attention_subset_key
                    candidate_df["feature_space_attention_subset_title"] = feature_space_attention_subset_title
                    candidate_df["train_env"] = run_spec.train_env_label
                    candidate_df["source_envs"] = run_spec.source_envs_label
                    candidate_df["source_env_count"] = int(len(run_spec.source_envs))
                    candidate_df["heldout_env"] = run_spec.heldout_env if run_spec.heldout_env is not None else pd.NA
                    candidate_df["ood_env_count"] = int(len(run_spec.ood_envs))
                    candidate_df["feature_size"] = pd.NA if feature_size is None else int(feature_size)
                    candidate_df["feature_size_label"] = size_label
                    candidate_df["attention_feature_count"] = int(attention_dim)
                    candidate_df["activation_feature_count"] = int(activation_dim)
                    candidate_df["selected_feature_count"] = len(current_feature_names)
                    candidate_df["effective_activation_pca_dim"] = (
                        pd.NA if activation_bundle is None or activation_bundle.effective_pca_dim is None else int(activation_bundle.effective_pca_dim)
                    )
                    candidate_df.to_csv(candidate_path, index=False)

                    selected_path = selection_dir / f"{slugify(run_spec.train_env_label)}__selected_features.csv"
                    coefficients_path = selection_dir / f"{slugify(run_spec.train_env_label)}__coefficients.csv"
                    selection_summary_path = selection_dir / f"{slugify(run_spec.train_env_label)}__selection_summary.csv"
                    transfer_metrics_path = selection_dir / f"{slugify(run_spec.train_env_label)}__transfer_metrics.csv"

                    selected_df = pd.DataFrame(
                        {
                            "feature": current_feature_names,
                            "selected_rank": np.arange(1, len(current_feature_names) + 1, dtype=int),
                        }
                    )
                    if not selected_df.empty:
                        selected_df = selected_df.merge(
                            ranking_meta_df,
                            on="feature",
                            how="left",
                            validate="one_to_one",
                        )
                    selected_df["scenario_name"] = run_spec.scenario_name
                    selected_df["scenario_title"] = run_spec.scenario_title
                    selected_df["target_name"] = target_name
                    selected_df["feature_space"] = feature_space_name
                    selected_df["feature_space_title"] = feature_space.title
                    selected_df["feature_family_group"] = feature_space.family_title
                    selected_df["feature_space_attention_subset_key"] = feature_space_attention_subset_key
                    selected_df["feature_space_attention_subset_title"] = feature_space_attention_subset_title
                    selected_df["train_env"] = run_spec.train_env_label
                    selected_df["source_envs"] = run_spec.source_envs_label
                    selected_df["source_env_count"] = int(len(run_spec.source_envs))
                    selected_df["heldout_env"] = run_spec.heldout_env if run_spec.heldout_env is not None else pd.NA
                    selected_df["feature_size"] = pd.NA if feature_size is None else int(feature_size)
                    selected_df["feature_size_label"] = size_label
                    selected_df["attention_feature_count"] = int(attention_dim)
                    selected_df["activation_feature_count"] = int(activation_dim)
                    selected_df["selected_feature_count"] = len(current_feature_names)
                    selected_df.to_csv(selected_path, index=False)

                    coefficient_df = extract_feature_weights(
                        best_model,
                        feature_names=current_feature_names,
                        target_name=target_name,
                        feature_space=feature_space_name,
                        train_env=run_spec.train_env_label,
                    )
                    coefficient_df["scenario_name"] = run_spec.scenario_name
                    coefficient_df["scenario_title"] = run_spec.scenario_title
                    coefficient_df["feature_space_title"] = feature_space.title
                    coefficient_df["feature_family_group"] = feature_space.family_title
                    coefficient_df["feature_space_attention_subset_key"] = feature_space_attention_subset_key
                    coefficient_df["feature_space_attention_subset_title"] = feature_space_attention_subset_title
                    coefficient_df["source_envs"] = run_spec.source_envs_label
                    coefficient_df["source_env_count"] = int(len(run_spec.source_envs))
                    coefficient_df["heldout_env"] = run_spec.heldout_env if run_spec.heldout_env is not None else pd.NA
                    coefficient_df["feature_size"] = pd.NA if feature_size is None else int(feature_size)
                    coefficient_df["feature_size_label"] = size_label
                    coefficient_df["attention_feature_count"] = int(attention_dim)
                    coefficient_df["activation_feature_count"] = int(activation_dim)
                    coefficient_df["selected_feature_count"] = len(current_feature_names)
                    coefficient_df.to_csv(coefficients_path, index=False)
                    all_coefficient_frames.append(coefficient_df)

                    selection_summary_row = {
                        "scenario_name": run_spec.scenario_name,
                        "scenario_title": run_spec.scenario_title,
                        "target_name": target_name,
                        "target_title": target_title,
                        "feature_space": feature_space_name,
                        "feature_space_title": feature_space.title,
                        "feature_family_group": feature_space.family_title,
                        "feature_space_attention_subset_key": feature_space_attention_subset_key,
                        "feature_space_attention_subset_title": feature_space_attention_subset_title,
                        "train_env": run_spec.train_env_label,
                        "source_envs": run_spec.source_envs_label,
                        "source_env_count": int(len(run_spec.source_envs)),
                        "heldout_env": run_spec.heldout_env if run_spec.heldout_env is not None else pd.NA,
                        "ood_env_count": int(len(run_spec.ood_envs)),
                        "feature_size": pd.NA if feature_size is None else int(feature_size),
                        "feature_size_label": size_label,
                        "attention_feature_count": int(attention_dim),
                        "activation_feature_count": int(activation_dim),
                        "selected_feature_count": len(current_feature_names),
                        "model_family": best_model.model_family,
                        "model_family_title": best_model.model_family_title,
                        "model_weight_kind": best_model.model_weight_kind,
                        "candidate_key": best_model.candidate_key,
                        "candidate_label": best_model.candidate_label,
                        "candidate_complexity": float(best_model.candidate_complexity),
                        "candidate_params_json": serialize_candidate_params(best_model.candidate_params),
                        "chosen_c": pd.NA if best_model.chosen_c is None else float(best_model.chosen_c),
                        "chosen_max_depth": pd.NA if best_model.candidate_max_depth is None else int(best_model.candidate_max_depth),
                        "decision_threshold": float(best_model.decision_threshold),
                        "effective_activation_pca_dim": (
                            pd.NA if activation_bundle is None or activation_bundle.effective_pca_dim is None else int(activation_bundle.effective_pca_dim)
                        ),
                        "model_selection_objective": MODEL_SELECTION_OBJECTIVE,
                        "oracle_mean_ood_auroc_selected": float(best_candidate_oracle_metrics["oracle_mean_ood_auroc"]),
                        "oracle_min_ood_auroc_selected": float(best_candidate_oracle_metrics["oracle_min_ood_auroc"]),
                        "oracle_std_ood_auroc_selected": float(best_candidate_oracle_metrics["oracle_std_ood_auroc"]),
                        "oracle_mean_ood_average_precision_selected": float(best_candidate_oracle_metrics["oracle_mean_ood_average_precision"]),
                        "candidate_path": str(candidate_path),
                        "selected_features_path": str(selected_path),
                        "coefficients_path": str(coefficients_path),
                        "selection_summary_path": str(selection_summary_path),
                        "transfer_metrics_path": str(transfer_metrics_path),
                        **best_model.validation_metrics,
                    }
                    pd.DataFrame([selection_summary_row]).to_csv(selection_summary_path, index=False)
                    all_model_selection_rows.append(selection_summary_row)

                    current_transfer_rows: list[dict[str, Any]] = []
                    for eval_role, eval_envs in (("val", run_spec.source_envs), ("ood", run_spec.ood_envs)):
                        for test_env in eval_envs:
                            eval_cache = env_pool_cache[test_env]
                            x_eval = assemble_split_matrix(
                                eval_cache,
                                split_name="val",
                                feature_space=feature_space,
                                attention_dim=attention_dim,
                                activation_dim=activation_dim,
                            )
                            eval_scores = best_model.estimator.predict_proba(x_eval)[:, 1].astype(np.float32)
                            eval_metrics = summarize_score_metrics(
                                eval_cache["y_val"],
                                eval_scores,
                                decision_threshold=float(best_model.decision_threshold),
                            )
                            transfer_row = {
                                "scenario_name": run_spec.scenario_name,
                                "scenario_title": run_spec.scenario_title,
                                "target_name": target_name,
                                "target_title": target_title,
                                "feature_space": feature_space_name,
                                "feature_space_title": feature_space.title,
                                "feature_family_group": feature_space.family_title,
                                "feature_space_attention_subset_key": feature_space_attention_subset_key,
                                "feature_space_attention_subset_title": feature_space_attention_subset_title,
                                "train_env": run_spec.train_env_label,
                                "source_envs": run_spec.source_envs_label,
                                "source_env_count": int(len(run_spec.source_envs)),
                                "heldout_env": run_spec.heldout_env if run_spec.heldout_env is not None else pd.NA,
                                "test_env": test_env,
                                "eval_role": eval_role,
                                "feature_size": pd.NA if feature_size is None else int(feature_size),
                                "feature_size_label": size_label,
                                "attention_feature_count": int(attention_dim),
                                "activation_feature_count": int(activation_dim),
                                "selected_feature_count": len(current_feature_names),
                                "model_family": best_model.model_family,
                                "model_family_title": best_model.model_family_title,
                                "model_weight_kind": best_model.model_weight_kind,
                                "candidate_key": best_model.candidate_key,
                                "candidate_label": best_model.candidate_label,
                                "candidate_complexity": float(best_model.candidate_complexity),
                                "candidate_params_json": serialize_candidate_params(best_model.candidate_params),
                                "chosen_c": pd.NA if best_model.chosen_c is None else float(best_model.chosen_c),
                                "chosen_max_depth": pd.NA if best_model.candidate_max_depth is None else int(best_model.candidate_max_depth),
                                "decision_threshold": float(best_model.decision_threshold),
                                "effective_activation_pca_dim": (
                                    pd.NA if activation_bundle is None or activation_bundle.effective_pca_dim is None else int(activation_bundle.effective_pca_dim)
                                ),
                                "model_selection_objective": MODEL_SELECTION_OBJECTIVE,
                                "oracle_mean_ood_auroc_selected": float(best_candidate_oracle_metrics["oracle_mean_ood_auroc"]),
                                "selected_features_path": str(selected_path),
                                "coefficients_path": str(coefficients_path),
                                "transfer_metrics_path": str(transfer_metrics_path),
                                **eval_metrics,
                            }
                            current_transfer_rows.append(transfer_row)
                            all_transfer_metric_rows.append(transfer_row)

                    pd.DataFrame(current_transfer_rows).to_csv(transfer_metrics_path, index=False)

                    gc.collect()
                    completed_model_selection_count += 1
                    if completed_model_selection_count % CHECKPOINT_EVERY_MODEL_SELECTIONS == 0:
                        persist_core_artifacts(
                            transfer_rows=all_transfer_metric_rows,
                            model_selection_rows=all_model_selection_rows,
                            coefficient_frames=all_coefficient_frames,
                        )
finally:
    all_transfer_metrics_df, all_model_selection_df, all_coefficients_df = persist_core_artifacts(
        transfer_rows=all_transfer_metric_rows,
        model_selection_rows=all_model_selection_rows,
        coefficient_frames=all_coefficient_frames,
    )


# %%
def family_uses_requested_feature_size(feature_family_group: str) -> bool:
    return feature_family_group in {"activation_only", "attention_plus_activation"}


def optional_float(value: Any) -> float:
    return float(value) if pd.notna(value) else float("nan")


# %%
def summarize_transfer_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows: list[dict[str, Any]] = []
    group_cols = [
        "scenario_name",
        "scenario_title",
        "target_name",
        "target_title",
        "feature_space",
        "feature_space_title",
        "feature_family_group",
        "feature_size",
        "feature_size_label",
    ]
    for group_values, group_df in metrics_df.groupby(group_cols, dropna=False, sort=False):
        (
            scenario_name,
            scenario_title,
            target_name,
            target_title,
            feature_space,
            feature_space_title,
            feature_family_group,
            feature_size,
            feature_size_label,
        ) = group_values
        diagonal_df = group_df.loc[group_df["eval_role"].eq("val")].copy()
        ood_df = group_df.loc[group_df["eval_role"].eq("ood")].copy()
        meta = group_df.iloc[0]
        summary_rows.append(
            {
                "scenario_name": scenario_name,
                "scenario_title": scenario_title,
                "target_name": target_name,
                "target_title": target_title,
                "feature_space": feature_space,
                "feature_space_title": feature_space_title,
                "feature_family_group": feature_family_group,
                "feature_size": feature_size,
                "feature_size_label": feature_size_label,
                "attention_feature_count": int(meta["attention_feature_count"]),
                "activation_feature_count": int(meta["activation_feature_count"]),
                "selected_feature_count": int(meta["selected_feature_count"]),
                "effective_activation_pca_dim": meta.get("effective_activation_pca_dim", pd.NA),
                "n_source_val_envs": int(diagonal_df["test_env"].nunique()),
                "n_ood_envs": int(ood_df["test_env"].nunique()),
                "mean_val_auroc": safe_metric_mean(diagonal_df["auroc"]),
                "min_val_auroc": safe_metric_min(diagonal_df["auroc"]),
                "mean_ood_auroc": safe_metric_mean(ood_df["auroc"]),
                "min_ood_auroc": safe_metric_min(ood_df["auroc"]),
                "std_ood_auroc": safe_metric_std(ood_df["auroc"]),
                "mean_val_average_precision": safe_metric_mean(diagonal_df["average_precision"]),
                "mean_ood_average_precision": safe_metric_mean(ood_df["average_precision"]),
                "mean_val_balanced_accuracy": safe_metric_mean(diagonal_df["balanced_accuracy"]),
                "mean_ood_balanced_accuracy": safe_metric_mean(ood_df["balanced_accuracy"]),
            }
        )
    if not summary_rows:
        return pd.DataFrame()
    out = pd.DataFrame(summary_rows)
    out["_feature_size_sort"] = pd.to_numeric(out["feature_size"], errors="coerce").fillna(-1).astype(int)
    out = out.sort_values(
        ["scenario_name", "target_name", "_feature_size_sort", "mean_ood_auroc", "min_ood_auroc", "mean_val_auroc"],
        ascending=[True, True, True, False, False, False],
    ).drop(columns="_feature_size_sort")
    return out.reset_index(drop=True)


def summarize_train_env_models(metrics_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows: list[dict[str, Any]] = []
    group_cols = [
        "scenario_name",
        "scenario_title",
        "target_name",
        "target_title",
        "feature_space",
        "feature_space_title",
        "feature_family_group",
        "feature_size",
        "feature_size_label",
        "train_env",
        "source_envs",
        "source_env_count",
        "heldout_env",
    ]
    for group_values, group_df in metrics_df.groupby(group_cols, dropna=False, sort=False):
        (
            scenario_name,
            scenario_title,
            target_name,
            target_title,
            feature_space,
            feature_space_title,
            feature_family_group,
            feature_size,
            feature_size_label,
            train_env,
            source_envs,
            source_env_count,
            heldout_env,
        ) = group_values
        diagonal_df = group_df.loc[group_df["eval_role"].eq("val")].copy()
        ood_df = group_df.loc[group_df["eval_role"].eq("ood")].copy()
        selection_meta = group_df.iloc[0]
        summary_rows.append(
            {
                "scenario_name": scenario_name,
                "scenario_title": scenario_title,
                "target_name": target_name,
                "target_title": target_title,
                "feature_space": feature_space,
                "feature_space_title": feature_space_title,
                "feature_family_group": feature_family_group,
                "feature_size": feature_size,
                "feature_size_label": feature_size_label,
                "train_env": train_env,
                "source_envs": source_envs,
                "source_env_count": int(source_env_count),
                "heldout_env": heldout_env,
                "model_selection_objective": selection_meta.get("model_selection_objective", MODEL_SELECTION_OBJECTIVE),
                "source_val_auroc": safe_metric_mean(diagonal_df["auroc"]),
                "source_val_average_precision": safe_metric_mean(diagonal_df["average_precision"]),
                "source_val_balanced_accuracy": safe_metric_mean(diagonal_df["balanced_accuracy"]),
                "mean_ood_auroc": safe_metric_mean(ood_df["auroc"]),
                "min_ood_auroc": safe_metric_min(ood_df["auroc"]),
                "std_ood_auroc": safe_metric_std(ood_df["auroc"]),
                "mean_ood_average_precision": safe_metric_mean(ood_df["average_precision"]),
                "mean_ood_balanced_accuracy": safe_metric_mean(ood_df["balanced_accuracy"]),
                "n_source_val_envs": int(diagonal_df["test_env"].nunique()),
                "n_ood_envs": int(ood_df["test_env"].nunique()),
                "attention_feature_count": int(selection_meta["attention_feature_count"]),
                "activation_feature_count": int(selection_meta["activation_feature_count"]),
                "selected_feature_count": int(selection_meta["selected_feature_count"]),
                "chosen_c": optional_float(selection_meta.get("chosen_c", pd.NA)),
                "chosen_max_depth": optional_float(selection_meta.get("chosen_max_depth", pd.NA)),
                "decision_threshold": float(selection_meta["decision_threshold"]),
                "effective_activation_pca_dim": selection_meta.get("effective_activation_pca_dim", pd.NA),
                "oracle_mean_ood_auroc_selected": optional_float(selection_meta.get("oracle_mean_ood_auroc_selected", pd.NA)),
                "selected_features_path": str(selection_meta["selected_features_path"]),
                "coefficients_path": str(selection_meta.get("coefficients_path", "")),
            }
        )
    if not summary_rows:
        return pd.DataFrame()
    out = pd.DataFrame(summary_rows)
    out["_feature_size_sort"] = pd.to_numeric(out["feature_size"], errors="coerce").fillna(-1).astype(int)
    out = out.sort_values(
        ["scenario_name", "target_name", "_feature_size_sort", "mean_ood_auroc", "min_ood_auroc", "source_val_auroc"],
        ascending=[True, True, True, False, False, False],
    ).drop(columns="_feature_size_sort")
    return out.reset_index(drop=True)


def summarize_confusion_counts(metrics_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows: list[dict[str, Any]] = []
    group_cols = [
        "scenario_name",
        "scenario_title",
        "target_name",
        "target_title",
        "feature_space",
        "feature_space_title",
        "feature_family_group",
        "feature_size",
        "feature_size_label",
        "eval_role",
    ]
    for group_values, group_df in metrics_df.groupby(group_cols, dropna=False, sort=False):
        (
            scenario_name,
            scenario_title,
            target_name,
            target_title,
            feature_space,
            feature_space_title,
            feature_family_group,
            feature_size,
            feature_size_label,
            eval_role,
        ) = group_values
        meta = group_df.iloc[0]
        summary_rows.append(
            {
                "scenario_name": scenario_name,
                "scenario_title": scenario_title,
                "target_name": target_name,
                "target_title": target_title,
                "feature_space": feature_space,
                "feature_space_title": feature_space_title,
                "feature_family_group": feature_family_group,
                "feature_size": feature_size,
                "feature_size_label": feature_size_label,
                "eval_role": eval_role,
                "attention_feature_count": int(meta["attention_feature_count"]),
                "activation_feature_count": int(meta["activation_feature_count"]),
                "selected_feature_count": int(meta["selected_feature_count"]),
                "sum_tn": int(group_df["tn"].sum()),
                "sum_fp": int(group_df["fp"].sum()),
                "sum_fn": int(group_df["fn"].sum()),
                "sum_tp": int(group_df["tp"].sum()),
                "n_pairs": int(len(group_df)),
            }
        )
    if not summary_rows:
        return pd.DataFrame()
    out = pd.DataFrame(summary_rows)
    out["_feature_size_sort"] = pd.to_numeric(out["feature_size"], errors="coerce").fillna(-1).astype(int)
    out = out.sort_values(
        ["scenario_name", "target_name", "_feature_size_sort", "eval_role", "feature_space"],
        ascending=[True, True, True, True, True],
    ).drop(columns="_feature_size_sort")
    return out.reset_index(drop=True)


def summarize_coefficients(coefficients_df: pd.DataFrame, ranking_df: pd.DataFrame) -> pd.DataFrame:
    if coefficients_df.empty:
        return pd.DataFrame()
    coefficient_summary_df = (
        coefficients_df.groupby(
            ["scenario_name", "scenario_title", "target_name", "feature_space", "feature_size", "feature_size_label", "feature"],
            as_index=False,
        )
        .agg(
            mean_coefficient=("coefficient", "mean"),
            mean_abs_coefficient=("abs_coefficient", "mean"),
            max_abs_coefficient=("abs_coefficient", "max"),
            selected_in_sources=("train_env", "nunique"),
        )
        .sort_values(
            ["scenario_name", "target_name", "feature_space", "feature_size_label", "mean_abs_coefficient"],
            ascending=[True, True, True, True, False],
        )
        .reset_index(drop=True)
    )
    ranking_cols = [
        "target_name",
        "feature_space",
        "feature",
        "feature_root",
        "family",
        "metric_name",
        "head_summary",
        "band",
        "band_stat",
        "global_rank",
        "same_sign_all",
        "sign_direction",
        "consistency_score",
        "min_abs_effect",
        "mean_abs_effect",
        "median_abs_effect",
        "std_effect",
    ]
    ranking_summary_df = ranking_df.loc[:, ranking_cols].drop_duplicates(
        subset=["target_name", "feature_space", "feature"],
        keep="first",
    )
    return coefficient_summary_df.merge(
        ranking_summary_df,
        on=["target_name", "feature_space", "feature"],
        how="left",
        validate="many_to_one",
    )


def build_family_panel_selection(transfer_summary_df: pd.DataFrame) -> pd.DataFrame:
    family_sort = {family: idx for idx, family in enumerate(FAMILY_PANEL_ORDER)}
    rows: list[dict[str, Any]] = []
    for scenario_name, scenario_title in SELECTED_SCENARIO_TITLES.items():
        for target_name, target_spec in TARGET_SPECS.items():
            target_title = str(target_spec["title"])
            for requested_feature_size in FEATURE_SIZE_GRID:
                for family in FAMILY_PANEL_ORDER:
                    subset = transfer_summary_df.loc[
                        transfer_summary_df["scenario_name"].eq(scenario_name)
                        & transfer_summary_df["target_name"].eq(target_name)
                        & transfer_summary_df["feature_family_group"].eq(family)
                    ].copy()
                    if family_uses_requested_feature_size(family):
                        subset = subset.loc[pd.to_numeric(subset["feature_size"], errors="coerce").eq(int(requested_feature_size))]
                    if subset.empty:
                        continue
                    subset = subset.sort_values(
                        ["mean_ood_auroc", "min_ood_auroc", "mean_val_auroc", "selected_feature_count"],
                        ascending=[False, False, False, True],
                    ).reset_index(drop=True)
                    best = subset.iloc[0]
                    rows.append(
                        {
                            "scenario_name": scenario_name,
                            "scenario_title": scenario_title,
                            "target_name": target_name,
                            "target_title": target_title,
                            "requested_feature_size": int(requested_feature_size),
                            "requested_feature_size_label": feature_size_to_label(int(requested_feature_size)),
                            "feature_family_group": family,
                            "selected_feature_space": str(best["feature_space"]),
                            "selected_feature_space_title": str(best["feature_space_title"]),
                            "source_feature_size": best["feature_size"],
                            "source_feature_size_label": str(best["feature_size_label"]),
                            "attention_feature_count": int(best["attention_feature_count"]),
                            "activation_feature_count": int(best["activation_feature_count"]),
                            "selected_feature_count": int(best["selected_feature_count"]),
                            "effective_activation_pca_dim": best.get("effective_activation_pca_dim", pd.NA),
                            "mean_val_auroc": float(best["mean_val_auroc"]),
                            "mean_ood_auroc": float(best["mean_ood_auroc"]),
                            "min_ood_auroc": float(best["min_ood_auroc"]),
                            "std_ood_auroc": float(best["std_ood_auroc"]),
                            "mean_ood_average_precision": float(best["mean_ood_average_precision"]),
                        }
                    )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["_family_sort"] = out["feature_family_group"].map(family_sort).fillna(len(family_sort)).astype(int)
    out = out.sort_values(
        ["scenario_name", "target_name", "requested_feature_size", "_family_sort", "mean_ood_auroc"],
        ascending=[True, True, True, True, False],
    ).drop(columns="_family_sort")
    return out.reset_index(drop=True)


def build_best_family_models(
    train_env_model_summary_df: pd.DataFrame,
    panel_selection_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for panel_row in panel_selection_df.itertuples(index=False):
        subset = train_env_model_summary_df.loc[
            train_env_model_summary_df["scenario_name"].eq(panel_row.scenario_name)
            & train_env_model_summary_df["target_name"].eq(panel_row.target_name)
            & train_env_model_summary_df["feature_space"].eq(panel_row.selected_feature_space)
            & train_env_model_summary_df["feature_size_label"].eq(panel_row.source_feature_size_label)
        ].copy()
        if subset.empty:
            continue
        subset = subset.sort_values(
            ["mean_ood_auroc", "min_ood_auroc", "source_val_auroc"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        best = subset.iloc[0]
        rows.append(
            {
                "scenario_name": panel_row.scenario_name,
                "scenario_title": panel_row.scenario_title,
                "target_name": panel_row.target_name,
                "target_title": panel_row.target_title,
                "requested_feature_size": int(panel_row.requested_feature_size),
                "requested_feature_size_label": panel_row.requested_feature_size_label,
                "feature_family_group": panel_row.feature_family_group,
                "feature_space": str(best["feature_space"]),
                "feature_space_title": str(best["feature_space_title"]),
                "feature_size": best["feature_size"],
                "feature_size_label": str(best["feature_size_label"]),
                "train_env": str(best["train_env"]),
                "source_envs": str(best["source_envs"]),
                "source_env_count": int(best["source_env_count"]),
                "heldout_env": best["heldout_env"],
                "source_val_auroc": float(best["source_val_auroc"]),
                "mean_ood_auroc": float(best["mean_ood_auroc"]),
                "min_ood_auroc": float(best["min_ood_auroc"]),
                "std_ood_auroc": float(best["std_ood_auroc"]),
                "mean_ood_average_precision": float(best["mean_ood_average_precision"]),
                "mean_ood_balanced_accuracy": float(best["mean_ood_balanced_accuracy"]),
                "attention_feature_count": int(best["attention_feature_count"]),
                "activation_feature_count": int(best["activation_feature_count"]),
                "selected_feature_count": int(best["selected_feature_count"]),
                "chosen_c": optional_float(best["chosen_c"]),
                "chosen_max_depth": optional_float(best["chosen_max_depth"]),
                "decision_threshold": float(best["decision_threshold"]),
                "effective_activation_pca_dim": best.get("effective_activation_pca_dim", pd.NA),
                "selected_features_path": str(best["selected_features_path"]),
                "coefficients_path": str(best["coefficients_path"]),
            }
        )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    family_sort = {family: idx for idx, family in enumerate(FAMILY_PANEL_ORDER)}
    out["_family_sort"] = out["feature_family_group"].map(family_sort).fillna(len(family_sort)).astype(int)
    out = out.sort_values(
        ["scenario_name", "target_name", "requested_feature_size", "_family_sort", "mean_ood_auroc"],
        ascending=[True, True, True, True, False],
    ).drop(columns="_family_sort")
    return out.reset_index(drop=True)


def build_selected_panel_confusion_summary(
    metrics_df: pd.DataFrame,
    panel_selection_df: pd.DataFrame,
    *,
    eval_role: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for panel_row in panel_selection_df.itertuples(index=False):
        subset = metrics_df.loc[
            metrics_df["scenario_name"].eq(panel_row.scenario_name)
            & metrics_df["target_name"].eq(panel_row.target_name)
            & metrics_df["feature_space"].eq(panel_row.selected_feature_space)
            & metrics_df["feature_size_label"].eq(panel_row.source_feature_size_label)
            & metrics_df["eval_role"].eq(eval_role)
        ].copy()
        if subset.empty:
            continue
        rows.append(
            {
                "scenario_name": panel_row.scenario_name,
                "scenario_title": panel_row.scenario_title,
                "target_name": panel_row.target_name,
                "target_title": panel_row.target_title,
                "requested_feature_size": int(panel_row.requested_feature_size),
                "requested_feature_size_label": panel_row.requested_feature_size_label,
                "feature_family_group": panel_row.feature_family_group,
                "feature_space": panel_row.selected_feature_space,
                "feature_space_title": panel_row.selected_feature_space_title,
                "feature_size_label": panel_row.source_feature_size_label,
                "eval_role": eval_role,
                "sum_tn": int(subset["tn"].sum()),
                "sum_fp": int(subset["fp"].sum()),
                "sum_fn": int(subset["fn"].sum()),
                "sum_tp": int(subset["tp"].sum()),
                "n_pairs": int(len(subset)),
            }
        )
    if not rows:
        return pd.DataFrame()
    family_sort = {family: idx for idx, family in enumerate(FAMILY_PANEL_ORDER)}
    out = pd.DataFrame(rows)
    out["_family_sort"] = out["feature_family_group"].map(family_sort).fillna(len(family_sort)).astype(int)
    out = out.sort_values(
        ["scenario_name", "target_name", "requested_feature_size", "eval_role", "_family_sort"],
        ascending=[True, True, True, True, True],
    ).drop(columns="_family_sort")
    return out.reset_index(drop=True)


transfer_summary_df = summarize_transfer_metrics(all_transfer_metrics_df)
train_env_model_summary_df = summarize_train_env_models(all_transfer_metrics_df)
confusion_summary_df = summarize_confusion_counts(all_transfer_metrics_df)
coefficient_summary_df = summarize_coefficients(all_coefficients_df, ranking_df_all)
family_panel_selection_df = build_family_panel_selection(transfer_summary_df)
best_family_models_df = build_best_family_models(train_env_model_summary_df, family_panel_selection_df)
selected_panel_confusion_val_df = build_selected_panel_confusion_summary(
    all_transfer_metrics_df,
    family_panel_selection_df,
    eval_role="val",
)
selected_panel_confusion_ood_df = build_selected_panel_confusion_summary(
    all_transfer_metrics_df,
    family_panel_selection_df,
    eval_role="ood",
)
selected_panel_confusion_df = pd.concat(
    [selected_panel_confusion_val_df, selected_panel_confusion_ood_df],
    ignore_index=True,
) if not selected_panel_confusion_val_df.empty or not selected_panel_confusion_ood_df.empty else pd.DataFrame()

top_feature_tables: list[pd.DataFrame] = []
top_feature_dir = OUTPUT_ROOT / "best_features"
top_feature_dir.mkdir(parents=True, exist_ok=True)
for row in best_family_models_df.itertuples(index=False):
    feature_table = (
        all_coefficients_df.loc[
            all_coefficients_df["scenario_name"].eq(row.scenario_name)
            & all_coefficients_df["target_name"].eq(row.target_name)
            & all_coefficients_df["feature_space"].eq(row.feature_space)
            & all_coefficients_df["feature_size_label"].eq(row.feature_size_label)
            & all_coefficients_df["train_env"].eq(row.train_env)
        ]
        .merge(
            ranking_df_all.loc[
                ranking_df_all["target_name"].eq(row.target_name)
                & ranking_df_all["feature_space"].eq(row.feature_space)
            ].drop(columns=["target_name", "feature_space"], errors="ignore"),
            on="feature",
            how="left",
            validate="many_to_one",
        )
        .sort_values(["abs_coefficient", "global_rank"], ascending=[False, True], na_position="last")
        .reset_index(drop=True)
    )
    if feature_table.empty:
        continue
    feature_table["scenario_name"] = row.scenario_name
    feature_table["scenario_title"] = row.scenario_title
    feature_table["requested_feature_size"] = int(row.requested_feature_size)
    feature_table["requested_feature_size_label"] = row.requested_feature_size_label
    feature_table["feature_family_group"] = row.feature_family_group
    feature_table.insert(0, "importance_rank", np.arange(1, len(feature_table) + 1, dtype=int))
    top_feature_df = feature_table.head(int(TOP_FEATURES_TO_SHOW)).copy()
    top_feature_tables.append(top_feature_df)
    scenario_dir = top_feature_dir / str(row.scenario_name)
    scenario_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"{row.target_name}__{row.feature_family_group}__{row.requested_feature_size_label}__top_features.csv"
    top_feature_df.to_csv(scenario_dir / out_name, index=False)

top_features_for_best_models_df = pd.concat(top_feature_tables, ignore_index=True) if top_feature_tables else pd.DataFrame()

display(transfer_summary_df)
display(train_env_model_summary_df)
display(confusion_summary_df)
display(family_panel_selection_df)
display(best_family_models_df)
display(selected_panel_confusion_val_df)
display(selected_panel_confusion_ood_df)
display(top_features_for_best_models_df)

transfer_summary_df.to_csv(OUTPUT_ROOT / "transfer_summary.csv", index=False)
train_env_model_summary_df.to_csv(OUTPUT_ROOT / "train_env_model_summary.csv", index=False)
confusion_summary_df.to_csv(OUTPUT_ROOT / "confusion_summary.csv", index=False)
coefficient_summary_df.to_csv(OUTPUT_ROOT / "coefficient_summary.csv", index=False)
family_panel_selection_df.to_csv(OUTPUT_ROOT / "best_feature_space_by_target_size_family.csv", index=False)
best_family_models_df.to_csv(OUTPUT_ROOT / "best_model_by_target_size_family.csv", index=False)
selected_panel_confusion_val_df.to_csv(OUTPUT_ROOT / "selected_panel_confusion_val.csv", index=False)
selected_panel_confusion_ood_df.to_csv(OUTPUT_ROOT / "selected_panel_confusion_ood.csv", index=False)
selected_panel_confusion_df.to_csv(OUTPUT_ROOT / "selected_panel_confusion_all.csv", index=False)
top_features_for_best_models_df.to_csv(OUTPUT_ROOT / "top_features_for_best_models.csv", index=False)


# %%
def target_confusion_labels(target_name: str) -> tuple[str, str]:
    target_spec = TARGET_SPECS[target_name]
    return str(target_spec["negative_label"]), str(target_spec["positive_label"])


def build_transfer_matrix_for_selection(
    metrics_df: pd.DataFrame,
    *,
    scenario_name: str,
    target_name: str,
    feature_space: str,
    feature_size_label: str,
) -> pd.DataFrame:
    subset = metrics_df.loc[
        metrics_df["scenario_name"].eq(scenario_name)
        & metrics_df["target_name"].eq(target_name)
        & metrics_df["feature_space"].eq(feature_space)
        & metrics_df["feature_size_label"].eq(feature_size_label)
    ].copy()
    row_labels = TRAIN_AXIS_LABELS_BY_SCENARIO[str(scenario_name)]
    matrix = pd.DataFrame(index=row_labels, columns=ENV_ORDER, dtype=float)
    for row in subset.itertuples(index=False):
        matrix.loc[str(row.train_env), str(row.test_env)] = float(row.auroc) if pd.notna(row.auroc) else float("nan")
    return matrix


def build_confusion_matrix_df(
    confusion_df: pd.DataFrame,
    *,
    scenario_name: str,
    target_name: str,
    requested_feature_size: int,
    feature_family_group: str,
    eval_role: str,
) -> pd.DataFrame | None:
    subset = confusion_df.loc[
        confusion_df["scenario_name"].eq(scenario_name)
        & confusion_df["target_name"].eq(target_name)
        & confusion_df["requested_feature_size"].eq(int(requested_feature_size))
        & confusion_df["feature_family_group"].eq(feature_family_group)
        & confusion_df["eval_role"].eq(eval_role)
    ].copy()
    if subset.empty:
        return None
    row = subset.iloc[0]
    negative_label, positive_label = target_confusion_labels(target_name)
    return pd.DataFrame(
        [
            [int(row["sum_tn"]), int(row["sum_fp"])],
            [int(row["sum_fn"]), int(row["sum_tp"])],
        ],
        index=[f"Actual {negative_label}", f"Actual {positive_label}"],
        columns=[f"Pred {negative_label}", f"Pred {positive_label}"],
    )


def export_selected_family_panel_tables(
    metrics_df: pd.DataFrame,
    panel_selection_df: pd.DataFrame,
    confusion_df: pd.DataFrame,
) -> None:
    export_root = OUTPUT_ROOT / "selected_family_panel_tables"
    export_root.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    for panel_row in panel_selection_df.itertuples(index=False):
        panel_dir = export_root / str(panel_row.scenario_name) / panel_row.target_name / str(panel_row.requested_feature_size_label)
        panel_dir.mkdir(parents=True, exist_ok=True)

        matrix_df = build_transfer_matrix_for_selection(
            metrics_df,
            scenario_name=str(panel_row.scenario_name),
            target_name=str(panel_row.target_name),
            feature_space=str(panel_row.selected_feature_space),
            feature_size_label=str(panel_row.source_feature_size_label),
        )
        matrix_path = panel_dir / f"{panel_row.feature_family_group}__auroc_matrix.csv"
        matrix_df.to_csv(matrix_path, index=True)

        val_confusion_matrix_df = build_confusion_matrix_df(
            confusion_df,
            scenario_name=str(panel_row.scenario_name),
            target_name=str(panel_row.target_name),
            requested_feature_size=int(panel_row.requested_feature_size),
            feature_family_group=str(panel_row.feature_family_group),
            eval_role="val",
        )
        val_confusion_path = panel_dir / f"{panel_row.feature_family_group}__val_confusion_counts.csv"
        if val_confusion_matrix_df is not None:
            val_confusion_matrix_df.to_csv(val_confusion_path, index=True)

        ood_confusion_matrix_df = build_confusion_matrix_df(
            confusion_df,
            scenario_name=str(panel_row.scenario_name),
            target_name=str(panel_row.target_name),
            requested_feature_size=int(panel_row.requested_feature_size),
            feature_family_group=str(panel_row.feature_family_group),
            eval_role="ood",
        )
        ood_confusion_path = panel_dir / f"{panel_row.feature_family_group}__ood_confusion_counts.csv"
        if ood_confusion_matrix_df is not None:
            ood_confusion_matrix_df.to_csv(ood_confusion_path, index=True)

        manifest_rows.append(
            {
                "scenario_name": panel_row.scenario_name,
                "scenario_title": panel_row.scenario_title,
                "target_name": panel_row.target_name,
                "target_title": panel_row.target_title,
                "requested_feature_size": int(panel_row.requested_feature_size),
                "requested_feature_size_label": panel_row.requested_feature_size_label,
                "feature_family_group": panel_row.feature_family_group,
                "selected_feature_space": panel_row.selected_feature_space,
                "selected_feature_space_title": panel_row.selected_feature_space_title,
                "source_feature_size_label": panel_row.source_feature_size_label,
                "attention_feature_count": int(panel_row.attention_feature_count),
                "activation_feature_count": int(panel_row.activation_feature_count),
                "selected_feature_count": int(panel_row.selected_feature_count),
                "auroc_matrix_path": str(matrix_path),
                "val_confusion_counts_path": str(val_confusion_path if val_confusion_matrix_df is not None else ""),
                "ood_confusion_counts_path": str(ood_confusion_path if ood_confusion_matrix_df is not None else ""),
            }
        )
    pd.DataFrame(manifest_rows).to_csv(export_root / "panel_table_manifest.csv", index=False)


def plot_feature_size_family_transfer_panels(
    metrics_df: pd.DataFrame,
    panel_selection_df: pd.DataFrame,
    *,
    scenario_name: str,
    target_name: str,
    requested_feature_size: int,
) -> None:
    selection_subset = (
        panel_selection_df.loc[
            panel_selection_df["scenario_name"].eq(scenario_name)
            & panel_selection_df["target_name"].eq(target_name)
            & panel_selection_df["requested_feature_size"].eq(int(requested_feature_size))
        ]
        .set_index("feature_family_group")
        .reindex(list(FAMILY_PANEL_ORDER))
        .reset_index()
    )
    matrices: dict[str, pd.DataFrame] = {}
    finite_values_list: list[np.ndarray] = []
    row_labels = TRAIN_AXIS_LABELS_BY_SCENARIO[str(scenario_name)]
    for row in selection_subset.itertuples(index=False):
        if pd.isna(row.selected_feature_space):
            continue
        matrix_df = build_transfer_matrix_for_selection(
            metrics_df,
            scenario_name=str(scenario_name),
            target_name=target_name,
            feature_space=str(row.selected_feature_space),
            feature_size_label=str(row.source_feature_size_label),
        )
        matrices[str(row.feature_family_group)] = matrix_df
        finite_values_list.append(matrix_df.to_numpy(dtype=float).ravel())

    finite_values = np.concatenate(finite_values_list) if finite_values_list else np.array([], dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    vmin = float(np.min(finite_values)) if finite_values.size else 0.0
    vmax = float(np.max(finite_values)) if finite_values.size else 1.0
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="lightgray")

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 10.2), constrained_layout=True)
    axes = np.asarray(axes).reshape(2, 2)
    image = None

    for idx, family in enumerate(FAMILY_PANEL_ORDER):
        ax = axes.flat[idx]
        row_df = selection_subset.loc[selection_subset["feature_family_group"].eq(family)]
        if row_df.empty or family not in matrices:
            ax.axis("off")
            continue
        row = row_df.iloc[0]
        matrix_df = matrices[family].reindex(index=row_labels, columns=ENV_ORDER)
        matrix = np.ma.masked_invalid(matrix_df.to_numpy(dtype=float))
        image = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(np.arange(len(ENV_ORDER)))
        ax.set_xticklabels(ENV_ORDER, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_yticklabels(row_labels)
        ax.set_xlabel("Evaluation env")
        ax.set_ylabel("Training source(s)")
        title = (
            f"{FAMILY_DISPLAY_TITLES.get(family, str(row['feature_family_group']))}\n"
            f"{row['selected_feature_space_title']}\n"
            f"selected={int(row['selected_feature_count'])}, attn={int(row['attention_feature_count'])}, act={int(row['activation_feature_count'])}"
        )
        ax.set_title(title, fontsize=10.0)
        midpoint = (vmin + vmax) / 2.0 if np.isfinite(vmin) and np.isfinite(vmax) else 0.5
        for r in range(matrix_df.shape[0]):
            for c in range(matrix_df.shape[1]):
                value = matrix_df.iat[r, c]
                text = "nan" if not np.isfinite(value) else f"{value:.3f}"
                text_color = "white" if np.isfinite(value) and value < midpoint else "black"
                ax.text(c, r, text, ha="center", va="center", color=text_color, fontsize=8.0)

    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02, label="AUROC")

    scenario_note = (
        "source-env columns = validation AUROC; held-out column = OOD AUROC"
        if scenario_name == "holdout_env_ood"
        else "diagonal = source validation AUROC; off-diagonal = OOD AUROC"
    )
    fig.suptitle(
        f"{SELECTED_SCENARIO_TITLES[str(scenario_name)]}\n"
        f"{TARGET_SPECS[target_name]['title']} | feature size {requested_feature_size}\n"
        f"{scenario_note}",
        fontsize=14,
    )
    plot_dir = OUTPUT_ROOT / "plots" / str(scenario_name)
    plot_dir.mkdir(parents=True, exist_ok=True)
    out_path = plot_dir / f"{target_name}__k{int(requested_feature_size):03d}__family_transfer_heatmaps.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.show()


def plot_feature_size_family_confusion_panels(
    confusion_df: pd.DataFrame,
    *,
    scenario_name: str,
    target_name: str,
    requested_feature_size: int,
    eval_role: str,
) -> None:
    subset = (
        confusion_df.loc[
            confusion_df["scenario_name"].eq(scenario_name)
            & confusion_df["target_name"].eq(target_name)
            & confusion_df["requested_feature_size"].eq(int(requested_feature_size))
            & confusion_df["eval_role"].eq(eval_role)
        ]
        .set_index("feature_family_group")
        .reindex(list(FAMILY_PANEL_ORDER))
        .reset_index()
    )
    negative_label, positive_label = target_confusion_labels(target_name)
    vmax = 1.0
    for row in subset.itertuples(index=False):
        if pd.isna(row.sum_tn):
            continue
        vmax = max(vmax, float(max(row.sum_tn, row.sum_fp, row.sum_fn, row.sum_tp)))

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 9.6), constrained_layout=True)
    axes = np.asarray(axes).reshape(2, 2)
    image = None

    for idx, family in enumerate(FAMILY_PANEL_ORDER):
        ax = axes.flat[idx]
        row_df = subset.loc[subset["feature_family_group"].eq(family)]
        if row_df.empty or row_df[["sum_tn", "sum_fp", "sum_fn", "sum_tp"]].isna().all(axis=None):
            ax.axis("off")
            continue
        row = row_df.iloc[0]
        matrix = np.array(
            [
                [float(row["sum_tn"]), float(row["sum_fp"])],
                [float(row["sum_fn"]), float(row["sum_tp"])],
            ],
            dtype=float,
        )
        image = ax.imshow(matrix, vmin=0.0, vmax=vmax, cmap=plt.cm.Blues)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"Pred {negative_label}", f"Pred {positive_label}"], rotation=15, ha="right")
        ax.set_yticks([0, 1])
        ax.set_yticklabels([f"Actual {negative_label}", f"Actual {positive_label}"])
        ax.set_title(
            f"{FAMILY_DISPLAY_TITLES.get(family, str(row['feature_family_group']))}\n"
            f"{row['feature_space_title']}\n"
            f"n_pairs={int(row['n_pairs'])}",
            fontsize=10.0,
        )
        midpoint = vmax / 2.0
        for r in range(2):
            for c in range(2):
                value = matrix[r, c]
                ax.text(
                    c,
                    r,
                    f"{int(round(value))}",
                    ha="center",
                    va="center",
                    color="white" if value > midpoint else "black",
                    fontsize=9.0,
                )

    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02, label="Count")
    eval_role_title = "Source validation" if eval_role == "val" else "OOD"
    fig.suptitle(
        f"{SELECTED_SCENARIO_TITLES[str(scenario_name)]}\n"
        f"Summed {eval_role_title.lower()} raw confusion counts for {TARGET_SPECS[target_name]['title']} | feature size {requested_feature_size}",
        fontsize=14,
    )
    plot_dir = OUTPUT_ROOT / "plots" / str(scenario_name)
    plot_dir.mkdir(parents=True, exist_ok=True)
    out_path = plot_dir / f"{target_name}__k{int(requested_feature_size):03d}__{eval_role}_family_confusion_counts.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.show()


export_selected_family_panel_tables(
    all_transfer_metrics_df,
    family_panel_selection_df,
    selected_panel_confusion_df,
)


for scenario_name, scenario_title in SELECTED_SCENARIO_TITLES.items():
    print(f"\nScenario: {scenario_title}")
    for target_name in TARGET_SPECS:
        print(f"\nTarget: {TARGET_SPECS[target_name]['title']}")
        for requested_feature_size in FEATURE_SIZE_GRID:
            print(f"\nFeature size: {requested_feature_size}")
            selection_view = family_panel_selection_df.loc[
                family_panel_selection_df["scenario_name"].eq(scenario_name)
                & family_panel_selection_df["target_name"].eq(target_name)
                & family_panel_selection_df["requested_feature_size"].eq(int(requested_feature_size))
            ]
            display(selection_view)
            plot_feature_size_family_transfer_panels(
                all_transfer_metrics_df,
                family_panel_selection_df,
                scenario_name=scenario_name,
                target_name=target_name,
                requested_feature_size=int(requested_feature_size),
            )
            plot_feature_size_family_confusion_panels(
                selected_panel_confusion_val_df,
                scenario_name=scenario_name,
                target_name=target_name,
                requested_feature_size=int(requested_feature_size),
                eval_role="val",
            )
            plot_feature_size_family_confusion_panels(
                selected_panel_confusion_ood_df,
                scenario_name=scenario_name,
                target_name=target_name,
                requested_feature_size=int(requested_feature_size),
                eval_role="ood",
            )

print(f"Scenarios: {', '.join(SELECTED_SCENARIO_TITLES.values())}")
print(f"Model selection objective: {MODEL_SELECTION_OBJECTIVE}")
print(f"Fixed logistic C: {LOGREG_C:g}")
print(f"Fixed XGBoost max_depth: {XGB_MAX_DEPTH}")
print("\nBest feature-space variant by scenario, target, feature size, and family:")
display(family_panel_selection_df)
print("\nBest train-env model by scenario, target, feature size, and family:")
display(best_family_models_df)
print(f"\nTop {TOP_FEATURES_TO_SHOW} features for the best model in each scenario/target/size/family panel:")
display(top_features_for_best_models_df)
print("\nOutputs written to:")
print(OUTPUT_ROOT)
