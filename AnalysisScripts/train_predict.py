#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import re
import types
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]
OOD_SUPPORT_DIR = THIS_FILE.with_name("ood_support")
OOD_MODELING_LIB_PATH = OOD_SUPPORT_DIR / "ood_modeling_lib.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the OOD modeling pipeline and write the CSV/PNG artifacts "
            "used to analyze transfer AUROC, confusion summaries, and top features."
        )
    )
    parser.add_argument(
        "--model-dirname",
        required=True,
        help="DatasetMain model subdirectory name, e.g. DeepSeek-R1-Distill-Qwen-7B or gpt-oss-20b.",
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Explicit DatasetMain root. Useful when data lives outside the repo checkout.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional explicit output directory. Defaults to the standard scenario-aware OOD output path.",
    )
    parser.add_argument(
        "--structural-baseline-filename",
        default=None,
        help=(
            "Optional override for the companion structural-baseline parquet filename inside each dataset "
            "directory. This is used to align TF-IDF baseline rows; it does not enable a separate "
            "structural baseline sweep."
        ),
    )
    parser.add_argument(
        "--tfidf-cache-dirname",
        default=None,
        help="Optional override for the TF-IDF cache directory name inside each dataset directory.",
    )
    parser.add_argument(
        "--tfidf-text-fields",
        default=None,
        help="Optional comma-separated TF-IDF text fields to consider, e.g. last_sentence_text,prefix_text.",
    )
    parser.add_argument(
        "--only-tfidf",
        action="store_true",
        help=(
            "Run only the discovered TF-IDF baseline feature spaces. "
            "This is useful when the full attention/activation sweep already exists."
        ),
    )
    parser.add_argument(
        "--model-family",
        required=True,
        help="Model family to train for this run, e.g. logreg or xgb.",
    )
    parser.add_argument(
        "--feature-sizes",
        required=True,
        help="Comma-separated PCA feature-size sweep for activation-based feature spaces.",
    )
    parser.add_argument(
        "--scenarios",
        required=True,
        help=(
            "Comma-separated scenario keys. Supported: single_source_ood, holdout_env_ood. "
            "Example: single_source_ood,holdout_env_ood"
        ),
    )
    parser.add_argument(
        "--logreg-c",
        type=float,
        default=None,
        help="Fixed logistic-regression C value. Required when --model-family is logistic regression.",
    )
    parser.add_argument(
        "--xgb-max-depth",
        type=int,
        default=None,
        help="Fixed XGBoost max_depth value. Required when --model-family is XGBoost.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Random seed for train/val splits and PCA.",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        required=True,
        help="Validation split fraction within each environment.",
    )
    parser.add_argument(
        "--delta-threshold",
        type=float,
        required=True,
        help="Threshold for the delta deception-rate targets.",
    )
    parser.add_argument(
        "--root-batch-size",
        type=int,
        required=True,
        help="Attention reduction batch size over layer roots.",
    )
    parser.add_argument(
        "--decision-threshold-mode",
        required=True,
        help="Decision-threshold selection mode passed through to the experiment.",
    )
    parser.add_argument(
        "--model-selection-objective",
        required=True,
        choices=["mean_ood_auroc_oracle", "source_val_auroc"],
        help="Model-selection objective inside the experiment.",
    )
    parser.add_argument(
        "--top-features-to-show",
        type=int,
        required=True,
        help="How many top-coefficient features to export per winning panel.",
    )
    parser.add_argument(
        "--min-num-valid",
        type=int,
        default=None,
        help="Optional minimum num_valid filter applied to both the current and previous sentence rows.",
    )
    parser.add_argument(
        "--min-sentence-alpha-words",
        type=int,
        default=None,
        help="Optional minimum alphabetic word count required for a sentence to stay in the dataset.",
    )
    if hasattr(argparse, "BooleanOptionalAction"):
        parser.add_argument(
            "--exclude-multiline-sentences",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Whether to drop multiline sentences during metadata filtering.",
        )
    parser.add_argument(
        "--xgb-n-estimators",
        type=int,
        default=None,
        help="Optional XGBoost n_estimators override.",
    )
    parser.add_argument(
        "--xgb-learning-rate",
        type=float,
        default=None,
        help="Optional XGBoost learning_rate override.",
    )
    parser.add_argument(
        "--xgb-subsample",
        type=float,
        default=None,
        help="Optional XGBoost subsample override.",
    )
    parser.add_argument(
        "--xgb-colsample-bytree",
        type=float,
        default=None,
        help="Optional XGBoost colsample_bytree override.",
    )
    parser.add_argument(
        "--xgb-reg-lambda",
        type=float,
        default=None,
        help="Optional XGBoost reg_lambda override.",
    )
    parser.add_argument(
        "--xgb-min-child-weight",
        type=float,
        default=None,
        help="Optional XGBoost min_child_weight override.",
    )
    parser.add_argument(
        "--xgb-gamma",
        type=float,
        default=None,
        help="Optional XGBoost gamma override.",
    )
    parser.add_argument(
        "--xgb-n-jobs",
        type=int,
        default=None,
        help="Optional XGBoost n_jobs override.",
    )
    parser.add_argument(
        "--xgb-importance-type",
        default=None,
        help="Optional XGBoost importance_type override.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=None,
        help="Optional checkpoint frequency for intermediate OOD model-selection artifacts.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Rebuild cached reduced-attention parquet files under the output cache directory.",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Disable progress bars.",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Allow interactive plot display. By default the script uses Agg and only saves figures.",
    )
    return parser.parse_args()


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


def normalize_model_family(raw_value: str) -> str:
    value = str(raw_value).strip().lower()
    if value in {"logreg", "logistic", "logistic_regression", "lr"}:
        return "logreg"
    if value in {"xgboost", "xgb"}:
        return "xgboost"
    raise ValueError(f"Unsupported model family: {raw_value!r}")


def normalize_scenario_key(raw_value: str) -> str:
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
        raise ValueError(f"Unsupported scenario key: {raw_value!r}")
    return aliases[value]


def parse_scenarios(raw_value: str) -> list[str]:
    return [normalize_scenario_key(part.strip()) for part in str(raw_value).split(",") if part.strip()]


def resolve_output_root(
    *,
    model_dirname: str,
    model_family: str,
    scenarios: list[str],
    explicit_output_root: str | None,
    only_tfidf: bool,
) -> Path:
    if explicit_output_root:
        return Path(explicit_output_root)
    scenario_slug = "__".join(slugify(scenario_name) for scenario_name in scenarios)
    suffix = "__only_tfidf" if only_tfidf else ""
    return THIS_FILE.parent / (
        f"ood_modeling_outputs__{slugify(model_dirname)}__{slugify(model_family)}__{scenario_slug}"
        f"{suffix}"
    )


def build_runtime_config(
    *,
    args: argparse.Namespace,
    model_family: str,
    feature_sizes: list[int],
    scenarios: list[str],
    output_root: Path,
) -> dict[str, object]:
    runtime_config: dict[str, object] = {
        "model_dirname": str(args.model_dirname),
        "model_family": model_family,
        "source_path": str(OOD_MODELING_LIB_PATH),
        "repo_root": str(REPO_ROOT),
        "analysis_root": str(THIS_FILE.parent),
        "output_root": str(output_root),
        "feature_sizes": tuple(int(value) for value in feature_sizes),
        "scenarios": tuple(str(value) for value in scenarios),
        "seed": int(args.seed),
        "val_size": float(args.val_size),
        "delta_threshold": float(args.delta_threshold),
        "root_batch_size": int(args.root_batch_size),
        "decision_threshold_mode": str(args.decision_threshold_mode),
        "model_selection_objective": str(args.model_selection_objective),
        "top_features_to_show": int(args.top_features_to_show),
        "force_rebuild": bool(args.force_rebuild),
        "disable_tqdm": bool(args.disable_tqdm),
        "feature_space_mode": "only_tfidf" if args.only_tfidf else "all",
    }
    if args.dataset_root:
        runtime_config["dataset_root"] = str(Path(args.dataset_root))
    if args.structural_baseline_filename:
        runtime_config["structural_baseline_filename"] = str(args.structural_baseline_filename)
    if args.tfidf_cache_dirname:
        runtime_config["tfidf_cache_dirname"] = str(args.tfidf_cache_dirname)
    if args.tfidf_text_fields:
        runtime_config["tfidf_text_fields"] = tuple(
            part.strip() for part in str(args.tfidf_text_fields).split(",") if part.strip()
        )
    if args.min_num_valid is not None:
        runtime_config["min_num_valid"] = int(args.min_num_valid)
    if args.min_sentence_alpha_words is not None:
        runtime_config["min_sentence_alpha_words"] = int(args.min_sentence_alpha_words)
    if getattr(args, "exclude_multiline_sentences", None) is not None:
        runtime_config["exclude_multiline_sentences"] = bool(args.exclude_multiline_sentences)
    if args.logreg_c is not None:
        runtime_config["logreg_c"] = float(args.logreg_c)
    if args.xgb_max_depth is not None:
        runtime_config["xgb_max_depth"] = int(args.xgb_max_depth)
    if args.xgb_n_estimators is not None:
        runtime_config["xgb_n_estimators"] = int(args.xgb_n_estimators)
    if args.xgb_learning_rate is not None:
        runtime_config["xgb_learning_rate"] = float(args.xgb_learning_rate)
    if args.xgb_subsample is not None:
        runtime_config["xgb_subsample"] = float(args.xgb_subsample)
    if args.xgb_colsample_bytree is not None:
        runtime_config["xgb_colsample_bytree"] = float(args.xgb_colsample_bytree)
    if args.xgb_reg_lambda is not None:
        runtime_config["xgb_reg_lambda"] = float(args.xgb_reg_lambda)
    if args.xgb_min_child_weight is not None:
        runtime_config["xgb_min_child_weight"] = float(args.xgb_min_child_weight)
    if args.xgb_gamma is not None:
        runtime_config["xgb_gamma"] = float(args.xgb_gamma)
    if args.xgb_n_jobs is not None:
        runtime_config["xgb_n_jobs"] = int(args.xgb_n_jobs)
    if args.xgb_importance_type is not None:
        runtime_config["xgb_importance_type"] = str(args.xgb_importance_type)
    if args.checkpoint_every is not None:
        runtime_config["checkpoint_every"] = int(args.checkpoint_every)
    return runtime_config


def execute_ood_modeling_library(runtime_config: dict[str, object]) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("ood_modeling_lib_runtime", OOD_MODELING_LIB_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load OOD modeling library from {OOD_MODELING_LIB_PATH}")
    module = importlib.util.module_from_spec(spec)
    module.RUNTIME_CONFIG = runtime_config
    spec.loader.exec_module(module)
    return module


def main() -> None:
    args = parse_args()
    if not OOD_MODELING_LIB_PATH.exists():
        raise FileNotFoundError(f"OOD modeling library not found: {OOD_MODELING_LIB_PATH}")

    if not args.show_plots:
        import matplotlib

        matplotlib.use("Agg")

    model_family = normalize_model_family(args.model_family)
    feature_sizes = [int(part.strip()) for part in str(args.feature_sizes).split(",") if part.strip()]
    scenarios = parse_scenarios(args.scenarios)
    if model_family == "logreg" and args.logreg_c is None:
        raise ValueError("--logreg-c is required when --model-family is logistic regression.")
    if model_family == "xgboost" and args.xgb_max_depth is None:
        raise ValueError("--xgb-max-depth is required when --model-family is XGBoost.")
    output_root = resolve_output_root(
        model_dirname=str(args.model_dirname),
        model_family=model_family,
        scenarios=scenarios,
        explicit_output_root=args.output_root,
        only_tfidf=bool(args.only_tfidf),
    )
    runtime_config = build_runtime_config(
        args=args,
        model_family=model_family,
        feature_sizes=feature_sizes,
        scenarios=scenarios,
        output_root=output_root,
    )

    print("Running OOD modeling pipeline")
    print(f"Library: {OOD_MODELING_LIB_PATH}")
    print(f"Model: {args.model_dirname}")
    print(f"Model family: {model_family}")
    print(f"Scenarios: {scenarios}")
    print(f"Feature sizes: {feature_sizes}")
    if args.logreg_c is not None:
        print(f"Fixed logistic C: {float(args.logreg_c):g}")
    if args.xgb_max_depth is not None:
        print(f"Fixed XGBoost max_depth: {int(args.xgb_max_depth)}")
    if args.dataset_root:
        print(f"Dataset root: {Path(args.dataset_root)}")
    if args.structural_baseline_filename:
        print(f"Companion structural parquet filename: {args.structural_baseline_filename}")
    if args.tfidf_cache_dirname:
        print(f"TF-IDF cache dirname: {args.tfidf_cache_dirname}")
    if args.tfidf_text_fields:
        print(f"TF-IDF text fields: {args.tfidf_text_fields}")
    print(f"Feature space mode: {'only_tfidf' if args.only_tfidf else 'all'}")
    print(f"Output root: {output_root}")

    execute_ood_modeling_library(runtime_config)

    required_outputs = [
        output_root / "config.csv",
        output_root / "all_transfer_metrics.csv",
        output_root / "all_model_selection.csv",
        output_root / "all_coefficients.csv",
    ]
    missing_outputs = [str(path) for path in required_outputs if not path.exists()]
    if missing_outputs:
        raise FileNotFoundError(
            "Run finished without writing the expected aggregate outputs:\n"
            + "\n".join(missing_outputs)
        )
    print(f"Verified aggregate outputs under: {output_root}")


if __name__ == "__main__":
    main()
