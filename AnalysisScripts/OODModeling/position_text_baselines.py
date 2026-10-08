#!/usr/bin/env python3
"""Position / length / sentence-embedding baselines for commitment-juncture prediction.

A row is a probed reasoning prefix compared with the previous probed prefix of the same trace. The label is
    deceptive commitment:  delta_deception_rate >  tau
    honest commitment:     delta_deception_rate < -tau
Each baseline gets one cheap per-row feature (no model internals) and an XGBoost classifier:

    normalized_position   (probe index + 1) / number of probes in the trace
    prefix_word_count     words in the reasoning prefix
    reasoning_word_count  words in the longest probed prefix of the trace
    sentence_embedding    all-MiniLM-L6-v2 embedding of the candidate sentence (384-d)

Inputs are the files written by text_structural_feature_extractor.py:
    <dataset-root>/<environment>/<model>/commitment_text_structural_baselines.parquet

Protocols (train on training traces, evaluate on the held-out 20% validation traces):
    leave_one_env_out   train on four environments, test on the fifth
    single_source       train on one environment, test on each of the other four

Example:
    python AnalysisScripts/OODModeling/position_text_baselines.py --dataset-root DatasetMain \
        --models DeepSeek-R1-Distill-Qwen-7B --protocol single_source --output baselines.csv
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier

ENVIRONMENTS = ["advisor_audit", "bs", "car_sales", "gridworld", "interview"]
MODELS = ["DeepSeek-R1-Distill-Llama-8B", "DeepSeek-R1-Distill-Qwen-7B", "DeepSeek-R1-Distill-Qwen-14B", "gpt-oss-20b"]
STRUCTURAL_FILE = "commitment_text_structural_baselines.parquet"
COLUMNS = ["example_id", "sentence_idx", "last_sentence_text", "prefix_word_count", "normalized_position",
           "delta_deception_rate", "num_valid"]
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMB = [f"emb_{i:03d}" for i in range(384)]
FEATURES = {
    "baseline_normalized_position": ["normalized_position"],
    "baseline_prefix_word_count": ["prefix_word_count"],
    "baseline_reasoning_word_count": ["reasoning_word_count"],
    "baseline_sentence_embedding": EMB,
}
MIN_VALID = 11          # sampled continuations required at both prefixes of a row
VAL_FRACTION, SPLIT_SEED = 0.2, 42
RECALLS = (0.5, 0.8, 0.9, 0.95)
XGB_PARAMS = dict(objective="binary:logistic", n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
                  colsample_bytree=0.8, tree_method="hist", eval_metric="logloss", random_state=42)


def load_rows(root: Path, env: str, model: str, tau: float) -> pd.DataFrame:
    """Rows of one (environment, model) bundle: filtered, split by trace, labelled."""
    d = pd.read_parquet(root / env / model / STRUCTURAL_FILE, columns=COLUMNS)
    d = d.sort_values(["example_id", "sentence_idx"]).reset_index(drop=True)
    by_trace = d.groupby("example_id")
    d["prev_num_valid"] = by_trace.num_valid.shift(1)
    d["reasoning_word_count"] = by_trace.prefix_word_count.transform("max")
    text = d.last_sentence_text.fillna("").str.strip()
    keep = (d.delta_deception_rate.notna() & (d.num_valid >= MIN_VALID) & (d.prev_num_valid >= MIN_VALID)
            & text.ne("") & ~text.str.contains("\n") & text.map(lambda s: len(re.findall(r"[A-Za-z]+", s)) >= 4))
    ids = np.array(sorted(d.example_id.unique()))
    val = set(ids[np.random.RandomState(SPLIT_SEED).permutation(len(ids))[: int(round(VAL_FRACTION * len(ids)))]])
    delta = d.delta_deception_rate.astype("float32")
    d["split"] = np.where(d.example_id.isin(val), "val", "train")
    d["deceptive"] = (delta > np.float32(tau)).astype(int)
    d["honest"] = (delta < np.float32(-tau)).astype(int)
    d["env"] = env
    return d[keep].reset_index(drop=True)


def embed(texts: list[str], batch_size: int = 256) -> np.ndarray:
    """L2-normalised mean-pooled MiniLM sentence embeddings (same pooling as sentence-transformers)."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok, net = AutoTokenizer.from_pretrained(EMBEDDING_MODEL), AutoModel.from_pretrained(EMBEDDING_MODEL).to(device).eval()
    order = np.argsort([len(t) for t in texts], kind="stable")      # similar lengths per batch
    out = np.zeros((len(texts), 384), dtype=np.float32)
    for i in range(0, len(texts), batch_size):
        idx = order[i:i + batch_size]
        enc = tok([texts[j] for j in idx], padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        with torch.no_grad():
            hidden = net(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        vec = torch.nn.functional.normalize((hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9), dim=1)
        out[idx] = vec.cpu().numpy()
    return out


def fpr_at_recall(y: np.ndarray, score: np.ndarray, recall: float) -> float:
    pos, neg = score[y == 1], score[y == 0]
    return float("nan") if not len(pos) or not len(neg) else float((neg >= np.quantile(pos, 1 - recall)).mean())


def fit_eval(train: pd.DataFrame, test: pd.DataFrame, cols: list[str], target: str, n_jobs: int) -> dict | None:
    y = train[target].to_numpy()
    if y.sum() < 5:
        return None
    clf = make_pipeline(SimpleImputer(strategy="median"),
                        XGBClassifier(scale_pos_weight=(len(y) - y.sum()) / y.sum(), n_jobs=n_jobs, **XGB_PARAMS)).fit(train[cols], y)
    score, yt = clf.predict_proba(test[cols])[:, 1], test[target].to_numpy()
    ok = yt.sum() > 0
    return dict(n=len(yt), n_pos=int(yt.sum()),
                auroc=roc_auc_score(yt, score) if ok else np.nan,
                average_precision=average_precision_score(yt, score) if ok else np.nan,
                **{f"fpr_at_recall_{r}": fpr_at_recall(yt, score, r) for r in RECALLS})


def splits(protocol: str, envs: list[str]):
    """Yield (train environments, test environment)."""
    for held_out in envs:
        if protocol == "leave_one_env_out":
            yield [e for e in envs if e != held_out], held_out
        else:
            for source in envs:
                if source != held_out:
                    yield [source], held_out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", required=True, help="Directory holding <environment>/<model>/" + STRUCTURAL_FILE)
    ap.add_argument("--output", required=True, help="CSV with one row per (model, train envs, test env, feature space, target).")
    ap.add_argument("--protocol", choices=["leave_one_env_out", "single_source"], default="leave_one_env_out")
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--environments", nargs="+", default=ENVIRONMENTS)
    ap.add_argument("--feature-spaces", nargs="+", default=list(FEATURES), choices=list(FEATURES))
    ap.add_argument("--targets", nargs="+", default=["deceptive", "honest"], choices=["deceptive", "honest"])
    ap.add_argument("--tau", type=float, default=0.3, help="Commitment threshold on the change in deception rate.")
    ap.add_argument("--n-jobs", type=int, default=16, help="XGBoost threads.")
    args = ap.parse_args()

    root, results = Path(args.dataset_root), []
    for model in args.models:
        data = {env: load_rows(root, env, model, args.tau) for env in args.environments}
        if "baseline_sentence_embedding" in args.feature_spaces:
            texts = sorted({t for d in data.values() for t in d.last_sentence_text})
            emb = pd.DataFrame(embed(texts), index=texts, columns=EMB)
            data = {env: pd.concat([d, emb.loc[d.last_sentence_text].reset_index(drop=True)], axis=1) for env, d in data.items()}
        for train_envs, test_env in splits(args.protocol, args.environments):
            train = pd.concat([data[e][data[e].split == "train"] for e in train_envs])
            test = data[test_env][data[test_env].split == "val"]
            for space in args.feature_spaces:
                for target in args.targets:
                    r = fit_eval(train, test, FEATURES[space], target, args.n_jobs)
                    if r:
                        results.append(dict(model=model, protocol=args.protocol, train_envs="+".join(train_envs),
                                            test_env=test_env, feature_space=space, target=target, tau=args.tau, **r))
        print(f"done: {model}", flush=True)

    table = pd.DataFrame(results)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(table.groupby(["target", "feature_space", "model"]).auroc.mean().groupby(["target", "feature_space"]).mean().round(3))


if __name__ == "__main__":
    main()
