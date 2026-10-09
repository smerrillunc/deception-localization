#!/usr/bin/env python3
"""Read localization files from a local copy of the dataset and write the two inputs of build_data.py.

    harvest.jsonl   one record per trace: probe curve (rate, interval, counts, sentence) + trace metadata
    detail.jsonl    full trace text, prompt, and a few sampled continuations around each trace's juncture

The traces are the ones already listed in data/index.json (a fixed, stratified sample), so a rebuild
shows the same traces as the previous build and only the underlying data changes.

    python3 harvest_local.py --root <dataset-root> [--index data/index.json] [--out .]

<dataset-root> holds <environment>/<model>/localization/*.json.gz, e.g. a local download of the Hub dataset.
Continuations shown per probe: up to 2 deceptive + the rest truthful, 4 in total, drawn with a fixed seed
from the evaluable ones, each cut to 420 characters. Probes with continuations: the juncture's later boundary
(or the sharpest single-sentence step when there is no juncture) and its neighbouring probes.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_data as bd   # reuse the exact curve / juncture definitions

TEXT_CAP = 420        # characters of each continuation kept
GENS_PER_PROBE = 4
DEC_PER_PROBE = 2


def load(path):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        raw = fh.read()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:                 # complete object followed by stray bytes
        return json.JSONDecoder().raw_decode(raw)[0]


def curve_of(hist):
    pts = []
    for h in sorted(hist, key=lambda h: h["sentence_end_idx"]):
        if h.get("deception_rate") is None or not (h.get("num_valid") or 0):
            continue
        pts.append({"i": h["sentence_end_idx"], "r": round(h["deception_rate"], 4),
                    "lo": None if h.get("ci_low") is None else round(h["ci_low"], 4),
                    "hi": None if h.get("ci_high") is None else round(h["ci_high"], 4),
                    "nv": h["num_valid"], "nt": h.get("num_truthful"),
                    "s": h.get("target_sentence_text") or h.get("sentence_text") or ""})
    return pts


def pick_gens(h, rng):
    gens = [g for g in h.get("generations", []) if g.get("is_truthful") is not None]
    dec = [g for g in gens if g["deceptive"]]
    tru = [g for g in gens if not g["deceptive"]]
    rng.shuffle(dec), rng.shuffle(tru)
    chosen = dec[:DEC_PER_PROBE]
    chosen += tru[:GENS_PER_PROBE - len(chosen)]
    chosen += dec[len(chosen):GENS_PER_PROBE]          # not enough truthful ones: top up with deceptive
    out = []
    for g in chosen:
        t = g.get("gen_text") or ""
        out.append({"t": t if len(t) <= TEXT_CAP else t[:TEXT_CAP] + "…", "d": bool(g["deceptive"]), "e": str(g.get("evaluation"))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--index", default=os.path.join(bd.HERE, "data", "index.json"))
    ap.add_argument("--out", default=bd.HERE)
    a = ap.parse_args()
    paths = [r["path"] for r in json.load(open(a.index))["rows"]]
    n = 0
    with open(os.path.join(a.out, "harvest.jsonl"), "w") as hv, open(os.path.join(a.out, "detail.jsonl"), "w") as dt:
        for path in paths:
            env, model = path.split("/")[:2]
            j = load(os.path.join(a.root, path))
            hist = sorted(j["history"], key=lambda h: h["sentence_end_idx"])
            fs = j.get("full_score") or {}
            rec = {"env": env, "model": model, "example_id": j["example_id"], "path": path, "curve": curve_of(hist),
                   "n_sent_total": len(hist), "full_rate": fs.get("deception_rate"), "raw_len": len(j.get("raw_text") or "")}
            hv.write(json.dumps(rec) + "\n")
            m = bd.trace_metrics(rec)
            row = m["row"] if m else None
            centre = None if row is None else (row["j"] + 1 if row["j"] is not None else row["jump_at"])
            probed = [p["i"] for p in rec["curve"]]
            show = set()
            if centre in probed:
                k = probed.index(centre)
                lo = min(max(k - 1, 0), max(len(probed) - 3, 0))
                show = set(probed[lo:lo + 3])
            probes = []
            for h in hist:
                if h["sentence_end_idx"] not in probed:
                    continue
                p = {"i": h["sentence_end_idx"], "span": h.get("char_span"), "s": h.get("sentence_text") or ""}
                if h["sentence_end_idx"] in show:
                    p["g"] = pick_gens(h, random.Random(f"{path}-{h['sentence_end_idx']}"))
                probes.append(p)
            dt.write(json.dumps({"env": env, "model": model, "path": path, "raw": j.get("raw_text") or "",
                                 "prompt": j.get("prompt") or "", "ctx": j.get("eval_context"), "probes": probes}) + "\n")
            n += 1
    print(f"harvested {n} traces -> {a.out}/harvest.jsonl, detail.jsonl")


if __name__ == "__main__":
    main()
