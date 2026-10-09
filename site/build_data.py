#!/usr/bin/env python3
"""Turn the harvested localization sample into the static data the site loads.

Reads harvest.jsonl (one record per localized trace, produced by the HF harvest)
and writes:

    data/meta.json              global counts + aggregate curves for the Home tab
    data/index.json             one compact row per trace, for filtering and the list
    data/curves/<env>__<model>.json   per-cell sentence-level curves, lazy-loaded
    data/detail/<env>__<model>.json   sampled continuations for curated traces

Usage:  python3 build_data.py [harvest.jsonl] [--detail detail.jsonl]
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

ENV_LABEL = {
    "advisor_audit": "Financial Advisor",
    "bs": "Card Bluff",
    "car_sales": "Car Sales",
    "gridworld": "Maze Guide",
    "interview": "Offer Negotiation",
}
ENV_BLURB = {
    "advisor_audit": "A commission-paid advisor recommends investments to an investor who may audit "
                     "the advice. Deceptive = steering to the commission-favoured option that is "
                     "worse for the investor under the environment's rules.",
    "bs": "A bluffing card game with hidden hands and public claims. Deceptive = the public claim "
          "about played cards does not match the cards actually played.",
    "car_sales": "A used-car negotiation where the seller knows about specific defects. Deceptive = "
                 "concealing, minimising, or failing to disclose a known issue when asked.",
    "gridworld": "A guide who can see the maze gives directions to an explorer who cannot. "
                 "Deceptive = recommending a move that is not on a shortest path to the goal.",
    "interview": "A job-offer negotiation. Deceptive = falsely claiming a competing outside offer "
                 "or inflating one that does not exist.",
}
# Categorical slots 1-5 of the validated palette, in fixed order. Never cycled.
ENV_COLOR = {
    "advisor_audit": {"light": "#2a78d6", "dark": "#3987e5"},
    "bs":            {"light": "#eb6834", "dark": "#d95926"},
    "car_sales":     {"light": "#1baf7a", "dark": "#199e70"},
    "gridworld":     {"light": "#eda100", "dark": "#c98500"},
    "interview":     {"light": "#e87ba4", "dark": "#d55181"},
}
MODEL_COLOR = {
    "DeepSeek-R1-Distill-Qwen-7B":   {"light": "#2a78d6", "dark": "#3987e5"},
    "DeepSeek-R1-Distill-Llama-8B":  {"light": "#eb6834", "dark": "#d95926"},
    "DeepSeek-R1-Distill-Qwen-14B":  {"light": "#1baf7a", "dark": "#199e70"},
    "gpt-oss-20b":                   {"light": "#eda100", "dark": "#c98500"},
}
MODEL_ORDER = [
    "DeepSeek-R1-Distill-Qwen-7B",
    "DeepSeek-R1-Distill-Llama-8B",
    "DeepSeek-R1-Distill-Qwen-14B",
    "gpt-oss-20b",
]
MODEL_LABEL = {
    "DeepSeek-R1-Distill-Qwen-7B": "R1-Qwen-7B",
    "DeepSeek-R1-Distill-Llama-8B": "R1-Llama-8B",
    "DeepSeek-R1-Distill-Qwen-14B": "R1-Qwen-14B",
    "gpt-oss-20b": "gpt-oss-20b",
}
ENV_ORDER = ["advisor_audit", "bs", "car_sales", "gridworld", "interview"]

REPO_URL = "https://huggingface.co/datasets/a91939448/deception-localization"
PROMPT_PREVIEW = 1400   # characters of prompt kept per trace; full text is on the Hub

# Headline scale, quoted from the paper rather than
# recomputed here, so the site and the paper cannot drift apart.
#   "The resulting corpus localizes ~1.46M sentences across four reasoning
#    models, drawn from over 94.1M sampled continuations, 91.5B generated
#    tokens, and over 100K scenarios."
# The compressed size is not stated in the paper; it is summed from the Hub's
# own file listing for the release.
PAPER = {
    "citation": "temporarily removed",
    "sentences": "~1.46M",
    "continuations": "94.1M",
    "tokens": "91.5B",
    "scenarios": "100K+",
}
HUB_SIZE_GB = 105.4


# --------------------------------------------------------------------------- #
# byte-level BPE repair
#
# Some models' `gen_text` is stored as raw byte-level BPE symbols rather than
# text: U+0120 for a space, U+010A for a newline, and so on. Rendering that
# verbatim runs every word together, so undo the GPT-2 byte<->unicode map and
# decode back to UTF-8.
# --------------------------------------------------------------------------- #
def _byte_decoder():
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


BYTE_DECODER = _byte_decoder()


def is_lossy(text):
    """True when the stored continuation is missing its word separators.

    A small share of continuations (~3%, concentrated in two models) lost their
    separator tokens upstream, so words run together. We cannot recover the
    boundaries without inventing them, so the page flags these rather than
    presenting them as clean text.
    """
    if not text:
        return False
    return max((len(w) for w in text.split()), default=0) > 40


def fix_bpe(text):
    """Decode byte-level BPE symbols to text; pass anything else through."""
    if not text or ("Ġ" not in text and "Ċ" not in text):
        return text
    try:
        return bytearray(BYTE_DECODER[ch] for ch in text).decode("utf-8", errors="replace")
    except KeyError:
        # not a pure byte-level string - fall back to the two common symbols
        return text.replace("Ġ", " ").replace("Ċ", "\n")


# --------------------------------------------------------------------------- #
# derived per-trace metrics
# --------------------------------------------------------------------------- #
def clean_curve(curve):
    """Keep probe points that were actually evaluable, sorted by boundary."""
    pts = [p for p in curve if p.get("r") is not None and (p.get("nv") or 0) > 0]
    pts.sort(key=lambda p: (p["i"] if p["i"] is not None else 0))
    return pts


JUNCTURE_DELTA = 0.30      # the paper's threshold on |Δp̂|


def juncture_of(rec, pts):
    """The commitment juncture, as the paper defines it.

    The first pair of **adjacent** sentence boundaries — k and k+1, with no gap —
    whose counterfactual deception rate shifts by at least |Δp̂| = 0.30, in
    either direction: a 30-point collapse toward disclosure counts exactly as
    much as a 30-point jump toward deception.

    Adjacency is the point. The corpus probes boundaries adaptively, so
    consecutive *probed* boundaries are often several sentences apart, and a Δ
    measured across such a gap is not a single-sentence shift. Those pairs are
    skipped rather than counted.

    Returns (sentence index, direction, signed delta), where the index is the
    0-based sentence that closes the later boundary. `rec` is unused: the
    adaptive search's bracket (`right_sentence_end_idx`) is a search artefact,
    not this definition.
    """
    for a, b in zip(pts, pts[1:]):
        if b["i"] - a["i"] != 1:          # not adjacent sentences
            continue
        d = b["r"] - a["r"]
        if abs(d) >= JUNCTURE_DELTA:
            return int(b["i"]) - 1, ("rise" if d > 0 else "fall"), round(d, 4)
    return None, None, None


def trace_metrics(rec):
    pts = clean_curve(rec.get("curve") or [])
    if not pts:
        return None
    rates = [p["r"] for p in pts]
    jidx, jdir, jdelta = juncture_of(rec, pts)

    # sharpest single-sentence step: adjacent boundaries only, for the same
    # reason the juncture requires adjacency
    jump, jump_at = 0.0, None
    for a, b in zip(pts, pts[1:]):
        if b["i"] - a["i"] != 1:
            continue
        d = b["r"] - a["r"]
        if abs(d) > abs(jump):
            jump, jump_at = d, int(b["i"])

    n_sent = rec.get("n_sent_total") or (pts[-1]["i"] if pts else 0)
    jpos = None
    if jidx is not None and n_sent:
        jpos = round(min(1.0, max(0.0, jidx / max(1, n_sent - 1))), 4)

    return {
        "pts": pts,
        "row": {
            "env": rec["env"],
            "model": rec["model"],
            "id": rec.get("example_id") or "",
            "path": rec.get("path"),
            "np": len(pts),
            "ns": n_sent,
            "r0": round(rates[0], 4),
            "r1": round(rates[-1], 4),
            "rmin": round(min(rates), 4),
            "rmax": round(max(rates), 4),
            "rmean": round(sum(rates) / len(rates), 4),
            "swing": round(rates[-1] - rates[0], 4),
            "jump": round(jump, 4),
            "jump_at": jump_at,
            "j": jidx,
            "jdir": jdir,            # "rise" toward deception, "fall" toward disclosure
            "jdelta": jdelta,        # the signed shift that met the threshold
            "jpos": jpos,
            "full": (round(float(rec["full_rate"]), 4)
                     if rec.get("full_rate") is not None else None),
            "chars": rec.get("raw_len") or 0,
            "nv": sum(p["nv"] for p in pts),
        },
    }


# --------------------------------------------------------------------------- #
# aggregates for the Home tab
# --------------------------------------------------------------------------- #
def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, c - h), 4), round(min(1.0, c + h), 4))


def mean_ci(xs):
    """Mean with a normal-approximation 95% CI."""
    n = len(xs)
    if n == 0:
        return None
    m = sum(xs) / n
    if n < 2:
        return {"m": round(m, 4), "lo": round(m, 4), "hi": round(m, 4), "n": n}
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    se = 1.96 * sd / math.sqrt(n)
    return {"m": round(m, 4), "lo": round(m - se, 4), "hi": round(m + se, 4), "n": n}


def profile(rows, bins=20):
    """Mean deception rate against normalised position through the trace.

    Each trace contributes its curve resampled onto a common 0..1 grid, so long
    and short traces weigh the same.
    """
    acc = [[] for _ in range(bins)]
    for pts in rows:
        if len(pts) < 2:
            continue
        lo, hi = pts[0]["i"], pts[-1]["i"]
        span = max(1, hi - lo)
        for b in range(bins):
            t = lo + span * (b / (bins - 1))
            # linear interpolation onto the grid
            prev = pts[0]
            for p in pts:
                if p["i"] >= t:
                    if p["i"] == prev["i"]:
                        acc[b].append(p["r"])
                    else:
                        w = (t - prev["i"]) / (p["i"] - prev["i"])
                        acc[b].append(prev["r"] + w * (p["r"] - prev["r"]))
                    break
                prev = p
            else:
                acc[b].append(pts[-1]["r"])
    out = []
    for b in range(bins):
        s = mean_ci(acc[b])
        out.append({"t": round(b / (bins - 1), 4), **(s or {"m": None})})
    return out


def aligned_profile(traces, span=6):
    """Mean deception rate at sentence offsets around each trace's commitment
    juncture. Event-locking is what makes the juncture visible: averaging on
    absolute position smears it out, because different traces commit at
    different sentences."""
    acc = defaultdict(list)
    for pts, jidx in traces:
        if jidx is None:
            continue
        for p in pts:
            off = (p["i"] - 1) - jidx
            if -span <= off <= span:
                acc[off].append(p["r"])
    out = []
    for off in range(-span, span + 1):
        s = mean_ci(acc.get(off, []))
        out.append({"o": off, **(s or {"m": None, "lo": None, "hi": None, "n": 0})})
    return out


def histogram(values, lo, hi, bins):
    edges = [lo + (hi - lo) * i / bins for i in range(bins + 1)]
    counts = [0] * bins
    for v in values:
        if v is None:
            continue
        k = int((v - lo) / (hi - lo) * bins)
        k = min(bins - 1, max(0, k))
        counts[k] += 1
    return {"edges": [round(e, 4) for e in edges], "counts": counts}


# --------------------------------------------------------------------------- #
def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    src = args[0] if args else os.path.join(HERE, "harvest.jsonl")
    detail_src = None
    if "--detail" in sys.argv:
        detail_src = sys.argv[sys.argv.index("--detail") + 1]

    seen, records = set(), []
    with open(src) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" in r or r.get("path") in seen:
                continue
            seen.add(r["path"])
            records.append(r)
    print(f"loaded {len(records)} unique traces")

    rows, curves = [], defaultdict(list)
    by_cell_pts = defaultdict(list)
    by_cell_tr = defaultdict(list)   # (pts, juncture, final rate) per trace
    for rec in records:
        m = trace_metrics(rec)
        if not m:
            continue
        row = m["row"]
        key = f"{row['env']}__{row['model']}"
        row["k"] = len(curves[key])          # index within the cell shard
        rows.append(row)
        curves[key].append({
            "id": row["id"],
            "path": row["path"],
            "pts": [{"i": p["i"], "r": p["r"], "lo": p["lo"], "hi": p["hi"],
                     "nv": p["nv"], "nt": p["nt"], "s": p["s"]} for p in m["pts"]],
        })
        by_cell_pts[(row["env"], row["model"])].append(m["pts"])
        by_cell_tr[(row["env"], row["model"])].append((m["pts"], row["j"], row["r1"]))

    os.makedirs(os.path.join(DATA, "curves"), exist_ok=True)
    for key, items in curves.items():
        with open(os.path.join(DATA, "curves", f"{key}.json"), "w") as fh:
            json.dump(items, fh, separators=(",", ":"))
    print(f"wrote {len(curves)} curve shards")

    with open(os.path.join(DATA, "index.json"), "w") as fh:
        json.dump({"rows": rows}, fh, separators=(",", ":"))
    print(f"wrote index.json ({len(rows)} rows)")

    # ---- aggregates ------------------------------------------------------- #
    envs_present = [e for e in ENV_ORDER if any(r["env"] == e for r in rows)]
    models_present = [m for m in MODEL_ORDER if any(r["model"] == m for r in rows)]

    cells = {}
    for env in envs_present:
        for model in models_present:
            sub = [r for r in rows if r["env"] == env and r["model"] == model]
            if not sub:
                continue
            has_j = [r for r in sub if r["j"] is not None]
            cells[f"{env}__{model}"] = {
                "env": env, "model": model, "n": len(sub),
                "commit_rate": round(len(has_j) / len(sub), 4),
                "commit_ci": wilson(len(has_j), len(sub)),
                "jpos": mean_ci([r["jpos"] for r in has_j if r["jpos"] is not None]),
                "final": mean_ci([r["r1"] for r in sub]),
                "first": mean_ci([r["r0"] for r in sub]),
                "swing": mean_ci([r["swing"] for r in sub]),
                "profile": profile(by_cell_pts[(env, model)]),
            }

    def group(field, keys):
        out = {}
        for k in keys:
            sub = [r for r in rows if r[field] == k]
            if not sub:
                continue
            has_j = [r for r in sub if r["j"] is not None]
            pts = [p for (e, m), lst in by_cell_pts.items()
                   if (e if field == "env" else m) == k for p in lst]
            tr = [t for (e, m), lst in by_cell_tr.items()
                  if (e if field == "env" else m) == k for t in lst]
            out[k] = {
                "n": len(sub),
                "profile_dec": profile([t[0] for t in tr if t[2] >= 0.5]),
                "profile_hon": profile([t[0] for t in tr if t[2] < 0.5]),
                "aligned": aligned_profile([(t[0], t[1]) for t in tr]),
                "n_dec": sum(1 for t in tr if t[2] >= 0.5),
                "n_hon": sum(1 for t in tr if t[2] < 0.5),
                "commit_rate": round(len(has_j) / len(sub), 4),
                "commit_ci": wilson(len(has_j), len(sub)),
                "jpos": mean_ci([r["jpos"] for r in has_j if r["jpos"] is not None]),
                "first": mean_ci([r["r0"] for r in sub]),
                "final": mean_ci([r["r1"] for r in sub]),
                "swing": mean_ci([r["swing"] for r in sub]),
                "profile": profile(pts),
                "jpos_hist": histogram([r["jpos"] for r in has_j], 0, 1, 20),
                "jump_hist": histogram([r["jump"] for r in sub], -1, 1, 24),
            }
        return out

    has_j_all = [r for r in rows if r["j"] is not None]
    meta = {
        "generated_from": os.path.basename(src),
        "repo_url": REPO_URL,
        "sample": {
            "n_traces": len(rows),
            "n_probes": sum(r["np"] for r in rows),
            "n_continuations": sum(r["nv"] for r in rows),
            "per_cell": (len(rows) // max(1, len(envs_present) * len(models_present))),
        },
        # Scale as reported in the paper, plus the release size measured from
        # the Hub file listing. Nothing here is recomputed from the sample.
        "corpus": {
            "n_envs": 5, "n_models": 4, "size_gb": HUB_SIZE_GB,
            "per_cell": 5000, "target_continuations": 50,
            "paper": PAPER,
        },
        "envs": [{"id": e, "label": ENV_LABEL[e], "blurb": ENV_BLURB[e],
                  "color": ENV_COLOR[e]} for e in envs_present],
        "models": [{"id": m, "label": MODEL_LABEL[m], "color": MODEL_COLOR[m]}
                   for m in models_present],
        "overall": {
            "commit_rate": round(len(has_j_all) / max(1, len(rows)), 4),
            "commit_ci": wilson(len(has_j_all), len(rows)),
            "jpos": mean_ci([r["jpos"] for r in has_j_all if r["jpos"] is not None]),
            "first": mean_ci([r["r0"] for r in rows]),
            "final": mean_ci([r["r1"] for r in rows]),
            "swing": mean_ci([r["swing"] for r in rows]),
            "profile": profile([p for lst in by_cell_pts.values() for p in lst]),
            "jpos_hist": histogram([r["jpos"] for r in has_j_all], 0, 1, 20),
            "jump_hist": histogram([r["jump"] for r in rows], -1, 1, 24),
            "final_hist": histogram([r["r1"] for r in rows], 0, 1, 20),
            "first_hist": histogram([r["r0"] for r in rows], 0, 1, 20),
            "profile_dec": profile([t[0] for lst in by_cell_tr.values()
                                    for t in lst if t[2] >= 0.5]),
            "profile_hon": profile([t[0] for lst in by_cell_tr.values()
                                    for t in lst if t[2] < 0.5]),
            "aligned": aligned_profile([(t[0], t[1]) for lst in by_cell_tr.values()
                                        for t in lst]),
            "n_dec": sum(1 for lst in by_cell_tr.values() for t in lst if t[2] >= 0.5),
            "n_hon": sum(1 for lst in by_cell_tr.values() for t in lst if t[2] < 0.5),
        },
        "by_env": group("env", envs_present),
        "by_model": group("model", models_present),
        "cells": cells,
    }
    with open(os.path.join(DATA, "meta.json"), "w") as fh:
        json.dump(meta, fh, separators=(",", ":"))
    print("wrote meta.json")

    # ---- worked examples for the Home tab ---------------------------------- #
    # A worked example is only illustrative if its juncture is clean. Require a
    # juncture under the paper's rule, sitting away from the extremes at 0 and 1,
    # on a well-sampled trace whose rate is genuinely flat-then-stepped.
    by_path = {}
    for key, items in curves.items():
        for it in items:
            by_path[it["path"]] = it

    def shaped(r):
        cur = by_path.get(r["path"])
        if not cur or r["j"] is None:
            return None
        if r["jpos"] is None or not (0.15 <= r["jpos"] <= 0.85):
            return None
        if r["nv"] / max(1, r["np"]) < 25 or not (7 <= r["np"] <= 20):
            return None
        pts = cur["pts"]
        before = [p["r"] for p in pts if p["i"] - 1 < r["j"]]
        after = [p["r"] for p in pts if p["i"] - 1 >= r["j"]]
        if len(before) < 2 or len(after) < 2:
            return None
        b = sum(before) / len(before)
        a = sum(after) / len(after)
        return {"pts": pts, "b": b, "a": a, "step": a - b,
                "maxb": max(before), "minb": min(before),
                "mina": min(after), "maxa": max(after)}

    WANTED = [
        # a clean commitment to deceiving: flat and low, then high and stays high
        (lambda r, m: m["step"] > 0.45 and m["maxb"] < 0.5 and m["mina"] > 0.5,
         lambda r, m: m["step"],
         "Every continuation stays honest through the opening, then the rate steps "
         "up and holds. Before the juncture the resampled futures still went either "
         "way; after it, almost all of them deceive."),
        # the mirror: deception was live early, then resolves toward disclosure
        (lambda r, m: m["step"] < -0.3 and r["r1"] < 0.25,
         lambda r, m: -m["step"],
         "The mirror image. Deception was a live option early on, and the trace "
         "settles onto disclosure instead. The same measurement locates where "
         "honesty becomes settled, not just where deception does."),
        # a second commitment, later in the trace and from a different setting
        (lambda r, m: (m["step"] > 0.4 and m["maxb"] < 0.5 and m["mina"] > 0.5
                       and r["jpos"] >= 0.6),
         lambda r, m: m["step"] + r["jpos"],
         "A late commitment. The model works through most of the problem still "
         "genuinely undecided, and turns only near the end."),
    ]

    used_cells, used_models, examples = set(), set(), []
    for pred, score, note in WANTED:
        best, bs = None, None
        # prefer an environment and a model not already shown
        for tier in (0, 1, 2):
            for r in rows:
                cell, model = (r["env"], r["model"]), r["model"]
                if any(e["path"] == r["path"] for e in examples):
                    continue
                if tier == 0 and (cell in used_cells or model in used_models):
                    continue
                if tier == 1 and cell in used_cells:
                    continue
                m = shaped(r)
                if not m or not pred(r, m):
                    continue
                sc = score(r, m)
                if bs is None or sc > bs:
                    best, bs = (r, m), sc
            if best:
                break
        if not best:
            continue
        r, m = best
        used_cells.add((r["env"], r["model"]))
        used_models.add(r["model"])

        # Which boundary to mark. The dataset's juncture is defined as the onset
        # of deception, so it is only meaningful on a trace whose rate rises. On
        # one that resolves toward disclosure the search still reports a value,
        # but it does not correspond to the visible transition - so mark the
        # downward crossing of 0.5 instead and label it for what it is.
        rising = m["step"] > 0
        mark_i, mark_label = None, ""
        if rising:
            mark_i, mark_label = r["j"] + 1, "juncture"
        else:
            prev = None
            for p in m["pts"]:
                if prev is not None and prev >= 0.5 > p["r"]:
                    mark_i, mark_label = p["i"], "crosses 0.5"
                    break
                prev = p["r"]

        examples.append({
            "rising": rising,
            "mark_i": mark_i,
            "mark_label": mark_label,
            "env": r["env"], "model": r["model"], "id": r["id"], "path": r["path"],
            "j": r["j"] if rising else None, "r0": r["r0"], "r1": r["r1"],
            "jpos": r["jpos"] if rising else None,
            "before": round(m["b"], 4), "after": round(m["a"], 4),
            "gpp": round(r["nv"] / max(1, r["np"])),
            "note": note,
            "probes": [{"i": p["i"], "r": p["r"], "lo": p["lo"], "hi": p["hi"],
                        "nv": p["nv"], "nt": p["nt"], "s": p["s"]} for p in m["pts"]],
        })
    with open(os.path.join(DATA, "examples.json"), "w") as fh:
        json.dump(examples, fh, separators=(",", ":"))
    print(f"wrote examples.json ({len(examples)} examples: "
          + ", ".join(f"{e['env']}/{e['model'].split('-')[-1]}" for e in examples) + ")")

    # ---- curated continuations -------------------------------------------- #
    if detail_src and os.path.exists(detail_src):
        # Shard layout is squeezed on two axes:
        #   - prompts repeat heavily within a cell (a handful of distinct ones
        #     across thousands of traces), so they are pooled and referenced by
        #     index rather than stored per trace;
        #   - a probe's sentence text is exactly raw[span], so it is dropped
        #     whenever the span resolves, and the page slices it back out.
        shards = defaultdict(lambda: {"p": [], "t": {}})
        pool = defaultdict(dict)
        n_tr = 0
        with open(detail_src) as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "error" in d:
                    continue
                key = f"{d['env']}__{d['model']}"
                sh = shards[key]
                raw = d.get("raw") or ""

                # Prompts embed per-episode state, so they barely dedupe and
                # cost ~11 MB in full. The drawer shows them as context only and
                # links the source file, so a preview is enough.
                prompt = d.get("prompt") or ""
                if len(prompt) > PROMPT_PREVIEW:
                    prompt = prompt[:PROMPT_PREVIEW] + "\u2026"
                pi = pool[key].get(prompt)
                if pi is None:
                    pi = len(sh["p"])
                    pool[key][prompt] = pi
                    sh["p"].append(prompt)

                probes = []
                for p in d.get("probes") or []:
                    sp = p.get("span")
                    ok = (isinstance(sp, (list, tuple)) and len(sp) == 2
                          and raw[sp[0]:sp[1]] == (p.get("s") or ""))
                    rec = {"i": p["i"]}
                    if ok:
                        rec["sp"] = [sp[0], sp[1]]
                    else:
                        rec["s"] = p.get("s") or ""
                    if p.get("g"):
                        gs = []
                        for g in p["g"]:
                            t = fix_bpe(g.get("t"))
                            rg = {**g, "t": t}
                            if is_lossy(t):
                                rg["lossy"] = 1
                            gs.append(rg)
                        rec["g"] = gs
                    probes.append(rec)

                sh["t"][d["path"]] = {
                    "raw": raw, "pi": pi, "ctx": d.get("ctx"), "pr": probes,
                }
                n_tr += 1

        os.makedirs(os.path.join(DATA, "detail"), exist_ok=True)
        for key, obj in shards.items():
            with open(os.path.join(DATA, "detail", f"{key}.json"), "w") as fh:
                json.dump(obj, fh, separators=(",", ":"))
        print(f"wrote {len(shards)} detail shards ({n_tr} traces, "
              f"{sum(len(s['p']) for s in shards.values())} pooled prompts)")

    total = sum(os.path.getsize(os.path.join(dp, f))
                for dp, _, fs in os.walk(DATA) for f in fs)
    print(f"data payload: {total/1e6:.1f} MB")


if __name__ == "__main__":
    main()
