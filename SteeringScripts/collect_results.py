#!/usr/bin/env python3
"""Aggregate steering runs into one results table.

Each run file holds one row per prefix, with the deceptive/truthful/invalid
counts of both arms. Rates are over VALID responses only, and validity is
reported separately, because an intervention that suppresses deception by
making the model unparseable has not suppressed anything.

Intervals come from a paired cluster bootstrap over prefixes rather than a
pooled binomial. Samples drawn from one prefix share a scenario and a cut point,
so pooling them would claim a precision the design does not support; resampling
whole prefixes keeps the clustering, and taking both arms of each drawn prefix
cancels the prefix-to-prefix variation they share.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def rate(c: dict) -> float | None:
    v = c["n_deceptive"] + c["n_truthful"]
    return c["n_deceptive"] / v if v else None


def validity(c: dict) -> float | None:
    return (c["n_deceptive"] + c["n_truthful"]) / c["n"] if c["n"] else None


def paired_bootstrap(recs: list[dict], n_boot: int = 2000, seed: int = 0):
    """95% CI on the steered-minus-unsteered difference, resampling prefixes."""
    if len(recs) < 2:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    idx = range(len(recs))
    deltas = []
    for _ in range(n_boot):
        pick = [recs[rng.choice(idx)] for _ in idx]
        du = sum(r["unsteered"]["n_deceptive"] for r in pick)
        vu = sum(r["unsteered"]["n_deceptive"] + r["unsteered"]["n_truthful"] for r in pick)
        ds = sum(r["steered"]["n_deceptive"] for r in pick)
        vs = sum(r["steered"]["n_deceptive"] + r["steered"]["n_truthful"] for r in pick)
        if vu and vs:
            deltas.append(ds / vs - du / vu)
    if len(deltas) < 100:
        return float("nan"), float("nan")
    deltas.sort()
    return deltas[int(.025 * len(deltas))], deltas[int(.975 * len(deltas))]


def summarize(path: Path) -> dict | None:
    recs = []
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not recs:
        return None
    first = recs[0]
    pool = lambda arm, key: sum(r[arm][key] for r in recs)
    u = {k: pool("unsteered", k) for k in ("n", "n_deceptive", "n_truthful")}
    s = {k: pool("steered", k) for k in ("n", "n_deceptive", "n_truthful")}
    lo, hi = paired_bootstrap(recs)
    ru, rs = rate(u), rate(s)
    return {
        "tag": path.stem, "env": first["env"], "alpha": first["alpha"],
        "window": first["window"], "delta_mode": first["delta_mode"],
        "n_prefixes": len(recs),
        "unsteered": ru, "steered": rs,
        "delta": (rs - ru) if (ru is not None and rs is not None) else None,
        "delta_lo": lo, "delta_hi": hi,
        "valid_unsteered": validity(u), "valid_steered": validity(s),
        "counts": {"unsteered": u, "steered": s},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="Results/Steering/runs",
                    help="directory of run JSONL files written by prefix_steering.py")
    ap.add_argument("--out", default="Results/Steering/sweep_results.json")
    ap.add_argument("--min-prefixes", dest="min_prefixes", type=int, default=1,
                    help="skip runs that did not reach this many prefixes")
    args = ap.parse_args()

    rows = []
    for f in sorted(Path(args.runs).glob("*.jsonl")):
        if f.name.endswith(".gen.jsonl") or f.name.endswith(".prefixes.jsonl"):
            continue
        r = summarize(f)
        if r and r["n_prefixes"] >= args.min_prefixes:
            rows.append(r)
    if not rows:
        print(f"no runs found under {args.runs}")
        return 1

    rows.sort(key=lambda r: (r["env"], r["alpha"], r["window"]))
    hdr = (f"{'env':>14} {'alpha':>6} {'window':>12} {'prefixes':>9} "
           f"{'unsteered':>10} {'steered':>8} {'delta':>8} {'95% CI':>18} {'valid':>6}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        ci = ("        --        " if r["delta_lo"] != r["delta_lo"]
              else f"[{r['delta_lo']:+.3f}, {r['delta_hi']:+.3f}]".rjust(18))
        print(f"{r['env']:>14} {r['alpha']:>6.1f} {r['window']:>12} {r['n_prefixes']:>9} "
              f"{r['unsteered']:>10.3f} {r['steered']:>8.3f} {r['delta']:>+8.3f} {ci} "
              f"{r['valid_steered']:>6.2f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {out}: {len(rows)} runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
