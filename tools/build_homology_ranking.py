#!/usr/bin/env python3
"""Combine general router-hit histograms and identity/homology router-hit histograms
into a per-layer [24][128] homology ranking JSON for `atlas-6g-homology` (`--expert-ranking`).

Usage:
  # 1. Collect general router stats via ingine-bench --atlas-6g --dump-router-stats general.json
  # 2. Collect identity/persona/ledger router stats via --dump-router-stats identity.json
  # 3. Combine:
  python3 tools/build_homology_ranking.py \
      --general general.json \
      --identity identity.json \
      --out atlas-6g-homology-ranking.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_hits(path: Path) -> list[list[float]]:
    data = json.loads(path.read_text())
    arr = data.get("router_hits") or data.get("scores")
    if not isinstance(arr, list) or len(arr) != 24:
        raise ValueError(f"{path}: expected 24-layer array in 'router_hits' or 'scores'")
    return [[float(x) for x in row] for row in arr]


def normalize_layer(row: list[float]) -> list[float]:
    s = sum(row)
    return [x / s if s > 0.0 else 0.0 for x in row]


def main() -> None:
    ap = argparse.ArgumentParser(description="Build homology-weighted expert ranking JSON for atlas-6g-homology")
    ap.add_argument("--general", required=True, help="Router stats JSON from general chat/code/reasoning prompts")
    ap.add_argument("--identity", default="", help="Optional router stats JSON from identity/who-you-are/ledger prompts")
    ap.add_argument("--identity-weight", type=float, default=1.0, help="Weight on identity frequency (default 1.0)")
    ap.add_argument("--lift-weight", type=float, default=1.5, help="Bonus weight on positive identity lift (default 1.5)")
    ap.add_argument("--tail-boost", type=float, default=1.35, help="Output tail L20-23 priority multiplier (default 1.35)")
    ap.add_argument("--out", required=True, help="Output ranking JSON path")
    args = ap.parse_args()

    gen = load_hits(Path(args.general))
    ident = load_hits(Path(args.identity)) if args.identity else None

    scores: list[list[float]] = []
    for l in range(24):
        visit_mult = 3.0 if 6 <= l < 18 else (args.tail_boost if l >= 20 else 1.0)
        g_norm = normalize_layer(gen[l])
        i_norm = normalize_layer(ident[l]) if ident is not None else g_norm
        row: list[float] = []
        for e in range(128):
            lift = max(0.0, i_norm[e] - g_norm[e])
            val = visit_mult * (g_norm[e] + args.identity_weight * i_norm[e] + args.lift_weight * lift)
            row.append(round(val, 8))
        scores.append(row)

    out_obj = {
        "schema": "atlas-6g-homology-ranking-v1",
        "layers": 24,
        "experts": 128,
        "params": {
            "identity_weight": args.identity_weight,
            "lift_weight": args.lift_weight,
            "tail_boost": args.tail_boost,
            "general_src": str(args.general),
            "identity_src": str(args.identity),
        },
        "scores": scores,
    }
    Path(args.out).write_text(json.dumps(out_obj, indent=2) + "\n")
    # Print top-5 summary per tier
    flat = [(scores[l][e], l, e) for l in range(24) for e in range(128)]
    flat.sort(reverse=True)
    top500_loop = sum(1 for _, l, _ in flat[:500] if 6 <= l < 18)
    top500_tail = sum(1 for _, l, _ in flat[:500] if l >= 20)
    print(f"Wrote {args.out}: top-500 breakdown -> loop(L6-17)={top500_loop}, tail(L20-23)={top500_tail}, other={500 - top500_loop - top500_tail}")


if __name__ == "__main__":
    main()
