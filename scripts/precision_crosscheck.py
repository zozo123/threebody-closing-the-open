#!/usr/bin/env python3
"""Cross-check selected published rows with the independent mpmath variational code."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from threebody_atlas.baseline import iter_baseline
from threebody_atlas.high_precision_reduced import verify_reduced_floquet
from threebody_atlas.liao_family import correct_family_point
from threebody_atlas.reduced import compute_reduced_floquet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--rows", nargs="+", type=int, default=[7, 12])
    parser.add_argument("--dps", type=int, default=40)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    wanted = set(args.rows)
    rows = {row.index: row for row in iter_baseline(args.dataset) if row.index in wanted}
    if set(rows) != wanted:
        raise SystemExit(f"missing requested rows: {sorted(wanted - set(rows))}")

    results = []
    for idx in args.rows:
        row = rows[idx]
        point = correct_family_point(
            (row.m1, row.m2, row.m3),
            (row.x1, row.v1, row.v2, row.period),
        )
        float_result = compute_reduced_floquet(point.state(), point.masses, point.period)
        mp_result = verify_reduced_floquet(
            tuple(str(x) for x in point.state()),
            tuple(str(x) for x in point.masses),
            str(point.period),
            dps=args.dps,
            steps=args.steps,
        )
        float_score = min(
            float_result.discriminant,
            2.0 - abs(float_result.trace_roots[0]),
            2.0 - abs(float_result.trace_roots[1]),
        )
        mp_score = float(mp_result.stability_score)
        results.append(
            {
                "baseline_row": idx,
                "published_stability": row.published_stability,
                "shooting_residual": point.residual_norm,
                "float64_score": float_score,
                "mpmath_score": mp_result.stability_score,
                "score_sign_agrees": (float_score > 0) == (mp_score > 0),
                "mpmath": mp_result.__dict__,
                "status": "cross-implementation diagnostic; not yet release-grade",
            }
        )

    payload = {
        "dps": args.dps,
        "coarse_steps": args.steps,
        "rows": results,
        "all_score_signs_agree": all(item["score_sign_agrees"] for item in results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
