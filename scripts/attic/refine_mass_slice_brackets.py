#!/usr/bin/env python3
"""Refine published S/U transition brackets across selected m1 slices."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from threebody_atlas.boundary import refine_m2_boundary
from threebody_atlas.liao_family import FamilyPoint, correct_family_point


def point_from_side(row: dict[str, str], side: str) -> FamilyPoint:
    masses = (float(row["m1"]), float(row[f"{side}_m2"]), float(row["m3"]))
    guess = (
        float(row[f"{side}_x1"]),
        float(row[f"{side}_v1"]),
        float(row[f"{side}_v2"]),
        float(row[f"{side}_period"]),
    )
    return correct_family_point(masses, guess, max_nfev=60)


def serialize_sample(sample) -> dict:
    p, f = sample.point, sample.floquet
    return {
        "masses": p.masses,
        "x1": p.x1,
        "v1": p.v1,
        "v2": p.v2,
        "period": p.period,
        "shooting_residual": p.residual_norm,
        "stability_score": sample.score,
        "alpha": f.alpha,
        "beta": f.beta,
        "discriminant": f.discriminant,
        "trace_roots": [[z.real, z.imag] for z in f.trace_roots],
        "screening_stable": f.linearly_stable,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("brackets_tsv")
    parser.add_argument("output")
    parser.add_argument("--tolerance", type=float, default=2e-7)
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()

    with Path(args.brackets_tsv).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    results = []
    for row in rows[: args.limit]:
        left = point_from_side(row, "left")
        right = point_from_side(row, "right")
        left_label, right_label = row["left_label"], row["right_label"]
        stable = left if left_label == "S" else right
        unstable = left if left_label == "U" else right
        refined = refine_m2_boundary(stable, unstable, m2_tolerance=args.tolerance)
        results.append(
            {
                "m1": float(row["m1"]),
                "published_labels": [left_label, right_label],
                "published_bracket": [float(row["left_m2"]), float(row["right_m2"])],
                "stable_side": serialize_sample(refined.stable_side),
                "unstable_side": serialize_sample(refined.unstable_side),
                "width": refined.parameter_width,
            }
        )
    payload = {
        "count": len(results),
        "claim_status": "screening-only",
        "refined_brackets": results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"count": len(results), "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
