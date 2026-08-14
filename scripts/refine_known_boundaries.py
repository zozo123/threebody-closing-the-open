#!/usr/bin/env python3
"""Refine the two m1=0.8 stability brackets visible in the published grid."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from threebody_atlas.baseline import iter_baseline
from threebody_atlas.boundary import BoundaryResult, refine_m2_boundary
from threebody_atlas.liao_family import FamilyPoint, correct_family_point


def corrected(row) -> FamilyPoint:
    return correct_family_point(
        (row.m1, row.m2, row.m3),
        (row.x1, row.v1, row.v2, row.period),
        max_nfev=60,
    )


def serialize(result: BoundaryResult) -> dict:
    def side(sample):
        p = sample.point
        f = sample.floquet
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
    return {
        "stable_side": side(result.stable_side),
        "unstable_side": side(result.unstable_side),
        "iterations": result.iterations,
        "parameter_width": result.parameter_width,
        "claim_status": "screening-only; requires high-precision variational verification",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=2e-7)
    args = parser.parse_args()

    wanted = {6, 7, 11, 12}
    rows = {row.index: row for row in iter_baseline(args.dataset) if row.index in wanted}
    if set(rows) != wanted:
        raise SystemExit(f"required baseline rows not found: have {sorted(rows)}")

    endpoints = {idx: corrected(row) for idx, row in rows.items()}
    lower = refine_m2_boundary(
        stable=endpoints[7],
        unstable=endpoints[6],
        m2_tolerance=args.tolerance,
    )
    upper = refine_m2_boundary(
        stable=endpoints[11],
        unstable=endpoints[12],
        m2_tolerance=args.tolerance,
    )

    payload = {
        "experiment": "m1=0.8 published-grid boundary refinement",
        "published_brackets": [[0.755, 0.756], [0.760, 0.761]],
        "lower_boundary": serialize(lower),
        "upper_boundary": serialize(upper),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
