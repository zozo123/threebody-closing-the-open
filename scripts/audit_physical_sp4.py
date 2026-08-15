#!/usr/bin/env python3
"""Cross-check the physical 4D symplectic quotient against existing invariants.

The current reduced 8D trace formulation encodes four neutral unit multipliers
algebraically.  This audit constructs the physical quotient ``E^omega/E``
directly from the canonical Jacobi monodromy and checks that its 4x4 invariant
coefficients agree with the neutral-factor removal formulas

    a = alpha - 4,
    b = beta - 4 alpha + 10.

It deliberately uses tighter float64 orbit/tangent tolerances than the discovery
stack.  The scientific acceptance thresholds are not relaxed when a run fails.
Published stable/unstable anchors and the two frozen critical representatives
are reported individually so a precision-sensitive point cannot hide inside a
single maximum statistic.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from threebody_atlas.baseline import iter_baseline
from threebody_atlas.liao_family import correct_family_point, state_from_chart
from threebody_atlas.physical_floquet import compute_physical_floquet
from threebody_atlas.reduced import compute_reduced_floquet

CORRECT_RTOL = 2e-12
CORRECT_ATOL = 2e-14
FLOQUET_RTOL = 5e-13
FLOQUET_ATOL = 5e-15


def record_point(name: str, masses, chart, *, max_closure: float) -> dict:
    point = correct_family_point(
        tuple(masses),
        tuple(chart),
        max_nfev=120,
        screening_rtol=CORRECT_RTOL,
        screening_atol=CORRECT_ATOL,
    )
    if not point.success or point.residual_norm > max_closure:
        raise RuntimeError(f"periodic correction failed for {name}: {point.residual_norm:.3e}")
    state0 = state_from_chart(point.masses, point.x1, point.v1, point.v2)
    mass_array = np.asarray(point.masses, dtype=float)
    reduced = compute_reduced_floquet(
        state0,
        mass_array,
        point.period,
        rtol=FLOQUET_RTOL,
        atol=FLOQUET_ATOL,
    )
    physical = compute_physical_floquet(
        state0,
        mass_array,
        point.period,
        rtol=FLOQUET_RTOL,
        atol=FLOQUET_ATOL,
    )

    expected_a = reduced.alpha - 4.0
    expected_b = reduced.beta - 4.0 * reduced.alpha + 10.0
    expected_plus = reduced.beta - 6.0 * reduced.alpha + 20.0
    expected_minus = reduced.beta - 2.0 * reduced.alpha + 4.0
    expected_collision = reduced.discriminant
    mismatches = {
        "a": float(physical.trace_a - expected_a),
        "b": float(physical.trace_b - expected_b),
        "plus_one": float(physical.plus_one_event - expected_plus),
        "minus_one": float(physical.minus_one_event - expected_minus),
        "collision": float(physical.collision_event - expected_collision),
    }
    record = {
        "name": name,
        "masses": [float(x) for x in point.masses],
        "chart": {
            "x1": float(point.x1),
            "v1": float(point.v1),
            "v2": float(point.v2),
            "period": float(point.period),
        },
        "shooting_residual": float(point.residual_norm),
        "canonical_closure": float(physical.canonical.closure_norm),
        "canonical_symplectic_defect": float(physical.canonical.symplectic_defect),
        "physical_symplectic_defect": float(physical.quotient_symplectic_defect),
        "quotient_leakage": float(physical.quotient_leakage),
        "physical_pairing_error": float(physical.reciprocal_pairing_error),
        "neutral_isotropy_defect": float(physical.neutral_isotropy_defect),
        "neutral_invariance_defect": float(physical.neutral_invariance_defect),
        "reduced": {
            "alpha": float(reduced.alpha),
            "beta": float(reduced.beta),
            "discriminant": float(reduced.discriminant),
            "plus_one": float(expected_plus),
            "minus_one": float(expected_minus),
        },
        "physical": {
            "a": float(physical.trace_a),
            "b": float(physical.trace_b),
            "discriminant": float(physical.discriminant),
            "plus_one": float(physical.plus_one_event),
            "minus_one": float(physical.minus_one_event),
            "trace_roots": [
                [float(root.real), float(root.imag)] for root in physical.trace_roots
            ],
            "multipliers": [
                [float(value.real), float(value.imag)] for value in physical.multipliers
            ],
        },
        "invariant_mismatches": mismatches,
    }
    print(
        json.dumps(
            {
                "name": name,
                "shooting_residual": record["shooting_residual"],
                "canonical_closure": record["canonical_closure"],
                "canonical_symplectic_defect": record["canonical_symplectic_defect"],
                "physical_symplectic_defect": record["physical_symplectic_defect"],
                "quotient_leakage": record["quotient_leakage"],
                "pairing": record["physical_pairing_error"],
                "neutral_invariance": record["neutral_invariance_defect"],
                "max_invariant_mismatch": max(abs(v) for v in mismatches.values()),
            }
        ),
        flush=True,
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("representatives_tsv")
    parser.add_argument("output")
    parser.add_argument("--max-closure", type=float, default=2e-7)
    parser.add_argument("--max-invariant-mismatch", type=float, default=5e-4)
    parser.add_argument("--max-physical-defect", type=float, default=5e-6)
    parser.add_argument("--max-leakage", type=float, default=5e-6)
    parser.add_argument("--max-pairing", type=float, default=5e-5)
    parser.add_argument("--max-neutral-invariance", type=float, default=5e-5)
    args = parser.parse_args()

    published = []
    for index, row in enumerate(iter_baseline(args.dataset), start=1):
        if index in {7, 11, 12}:
            published.append(
                record_point(
                    f"published_row_{index}_{row.published_stability}",
                    (row.m1, row.m2, row.m3),
                    (row.x1, row.v1, row.v2, row.period),
                    max_closure=args.max_closure,
                )
            )
        if index > 12:
            break
    if len(published) != 3:
        raise RuntimeError("failed to load the three published audit anchors")

    representatives = []
    with Path(args.representatives_tsv).open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            record = record_point(
                row["name"],
                (float(row["m1"]), float(row["m2"]), float(row["m3"])),
                (
                    float(row["x1"]),
                    float(row["v1"]),
                    float(row["v2"]),
                    float(row["period"]),
                ),
                max_closure=args.max_closure,
            )
            record["screening_event_mode"] = row["event_mode"]
            record["source_screening_event"] = float(row["screening_event"])
            representatives.append(record)

    records = published + representatives
    max_mismatch = max(
        abs(value)
        for record in records
        for value in record["invariant_mismatches"].values()
    )
    max_canonical_defect = max(record["canonical_symplectic_defect"] for record in records)
    max_physical_defect = max(record["physical_symplectic_defect"] for record in records)
    max_leakage = max(record["quotient_leakage"] for record in records)
    max_pairing = max(record["physical_pairing_error"] for record in records)
    max_neutral_invariance = max(record["neutral_invariance_defect"] for record in records)
    max_neutral_isotropy = max(record["neutral_isotropy_defect"] for record in records)

    gates = {
        "invariant_mismatch": max_mismatch <= args.max_invariant_mismatch,
        "physical_symplectic_defect": max_physical_defect <= args.max_physical_defect,
        "quotient_leakage": max_leakage <= args.max_leakage,
        "reciprocal_pairing": max_pairing <= args.max_pairing,
        "neutral_invariance": max_neutral_invariance <= args.max_neutral_invariance,
        "neutral_isotropy": max_neutral_isotropy <= 5e-9,
    }
    payload = {
        "method": "canonical Jacobi monodromy -> E^omega/E physical 4D quotient",
        "neutral_subspace": "E=span{X_H,X_L}",
        "float64_tolerances": {
            "correct_rtol": CORRECT_RTOL,
            "correct_atol": CORRECT_ATOL,
            "floquet_rtol": FLOQUET_RTOL,
            "floquet_atol": FLOQUET_ATOL,
        },
        "coefficient_relation": {
            "a": "alpha-4",
            "b": "beta-4*alpha+10",
        },
        "event_equations_physical": {
            "plus_one": "b-2*a+2",
            "minus_one": "b+2*a+2",
            "collision": "a^2-4*b+8",
        },
        "records": records,
        "summary": {
            "max_invariant_mismatch": max_mismatch,
            "max_canonical_symplectic_defect": max_canonical_defect,
            "max_physical_symplectic_defect": max_physical_defect,
            "max_quotient_leakage": max_leakage,
            "max_reciprocal_pairing_error": max_pairing,
            "max_neutral_invariance_defect": max_neutral_invariance,
            "max_neutral_isotropy_defect": max_neutral_isotropy,
            "gates": gates,
            "passed": all(gates.values()),
        },
        "claim_status": (
            "tight-float64 structural cross-check of the physical Sp(4) quotient; exact critical "
            "mechanisms still require independent BigFloat/canonical reproduction"
        ),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2), flush=True)
    if not payload["summary"]["passed"]:
        raise SystemExit("physical Sp(4) cross-check failed one or more structural gates")


if __name__ == "__main__":
    main()
