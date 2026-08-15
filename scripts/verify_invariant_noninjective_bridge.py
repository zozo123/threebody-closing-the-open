#!/usr/bin/env python3
"""Continuation bridge between far-apart masses with nearly identical invariants.

The full invariant-projection audit found a particularly sharp non-injectivity
certificate: public rows at (m1,m2)=(1.072,1.066) and (1.100,1.121) are far
apart in mass space but almost indistinguishable in the corrected scale-
invariant (T_si,L_si) projection.  This script asks the dynamical question the
projection cannot answer: are those two periodic orbits explicitly connected by
branch-preserving shooting continuation?

Both endpoints are independently corrected.  We then walk the straight mass
segment in both directions with sub-grid steps and require the terminal
shooting coordinates to agree with the independently corrected opposite endpoint
at a strict dimensionless tolerance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from threebody_atlas.baseline import iter_baseline
from threebody_atlas.liao_family import FamilyPoint, correct_family_point

LEFT = (1.072, 1.066, 1.0)
RIGHT = (1.100, 1.121, 1.0)


def params(p: FamilyPoint) -> np.ndarray:
    return np.asarray([p.x1, p.v1, p.v2, p.period], dtype=float)


def corrected_from_row(row, max_residual: float) -> FamilyPoint:
    point = correct_family_point(
        (row.m1, row.m2, row.m3),
        (row.x1, row.v1, row.v2, row.period),
        max_nfev=80,
    )
    if not point.success or point.residual_norm > max_residual:
        raise RuntimeError(f"endpoint correction failed at {(row.m1,row.m2)}: {point.residual_norm:.3e}")
    return point


def distance(a: FamilyPoint, b: FamilyPoint) -> float:
    pa, pb = params(a), params(b)
    scale = np.maximum(np.maximum(np.abs(pa), np.abs(pb)), np.asarray([0.05, 0.5, 0.1, 1.0]))
    return float(np.linalg.norm((pa - pb) / scale))


def walk(start: FamilyPoint, target_masses: tuple[float, float, float], steps: int, max_residual: float):
    m0 = np.asarray(start.masses, dtype=float)
    m1 = np.asarray(target_masses, dtype=float)
    current = start
    path = []
    max_closure = current.residual_norm
    for k in range(1, steps + 1):
        theta = k / steps
        masses = tuple(float(x) for x in ((1.0 - theta) * m0 + theta * m1))
        current = correct_family_point(
            masses,
            (current.x1, current.v1, current.v2, current.period),
            max_nfev=70,
        )
        if not current.success or current.residual_norm > max_residual:
            raise RuntimeError(
                f"continuation failed theta={theta:.6f} masses={masses}: residual={current.residual_norm:.3e}"
            )
        max_closure = max(max_closure, current.residual_norm)
        path.append(
            {
                "theta": theta,
                "masses": masses,
                "x1": current.x1,
                "v1": current.v1,
                "v2": current.v2,
                "period": current.period,
                "shooting_residual": current.residual_norm,
            }
        )
    return current, path, max_closure


def corrected_invariants(row) -> tuple[float, float]:
    m1, m2, m3 = row.m1, row.m2, row.m3
    v3 = -(m1 * row.v1 + m2 * row.v2) / m3
    kinetic = 0.5 * (m1 * row.v1**2 + m2 * row.v2**2 + m3 * v3**2)
    potential = -(m1 * m2 / abs(1.0 - row.x1) + m1 * m3 / abs(row.x1) + m2 * m3)
    energy = kinetic + potential
    angular = m1 * row.x1 * row.v1 + m2 * row.v2
    mt = m1 + m2 + m3
    return (
        row.period * abs(energy) ** 1.5 / mt**2.5,
        angular * abs(energy) ** 0.5 / mt ** (13.0 / 6.0),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("output")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--max-residual", type=float, default=2e-7)
    parser.add_argument("--match-tolerance", type=float, default=2e-5)
    args = parser.parse_args()

    wanted = {LEFT, RIGHT}
    rows = {}
    for row in iter_baseline(args.dataset):
        key = (round(row.m1, 3), round(row.m2, 3), round(row.m3, 3))
        if key in wanted:
            rows[key] = row
    if len(rows) != 2:
        raise RuntimeError(f"failed to locate both public endpoints: {list(rows)}")

    left_row, right_row = rows[LEFT], rows[RIGHT]
    left = corrected_from_row(left_row, args.max_residual)
    right = corrected_from_row(right_row, args.max_residual)
    forward, forward_path, max_forward_closure = walk(left, RIGHT, args.steps, args.max_residual)
    reverse, reverse_path, max_reverse_closure = walk(right, LEFT, args.steps, args.max_residual)
    forward_match = distance(forward, right)
    reverse_match = distance(reverse, left)
    if forward_match > args.match_tolerance or reverse_match > args.match_tolerance:
        raise RuntimeError(
            f"bridge hysteresis gate failed: forward={forward_match:.3e} reverse={reverse_match:.3e}"
        )

    inv_left = np.asarray(corrected_invariants(left_row))
    inv_right = np.asarray(corrected_invariants(right_row))
    invariant_delta = inv_right - inv_left
    mass_distance = float(np.linalg.norm(np.asarray(RIGHT[:2]) - np.asarray(LEFT[:2])))
    payload = {
        "left_masses": LEFT,
        "right_masses": RIGHT,
        "mass_distance": mass_distance,
        "left_invariants": inv_left.tolist(),
        "right_invariants": inv_right.tolist(),
        "raw_invariant_delta": invariant_delta.tolist(),
        "left_corrected_residual": left.residual_norm,
        "right_corrected_residual": right.residual_norm,
        "steps_each_direction": args.steps,
        "max_forward_shooting_residual": max_forward_closure,
        "max_reverse_shooting_residual": max_reverse_closure,
        "forward_terminal_match": forward_match,
        "reverse_terminal_match": reverse_match,
        "passed": True,
        "forward_path": forward_path,
        "reverse_path": reverse_path,
        "claim_status": (
            "direct float64 continuation bridge showing that far-separated masses with nearly "
            "identical invariant coordinates can lie on the same numerically continued orbit sheet"
        ),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "mass_distance": mass_distance,
        "raw_invariant_delta": invariant_delta.tolist(),
        "steps_each_direction": args.steps,
        "max_forward_shooting_residual": max_forward_closure,
        "max_reverse_shooting_residual": max_reverse_closure,
        "forward_terminal_match": forward_match,
        "reverse_terminal_match": reverse_match,
        "passed": True,
    }, indent=2))


if __name__ == "__main__":
    main()
