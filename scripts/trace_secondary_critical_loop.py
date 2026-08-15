#!/usr/bin/env python3
"""Trace the secondary stability-lobe critical boundary through its turning points.

The coarse 0.001 grid shows an extra pair of S/U crossings for
m1≈0.996--1.042.  This driver seeds the *lower* extra crossing on two adjacent
slices and follows the smooth Floquet event with nested pseudo-arclength.  It is
therefore free to turn in m1 and continue onto the upper extra crossing, unlike
fixed-m1 bisection.

If the trajectory returns to the seed with compatible tangent orientation we
record a closed-loop screen.  Final loop geometry still requires BigFloat
verification at representative points and folds.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from threebody_atlas.critical_manifold import localize_critical_point
from threebody_atlas.critical_massplane import advance_massplane_critical
from threebody_atlas.liao_family import FamilyPoint


def midpoint(row: dict[str, str]) -> float:
    return 0.5 * (float(row["left_m2"]) + float(row["right_m2"]))


def point(row: dict[str, str], side: str) -> FamilyPoint:
    return FamilyPoint(
        masses=(float(row["m1"]), float(row[f"{side}_m2"]), float(row["m3"])),
        x1=float(row[f"{side}_x1"]),
        v1=float(row[f"{side}_v1"]),
        v2=float(row[f"{side}_v2"]),
        period=float(row[f"{side}_period"]),
        residual_norm=float("nan"),
        nfev=0,
        success=True,
    )


def localize(row: dict[str, str], mode=None):
    left, right = point(row, "left"), point(row, "right")
    stable = left if row["left_label"] == "S" else right
    unstable = left if row["left_label"] == "U" else right
    return localize_critical_point(
        stable,
        unstable,
        event_mode=mode,
        m2_tolerance=5e-9,
        event_tolerance=5e-8,
        max_iterations=32,
        max_closure=1e-7,
    )


def serialize(p) -> dict:
    q, f = p.sample.point, p.sample.floquet
    return {
        "masses": q.masses,
        "x1": q.x1,
        "v1": q.v1,
        "v2": q.v2,
        "period": q.period,
        "shooting_residual": q.residual_norm,
        "event_mode": p.event_mode,
        "event_value": p.event_value,
        "alpha": f.alpha,
        "beta": f.beta,
        "discriminant": f.discriminant,
        "trace_roots": [[z.real, z.imag] for z in f.trace_roots],
        "mass_tangent": getattr(p, "mass_tangent", None),
        "arclength_residual": getattr(p, "arclength_residual", None),
        "step": getattr(p, "step", None),
        "outer_nfev": getattr(p, "outer_nfev", None),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("brackets_tsv")
    parser.add_argument("output")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--step", type=float, default=0.001)
    parser.add_argument("--min-step", type=float, default=6.25e-5)
    args = parser.parse_args()

    with Path(args.brackets_tsv).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    by_m1 = {}
    for m1 in (0.996, 0.997):
        candidates = [
            row for row in rows
            if abs(float(row["m1"]) - m1) < 5e-7
            and row["left_label"] == "U"
            and row["right_label"] == "S"
        ]
        if len(candidates) < 2:
            raise RuntimeError(f"expected two U->S crossings at m1={m1}, found {len(candidates)}")
        by_m1[m1] = min(candidates, key=midpoint)

    first = localize(by_m1[0.996])
    second = localize(by_m1[0.997], mode=first.event_mode)
    previous, current = first, second
    points = []
    requested_step = args.step
    current_step = requested_step
    initial_pair = np.asarray(first.sample.point.masses[:2], dtype=float)
    initial_secant = np.asarray(second.sample.point.masses[:2], dtype=float) - initial_pair
    initial_tangent = initial_secant / np.linalg.norm(initial_secant)
    closed = False
    stopped_reason = "max_steps_reached"
    min_m1 = min(first.sample.point.masses[0], second.sample.point.masses[0])
    max_m1 = max(first.sample.point.masses[0], second.sample.point.masses[0])

    for k in range(args.max_steps):
        trial = current_step
        accepted = None
        last_error = None
        for _retry in range(5):
            try:
                accepted = advance_massplane_critical(previous, current, step=trial)
                break
            except (RuntimeError, ValueError) as exc:
                last_error = exc
                trial *= 0.5
                if trial < args.min_step:
                    break
        if accepted is None:
            stopped_reason = f"continuation_failed: {last_error}"
            break
        points.append(accepted)
        pair = accepted.mass_pair
        min_m1 = min(min_m1, float(pair[0]))
        max_m1 = max(max_m1, float(pair[0]))

        # After moving a meaningful fraction of the coarse loop, accept closure
        # only if position and tangent orientation both return to the seed.
        if k >= 30:
            distance = float(np.linalg.norm(pair - initial_pair))
            tangent = np.asarray(accepted.mass_tangent, dtype=float)
            alignment = float(np.dot(tangent, initial_tangent))
            if distance <= 2.0 * requested_step and alignment >= 0.5:
                closed = True
                stopped_reason = "returned_to_seed_with_aligned_tangent"
                break

        previous, current = current, accepted
        current_step = min(requested_step, trial * 1.25)

    payload = {
        "implementation": "nested mass-plane pseudo-arclength",
        "event_mode": first.event_mode,
        "closed_loop_screen": closed,
        "stopped_reason": stopped_reason,
        "requested_step": requested_step,
        "points": len(points),
        "m1_range_reached": [min_m1, max_m1],
        "localized_seeds": [serialize(first), serialize(second)],
        "trace": [serialize(p) for p in points],
        "claim_status": "float64 screening; folds/closure require independent BigFloat validation",
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "event_mode": first.event_mode,
        "closed_loop_screen": closed,
        "stopped_reason": stopped_reason,
        "points": len(points),
        "m1_range_reached": [min_m1, max_m1],
    }, indent=2))
    if len(points) < 5:
        raise RuntimeError("secondary-loop trace made fewer than five accepted pseudo-arclength steps")


if __name__ == "__main__":
    main()
