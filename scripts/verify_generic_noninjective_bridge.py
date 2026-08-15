#!/usr/bin/env python3
"""Repeat the pathological invariant bridge outside the Li shooting ansatz.

The existing bridge establishes bidirectional continuation between public rows
``(m1,m2)=(1.072,1.066)`` and ``(1.100,1.121)`` using the Li chart.  This script
repeats the same mass-space path with the generic 8D translation-reduced
periodic corrector.  Li solutions are used only as endpoint/initial warm starts;
the correction itself does not impose collinearity or the Li velocity pattern.

Passing this experiment is stronger evidence against a chart-induced family
split.  It is still a finite path experiment, not a theorem of global catalog
connectedness.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from threebody_atlas.baseline import iter_baseline
from threebody_atlas.generic_periodic import GenericPeriodicPoint, correct_generic_periodic
from threebody_atlas.liao_family import correct_family_point, state_from_chart
from threebody_atlas.reduced import full_to_reduced

LEFT = (1.072, 1.066, 1.0)
RIGHT = (1.100, 1.121, 1.0)


def endpoint_from_row(row, max_residual: float) -> GenericPeriodicPoint:
    li = correct_family_point(
        (row.m1, row.m2, row.m3),
        (row.x1, row.v1, row.v2, row.period),
        max_nfev=80,
    )
    if not li.success or li.residual_norm > max_residual:
        raise RuntimeError(f"Li endpoint correction failed: {li.residual_norm:.3e}")
    reduced = full_to_reduced(
        state_from_chart(li.masses, li.x1, li.v1, li.v2)
    )
    generic = correct_generic_periodic(
        li.masses,
        reduced,
        li.period,
        reference_state=reduced,
        max_nfev=80,
        max_closure=max_residual,
    )
    if not generic.success:
        raise RuntimeError(
            "generic endpoint correction failed: "
            f"closure={generic.closure_norm:.3e} gauge={generic.gauge_norm:.3e} "
            f"phase={generic.phase_residual:.3e}"
        )
    return generic


def distance(a: GenericPeriodicPoint, b: GenericPeriodicPoint) -> float:
    va, vb = a.vector, b.vector
    floors = np.asarray([0.2, 0.2, 0.5, 0.2, 0.5, 0.5, 0.5, 0.5, 1.0])
    scale = np.maximum(np.maximum(np.abs(va), np.abs(vb)), floors)
    return float(np.linalg.norm((va - vb) / scale))


def off_li_norm(point: GenericPeriodicPoint) -> float:
    z = np.asarray(point.state, dtype=float)
    # In this phase/gauge the Li ansatz additionally has r13_y=v13_x=v23_x=0.
    return float(np.linalg.norm(z[[1, 4, 6]]))


def walk(
    start: GenericPeriodicPoint,
    target_masses: tuple[float, float, float],
    *,
    steps: int,
    max_residual: float,
) -> tuple[GenericPeriodicPoint, list[dict], float, float]:
    m0 = np.asarray(start.masses, dtype=float)
    m1 = np.asarray(target_masses, dtype=float)
    current = start
    path: list[dict] = []
    max_closure = current.closure_norm
    max_off_li = off_li_norm(current)
    for k in range(1, steps + 1):
        theta = k / steps
        masses = tuple(float(x) for x in ((1.0 - theta) * m0 + theta * m1))
        reference = np.asarray(current.state, dtype=float)
        next_point = correct_generic_periodic(
            masses,
            reference,
            current.period,
            reference_state=reference,
            max_nfev=55,
            max_closure=max_residual,
        )
        if not next_point.success:
            raise RuntimeError(
                f"generic continuation failed at theta={theta:.6f}, masses={masses}: "
                f"closure={next_point.closure_norm:.3e} gauge={next_point.gauge_norm:.3e} "
                f"phase={next_point.phase_residual:.3e}"
            )
        current = next_point
        max_closure = max(max_closure, current.closure_norm)
        max_off_li = max(max_off_li, off_li_norm(current))
        path.append(
            {
                "theta": theta,
                "masses": masses,
                "period": current.period,
                "closure_norm": current.closure_norm,
                "gauge_norm": current.gauge_norm,
                "phase_residual": current.phase_residual,
                "nfev": current.nfev,
                "off_li_norm": off_li_norm(current),
            }
        )
    return current, path, max_closure, max_off_li


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("output")
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--max-residual", type=float, default=2e-7)
    parser.add_argument("--match-tolerance", type=float, default=5e-5)
    args = parser.parse_args()
    if args.steps < 2:
        raise SystemExit("steps must be at least 2")

    wanted = {LEFT, RIGHT}
    rows = {}
    for row in iter_baseline(args.dataset):
        key = (round(row.m1, 3), round(row.m2, 3), round(row.m3, 3))
        if key in wanted:
            rows[key] = row
    if len(rows) != 2:
        raise RuntimeError(f"failed to locate both public endpoints: {list(rows)}")

    left = endpoint_from_row(rows[LEFT], args.max_residual)
    right = endpoint_from_row(rows[RIGHT], args.max_residual)
    forward, forward_path, max_forward, max_forward_off = walk(
        left,
        RIGHT,
        steps=args.steps,
        max_residual=args.max_residual,
    )
    reverse, reverse_path, max_reverse, max_reverse_off = walk(
        right,
        LEFT,
        steps=args.steps,
        max_residual=args.max_residual,
    )
    forward_match = distance(forward, right)
    reverse_match = distance(reverse, left)
    passed = forward_match <= args.match_tolerance and reverse_match <= args.match_tolerance
    if not passed:
        raise RuntimeError(
            "generic bridge hysteresis gate failed: "
            f"forward={forward_match:.3e} reverse={reverse_match:.3e}"
        )

    payload = {
        "left_masses": LEFT,
        "right_masses": RIGHT,
        "formulation": (
            "8D translation-reduced strict periodic single shooting with local scale, rotation, "
            "and time-phase gauges; no Li collinearity/velocity ansatz in the corrector"
        ),
        "steps_each_direction": args.steps,
        "left_endpoint": {
            "closure": left.closure_norm,
            "gauge": left.gauge_norm,
            "phase": left.phase_residual,
            "off_li_norm": off_li_norm(left),
        },
        "right_endpoint": {
            "closure": right.closure_norm,
            "gauge": right.gauge_norm,
            "phase": right.phase_residual,
            "off_li_norm": off_li_norm(right),
        },
        "max_forward_closure": max_forward,
        "max_reverse_closure": max_reverse,
        "max_forward_off_li_norm": max_forward_off,
        "max_reverse_off_li_norm": max_reverse_off,
        "forward_terminal_match": forward_match,
        "reverse_terminal_match": reverse_match,
        "passed": passed,
        "forward_path": forward_path,
        "reverse_path": reverse_path,
        "claim_status": (
            "cross-chart float64 continuation evidence; finite-path result only, not a global "
            "connectivity theorem"
        ),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "steps_each_direction": args.steps,
                "max_forward_closure": max_forward,
                "max_reverse_closure": max_reverse,
                "forward_terminal_match": forward_match,
                "reverse_terminal_match": reverse_match,
                "max_off_li_norm": max(max_forward_off, max_reverse_off),
                "passed": passed,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
