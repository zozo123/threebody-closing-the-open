#!/usr/bin/env python3
"""Search the right endpoint of the secondary stability island for a hidden mixed organizer.

At m1=1.042 the coarse global event network still has a lower ``+1`` wall and
an upper ``-1`` wall.  At m1=1.043 both secondary walls have disappeared while
the principal boundaries remain.  This script treats that disappearance as a
falsifiable codimension-two hypothesis rather than silently ending the tracks.

It localizes both secondary roots at m1=1.042, continues them upward in fixed
m1 with closure+event correction, and seeds the simultaneous ``G+=G-=0`` solve
from their closest approach.  The output TSV is only a seed for the independent
Julia BigFloat verifier; Python acceptance is never publication truth.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from threebody_atlas.boundary import BoundarySample, evaluate, stability_score
from threebody_atlas.critical_manifold import (
    LocalizedCriticalPoint,
    _flow_for_vector,
    event_value,
    localize_critical_point,
)
from threebody_atlas.hybrid_vertices import solve_direct_vertex
from threebody_atlas.liao_family import FamilyPoint


def family_point(row: dict[str, str], side: str) -> FamilyPoint:
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


def sign_modes(row: dict[str, str]) -> list[str]:
    left = evaluate(family_point(row, "left"))
    right = evaluate(family_point(row, "right"))
    out: list[str] = []
    for mode in ("plus_one", "minus_one", "trace_collision"):
        a, b = event_value(left.floquet, mode), event_value(right.floquet, mode)
        if a == 0.0 or b == 0.0 or a * b < 0.0:
            out.append(mode)
    return out


def localize(row: dict[str, str], mode: str) -> LocalizedCriticalPoint:
    left = family_point(row, "left")
    right = family_point(row, "right")
    stable = left if row["left_label"] == "S" else right
    unstable = left if row["left_label"] == "U" else right
    return localize_critical_point(
        stable,
        unstable,
        event_mode=mode,
        m2_tolerance=2e-9,
        event_tolerance=2e-8,
        max_iterations=40,
        max_closure=1e-7,
    )


def fixed_m1_root(
    anchor: LocalizedCriticalPoint,
    target_m1: float,
    mode: str,
    *,
    max_m2_jump: float,
) -> LocalizedCriticalPoint:
    q = anchor.sample.point
    m3 = float(q.masses[2])
    x0 = np.asarray([q.x1, q.v1, q.v2, q.period, q.masses[1]], dtype=float)
    floors = np.asarray([0.05, 0.5, 0.1, 1.0, 0.1], dtype=float)
    scales = np.maximum(np.abs(x0), floors)

    def residual(u: np.ndarray) -> np.ndarray:
        y = np.asarray([u[0], u[1], u[2], u[3], target_m1, u[4]], dtype=float)
        closure, floquet = _flow_for_vector(y, m3=m3, rtol=3e-10, atol=3e-12)
        return np.concatenate((closure / 1e-6, [event_value(floquet, mode) / 2e-4]))

    fit = least_squares(
        residual,
        x0,
        method="trf",
        bounds=(
            np.asarray([-2.0, -10.0, -10.0, 0.1, 0.5]),
            np.asarray([2.0, 10.0, 10.0, 20.0, 1.5]),
        ),
        x_scale=scales,
        xtol=2e-11,
        ftol=2e-11,
        gtol=2e-11,
        max_nfev=80,
    )
    y = np.asarray([fit.x[0], fit.x[1], fit.x[2], fit.x[3], target_m1, fit.x[4]], dtype=float)
    closure, floquet = _flow_for_vector(y, m3=m3, rtol=3e-10, atol=3e-12)
    cn = float(np.linalg.norm(closure))
    ev = float(event_value(floquet, mode))
    jump = abs(float(fit.x[4]) - float(q.masses[1]))
    if not fit.success:
        raise RuntimeError(f"{mode} fixed-m1 correction failed: {fit.message}")
    if cn > 2e-7 or abs(ev) > 2e-6:
        raise RuntimeError(f"{mode} fixed-m1 gates failed: closure={cn:.3e} event={ev:.3e}")
    if jump > max_m2_jump:
        raise RuntimeError(f"{mode} fixed-m1 branch jump {jump:.3e} > {max_m2_jump:.3e}")
    point = FamilyPoint(
        masses=(float(target_m1), float(fit.x[4]), m3),
        x1=float(fit.x[0]),
        v1=float(fit.x[1]),
        v2=float(fit.x[2]),
        period=float(fit.x[3]),
        residual_norm=cn,
        nfev=int(fit.nfev),
        success=True,
    )
    return LocalizedCriticalPoint(BoundarySample(point, floquet, stability_score(floquet)), mode, ev, float("nan"))


def vector(p: LocalizedCriticalPoint) -> np.ndarray:
    q = p.sample.point
    return np.asarray([q.x1, q.v1, q.v2, q.period, q.masses[0], q.masses[1]], dtype=float)


def serialize(p: LocalizedCriticalPoint) -> dict[str, Any]:
    q, f = p.sample.point, p.sample.floquet
    return {
        "masses": [float(x) for x in q.masses],
        "x1": float(q.x1),
        "v1": float(q.v1),
        "v2": float(q.v2),
        "period": float(q.period),
        "closure": float(q.residual_norm),
        "event_mode": p.event_mode,
        "event": float(p.event_value),
        "alpha": float(f.alpha),
        "beta": float(f.beta),
        "discriminant": float(f.discriminant),
        "trace_roots": [[float(z.real), float(z.imag)] for z in f.trace_roots],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("brackets_tsv")
    parser.add_argument("output_json")
    parser.add_argument("output_seed_tsv")
    parser.add_argument("--seed-m1", type=float, default=1.042)
    parser.add_argument("--m1-step", type=float, default=2e-5)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--m2-min", type=float, default=1.0)
    parser.add_argument("--m2-max", type=float, default=1.07)
    parser.add_argument("--max-m2-jump", type=float, default=0.004)
    args = parser.parse_args()

    with Path(args.brackets_tsv).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    candidates: dict[str, LocalizedCriticalPoint] = {}
    for row in rows:
        if abs(float(row["m1"]) - args.seed_m1) > 5e-10:
            continue
        midpoint = 0.5 * (float(row["left_m2"]) + float(row["right_m2"]))
        if not (args.m2_min <= midpoint <= args.m2_max):
            continue
        modes = sign_modes(row)
        if len(modes) == 1 and modes[0] in ("plus_one", "minus_one"):
            candidates[modes[0]] = localize(row, modes[0])
    if set(candidates) != {"plus_one", "minus_one"}:
        raise RuntimeError(f"did not isolate secondary +1/-1 seed roots: {sorted(candidates)}")

    plus = candidates["plus_one"]
    minus = candidates["minus_one"]
    approach: list[dict[str, Any]] = []
    best_pair = (float("inf"), plus, minus)
    for step in range(args.steps + 1):
        gap = float(np.linalg.norm(np.asarray(plus.sample.point.masses[:2]) - np.asarray(minus.sample.point.masses[:2])))
        approach.append({"step": step, "plus": serialize(plus), "minus": serialize(minus), "mass_gap": gap})
        if gap < best_pair[0]:
            best_pair = (gap, plus, minus)
        if step == args.steps:
            break
        target_m1 = args.seed_m1 + (step + 1) * args.m1_step
        try:
            next_plus = fixed_m1_root(plus, target_m1, "plus_one", max_m2_jump=args.max_m2_jump)
            next_minus = fixed_m1_root(minus, target_m1, "minus_one", max_m2_jump=args.max_m2_jump)
        except Exception as exc:
            approach.append({"step": step + 1, "target_m1": target_m1, "stopped": f"{type(exc).__name__}: {exc}"})
            break
        plus, minus = next_plus, next_minus

    closest_gap, close_plus, close_minus = best_pair
    seed = 0.5 * (vector(close_plus) + vector(close_minus))
    direct = solve_direct_vertex(
        seed,
        "mixed_plus_minus_one",
        m3=1.0,
        mass_bounds=((1.0415, 1.0435), (1.02, 1.06)),
        max_nfev=100,
    )
    result = {
        "claim_status": "float64 direct candidate only; independent Julia BigFloat/canonical verification required",
        "coarse_gap_hypothesis": "secondary +1 and -1 walls both disappear between m1=1.042 and 1.043",
        "closest_approach_mass_gap": closest_gap,
        "approach": approach,
        "direct_candidate": {
            "masses": [float(x) for x in direct.point.masses],
            "x1": float(direct.point.x1),
            "v1": float(direct.point.v1),
            "v2": float(direct.point.v2),
            "period": float(direct.point.period),
            "closure": float(direct.point.residual_norm),
            "alpha": float(direct.alpha),
            "beta": float(direct.beta),
            "event_values": [float(x) for x in direct.event_values],
            "invariant_error": float(direct.invariant_error),
            "nfev": int(direct.nfev),
        },
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    q = direct.point
    with Path(args.output_seed_tsv).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["name", "m1", "m2", "m3", "x1", "v1", "v2", "period"])
        writer.writerow([
            "secondary-right-mixed",
            f"{q.masses[0]:.17g}", f"{q.masses[1]:.17g}", f"{q.masses[2]:.17g}",
            f"{q.x1:.17g}", f"{q.v1:.17g}", f"{q.v2:.17g}", f"{q.period:.17g}",
        ])
    print(json.dumps(result["direct_candidate"], indent=2))


if __name__ == "__main__":
    main()
