#!/usr/bin/env python3
"""Trace the secondary ``-1`` critical curve through its suspected m1 fold.

The coarse event network has two distinct minus-one S/U transition roots at
m1=0.996, one on each secondary boundary.  If those roots are the two sides of
one folded critical component, decreasing m1 should drive them together and a
six-dimensional pseudo-arclength trace should turn in m1 and emerge on the
opposite branch.

This is screening/geometry evidence.  SciPy DOP853 residual values remain the
authority; JAX x64/Diffrax is used only for tangent/Jacobian information through
the already-audited hybrid continuation path.  Any retained fold still needs an
independent BigFloat nondegeneracy check before release.
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
from threebody_atlas.hybrid_critical import trace_hybrid_critical
from threebody_atlas.liao_family import FamilyPoint

MODE = "minus_one"


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


def orientation(row: dict[str, str]) -> str:
    return f'{row["left_label"]}->{row["right_label"]}'


def select_secondary_pair(
    roots: list[tuple[str, dict[str, str], LocalizedCriticalPoint]],
) -> tuple[
    tuple[str, dict[str, str], LocalizedCriticalPoint],
    tuple[str, dict[str, str], LocalizedCriticalPoint],
]:
    """Select the first adjacent stable interval, never the slice extremes."""
    pairs = [
        (lower, upper)
        for lower, upper in zip(roots, roots[1:], strict=False)
        if lower[0] == "U->S" and upper[0] == "S->U"
    ]
    if not pairs:
        raise RuntimeError(
            "expected an adjacent U->S / S->U secondary pair; "
            f"found {[item[0] for item in roots]}"
        )
    return pairs[0]


def localize_minus(row: dict[str, str]) -> LocalizedCriticalPoint:
    left = family_point(row, "left")
    right = family_point(row, "right")
    stable = left if row["left_label"] == "S" else right
    unstable = left if row["left_label"] == "U" else right
    return localize_critical_point(
        stable,
        unstable,
        event_mode=MODE,
        m2_tolerance=2e-9,
        event_tolerance=2e-8,
        max_iterations=40,
        max_closure=1e-7,
    )


def record(p: LocalizedCriticalPoint) -> dict[str, Any]:
    q = p.sample.point
    f = p.sample.floquet
    return {
        "masses": [float(x) for x in q.masses],
        "x1": float(q.x1),
        "v1": float(q.v1),
        "v2": float(q.v2),
        "period": float(q.period),
        "closure": float(q.residual_norm),
        "event": float(p.event_value),
        "alpha": float(f.alpha),
        "beta": float(f.beta),
        "discriminant": float(f.discriminant),
        "trace_roots": [[float(z.real), float(z.imag)] for z in f.trace_roots],
    }


def fixed_m1_root(
    anchor: LocalizedCriticalPoint,
    target_m1: float,
    *,
    max_m2_jump: float,
    max_nfev: int = 70,
) -> LocalizedCriticalPoint:
    """Correct closure + G- at fixed m1, with m2 and Li chart free."""
    q = anchor.sample.point
    m3 = float(q.masses[2])
    x0 = np.asarray([q.x1, q.v1, q.v2, q.period, q.masses[1]], dtype=float)
    floors = np.asarray([0.05, 0.5, 0.1, 1.0, 0.1], dtype=float)
    scales = np.maximum(np.abs(x0), floors)

    def residual(u: np.ndarray) -> np.ndarray:
        y = np.asarray([u[0], u[1], u[2], u[3], target_m1, u[4]], dtype=float)
        closure, floquet = _flow_for_vector(y, m3=m3, rtol=3e-10, atol=3e-12)
        return np.concatenate((closure / 1e-6, [event_value(floquet, MODE) / 2e-4]))

    lower = np.asarray([-2.0, -10.0, -10.0, 0.1, 0.5], dtype=float)
    upper = np.asarray([2.0, 10.0, 10.0, 20.0, 1.5], dtype=float)
    fit = least_squares(
        residual,
        x0,
        method="trf",
        bounds=(lower, upper),
        x_scale=scales,
        xtol=2e-11,
        ftol=2e-11,
        gtol=2e-11,
        max_nfev=max_nfev,
    )
    y = np.asarray([fit.x[0], fit.x[1], fit.x[2], fit.x[3], target_m1, fit.x[4]], dtype=float)
    closure, floquet = _flow_for_vector(y, m3=m3, rtol=3e-10, atol=3e-12)
    closure_norm = float(np.linalg.norm(closure))
    critical = float(event_value(floquet, MODE))
    m2_jump = abs(float(fit.x[4]) - float(q.masses[1]))
    if not fit.success:
        raise RuntimeError(f"fixed-m1 correction failed: {fit.message}")
    if closure_norm > 2e-7 or abs(critical) > 2e-6:
        raise RuntimeError(
            f"fixed-m1 correction missed gates: closure={closure_norm:.3e}, event={critical:.3e}"
        )
    if m2_jump > max_m2_jump:
        raise RuntimeError(f"fixed-m1 correction jumped branches in m2: {m2_jump:.3e}")

    point = FamilyPoint(
        masses=(float(target_m1), float(fit.x[4]), m3),
        x1=float(fit.x[0]),
        v1=float(fit.x[1]),
        v2=float(fit.x[2]),
        period=float(fit.x[3]),
        residual_norm=closure_norm,
        nfev=int(fit.nfev),
        success=True,
    )
    sample = BoundarySample(point, floquet, stability_score(floquet))
    return LocalizedCriticalPoint(sample, MODE, critical, float("nan"))


def descend_branch(
    seed: LocalizedCriticalPoint,
    *,
    m1_step: float,
    max_steps: int,
    max_m2_jump: float,
) -> tuple[list[LocalizedCriticalPoint], str]:
    points = [seed]
    reason = "requested_steps_completed"
    current = seed
    for _ in range(max_steps):
        target_m1 = float(current.sample.point.masses[0]) - m1_step
        try:
            nxt = fixed_m1_root(current, target_m1, max_m2_jump=max_m2_jump)
        except Exception as exc:
            reason = f"fixed-m1 branch ended: {type(exc).__name__}: {exc}"
            break
        points.append(nxt)
        current = nxt
    return points, reason


def chart_vector(p: LocalizedCriticalPoint | Any) -> np.ndarray:
    q = p.sample.point
    return np.asarray([q.x1, q.v1, q.v2, q.period, q.masses[0], q.masses[1]], dtype=float)


def normalized_gap(a: LocalizedCriticalPoint | Any, b: LocalizedCriticalPoint | Any) -> float:
    va, vb = chart_vector(a), chart_vector(b)
    floors = np.asarray([0.05, 0.5, 0.1, 1.0, 0.1, 0.1], dtype=float)
    scales = np.maximum(np.maximum(np.abs(va), np.abs(vb)), floors)
    return float(np.linalg.norm((va - vb) / scales))


def mass_gap(a: LocalizedCriticalPoint | Any, b: LocalizedCriticalPoint | Any) -> float:
    ma = np.asarray(a.sample.point.masses[:2], dtype=float)
    mb = np.asarray(b.sample.point.masses[:2], dtype=float)
    return float(np.linalg.norm(ma - mb))


def closest_gap(trace_points: list[Any], branch: list[LocalizedCriticalPoint]) -> dict[str, Any]:
    best: tuple[float, float, int, int] | None = None
    for i, a in enumerate(trace_points):
        for j, b in enumerate(branch):
            mg = mass_gap(a, b)
            ng = normalized_gap(a, b)
            key = mg + 0.05 * ng
            if best is None or key < best[0]:
                best = (key, mg, ng, i, j)  # type: ignore[assignment]
    if best is None:
        return {"mass_gap": None, "normalized_chart_gap": None}
    _key, mg, ng, i, j = best
    return {
        "mass_gap": float(mg),
        "normalized_chart_gap": float(ng),
        "trace_index": int(i),
        "branch_index": int(j),
    }


def serialize_trace(trace, diagnostics) -> dict[str, Any]:
    points: list[dict[str, Any]] = []
    for i, p in enumerate(trace.points):
        q, f = p.sample.point, p.sample.floquet
        diag = diagnostics[i]
        points.append(
            {
                "masses": [float(x) for x in q.masses],
                "x1": float(q.x1),
                "v1": float(q.v1),
                "v2": float(q.v2),
                "period": float(q.period),
                "closure": float(q.residual_norm),
                "event": float(p.event_value),
                "scaled_tangent": [float(x) for x in p.tangent_scaled],
                "dm1_scaled": float(p.tangent_scaled[4]),
                "dm2_scaled": float(p.tangent_scaled[5]),
                "null_residual": float(diag.null_residual),
                "spectral_gap": float(diag.spectral_gap),
                "alpha": float(f.alpha),
                "beta": float(f.beta),
                "discriminant": float(f.discriminant),
            }
        )
    fold_pairs: list[dict[str, Any]] = []
    for i in range(1, len(points)):
        a, b = points[i - 1], points[i]
        ta, tb = a["dm1_scaled"], b["dm1_scaled"]
        if ta == 0.0 or tb == 0.0 or ta * tb < 0.0:
            transverse = min(abs(a["dm2_scaled"]), abs(b["dm2_scaled"]))
            fold_pairs.append(
                {
                    "indices": [i - 1, i],
                    "dm1_scaled": [ta, tb],
                    "minimum_abs_dm2_scaled": transverse,
                    "generic_fold_screen": bool(transverse >= 1e-4),
                }
            )
    return {"points": points, "stopped_reason": trace.stopped_reason, "fold_pairs": fold_pairs}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("brackets_tsv")
    parser.add_argument("output")
    parser.add_argument("--seed-m1", type=float, default=0.996)
    parser.add_argument("--orientation-m1", type=float, default=0.997)
    parser.add_argument("--m1-step", type=float, default=2e-5)
    parser.add_argument("--descent-steps", type=int, default=20)
    parser.add_argument("--arclength-steps", type=int, default=40)
    parser.add_argument("--arclength-step", type=float, default=5e-4)
    parser.add_argument("--max-m2-jump", type=float, default=0.004)
    parser.add_argument("--reconnect-mass-gap", type=float, default=1.5e-3)
    parser.add_argument("--reconnect-chart-gap", type=float, default=2e-2)
    args = parser.parse_args()

    with Path(args.brackets_tsv).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    for cell_id, row in enumerate(rows):
        row["_cell_id"] = str(cell_id)

    def minus_roots_at(target_m1: float) -> list[tuple[str, dict[str, str], LocalizedCriticalPoint]]:
        found: list[tuple[str, dict[str, str], LocalizedCriticalPoint]] = []
        for row in rows:
            if abs(float(row["m1"]) - target_m1) > 5e-10:
                continue
            left, right = evaluate(family_point(row, "left")), evaluate(family_point(row, "right"))
            va, vb = event_value(left.floquet, MODE), event_value(right.floquet, MODE)
            if va == 0.0 or vb == 0.0 or va * vb < 0.0:
                found.append((orientation(row), row, localize_minus(row)))
        found.sort(key=lambda item: item[2].sample.point.masses[1])
        return found

    seed_rows = minus_roots_at(args.seed_m1)
    orientation_rows = minus_roots_at(args.orientation_m1)
    lower_seed_row, upper_seed_row = select_secondary_pair(seed_rows)
    lower_seed = lower_seed_row[2]
    upper_seed = upper_seed_row[2]

    def nearby_orientation_seed(
        kind: str, seed: LocalizedCriticalPoint
    ) -> LocalizedCriticalPoint | None:
        candidates = [item[2] for item in orientation_rows if item[0] == kind]
        if not candidates:
            return None
        nearest = min(candidates, key=lambda point: mass_gap(point, seed))
        return nearest if mass_gap(nearest, seed) <= 0.02 else None

    lower_orient = nearby_orientation_seed("U->S", lower_seed)
    upper_orient = nearby_orientation_seed("S->U", upper_seed)

    lower_branch, lower_reason = descend_branch(
        lower_seed,
        m1_step=args.m1_step,
        max_steps=args.descent_steps,
        max_m2_jump=args.max_m2_jump,
    )
    upper_branch, upper_reason = descend_branch(
        upper_seed,
        m1_step=args.m1_step,
        max_steps=args.descent_steps,
        max_m2_jump=args.max_m2_jump,
    )
    if len(lower_branch) < 2 and lower_orient is not None:
        lower_branch = [lower_orient, lower_seed]
        lower_reason = "used orientation-m1 seed because fixed-m1 descent was short"
    if len(upper_branch) < 2 and upper_orient is not None:
        upper_branch = [upper_orient, upper_seed]
        upper_reason = "used orientation-m1 seed because fixed-m1 descent was short"

    # Start each hybrid trace from the last two branch points so its orientation
    # is explicitly toward the suspected fold.  If descent failed, the published
    # 0.997 -> 0.996 pair points decreasing m1 into the birth neighborhood.
    trace_error = None
    lower_serialized: dict[str, Any] = {"points": [], "stopped_reason": "not_started", "fold_pairs": []}
    upper_serialized: dict[str, Any] = {"points": [], "stopped_reason": "not_started", "fold_pairs": []}
    lower_to_upper = {"mass_gap": None, "normalized_chart_gap": None}
    upper_to_lower = {"mass_gap": None, "normalized_chart_gap": None}
    lower_fold = False
    upper_fold = False
    reconnect = False
    if len(lower_branch) < 2 or len(upper_branch) < 2:
        trace_error = (
            f"insufficient points to start hybrid traces: lower={len(lower_branch)} "
            f"upper={len(upper_branch)} lower_reason={lower_reason} upper_reason={upper_reason}"
        )
    else:
        try:
            lower_trace, lower_diag = trace_hybrid_critical(
                lower_branch[-2],
                lower_branch[-1],
                steps=args.arclength_steps,
                normalized_step=args.arclength_step,
            )
            upper_trace, upper_diag = trace_hybrid_critical(
                upper_branch[-2],
                upper_branch[-1],
                steps=args.arclength_steps,
                normalized_step=args.arclength_step,
            )
            lower_serialized = serialize_trace(lower_trace, lower_diag)
            upper_serialized = serialize_trace(upper_trace, upper_diag)
            lower_to_upper = closest_gap(list(lower_trace.points), upper_branch)
            upper_to_lower = closest_gap(list(upper_trace.points), lower_branch)
            lower_fold = any(x["generic_fold_screen"] for x in lower_serialized["fold_pairs"])
            upper_fold = any(x["generic_fold_screen"] for x in upper_serialized["fold_pairs"])
            reconnect = bool(
                (
                    lower_to_upper["mass_gap"] is not None
                    and lower_to_upper["mass_gap"] <= args.reconnect_mass_gap
                    and lower_to_upper["normalized_chart_gap"] <= args.reconnect_chart_gap
                )
                or (
                    upper_to_lower["mass_gap"] is not None
                    and upper_to_lower["mass_gap"] <= args.reconnect_mass_gap
                    and upper_to_lower["normalized_chart_gap"] <= args.reconnect_chart_gap
                )
            )
        except Exception as exc:
            trace_error = f"{type(exc).__name__}: {exc}"
    passed = bool(trace_error is None and (lower_fold or upper_fold) and reconnect)

    output = {
        "claim_status": "event-specific float64/JAX-derivative geometry screen; independent BigFloat fold/nondegeneracy verification required",
        "event_mode": MODE,
        "seed_m1": args.seed_m1,
        "orientation_m1": args.orientation_m1,
        "seed_orientations": [x[0] for x in seed_rows],
        "orientation_seed_count": len(orientation_rows),
        "trace_error": trace_error,
        "localized_seeds": [
            {
                **record(lower_seed),
                "source_cell_id": int(lower_seed_row[1]["_cell_id"]),
                "orientation": lower_seed_row[0],
            },
            {
                **record(upper_seed),
                "source_cell_id": int(upper_seed_row[1]["_cell_id"]),
                "orientation": upper_seed_row[0],
            },
        ],
        "edge_endpoint_bindings": [
            {
                "cell_id": int(lower_seed_row[1]["_cell_id"]),
                "side": "start",
                "mechanism": MODE,
                "orientation": lower_seed_row[0],
            },
            {
                "cell_id": int(upper_seed_row[1]["_cell_id"]),
                "side": "start",
                "mechanism": MODE,
                "orientation": upper_seed_row[0],
            },
        ],
        "fixed_m1_descent": {
            "m1_step": args.m1_step,
            "lower": [record(x) for x in lower_branch],
            "upper": [record(x) for x in upper_branch],
            "lower_stopped_reason": lower_reason,
            "upper_stopped_reason": upper_reason,
        },
        "lower_to_upper_trace": lower_serialized,
        "upper_to_lower_trace": upper_serialized,
        "lower_to_upper_closest_gap": lower_to_upper,
        "upper_to_lower_closest_gap": upper_to_lower,
        "generic_m1_fold_screen": bool(lower_fold or upper_fold),
        "opposite_branch_reconnection_screen": reconnect,
        "passed": passed,
        "interpretation": (
            "The two secondary minus-one boundaries screen as opposite sides of one m1-projection fold."
            if passed
            else "The declared bidirectional fold/reconnection screen did not pass; do not retain the fold label from historical finite differences alone."
        ),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: output[k] for k in ("generic_m1_fold_screen", "opposite_branch_reconnection_screen", "passed", "interpretation")}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
