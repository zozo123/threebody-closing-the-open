#!/usr/bin/env python3
"""Continuously test whether sweep components 12 and 11 are one G+ wall.

The committed event-sign sweep contains one isolated plus-one sample
(component 11) immediately to the left of a two-sample component 12.  Raster
clustering is not a topological witness.  This script performs the decisive
experiment instead:

1. re-correct the two component-12 roots on the strict periodic sheet;
2. launch *backwards* from component 12 using the local critical-set null
   direction d[closure,G+]/d(x1,v1,v2,T,m1,m2), not a mass-plane join rule;
3. pseudo-arclength continue through the projection turn with SciPy residuals
   and JAX/Diffrax derivatives;
4. independently repeat the traversal with the nested mass-plane corrector;
5. ask whether the resulting continuous traces pass through the independently
   re-corrected component-11 root with matching local tangent.

No graph endpoint is invented here.  A pass is a continuous incidence witness;
a failure remains an explicit contradiction/unresolved result.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from threebody_atlas.critical_geometry import continuation_scales, critical_tangent
from threebody_atlas.critical_manifold import (
    LocalizedCriticalPoint,
    _polish_event_root,
    _precise_evaluate,
    event_value,
)
from threebody_atlas.critical_massplane import advance_massplane_critical
from threebody_atlas.hybrid_critical import advance_hybrid_critical
from threebody_atlas.jax_diffrax import (
    adaptive_closure_and_jacobian,
    adaptive_event_and_gradient,
    require_accelerated_x64,
)
from threebody_atlas.liao_family import FamilyPoint, correct_family_point

EVENT_GATE = 2e-8
CLOSURE_GATE = 1e-7
TARGET_COMPONENT = 11
SOURCE_COMPONENT = 12
TARGET_CELL = 10130
SOURCE_CELLS = (10131, 10132)
DECLARED_DOMAIN = ((0.8, 1.1), (0.7, 1.2))


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} JSON root must be an object")
    return payload


def _root_by_cell(payload: dict[str, Any], cell_id: int) -> dict[str, Any]:
    rows = [row for row in payload.get("roots", []) if int(row.get("cell_id", -1)) == cell_id]
    if len(rows) != 1:
        raise RuntimeError(f"expected exactly one supplemental root for cell {cell_id}, got {len(rows)}")
    row = rows[0]
    if not row.get("passed") or row.get("status") != "ok":
        raise RuntimeError(f"supplemental root {cell_id} is not certified")
    return row


def _corrected_seed(row: dict[str, Any]) -> LocalizedCriticalPoint:
    masses = tuple(float(x) for x in row["masses"])
    guess = tuple(float(row[key]) for key in ("x1", "v1", "v2", "period"))
    corrected = correct_family_point(masses, guess, max_nfev=80)
    if not corrected.success or corrected.residual_norm > CLOSURE_GATE:
        raise RuntimeError(
            f"cell {row['cell_id']} failed periodic re-correction: "
            f"closure={corrected.residual_norm:.3e}"
        )
    mode = str(row["event_mode"])
    sample = _precise_evaluate(corrected)
    value = event_value(sample.floquet, mode)
    if abs(value) > EVENT_GATE:
        # The sweep roots are already certified, but a different runner can move
        # a near-gate Float64 event slightly.  Re-polish only along the original
        # local m2 neighbourhood; never relax the frozen event gate.
        m2 = masses[1]
        polished = _polish_event_root(
            sample,
            mode,
            event_tolerance=EVENT_GATE,
            max_closure=CLOSURE_GATE,
            max_steps=8,
            m2_bounds=(m2 - 1e-3, m2 + 1e-3),
            precise=True,
        )
        sample = polished.sample
        value = polished.event_value
    if sample.point.residual_norm > CLOSURE_GATE or abs(value) > EVENT_GATE:
        raise RuntimeError(
            f"cell {row['cell_id']} misses frozen gates after re-correction: "
            f"closure={sample.point.residual_norm:.3e}, event={value:.3e}"
        )
    return LocalizedCriticalPoint(sample, mode, float(value), 0.0)


def _vector(point: LocalizedCriticalPoint | Any) -> np.ndarray:
    return np.asarray(point.vector, dtype=float)


def _local_tangent(point: LocalizedCriticalPoint | Any, reference: np.ndarray | None = None):
    y = _vector(point)
    m3 = float(point.sample.point.masses[2])
    _closure, closure_jac = adaptive_closure_and_jacobian(
        y,
        m3=m3,
        rtol=1e-10,
        atol=1e-12,
        max_steps=1 << 18,
    )
    _event, event_grad = adaptive_event_and_gradient(
        y,
        point.event_mode,
        m3=m3,
        rtol=5e-10,
        atol=5e-12,
        max_steps=1 << 18,
    )
    return critical_tangent(
        np.vstack((closure_jac, event_grad[None, :])),
        scales=continuation_scales(y),
        reference=reference,
    )


def _serialize_point(point: LocalizedCriticalPoint | Any) -> dict[str, Any]:
    p = point.sample.point
    out: dict[str, Any] = {
        "masses": [float(x) for x in p.masses],
        "x1": float(p.x1),
        "v1": float(p.v1),
        "v2": float(p.v2),
        "period": float(p.period),
        "closure": float(p.residual_norm),
        "event_mode": str(point.event_mode),
        "event": float(point.event_value),
    }
    if hasattr(point, "normalized_step"):
        out["normalized_step"] = float(point.normalized_step)
        out["arclength_residual"] = float(point.arclength_residual)
        out["nfev"] = int(point.nfev)
    if hasattr(point, "step"):
        out["massplane_step"] = float(point.step)
        out["arclength_residual"] = float(point.arclength_residual)
        out["outer_nfev"] = int(point.outer_nfev)
    return out


def _segment_distance_scaled(a: np.ndarray, b: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    scale = continuation_scales(target)
    aa = (a - target) / scale
    bb = (b - target) / scale
    direction = bb - aa
    denom = float(np.dot(direction, direction))
    if denom == 0.0:
        return float(np.linalg.norm(aa)), 0.0
    t = float(np.clip(-np.dot(aa, direction) / denom, 0.0, 1.0))
    miss = aa + t * direction
    return float(np.linalg.norm(miss)), t


def _segment_distance_mass(a: np.ndarray, b: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    direction = b - a
    denom = float(np.dot(direction, direction))
    if denom == 0.0:
        return float(np.linalg.norm(a - target)), 0.0
    t = float(np.clip(np.dot(target - a, direction) / denom, 0.0, 1.0))
    miss = a + t * direction - target
    return float(np.linalg.norm(miss)), t


def _trace_augmented(
    first: LocalizedCriticalPoint,
    second: LocalizedCriticalPoint,
    target: LocalizedCriticalPoint,
    *,
    steps: int,
    requested_step: float,
) -> dict[str, Any]:
    previous: LocalizedCriticalPoint | Any = first
    current: LocalizedCriticalPoint | Any = second
    target_y = _vector(target)
    target_tangent = _local_tangent(target, reference=_vector(second) - _vector(first))
    accepted: list[Any] = []
    diagnostics: list[dict[str, Any]] = []
    step = float(requested_step)
    best = {"segment_miss_scaled": float("inf"), "segment_index": None, "segment_fraction": None}
    reason = "requested_steps_completed"

    for index in range(steps):
        trial = step
        point = None
        diag = None
        last_error: Exception | None = None
        for _retry in range(7):
            try:
                point, diag = advance_hybrid_critical(
                    previous,
                    current,
                    normalized_step=trial,
                    max_closure=CLOSURE_GATE,
                    max_event=EVENT_GATE,
                    max_arc=max(5e-6, 0.05 * trial),
                    max_nfev=55,
                    screening_rtol=5e-12,
                    screening_atol=5e-14,
                )
                break
            except (RuntimeError, ValueError, FloatingPointError) as exc:
                last_error = exc
                trial *= 0.5
                if trial < 6.25e-5:
                    break
        if point is None or diag is None:
            reason = f"augmented_corrector_failed: {last_error}"
            break

        a = _vector(current)
        b = _vector(point)
        miss, fraction = _segment_distance_scaled(a, b, target_y)
        if miss < float(best["segment_miss_scaled"]):
            best = {
                "segment_miss_scaled": miss,
                "segment_index": index,
                "segment_fraction": fraction,
                "segment_start": _serialize_point(current),
                "segment_end": _serialize_point(point),
            }

        accepted.append(point)
        diagnostics.append(
            {
                "null_residual": float(diag.null_residual),
                "spectral_gap": float(diag.spectral_gap),
                "singular_values": [float(x) for x in diag.singular_values],
            }
        )
        previous, current = current, point
        step = min(requested_step, trial * 1.25)

        m1, m2, _m3 = current.sample.point.masses
        if not (DECLARED_DOMAIN[0][0] <= m1 <= DECLARED_DOMAIN[0][1]) or not (
            DECLARED_DOMAIN[1][0] <= m2 <= DECLARED_DOMAIN[1][1]
        ):
            reason = "left_declared_mass_domain"
            break
        # Once the target lies tightly on a traced segment, continue one more
        # accepted step so local tangent comparison is not a one-sided accident.
        if miss <= 1.5e-3 and index >= 1:
            reason = "target_neighbourhood_crossed"
            break

    if accepted:
        closest = min(accepted, key=lambda p: float(np.linalg.norm((_vector(p) - target_y) / continuation_scales(target_y))))
        closest_distance = float(
            np.linalg.norm((_vector(closest) - target_y) / continuation_scales(target_y))
        )
        closest_tangent = _local_tangent(closest, reference=_vector(closest) - _vector(previous))
        tangent_cosine = abs(float(np.dot(closest_tangent.physical, target_tangent.physical)))
        closest_tangent_diag = {
            "distance_scaled": closest_distance,
            "tangent_abs_cosine": tangent_cosine,
            "target_null_residual": float(target_tangent.null_residual),
            "target_spectral_gap": float(target_tangent.spectral_gap),
            "trace_null_residual": float(closest_tangent.null_residual),
            "trace_spectral_gap": float(closest_tangent.spectral_gap),
        }
    else:
        closest_tangent_diag = {
            "distance_scaled": float("inf"),
            "tangent_abs_cosine": 0.0,
        }

    passed = bool(
        float(best["segment_miss_scaled"]) <= 1.5e-3
        and float(closest_tangent_diag["tangent_abs_cosine"]) >= 0.98
    )
    return {
        "method": "full-state augmented pseudo-arclength; SciPy values + JAX/Diffrax local null tangent",
        "strict_gates": {"max_abs_event": EVENT_GATE, "max_closure": CLOSURE_GATE},
        "points": [_serialize_point(p) for p in accepted],
        "diagnostics": diagnostics,
        "stopped_reason": reason,
        "best_target_segment": best,
        "closest_target_tangent": closest_tangent_diag,
        "continuous_target_incidence": passed,
    }


def _trace_nested(
    first: LocalizedCriticalPoint,
    second: LocalizedCriticalPoint,
    target: LocalizedCriticalPoint,
    *,
    steps: int,
    requested_step: float,
) -> dict[str, Any]:
    previous: LocalizedCriticalPoint | Any = first
    current: LocalizedCriticalPoint | Any = second
    target_mass = np.asarray(target.sample.point.masses[:2], dtype=float)
    accepted: list[Any] = []
    step = float(requested_step)
    best = {"segment_miss_mass": float("inf"), "segment_index": None, "segment_fraction": None}
    reason = "requested_steps_completed"

    for index in range(steps):
        trial = step
        point = None
        last_error: Exception | None = None
        for _retry in range(7):
            try:
                point = advance_massplane_critical(
                    previous,
                    current,
                    step=trial,
                    max_closure=CLOSURE_GATE,
                    max_event=EVENT_GATE,
                    max_arc=max(1e-7, 0.02 * trial),
                    max_outer_nfev=28,
                    max_inner_nfev=60,
                )
                break
            except (RuntimeError, ValueError, FloatingPointError) as exc:
                last_error = exc
                trial *= 0.5
                if trial < 3.125e-5:
                    break
        if point is None:
            reason = f"nested_corrector_failed: {last_error}"
            break

        a = np.asarray(current.sample.point.masses[:2], dtype=float)
        b = np.asarray(point.sample.point.masses[:2], dtype=float)
        miss, fraction = _segment_distance_mass(a, b, target_mass)
        if miss < float(best["segment_miss_mass"]):
            best = {
                "segment_miss_mass": miss,
                "segment_index": index,
                "segment_fraction": fraction,
                "segment_start": _serialize_point(current),
                "segment_end": _serialize_point(point),
            }
        accepted.append(point)
        previous, current = current, point
        step = min(requested_step, trial * 1.25)
        if miss <= 2.5e-4 and index >= 1:
            reason = "target_neighbourhood_crossed"
            break

    passed = bool(float(best["segment_miss_mass"]) <= 2.5e-4)
    return {
        "method": "nested mass-plane pseudo-arclength with independent periodic re-correction",
        "strict_gates": {"max_abs_event": EVENT_GATE, "max_closure": CLOSURE_GATE},
        "points": [_serialize_point(p) for p in accepted],
        "stopped_reason": reason,
        "best_target_segment": best,
        "continuous_target_incidence": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("supplemental_roots")
    parser.add_argument("output")
    parser.add_argument("--augmented-steps", type=int, default=40)
    parser.add_argument("--augmented-step", type=float, default=2e-3)
    parser.add_argument("--nested-steps", type=int, default=80)
    parser.add_argument("--nested-step", type=float, default=5e-4)
    args = parser.parse_args()

    require_accelerated_x64()
    payload = _load(Path(args.supplemental_roots))
    rows = {cell: _root_by_cell(payload, cell) for cell in (TARGET_CELL, *SOURCE_CELLS)}
    if int(rows[TARGET_CELL]["sweep_component"]) != TARGET_COMPONENT:
        raise RuntimeError("target cell is no longer sweep component 11")
    if any(int(rows[cell]["sweep_component"]) != SOURCE_COMPONENT for cell in SOURCE_CELLS):
        raise RuntimeError("source cells are no longer sweep component 12")
    if any(str(rows[cell]["event_mode"]) != "plus_one" for cell in rows):
        raise RuntimeError("right-hand wall seeds no longer share the plus_one event")

    target = _corrected_seed(rows[TARGET_CELL])
    # Reverse the two component-12 samples: 10132 -> 10131 points the local
    # continuation back toward smaller arclength, without using component 11 to
    # choose the branch direction.
    first = _corrected_seed(rows[SOURCE_CELLS[1]])
    second = _corrected_seed(rows[SOURCE_CELLS[0]])

    augmented = _trace_augmented(
        first,
        second,
        target,
        steps=args.augmented_steps,
        requested_step=args.augmented_step,
    )
    nested = _trace_nested(
        first,
        second,
        target,
        steps=args.nested_steps,
        requested_step=args.nested_step,
    )
    overall = bool(augmented["continuous_target_incidence"] and nested["continuous_target_incidence"])

    result = {
        "schema": "atlas.v1.continuous-event-witness/1",
        "claim": "right-hand interior plus-one wall: sweep component 12 continuously reconnects to isolated component 11",
        "source": str(args.supplemental_roots),
        "source_cells": list(SOURCE_CELLS),
        "target_cell": TARGET_CELL,
        "source_component": SOURCE_COMPONENT,
        "target_component": TARGET_COMPONENT,
        "frozen_gates": {"maximum_absolute_event": EVENT_GATE, "maximum_periodic_closure": CLOSURE_GATE},
        "seed_component_12": [_serialize_point(first), _serialize_point(second)],
        "independent_target_component_11": _serialize_point(target),
        "augmented_trace": augmented,
        "nested_massplane_trace": nested,
        "continuous_witness_passed": overall,
        "claim_status": (
            "continuous_incidence_witness_passed_two_formulations"
            if overall
            else "unresolved_continuous_incidence_not_established"
        ),
        "interpretation": (
            "A pass joins components 11 and 12 by continuous corrected G+=0 geometry; it does not classify either remote terminus."
        ),
    }
    if not math.isfinite(float(augmented["best_target_segment"]["segment_miss_scaled"])):
        result["continuous_witness_passed"] = False
        result["claim_status"] = "unresolved_augmented_trace_produced_no_target_segment"

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "continuous_witness_passed": result["continuous_witness_passed"],
                "augmented_points": len(augmented["points"]),
                "augmented_miss_scaled": augmented["best_target_segment"]["segment_miss_scaled"],
                "augmented_tangent_cosine": augmented["closest_target_tangent"]["tangent_abs_cosine"],
                "nested_points": len(nested["points"]),
                "nested_miss_mass": nested["best_target_segment"]["segment_miss_mass"],
                "augmented_stop": augmented["stopped_reason"],
                "nested_stop": nested["stopped_reason"],
            },
            indent=2,
        )
    )
    raise SystemExit(0 if result["continuous_witness_passed"] else 3)


if __name__ == "__main__":
    main()
