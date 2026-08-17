#!/usr/bin/env python3
"""Build continuous witnesses for the label-invisible Floquet critical set.

This is the continuation campaign required by #192, not another raster scan.

The long-range predictor is derived from the *total* event gradient along the
corrected periodic family.  ``mass_sensitivity`` integrates the state,
monodromy, mass sensitivities and monodromy sensitivities, includes the
mass-dependence of the shooting chart and implicit periodic-orbit correction,
and returns dG/d(m1,m2).  Therefore

    tau_m = (dG/dm2, -dG/dm1) / ||dG/dm||

is a coordinate-free local tangent of the critical curve in the mass plane.
Each step predicts along tau_m, corrects the periodic orbit, and corrects only
along the event-gradient normal, with a safeguarded scalar sign-bracket fallback
at the float64 event floor.  This passes through m1- or m2-projection folds
without treating either mass as a graph coordinate.

At the nine contradiction seeds we independently compare that variational
mass-plane tangent with the mass projection of the six-dimensional JAX/Diffrax
null tangent of d[closure,G]/d(x1,v1,v2,T,m1,m2).  Residual values and acceptance
remain the canonical SciPy path; JAX supplies derivatives only.

Branches are launched from opposite pseudo-arclength germs that already straddle
independently reproduced mixed organizers.  The campaign covers every
multi-point supplemental sweep component (0, 1, 3, 4, 10, and 12), including
the two branches whose sampled endpoints happened to lie near a domain face.
Legitimate stops are another canonically bound germ or a declared-domain face.
Newton failure is never a scientific terminus.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from threebody_atlas.boundary import BoundarySample
from threebody_atlas.critical_geometry import continuation_scales, critical_tangent
from threebody_atlas.critical_manifold import (
    LocalizedCriticalPoint,
    _polish_event_root,
    _precise_bracket_search,
    _precise_evaluate,
    event_value,
)
from threebody_atlas.jax_diffrax import (
    adaptive_closure_and_jacobian,
    adaptive_event_and_gradient,
    require_accelerated_x64,
)
from threebody_atlas.liao_family import FamilyPoint, correct_family_point
from threebody_atlas.mass_sensitivity import mass_sensitivity

EVENT_GATE = 2e-8
CLOSURE_GATE = 1e-7
DECLARED_DOMAIN = {"m1": (0.8, 1.1), "m2": (0.7, 1.2)}
MIN_EVENT_GRADIENT = 1e-5
MIN_CLOSURE_SINGULAR = 1e-9
# The independent JAX/Diffrax tangent is evidentiary only when the smallest
# direction is both a genuine numerical null direction and well separated from
# the next singular direction. These do not relax the event/closure gates.
MAX_JAX_RELATIVE_NULL_RESIDUAL = 1e-6
MIN_JAX_SPECTRAL_GAP = 100.0


@dataclass(frozen=True)
class StrictPoint:
    localized: LocalizedCriticalPoint
    source: str
    source_id: str

    @property
    def vector(self) -> np.ndarray:
        return np.asarray(self.localized.vector, dtype=float)

    @property
    def masses2(self) -> np.ndarray:
        return np.asarray(self.localized.sample.point.masses[:2], dtype=float)


@dataclass(frozen=True)
class VariationalPoint:
    localized: LocalizedCriticalPoint
    gradient: np.ndarray
    tangent_mass: np.ndarray
    closure_singular_values: np.ndarray
    closure_relative_residual: float
    step: float
    corrector_iterations: int

    @property
    def vector(self) -> np.ndarray:
        return np.asarray(self.localized.vector, dtype=float)

    @property
    def masses2(self) -> np.ndarray:
        return np.asarray(self.localized.sample.point.masses[:2], dtype=float)


@dataclass(frozen=True)
class _NormalEvaluation:
    lam: float
    value: float
    sample: BoundarySample


def _tightest_sign_bracket(
    evaluations: list[_NormalEvaluation],
) -> tuple[_NormalEvaluation, _NormalEvaluation] | None:
    """Return the narrowest sampled interval that brackets a scalar zero."""
    ordered = sorted(evaluations, key=lambda item: item.lam)
    brackets = [
        (left, right)
        for left, right in zip(ordered, ordered[1:], strict=False)
        if left.value * right.value < 0.0
    ]
    if not brackets:
        return None
    return min(brackets, key=lambda pair: pair[1].lam - pair[0].lam)


def _bracket_trial_lambda(
    left: _NormalEvaluation,
    right: _NormalEvaluation,
) -> float:
    """Secant trial inside a sign bracket, with bisection as a safe fallback."""
    width = float(right.lam - left.lam)
    denominator = float(right.value - left.value)
    if not math.isfinite(denominator) or denominator == 0.0:
        return float(left.lam + 0.5 * width)
    trial = float((left.lam * right.value - right.lam * left.value) / denominator)
    margin = max(1e-15, 1e-8 * abs(width))
    if not math.isfinite(trial) or trial <= left.lam + margin or trial >= right.lam - margin:
        return float(left.lam + 0.5 * width)
    return trial


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path}: JSON root must be an object")
    return payload


def _sanitize_json(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, list):
        return [_sanitize_json(item) for item in value]
    if isinstance(value, dict):
        return {key: _sanitize_json(item) for key, item in value.items()}
    return value


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_sanitize_json(payload), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _family_from_record(record: dict[str, Any]) -> FamilyPoint:
    masses = tuple(float(x) for x in record["masses"])
    guess = tuple(float(record[key]) for key in ("x1", "v1", "v2", "period"))
    return FamilyPoint(
        masses=masses,
        x1=guess[0],
        v1=guess[1],
        v2=guess[2],
        period=guess[3],
        residual_norm=float(record.get("closure") or record.get("periodic_closure") or record.get("shooting_residual") or 0.0),
        nfev=0,
        success=True,
    )


def _strict_localize(
    record: dict[str, Any],
    mode: str,
    *,
    source: str,
    source_id: str,
    m2_bounds: tuple[float, float] | None = None,
) -> StrictPoint:
    seed = _family_from_record(record)
    guess = (seed.x1, seed.v1, seed.v2, seed.period)
    corrected = correct_family_point(seed.masses, guess, max_nfev=90)
    if not corrected.success or corrected.residual_norm > CLOSURE_GATE:
        raise RuntimeError(
            f"{source_id}: periodic re-correction failed: closure={corrected.residual_norm:.3e}"
        )
    sample = _precise_evaluate(corrected)
    value = float(event_value(sample.floquet, mode))
    if abs(value) > EVENT_GATE:
        m2 = float(corrected.masses[1])
        bounds = m2_bounds or (m2 - 1.25e-3, m2 + 1.25e-3)
        polished = _polish_event_root(
            sample,
            mode,
            event_tolerance=EVENT_GATE,
            max_closure=CLOSURE_GATE,
            max_steps=8,
            m2_bounds=bounds,
            precise=True,
        )
        sample = polished.sample
        value = float(polished.event_value)
        # A point seed near the float64 event floor can move by a few 1e-8
        # across platforms.  When the evidence supplies the original signed
        # cell, reopen that cell with tight evaluations rather than rejecting a
        # regular zero or relaxing the frozen gate.
        if abs(value) > EVENT_GATE and m2_bounds is not None:
            lo, hi = sorted(float(bound) for bound in m2_bounds)
            guess = (corrected.x1, corrected.v1, corrected.v2, corrected.period)
            left_point = correct_family_point(
                (float(corrected.masses[0]), lo, float(corrected.masses[2])),
                guess,
                max_nfev=90,
            )
            right_point = correct_family_point(
                (float(corrected.masses[0]), hi, float(corrected.masses[2])),
                guess,
                max_nfev=90,
            )
            if (
                left_point.success
                and right_point.success
                and left_point.residual_norm <= CLOSURE_GATE
                and right_point.residual_norm <= CLOSURE_GATE
            ):
                left = _precise_evaluate(left_point)
                right = _precise_evaluate(right_point)
                left_value = float(event_value(left.floquet, mode))
                right_value = float(event_value(right.floquet, mode))
                if left_value * right_value <= 0.0:
                    recovered = _precise_bracket_search(
                        left,
                        right,
                        mode,
                        seed=sample,
                        event_tolerance=EVENT_GATE,
                        max_closure=CLOSURE_GATE,
                        # Twelve safeguarded iterations were enough on macOS,
                        # but Ubuntu's tight-Floquet evaluation moves this
                        # cancellation-sensitive G- zero by about 1e-10 in m2.
                        # Keep refining the original signed cell instead of
                        # treating that platform shift as a failed seed (or,
                        # worse, relaxing the frozen event gate).
                        max_steps=32,
                    )
                    if abs(recovered.event_value) < abs(value):
                        sample = recovered.sample
                        value = float(recovered.event_value)
    if sample.point.residual_norm > CLOSURE_GATE or abs(value) > EVENT_GATE:
        raise RuntimeError(
            f"{source_id}: strict frozen gates failed after re-correction: "
            f"closure={sample.point.residual_norm:.3e} event={value:.3e}"
        )
    return StrictPoint(
        LocalizedCriticalPoint(sample, mode, value, 0.0),
        source=source,
        source_id=source_id,
    )


def _germ_point(path: Path, mode: str, direction: str) -> StrictPoint:
    payload = _load(path)
    rows = [
        row
        for row in payload.get("germs", [])
        if str(row.get("event_mode")) == mode and str(row.get("direction")) == direction
    ]
    if len(rows) != 1:
        raise RuntimeError(f"{path}: expected one {mode}/{direction} germ, got {len(rows)}")
    row = rows[0]
    if row.get("status") != "traced" or not row.get("canonical_bound") or not row.get("canonical_bracketed"):
        raise RuntimeError(f"{path}: germ {mode}/{direction} is not canonically bound")
    # The stored germ is a pseudo-arclength point, not necessarily at the m1 of
    # its source raster cell.  Reopen a local fixed-m1 interval around the germ
    # itself when tight Floquet evaluation moves it across the cancellation-
    # sensitive float64 gate.  Using the source cell's m2 bracket here would
    # pull the germ back toward the catalog root and erase its signed direction.
    germ_m2 = float(row["masses"][1])
    return _strict_localize(
        row,
        mode,
        source=str(path),
        source_id=f"{payload.get('mixed_node')}:{mode}:{direction}",
        m2_bounds=(germ_m2 - 1.25e-3, germ_m2 + 1.25e-3),
    )


def _strict_from_certification(
    cert: dict[str, Any],
    *,
    source_id: str,
    m2_bounds: tuple[float, float] | None = None,
) -> StrictPoint:
    record = {
        "masses": cert["masses"],
        "x1": cert["x1"],
        "v1": cert["v1"],
        "v2": cert["v2"],
        "period": cert["period"],
        "closure": cert.get("closure"),
    }
    return _strict_localize(
        record,
        str(cert["event_mode"]),
        source="research/evidence/V1_BRACKET_CRITERION_COMPARISON_2026-08-16.json",
        source_id=source_id,
        m2_bounds=m2_bounds,
    )


def _serialize_localized(point: LocalizedCriticalPoint) -> dict[str, Any]:
    p = point.sample.point
    return {
        "masses": [float(x) for x in p.masses],
        "x1": float(p.x1),
        "v1": float(p.v1),
        "v2": float(p.v2),
        "period": float(p.period),
        "closure": float(p.residual_norm),
        "event_mode": str(point.event_mode),
        "event": float(point.event_value),
    }


def _serialize_variational(point: VariationalPoint) -> dict[str, Any]:
    out = _serialize_localized(point.localized)
    out.update(
        {
            "d_event_dm": [float(x) for x in point.gradient],
            "mass_tangent": [float(x) for x in point.tangent_mass],
            "closure_jacobian_singular_values": [float(x) for x in point.closure_singular_values],
            "closure_family_relative_residual": float(point.closure_relative_residual),
            "mass_step": float(point.step),
            "event_normal_corrector_iterations": int(point.corrector_iterations),
        }
    )
    return out


def _variational_geometry(point: LocalizedCriticalPoint, reference_mass: np.ndarray) -> tuple[np.ndarray, np.ndarray, Any]:
    p = point.sample.point
    sens = mass_sensitivity(
        (p.x1, p.v1, p.v2, p.period),
        p.masses,
        rtol=1e-12,
        atol=1e-14,
    )
    gradient = np.asarray(sens.d_events_dm[point.event_mode], dtype=float)
    norm = float(np.linalg.norm(gradient))
    if not np.isfinite(norm) or norm < MIN_EVENT_GRADIENT:
        raise RuntimeError(f"event gradient collapsed: mode={point.event_mode} norm={norm:.3e}")
    tangent = np.asarray([gradient[1], -gradient[0]], dtype=float) / norm
    if float(np.dot(tangent, reference_mass)) < 0.0:
        tangent *= -1.0
    singular = np.asarray(sens.closure_jacobian_singular_values, dtype=float)
    if singular.size == 0 or float(singular[-1]) < MIN_CLOSURE_SINGULAR:
        raise RuntimeError(
            f"periodic-family closure Jacobian is nearly singular: sigma_min={singular[-1] if singular.size else float('nan'):.3e}"
        )
    if sens.dp_lstsq_relative_residual > 1e-6:
        raise RuntimeError(
            f"implicit family sensitivity is inconsistent: rel={sens.dp_lstsq_relative_residual:.3e}"
        )
    return gradient, tangent, sens


def _advance_variational(
    previous: LocalizedCriticalPoint,
    current: LocalizedCriticalPoint,
    *,
    requested_step: float,
    max_corrector_iterations: int = 5,
    max_bracket_iterations: int = 8,
) -> VariationalPoint:
    reference_mass = (
        np.asarray(current.sample.point.masses[:2], dtype=float)
        - np.asarray(previous.sample.point.masses[:2], dtype=float)
    )
    gradient, tangent, sens0 = _variational_geometry(current, reference_mass)
    normal = gradient / float(np.linalg.norm(gradient))
    current_mass = np.asarray(current.sample.point.masses[:2], dtype=float)
    predictor = current_mass + float(requested_step) * tangent
    lam = 0.0
    pcur = current.sample.point
    guess = (pcur.x1, pcur.v1, pcur.v2, pcur.period)
    accepted_sample: BoundarySample | None = None
    accepted_value = float("inf")
    accepted_sens = None
    evaluations: list[_NormalEvaluation] = []
    tube_radius = max(2.5 * requested_step, 2e-3)

    def evaluate_normal(candidate_lam: float) -> _NormalEvaluation:
        nonlocal guess
        masses2 = predictor + candidate_lam * normal
        masses = (float(masses2[0]), float(masses2[1]), float(pcur.masses[2]))
        corrected = correct_family_point(masses, guess, max_nfev=70)
        if not corrected.success or corrected.residual_norm > CLOSURE_GATE:
            raise RuntimeError(
                f"periodic corrector failed at masses={masses}: closure={corrected.residual_norm:.3e}"
            )
        sample = _precise_evaluate(corrected)
        value = float(event_value(sample.floquet, current.event_mode))
        guess = (corrected.x1, corrected.v1, corrected.v2, corrected.period)
        evaluation = _NormalEvaluation(float(candidate_lam), value, sample)
        evaluations.append(evaluation)
        return evaluation

    for iteration in range(1, max_corrector_iterations + 1):
        evaluation = evaluate_normal(lam)
        value = evaluation.value
        if abs(value) < abs(accepted_value):
            accepted_value = value
        if abs(value) <= EVENT_GATE:
            accepted_sample = evaluation.sample
            break

        # Newton correction is only in the predictor-normal direction.  The
        # derivative is the total event gradient on the corrected family.
        if iteration == 1:
            derivative = float(np.dot(gradient, normal))
        else:
            sens_iter = mass_sensitivity(
                (
                    evaluation.sample.point.x1,
                    evaluation.sample.point.v1,
                    evaluation.sample.point.v2,
                    evaluation.sample.point.period,
                ),
                evaluation.sample.point.masses,
                rtol=1e-12,
                atol=1e-14,
            )
            grad_iter = np.asarray(sens_iter.d_events_dm[current.event_mode], dtype=float)
            derivative = float(np.dot(grad_iter, normal))
        if not math.isfinite(derivative) or abs(derivative) < MIN_EVENT_GRADIENT:
            raise RuntimeError(f"event-normal Newton derivative collapsed: {derivative:.3e}")
        delta = -value / derivative
        lam += float(delta)
        if abs(lam) > tube_radius:
            raise RuntimeError(
                f"event-normal correction left local tube: lambda={lam:.3e}, step={requested_step:.3e}"
            )

    # Near the float64 event floor, Newton may alternate across a perfectly
    # regular zero without landing below the frozen gate.  Preserve the same
    # gate and refine the sampled sign bracket with a safeguarded scalar secant.
    if accepted_sample is None:
        bracket = _tightest_sign_bracket(evaluations)
        if bracket is None:
            probe = min(0.5 * tube_radius, max(0.1 * requested_step, 2e-5))
            sampled_lambdas = {item.lam for item in evaluations}
            for probe_lam in (-probe, probe):
                if all(abs(probe_lam - item) > 1e-15 for item in sampled_lambdas):
                    evaluation = evaluate_normal(probe_lam)
                    if abs(evaluation.value) < abs(accepted_value):
                        accepted_value = evaluation.value
                    if abs(evaluation.value) <= EVENT_GATE:
                        accepted_sample = evaluation.sample
                        break
            bracket = _tightest_sign_bracket(evaluations)

        for _ in range(max_bracket_iterations):
            if accepted_sample is not None or bracket is None:
                break
            left, right = bracket
            trial_lam = _bracket_trial_lambda(left, right)
            evaluation = evaluate_normal(trial_lam)
            if abs(evaluation.value) < abs(accepted_value):
                accepted_value = evaluation.value
            if abs(evaluation.value) <= EVENT_GATE:
                accepted_sample = evaluation.sample
                break
            bracket = _tightest_sign_bracket(evaluations)

    if accepted_sample is None:
        raise RuntimeError(
            "event-normal corrector missed frozen gate after "
            f"{len(evaluations)} evaluations: best_event={accepted_value:.3e}"
        )
    corrected = accepted_sample.point
    accepted_value = float(event_value(accepted_sample.floquet, current.event_mode))
    # Compute geometry at every accepted point; it is both the next predictor
    # and a conditioning certificate for this step.
    accepted_sens = mass_sensitivity(
        (corrected.x1, corrected.v1, corrected.v2, corrected.period),
        corrected.masses,
        rtol=1e-12,
        atol=1e-14,
    )
    accepted_gradient = np.asarray(
        accepted_sens.d_events_dm[current.event_mode], dtype=float
    )
    accepted_norm = float(np.linalg.norm(accepted_gradient))
    if accepted_norm < MIN_EVENT_GRADIENT:
        raise RuntimeError(f"accepted event gradient collapsed: {accepted_norm:.3e}")
    accepted_tangent = np.asarray(
        [accepted_gradient[1], -accepted_gradient[0]], dtype=float
    ) / accepted_norm
    if float(np.dot(accepted_tangent, tangent)) < 0.0:
        accepted_tangent *= -1.0
    singular = np.asarray(accepted_sens.closure_jacobian_singular_values, dtype=float)
    if singular.size == 0 or float(singular[-1]) < MIN_CLOSURE_SINGULAR:
        raise RuntimeError(f"accepted closure Jacobian nearly singular: {singular[-1] if singular.size else float('nan'):.3e}")
    if accepted_sens.dp_lstsq_relative_residual > 1e-6:
        raise RuntimeError(
            f"accepted implicit-family sensitivity inconsistent: rel={accepted_sens.dp_lstsq_relative_residual:.3e}"
        )
    localized = LocalizedCriticalPoint(
        accepted_sample,
        current.event_mode,
        float(accepted_value),
        0.0,
    )
    return VariationalPoint(
        localized=localized,
        gradient=accepted_gradient,
        tangent_mass=accepted_tangent,
        closure_singular_values=singular,
        closure_relative_residual=float(accepted_sens.dp_lstsq_relative_residual),
        step=float(requested_step),
        corrector_iterations=len(evaluations),
    )


def _vector_from_record(row: dict[str, Any]) -> np.ndarray:
    masses = row["masses"]
    return np.asarray(
        [row["x1"], row["v1"], row["v2"], row["period"], masses[0], masses[1]],
        dtype=float,
    )


def _segment_distance_scaled(a: np.ndarray, b: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    scale = continuation_scales(target)
    aa = (a - target) / scale
    bb = (b - target) / scale
    d = bb - aa
    denom = float(np.dot(d, d))
    if denom == 0.0:
        return float(np.linalg.norm(aa)), 0.0
    fraction = float(np.clip(-np.dot(aa, d) / denom, 0.0, 1.0))
    return float(np.linalg.norm(aa + fraction * d)), fraction


def _mass_segment_tangent(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = np.asarray(b[4:6] - a[4:6], dtype=float)
    norm = float(np.linalg.norm(d))
    return d / norm if norm > 0.0 else np.asarray([0.0, 0.0])


def _jax_mass_tangent(point: LocalizedCriticalPoint, reference_mass: np.ndarray) -> dict[str, Any]:
    reference_mass = np.asarray(reference_mass, dtype=float)
    if reference_mass.shape != (2,) or not np.isfinite(reference_mass).all():
        raise ValueError("reference mass tangent must have two finite components")
    reference = np.zeros(6, dtype=float)
    reference[4:6] = reference_mass
    y = np.asarray(point.vector, dtype=float)
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
    jacobian = np.vstack((closure_jac, event_grad[None, :]))
    tangent = critical_tangent(
        jacobian,
        scales=continuation_scales(y),
        reference=reference,
    )
    mass = np.asarray(tangent.physical[4:6], dtype=float)
    norm = float(np.linalg.norm(mass))
    if norm == 0.0:
        raise RuntimeError("JAX six-dimensional null tangent has zero mass projection")
    mass /= norm
    jacobian_norm = float(np.linalg.norm(jacobian, ord=2))
    relative_null_residual = float(
        tangent.null_residual / max(jacobian_norm, np.finfo(float).tiny)
    )
    return {
        "mass_tangent": mass,
        "null_residual": float(tangent.null_residual),
        "relative_null_residual": relative_null_residual,
        "spectral_gap": float(tangent.spectral_gap),
        "singular_values": [float(x) for x in tangent.singular_values],
    }


def _jax_diagnostics_pass(diagnostics: dict[str, Any]) -> bool:
    residual = float(diagnostics.get("relative_null_residual", float("inf")))
    gap = float(diagnostics.get("spectral_gap", 0.0))
    return bool(
        math.isfinite(residual)
        and residual <= MAX_JAX_RELATIVE_NULL_RESIDUAL
        and math.isfinite(gap)
        and gap >= MIN_JAX_SPECTRAL_GAP
    )


def _domain_crossing(a: np.ndarray, b: np.ndarray) -> dict[str, Any] | None:
    ma, mb = a[4:6], b[4:6]
    candidates: list[tuple[float, str, np.ndarray]] = []
    faces = (
        (0, DECLARED_DOMAIN["m1"][0], "domain_m1_min"),
        (0, DECLARED_DOMAIN["m1"][1], "domain_m1_max"),
        (1, DECLARED_DOMAIN["m2"][0], "domain_m2_min"),
        (1, DECLARED_DOMAIN["m2"][1], "domain_m2_max"),
    )
    for axis, value, name in faces:
        da, db = float(ma[axis] - value), float(mb[axis] - value)
        if da == 0.0:
            t = 0.0
        elif da * db > 0.0 or float(mb[axis] - ma[axis]) == 0.0:
            continue
        else:
            t = float((value - ma[axis]) / (mb[axis] - ma[axis]))
        if not (0.0 <= t <= 1.0):
            continue
        mass = ma + t * (mb - ma)
        other = 1 - axis
        bounds = DECLARED_DOMAIN["m2"] if other == 1 else DECLARED_DOMAIN["m1"]
        if bounds[0] - 1e-10 <= mass[other] <= bounds[1] + 1e-10:
            candidates.append((t, name, mass))
    if not candidates:
        return None
    t, name, mass = min(candidates, key=lambda item: item[0])
    return {"face": name, "fraction": t, "masses": [float(mass[0]), float(mass[1]), 1.0]}


def _comparison_targets(path: Path) -> list[StrictPoint]:
    payload = _load(path)
    out: list[StrictPoint] = []
    for slc in payload.get("slices", []):
        for bracket in slc.get("brackets", []):
            if bracket.get("reachable_by_published_label_pipeline") is not False:
                continue
            cert = bracket.get("certification")
            if not isinstance(cert, dict) or cert.get("status") != "passed":
                raise RuntimeError("label-invisible bracket lacks passed certification")
            source_id = f"m1={float(cert['masses'][0]):.3f}:{cert['event_mode']}"
            bounds = tuple(float(value) for value in bracket.get("m2_bracket", ()))
            if len(bounds) != 2:
                raise RuntimeError(f"{source_id}: label-invisible seed lacks its signed m2 bracket")
            out.append(
                _strict_from_certification(
                    cert,
                    source_id=source_id,
                    m2_bounds=(bounds[0], bounds[1]),
                )
            )
    if len(out) != 9:
        raise RuntimeError(f"expected nine label-invisible certified seeds, got {len(out)}")
    return out


def _mesh_rows(payload: dict[str, Any], components: set[int], *, minimum_m1: float | None = None) -> list[dict[str, Any]]:
    rows = []
    for row in payload.get("roots", []):
        if int(row.get("sweep_component", -1)) not in components:
            continue
        if row.get("status") != "ok" or not row.get("passed"):
            continue
        if minimum_m1 is not None and float(row["masses"][0]) < minimum_m1:
            continue
        rows.append(row)
    return rows


def _target_state(target: StrictPoint) -> dict[str, Any]:
    p = target.localized.sample.point
    sens = mass_sensitivity(
        (p.x1, p.v1, p.v2, p.period),
        p.masses,
        rtol=1e-12,
        atol=1e-14,
    )
    gradient = np.asarray(sens.d_events_dm[target.localized.event_mode], dtype=float)
    norm = float(np.linalg.norm(gradient))
    if norm < MIN_EVENT_GRADIENT:
        raise RuntimeError(f"{target.source_id}: target event gradient collapsed")
    vt = np.asarray([gradient[1], -gradient[0]], dtype=float) / norm
    jax = _jax_mass_tangent(target.localized, reference_mass=vt)
    cosine = abs(float(np.dot(vt, jax["mass_tangent"])))
    return {
        "point": target,
        "variational_mass_tangent": vt,
        "d_event_dm": gradient,
        "closure_family_relative_residual": float(sens.dp_lstsq_relative_residual),
        "closure_jacobian_singular_values": [float(x) for x in sens.closure_jacobian_singular_values],
        "jax": jax,
        "variational_vs_jax_mass_tangent_abs_cosine": cosine,
        "best_segment_miss_scaled": float("inf"),
        "best_segment_tangent_abs_cosine": 0.0,
        "best_segment_index": None,
    }


def _update_target(target: dict[str, Any], a: np.ndarray, b: np.ndarray, index: int) -> None:
    point: StrictPoint = target["point"]
    miss, _fraction = _segment_distance_scaled(a, b, point.vector)
    segment_tangent = _mass_segment_tangent(a, b)
    cosine = abs(float(np.dot(segment_tangent, target["variational_mass_tangent"])))
    if miss < float(target["best_segment_miss_scaled"]):
        target["best_segment_miss_scaled"] = miss
        target["best_segment_tangent_abs_cosine"] = cosine
        target["best_segment_index"] = index


def _trace_branch(
    *,
    branch_id: str,
    previous: StrictPoint,
    current: StrictPoint,
    targets: list[StrictPoint],
    mesh_rows: list[dict[str, Any]],
    endpoint: str,
    endpoint_target: StrictPoint | None,
    requested_step: float,
    max_steps: int,
    checkpoint: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if previous.localized.event_mode != current.localized.event_mode:
        raise RuntimeError(f"{branch_id}: germ event modes differ")
    mode = current.localized.event_mode
    target_states = [_target_state(target) for target in targets]
    mesh = [
        {
            "cell_id": int(row["cell_id"]),
            "component": int(row["sweep_component"]),
            "target": _vector_from_record(row),
            "best_miss_scaled": float("inf"),
        }
        for row in mesh_rows
    ]
    endpoint_state = _target_state(endpoint_target) if endpoint_target is not None else None

    points: list[VariationalPoint] = []
    prev = previous.localized
    cur = current.localized
    step = float(requested_step)
    stop_reason = "max_steps_exhausted"
    terminal: dict[str, Any] | None = None
    retry_history: list[dict[str, Any]] = []

    def emit_checkpoint(status: str) -> None:
        if checkpoint is None:
            return
        checkpoint(
            {
                "schema": "atlas.v1.continuation-branch-checkpoint/1",
                "branch_id": branch_id,
                "event_mode": mode,
                "status": status,
                "requested_mass_step": requested_step,
                "accepted_points": [_serialize_variational(point) for point in points],
                "retry_history": retry_history,
                "last_accepted_point": (
                    _serialize_variational(points[-1]) if points else None
                ),
            }
        )

    emit_checkpoint("initialized")

    for index in range(max_steps):
        accepted: VariationalPoint | None = None
        last_error: Exception | None = None
        trial = step
        for _retry in range(8):
            try:
                accepted = _advance_variational(prev, cur, requested_step=trial)
                break
            except (RuntimeError, ValueError, FloatingPointError) as exc:
                last_error = exc
                retry_history.append(
                    {
                        "step_index": index,
                        "trial_mass_step": float(trial),
                        "failure_class": type(exc).__name__,
                        "message": str(exc),
                    }
                )
                emit_checkpoint("retrying_after_numerical_failure")
                trial *= 0.5
                if trial < 5e-5:
                    break
        if accepted is None:
            stop_reason = f"continuation_failure_not_a_scientific_terminus: {last_error}"
            break

        a, b = np.asarray(cur.vector, dtype=float), accepted.vector
        for target in target_states:
            _update_target(target, a, b, index)
        for item in mesh:
            miss, _fraction = _segment_distance_scaled(a, b, item["target"])
            item["best_miss_scaled"] = min(float(item["best_miss_scaled"]), miss)
        if endpoint_state is not None:
            _update_target(endpoint_state, a, b, index)
            if (
                endpoint_state["best_segment_miss_scaled"] <= 4e-3
                and endpoint_state["best_segment_tangent_abs_cosine"] >= 0.95
            ):
                terminal = {
                    "kind": "canonically_bound_continuation_germ",
                    "target": endpoint_target.source_id,
                    "miss_scaled": float(endpoint_state["best_segment_miss_scaled"]),
                    "tangent_abs_cosine": float(endpoint_state["best_segment_tangent_abs_cosine"]),
                }
                stop_reason = "classified_organizer_germ_reached"
        else:
            crossing = _domain_crossing(a, b)
            if crossing is not None:
                terminal = {"kind": "declared_domain_boundary", **crossing}
                stop_reason = "declared_domain_boundary_reached"

        points.append(accepted)
        emit_checkpoint("accepted_step")
        prev, cur = cur, accepted.localized
        step = min(float(requested_step), trial * 1.35)
        if terminal is not None:
            break

    target_records = []
    all_targets = True
    for target in target_states:
        p: StrictPoint = target["point"]
        passed = bool(
            target["best_segment_miss_scaled"] <= 4e-3
            and target["best_segment_tangent_abs_cosine"] >= 0.95
            and target["variational_vs_jax_mass_tangent_abs_cosine"] >= 0.95
            and _jax_diagnostics_pass(target["jax"])
        )
        all_targets &= passed
        target_records.append(
            {
                "source_id": p.source_id,
                "source": p.source,
                "point": _serialize_localized(p.localized),
                "d_event_dm": [float(x) for x in target["d_event_dm"]],
                "variational_mass_tangent": [float(x) for x in target["variational_mass_tangent"]],
                "variational_vs_jax_mass_tangent_abs_cosine": float(target["variational_vs_jax_mass_tangent_abs_cosine"]),
                "jax_null_residual": float(target["jax"]["null_residual"]),
                "jax_relative_null_residual": float(
                    target["jax"]["relative_null_residual"]
                ),
                "jax_spectral_gap": float(target["jax"]["spectral_gap"]),
                "jax_diagnostics_passed": _jax_diagnostics_pass(target["jax"]),
                "best_segment_miss_scaled": float(target["best_segment_miss_scaled"]),
                "best_segment_tangent_abs_cosine": float(target["best_segment_tangent_abs_cosine"]),
                "best_segment_index": target["best_segment_index"],
                "continuous_incidence_passed": passed,
            }
        )

    mesh_records = [
        {
            "cell_id": item["cell_id"],
            "component": item["component"],
            "best_miss_scaled": float(item["best_miss_scaled"]),
            "covered": bool(item["best_miss_scaled"] <= 1.2e-2),
        }
        for item in mesh
    ]
    mesh_passed = bool(mesh_records) and all(item["covered"] for item in mesh_records)
    terminal_passed = terminal is not None and terminal["kind"] in {
        "canonically_bound_continuation_germ",
        "declared_domain_boundary",
    }
    branch_passed = bool(all_targets and mesh_passed and terminal_passed)

    return {
        "branch_id": branch_id,
        "event_mode": mode,
        "method": "variational dG/dm predictor + periodic Newton + safeguarded event-normal correction",
        "start_germs": {
            "previous": {"source_id": previous.source_id, "point": _serialize_localized(previous.localized)},
            "current": {"source_id": current.source_id, "point": _serialize_localized(current.localized)},
        },
        "requested_mass_step": requested_step,
        "accepted_points": [_serialize_variational(p) for p in points],
        "retry_history": retry_history,
        "issue_seed_witnesses": target_records,
        "sampled_component_overlap": mesh_records,
        "terminal": terminal,
        "stopped_reason": stop_reason,
        "continuous_branch_passed": branch_passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("comparison")
    parser.add_argument("supplemental_roots")
    parser.add_argument("output")
    parser.add_argument("--mass-step", type=float, default=0.0035)
    parser.add_argument("--max-steps", type=int, default=120)
    args = parser.parse_args()

    out = Path(args.output)
    initialization_stage = "accelerated_runtime"
    original_excepthook = sys.excepthook

    def checkpoint_unhandled_exception(exc_type, exc, traceback) -> None:
        if not out.exists():
            _write_json_atomic(
                out,
                {
                    "schema": "atlas.v1.label-invisible-continuation-checkpoint/1",
                    "claim_status": "failed_initialization_partial_continuation_evidence",
                    "frozen_gates": {
                        "maximum_absolute_event": EVENT_GATE,
                        "maximum_periodic_closure": CLOSURE_GATE,
                    },
                    "completed_branches": [],
                    "active_branch": None,
                    "campaign_error": {
                        "stage": initialization_stage,
                        "failure_class": exc_type.__name__,
                        "message": str(exc),
                    },
                },
            )
        original_excepthook(exc_type, exc, traceback)

    sys.excepthook = checkpoint_unhandled_exception
    require_accelerated_x64()
    initialization_stage = "comparison_seed_recertification"
    comparison = Path(args.comparison)
    supplemental = _load(Path(args.supplemental_roots))
    issue_targets = _comparison_targets(comparison)

    initialization_stage = "canonical_germ_recertification"
    root = Path("research/evidence")
    pleft = root / "V1_MIXED_GERMS_PRINCIPAL_LEFT_2026-08-16.json"
    sleft = root / "V1_MIXED_GERMS_SECONDARY_LEFT_2026-08-16.json"
    sright = root / "V1_SECONDARY_RIGHT_GERMS_2026-08-16.json"
    pright = root / "V1_MIXED_GERMS_PRINCIPAL_RIGHT_2026-08-16.json"

    pleft_minus_prev = _germ_point(pleft, "minus_one", "-")
    pleft_minus_cur = _germ_point(pleft, "minus_one", "+")
    pleft_plus_prev = _germ_point(pleft, "plus_one", "-")
    pleft_plus_cur = _germ_point(pleft, "plus_one", "+")
    sleft_plus_target = _germ_point(sleft, "plus_one", "+")
    sleft_minus_prev = _germ_point(sleft, "minus_one", "-")
    sleft_minus_cur = _germ_point(sleft, "minus_one", "+")
    sright_minus_prev = _germ_point(sright, "minus_one", "-")
    sright_minus_cur = _germ_point(sright, "minus_one", "+")
    sright_plus_prev = _germ_point(sright, "plus_one", "-")
    sright_plus_cur = _germ_point(sright, "plus_one", "+")
    pright_minus_prev = _germ_point(pright, "minus_one", "-")
    pright_minus_cur = _germ_point(pright, "minus_one", "+")
    pright_plus_target = _germ_point(pright, "plus_one", "+")

    minus_left_targets = [
        target
        for target in issue_targets
        if target.localized.event_mode == "minus_one" and target.masses2[0] < 0.95
    ]
    plus_targets = [target for target in issue_targets if target.localized.event_mode == "plus_one"]
    minus_right_targets = [
        target
        for target in issue_targets
        if target.localized.event_mode == "minus_one" and target.masses2[0] >= 0.99
    ]
    if (len(minus_left_targets), len(plus_targets), len(minus_right_targets)) != (3, 4, 2):
        raise RuntimeError(
            "unexpected nine-seed partition: "
            f"{len(minus_left_targets)}, {len(plus_targets)}, {len(minus_right_targets)}"
        )

    branch_specs = [
        dict(
            branch_id="principal_left_minus_to_domain",
            previous=pleft_minus_prev,
            current=pleft_minus_cur,
            targets=minus_left_targets,
            mesh_rows=_mesh_rows(supplemental, {0}),
            endpoint="domain",
            endpoint_target=None,
            requested_step=args.mass_step,
            max_steps=args.max_steps,
        ),
        dict(
            branch_id="principal_left_plus_to_secondary_left",
            previous=pleft_plus_prev,
            current=pleft_plus_cur,
            targets=plus_targets,
            mesh_rows=_mesh_rows(supplemental, {10}),
            endpoint="germ",
            endpoint_target=sleft_plus_target,
            requested_step=args.mass_step,
            max_steps=args.max_steps,
        ),
        dict(
            branch_id="secondary_left_minus_to_domain",
            previous=sleft_minus_prev,
            current=sleft_minus_cur,
            targets=minus_right_targets,
            mesh_rows=_mesh_rows(
                supplemental,
                {1},
                minimum_m1=float(_load(sleft)["canonical_masses"][0]),
            ),
            endpoint="domain",
            endpoint_target=None,
            requested_step=args.mass_step,
            max_steps=args.max_steps,
        ),
        dict(
            branch_id="secondary_right_plus_to_principal_right",
            previous=sright_plus_prev,
            current=sright_plus_cur,
            targets=[],
            mesh_rows=_mesh_rows(supplemental, {11, 12}),
            endpoint="germ",
            endpoint_target=pright_plus_target,
            requested_step=args.mass_step,
            max_steps=args.max_steps,
        ),
        dict(
            branch_id="secondary_right_minus_to_domain",
            previous=sright_minus_prev,
            current=sright_minus_cur,
            targets=[],
            mesh_rows=_mesh_rows(supplemental, {3}),
            endpoint="domain",
            endpoint_target=None,
            requested_step=args.mass_step,
            max_steps=args.max_steps,
        ),
        dict(
            branch_id="principal_right_minus_to_domain",
            previous=pright_minus_prev,
            current=pright_minus_cur,
            targets=[],
            mesh_rows=_mesh_rows(supplemental, {4}),
            endpoint="domain",
            endpoint_target=None,
            requested_step=args.mass_step,
            max_steps=args.max_steps,
        ),
    ]

    branches: list[dict[str, Any]] = []

    def write_checkpoint(
        active_branch: dict[str, Any] | None,
        campaign_error: dict[str, Any] | None = None,
    ) -> None:
        _write_json_atomic(
            out,
            {
                "schema": "atlas.v1.label-invisible-continuation-checkpoint/1",
                "claim_status": "running_or_failed_partial_continuation_evidence",
                "frozen_gates": {
                    "maximum_absolute_event": EVENT_GATE,
                    "maximum_periodic_closure": CLOSURE_GATE,
                },
                "completed_branches": branches,
                "active_branch": active_branch,
                "campaign_error": campaign_error,
            },
        )

    for spec in branch_specs:
        branch_id = str(spec["branch_id"])
        try:
            branch = _trace_branch(
                **spec,
                checkpoint=lambda partial: write_checkpoint(partial),
            )
        except Exception as exc:
            write_checkpoint(
                None,
                {
                    "branch_id": branch_id,
                    "failure_class": type(exc).__name__,
                    "message": str(exc),
                },
            )
            raise
        branches.append(branch)
        write_checkpoint(None)

    issue_witnesses = [w for branch in branches for w in branch["issue_seed_witnesses"]]
    all_nine = len(issue_witnesses) == 9 and all(w["continuous_incidence_passed"] for w in issue_witnesses)
    all_branches = all(branch["continuous_branch_passed"] for branch in branches)
    right_wall = next(branch for branch in branches if branch["branch_id"] == "secondary_right_plus_to_principal_right")
    passed = bool(all_nine and all_branches)

    result = {
        "schema": "atlas.v1.label-invisible-continuation/2",
        "claim": (
            "continuous reconciliation of all nine label-invisible roots and "
            "all six multi-point supplemental sweep components"
        ),
        "frozen_gates": {
            "maximum_absolute_event": EVENT_GATE,
            "maximum_periodic_closure": CLOSURE_GATE,
        },
        "method": {
            "long_range": "integrated variational total dG/dm on corrected periodic family; tangent perpendicular to gradient; safeguarded event-normal scalar correction",
            "independent_tangent_crosscheck": "JAX x64 + Diffrax derivative of six-dimensional closure+event system; SciPy values remain authoritative",
            "organizer_seed": "opposite canonically bound pseudo-arclength germs straddling independently reproduced mixed organizer",
            "termini": "only canonically bound organizer germ or declared-domain face counts as terminal",
            "independent_tangent_gates": {
                "maximum_relative_null_residual": MAX_JAX_RELATIVE_NULL_RESIDUAL,
                "minimum_spectral_gap": MIN_JAX_SPECTRAL_GAP,
            },
        },
        "branches": branches,
        "all_nine_seed_continuations_passed": all_nine,
        "right_plus_one_wall_continuous_between_organizers": bool(right_wall["continuous_branch_passed"]),
        "continuous_witness_passed": passed,
        "claim_status": (
            "all_nine_label_invisible_roots_continuously_reconciled"
            if passed
            else "open_contradiction_continuous_witness_incomplete"
        ),
    }
    _write_json_atomic(out, result)
    print(
        json.dumps(
            {
                "continuous_witness_passed": passed,
                "all_nine_seed_continuations_passed": all_nine,
                "branches": {
                    branch["branch_id"]: {
                        "passed": branch["continuous_branch_passed"],
                        "accepted_points": len(branch["accepted_points"]),
                        "stopped_reason": branch["stopped_reason"],
                        "terminal": branch["terminal"],
                    }
                    for branch in branches
                },
            },
            indent=2,
        )
    )
    raise SystemExit(0 if passed else 3)


if __name__ == "__main__":
    main()
