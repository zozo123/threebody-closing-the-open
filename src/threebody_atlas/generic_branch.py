"""Pseudo-arclength continuation for generic periodic daughter branches.

The amplitude-constrained solver in :mod:`threebody_atlas.branch_switch` is a
branch *switch*: it creates distinct nearby candidates by excluding the parent
orbit with a signed transverse-amplitude condition.  It is not a continuation
law.  This module removes that artificial amplitude condition after two
accepted daughter seeds and follows the generic strict-periodic solution set in

    y = (z0[0:8], T, m2)

at fixed ``m1`` and ``m3``.  Periodic closure plus the generic scale/rotation/
phase gauges are solved together with one pseudo-arclength condition.

This is float64 discovery/falsification machinery.  A traced daughter is not a
release claim until representative points are independently reproduced and its
relation to the Li parent sheet is established.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.optimize import least_squares

from .generic_periodic import generic_periodic_residual

Array = np.ndarray
_STATE_FLOORS = np.asarray([0.2, 0.2, 0.5, 0.2, 0.5, 0.5, 0.5, 0.5], dtype=float)


class _BranchPointLike(Protocol):
    masses: tuple[float, float, float]
    state: tuple[float, float, float, float, float, float, float, float]
    period: float


@dataclass(frozen=True)
class GenericBranchTracePoint:
    masses: tuple[float, float, float]
    state: tuple[float, float, float, float, float, float, float, float]
    period: float
    closure_norm: float
    gauge_norm: float
    phase_residual: float
    arclength_residual: float
    normalized_step: float
    nfev: int
    optimality: float
    success: bool

    @property
    def vector(self) -> Array:
        """Return the 10D continuation vector ``(z0,T,m2)``."""
        return np.asarray((*self.state, self.period, self.masses[1]), dtype=float)


@dataclass(frozen=True)
class GenericBranchTrace:
    points: tuple[GenericBranchTracePoint, ...]
    stopped_reason: str


def continuation_vector(point: _BranchPointLike) -> Array:
    """Return ``(z0,T,m2)`` for an amplitude seed or traced branch point."""
    state = np.asarray(point.state, dtype=float)
    if state.shape != (8,):
        raise ValueError("generic branch state must have shape (8,)")
    return np.concatenate((state, [float(point.period), float(point.masses[1])]))


def continuation_scales(y: Array) -> Array:
    """Coordinate scales for the 10D generic-branch continuation chart."""
    vector = np.asarray(y, dtype=float)
    if vector.shape != (10,):
        raise ValueError("generic branch continuation vector must have shape (10,)")
    state_scale = np.maximum(np.abs(vector[:8]), _STATE_FLOORS)
    return np.concatenate((state_scale, [max(abs(vector[8]), 1.0), max(abs(vector[9]), 0.05)]))


def advance_generic_branch(
    previous: _BranchPointLike,
    current: _BranchPointLike,
    *,
    reference_state: Array,
    m1: float,
    m3: float,
    normalized_step: float = 2e-3,
    m2_bounds: tuple[float, float] = (0.5, 1.5),
    max_closure: float = 3e-7,
    max_gauge: float = 3e-7,
    max_phase: float = 3e-7,
    max_arc: float = 2e-4,
    max_nfev: int = 140,
    rtol: float = 2e-10,
    atol: float = 2e-12,
) -> GenericBranchTracePoint:
    """Take one pseudo-arclength step on a generic daughter branch.

    The scale/rotation/time gauges use the fixed branch-point reference state.
    This keeps the local phase convention identical to the branch-switch seeds
    while the pseudo-arclength equation, not a transverse amplitude constraint,
    selects the next point.
    """
    if normalized_step <= 0.0:
        raise ValueError("normalized_step must be positive")
    reference = np.asarray(reference_state, dtype=float)
    if reference.shape != (8,):
        raise ValueError("reference_state must have shape (8,)")
    m2_lo, m2_hi = (float(m2_bounds[0]), float(m2_bounds[1]))
    if not (0.0 < m2_lo < m2_hi):
        raise ValueError("m2_bounds must be positive and increasing")

    yp = continuation_vector(previous)
    yc = continuation_vector(current)
    scales = continuation_scales(yc)
    secant = (yc - yp) / scales
    secant_norm = float(np.linalg.norm(secant))
    if secant_norm == 0.0 or not np.isfinite(secant_norm):
        raise ValueError("generic branch seeds must be distinct")
    tangent = secant / secant_norm
    predictor = yc + scales * float(normalized_step) * tangent

    closure_scale = np.maximum(np.abs(reference), 1.0)
    arc_scale = max(float(normalized_step), 1e-4)

    def residual(y: Array) -> Array:
        z0 = np.asarray(y[:8], dtype=float)
        period = float(y[8])
        m2 = float(y[9])
        base = generic_periodic_residual(
            np.concatenate((z0, [period])),
            (float(m1), m2, float(m3)),
            reference,
            rtol=rtol,
            atol=atol,
        )
        scaled = base.copy()
        scaled[:8] /= closure_scale
        arc = float(np.dot((np.asarray(y, dtype=float) - predictor) / scales, tangent))
        return np.concatenate((scaled, [arc / arc_scale]))

    lower = np.asarray([-20.0] * 8 + [0.1, m2_lo], dtype=float)
    upper = np.asarray([20.0] * 8 + [30.0, m2_hi], dtype=float)
    start = np.clip(predictor, lower + 1e-10, upper - 1e-10)
    fit = least_squares(
        residual,
        start,
        method="trf",
        bounds=(lower, upper),
        x_scale=scales,
        xtol=2e-11,
        ftol=2e-11,
        gtol=2e-11,
        max_nfev=max_nfev,
    )

    z0 = np.asarray(fit.x[:8], dtype=float)
    period = float(fit.x[8])
    m2 = float(fit.x[9])
    raw = generic_periodic_residual(
        np.concatenate((z0, [period])),
        (float(m1), m2, float(m3)),
        reference,
        rtol=rtol,
        atol=atol,
    )
    closure_norm = float(np.linalg.norm(raw[:8]))
    gauge_norm = float(np.linalg.norm(raw[8:10]))
    phase = float(raw[10])
    arc = float(np.dot((fit.x - predictor) / scales, tangent))
    accepted = bool(
        fit.success
        and closure_norm <= max_closure
        and gauge_norm <= max_gauge
        and abs(phase) <= max_phase
        and abs(arc) <= max_arc
        and m2_lo < m2 < m2_hi
        and z0[2] > 0.0
    )
    if not accepted:
        raise RuntimeError(
            "generic daughter pseudo-arclength correction missed gates: "
            f"success={fit.success} closure={closure_norm:.3e} gauge={gauge_norm:.3e} "
            f"phase={phase:.3e} arc={arc:.3e} m2={m2:.12g}"
        )

    return GenericBranchTracePoint(
        masses=(float(m1), m2, float(m3)),
        state=tuple(float(x) for x in z0),
        period=period,
        closure_norm=closure_norm,
        gauge_norm=gauge_norm,
        phase_residual=phase,
        arclength_residual=arc,
        normalized_step=float(normalized_step),
        nfev=int(fit.nfev),
        optimality=float(fit.optimality),
        success=True,
    )


def trace_generic_branch(
    first: _BranchPointLike,
    second: _BranchPointLike,
    *,
    reference_state: Array,
    m1: float,
    m3: float,
    steps: int,
    normalized_step: float = 2e-3,
    min_step: float = 1.25e-4,
    max_retries: int = 5,
    m2_bounds: tuple[float, float] = (0.5, 1.5),
) -> GenericBranchTrace:
    """Trace a daughter branch with adaptive step reduction and fixed gates."""
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if normalized_step <= 0.0 or min_step <= 0.0:
        raise ValueError("continuation steps must be positive")
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")

    previous: _BranchPointLike = first
    current: _BranchPointLike = second
    accepted_points: list[GenericBranchTracePoint] = []
    step = float(normalized_step)
    reason = "requested_steps_completed"

    for _ in range(steps):
        accepted: GenericBranchTracePoint | None = None
        trial_step = step
        last_error: Exception | None = None
        for _retry in range(max_retries + 1):
            try:
                accepted = advance_generic_branch(
                    previous,
                    current,
                    reference_state=reference_state,
                    m1=m1,
                    m3=m3,
                    normalized_step=trial_step,
                    m2_bounds=m2_bounds,
                )
                break
            except (RuntimeError, ValueError) as exc:
                last_error = exc
                trial_step *= 0.5
                if trial_step < min_step:
                    break
        if accepted is None:
            reason = f"pseudo-arclength correction failed: {last_error}"
            break
        accepted_points.append(accepted)
        previous, current = current, accepted
        step = min(float(normalized_step), trial_step * 1.25)

    return GenericBranchTrace(tuple(accepted_points), reason)
