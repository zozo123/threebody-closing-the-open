"""Amplitude-constrained branch switching outside the Li shooting ansatz.

A physical Floquet multiplier at ``+1`` is a necessary local signal for a
same-period bifurcation, but it is not by itself evidence for a new periodic
family.  To test whether a physical critical direction generates a distinct
branch, this module augments the generic 8D periodic corrector with

* one continuation parameter (``m2``), and
* one signed amplitude condition along a supplied transverse direction.

The parent orbit is therefore excluded when the target amplitude is nonzero.
A converged solution must still be compared with the independently corrected
Li parent at the returned masses: the augmented solve can otherwise simply
move along the known parent sheet.

The solve is a float64 discovery/falsification tool.  A daughter candidate is
not a scientific claim until it survives path continuation and independent
high-precision verification.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .generic_periodic import generic_periodic_residual

Array = np.ndarray

_STATE_FLOORS = np.asarray([0.2, 0.2, 0.5, 0.2, 0.5, 0.5, 0.5, 0.5], dtype=float)


@dataclass(frozen=True)
class GenericBranchPoint:
    masses: tuple[float, float, float]
    state: tuple[float, float, float, float, float, float, float, float]
    period: float
    target_amplitude: float
    achieved_amplitude: float
    closure_norm: float
    gauge_norm: float
    phase_residual: float
    amplitude_residual: float
    nfev: int
    optimality: float
    success: bool

    @property
    def vector(self) -> Array:
        return np.asarray((*self.state, self.period), dtype=float)


def scaled_branch_direction(reference_state: Array, direction: Array) -> tuple[Array, Array]:
    """Return coordinate scales and a unit direction in those scaled coordinates."""
    reference = np.asarray(reference_state, dtype=float)
    tangent = np.asarray(direction, dtype=float)
    if reference.shape != (8,) or tangent.shape != (8,):
        raise ValueError("reference_state and direction must have shape (8,)")
    state_scale = np.maximum(np.abs(reference), _STATE_FLOORS)
    scaled = tangent / state_scale
    norm = float(np.linalg.norm(scaled))
    if norm == 0.0 or not np.isfinite(norm):
        raise ValueError("branch direction must be nonzero and finite")
    return state_scale, scaled / norm


def branch_amplitude(
    state: Array,
    reference_state: Array,
    direction: Array,
) -> float:
    state_scale, unit_direction = scaled_branch_direction(reference_state, direction)
    delta = (np.asarray(state, dtype=float) - np.asarray(reference_state, dtype=float)) / state_scale
    return float(np.dot(delta, unit_direction))


def correct_generic_branch_amplitude(
    *,
    m1: float,
    m3: float,
    reference_m2: float,
    reference_state: Array,
    reference_period: float,
    direction: Array,
    target_amplitude: float,
    m2_halfwidth: float = 0.004,
    max_nfev: int = 140,
    max_closure: float = 3e-7,
    max_gauge: float = 3e-7,
    max_phase: float = 3e-7,
    max_amplitude_residual: float = 3e-7,
    rtol: float = 2e-10,
    atol: float = 2e-12,
) -> GenericBranchPoint:
    """Solve periodicity + gauges + signed amplitude while allowing ``m2`` to move."""
    reference = np.asarray(reference_state, dtype=float)
    if reference.shape != (8,):
        raise ValueError("reference_state must have shape (8,)")
    if reference_period <= 0.0:
        raise ValueError("reference_period must be positive")
    if m2_halfwidth <= 0.0:
        raise ValueError("m2_halfwidth must be positive")

    state_scale, unit_direction = scaled_branch_direction(reference, direction)
    seed_state = reference + float(target_amplitude) * state_scale * unit_direction
    y0 = np.concatenate((seed_state, [float(reference_period), float(reference_m2)]))
    closure_scale = np.maximum(np.abs(reference), 1.0)

    lower_m2 = max(1e-6, float(reference_m2) - m2_halfwidth)
    upper_m2 = float(reference_m2) + m2_halfwidth
    lower = np.asarray([-20.0] * 8 + [0.1, lower_m2], dtype=float)
    upper = np.asarray([20.0] * 8 + [30.0, upper_m2], dtype=float)
    x_scale = np.concatenate(
        (
            state_scale,
            [max(abs(reference_period), 1.0), max(m2_halfwidth, 1e-3)],
        )
    )

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
        scaled_base = base.copy()
        scaled_base[:8] /= closure_scale
        amplitude = float(np.dot((z0 - reference) / state_scale, unit_direction))
        return np.concatenate((scaled_base, [amplitude - float(target_amplitude)]))

    fit = least_squares(
        residual,
        y0,
        method="trf",
        bounds=(lower, upper),
        x_scale=x_scale,
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
    achieved = float(np.dot((z0 - reference) / state_scale, unit_direction))
    closure_norm = float(np.linalg.norm(raw[:8]))
    gauge_norm = float(np.linalg.norm(raw[8:10]))
    phase = float(raw[10])
    amplitude_residual = float(achieved - target_amplitude)
    accepted = bool(
        fit.success
        and closure_norm <= max_closure
        and gauge_norm <= max_gauge
        and abs(phase) <= max_phase
        and abs(amplitude_residual) <= max_amplitude_residual
        and lower_m2 < m2 < upper_m2
        and z0[2] > 0.0
    )
    return GenericBranchPoint(
        masses=(float(m1), m2, float(m3)),
        state=tuple(float(x) for x in z0),
        period=period,
        target_amplitude=float(target_amplitude),
        achieved_amplitude=achieved,
        closure_norm=closure_norm,
        gauge_norm=gauge_norm,
        phase_residual=phase,
        amplitude_residual=amplitude_residual,
        nfev=int(fit.nfev),
        optimality=float(fit.optimality),
        success=accepted,
    )
