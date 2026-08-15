"""Periodic shooting and a conservative continuation corrector.

This is a float64 discovery/screening corrector.  Any converged orbit intended for
publication must be re-integrated by the independent high-precision verifier.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .dynamics import center_of_mass, integrate_orbit, rhs, total_energy

Array = np.ndarray


@dataclass(frozen=True)
class ShootingResult:
    state: Array
    period: float
    residual_norm: float
    nfev: int
    success: bool
    message: str


def refine_periodic_orbit(
    state_guess: Array,
    period_guess: float,
    masses: Array,
    *,
    reference_state: Array | None = None,
    target_energy: float | None = None,
    max_nfev: int = 80,
) -> ShootingResult:
    """Correct a nearby periodic orbit with phase/COM/energy gauge conditions.

    The residual is intentionally overdetermined: periodic closure is augmented
    with physically valid center-of-mass constraints, an energy normalization,
    and a phase condition relative to the reference orbit.  This removes the
    most troublesome symmetry/scaling null directions for local continuation.
    """
    masses = np.asarray(masses, dtype=float)
    state_guess = np.asarray(state_guess, dtype=float)
    reference_state = state_guess.copy() if reference_state is None else np.asarray(reference_state, dtype=float)
    if target_energy is None:
        target_energy = total_energy(reference_state, masses)
    phase_tangent = rhs(0.0, reference_state, masses)
    phase_scale = max(np.linalg.norm(phase_tangent), 1.0)

    x0 = np.concatenate((state_guess, [period_guess]))

    def residual(x: Array) -> Array:
        state = x[:12]
        period = float(x[12])
        if period <= 0:
            return np.full(18, 1e6 + abs(period))
        try:
            orbit = integrate_orbit(state, masses, period, rtol=3e-10, atol=3e-12)
            qcm, vcm = center_of_mass(state, masses)
            closure = orbit.final_state - state
            energy_res = (total_energy(state, masses) - target_energy) / max(abs(target_energy), 1.0)
            phase_res = float((state - reference_state) @ phase_tangent) / phase_scale
            return np.concatenate((closure, qcm, vcm, [energy_res, phase_res]))
        except (RuntimeError, FloatingPointError, ValueError):
            return np.full(18, 1e6)

    fit = least_squares(
        residual,
        x0,
        method="trf",
        xtol=1e-11,
        ftol=1e-11,
        gtol=1e-11,
        max_nfev=max_nfev,
        x_scale="jac",
    )
    return ShootingResult(
        state=fit.x[:12].copy(),
        period=float(fit.x[12]),
        residual_norm=float(np.linalg.norm(fit.fun)),
        nfev=int(fit.nfev),
        success=bool(fit.success),
        message=str(fit.message),
    )
