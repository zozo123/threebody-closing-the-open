"""Continuation chart matching the Li--Li--Liao unequal-mass family.

Published baseline points use
  r1=(x1,0), r2=(1,0), r3=(0,0),
  v1=(0,v1), v2=(0,v2), v3=(0,-(m1*v1+m2*v2)/m3).
For fixed masses this leaves (x1,v1,v2,T) as the shooting variables.  The
corrector minimizes full-period closure and is therefore directly initialized
from the public supplementary data.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .dynamics import integrate_orbit

Array = np.ndarray


@dataclass(frozen=True)
class FamilyPoint:
    masses: tuple[float, float, float]
    x1: float
    v1: float
    v2: float
    period: float
    residual_norm: float
    nfev: int
    success: bool

    def state(self) -> Array:
        return state_from_chart(self.masses, self.x1, self.v1, self.v2)


def state_from_chart(
    masses: tuple[float, float, float], x1: float, v1: float, v2: float
) -> Array:
    m1, m2, m3 = masses
    v3 = -(m1 * v1 + m2 * v2) / m3
    return np.asarray(
        [x1, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, v1, 0.0, v2, 0.0, v3],
        dtype=float,
    )


def correct_family_point(
    masses: tuple[float, float, float],
    guess: tuple[float, float, float, float],
    *,
    max_nfev: int = 80,
    screening_rtol: float = 2e-10,
    screening_atol: float = 2e-12,
) -> FamilyPoint:
    """Newton/least-squares shooting correction in the published family chart."""

    def residual(p: Array) -> Array:
        x1, v1, v2, period = [float(x) for x in p]
        if period <= 0.0:
            return np.full(12, 1e6 + abs(period))
        state = state_from_chart(masses, x1, v1, v2)
        try:
            orbit = integrate_orbit(
                state,
                np.asarray(masses),
                period,
                rtol=screening_rtol,
                atol=screening_atol,
            )
        except (RuntimeError, FloatingPointError, ValueError):
            return np.full(12, 1e6)
        return orbit.final_state - state

    fit = least_squares(
        residual,
        np.asarray(guess, dtype=float),
        method="trf",
        xtol=2e-12,
        ftol=2e-12,
        gtol=2e-12,
        max_nfev=max_nfev,
        x_scale="jac",
    )
    return FamilyPoint(
        masses=masses,
        x1=float(fit.x[0]),
        v1=float(fit.x[1]),
        v2=float(fit.x[2]),
        period=float(fit.x[3]),
        residual_norm=float(np.linalg.norm(fit.fun)),
        nfev=int(fit.nfev),
        success=bool(fit.success),
    )


def continue_mass_segment(
    seed: FamilyPoint,
    target_masses: tuple[float, float, float],
    *,
    steps: int,
    max_residual: float = 1e-7,
) -> list[FamilyPoint]:
    if steps < 1:
        raise ValueError("steps must be positive")
    start = np.asarray(seed.masses, dtype=float)
    target = np.asarray(target_masses, dtype=float)
    current = seed
    out: list[FamilyPoint] = []
    for k in range(1, steps + 1):
        masses = tuple((start + (target - start) * (k / steps)).tolist())
        guess = (current.x1, current.v1, current.v2, current.period)
        point = correct_family_point(masses, guess)
        if not point.success or point.residual_norm > max_residual:
            raise RuntimeError(
                f"continuation failed at step {k}/{steps}: residual={point.residual_norm:.3e}"
            )
        out.append(point)
        current = point
    return out
