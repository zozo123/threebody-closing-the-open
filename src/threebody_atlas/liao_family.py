"""Continuation chart matching the Li--Li--Liao unequal-mass family.

Published baseline points use
  r1=(x1,0), r2=(1,0), r3=(0,0),
  v1=(0,v1), v2=(0,v2), v3=(0,-(m1*v1+m2*v2)/m3).
For fixed masses this leaves (x1,v1,v2,T) as the shooting variables.  The
corrector minimizes COM-reduced full-period closure and supplies the exact
shooting Jacobian from the variational flow rather than finite differences.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from .reduced import full_to_reduced, reduced_jacobian, reduced_rhs, reduction_matrix

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


def _chart_tangent(masses: tuple[float, float, float]) -> Array:
    """Return dz0/d(x1,v1,v2), an 8x3 matrix in reduced coordinates."""
    m1, m2, m3 = masses
    full = np.zeros((12, 3), dtype=float)
    full[0, 0] = 1.0
    full[7, 1] = 1.0
    full[11, 1] = -m1 / m3
    full[9, 2] = 1.0
    full[11, 2] = -m2 / m3
    return reduction_matrix() @ full


def _flow_and_shooting_jacobian(
    masses: tuple[float, float, float],
    parameters: Array,
    *,
    rtol: float,
    atol: float,
) -> tuple[Array, Array]:
    x1, v1, v2, period = [float(x) for x in parameters]
    if period <= 0.0:
        return np.full(8, 1e6 + abs(period)), np.zeros((8, 4), dtype=float)
    mass_array = np.asarray(masses, dtype=float)
    z0 = full_to_reduced(state_from_chart(masses, x1, v1, v2))
    y0 = np.concatenate((z0, np.eye(8).ravel()))

    def augmented(t: float, y: Array) -> Array:
        z = y[:8]
        phi = y[8:].reshape(8, 8)
        dz = reduced_rhs(t, z, mass_array)
        dphi = reduced_jacobian(z, mass_array) @ phi
        return np.concatenate((dz, dphi.ravel()))

    sol = solve_ivp(
        augmented,
        (0.0, period),
        y0,
        method="DOP853",
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    zt = sol.y[:8, -1]
    monodromy = sol.y[8:, -1].reshape(8, 8)
    closure = zt - z0
    chart = _chart_tangent(masses)
    jac = np.empty((8, 4), dtype=float)
    jac[:, :3] = (monodromy - np.eye(8)) @ chart
    jac[:, 3] = reduced_rhs(period, zt, mass_array)
    return closure, jac


def correct_family_point(
    masses: tuple[float, float, float],
    guess: tuple[float, float, float, float],
    *,
    max_nfev: int = 40,
    screening_rtol: float = 2e-10,
    screening_atol: float = 2e-12,
) -> FamilyPoint:
    """Variational Newton/least-squares correction in the published family chart."""
    cached_p: tuple[float, ...] | None = None
    cached_result: tuple[Array, Array] | None = None

    def evaluate(p: Array) -> tuple[Array, Array]:
        nonlocal cached_p, cached_result
        key = tuple(float(x) for x in p)
        if cached_p != key:
            cached_p = key
            cached_result = _flow_and_shooting_jacobian(
                masses,
                p,
                rtol=screening_rtol,
                atol=screening_atol,
            )
        assert cached_result is not None
        return cached_result

    fit = least_squares(
        lambda p: evaluate(p)[0],
        np.asarray(guess, dtype=float),
        jac=lambda p: evaluate(p)[1],
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
