"""Generic gauge-fixed periodic-orbit correction in translation-reduced coordinates.

The Li shooting chart is extremely effective but encodes a special collinear
initial section and velocity pattern.  Near branch points or symmetry-breaking
solutions that specialization can become the scientific question rather than a
harmless coordinate choice.

This module provides a deliberately less specialized single-shooting BVP.  The
unknown is ``(z0,T)`` with ``z0`` the full 8D translation-reduced relative state.
No Li collinearity or velocity ansatz is imposed.  Three continuous degeneracies
are fixed locally:

* Newtonian similarity: ``|r2-r3| = 1``;
* planar rotation: ``(r2-r3)_y = 0`` (the nearby positive-x representative);
* time phase: an orthogonality condition to a supplied reference orbit.

The periodic closure contributes eight residuals and the three gauges add three
more.  The system is intentionally solved as an overdetermined least-squares
problem: first integrals make the closure rows dependent on the exact periodic
solution manifold, while the three gauges remove the local time/rotation/scale
directions.

This is a chart-independence / branch-discovery tool, not an arbitrary-precision
publication verifier.  Headline results still require the independent Julia
BigFloat/canonical path.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from .conditioning import SolveConditioning, condition_report
from .reduced import reduced_rhs

Array = np.ndarray


@dataclass(frozen=True)
class GenericPeriodicPoint:
    masses: tuple[float, float, float]
    state: tuple[float, float, float, float, float, float, float, float]
    period: float
    closure_norm: float
    gauge_norm: float
    phase_residual: float
    nfev: int
    optimality: float
    success: bool
    #: Conditioning of the augmented residual Jacobian at the corrected orbit.
    #: This corrector supplies no analytic Jacobian, so the report is built from
    #: the optimizer's finite-difference Jacobian and inherits its accuracy.
    conditioning: SolveConditioning | None = None

    @property
    def vector(self) -> Array:
        return np.asarray((*self.state, self.period), dtype=float)


def integrate_reduced_period(
    masses: tuple[float, float, float],
    state: Array,
    period: float,
    *,
    rtol: float = 3e-10,
    atol: float = 3e-12,
) -> Array:
    """Integrate one candidate reduced orbit period and return the final state."""
    if period <= 0.0:
        raise ValueError("period must be positive")
    z0 = np.asarray(state, dtype=float)
    if z0.shape != (8,):
        raise ValueError("reduced state must have shape (8,)")
    mass_array = np.asarray(masses, dtype=float)
    sol = solve_ivp(
        lambda t, z: reduced_rhs(t, z, mass_array),
        (0.0, float(period)),
        z0,
        method="DOP853",
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    return np.asarray(sol.y[:, -1], dtype=float)


def generic_periodic_residual(
    y: Array,
    masses: tuple[float, float, float],
    reference_state: Array,
    *,
    rtol: float = 3e-10,
    atol: float = 3e-12,
) -> Array:
    """Return periodic closure plus local scale/rotation/time-phase gauges."""
    vector = np.asarray(y, dtype=float)
    if vector.shape != (9,):
        raise ValueError("generic periodic vector must have shape (9,)")
    z0 = vector[:8]
    period = float(vector[8])
    reference = np.asarray(reference_state, dtype=float)
    if reference.shape != (8,):
        raise ValueError("reference_state must have shape (8,)")

    zf = integrate_reduced_period(masses, z0, period, rtol=rtol, atol=atol)
    closure = zf - z0

    # In the reduced convention z[2:4] is r2-r3.  Its norm fixes Newtonian
    # similarity and its y component fixes the SO(2) rotation representative.
    q23 = z0[2:4]
    scale = float(np.dot(q23, q23) - 1.0)
    rotation = float(q23[1])

    flow_reference = reduced_rhs(0.0, reference, np.asarray(masses, dtype=float))
    phase_scale = max(float(np.linalg.norm(flow_reference)), 1.0)
    phase = float(np.dot(z0 - reference, flow_reference) / phase_scale)
    return np.concatenate((closure, np.asarray([scale, rotation, phase], dtype=float)))


def correct_generic_periodic(
    masses: tuple[float, float, float],
    state_guess: Array,
    period_guess: float,
    *,
    reference_state: Array | None = None,
    max_nfev: int = 120,
    max_closure: float = 2e-7,
    max_gauge: float = 2e-7,
    max_phase: float = 2e-7,
    rtol: float = 3e-10,
    atol: float = 3e-12,
) -> GenericPeriodicPoint:
    """Correct a nearby strict periodic orbit without imposing the Li ansatz."""
    state0 = np.asarray(state_guess, dtype=float)
    if state0.shape != (8,):
        raise ValueError("state_guess must have shape (8,)")
    if period_guess <= 0.0:
        raise ValueError("period_guess must be positive")
    reference = state0.copy() if reference_state is None else np.asarray(reference_state, dtype=float)
    if reference.shape != (8,):
        raise ValueError("reference_state must have shape (8,)")

    y0 = np.concatenate((state0, [float(period_guess)]))
    floors = np.asarray([0.2, 0.2, 0.5, 0.2, 0.5, 0.5, 0.5, 0.5, 1.0])
    x_scale = np.maximum(np.abs(y0), floors)
    lower = np.asarray([-20.0] * 8 + [0.1], dtype=float)
    upper = np.asarray([20.0] * 8 + [30.0], dtype=float)

    fit = least_squares(
        lambda y: generic_periodic_residual(
            y,
            masses,
            reference,
            rtol=rtol,
            atol=atol,
        ),
        y0,
        method="trf",
        bounds=(lower, upper),
        x_scale=x_scale,
        xtol=2e-11,
        ftol=2e-11,
        gtol=2e-11,
        max_nfev=max_nfev,
    )
    residual = generic_periodic_residual(
        fit.x,
        masses,
        reference,
        rtol=rtol,
        atol=atol,
    )
    closure_norm = float(np.linalg.norm(residual[:8]))
    gauge_norm = float(np.linalg.norm(residual[8:10]))
    phase_residual = float(residual[10])
    accepted = bool(
        fit.success
        and closure_norm <= max_closure
        and gauge_norm <= max_gauge
        and abs(phase_residual) <= max_phase
        and fit.x[2] > 0.0
    )
    return GenericPeriodicPoint(
        masses=tuple(float(x) for x in masses),
        state=tuple(float(x) for x in fit.x[:8]),
        period=float(fit.x[8]),
        closure_norm=closure_norm,
        gauge_norm=gauge_norm,
        phase_residual=phase_residual,
        nfev=int(fit.nfev),
        optimality=float(fit.optimality),
        success=accepted,
        conditioning=condition_report(getattr(fit, "jac", None), residual),
    )
