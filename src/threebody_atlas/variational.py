"""Variational equations and Floquet diagnostics for the screening integrator."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .dynamics import rhs, unpack_state

Array = np.ndarray


@dataclass(frozen=True)
class FloquetResult:
    monodromy: Array
    multipliers: Array
    max_modulus_error: float
    symplectic_defect: float


def state_jacobian(state: Array, masses: Array, *, g: float = 1.0) -> Array:
    """Jacobian of the 12D first-order equations in (q, v) coordinates."""
    positions, _ = unpack_state(state)
    masses = np.asarray(masses, dtype=float)
    jac = np.zeros((12, 12), dtype=float)
    jac[:6, 6:] = np.eye(6)
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            d = positions[j] - positions[i]
            r2 = float(d @ d)
            if r2 == 0.0:
                raise FloatingPointError("binary collision")
            r = np.sqrt(r2)
            block = g * masses[j] * (np.eye(2) / r**3 - 3.0 * np.outer(d, d) / r**5)
            ri = slice(2 * i, 2 * i + 2)
            rj = slice(2 * j, 2 * j + 2)
            jac[6 + ri.start : 6 + ri.stop, rj] += block
            jac[6 + ri.start : 6 + ri.stop, ri] -= block
    return jac


def augmented_rhs(t: float, augmented: Array, masses: Array) -> Array:
    state = augmented[:12]
    phi = augmented[12:].reshape(12, 12)
    dstate = rhs(t, state, masses)
    dphi = state_jacobian(state, masses) @ phi
    return np.concatenate((dstate, dphi.ravel()))


def _canonical_monodromy(monodromy_qv: Array, masses: Array) -> Array:
    """Transform tangent map from (q,v) to canonical (q,p) coordinates."""
    mdiag = np.repeat(np.asarray(masses, dtype=float), 2)
    s = np.diag(np.concatenate((np.ones(6), mdiag)))
    return s @ monodromy_qv @ np.linalg.inv(s)


def compute_floquet(
    state0: Array,
    masses: Array,
    period: float,
    *,
    rtol: float = 2e-11,
    atol: float = 2e-13,
) -> FloquetResult:
    """Integrate state + 12x12 fundamental matrix over one proposed period.

    This full-coordinate spectrum contains symmetry/trivial directions.  It is a
    validation/screening diagnostic; publishable classification should use the
    reduced problem and high-precision reruns.
    """
    phi0 = np.eye(12)
    y0 = np.concatenate((np.asarray(state0, dtype=float), phi0.ravel()))
    sol = solve_ivp(
        lambda t, y: augmented_rhs(t, y, np.asarray(masses, dtype=float)),
        (0.0, period),
        y0,
        method="DOP853",
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    monodromy = sol.y[12:, -1].reshape(12, 12)
    multipliers = np.linalg.eigvals(monodromy)
    canonical = _canonical_monodromy(monodromy, masses)
    j = np.block([[np.zeros((6, 6)), np.eye(6)], [-np.eye(6), np.zeros((6, 6))]])
    symplectic_defect = float(np.linalg.norm(canonical.T @ j @ canonical - j, ord=np.inf))
    max_modulus_error = float(np.max(np.abs(np.abs(multipliers) - 1.0)))
    return FloquetResult(monodromy, multipliers, max_modulus_error, symplectic_defect)
