"""Center-of-mass reduced 8D dynamics and Floquet stability invariants.

The reduced coordinates are (r1-r3, r2-r3, v1-v3, v2-v3).  They remove the
four translational/total-momentum directions exactly.  The periodic-orbit
monodromy then has the four symmetry-related unit multipliers associated with
autonomous Hamiltonian dynamics and planar rotational symmetry; the remaining
four multipliers are captured by the Kapela--Simo trace invariants.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .dynamics import rhs
from .variational import state_jacobian

Array = np.ndarray


@dataclass(frozen=True)
class ReducedFloquetResult:
    monodromy: Array
    multipliers: Array
    alpha: float
    beta: float
    discriminant: float
    trace_roots: tuple[complex, complex]
    linearly_stable: bool | None
    stability_margin: float


def reconstruction_matrix(masses: Array) -> Array:
    """Map 8 relative coordinates into 12 full COM-zero coordinates."""
    m1, m2, m3 = np.asarray(masses, dtype=float)
    mt = m1 + m2 + m3
    p = np.zeros((12, 8), dtype=float)

    # Position blocks. q1=r1-r3, q2=r2-r3.
    for axis in range(2):
        q1 = axis
        q2 = 2 + axis
        r3_coeff_q1 = -m1 / mt
        r3_coeff_q2 = -m2 / mt
        p[4 + axis, q1] = r3_coeff_q1
        p[4 + axis, q2] = r3_coeff_q2
        p[axis, q1] = 1.0 + r3_coeff_q1
        p[axis, q2] = r3_coeff_q2
        p[2 + axis, q1] = r3_coeff_q1
        p[2 + axis, q2] = 1.0 + r3_coeff_q2

    # Velocity blocks use the same linear transformation.
    p[6:, 4:] = p[:6, :4]
    return p


def reduction_matrix() -> Array:
    """Map full coordinates to (r1-r3,r2-r3,v1-v3,v2-v3)."""
    r = np.zeros((8, 12), dtype=float)
    for axis in range(2):
        r[axis, axis] = 1.0
        r[axis, 4 + axis] = -1.0
        r[2 + axis, 2 + axis] = 1.0
        r[2 + axis, 4 + axis] = -1.0
        r[4 + axis, 6 + axis] = 1.0
        r[4 + axis, 10 + axis] = -1.0
        r[6 + axis, 8 + axis] = 1.0
        r[6 + axis, 10 + axis] = -1.0
    return r


def full_to_reduced(state: Array) -> Array:
    return reduction_matrix() @ np.asarray(state, dtype=float)


def reduced_to_full(state: Array, masses: Array) -> Array:
    state = np.asarray(state, dtype=float)
    if state.shape != (8,):
        raise ValueError("reduced state must contain 8 components")
    return reconstruction_matrix(masses) @ state


def reduced_rhs(_t: float, state: Array, masses: Array) -> Array:
    p = reconstruction_matrix(masses)
    r = reduction_matrix()
    full = p @ state
    return r @ rhs(0.0, full, masses)


def reduced_jacobian(state: Array, masses: Array) -> Array:
    p = reconstruction_matrix(masses)
    r = reduction_matrix()
    full = p @ state
    return r @ state_jacobian(full, masses) @ p


def stability_invariants(monodromy: Array, *, tolerance: float = 1e-7) -> ReducedFloquetResult:
    a = np.asarray(monodromy, dtype=float)
    if a.shape != (8, 8):
        raise ValueError("monodromy must be 8x8")
    alpha = float(np.trace(a))
    beta = float(0.5 * (alpha * alpha - np.trace(a @ a)))
    disc = float((alpha - 4.0) ** 2 - 4.0 * (beta - 4.0 * alpha + 8.0))
    roots = np.roots([1.0, -(alpha - 4.0), beta - 4.0 * alpha + 8.0])
    t1, t2 = complex(roots[0]), complex(roots[1])
    margins = [2.0 - abs(t1), 2.0 - abs(t2), disc]
    margin = float(min(margins))
    if abs(disc) <= tolerance or abs(abs(t1) - 2.0) <= tolerance or abs(abs(t2) - 2.0) <= tolerance:
        stable: bool | None = None
    else:
        stable = bool(disc > 0.0 and abs(t1.imag) <= tolerance and abs(t2.imag) <= tolerance and abs(t1) < 2.0 and abs(t2) < 2.0)
    eig = np.linalg.eigvals(a)
    return ReducedFloquetResult(a, eig, alpha, beta, disc, (t1, t2), stable, margin)


def compute_reduced_floquet(
    full_state0: Array,
    masses: Array,
    period: float,
    *,
    rtol: float = 2e-11,
    atol: float = 2e-13,
) -> ReducedFloquetResult:
    z0 = full_to_reduced(full_state0)
    phi0 = np.eye(8)
    y0 = np.concatenate((z0, phi0.ravel()))

    def augmented(t: float, y: Array) -> Array:
        z = y[:8]
        phi = y[8:].reshape(8, 8)
        return np.concatenate((reduced_rhs(t, z, masses), (reduced_jacobian(z, masses) @ phi).ravel()))

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
    return stability_invariants(sol.y[8:, -1].reshape(8, 8))
