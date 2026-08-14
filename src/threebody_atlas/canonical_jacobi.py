"""Canonical translation-reduced Jacobi dynamics and Floquet diagnostics.

Coordinates are z=(rho, lam, p_rho, p_lam), each planar. Here
rho=r2-r1 and lam=r3-(m1*r1+m2*r2)/(m1+m2). Their conjugate momenta are
p_rho=mu12*(v2-v1) and p_lam=mu3_12*(v3-v12).

This is a canonical four-degree-of-freedom translation reduction. It provides
the standard symplectic form needed for publication-grade symplectic-defect and
Krein diagnostics. Float64 routines here are screening/reference routines; the
critical claim path must be independently reproduced in arbitrary precision.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

Array = np.ndarray


def symplectic_matrix() -> Array:
    j = np.zeros((8, 8), dtype=float)
    j[:4, 4:] = np.eye(4)
    j[4:, :4] = -np.eye(4)
    return j


def _g_and_dg(x: Array) -> tuple[Array, Array]:
    r2 = float(x @ x)
    if r2 == 0.0:
        raise ZeroDivisionError("binary collision")
    r = np.sqrt(r2)
    inv3 = 1.0 / r**3
    inv5 = 1.0 / r**5
    g = x * inv3
    dg = np.eye(2) * inv3 - 3.0 * np.outer(x, x) * inv5
    return g, dg


def full_to_jacobi(state: Array, masses: Array) -> Array:
    state = np.asarray(state, dtype=float)
    m1, m2, m3 = np.asarray(masses, dtype=float)
    m12 = m1 + m2
    mt = m12 + m3
    mu12 = m1 * m2 / m12
    mu3 = m3 * m12 / mt
    r1, r2, r3 = state[0:2], state[2:4], state[4:6]
    v1, v2, v3 = state[6:8], state[8:10], state[10:12]
    rho = r2 - r1
    c12 = (m1 * r1 + m2 * r2) / m12
    v12 = (m1 * v1 + m2 * v2) / m12
    lam = r3 - c12
    p_rho = mu12 * (v2 - v1)
    p_lam = mu3 * (v3 - v12)
    return np.concatenate((rho, lam, p_rho, p_lam))


def jacobi_to_full_com(state: Array, masses: Array) -> Array:
    """Reconstruct a COM-zero and total-momentum-zero full state."""
    z = np.asarray(state, dtype=float)
    m1, m2, m3 = np.asarray(masses, dtype=float)
    m12 = m1 + m2
    mt = m12 + m3
    mu12 = m1 * m2 / m12
    mu3 = m3 * m12 / mt
    rho, lam = z[0:2], z[2:4]
    p_rho, p_lam = z[4:6], z[6:8]

    c12 = -(m3 / mt) * lam
    r1 = c12 - (m2 / m12) * rho
    r2 = c12 + (m1 / m12) * rho
    r3 = (m12 / mt) * lam

    rel12 = p_rho / mu12
    rel3 = p_lam / mu3
    v12 = -(m3 / mt) * rel3
    v3 = (m12 / mt) * rel3
    v1 = v12 - (m2 / m12) * rel12
    v2 = v12 + (m1 / m12) * rel12
    return np.concatenate((r1, r2, r3, v1, v2, v3))


def rhs_and_jacobian(state: Array, masses: Array) -> tuple[Array, Array]:
    z = np.asarray(state, dtype=float)
    m1, m2, m3 = np.asarray(masses, dtype=float)
    m12 = m1 + m2
    mt = m12 + m3
    mu12 = m1 * m2 / m12
    mu3 = m3 * m12 / mt
    a = m2 / m12
    b = m1 / m12

    rho, lam = z[0:2], z[2:4]
    p_rho, p_lam = z[4:6], z[6:8]
    x13 = lam + a * rho
    x23 = lam - b * rho
    g12, d12 = _g_and_dg(rho)
    g13, d13 = _g_and_dg(x13)
    g23, d23 = _g_and_dg(x23)

    grad_rho = m1 * m2 * g12 + m1 * m3 * a * g13 - m2 * m3 * b * g23
    grad_lam = m1 * m3 * g13 + m2 * m3 * g23

    out = np.empty(8, dtype=float)
    out[0:2] = p_rho / mu12
    out[2:4] = p_lam / mu3
    out[4:6] = -grad_rho
    out[6:8] = -grad_lam

    hrr = m1 * m2 * d12 + m1 * m3 * a * a * d13 + m2 * m3 * b * b * d23
    hll = m1 * m3 * d13 + m2 * m3 * d23
    hrl = m1 * m3 * a * d13 - m2 * m3 * b * d23

    jac = np.zeros((8, 8), dtype=float)
    jac[0:2, 4:6] = np.eye(2) / mu12
    jac[2:4, 6:8] = np.eye(2) / mu3
    jac[4:6, 0:2] = -hrr
    jac[4:6, 2:4] = -hrl
    jac[6:8, 0:2] = -hrl.T
    jac[6:8, 2:4] = -hll
    return out, jac


def rhs(_t: float, state: Array, masses: Array) -> Array:
    return rhs_and_jacobian(state, masses)[0]


@dataclass(frozen=True)
class CanonicalFloquetResult:
    monodromy: Array
    multipliers: Array
    closure_norm: float
    symplectic_defect: float
    reciprocal_pairing_error: float


def _reciprocal_pairing_error(eig: Array) -> float:
    remaining = list(complex(x) for x in eig)
    worst = 0.0
    while remaining:
        lam = remaining.pop()
        target = 1.0 / lam
        if not remaining:
            return float("inf")
        idx = min(range(len(remaining)), key=lambda i: abs(remaining[i] - target))
        mate = remaining.pop(idx)
        worst = max(worst, abs(mate - target))
    return float(worst)


def compute_canonical_floquet(
    full_state0: Array,
    masses: Array,
    period: float,
    *,
    rtol: float = 2e-11,
    atol: float = 2e-13,
) -> CanonicalFloquetResult:
    z0 = full_to_jacobi(full_state0, masses)
    y0 = np.concatenate((z0, np.eye(8).ravel()))

    def augmented(t: float, y: Array) -> Array:
        dz, jac = rhs_and_jacobian(y[:8], masses)
        phi = y[8:].reshape(8, 8)
        return np.concatenate((dz, (jac @ phi).ravel()))

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
    zf = sol.y[:8, -1]
    monodromy = sol.y[8:, -1].reshape(8, 8)
    eig = np.linalg.eigvals(monodromy)
    j = symplectic_matrix()
    defect = np.linalg.norm(monodromy.T @ j @ monodromy - j, ord=np.inf)
    return CanonicalFloquetResult(
        monodromy=monodromy,
        multipliers=eig,
        closure_norm=float(np.linalg.norm(zf - z0)),
        symplectic_defect=float(defect),
        reciprocal_pairing_error=_reciprocal_pairing_error(eig),
    )


def krein_form(eigenvector: Array) -> float:
    """Return real Krein-form value -i v*Jv for a complex eigenvector.

    The sign is meaningful for a simple unit-circle eigenmode after consistent
    canonical normalization. Near collisions/defective modes, publication code
    must use invariant subspace diagnostics rather than trusting this scalar.
    """
    v = np.asarray(eigenvector, dtype=complex)
    value = -1j * np.vdot(v, symplectic_matrix() @ v)
    return float(np.real_if_close(value).real)
