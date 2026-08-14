"""Independent arbitrary-precision reduced variational verifier.

This module does not call SciPy, NumPy's eigensolver, or the float64 Jacobian.  It
implements the center-of-mass relative equations and their analytic 8x8
Jacobian directly in mpmath, then integrates the state and fundamental matrix
with fixed-step RK4.  Step doubling provides a transparent convergence check.

It is intentionally slower than the screening layer and is meant for a small
set of publication-critical points selected by the float64 pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass

import mpmath as mp


@dataclass(frozen=True)
class HighPrecisionFloquetResult:
    closure_norm: str
    state_step_convergence_norm: str
    monodromy_step_convergence_norm: str
    alpha: str
    beta: str
    discriminant: str
    trace_roots: tuple[tuple[str, str], tuple[str, str]]
    stability_score: str
    dps: int
    steps: int


def _norm(values: list[mp.mpf]) -> mp.mpf:
    return mp.sqrt(mp.fsum(v * v for v in values))


def _f_and_d(x: tuple[mp.mpf, mp.mpf]) -> tuple[list[mp.mpf], list[list[mp.mpf]]]:
    x0, x1 = x
    r2 = x0 * x0 + x1 * x1
    if r2 == 0:
        raise ZeroDivisionError("binary collision")
    r = mp.sqrt(r2)
    inv3 = 1 / (r**3)
    inv5 = 1 / (r**5)
    f = [x0 * inv3, x1 * inv3]
    d = [
        [inv3 - 3 * x0 * x0 * inv5, -3 * x0 * x1 * inv5],
        [-3 * x1 * x0 * inv5, inv3 - 3 * x1 * x1 * inv5],
    ]
    return f, d


def _add_block(j: list[list[mp.mpf]], row: int, col: int, block, scale: mp.mpf) -> None:
    for a in range(2):
        for b in range(2):
            j[row + a][col + b] += scale * block[a][b]


def reduced_rhs_and_jacobian(
    z: list[mp.mpf], masses: list[mp.mpf]
) -> tuple[list[mp.mpf], list[list[mp.mpf]]]:
    """Return the 8D relative vector field and analytic Jacobian.

    z=(q1x,q1y,q2x,q2y,u1x,u1y,u2x,u2y), where qi=ri-r3 and
    ui=vi-v3.  Translation and total-momentum directions are absent.
    """
    if len(z) != 8 or len(masses) != 3:
        raise ValueError("expected 8D state and three masses")
    m1, m2, m3 = masses
    q1 = (z[0], z[1])
    q2 = (z[2], z[3])
    d12 = (q2[0] - q1[0], q2[1] - q1[1])
    f1, d1 = _f_and_d(q1)
    f2, d2 = _f_and_d(q2)
    fd, dd = _f_and_d(d12)

    du1 = [
        m2 * fd[k] - (m1 + m3) * f1[k] - m2 * f2[k] for k in range(2)
    ]
    du2 = [
        -m1 * fd[k] - m1 * f1[k] - (m2 + m3) * f2[k] for k in range(2)
    ]
    rhs = [z[4], z[5], z[6], z[7], *du1, *du2]

    zero = mp.mpf("0")
    one = mp.mpf("1")
    jac = [[zero for _ in range(8)] for _ in range(8)]
    for k in range(4):
        jac[k][4 + k] = one

    # du1/dq1, du1/dq2
    _add_block(jac, 4, 0, dd, -m2)
    _add_block(jac, 4, 0, d1, -(m1 + m3))
    _add_block(jac, 4, 2, dd, m2)
    _add_block(jac, 4, 2, d2, -m2)

    # du2/dq1, du2/dq2.  du2 contains -m1*F(q2-q1), hence +m1*Dd wrt q1.
    _add_block(jac, 6, 0, dd, m1)
    _add_block(jac, 6, 0, d1, -m1)
    _add_block(jac, 6, 2, dd, -m1)
    _add_block(jac, 6, 2, d2, -(m2 + m3))
    return rhs, jac


def _augmented_rhs(y: list[mp.mpf], masses: list[mp.mpf]) -> list[mp.mpf]:
    z = y[:8]
    phi_flat = y[8:]
    rhs, jac = reduced_rhs_and_jacobian(z, masses)
    dphi: list[mp.mpf] = []
    for i in range(8):
        for k in range(8):
            dphi.append(mp.fsum(jac[i][j] * phi_flat[8 * j + k] for j in range(8)))
    return rhs + dphi


def _rk4_augmented(
    z0: list[mp.mpf], masses: list[mp.mpf], period: mp.mpf, steps: int
) -> tuple[list[mp.mpf], list[mp.mpf]]:
    identity = [mp.mpf(int(i == j)) for i in range(8) for j in range(8)]
    y = list(z0) + identity
    h = period / steps
    for _ in range(steps):
        k1 = _augmented_rhs(y, masses)
        k2 = _augmented_rhs([y[i] + h * k1[i] / 2 for i in range(72)], masses)
        k3 = _augmented_rhs([y[i] + h * k2[i] / 2 for i in range(72)], masses)
        k4 = _augmented_rhs([y[i] + h * k3[i] for i in range(72)], masses)
        y = [
            y[i] + h * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) / 6
            for i in range(72)
        ]
    return y[:8], y[8:]


def _invariants(phi: list[mp.mpf]):
    alpha = mp.fsum(phi[8 * i + i] for i in range(8))
    trace_a2 = mp.fsum(
        phi[8 * i + j] * phi[8 * j + i] for i in range(8) for j in range(8)
    )
    beta = (alpha * alpha - trace_a2) / 2
    c = beta - 4 * alpha + 8
    b = -(alpha - 4)
    discriminant = b * b - 4 * c
    sqrt_disc = mp.sqrt(discriminant)
    t1 = (-b + sqrt_disc) / 2
    t2 = (-b - sqrt_disc) / 2
    score = min(discriminant, 2 - abs(t1), 2 - abs(t2))
    return alpha, beta, discriminant, t1, t2, score


def _complex_strings(value, digits: int) -> tuple[str, str]:
    return mp.nstr(mp.re(value), digits), mp.nstr(mp.im(value), digits)


def verify_reduced_floquet(
    full_initial_state: tuple[float | str, ...],
    masses: tuple[float | str, float | str, float | str],
    period: float | str,
    *,
    dps: int = 50,
    steps: int = 512,
) -> HighPrecisionFloquetResult:
    """Run two independent-resolution RK4 integrations and report convergence."""
    if len(full_initial_state) != 12:
        raise ValueError("full_initial_state must contain 12 components")
    if steps < 16:
        raise ValueError("steps must be at least 16")
    with mp.workdps(dps):
        x = [mp.mpf(str(v)) for v in full_initial_state]
        m = [mp.mpf(str(v)) for v in masses]
        t = mp.mpf(str(period))
        z0 = [
            x[0] - x[4],
            x[1] - x[5],
            x[2] - x[4],
            x[3] - x[5],
            x[6] - x[10],
            x[7] - x[11],
            x[8] - x[10],
            x[9] - x[11],
        ]
        z_coarse, phi_coarse = _rk4_augmented(z0, m, t, steps)
        z_fine, phi_fine = _rk4_augmented(z0, m, t, steps * 2)
        closure = _norm([z_fine[i] - z0[i] for i in range(8)])
        state_conv = _norm([z_fine[i] - z_coarse[i] for i in range(8)])
        mono_conv = _norm([phi_fine[i] - phi_coarse[i] for i in range(64)])
        alpha, beta, disc, t1, t2, score = _invariants(phi_fine)
        digits = max(20, dps - 8)
        return HighPrecisionFloquetResult(
            closure_norm=mp.nstr(closure, digits),
            state_step_convergence_norm=mp.nstr(state_conv, digits),
            monodromy_step_convergence_norm=mp.nstr(mono_conv, digits),
            alpha=mp.nstr(alpha, digits),
            beta=mp.nstr(beta, digits),
            discriminant=mp.nstr(disc, digits),
            trace_roots=(_complex_strings(t1, digits), _complex_strings(t2, digits)),
            stability_score=mp.nstr(score, digits),
            dps=dps,
            steps=steps * 2,
        )
