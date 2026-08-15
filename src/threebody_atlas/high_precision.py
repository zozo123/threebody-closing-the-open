"""Independent arbitrary-precision verifier using mpmath.

The implementation favors auditability over speed.  It uses fixed-step RK4 with
step-doubling convergence checks.  For publishable results, the workflow records
both the requested decimal precision and the step-convergence discrepancy.
"""
from __future__ import annotations

from dataclasses import dataclass

import mpmath as mp


@dataclass(frozen=True)
class HighPrecisionResult:
    final_state: tuple[str, ...]
    closure_norm: str
    energy_defect: str
    step_convergence_norm: str
    dps: int
    steps: int


def _rhs(state: list[mp.mpf], masses: list[mp.mpf]) -> list[mp.mpf]:
    q = [[state[2 * i], state[2 * i + 1]] for i in range(3)]
    v = [[state[6 + 2 * i], state[6 + 2 * i + 1]] for i in range(3)]
    a = [[mp.mpf("0"), mp.mpf("0")] for _ in range(3)]
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            dx = q[j][0] - q[i][0]
            dy = q[j][1] - q[i][1]
            r2 = dx * dx + dy * dy
            if r2 == 0:
                raise ZeroDivisionError("binary collision")
            inv_r3 = r2 ** mp.mpf("-1.5")
            a[i][0] += masses[j] * dx * inv_r3
            a[i][1] += masses[j] * dy * inv_r3
    return [c for pair in v for c in pair] + [c for pair in a for c in pair]


def _energy(state: list[mp.mpf], masses: list[mp.mpf]) -> mp.mpf:
    kinetic = mp.mpf("0")
    for i in range(3):
        vx, vy = state[6 + 2 * i], state[6 + 2 * i + 1]
        kinetic += mp.mpf("0.5") * masses[i] * (vx * vx + vy * vy)
    potential = mp.mpf("0")
    for i in range(3):
        for j in range(i + 1, 3):
            dx = state[2 * i] - state[2 * j]
            dy = state[2 * i + 1] - state[2 * j + 1]
            potential -= masses[i] * masses[j] / mp.sqrt(dx * dx + dy * dy)
    return kinetic + potential


def _rk4(state0: list[mp.mpf], masses: list[mp.mpf], period: mp.mpf, steps: int) -> list[mp.mpf]:
    h = period / steps
    y = list(state0)
    for _ in range(steps):
        k1 = _rhs(y, masses)
        k2 = _rhs([y[i] + h * k1[i] / 2 for i in range(12)], masses)
        k3 = _rhs([y[i] + h * k2[i] / 2 for i in range(12)], masses)
        k4 = _rhs([y[i] + h * k3[i] for i in range(12)], masses)
        y = [y[i] + h * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]) / 6 for i in range(12)]
    return y


def _norm(values: list[mp.mpf]) -> mp.mpf:
    return mp.sqrt(mp.fsum(v * v for v in values))


def verify_closure(
    initial_state: tuple[float | str, ...],
    masses: tuple[float | str, float | str, float | str],
    period: float | str,
    *,
    dps: int = 60,
    steps: int = 4096,
) -> HighPrecisionResult:
    """Re-integrate with arbitrary precision and explicit step-doubling.

    Inputs may be decimal strings; this is preferred for scientific evidence so
    conversion does not inherit binary64 rounding.
    """
    if len(initial_state) != 12:
        raise ValueError("initial_state must contain 12 components")
    if steps < 8:
        raise ValueError("steps must be at least 8")
    with mp.workdps(dps):
        y0 = [mp.mpf(str(x)) for x in initial_state]
        m = [mp.mpf(str(x)) for x in masses]
        t = mp.mpf(str(period))
        coarse = _rk4(y0, m, t, steps)
        fine = _rk4(y0, m, t, steps * 2)
        closure = _norm([fine[i] - y0[i] for i in range(12)])
        convergence = _norm([fine[i] - coarse[i] for i in range(12)])
        energy_defect = abs(_energy(fine, m) - _energy(y0, m))
        digits = max(20, dps - 5)
        return HighPrecisionResult(
            final_state=tuple(mp.nstr(x, digits) for x in fine),
            closure_norm=mp.nstr(closure, digits),
            energy_defect=mp.nstr(energy_defect, digits),
            step_convergence_norm=mp.nstr(convergence, digits),
            dps=dps,
            steps=steps * 2,
        )
