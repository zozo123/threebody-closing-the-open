"""Planar Newtonian three-body dynamics.

The float64 integrator in this module is intentionally the *screening* layer.  It is
not sufficient evidence for a publishable high-precision periodic-orbit claim.
High-precision verification is handled separately.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp


Array = np.ndarray


@dataclass(frozen=True)
class OrbitResult:
    t: Array
    y: Array
    initial_state: Array
    final_state: Array
    period: float
    closure_norm: float
    energy_initial: float
    energy_final: float
    angular_momentum_initial: float
    angular_momentum_final: float


def unpack_state(state: Array) -> tuple[Array, Array]:
    """Return positions and velocities as arrays with shape (3, 2)."""
    state = np.asarray(state, dtype=float)
    if state.shape != (12,):
        raise ValueError("state must contain 12 components")
    return state[:6].reshape(3, 2), state[6:].reshape(3, 2)


def acceleration(positions: Array, masses: Array, *, g: float = 1.0) -> Array:
    masses = np.asarray(masses, dtype=float)
    positions = np.asarray(positions, dtype=float)
    if masses.shape != (3,) or positions.shape != (3, 2):
        raise ValueError("masses must have shape (3,) and positions shape (3, 2)")
    out = np.zeros_like(positions)
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            delta = positions[j] - positions[i]
            r2 = float(delta @ delta)
            if r2 == 0.0:
                raise FloatingPointError("binary collision")
            out[i] += g * masses[j] * delta / (r2 ** 1.5)
    return out


def rhs(_t: float, state: Array, masses: Array, *, g: float = 1.0) -> Array:
    positions, velocities = unpack_state(state)
    return np.concatenate((velocities.ravel(), acceleration(positions, masses, g=g).ravel()))


def total_energy(state: Array, masses: Array, *, g: float = 1.0) -> float:
    positions, velocities = unpack_state(state)
    masses = np.asarray(masses, dtype=float)
    kinetic = 0.5 * np.sum(masses[:, None] * velocities**2)
    potential = 0.0
    for i in range(3):
        for j in range(i + 1, 3):
            r = np.linalg.norm(positions[i] - positions[j])
            potential -= g * masses[i] * masses[j] / r
    return float(kinetic + potential)


def angular_momentum(state: Array, masses: Array) -> float:
    positions, velocities = unpack_state(state)
    masses = np.asarray(masses, dtype=float)
    return float(np.sum(masses * (positions[:, 0] * velocities[:, 1] - positions[:, 1] * velocities[:, 0])))


def center_of_mass(state: Array, masses: Array) -> tuple[Array, Array]:
    positions, velocities = unpack_state(state)
    masses = np.asarray(masses, dtype=float)
    mtot = masses.sum()
    return (np.sum(masses[:, None] * positions, axis=0) / mtot,
            np.sum(masses[:, None] * velocities, axis=0) / mtot)


def integrate_orbit(
    state0: Array,
    masses: Array,
    period: float,
    *,
    rtol: float = 1e-11,
    atol: float = 1e-13,
    max_step: float | None = None,
    samples: int = 0,
) -> OrbitResult:
    """Integrate one proposed period and report closure/conservation diagnostics."""
    state0 = np.asarray(state0, dtype=float)
    masses = np.asarray(masses, dtype=float)
    if period <= 0:
        raise ValueError("period must be positive")
    t_eval = np.linspace(0.0, period, samples + 1) if samples > 0 else None
    kwargs = {}
    if max_step is not None:
        kwargs["max_step"] = max_step
    sol = solve_ivp(
        lambda t, y: rhs(t, y, masses),
        (0.0, period),
        state0,
        method="DOP853",
        rtol=rtol,
        atol=atol,
        t_eval=t_eval,
        **kwargs,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    final = sol.y[:, -1]
    return OrbitResult(
        t=sol.t,
        y=sol.y,
        initial_state=state0.copy(),
        final_state=final.copy(),
        period=float(period),
        closure_norm=float(np.linalg.norm(final - state0)),
        energy_initial=total_energy(state0, masses),
        energy_final=total_energy(final, masses),
        angular_momentum_initial=angular_momentum(state0, masses),
        angular_momentum_final=angular_momentum(final, masses),
    )
