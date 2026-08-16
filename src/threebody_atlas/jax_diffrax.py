"""Differentiable adaptive reduced three-body flow for screening.

This module is deliberately *not* a publication truth path. It provides a JAX
x64 + Diffrax implementation of the COM-reduced dynamics so that continuation
can obtain high-quality derivatives without finite-differencing every shooting
variable. Every use must be cross-checked against the canonical SciPy/Julia
paths before it is allowed to support a scientific claim.

The integration interval is normalized to s in [0, 1], with

    dz/ds = T f(z; m1, m2, m3),

so the period T is an ordinary differentiable parameter instead of the terminal
integration time. This lets Diffrax ForwardMode differentiate the full closure
map with respect to (x1, v1, v2, T, m1, m2).

For Floquet-event gradients we integrate the 8D state together with its 8x8
fundamental matrix. JAX then differentiates the smooth event scalar through that
adaptive augmented solve. This requires second derivatives of the vector field,
which JAX obtains automatically; the resulting gradients remain screening-only
until independently checked against the SciPy variational path.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

import numpy as np

try:  # Optional accelerated dependency.
    import diffrax
    import jax
    import jax.numpy as jnp
except ImportError:  # pragma: no cover - exercised in base-only environments.
    diffrax = None
    jax = None
    jnp = None

Array = np.ndarray
EventMode = Literal["plus_one", "minus_one", "trace_collision"]


def accelerated_available() -> bool:
    return jax is not None and jnp is not None and diffrax is not None


def require_accelerated_x64() -> None:
    if not accelerated_available():
        raise RuntimeError("JAX + Diffrax are required; install the accelerated extra")
    if not bool(jax.config.x64_enabled):
        raise RuntimeError("JAX x64 is mandatory for three-body accelerated screening")


def _force(x):
    r2 = jnp.dot(x, x)
    return x / (r2 * jnp.sqrt(r2))


def reduced_rhs_jax(_t, state, masses):
    """JAX form of the 8D COM-reduced Newtonian vector field."""
    q1 = state[0:2]
    q2 = state[2:4]
    d12 = q2 - q1
    f1 = _force(q1)
    f2 = _force(q2)
    fd = _force(d12)
    m1, m2, m3 = masses
    a1 = m2 * fd - (m1 + m3) * f1 - m2 * f2
    a2 = -m1 * fd - m1 * f1 - (m2 + m3) * f2
    return jnp.concatenate((state[4:8], a1, a2))


def chart_state_jax(y, m3):
    """Map y=(x1,v1,v2,T,m1,m2) into the 8D reduced initial state."""
    x1, v1, v2, _period, m1, m2 = y
    v3 = -(m1 * v1 + m2 * v2) / m3
    return jnp.asarray(
        [x1, 0.0, 1.0, 0.0, 0.0, v1 - v3, 0.0, v2 - v3],
        dtype=jnp.float64,
    )


def _normalized_rhs(s, state, args):
    masses, period = args
    return period * reduced_rhs_jax(s, state, masses)


def _solve(term, y0, args, *, rtol: float, atol: float, max_steps: int):
    return diffrax.diffeqsolve(
        term,
        diffrax.Dopri8(),
        t0=0.0,
        t1=1.0,
        dt0=None,
        y0=y0,
        args=args,
        saveat=diffrax.SaveAt(t1=True),
        stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
        adjoint=diffrax.ForwardMode(),
        max_steps=max_steps,
        throw=True,
    )


def _closure_impl(y, m3, *, rtol: float, atol: float, max_steps: int):
    z0 = chart_state_jax(y, m3)
    masses = jnp.asarray([y[4], y[5], m3], dtype=jnp.float64)
    sol = _solve(
        diffrax.ODETerm(_normalized_rhs),
        z0,
        (masses, y[3]),
        rtol=rtol,
        atol=atol,
        max_steps=max_steps,
    )
    return sol.ys[0] - z0


def _normalized_augmented_rhs(s, augmented, args):
    masses, period = args
    z = augmented[:8]
    phi = augmented[8:].reshape(8, 8)
    rhs = reduced_rhs_jax(s, z, masses)
    local_jac = jax.jacfwd(reduced_rhs_jax, argnums=1)(s, z, masses)
    dphi = local_jac @ phi
    return period * jnp.concatenate((rhs, dphi.reshape(-1)))


def _floquet_invariants_jax(monodromy):
    alpha = jnp.trace(monodromy)
    beta = 0.5 * (alpha * alpha - jnp.trace(monodromy @ monodromy))
    discriminant = (alpha - 4.0) ** 2 - 4.0 * (beta - 4.0 * alpha + 8.0)
    return alpha, beta, discriminant


def _event_from_invariants(alpha, beta, discriminant, mode: EventMode):
    if mode == "plus_one":
        return beta - 6.0 * alpha + 20.0
    if mode == "minus_one":
        return beta - 2.0 * alpha + 4.0
    if mode == "trace_collision":
        return discriminant
    raise ValueError(f"unsupported event mode: {mode}")


def _event_impl(y, m3, mode: EventMode, *, rtol: float, atol: float, max_steps: int):
    z0 = chart_state_jax(y, m3)
    masses = jnp.asarray([y[4], y[5], m3], dtype=jnp.float64)
    augmented0 = jnp.concatenate((z0, jnp.eye(8, dtype=jnp.float64).reshape(-1)))
    sol = _solve(
        diffrax.ODETerm(_normalized_augmented_rhs),
        augmented0,
        (masses, y[3]),
        rtol=rtol,
        atol=atol,
        max_steps=max_steps,
    )
    monodromy = sol.ys[0][8:].reshape(8, 8)
    alpha, beta, discriminant = _floquet_invariants_jax(monodromy)
    return _event_from_invariants(alpha, beta, discriminant, mode)


def _mixed_system_impl(y, m3, *, rtol: float, atol: float, max_steps: int):
    """Integrate once for closure plus both smooth mixed-vertex events."""
    z0 = chart_state_jax(y, m3)
    masses = jnp.asarray([y[4], y[5], m3], dtype=jnp.float64)
    augmented0 = jnp.concatenate((z0, jnp.eye(8, dtype=jnp.float64).reshape(-1)))
    sol = _solve(
        diffrax.ODETerm(_normalized_augmented_rhs),
        augmented0,
        (masses, y[3]),
        rtol=rtol,
        atol=atol,
        max_steps=max_steps,
    )
    final = sol.ys[0]
    closure = final[:8] - z0
    monodromy = final[8:].reshape(8, 8)
    alpha, beta, discriminant = _floquet_invariants_jax(monodromy)
    plus = _event_from_invariants(alpha, beta, discriminant, "plus_one")
    minus = _event_from_invariants(alpha, beta, discriminant, "minus_one")
    return jnp.concatenate((closure, jnp.asarray([plus, minus], dtype=jnp.float64)))


@lru_cache(maxsize=8)
def _compiled_closure(rtol: float, atol: float, max_steps: int):
    require_accelerated_x64()

    def closure(y, m3):
        return _closure_impl(y, m3, rtol=rtol, atol=atol, max_steps=max_steps)

    return jax.jit(closure), jax.jit(jax.jacfwd(closure, argnums=0))


@lru_cache(maxsize=24)
def _compiled_event(mode: EventMode, rtol: float, atol: float, max_steps: int):
    require_accelerated_x64()

    def event(y, m3):
        return _event_impl(y, m3, mode, rtol=rtol, atol=atol, max_steps=max_steps)

    return jax.jit(event), jax.jit(jax.jacfwd(event, argnums=0))


@lru_cache(maxsize=8)
def _compiled_mixed_system(rtol: float, atol: float, max_steps: int):
    require_accelerated_x64()

    def mixed_system(y, m3):
        return _mixed_system_impl(y, m3, rtol=rtol, atol=atol, max_steps=max_steps)

    return jax.jit(mixed_system), jax.jit(jax.jacfwd(mixed_system, argnums=0))


def _validate_vector(y: Array) -> Array:
    y_arr = np.asarray(y, dtype=float)
    if y_arr.shape != (6,):
        raise ValueError("continuation vector must have six components")
    if y_arr[3] <= 0.0:
        raise ValueError("period must be positive")
    return y_arr


def adaptive_closure_and_jacobian(
    y: Array,
    *,
    m3: float = 1.0,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    max_steps: int = 1 << 18,
) -> tuple[Array, Array]:
    """Return adaptive Diffrax closure and d(closure)/d(x1,v1,v2,T,m1,m2)."""
    require_accelerated_x64()
    y_arr = _validate_vector(y)
    closure_fn, jacobian_fn = _compiled_closure(float(rtol), float(atol), int(max_steps))
    y_jax = jnp.asarray(y_arr, dtype=jnp.float64)
    m3_jax = jnp.asarray(float(m3), dtype=jnp.float64)
    closure = np.asarray(jax.device_get(closure_fn(y_jax, m3_jax)), dtype=float)
    jacobian = np.asarray(jax.device_get(jacobian_fn(y_jax, m3_jax)), dtype=float)
    return closure, jacobian


def adaptive_closure(
    y: Array,
    *,
    m3: float = 1.0,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    max_steps: int = 1 << 18,
) -> Array:
    closure, _ = adaptive_closure_and_jacobian(
        y, m3=m3, rtol=rtol, atol=atol, max_steps=max_steps
    )
    return closure


def adaptive_event_and_gradient(
    y: Array,
    mode: EventMode,
    *,
    m3: float = 1.0,
    rtol: float = 2e-9,
    atol: float = 2e-11,
    max_steps: int = 1 << 18,
) -> tuple[float, Array]:
    """Return a smooth Floquet event and its six-parameter JAX gradient.

    The slightly looser defaults than the state-only closure path control the
    cost of differentiating the 72D state+fundamental-matrix flow. Acceptance
    against the SciPy variational implementation is mandatory before use.
    """
    require_accelerated_x64()
    if mode not in ("plus_one", "minus_one", "trace_collision"):
        raise ValueError(f"unsupported event mode: {mode}")
    y_arr = _validate_vector(y)
    event_fn, gradient_fn = _compiled_event(mode, float(rtol), float(atol), int(max_steps))
    y_jax = jnp.asarray(y_arr, dtype=jnp.float64)
    m3_jax = jnp.asarray(float(m3), dtype=jnp.float64)
    value = float(jax.device_get(event_fn(y_jax, m3_jax)))
    gradient = np.asarray(jax.device_get(gradient_fn(y_jax, m3_jax)), dtype=float)
    return value, gradient


def adaptive_mixed_system_and_jacobian(
    y: Array,
    *,
    m3: float = 1.0,
    rtol: float = 5e-10,
    atol: float = 5e-12,
    max_steps: int = 1 << 18,
) -> tuple[Array, Array]:
    """Return ``[closure, G+, G-]`` and its six-parameter Jacobian.

    The mixed organizer previously differentiated three separate integrations:
    a state flow and two identical state+monodromy flows.  This joint path uses
    one augmented integration, preserving the same equations while removing
    duplicated work.  As elsewhere, SciPy values remain the acceptance truth.
    """
    require_accelerated_x64()
    y_arr = _validate_vector(y)
    value_fn, jacobian_fn = _compiled_mixed_system(
        float(rtol), float(atol), int(max_steps)
    )
    y_jax = jnp.asarray(y_arr, dtype=jnp.float64)
    m3_jax = jnp.asarray(float(m3), dtype=jnp.float64)
    value = np.asarray(jax.device_get(value_fn(y_jax, m3_jax)), dtype=float)
    jacobian = np.asarray(jax.device_get(jacobian_fn(y_jax, m3_jax)), dtype=float)
    return value, jacobian


def rhs_jacobian(state: Array, masses: Array) -> Array:
    """Autodifferentiate the local reduced vector field for implementation QA."""
    require_accelerated_x64()
    z = jnp.asarray(np.asarray(state, dtype=float), dtype=jnp.float64)
    m = jnp.asarray(np.asarray(masses, dtype=float), dtype=jnp.float64)
    jac = jax.jacfwd(reduced_rhs_jax, argnums=1)(0.0, z, m)
    return np.asarray(jax.device_get(jac), dtype=float)
