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
"""
from __future__ import annotations

from functools import lru_cache

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


def _closure_impl(y, m3, *, rtol: float, atol: float, max_steps: int):
    z0 = chart_state_jax(y, m3)
    masses = jnp.asarray([y[4], y[5], m3], dtype=jnp.float64)
    term = diffrax.ODETerm(_normalized_rhs)
    solver = diffrax.Dopri8()
    controller = diffrax.PIDController(rtol=rtol, atol=atol)
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=1.0,
        dt0=None,
        y0=z0,
        args=(masses, y[3]),
        saveat=diffrax.SaveAt(t1=True),
        stepsize_controller=controller,
        adjoint=diffrax.ForwardMode(),
        max_steps=max_steps,
        throw=True,
    )
    zf = sol.ys[0]
    return zf - z0


@lru_cache(maxsize=8)
def _compiled(rtol: float, atol: float, max_steps: int):
    require_accelerated_x64()

    def closure(y, m3):
        return _closure_impl(y, m3, rtol=rtol, atol=atol, max_steps=max_steps)

    compiled_closure = jax.jit(closure)
    compiled_jacobian = jax.jit(jax.jacfwd(closure, argnums=0))
    return compiled_closure, compiled_jacobian


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
    y_arr = np.asarray(y, dtype=float)
    if y_arr.shape != (6,):
        raise ValueError("continuation vector must have six components")
    if y_arr[3] <= 0.0:
        raise ValueError("period must be positive")
    closure_fn, jacobian_fn = _compiled(float(rtol), float(atol), int(max_steps))
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


def rhs_jacobian(state: Array, masses: Array) -> Array:
    """Autodifferentiate the local reduced vector field for implementation QA."""
    require_accelerated_x64()
    z = jnp.asarray(np.asarray(state, dtype=float), dtype=jnp.float64)
    m = jnp.asarray(np.asarray(masses, dtype=float), dtype=jnp.float64)
    jac = jax.jacfwd(reduced_rhs_jax, argnums=1)(0.0, z, m)
    return np.asarray(jax.device_get(jac), dtype=float)
