"""Direct six-variable solvers for codimension-two Floquet organizer vertices.

The reduced stable trace-root domain has three universal vertices:

* double -1:       t1=t2=-2,  (alpha,beta)=(0,-4)
* mixed +/-1:      {t1,t2}={-2,+2}, (alpha,beta)=(4,4)
* double +1:       t1=t2=+2,  (alpha,beta)=(8,28)

Rather than optimize only (m1,m2) while nesting a periodic-orbit solve inside
every function evaluation, we solve directly in

    y=(x1,v1,v2,T,m1,m2)

with SciPy closure/Floquet values and audited JAX derivative blocks.  The two
smooth event equations used at each vertex are:

    double -1:  P(-2)=0 and Delta=0
    mixed:      P(+2)=0 and P(-2)=0
    double +1:  P(+2)=0 and Delta=0

This remains float64 screening.  A candidate is not a publication result until
independent BigFloat/canonical reproduction passes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import least_squares

from .conditioning import SolveConditioning, condition_report
from .critical_manifold import _flow_for_vector, event_value
from .jax_diffrax import (
    adaptive_closure_and_jacobian,
    adaptive_event_and_gradient,
    adaptive_mixed_system_and_jacobian,
)
from .liao_family import FamilyPoint

Array = np.ndarray
VertexMode = Literal["double_minus_one", "mixed_plus_minus_one", "double_plus_one"]

VERTEX_EVENTS: dict[VertexMode, tuple[str, str]] = {
    "double_minus_one": ("minus_one", "trace_collision"),
    "mixed_plus_minus_one": ("plus_one", "minus_one"),
    "double_plus_one": ("plus_one", "trace_collision"),
}
VERTEX_TARGETS: dict[VertexMode, tuple[float, float]] = {
    "double_minus_one": (0.0, -4.0),
    "mixed_plus_minus_one": (4.0, 4.0),
    "double_plus_one": (8.0, 28.0),
}


@dataclass(frozen=True)
class DirectVertexResult:
    mode: VertexMode
    point: FamilyPoint
    alpha: float
    beta: float
    discriminant: float
    event_values: tuple[float, float]
    invariant_error: float
    nfev: int
    optimality: float
    cost: float
    optimizer_success: bool
    optimizer_message: str
    #: Conditioning of the scaled augmented mixed-vertex Jacobian at the
    #: solution.  A mixed vertex is where two events coincide, so this is the
    #: number that says how degenerate the solve there actually was.
    conditioning: SolveConditioning | None = None

    @property
    def vector(self) -> Array:
        p = self.point
        m1, m2, _ = p.masses
        return np.asarray([p.x1, p.v1, p.v2, p.period, m1, m2], dtype=float)


def _scales(y: Array) -> Array:
    floors = np.asarray([0.05, 0.5, 0.1, 1.0, 0.02, 0.02], dtype=float)
    return np.maximum(np.abs(np.asarray(y, dtype=float)), floors)


def solve_direct_vertex(
    seed: Array,
    mode: VertexMode,
    *,
    m3: float = 1.0,
    mass_bounds: tuple[tuple[float, float], tuple[float, float]],
    max_closure: float = 2e-7,
    max_event: float = 5e-5,
    max_invariant_error: float = 2e-4,
    max_nfev: int = 35,
    screening_rtol: float = 3e-10,
    screening_atol: float = 3e-12,
) -> DirectVertexResult:
    """Solve an isolated Floquet vertex directly in the six-variable chart."""
    if mode not in VERTEX_EVENTS:
        raise ValueError(f"unsupported vertex mode: {mode}")
    y0 = np.asarray(seed, dtype=float)
    if y0.shape != (6,):
        raise ValueError("vertex seed must have six components")
    if y0[3] <= 0.0:
        raise ValueError("seed period must be positive")

    event_a, event_b = VERTEX_EVENTS[mode]
    scales = _scales(y0)
    closure_scale = 1e-6
    event_scale = 2e-4

    def residual(y: Array) -> Array:
        closure, floquet = _flow_for_vector(
            y,
            m3=m3,
            rtol=screening_rtol,
            atol=screening_atol,
        )
        ea = event_value(floquet, event_a)
        eb = event_value(floquet, event_b)
        return np.concatenate((closure / closure_scale, [ea / event_scale, eb / event_scale]))

    def jacobian(y: Array) -> Array:
        if mode == "mixed_plus_minus_one":
            _values, joint = adaptive_mixed_system_and_jacobian(
                y,
                m3=m3,
                rtol=5e-10,
                atol=5e-12,
                max_steps=1 << 18,
            )
            return np.vstack((
                joint[:8] / closure_scale,
                joint[8:10] / event_scale,
            ))
        _closure_value, closure_jac = adaptive_closure_and_jacobian(
            y,
            m3=m3,
            rtol=1e-10,
            atol=1e-12,
            max_steps=1 << 18,
        )
        _ea_value, grad_a = adaptive_event_and_gradient(
            y,
            event_a,
            m3=m3,
            rtol=5e-10,
            atol=5e-12,
            max_steps=1 << 18,
        )
        _eb_value, grad_b = adaptive_event_and_gradient(
            y,
            event_b,
            m3=m3,
            rtol=5e-10,
            atol=5e-12,
            max_steps=1 << 18,
        )
        return np.vstack((
            closure_jac / closure_scale,
            grad_a[None, :] / event_scale,
            grad_b[None, :] / event_scale,
        ))

    (m1_lo, m1_hi), (m2_lo, m2_hi) = mass_bounds
    lower = np.asarray([-2.0, -10.0, -10.0, 0.1, m1_lo, m2_lo], dtype=float)
    upper = np.asarray([2.0, 10.0, 10.0, 20.0, m1_hi, m2_hi], dtype=float)
    start = np.clip(y0, lower + 1e-10, upper - 1e-10)
    fit = least_squares(
        residual,
        start,
        jac=jacobian,
        method="trf",
        bounds=(lower, upper),
        x_scale=scales,
        xtol=2e-11,
        ftol=2e-11,
        gtol=2e-11,
        max_nfev=max_nfev,
    )

    closure, floquet = _flow_for_vector(
        fit.x,
        m3=m3,
        rtol=screening_rtol,
        atol=screening_atol,
    )
    closure_norm = float(np.linalg.norm(closure))
    ea = event_value(floquet, event_a)
    eb = event_value(floquet, event_b)
    target_alpha, target_beta = VERTEX_TARGETS[mode]
    invariant_error = float(np.hypot(
        floquet.alpha - target_alpha,
        floquet.beta - target_beta,
    ))

    if closure_norm > max_closure or max(abs(ea), abs(eb)) > max_event:
        raise RuntimeError(
            f"direct vertex missed residual gates: closure={closure_norm:.3e}, "
            f"events=({ea:.3e},{eb:.3e}), masses=({fit.x[4]:.12g},{fit.x[5]:.12g}), "
            f"nfev={fit.nfev}, optimizer={fit.message}"
        )
    if invariant_error > max_invariant_error:
        raise RuntimeError(
            f"direct vertex missed invariant target: error={invariant_error:.3e}, "
            f"masses=({fit.x[4]:.12g},{fit.x[5]:.12g}), nfev={fit.nfev}, "
            f"optimizer={fit.message}"
        )

    point = FamilyPoint(
        masses=(float(fit.x[4]), float(fit.x[5]), float(m3)),
        x1=float(fit.x[0]),
        v1=float(fit.x[1]),
        v2=float(fit.x[2]),
        period=float(fit.x[3]),
        residual_norm=closure_norm,
        nfev=int(fit.nfev),
        success=True,
    )
    return DirectVertexResult(
        mode=mode,
        point=point,
        alpha=float(floquet.alpha),
        beta=float(floquet.beta),
        discriminant=float(floquet.discriminant),
        event_values=(float(ea), float(eb)),
        invariant_error=invariant_error,
        nfev=int(fit.nfev),
        optimality=float(fit.optimality),
        cost=float(fit.cost),
        optimizer_success=bool(fit.success),
        optimizer_message=str(fit.message),
        conditioning=condition_report(getattr(fit, "jac", None), fit.fun),
    )
