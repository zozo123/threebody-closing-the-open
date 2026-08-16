"""Hybrid SciPy-residual/JAX-Jacobian pseudo-arclength continuation.

The authoritative screening residual is always evaluated by the existing SciPy
DOP853 + variational path.  JAX x64 + Diffrax is used only for derivatives after
its Jacobian blocks have passed the independent derivative audit.

At each accepted point we form the critical residual Jacobian

    d[closure, event] / d(x1,v1,v2,T,m1,m2),

extract its scaled one-dimensional null direction by SVD, orient that tangent
with the previous secant, predict along it, and solve closure + event +
pseudo-arclength simultaneously.  This is the geometry of the critical curve
itself, rather than a mass-plane secant/normal heuristic.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .boundary import BoundarySample, stability_score
from .conditioning import SolveConditioning, condition_report
from .critical_geometry import continuation_scales, critical_tangent
from .critical_manifold import (
    AugmentedCriticalPoint,
    AugmentedCriticalTrace,
    LocalizedCriticalPoint,
    _flow_for_vector,
    event_value,
)
from .jax_diffrax import adaptive_closure_and_jacobian, adaptive_event_and_gradient
from .liao_family import FamilyPoint

Array = np.ndarray


@dataclass(frozen=True)
class HybridJacobianDiagnostics:
    null_residual: float
    spectral_gap: float
    singular_values: tuple[float, ...]
    #: Conditioning of the unscaled critical residual Jacobian
    #: d[closure, event]/d(x1,v1,v2,T,m1,m2) at the accepted point.  This is the
    #: geometry of the critical curve itself.
    predictor_conditioning: SolveConditioning | None = None
    #: Conditioning of the scaled augmented corrector Jacobian (closure, event,
    #: arclength) the least-squares solve actually saw, paired with its residual.
    #: ``displacement_bound`` is the backward-error displacement in the scaled
    #: continuation variables.
    corrector_conditioning: SolveConditioning | None = None


def _derivative_blocks(y: Array, mode: str, m3: float) -> tuple[Array, Array]:
    """Return JAX derivative blocks; never return JAX residual values."""
    _jax_closure, closure_jac = adaptive_closure_and_jacobian(
        y,
        m3=m3,
        rtol=1e-10,
        atol=1e-12,
        max_steps=1 << 18,
    )
    _jax_event, event_grad = adaptive_event_and_gradient(
        y,
        mode,
        m3=m3,
        rtol=5e-10,
        atol=5e-12,
        max_steps=1 << 18,
    )
    return closure_jac, event_grad


def advance_hybrid_critical(
    previous: LocalizedCriticalPoint | AugmentedCriticalPoint,
    current: LocalizedCriticalPoint | AugmentedCriticalPoint,
    *,
    normalized_step: float = 2e-3,
    max_closure: float = 2e-7,
    max_event: float = 2e-6,
    max_arc: float = 2e-4,
    max_nfev: int = 35,
    screening_rtol: float = 3e-10,
    screening_atol: float = 3e-12,
) -> tuple[AugmentedCriticalPoint, HybridJacobianDiagnostics]:
    """Advance one critical-curve step with SciPy values and JAX derivatives."""
    if previous.event_mode != current.event_mode:
        raise ValueError("critical seed event modes differ")
    if normalized_step <= 0.0:
        raise ValueError("normalized_step must be positive")

    mode = current.event_mode
    m3 = float(current.sample.point.masses[2])
    yp = previous.vector
    yc = current.vector
    scales = continuation_scales(yc)

    reference = yc - yp
    if float(np.linalg.norm(reference)) == 0.0:
        raise ValueError("critical seeds must be distinct in continuation space")

    closure_jac0, event_grad0 = _derivative_blocks(yc, mode, m3)
    critical_jac0 = np.vstack((closure_jac0, event_grad0[None, :]))
    tangent_info = critical_tangent(
        critical_jac0,
        scales=scales,
        reference=reference,
    )
    tangent_scaled = tangent_info.scaled
    predictor = yc + scales * normalized_step * tangent_scaled

    closure_scale = 1e-6
    event_scale = 2e-4
    arc_scale = max(normalized_step, 1e-4)

    def residual(y: Array) -> Array:
        # Values are authoritative SciPy values. JAX values are never substituted.
        closure, floquet = _flow_for_vector(
            y,
            m3=m3,
            rtol=screening_rtol,
            atol=screening_atol,
        )
        critical = event_value(floquet, mode)
        arc = float(np.dot((y - predictor) / scales, tangent_scaled))
        return np.concatenate(
            (closure / closure_scale, [critical / event_scale, arc / arc_scale])
        )

    def jacobian(y: Array) -> Array:
        closure_jac, event_grad = _derivative_blocks(y, mode, m3)
        arc_grad = tangent_scaled / scales
        return np.vstack(
            (
                closure_jac / closure_scale,
                event_grad[None, :] / event_scale,
                arc_grad[None, :] / arc_scale,
            )
        )

    lower = np.asarray([-2.0, -10.0, -10.0, 0.1, 0.5, 0.5], dtype=float)
    upper = np.asarray([2.0, 10.0, 10.0, 20.0, 1.5, 1.5], dtype=float)
    fit = least_squares(
        residual,
        predictor,
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
    critical = event_value(floquet, mode)
    arc = float(np.dot((fit.x - predictor) / scales, tangent_scaled))
    if not fit.success:
        raise RuntimeError(f"hybrid augmented least-squares failed: {fit.message}")
    if closure_norm > max_closure or abs(critical) > max_event or abs(arc) > max_arc:
        raise RuntimeError(
            "hybrid critical correction missed acceptance gates: "
            f"closure={closure_norm:.3e}, event={critical:.3e}, arc={arc:.3e}"
        )

    point = FamilyPoint(
        masses=(float(fit.x[4]), float(fit.x[5]), m3),
        x1=float(fit.x[0]),
        v1=float(fit.x[1]),
        v2=float(fit.x[2]),
        period=float(fit.x[3]),
        residual_norm=closure_norm,
        nfev=int(fit.nfev),
        success=True,
    )
    sample = BoundarySample(point, floquet, stability_score(floquet))
    accepted = AugmentedCriticalPoint(
        sample=sample,
        event_mode=mode,
        event_value=float(critical),
        tangent_scaled=tuple(float(x) for x in tangent_scaled),
        arclength_residual=float(arc),
        normalized_step=float(normalized_step),
        nfev=int(fit.nfev),
    )
    diagnostics = HybridJacobianDiagnostics(
        null_residual=float(tangent_info.null_residual),
        spectral_gap=float(tangent_info.spectral_gap),
        singular_values=tuple(float(x) for x in tangent_info.singular_values),
        predictor_conditioning=condition_report(
            critical_jac0,
            np.concatenate((closure, [critical])),
        ),
        corrector_conditioning=condition_report(getattr(fit, "jac", None), fit.fun),
    )
    return accepted, diagnostics


def trace_hybrid_critical(
    first: LocalizedCriticalPoint,
    second: LocalizedCriticalPoint,
    *,
    steps: int,
    normalized_step: float = 2e-3,
    min_step: float = 1.25e-4,
    max_retries: int = 5,
) -> tuple[AugmentedCriticalTrace, tuple[HybridJacobianDiagnostics, ...]]:
    """Trace a critical component with adaptive hybrid pseudo-arclength steps."""
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if first.event_mode != second.event_mode:
        raise ValueError("localized seeds identify different critical events")

    previous: LocalizedCriticalPoint | AugmentedCriticalPoint = first
    current: LocalizedCriticalPoint | AugmentedCriticalPoint = second
    points: list[AugmentedCriticalPoint] = []
    diagnostics: list[HybridJacobianDiagnostics] = []
    step = normalized_step
    reason = "requested_steps_completed"

    for _ in range(steps):
        last_error: Exception | None = None
        accepted: AugmentedCriticalPoint | None = None
        diag: HybridJacobianDiagnostics | None = None
        trial_step = step
        for _retry in range(max_retries + 1):
            try:
                accepted, diag = advance_hybrid_critical(
                    previous,
                    current,
                    normalized_step=trial_step,
                )
                break
            except (RuntimeError, ValueError, FloatingPointError) as exc:
                last_error = exc
                trial_step *= 0.5
                if trial_step < min_step:
                    break
        if accepted is None or diag is None:
            reason = f"hybrid pseudo-arclength correction failed: {last_error}"
            break
        points.append(accepted)
        diagnostics.append(diag)
        previous, current = current, accepted
        step = min(normalized_step, trial_step * 1.25)

    return AugmentedCriticalTrace(tuple(points), reason), tuple(diagnostics)
