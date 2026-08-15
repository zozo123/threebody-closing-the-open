"""Branch-preserving pseudo-arclength continuation in the two mass variables.

This is deliberately independent of the six-variable augmented corrector in
``critical_manifold.py``.  The periodic orbit is treated as an implicit chart
p=p(m1,m2): every mass-plane residual evaluation first Newton-corrects the
published shooting variables onto the periodic sheet, then evaluates one smooth
Floquet event.  The outer corrector therefore solves only

    event(m1,m2) = 0,
    arclength(m1,m2) = 0.

Agreement between this nested formulation and the full-state augmented
formulation is an implementation-independence check.  Publication claims still
require the Julia BigFloat path.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .boundary import BoundarySample, evaluate
from .critical_manifold import EventMode, LocalizedCriticalPoint, event_value
from .liao_family import FamilyPoint, correct_family_point

Array = np.ndarray


@dataclass(frozen=True)
class MassPlaneCriticalPoint:
    sample: BoundarySample
    event_mode: EventMode
    event_value: float
    mass_tangent: tuple[float, float]
    arclength_residual: float
    step: float
    outer_nfev: int

    @property
    def mass_pair(self) -> Array:
        return np.asarray(self.sample.point.masses[:2], dtype=float)


@dataclass(frozen=True)
class MassPlaneTrace:
    points: tuple[MassPlaneCriticalPoint, ...]
    stopped_reason: str


def _params(point: FamilyPoint) -> Array:
    return np.asarray([point.x1, point.v1, point.v2, point.period], dtype=float)


def _mass_pair(point: LocalizedCriticalPoint | MassPlaneCriticalPoint) -> Array:
    if isinstance(point, LocalizedCriticalPoint):
        return np.asarray(point.sample.point.masses[:2], dtype=float)
    return point.mass_pair


def _family(point: LocalizedCriticalPoint | MassPlaneCriticalPoint) -> FamilyPoint:
    return point.sample.point


def _mode(point: LocalizedCriticalPoint | MassPlaneCriticalPoint) -> EventMode:
    return point.event_mode


def advance_massplane_critical(
    previous: LocalizedCriticalPoint | MassPlaneCriticalPoint,
    current: LocalizedCriticalPoint | MassPlaneCriticalPoint,
    *,
    step: float = 5e-4,
    max_closure: float = 1e-7,
    max_event: float = 2e-6,
    max_arc: float = 2e-6,
    max_outer_nfev: int = 20,
    max_inner_nfev: int = 45,
) -> MassPlaneCriticalPoint:
    """Advance one nested pseudo-arclength step on a smooth critical curve."""
    if _mode(previous) != _mode(current):
        raise ValueError("critical seed event modes differ")
    mode = _mode(current)
    mp = _mass_pair(previous)
    mc = _mass_pair(current)
    secant = mc - mp
    norm = float(np.linalg.norm(secant))
    if norm == 0.0:
        raise ValueError("critical seeds must have distinct mass pairs")
    tangent = secant / norm
    predictor = mc + step * tangent

    fp = _family(previous)
    fc = _family(current)
    pp = _params(fp)
    pc = _params(fc)
    # Deterministic local affine predictor for the inner shooting solve.  The
    # predictor follows the previously observed critical-curve secant and does
    # not depend on optimizer evaluation order, which keeps the outer residual
    # deterministic.
    dparam_ds = (pc - pp) / norm
    m3 = fc.masses[2]
    cache: dict[tuple[float, float], tuple[FamilyPoint, BoundarySample]] = {}

    def corrected(pair: Array) -> tuple[FamilyPoint, BoundarySample]:
        key = (float(pair[0]), float(pair[1]))
        if key in cache:
            return cache[key]
        ds = float(np.dot(pair - mc, tangent))
        guess = pc + ds * dparam_ds
        masses = (key[0], key[1], m3)
        point = correct_family_point(
            masses,
            tuple(float(x) for x in guess),
            max_nfev=max_inner_nfev,
        )
        if not point.success or point.residual_norm > max_closure:
            raise RuntimeError(
                f"inner periodic correction failed at {masses}: residual={point.residual_norm:.3e}"
            )
        sample = evaluate(point)
        cache[key] = (point, sample)
        return point, sample

    event_scale = 2e-4
    arc_scale = max(step, 1e-5)

    def residual(pair: Array) -> Array:
        _, sample = corrected(pair)
        event = event_value(sample.floquet, mode)
        arc = float(np.dot(pair - predictor, tangent))
        return np.asarray([event / event_scale, arc / arc_scale], dtype=float)

    lower = np.asarray([0.5, 0.5], dtype=float)
    upper = np.asarray([1.5, 1.5], dtype=float)
    fit = least_squares(
        residual,
        predictor,
        bounds=(lower, upper),
        method="trf",
        x_scale=np.asarray([0.1, 0.1]),
        xtol=5e-11,
        ftol=5e-11,
        gtol=5e-11,
        max_nfev=max_outer_nfev,
    )
    point, sample = corrected(fit.x)
    critical = event_value(sample.floquet, mode)
    arc = float(np.dot(fit.x - predictor, tangent))
    if not fit.success:
        raise RuntimeError(f"outer mass-plane corrector failed: {fit.message}")
    if point.residual_norm > max_closure or abs(critical) > max_event or abs(arc) > max_arc:
        raise RuntimeError(
            "mass-plane corrector missed acceptance gates: "
            f"closure={point.residual_norm:.3e}, event={critical:.3e}, arc={arc:.3e}"
        )
    return MassPlaneCriticalPoint(
        sample=sample,
        event_mode=mode,
        event_value=float(critical),
        mass_tangent=(float(tangent[0]), float(tangent[1])),
        arclength_residual=float(arc),
        step=float(step),
        outer_nfev=int(fit.nfev),
    )


def trace_massplane_critical(
    first: LocalizedCriticalPoint,
    second: LocalizedCriticalPoint,
    *,
    steps: int,
    step: float = 5e-4,
    min_step: float = 6.25e-5,
    max_retries: int = 4,
) -> MassPlaneTrace:
    """Trace a component with adaptive nested pseudo-arclength continuation."""
    if first.event_mode != second.event_mode:
        raise ValueError("localized seeds identify different critical events")
    previous: LocalizedCriticalPoint | MassPlaneCriticalPoint = first
    current: LocalizedCriticalPoint | MassPlaneCriticalPoint = second
    out: list[MassPlaneCriticalPoint] = []
    requested = float(step)
    current_step = requested
    reason = "requested_steps_completed"
    for _ in range(steps):
        accepted: MassPlaneCriticalPoint | None = None
        last_error: Exception | None = None
        trial = current_step
        for _retry in range(max_retries + 1):
            try:
                accepted = advance_massplane_critical(previous, current, step=trial)
                break
            except (RuntimeError, ValueError) as exc:
                last_error = exc
                trial *= 0.5
                if trial < min_step:
                    break
        if accepted is None:
            reason = f"mass-plane correction failed: {last_error}"
            break
        out.append(accepted)
        previous, current = current, accepted
        current_step = min(requested, trial * 1.25)
    return MassPlaneTrace(tuple(out), reason)
