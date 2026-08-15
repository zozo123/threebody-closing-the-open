"""Trace connected Floquet stability-boundary components in mass space.

Fast screening only. Given two localized critical samples, use a secant predictor
in (m1,m2) and correct transversely to the zero set of the reduced Floquet
stability score. Every trial mass pair is first Newton-corrected onto the
periodic-orbit family. Publication claims still require the independent BigFloat
verification path.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .boundary import BoundarySample, evaluate
from .liao_family import FamilyPoint, correct_family_point


@dataclass(frozen=True)
class CriticalCurvePoint:
    sample: BoundarySample
    tangent: tuple[float, float]
    correction_offset: float
    bracket_width: float

    @property
    def mass_pair(self) -> tuple[float, float]:
        m1, m2, _ = self.sample.point.masses
        return m1, m2


@dataclass(frozen=True)
class CriticalCurveTrace:
    points: tuple[CriticalCurvePoint, ...]
    stopped_reason: str


def _guess(p: FamilyPoint) -> tuple[float, float, float, float]:
    return p.x1, p.v1, p.v2, p.period


def _at_masses(
    masses: tuple[float, float, float],
    anchor: FamilyPoint,
    *,
    max_residual: float,
) -> BoundarySample:
    point = correct_family_point(masses, _guess(anchor))
    if not point.success or point.residual_norm > max_residual:
        raise RuntimeError(
            f"periodic correction failed at {masses}: {point.residual_norm:.3e}"
        )
    return evaluate(point)


def advance_critical_curve(
    previous: BoundarySample,
    current: BoundarySample,
    *,
    arclength_step: float = 5e-4,
    normal_half_width: float = 1e-3,
    normal_tolerance: float = 1e-8,
    score_tolerance: float = 1e-8,
    max_bisections: int = 32,
    max_residual: float = 1e-7,
) -> CriticalCurvePoint:
    """Advance one screening step along a smooth critical-curve component."""
    p0 = np.asarray(previous.point.masses[:2], dtype=float)
    p1 = np.asarray(current.point.masses[:2], dtype=float)
    secant = p1 - p0
    secant_norm = float(np.linalg.norm(secant))
    if secant_norm == 0.0:
        raise ValueError("critical seeds must have distinct mass pairs")
    tangent = secant / secant_norm
    normal = np.asarray((-tangent[1], tangent[0]))
    predictor = p1 + arclength_step * tangent
    m3 = current.point.masses[2]

    def trial(offset: float, anchor: FamilyPoint) -> BoundarySample:
        pair = predictor + offset * normal
        masses = (float(pair[0]), float(pair[1]), m3)
        return _at_masses(masses, anchor, max_residual=max_residual)

    width = normal_half_width
    left = trial(-width, current.point)
    right = trial(width, current.point)
    for _ in range(6):
        if left.score == 0.0 or right.score == 0.0 or left.score * right.score < 0.0:
            break
        width *= 2.0
        left = trial(-width, left.point)
        right = trial(width, right.point)
    else:
        raise RuntimeError("normal corrector did not bracket the stability zero set")

    if left.score == 0.0:
        best, offset = left, -width
        bracket = 0.0
    elif right.score == 0.0:
        best, offset = right, width
        bracket = 0.0
    else:
        lo, hi = -width, width
        slo, shi = left, right
        best = slo if abs(slo.score) <= abs(shi.score) else shi
        offset = lo if best is slo else hi
        for _ in range(max_bisections):
            bracket = hi - lo
            if bracket <= normal_tolerance or abs(best.score) <= score_tolerance:
                break
            mid = 0.5 * (lo + hi)
            anchor = slo.point if abs(mid - lo) <= abs(hi - mid) else shi.point
            smid = trial(mid, anchor)
            if abs(smid.score) < abs(best.score):
                best, offset = smid, mid
            if slo.score * smid.score <= 0.0:
                hi, shi = mid, smid
            else:
                lo, slo = mid, smid
        bracket = hi - lo

    return CriticalCurvePoint(
        sample=best,
        tangent=(float(tangent[0]), float(tangent[1])),
        correction_offset=float(offset),
        bracket_width=float(bracket),
    )


def trace_critical_curve(
    first: BoundarySample,
    second: BoundarySample,
    *,
    steps: int,
    arclength_step: float = 5e-4,
    **kwargs: float | int,
) -> CriticalCurveTrace:
    """Trace a smooth component from two localized screening seeds."""
    if steps < 0:
        raise ValueError("steps must be non-negative")
    points: list[CriticalCurvePoint] = []
    previous, current = first, second
    reason = "requested_steps_completed"
    for _ in range(steps):
        try:
            nxt = advance_critical_curve(
                previous,
                current,
                arclength_step=arclength_step,
                **kwargs,
            )
        except RuntimeError as exc:
            reason = str(exc)
            break
        points.append(nxt)
        previous, current = current, nxt.sample
    return CriticalCurveTrace(tuple(points), reason)
