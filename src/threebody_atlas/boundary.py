"""Refine stability boundaries inside a continuation bracket."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .liao_family import FamilyPoint, correct_family_point
from .reduced import ReducedFloquetResult, compute_reduced_floquet


@dataclass(frozen=True)
class BoundarySample:
    point: FamilyPoint
    floquet: ReducedFloquetResult
    score: float


@dataclass(frozen=True)
class BoundaryResult:
    stable_side: BoundarySample
    unstable_side: BoundarySample
    iterations: int
    parameter_width: float


def stability_score(result: ReducedFloquetResult) -> float:
    """Positive in the strict elliptic region, negative outside it.

    The zero set captures the discriminant or |T_i|=2 boundaries used by the
    reduced trace criterion.  It is a localization score, not an uncertainty
    estimate.
    """
    t1, t2 = result.trace_roots
    return float(min(result.discriminant, 2.0 - abs(t1), 2.0 - abs(t2)))


def evaluate(point: FamilyPoint) -> BoundarySample:
    floquet = compute_reduced_floquet(point.state(), np.asarray(point.masses), point.period)
    return BoundarySample(point, floquet, stability_score(floquet))


def refine_m2_boundary(
    stable: FamilyPoint,
    unstable: FamilyPoint,
    *,
    m2_tolerance: float = 1e-8,
    max_iterations: int = 40,
) -> BoundaryResult:
    """Bisect an m2 stability bracket at fixed m1,m3 with shooting correction."""
    if stable.masses[0] != unstable.masses[0] or stable.masses[2] != unstable.masses[2]:
        raise ValueError("m1 and m3 must be fixed across the boundary bracket")
    stable_sample = evaluate(stable)
    unstable_sample = evaluate(unstable)
    if stable_sample.score <= 0:
        raise ValueError("stable endpoint does not satisfy the screening stability criterion")
    if unstable_sample.score >= 0:
        raise ValueError("unstable endpoint does not bracket the screening stability criterion")

    iterations = 0
    while abs(stable_sample.point.masses[1] - unstable_sample.point.masses[1]) > m2_tolerance:
        if iterations >= max_iterations:
            break
        iterations += 1
        m2 = 0.5 * (stable_sample.point.masses[1] + unstable_sample.point.masses[1])
        # Seed from the closer endpoint in chart coordinates.
        anchor = stable_sample.point if abs(m2 - stable_sample.point.masses[1]) <= abs(m2 - unstable_sample.point.masses[1]) else unstable_sample.point
        masses = (anchor.masses[0], m2, anchor.masses[2])
        guess = (anchor.x1, anchor.v1, anchor.v2, anchor.period)
        corrected = correct_family_point(masses, guess)
        if not corrected.success or corrected.residual_norm > 1e-7:
            raise RuntimeError(f"shooting failed near m2={m2:.12g}: {corrected.residual_norm:.3e}")
        sample = evaluate(corrected)
        if sample.score > 0:
            stable_sample = sample
        else:
            unstable_sample = sample

    return BoundaryResult(
        stable_side=stable_sample,
        unstable_side=unstable_sample,
        iterations=iterations,
        parameter_width=abs(stable_sample.point.masses[1] - unstable_sample.point.masses[1]),
    )
