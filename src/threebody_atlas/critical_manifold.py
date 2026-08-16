"""Augmented pseudo-arclength continuation of linear-stability critical sets.

This is a screening implementation, not the publication truth path.  The key
idea is to continue a *smooth* Floquet event equation together with periodic
closure.  We do not continue the nonsmooth minimum stability score.

For the two nontrivial reciprocal multiplier pairs, let t=lambda+1/lambda be
roots of

    P(t) = t^2 - (alpha-4)t + beta - 4alpha + 8.

Then useful smooth codimension-one event functions are

    P(+2) = beta - 6 alpha + 20       (nontrivial pair at lambda=+1)
    P(-2) = beta - 2 alpha + 4        (nontrivial pair at lambda=-1)
    Delta  = discriminant(P)           (collision of the two trace roots).

The continuation state is y=(x1,v1,v2,T,m1,m2), with m3 fixed.  Each corrector
solves periodic closure, one selected event equation, and one pseudo-arclength
condition simultaneously.  Independent Julia BigFloat verification and the
canonical Jacobi/Krein path remain required before publication claims.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from .boundary import BoundarySample, evaluate, stability_score
from .conditioning import SolveConditioning, scalar_condition_report
from .liao_family import FamilyPoint, correct_family_point, state_from_chart
from .reduced import (
    ReducedFloquetResult,
    compute_reduced_floquet,
    full_to_reduced,
    reduced_jacobian,
    reduced_rhs,
    stability_invariants,
)

Array = np.ndarray
EventMode = Literal["plus_one", "minus_one", "trace_collision"]
_EVENT_MODES: tuple[EventMode, ...] = ("plus_one", "minus_one", "trace_collision")


def classify_localized_cell(
    *,
    closure: float,
    event: float,
    m2: float,
    lo: float,
    hi: float,
    max_closure: float = 1e-7,
    event_tolerance: float = 2e-8,
    bracket_slop: float = 2e-9,
) -> str:
    """Classify one 1-D cell localization without aborting a 620-cell census."""
    if not (lo - bracket_slop <= m2 <= hi + bracket_slop):
        return "outside_bracket"
    if closure > max_closure:
        return "missed_closure"
    if abs(event) > event_tolerance:
        return "missed_event"
    return "ok"


@dataclass(frozen=True)
class LocalizedCriticalPoint:
    sample: BoundarySample
    event_mode: EventMode
    event_value: float
    source_width: float
    #: Conditioning of the one-dimensional event solve in m2.  ``event_value``
    #: is a backward error; ``conditioning.displacement_bound`` converts it into
    #: the m2 uncertainty it actually buys.  ``None`` when not requested, since
    #: measuring d(event)/d(m2) costs two extra corrected Floquet evaluations.
    conditioning: SolveConditioning | None = None

    @property
    def vector(self) -> Array:
        p = self.sample.point
        m1, m2, _ = p.masses
        return np.asarray([p.x1, p.v1, p.v2, p.period, m1, m2], dtype=float)


@dataclass(frozen=True)
class AugmentedCriticalPoint:
    sample: BoundarySample
    event_mode: EventMode
    event_value: float
    tangent_scaled: tuple[float, float, float, float, float, float]
    arclength_residual: float
    normalized_step: float
    nfev: int

    @property
    def vector(self) -> Array:
        p = self.sample.point
        m1, m2, _ = p.masses
        return np.asarray([p.x1, p.v1, p.v2, p.period, m1, m2], dtype=float)


@dataclass(frozen=True)
class AugmentedCriticalTrace:
    points: tuple[AugmentedCriticalPoint, ...]
    stopped_reason: str


def event_value(result: ReducedFloquetResult, mode: EventMode) -> float:
    """Return a smooth scalar whose zero set is a Floquet critical event."""
    if mode == "plus_one":
        return float(result.beta - 6.0 * result.alpha + 20.0)
    if mode == "minus_one":
        return float(result.beta - 2.0 * result.alpha + 4.0)
    if mode == "trace_collision":
        return float(result.discriminant)
    raise ValueError(f"unsupported event mode: {mode}")


def infer_event_mode(a: BoundarySample, b: BoundarySample) -> EventMode:
    """Infer which smooth critical equation changes sign across an S/U bracket."""
    crossings: list[tuple[float, EventMode]] = []
    for mode in _EVENT_MODES:
        va = event_value(a.floquet, mode)
        vb = event_value(b.floquet, mode)
        if va == 0.0 or vb == 0.0 or va * vb < 0.0:
            # Prefer the crossing already closest to zero at the supplied bracket.
            crossings.append((max(abs(va), abs(vb)), mode))
    if not crossings:
        details = {
            mode: (event_value(a.floquet, mode), event_value(b.floquet, mode))
            for mode in _EVENT_MODES
        }
        raise RuntimeError(f"no smooth Floquet event changes sign across bracket: {details}")
    crossings.sort(key=lambda item: item[0])
    return crossings[0][1]


def _interpolate_guess(a: FamilyPoint, b: FamilyPoint, m2: float) -> tuple[float, float, float, float]:
    ma = a.masses[1]
    mb = b.masses[1]
    if ma == mb:
        return a.x1, a.v1, a.v2, a.period
    theta = (m2 - ma) / (mb - ma)
    pa = np.asarray([a.x1, a.v1, a.v2, a.period], dtype=float)
    pb = np.asarray([b.x1, b.v1, b.v2, b.period], dtype=float)
    p = (1.0 - theta) * pa + theta * pb
    return tuple(float(x) for x in p)


def event_slope_in_m2(
    sample: BoundarySample,
    mode: EventMode,
    *,
    max_closure: float = 1e-7,
    step: float = 1e-7,
    precise: bool = True,
) -> float | None:
    """Central-difference d(event)/d(m2) at a corrected sample.

    Each evaluation re-corrects the periodic orbit at the shifted mass, so this
    is the derivative along the family, not a partial derivative at frozen
    chart coordinates.  Costs two corrected Floquet evaluations.
    """
    ev = _precise_evaluate if precise else evaluate
    m1, m2, m3 = (float(x) for x in sample.point.masses)
    guess = (sample.point.x1, sample.point.v1, sample.point.v2, sample.point.period)
    values: list[float] = []
    for shifted in (m2 + step, m2 - step):
        point = correct_family_point((m1, shifted, m3), guess, max_nfev=60)
        if not point.success or point.residual_norm > max_closure:
            return None
        values.append(event_value(ev(point).floquet, mode))
    return (values[0] - values[1]) / (2.0 * step)


def attach_event_conditioning(
    localized: LocalizedCriticalPoint,
    *,
    max_closure: float = 1e-7,
    step: float = 1e-7,
    precise: bool = True,
) -> LocalizedCriticalPoint:
    """Return ``localized`` with its 1-D event-solve conditioning filled in."""
    slope = event_slope_in_m2(
        localized.sample,
        localized.event_mode,
        max_closure=max_closure,
        step=step,
        precise=precise,
    )
    if slope is None:
        return localized
    return LocalizedCriticalPoint(
        localized.sample,
        localized.event_mode,
        localized.event_value,
        localized.source_width,
        scalar_condition_report(slope, localized.event_value),
    )


def localize_critical_point(
    stable: FamilyPoint,
    unstable: FamilyPoint,
    *,
    event_mode: EventMode | None = None,
    m2_tolerance: float = 2e-9,
    event_tolerance: float = 2e-8,
    max_iterations: int = 40,
    max_closure: float = 1e-7,
    conditioning: bool = False,
) -> LocalizedCriticalPoint:
    """Project an S/U mass-slice bracket onto one smooth Floquet event.

    With ``conditioning=True`` the returned point also carries the conditioning
    of its own solve, so the event residual is never reported alone.
    """
    localized = _localize_critical_point(
        stable,
        unstable,
        event_mode=event_mode,
        m2_tolerance=m2_tolerance,
        event_tolerance=event_tolerance,
        max_iterations=max_iterations,
        max_closure=max_closure,
    )
    if not conditioning:
        return localized
    return attach_event_conditioning(localized, max_closure=max_closure)


def _localize_critical_point(
    stable: FamilyPoint,
    unstable: FamilyPoint,
    *,
    event_mode: EventMode | None = None,
    m2_tolerance: float = 2e-9,
    event_tolerance: float = 2e-8,
    max_iterations: int = 40,
    max_closure: float = 1e-7,
) -> LocalizedCriticalPoint:
    """Illinois/Newton localization proper.  See ``localize_critical_point``."""
    a = evaluate(stable)
    b = evaluate(unstable)
    mode = event_mode or infer_event_mode(a, b)
    va = event_value(a.floquet, mode)
    vb = event_value(b.floquet, mode)
    if va != 0.0 and vb != 0.0 and va * vb > 0.0:
        raise RuntimeError(f"{mode} does not bracket zero: {va:.6e}, {vb:.6e}")

    left, right = (a, b) if a.point.masses[1] < b.point.masses[1] else (b, a)
    published_left, published_right = left, right
    lo0, hi0 = float(left.point.masses[1]), float(right.point.masses[1])
    vl = event_value(left.floquet, mode)
    vr = event_value(right.floquet, mode)
    best = left if abs(vl) <= abs(vr) else right
    best_v = vl if best is left else vr
    last_side = 0

    for _ in range(max_iterations):
        width = right.point.masses[1] - left.point.masses[1]
        if abs(best_v) <= event_tolerance:
            break
        if width <= m2_tolerance:
            # Width is not the scientific gate.  Cheap Illinois cannot split
            # further; hand the published cell to the precise refiner.
            break
        denom = vr - vl
        if denom != 0.0:
            secant = left.point.masses[1] - vl * width / denom
        else:
            secant = 0.5 * (left.point.masses[1] + right.point.masses[1])
        guard = 0.05 * width
        m2 = float(np.clip(secant, left.point.masses[1] + guard, right.point.masses[1] - guard))
        masses = (left.point.masses[0], m2, left.point.masses[2])
        guess = _interpolate_guess(left.point, right.point, m2)
        point = correct_family_point(masses, guess, max_nfev=40)
        if not point.success or point.residual_norm > max_closure:
            m2 = 0.5 * (left.point.masses[1] + right.point.masses[1])
            anchor = left.point if m2 - left.point.masses[1] <= right.point.masses[1] - m2 else right.point
            point = correct_family_point(
                (anchor.masses[0], m2, anchor.masses[2]),
                (anchor.x1, anchor.v1, anchor.v2, anchor.period),
                max_nfev=50,
            )
        if not point.success or point.residual_norm > max_closure:
            raise RuntimeError(f"periodic correction failed while localizing {mode} at m2={m2:.12g}")
        mid = evaluate(point)
        vm = event_value(mid.floquet, mode)
        if abs(vm) < abs(best_v):
            best, best_v = mid, vm
        if vl == 0.0 or vr == 0.0 or abs(best_v) <= event_tolerance:
            break
        if vl * vm <= 0.0:
            right, vr = mid, vm
            if last_side == 1:
                vl *= 0.5
            last_side = 1
        else:
            left, vl = mid, vm
            if last_side == -1:
                vr *= 0.5
            last_side = -1

    width = right.point.masses[1] - left.point.masses[1]
    final = left if abs(vl) <= abs(vr) else right
    if abs(best_v) < abs(event_value(final.floquet, mode)):
        final = best
    polished = _polish_event_root(
        final,
        mode,
        event_tolerance=event_tolerance,
        max_closure=max_closure,
        m2_bounds=(lo0, hi0),
    )
    chosen = polished.sample if abs(polished.event_value) <= abs(event_value(final.floquet, mode)) else final
    # One tight Floquet re-evaluation so a 5e-8 screening residual can still
    # fall under the 2e-8 gate without paying that cost on every Illinois step.
    tight = _precise_evaluate(chosen.point if hasattr(chosen, "point") else chosen)
    tight_v = event_value(tight.floquet, mode)
    if abs(tight_v) <= event_tolerance:
        return LocalizedCriticalPoint(tight, mode, float(tight_v), float(width))
    refined = _polish_event_root(
        tight,
        mode,
        event_tolerance=event_tolerance,
        max_closure=max_closure,
        max_steps=8,
        m2_bounds=(lo0, hi0),
        precise=True,
    )
    if abs(refined.event_value) <= event_tolerance:
        return LocalizedCriticalPoint(refined.sample, mode, float(refined.event_value), float(width))
    recovered = _precise_bracket_search(
        published_left,
        published_right,
        mode,
        seed=refined.sample if abs(refined.event_value) <= abs(tight_v) else tight,
        event_tolerance=event_tolerance,
        max_closure=max_closure,
    )
    candidates = [
        LocalizedCriticalPoint(tight, mode, float(tight_v), float(width)),
        refined,
        recovered,
    ]
    return min(candidates, key=lambda item: abs(item.event_value))


def _evaluate_at_m2(
    left: FamilyPoint,
    right: FamilyPoint,
    m2: float,
    *,
    max_closure: float,
    precise: bool,
) -> BoundarySample:
    masses = (float(left.masses[0]), float(m2), float(left.masses[2]))
    guess = _interpolate_guess(left, right, m2)
    point = correct_family_point(masses, guess, max_nfev=50)
    if not point.success or point.residual_norm > max_closure:
        anchor = left if abs(m2 - left.masses[1]) <= abs(m2 - right.masses[1]) else right
        point = correct_family_point(
            (float(anchor.masses[0]), float(m2), float(anchor.masses[2])),
            (anchor.x1, anchor.v1, anchor.v2, anchor.period),
            max_nfev=60,
        )
    if not point.success or point.residual_norm > max_closure:
        raise RuntimeError(f"periodic correction failed at m2={m2:.12g}")
    return _precise_evaluate(point) if precise else evaluate(point)


def _precise_bracket_search(
    published_left: BoundarySample,
    published_right: BoundarySample,
    mode: EventMode,
    *,
    seed: BoundarySample,
    event_tolerance: float,
    max_closure: float,
    max_steps: int = 12,
) -> LocalizedCriticalPoint:
    """Re-open the published cell and localize with tight Floquet only.

    Cheap Illinois can collapse a 1e-3 mass cell to 1e-14 while the event is
    still 1e-6.  The scientific object is the sign-changing published bracket,
    so rebuild that bracket at tight tolerance and Illinois/bisect there.
    """
    left_pt, right_pt = published_left.point, published_right.point
    lo = float(left_pt.masses[1])
    hi = float(right_pt.masses[1])
    left = _evaluate_at_m2(left_pt, right_pt, lo, max_closure=max_closure, precise=True)
    right = _evaluate_at_m2(left_pt, right_pt, hi, max_closure=max_closure, precise=True)
    vl = event_value(left.floquet, mode)
    vr = event_value(right.floquet, mode)
    seed_v = event_value(seed.floquet, mode)
    if vl * vr > 0.0 and seed_v * vl <= 0.0:
        right, vr = seed, seed_v
        hi = float(seed.point.masses[1])
        if hi < lo:
            left, right = right, left
            vl, vr = vr, vl
            lo, hi = hi, lo
    elif vl * vr > 0.0 and seed_v * vr <= 0.0:
        left, vl = seed, seed_v
        lo = float(seed.point.masses[1])
        if hi < lo:
            left, right = right, left
            vl, vr = vr, vl
            lo, hi = hi, lo
    best = left if abs(vl) <= abs(vr) else right
    best_v = vl if best is left else vr
    if abs(seed_v) < abs(best_v):
        best, best_v = seed, seed_v
    if abs(best_v) <= event_tolerance:
        return LocalizedCriticalPoint(best, mode, float(best_v), hi - lo)

    for _ in range(max_steps):
        width = float(right.point.masses[1] - left.point.masses[1])
        if abs(best_v) <= event_tolerance:
            break
        if width <= 1e-14:
            break
        denom = vr - vl
        if denom != 0.0 and width > 1e-10:
            secant = left.point.masses[1] - vl * width / denom
        else:
            secant = 0.5 * (left.point.masses[1] + right.point.masses[1])
        # No Illinois damping. Cheap-loop damping is what collapsed hard +1 cells.
        guard = 0.05 * width if width > 1e-10 else 0.0
        if guard > 0.0:
            m2 = float(np.clip(secant, left.point.masses[1] + guard, right.point.masses[1] - guard))
        else:
            m2 = float(np.clip(secant, left.point.masses[1], right.point.masses[1]))
        mid = _evaluate_at_m2(left.point, right.point, m2, max_closure=max_closure, precise=True)
        vm = event_value(mid.floquet, mode)
        if abs(vm) < abs(best_v):
            best, best_v = mid, vm
        if abs(best_v) <= event_tolerance:
            break
        if vl * vm <= 0.0:
            right, vr = mid, vm
        else:
            left, vl = mid, vm

    polished = _polish_event_root(
        best,
        mode,
        event_tolerance=event_tolerance,
        max_closure=max_closure,
        max_steps=8,
        m2_bounds=(float(published_left.point.masses[1]), float(published_right.point.masses[1])),
        precise=True,
    )
    return polished if abs(polished.event_value) <= abs(best_v) else LocalizedCriticalPoint(
        best, mode, float(best_v), float(right.point.masses[1] - left.point.masses[1])
    )


def _precise_evaluate(point: FamilyPoint) -> BoundarySample:
    """Recompute Floquet tightly so event Newton is not limited by monodromy noise."""
    floquet = compute_reduced_floquet(
        point.state(),
        np.asarray(point.masses, dtype=float),
        point.period,
        rtol=5e-13,
        atol=5e-15,
    )
    return BoundarySample(point, floquet, stability_score(floquet))


def _polish_event_root(
    sample: BoundarySample,
    mode: EventMode,
    *,
    event_tolerance: float,
    max_closure: float,
    max_steps: int = 4,
    m2_bounds: tuple[float, float] | None = None,
    precise: bool = False,
) -> LocalizedCriticalPoint:
    """In-bracket 1-D Newton. Never leave the published cell."""
    ev = _precise_evaluate if precise else evaluate
    current = sample
    value = event_value(current.floquet, mode)
    if abs(value) <= event_tolerance:
        return LocalizedCriticalPoint(current, mode, float(value), 0.0)

    m1, m2, m3 = (float(x) for x in current.point.masses)
    lo, hi = m2_bounds if m2_bounds is not None else (m2 - 5e-4, m2 + 5e-4)
    for _ in range(max_steps):
        step = max(1e-10, 1e-8 * max(abs(m2), 1.0))
        plus = correct_family_point(
            (m1, min(m2 + step, hi), m3),
            (current.point.x1, current.point.v1, current.point.v2, current.point.period),
            max_nfev=30,
        )
        if not plus.success or plus.residual_norm > max_closure:
            break
        plus_sample = ev(plus)
        if precise and m2 - step >= lo:
            minus = correct_family_point(
                (m1, m2 - step, m3),
                (current.point.x1, current.point.v1, current.point.v2, current.point.period),
                max_nfev=30,
            )
            if minus.success and minus.residual_norm <= max_closure:
                minus_sample = ev(minus)
                slope = (
                    event_value(plus_sample.floquet, mode)
                    - event_value(minus_sample.floquet, mode)
                ) / (2.0 * step)
            else:
                slope = (event_value(plus_sample.floquet, mode) - value) / step
        else:
            slope = (event_value(plus_sample.floquet, mode) - value) / step
        if slope == 0.0 or not np.isfinite(slope):
            break
        nxt_m2 = float(np.clip(m2 - value / slope, lo, hi))
        if not np.isfinite(nxt_m2) or abs(nxt_m2 - m2) < 1e-14:
            break
        corrected = correct_family_point(
            (m1, nxt_m2, m3),
            (current.point.x1, current.point.v1, current.point.v2, current.point.period),
            max_nfev=30,
        )
        if not corrected.success or corrected.residual_norm > max_closure:
            break
        trial = ev(corrected)
        trial_v = event_value(trial.floquet, mode)
        if abs(trial_v) >= abs(value):
            break
        current, value, m2 = trial, trial_v, nxt_m2
        if abs(value) <= event_tolerance:
            break
    return LocalizedCriticalPoint(current, mode, float(value), 0.0)


def _flow_for_vector(
    y: Array,
    *,
    m3: float,
    rtol: float,
    atol: float,
) -> tuple[Array, ReducedFloquetResult]:
    x1, v1, v2, period, m1, m2 = [float(x) for x in y]
    if period <= 0.0:
        raise RuntimeError("non-positive period in critical corrector")
    masses = np.asarray([m1, m2, m3], dtype=float)
    state0 = state_from_chart((m1, m2, m3), x1, v1, v2)
    z0 = full_to_reduced(state0)
    u0 = np.concatenate((z0, np.eye(8).ravel()))

    def augmented(t: float, u: Array) -> Array:
        z = u[:8]
        phi = u[8:].reshape(8, 8)
        dz = reduced_rhs(t, z, masses)
        dphi = reduced_jacobian(z, masses) @ phi
        return np.concatenate((dz, dphi.ravel()))

    sol = solve_ivp(
        augmented,
        (0.0, period),
        u0,
        method="DOP853",
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    zf = sol.y[:8, -1]
    monodromy = sol.y[8:, -1].reshape(8, 8)
    return zf - z0, stability_invariants(monodromy)


def _point_from_vector(y: Array, closure_norm: float) -> FamilyPoint:
    x1, v1, v2, period, m1, m2 = [float(x) for x in y]
    return FamilyPoint(
        masses=(m1, m2, 1.0),
        x1=x1,
        v1=v1,
        v2=v2,
        period=period,
        residual_norm=float(closure_norm),
        nfev=0,
        success=True,
    )


def _default_scales(y: Array) -> Array:
    floors = np.asarray([0.05, 0.5, 0.1, 1.0, 0.1, 0.1], dtype=float)
    return np.maximum(np.abs(y), floors)


def advance_augmented_critical(
    previous: LocalizedCriticalPoint | AugmentedCriticalPoint,
    current: LocalizedCriticalPoint | AugmentedCriticalPoint,
    *,
    normalized_step: float = 4e-3,
    max_closure: float = 2e-7,
    max_event: float = 2e-6,
    max_arc: float = 2e-4,
    max_nfev: int = 70,
    screening_rtol: float = 3e-10,
    screening_atol: float = 3e-12,
    use_jax_jacobian: bool = False,
) -> AugmentedCriticalPoint:
    """Take one full-state pseudo-arclength step on a critical component."""
    if previous.event_mode != current.event_mode:
        raise ValueError("critical seed event modes differ")
    mode = current.event_mode
    yp = previous.vector
    yc = current.vector
    scales = _default_scales(yc)
    secant = (yc - yp) / scales
    snorm = float(np.linalg.norm(secant))
    if snorm == 0.0:
        raise ValueError("critical seeds must be distinct in continuation space")
    tangent = secant / snorm
    predictor = yc + scales * normalized_step * tangent

    # Residual scales target the relative importance of the three constraints.
    closure_scale = 1e-6
    event_scale = 2e-4
    # Signed steps are useful when launching the two local germs from a known
    # organizer.  The sign belongs in the predictor; the residual scale must
    # remain positive.
    arc_scale = max(abs(normalized_step), 1e-4)
    last: dict[str, object] = {}

    def residual(y: Array) -> Array:
        closure, floquet = _flow_for_vector(
            y,
            m3=current.sample.point.masses[2],
            rtol=screening_rtol,
            atol=screening_atol,
        )
        critical = event_value(floquet, mode)
        arc = float(np.dot((y - predictor) / scales, tangent))
        last["closure"] = closure
        last["floquet"] = floquet
        last["event"] = critical
        last["arc"] = arc
        return np.concatenate((closure / closure_scale, [critical / event_scale, arc / arc_scale]))

    def jacobian(y: Array) -> Array:
        # Optional discovery accelerator.  Acceptance always uses the SciPy
        # DOP853 values above/below; JAX supplies derivatives only.
        from .jax_diffrax import (  # noqa: PLC0415
            adaptive_closure_and_jacobian,
            adaptive_event_and_gradient,
        )

        _closure, closure_jac = adaptive_closure_and_jacobian(
            y,
            m3=current.sample.point.masses[2],
            rtol=1e-10,
            atol=1e-12,
            max_steps=1 << 18,
        )
        _event, event_grad = adaptive_event_and_gradient(
            y,
            mode,
            m3=current.sample.point.masses[2],
            rtol=5e-10,
            atol=5e-12,
            max_steps=1 << 18,
        )
        arc_grad = tangent / scales
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
        jac=jacobian if use_jax_jacobian else "2-point",
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
        m3=current.sample.point.masses[2],
        rtol=screening_rtol,
        atol=screening_atol,
    )
    closure_norm = float(np.linalg.norm(closure))
    critical = event_value(floquet, mode)
    arc = float(np.dot((fit.x - predictor) / scales, tangent))
    if not fit.success:
        raise RuntimeError(f"augmented least-squares failed: {fit.message}")
    if closure_norm > max_closure or abs(critical) > max_event or abs(arc) > max_arc:
        raise RuntimeError(
            "augmented critical correction missed acceptance gates: "
            f"closure={closure_norm:.3e}, event={critical:.3e}, arc={arc:.3e}"
        )

    point = FamilyPoint(
        masses=(float(fit.x[4]), float(fit.x[5]), current.sample.point.masses[2]),
        x1=float(fit.x[0]),
        v1=float(fit.x[1]),
        v2=float(fit.x[2]),
        period=float(fit.x[3]),
        residual_norm=closure_norm,
        nfev=int(fit.nfev),
        success=True,
    )
    sample = BoundarySample(point, floquet, float(min(
        floquet.discriminant,
        2.0 - abs(floquet.trace_roots[0]),
        2.0 - abs(floquet.trace_roots[1]),
    )))
    return AugmentedCriticalPoint(
        sample=sample,
        event_mode=mode,
        event_value=float(critical),
        tangent_scaled=tuple(float(x) for x in tangent),
        arclength_residual=float(arc),
        normalized_step=float(normalized_step),
        nfev=int(fit.nfev),
    )


def trace_augmented_critical(
    first: LocalizedCriticalPoint,
    second: LocalizedCriticalPoint,
    *,
    steps: int,
    normalized_step: float = 4e-3,
    min_step: float = 2.5e-4,
    max_retries: int = 4,
) -> AugmentedCriticalTrace:
    """Trace a critical component with adaptive pseudo-arclength step reduction."""
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if first.event_mode != second.event_mode:
        raise ValueError("localized seeds identify different critical events")
    previous: LocalizedCriticalPoint | AugmentedCriticalPoint = first
    current: LocalizedCriticalPoint | AugmentedCriticalPoint = second
    points: list[AugmentedCriticalPoint] = []
    step = normalized_step
    reason = "requested_steps_completed"
    for _ in range(steps):
        last_error: Exception | None = None
        accepted: AugmentedCriticalPoint | None = None
        trial_step = step
        for _retry in range(max_retries + 1):
            try:
                accepted = advance_augmented_critical(
                    previous,
                    current,
                    normalized_step=trial_step,
                )
                break
            except (RuntimeError, ValueError) as exc:
                last_error = exc
                trial_step *= 0.5
                if trial_step < min_step:
                    break
        if accepted is None:
            reason = f"pseudo-arclength correction failed: {last_error}"
            break
        points.append(accepted)
        previous, current = current, accepted
        # Recover cautiously after a reduced step; never grow above the requested step.
        step = min(normalized_step, trial_step * 1.25)
    return AugmentedCriticalTrace(tuple(points), reason)
