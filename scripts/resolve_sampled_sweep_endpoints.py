#!/usr/bin/env python3
"""Replace sampled 'interior termini' with continuous scientific classifications.

Two supplemental minus-one components currently end strictly inside the scanned
mass support (component 0 at its low-m1 end; component 1 at its high-m1 end).
The assembler labels those finite-lattice ends as passed nodes.  This script
continues *away* from the sampled segment using the variational predictor in
``trace_label_invisible_continuous.py`` and asks what the zero set actually does.

Accepted scientific stops:
  * tangent-matched overlap with a committed catalog critical root/arc;
  * declared-domain boundary;
  * closed loop (return to an earlier continuation point with tangent match);
  * existence boundary of the periodic family -- the walk ran out of family,
    not out of corrector.  See ``threebody_atlas.existence_boundary`` for the
    fail-closed conditions; this script's job is to MEASURE them.
A projection turn is recorded but never treated as a stop.  Corrector failure is
reported as unresolved, never promoted to an endpoint.

THE FOURTH STOP, AND WHY IT IS NOT A LOOPHOLE.  "The corrector failed" and "the
family ceased to exist" produce the same symptom, so the fourth class is awarded
only against measurements taken on purpose at the failure, every one of which can
only ever REFUSE it:

  * the periodic orbit must fail to CLOSE at the nearest predictor the ladder
    ever tried, with a residual orders outside the frozen 1e-7 gate.  A point
    where the orbit closes is a point where the family exists, whatever the
    event corrector did there;
  * halving the mass step, twice more than the retry ladder already does, must
    not rescue the step;
  * a tightened float64 solve and the independent accelerated Diffrax
    integrator must both agree that nothing closes there;
  * probes further out must also fail to close, so the absence is a region;
  * the committed sign-topology audits, read per scan line, must independently
    show every probe they fired outward of that point failing to close;
  * the walk must have been leaving the support monotonically, by at least one
    mass grid step, from a certified point.

An unmeasurable condition refuses exactly like a failed one, so a run in which
the probes cannot be taken reports the endpoint unresolved, as today.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

import trace_label_invisible_continuous as cont
from threebody_atlas import existence_boundary as frontier
from threebody_atlas.critical_geometry import continuation_scales


CATALOG_MATCH_TOL = 7e-3
CATALOG_TANGENT_COS = 0.94
LOOP_MATCH_TOL = 5e-3
LOOP_MIN_SEPARATION = 12
ORGANIZER_MATCH_TOL = 8e-3
REPO_ROOT = Path(__file__).resolve().parents[1]
GRAPH_PATH = REPO_ROOT / "research/evidence/V1_CRITICAL_GRAPH.json"

# MINIMUM_RETRY_STEP -- the floor of the step-retry ladder, named rather than
# inlined because the existence probe deliberately reaches BELOW it.  The ladder
# stops here because a walk that needs a step this short is not making progress;
# that is a statement about progress, not about whether an orbit exists.
MINIMUM_RETRY_STEP = 5e-5

# EXISTENCE_EXTRA_REFINEMENTS -- two more halvings, past the ladder's floor.
# "Halve the step, and halve it again" has to be tried below the resolver's own
# progress floor, or the step-refinement condition would only be re-testing a
# constant of this script.  From the 7.8e-5 the ladder reaches, these two land at
# 3.9e-5 and 2.0e-5.  A step that succeeds here refuses the class: it means the
# walk was stopped by step control.
EXISTENCE_EXTRA_REFINEMENTS = 2

# EXISTENCE_PERSISTENCE_MULTIPLES -- of the failing step, outward along the
# tangent.  1x is where the walk was trying to go; 2x and 4x ask whether the
# absence persists or whether the family resumes just beyond.  A closing orbit at
# any of them refuses the class.
EXISTENCE_PERSISTENCE_MULTIPLES = (1.0, 2.0, 4.0)

# Tightened float64 tolerances for the precision cross-check.  These are the
# tolerances scripts/jax_diffrax_audit.py uses for its REFERENCE SciPy closure
# (rtol 2e-12, atol 2e-14), two orders inside the 2e-10/2e-12 screening
# tolerances liao_family.correct_family_point runs by default, so a residual that
# survives them is not a quadrature artifact of the default path.
TIGHTENED_RTOL = 2e-12
TIGHTENED_ATOL = 2e-14
TIGHTENED_MAX_NFEV = 200

# The independent accelerated path: Diffrax adaptive integration plus its own
# autodiff Jacobian, at the tolerances scripts/jax_diffrax_audit.py accepted it
# against finite differences.  Used here as a SOLVER -- damped Gauss-Newton on
# the four chart parameters at fixed masses -- because the question is whether
# another arithmetic can close what SciPy could not.
ACCELERATED_RTOL = 1e-10
ACCELERATED_ATOL = 1e-12
ACCELERATED_NEWTON_ITERATIONS = 12
ACCELERATED_BACKTRACKS = 6

#: Committed sign-topology audits whose per-probe records corroborate (or refute)
#: a claimed existence frontier.  Defaults, overridable on the command line; the
#: full-domain audit is the only one dense enough to corroborate, and either may
#: refute.  Both are read from the repository, never from a temp path.
DEFAULT_SIGN_TOPOLOGY_AUDITS = (
    "research/evidence/V1_SIGN_TOPOLOGY_AUDIT_INDEPENDENT_35LINE_2026-08-17.json",
    "research/evidence/V1_SIGN_TOPOLOGY_AUDIT_FULLDOMAIN_2026-08-18.json",
)


def load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path}: JSON root must be an object")
    return payload


def vector(row: dict[str, Any]) -> np.ndarray:
    masses = row["masses"]
    return np.asarray(
        [float(row["x1"]), float(row["v1"]), float(row["v2"]), float(row["period"]), float(masses[0]), float(masses[1])],
        dtype=float,
    )


def segment_distance(a: np.ndarray, b: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    scale = continuation_scales(target)
    aa = (a - target) / scale
    bb = (b - target) / scale
    d = bb - aa
    denom = float(np.dot(d, d))
    if denom == 0.0:
        return float(np.linalg.norm(aa)), 0.0
    t = float(np.clip(-np.dot(aa, d) / denom, 0.0, 1.0))
    return float(np.linalg.norm(aa + t * d)), t


def tangent_mass(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = b[4:6] - a[4:6]
    n = float(np.linalg.norm(d))
    return d / n if n else np.zeros(2)


def strict_supplemental(row: dict[str, Any]) -> cont.StrictPoint:
    return cont._strict_localize(
        row,
        str(row["event_mode"]),
        source="research/evidence/V1_SUPPLEMENTAL_EVENT_SIGN_ROOTS_2026-08-16.json",
        source_id=f"supplemental-cell-{int(row['cell_id'])}",
    )


def _direction_only_seed(certified: cont.StrictPoint, neighbor: dict[str, Any]) -> cont.StrictPoint:
    """Copy a re-certified chart onto a neighbor's masses for the first secant.

    The neighbor is not a scientific seed.  Only its mass-plane location orients
    the variational predictor.  The first accepted step still has to pass the
    frozen gates on its own.
    """
    masses = neighbor.get("masses") or []
    if len(masses) < 2:
        raise RuntimeError("direction-only neighbor is missing masses")
    point = certified.localized.sample.point
    dummy_point = replace(
        point,
        masses=(float(masses[0]), float(masses[1]), float(point.masses[2])),
    )
    dummy_sample = replace(certified.localized.sample, point=dummy_point)
    dummy_localized = replace(certified.localized, sample=dummy_sample)
    return replace(certified, localized=dummy_localized)


def mixed_organizers(mode: str) -> list[dict[str, Any]]:
    """Passed mixed organizers that a matching-mechanism branch may terminate on."""
    if not GRAPH_PATH.is_file():
        return []
    graph = load(GRAPH_PATH)
    rows = []
    for node in graph.get("nodes", []):
        kind = str(node.get("kind") or "")
        mechanism = str(node.get("mechanism") or "")
        if not node.get("passed"):
            continue
        if kind != "mixed_organizer" and mechanism != "mixed_organizer":
            continue
        masses = node.get("masses") or []
        if len(masses) < 2:
            continue
        rows.append(
            {
                "id": str(node["id"]),
                "masses": [float(masses[0]), float(masses[1])],
            }
        )
    return rows


def nearest_organizer(
    point: np.ndarray,
    organizers: list[dict[str, Any]],
) -> dict[str, Any] | None:
    best = None
    for item in organizers:
        target = np.asarray(item["masses"], dtype=float)
        miss = float(np.linalg.norm(point[4:6] - target))
        if best is None or miss < best["miss_mass"]:
            best = {"id": item["id"], "miss_mass": miss, "masses": item["masses"]}
    return best


def catalog_candidates(payload: dict[str, Any], mode: str) -> list[dict[str, Any]]:
    rows = []
    for row in payload.get("roots", []):
        if row.get("status") != "ok" or not row.get("passed") or row.get("event_mode") != mode:
            continue
        rows.append(
            {
                "cell_id": int(row["cell_id"]),
                "orientation": row.get("orientation"),
                "target": vector(row),
            }
        )
    return rows


def nearest_catalog_segment(
    a: np.ndarray,
    b: np.ndarray,
    candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    seg_tangent = tangent_mass(a, b)
    best = None
    for item in candidates:
        miss, fraction = segment_distance(a, b, item["target"])
        # A catalog root has no stored branch tangent.  Estimate it from the
        # local mass secant only after a close 6-D chart match is found; the
        # final branch tangent check is done against neighboring catalog roots
        # below by the caller when possible.
        if best is None or miss < best["miss_scaled"]:
            best = {
                "cell_id": item["cell_id"],
                "orientation": item["orientation"],
                "miss_scaled": miss,
                "fraction": fraction,
                "segment_mass_tangent": seg_tangent,
                "target": item["target"],
            }
    return best


def catalog_neighbor_tangent(
    hit: dict[str, Any], candidates: list[dict[str, Any]]
) -> tuple[float, list[int]]:
    target = hit["target"]
    same = sorted(
        candidates,
        key=lambda item: abs(float(item["target"][4] - target[4])) + abs(float(item["target"][5] - target[5])),
    )
    neighbors = [item for item in same if item["cell_id"] != hit["cell_id"]][:4]
    best_cos = 0.0
    ids: list[int] = []
    for item in neighbors:
        d = item["target"][4:6] - target[4:6]
        n = float(np.linalg.norm(d))
        if n == 0.0:
            continue
        ids.append(item["cell_id"])
        cos = abs(float(np.dot(hit["segment_mass_tangent"], d / n)))
        best_cos = max(best_cos, cos)
    return best_cos, ids


def loop_hit(history: list[np.ndarray], a: np.ndarray, b: np.ndarray) -> dict[str, Any] | None:
    if len(history) <= LOOP_MIN_SEPARATION:
        return None
    seg_tangent = tangent_mass(a, b)
    for index, target in enumerate(history[:-LOOP_MIN_SEPARATION]):
        miss, fraction = segment_distance(a, b, target)
        if miss > LOOP_MATCH_TOL:
            continue
        before = history[max(0, index - 1)]
        after = history[min(len(history) - 1, index + 1)]
        old_tangent = tangent_mass(before, after)
        cosine = abs(float(np.dot(seg_tangent, old_tangent)))
        if cosine >= 0.94:
            return {
                "kind": "closed_loop",
                "history_index": index,
                "miss_scaled": miss,
                "fraction": fraction,
                "tangent_abs_cosine": cosine,
            }
    return None


def _recorded_closure(value: float) -> tuple[float | None, str]:
    """A closure norm as it will survive this script's JSON sanitizer.

    ``sanitize`` below maps every non-finite float to null, so an in-memory inf
    would let the producer award a class that the artifact could no longer
    justify to a consumer re-deriving it.  A non-finite residual is therefore
    recorded as a missing measurement plus a note, which REFUSES -- the safe
    direction -- and keeps the in-memory verdict and the artifact identical.
    """
    number = float(value)
    if not math.isfinite(number):
        return None, f"non-finite closure {number!r}; recorded as unmeasured"
    return number, ""


def _closure_probe(
    masses: tuple[float, float, float],
    guess: tuple[float, float, float, float],
    mode: str,
    *,
    rtol: float | None = None,
    atol: float | None = None,
    max_nfev: int = 70,
) -> dict[str, Any]:
    """Does a periodic orbit close at this mass point?  The audits' own test.

    scripts/audit_sign_topology.evaluate_probes decides existence exactly this
    way -- ``correct_family_point`` from a neighbouring chart, then the closure
    residual against the frozen gate -- and its non-closing probes are what the
    committed audits report as order-one closure norms.  Using the same test here
    is what makes the audits admissible as corroboration at all.

    Every failure mode is recorded rather than raised: a probe that cannot be
    taken must refuse the terminus, not abort the run.
    """
    record: dict[str, Any] = {
        "masses": [float(masses[0]), float(masses[1])],
        "available": True,
        "closed": False,
        "closure": None,
        "event": None,
        "solver_success": None,
        "note": "",
        "tolerances": {"rtol": rtol, "atol": atol, "max_nfev": max_nfev},
    }
    kwargs: dict[str, Any] = {"max_nfev": max_nfev}
    if rtol is not None and atol is not None:
        kwargs["screening_rtol"] = float(rtol)
        kwargs["screening_atol"] = float(atol)
    try:
        corrected = cont.correct_family_point(masses, guess, **kwargs)
    except (RuntimeError, ValueError, FloatingPointError, ArithmeticError) as exc:
        record["note"] = f"{type(exc).__name__}: {exc}"
        return record
    record["solver_success"] = bool(corrected.success)
    closure, note = _recorded_closure(corrected.residual_norm)
    record["closure"] = closure
    record["note"] = note
    record["closed"] = frontier.closure_closes(closure)
    record["chart"] = [
        float(corrected.x1),
        float(corrected.v1),
        float(corrected.v2),
        float(corrected.period),
    ]
    if record["closed"]:
        # Only meaningful on a closed orbit; on a non-closing point the Floquet
        # invariants are read off a trajectory that is not periodic at all.
        try:
            sample = cont._precise_evaluate(corrected)
            record["event"] = float(cont.event_value(sample.floquet, mode))
        except (RuntimeError, ValueError, FloatingPointError, ArithmeticError) as exc:
            record["note"] = f"event unread: {type(exc).__name__}: {exc}"
    return record


def _accelerated_closure_probe(
    chart: tuple[float, float, float, float],
    masses: tuple[float, float, float],
) -> dict[str, Any]:
    """Try to close the orbit in the independent accelerated arithmetic.

    Diffrax adaptive integration with its own autodiff Jacobian, driven as a
    damped Gauss-Newton on the four chart parameters at FIXED masses.  This is
    the escape hatch for a float64 artifact: if another integrator and another
    derivative path find the periodic orbit that SciPy could not, then the
    continuation failure was arithmetic and the terminus is refused.

    Unavailability is not a pass.  If JAX/Diffrax cannot run, the record says so
    and the precision condition refuses on that basis.
    """
    record: dict[str, Any] = {
        "path": "threebody_atlas.jax_diffrax.adaptive_closure_and_jacobian",
        "available": False,
        "closed": False,
        "closure": None,
        "iterations": 0,
        "note": "",
        "tolerances": {"rtol": ACCELERATED_RTOL, "atol": ACCELERATED_ATOL},
    }
    try:
        from threebody_atlas.jax_diffrax import (
            adaptive_closure_and_jacobian,
            require_accelerated_x64,
        )

        require_accelerated_x64()
    except (ImportError, RuntimeError) as exc:
        record["note"] = f"{type(exc).__name__}: {exc}"
        return record
    record["available"] = True

    def residual(vector: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        closure, jacobian = adaptive_closure_and_jacobian(
            vector, m3=float(masses[2]), rtol=ACCELERATED_RTOL, atol=ACCELERATED_ATOL
        )
        closure = np.asarray(closure, dtype=float)
        return float(np.linalg.norm(closure)), closure, np.asarray(jacobian, dtype=float)

    vector = np.asarray(
        [chart[0], chart[1], chart[2], chart[3], masses[0], masses[1]], dtype=float
    )
    try:
        norm, closure, jacobian = residual(vector)
    except (RuntimeError, ValueError, FloatingPointError, ArithmeticError) as exc:
        record["note"] = f"initial evaluation failed: {type(exc).__name__}: {exc}"
        return record
    record["closure_at_float64_chart"] = norm
    for iteration in range(1, ACCELERATED_NEWTON_ITERATIONS + 1):
        record["iterations"] = iteration
        record["closure"], closure_note = _recorded_closure(norm)
        if closure_note:
            record["note"] = closure_note
        if norm <= frontier.CLOSURE_GATE:
            record["closed"] = True
            return record
        try:
            step, *_ = np.linalg.lstsq(jacobian[:, :4], -closure, rcond=None)
        except np.linalg.LinAlgError as exc:
            record["note"] = f"least squares failed: {exc}"
            return record
        damping = 1.0
        improved = False
        for _backtrack in range(ACCELERATED_BACKTRACKS):
            trial = vector.copy()
            trial[:4] += damping * step
            if trial[3] <= 0.0:
                damping *= 0.5
                continue
            try:
                trial_norm, trial_closure, trial_jacobian = residual(trial)
            except (RuntimeError, ValueError, FloatingPointError, ArithmeticError) as exc:
                record["note"] = f"{type(exc).__name__}: {exc}"
                damping *= 0.5
                continue
            if trial_norm < norm:
                vector, norm, closure, jacobian = trial, trial_norm, trial_closure, trial_jacobian
                improved = True
                break
            damping *= 0.5
        if not improved:
            record["note"] = (
                f"no descent after {ACCELERATED_BACKTRACKS} backtracks at closure "
                f"{norm:.3e}"
            )
            break
    record["closure"], closure_note = _recorded_closure(norm)
    if closure_note:
        record["note"] = closure_note
    record["closed"] = frontier.closure_closes(record["closure"])
    return record


def _existence_conditions(
    prev: Any,
    cur: Any,
    *,
    failing_step: float,
    ladder: list[dict[str, Any]],
    accepted: list[dict[str, Any]],
    seed_masses: list[float],
    audits: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Measure every existence-boundary condition at a continuation failure.

    Returns ``(conditions, measurements)``: the first is the per-condition
    payload ``threebody_atlas.existence_boundary.evaluate`` re-derives its verdict
    from, the second is the raw probe record kept for a reviewer.  Nothing here
    decides anything; a measurement that cannot be taken is written down as
    missing, which refuses.
    """
    point = cur.sample.point
    m3 = float(point.masses[2])
    cur_mass = np.asarray(point.masses[:2], dtype=float)
    guess = (float(point.x1), float(point.v1), float(point.v2), float(point.period))
    measurements: dict[str, Any] = {
        "failing_step": float(failing_step),
        "step_ladder": ladder,
        "seed_masses": [float(v) for v in seed_masses[:2]],
        "last_accepted_masses": [float(v) for v in cur_mass],
    }
    conditions: dict[str, Any] = {}

    # The predictor direction, recomputed exactly as _advance_variational does.
    try:
        reference = cur_mass - np.asarray(prev.sample.point.masses[:2], dtype=float)
        _gradient, tangent, _sens = cont._variational_geometry(cur, reference)
    except (RuntimeError, ValueError, FloatingPointError, ArithmeticError) as exc:
        measurements["tangent_error"] = f"{type(exc).__name__}: {exc}"
        # No direction means no frontier point, so no condition below can be
        # measured.  Every one of them is left unrecorded, and every one refuses.
        return conditions, measurements
    tangent = np.asarray(tangent, dtype=float)
    measurements["mass_tangent"] = [float(v) for v in tangent]

    # Offsets outward along the tangent.  The NEAREST one is the smallest step the
    # ladder ever requested, including the deliberate refinements below the retry
    # floor: claiming the family ends there is the tightest claim available, and
    # the easiest to refute, because an orbit that close to a certified one should
    # be the easiest to find.
    ladder_steps = [
        float(entry["requested_step"])
        for entry in ladder
        if isinstance(entry, dict) and entry.get("requested_step") is not None
    ]
    nearest_offset = min([*ladder_steps, float(failing_step)])
    persistence_offsets = [float(failing_step) * m for m in EXISTENCE_PERSISTENCE_MULTIPLES]
    probes: dict[str, dict[str, Any]] = {}
    # MARCHED, not fanned out from one chart.  A probe 1e-2 away that misses
    # because its SEED was 1e-2 stale is a statement about seeding, and it would
    # make the persistence condition EASIER to satisfy -- the one thing no
    # condition here is allowed to be.  So each probe is seeded from the last one
    # that closed, which is exactly how audit_sign_topology.evaluate_probes walks
    # a scan line, and the last good chart is retained across a failure.
    marching_guess = guess
    for label, offset in [("nearest", nearest_offset)] + [
        (f"persistence_{m:g}x", value)
        for m, value in zip(
            EXISTENCE_PERSISTENCE_MULTIPLES, persistence_offsets, strict=True
        )
    ]:
        target = cur_mass + offset * tangent
        probe = _closure_probe(
            (float(target[0]), float(target[1]), m3), marching_guess, str(cur.event_mode)
        )
        probe["distance_outward"] = float(offset)
        probe["seeded_from"] = list(marching_guess)
        probes[label] = probe
        if probe["closed"] and probe.get("chart"):
            marching_guess = tuple(float(value) for value in probe["chart"])
    measurements["closure_probes"] = probes

    frontier_masses = probes["nearest"]["masses"]
    measurements["frontier_masses"] = frontier_masses

    conditions["divergence"] = {
        "closure": probes["nearest"]["closure"],
        "event": probes["nearest"]["event"],
        "solver_success": probes["nearest"]["solver_success"],
        "distance_outward": nearest_offset,
        "note": probes["nearest"]["note"],
    }
    conditions["outward_persistence"] = {
        "probes": [
            {
                "distance_outward": probes[label]["distance_outward"],
                "closed": probes[label]["closed"],
                "closure": probes[label]["closure"],
                "note": probes[label]["note"],
            }
            for label in probes
            if label != "nearest"
        ]
    }
    conditions["step_refinement_invariance"] = {"ladder": ladder}
    conditions["inside_declared_domain"] = {"frontier_masses": frontier_masses}

    last = accepted[-1] if accepted else None
    conditions["certified_departure"] = {
        "accepted_points": len(accepted),
        "last_accepted_closure": (last or {}).get("closure"),
        "last_accepted_event": (last or {}).get("event"),
        "last_accepted_masses": (last or {}).get("masses"),
    }

    # Direction.  The outward coordinate is m2 because that is the coordinate the
    # committed audits sweep, so it is the only one along which their frontier is
    # measured.  The sign is the walk's own heading; the audits are asked
    # separately which side leaves the support, and the two must agree.
    walk_m2 = [float(seed_masses[1])] + [
        float((step.get("masses") or [0.0, 0.0])[1]) for step in accepted
    ]
    sign = 0
    if tangent[1] < 0.0:
        sign = -1
    elif tangent[1] > 0.0:
        sign = 1
    steps_outward = [
        sign * (walk_m2[index + 1] - walk_m2[index]) for index in range(len(walk_m2) - 1)
    ]
    conditions["outward_direction"] = {
        "axis": "m2",
        "outward_sign": sign if sign in (-1, 1) else None,
        "outward_travel": sign * (walk_m2[-1] - walk_m2[0]) if sign else None,
        "monotone_outward": bool(steps_outward) and all(value > 0.0 for value in steps_outward),
        "walk_m2": walk_m2,
        "per_step_outward": steps_outward,
        "tangent_m2": float(tangent[1]),
    }

    # Precision.  Two independent escapes from a float64 artifact: the same SciPy
    # path at tolerances two orders tighter, and a different integrator with a
    # different derivative path driven as a solver.
    tightened = _closure_probe(
        (float(frontier_masses[0]), float(frontier_masses[1]), m3),
        guess,
        str(cur.event_mode),
        rtol=TIGHTENED_RTOL,
        atol=TIGHTENED_ATOL,
        max_nfev=TIGHTENED_MAX_NFEV,
    )
    accelerated_chart = probes["nearest"].get("chart") or guess
    accelerated = _accelerated_closure_probe(
        (
            float(accelerated_chart[0]),
            float(accelerated_chart[1]),
            float(accelerated_chart[2]),
            float(accelerated_chart[3]),
        ),
        (float(frontier_masses[0]), float(frontier_masses[1]), m3),
    )
    conditions["precision_invariance"] = {
        "tightened_float64": tightened,
        "accelerated": accelerated,
    }

    corroboration = frontier.corroboration_measurement(
        audits, float(frontier_masses[0]), float(frontier_masses[1])
    )
    corroboration["outward_sign"] = sign if sign in (-1, 1) else None
    conditions["audit_corroboration"] = corroboration
    return conditions, measurements


def trace_endpoint(
    rows: list[dict[str, Any]],
    catalog: list[dict[str, Any]],
    *,
    outward_side: str,
    mass_step: float,
    max_steps: int,
    organizers: list[dict[str, Any]] | None = None,
    audits: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    rows = sorted(rows, key=lambda row: (float(row["masses"][0]), float(row["masses"][1])))
    if outward_side == "low":
        ordered = list(rows)
    elif outward_side == "high":
        ordered = list(reversed(rows))
    else:
        raise ValueError(outward_side)
    # A stored lattice sample can fail the frozen event gate on re-correction
    # (float64 evaluation floor).  Walk inward until a point re-certifies, then
    # continue outward.  A second re-certified neighbor is preferred for the
    # predictor direction; otherwise the unused lattice neighbor supplies only
    # a mass-plane secant (its chart is not treated as a seed).
    current_row = None
    current = None
    previous_row = None
    previous = None
    seed_errors: list[str] = []
    for row in ordered:
        try:
            point = strict_supplemental(row)
        except (RuntimeError, ValueError, FloatingPointError) as exc:
            seed_errors.append(f"cell {row.get('cell_id')}: {exc}")
            continue
        if current is None:
            current_row, current = row, point
            continue
        previous_row, previous = row, point
        break
    if current is None or current_row is None:
        raise RuntimeError(
            f"component {rows[0].get('sweep_component')} {outward_side}: "
            f"no lattice seed re-certified; {seed_errors}"
        )
    direction_only = False
    if previous is None or previous_row is None:
        neighbor = next((row for row in ordered if row is not current_row), rows[0])
        previous_row = neighbor
        previous = _direction_only_seed(current, neighbor)
        direction_only = True
    prev, cur = previous.localized, current.localized
    history = [previous.vector, current.vector]
    accepted = []
    folds = []
    step = float(mass_step)
    stop = "max_steps_exhausted"
    terminal = None
    existence: dict[str, Any] | None = None
    previous_dm1 = float(current.masses2[0] - previous.masses2[0])

    for index in range(max_steps):
        point = None
        last_error = None
        trial = step
        ladder: list[dict[str, Any]] = []
        for _retry in range(8):
            try:
                point = cont._advance_variational(prev, cur, requested_step=trial)
                break
            except (RuntimeError, ValueError, FloatingPointError) as exc:
                last_error = exc
                ladder.append(
                    {"requested_step": float(trial), "closed": False, "error": str(exc)}
                )
                trial *= 0.5
                if trial < MINIMUM_RETRY_STEP:
                    break
        if point is None:
            # Continue the ladder BELOW the retry floor on purpose: two more
            # halvings, recorded whether they fail or succeed.  A step that
            # succeeds down here refuses the existence class -- it says the walk
            # was stopped by step control, not by the family -- and is worth
            # recording either way, because today that distinction is invisible.
            # The walk is not resumed from it; step control is not this script's
            # subject, and a rescued step would still owe the frozen gates a full
            # accepted point at every subsequent step.
            for _extra in range(EXISTENCE_EXTRA_REFINEMENTS):
                try:
                    rescued = cont._advance_variational(prev, cur, requested_step=trial)
                except (RuntimeError, ValueError, FloatingPointError) as exc:
                    ladder.append(
                        {
                            "requested_step": float(trial),
                            "closed": False,
                            "error": str(exc),
                            "below_retry_floor": True,
                        }
                    )
                    trial *= 0.5
                    continue
                ladder.append(
                    {
                        "requested_step": float(trial),
                        "closed": True,
                        "closure": float(rescued.localized.sample.point.residual_norm),
                        "event": float(rescued.localized.event_value),
                        "below_retry_floor": True,
                    }
                )
                break
            conditions, measurements = _existence_conditions(
                prev,
                cur,
                failing_step=float(step),
                ladder=ladder,
                accepted=accepted,
                seed_masses=[float(v) for v in list(current.masses2)[:2]],
                audits=audits or [],
            )
            existence = frontier.evaluate(conditions)
            existence["measurements"] = measurements
            if existence["awarded"]:
                frontier_masses = measurements.get("frontier_masses") or []
                final_masses = [
                    float(v) for v in ((accepted[-1].get("masses") if accepted else []) or [])[:2]
                ]
                terminal = {
                    "kind": frontier.TERMINUS_KIND,
                    "frontier_masses": [float(v) for v in frontier_masses[:2]],
                    "final_masses": final_masses,
                    "miss_mass": math.dist(final_masses, [float(v) for v in frontier_masses[:2]]),
                    "outward_axis": "m2",
                    "outward_sign": conditions["outward_direction"]["outward_sign"],
                    "frontier_closure": conditions["divergence"]["closure"],
                    "conditions_passed": [
                        name for name in frontier.CONDITION_ORDER
                        if existence["conditions"][name]["passed"]
                    ],
                }
                stop = "existence_boundary_reached"
            else:
                # The refusal is part of the endpoint's record, not a footnote:
                # today's stopped_reason says only that the corrector failed, and
                # says nothing about whether anyone checked why.
                detail = measurements.get("tangent_error")
                stop = (
                    "continuation_failure_not_a_scientific_terminus: "
                    f"{last_error}; existence_boundary refused -- "
                    f"{existence['refusal_reason']}"
                    + (f" (predictor geometry unavailable: {detail})" if detail else "")
                )
            break

        a = np.asarray(cur.vector, dtype=float)
        b = point.vector
        dm1 = float(b[4] - a[4])
        if previous_dm1 != 0.0 and dm1 != 0.0 and previous_dm1 * dm1 < 0.0:
            folds.append(
                {
                    "kind": "m1_projection_turn_crossed",
                    "step_index": index,
                    "before_masses": [float(a[4]), float(a[5]), 1.0],
                    "after_masses": [float(b[4]), float(b[5]), 1.0],
                }
            )
        if dm1 != 0.0:
            previous_dm1 = dm1

        crossing = cont._domain_crossing(a, b)
        if crossing is not None:
            terminal = {"kind": "declared_domain_boundary", **crossing}
            stop = "declared_domain_boundary_reached"

        if terminal is None:
            hit = nearest_catalog_segment(a, b, catalog)
            if hit is not None and hit["miss_scaled"] <= CATALOG_MATCH_TOL:
                cosine, neighbor_ids = catalog_neighbor_tangent(hit, catalog)
                if cosine >= CATALOG_TANGENT_COS:
                    terminal = {
                        "kind": "existing_catalog_critical_curve",
                        "cell_id": hit["cell_id"],
                        "orientation": hit["orientation"],
                        "miss_scaled": float(hit["miss_scaled"]),
                        "segment_fraction": float(hit["fraction"]),
                        "tangent_abs_cosine": float(cosine),
                        "neighbor_cells_used_for_tangent": neighbor_ids,
                    }
                    stop = "existing_catalog_curve_reached"

        if terminal is None:
            hit = nearest_organizer(b, organizers or [])
            if hit is not None and hit["miss_mass"] <= ORGANIZER_MATCH_TOL:
                terminal = {
                    "kind": "mixed_organizer",
                    "node_id": hit["id"],
                    "miss_mass": float(hit["miss_mass"]),
                    "organizer_masses": hit["masses"],
                }
                stop = "mixed_organizer_reached"

        if terminal is None:
            loop = loop_hit(history, a, b)
            if loop is not None:
                terminal = loop
                stop = "closed_loop_reached"

        accepted.append(cont._serialize_variational(point))
        history.append(b)
        prev, cur = cur, point.localized
        step = min(float(mass_step), trial * 1.35)
        if terminal is not None:
            break

    passed = terminal is not None and terminal["kind"] in {
        "declared_domain_boundary",
        "existing_catalog_critical_curve",
        "mixed_organizer",
        "closed_loop",
        frontier.TERMINUS_KIND,
    }
    return {
        "source_component": int(rows[0]["sweep_component"]),
        "outward_side": outward_side,
        # The seed pair identifies the COMPUTATION.  main() refuses to report two
        # sides that share one -- see the degeneracy note there.
        "seed_cells": {
            "current": current_row.get("cell_id"),
            "previous": previous_row.get("cell_id"),
            "direction_only_neighbor": bool(direction_only),
        },
        "seed_rows": [int(previous_row["cell_id"]), int(current_row["cell_id"])],
        # A resolution is a WALK: it starts at the edge terminus being explained
        # and ends somewhere else.  Both ends must be recorded, because a
        # consumer has to check two different things -- that the walk began at
        # the endpoint it claims to resolve, and that it ended within reach of
        # the node it claims to have reached.  Reporting only the terminal miss
        # made those indistinguishable, which is how one end's distance came to
        # be attached to the other end's binding.
        "seed_masses": [float(v) for v in list(current.masses2)[:2]],
        "final_masses": (
            [float(v) for v in (accepted[-1].get("masses") or [])[:2]] if accepted else None
        ),
        "seed_previous": cont._serialize_localized(previous.localized),
        "seed_current": cont._serialize_localized(current.localized),
        "accepted_points": accepted,
        "projection_turns_crossed": folds,
        "terminal": terminal,
        # Recorded on every continuation failure, awarded or refused, so a
        # reviewer can re-derive the verdict from the numbers instead of trusting
        # it.  threebody_atlas.existence_boundary.recheck replays exactly this.
        "existence_boundary": existence,
        "stopped_reason": stop,
        "scientific_endpoint_resolved": bool(passed),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("supplemental_roots")
    parser.add_argument("catalog_roots")
    parser.add_argument("output")
    parser.add_argument("--mass-step", type=float, default=0.0025)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument(
        "--components",
        help=(
            "comma-separated sweep_component indices to resolve.  Default: every "
            "component in the artifact that carries at least two roots.  This used "
            "to be the literal tuple (0, 1, 12), which meant the script could only "
            "ever resolve the first supplemental artifact and raised "
            "'components no longer contain enough roots' on any other."
        ),
    )
    parser.add_argument(
        "--sign-topology-audit",
        action="append",
        default=None,
        help=(
            "committed sign-topology audit whose per-probe records may corroborate "
            "-- or refute -- a claimed existence frontier.  Repeatable; defaults to "
            "the two committed audits.  Only an audit dense enough in m2 can "
            "corroborate one; any audit can refute one with a single probe that "
            "closed an orbit outward of the claimed frontier."
        ),
    )
    args = parser.parse_args()

    cont.require_accelerated_x64()
    audit_paths = [
        Path(item) for item in (args.sign_topology_audit or DEFAULT_SIGN_TOPOLOGY_AUDITS)
    ]
    missing_audits = [str(path) for path in audit_paths if not path.is_file()]
    audits: list[dict[str, Any]] = []
    if not missing_audits:
        # Loaded once, up front, because the corroboration window is read per
        # endpoint and the full-domain audit is 16 MB.  A missing audit is not
        # fatal -- other endpoints may resolve on other stop classes -- but it
        # leaves the existence class with nothing to corroborate against, which
        # refuses.
        audits = frontier.load_audits(audit_paths)
    supplemental = load(Path(args.supplemental_roots))
    catalog_payload = load(Path(args.catalog_roots))
    minus_catalog = catalog_candidates(catalog_payload, "minus_one")
    plus_catalog = catalog_candidates(catalog_payload, "plus_one")
    minus_organizers = mixed_organizers("minus_one")
    plus_organizers = mixed_organizers("plus_one")

    by_component: dict[int, list[dict[str, Any]]] = {}
    for row in supplemental.get("roots", []):
        by_component.setdefault(int(row.get("sweep_component", -1)), []).append(row)
    if args.components:
        requested = [int(x) for x in args.components.split(",") if x.strip()]
        missing = [i for i in requested if len(by_component.get(i, [])) < 2]
        if missing:
            raise RuntimeError(
                f"supplemental components {missing} carry fewer than two roots"
            )
    else:
        # Every component with enough roots to have two distinguishable termini.
        requested = sorted(i for i, rows in by_component.items() if i >= 0 and len(rows) >= 2)
    if not requested:
        raise RuntimeError(
            "no sweep component in this artifact carries two or more roots, so it "
            "has no finite termini to continue"
        )

    # A component's mechanism comes from its own roots, not from a table keyed by
    # component index: indices are per-artifact and carry no cross-artifact meaning.
    def mechanism_of(rows: list[dict[str, Any]]) -> str:
        modes = {str(r.get("event_mode")) for r in rows}
        if len(modes) != 1:
            raise RuntimeError(f"component spans mixed mechanisms {sorted(modes)}")
        return modes.pop()

    jobs = []
    for component in requested:
        rows = by_component[component]
        mode = mechanism_of(rows)
        if mode == "minus_one":
            catalog, organizers = minus_catalog, minus_organizers
        elif mode == "plus_one":
            catalog, organizers = plus_catalog, plus_organizers
        else:
            raise RuntimeError(f"component {component} has unsupported mechanism {mode!r}")
        # Both finite termini of every component, low-m1 and high-m1 side.
        jobs.append((component, "low", catalog, organizers))
        jobs.append((component, "high", catalog, organizers))
    jobs = tuple(jobs)
    results = [
        trace_endpoint(
            by_component[component],
            catalog,
            outward_side=side,
            mass_step=args.mass_step,
            max_steps=args.max_steps,
            organizers=organizers,
            audits=audits,
        )
        for component, side, catalog, organizers in jobs
    ]
    # DEGENERACY REFUSAL.
    #
    # A low trace and a high trace are two different computations.  When a
    # component's requested tip fails the frozen event gate on re-correction,
    # the walk above starts from a point further in; that is legitimate only
    # while the two sides still begin from different seeds.  If both sides end
    # up on the SAME (seed, neighbour) pair they are literally one computation,
    # so at most one of them resolves its labelled terminus -- and nothing in
    # the run says which.  Both are therefore unresolved.
    #
    # plus_one component 12 is that case.  It has two roots; cell 10131
    # (m1 1.046) fails re-correction at event 2.919e-08 against the 2e-08 gate,
    # leaving cell 10132 (m1 1.048) as the only seed.  Both sides collapsed onto
    # it with the same direction-only neighbour and emitted byte-identical
    # results -- same node, same miss_mass to 18 digits, final_masses and steps
    # both null -- whose miss_mass was measured at the HIGH terminus.  Consumed
    # naively that binds a far endpoint on a near endpoint's evidence.
    by_seed: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for item in results:
        seed = item.get("seed_cells") or {}
        by_seed.setdefault(
            (item["source_component"], seed.get("current"), seed.get("previous")), []
        ).append(item)
    for (component, current_cell, previous_cell), group in by_seed.items():
        if len(group) < 2:
            continue
        sides = sorted(str(item["outward_side"]) for item in group)
        for item in group:
            item["scientific_endpoint_resolved"] = False
            item["stopped_reason"] = (
                "degenerate_sides_not_a_scientific_terminus: component "
                f"{component} sides {sides} share seed pair "
                f"(current={current_cell}, previous={previous_cell}); one "
                "computation cannot resolve two termini"
            )
            item["terminal"] = None

    passed = all(item["scientific_endpoint_resolved"] for item in results)
    payload = {
        "schema": "atlas.v1.sampled-endpoint-resolution/1",
        "claim": (
            "continuous classification of the finite-lattice endpoints of sweep "
            f"components {list(requested)} in {Path(args.supplemental_roots).name}"
        ),
        "components_resolved": list(requested),
        "frozen_gates": {
            "maximum_absolute_event": cont.EVENT_GATE,
            "maximum_periodic_closure": cont.CLOSURE_GATE,
        },
        "accepted_scientific_stops": [
            "existing_catalog_critical_curve",
            "declared_domain_boundary",
            "mixed_organizer",
            "closed_loop",
            frontier.TERMINUS_KIND,
        ],
        "existence_boundary_thresholds": frontier.evaluate({})["thresholds"],
        "existence_boundary_conditions": {
            name: frontier.CONDITION_INTENT[name] for name in frontier.CONDITION_ORDER
        },
        "sign_topology_audits_read": [
            {"path": audit["path"], "schema": audit["schema"], "probes": len(audit["probes"])}
            for audit in audits
        ],
        "sign_topology_audits_missing": missing_audits,
        "results": results,
        "all_sampled_endpoints_resolved": passed,
        "claim_status": "sampled_endpoints_replaced_by_continuous_scientific_termini" if passed else "unresolved_continuation_required",
    }

    def sanitize(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, list):
            return [sanitize(v) for v in value]
        if isinstance(value, dict):
            return {k: sanitize(v) for k, v in value.items()}
        return value

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sanitize(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"passed": passed, "results": [{"component": r["source_component"], "stop": r["stopped_reason"], "terminal": r["terminal"]} for r in results]}, indent=2))
    raise SystemExit(0 if passed else 3)


if __name__ == "__main__":
    main()
