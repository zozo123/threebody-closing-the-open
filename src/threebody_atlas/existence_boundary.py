"""Fail-closed test for one continuation stop class: ``existence_boundary_terminus``.

A continuation that stops because its corrector failed is not a scientific
terminus, and scripts/resolve_sampled_sweep_endpoints.py has always reported
that case as unresolved.  There is, however, a physically different way for an
outward walk to stop: the periodic family it is continuing along can cease to
exist.  The committed sign-topology audits measure exactly that -- probes whose
shooting solve leaves a closure norm of order 1 against a frozen 1e-7 gate, no
periodic orbit there at all -- and they show it happening on a lower frontier in
m2 on every scan line.

This module decides whether a particular continuation failure is that frontier.
It is written so that EVERY condition can only ever REJECT.  There is no
condition whose satisfaction is inferred, defaulted, or read off a producer's
own verdict: each one is a number that must be measured and must clear a stated
bar, and a missing measurement rejects exactly like a bad one.  The reason for
that asymmetry is that the two failure modes look identical from the outside --
"the corrector did not converge" -- and only one of them is a fact about the
three-body problem.  Awarding the class to a weak corrector would put a
non-existence claim into a scientific record on the strength of a numerical
shortfall, which is the one thing this repository is organised to prevent.

WHAT THIS CLASS DOES NOT COVER.  The zero set of the Floquet event can also end
or turn while the periodic family remains perfectly well defined -- a projection
fold, a codimension-2 degeneracy, a tangency.  Those are ends of the CURVE, not
of the FAMILY, and they leave a closed orbit sitting at the frontier probe.
This module refuses them (condition ``divergence``), because the evidence a
family-existence boundary requires -- no orbit closes out there -- is precisely
what such a case does not have.  They need their own class and their own
evidence.

Every threshold below is either frozen elsewhere in the repository or derived
from a committed artifact, and each carries the number that justifies it.
"""
from __future__ import annotations

import math
from pathlib import Path
from statistics import median
from typing import Any

# ---------------------------------------------------------------------------
# Frozen gates.  Mirrors of numbers that live elsewhere; never loosened here.
# ---------------------------------------------------------------------------
#: Periodic closure gate.  scripts/trace_label_invisible_continuous.CLOSURE_GATE
#: and scripts/audit_sign_topology.MAX_CLOSURE are both 1e-7; a probe at or under
#: it has a periodic orbit, full stop.
CLOSURE_GATE = 1e-7
#: Floquet event gate, scripts/trace_label_invisible_continuous.EVENT_GATE.
EVENT_GATE = 2e-8

# DIVERGENT_CLOSURE_FLOOR -- five orders above the closure gate.
#
# The bar that separates "no periodic orbit exists here" from "my corrector
# missed".  Derived from the two committed audits, not chosen:
#   V1_SIGN_TOPOLOGY_AUDIT_INDEPENDENT_35LINE_2026-08-17.json -- 100 of 465
#     probes failed; 97 carry a measured closure; ALL 97 exceed 1e-2 (min
#     3.882e-2, median 1.532e0, max 3.806e0).
#   V1_SIGN_TOPOLOGY_AUDIT_FULLDOMAIN_2026-08-18.json -- 2785 of 33701 failed;
#     2719 carry a measured closure; 99.2% exceed 1e-2 (median 1.825e0).
# So 1e-2 is the empirical separation the audits themselves exhibit between a
# non-closing probe and the gate.  A residual of 3e-7 -- above the gate, five
# orders below this floor -- is a corrector miss and is refused.  Raising this
# floor can only refuse more; lowering it would admit corrector misses, which is
# why the pin in tests/test_existence_boundary.py exists.
DIVERGENT_CLOSURE_FLOOR = 1e-2

# MIN_ACCEPTED_CONTINUATION_POINTS -- the walk must have worked here first.
#
# A corrector that fails on its first step has demonstrated nothing about the
# family; it has demonstrated something about itself.  Only a walk that already
# produced at least one gate-passing continuation point past its seed may have
# its subsequent failure read as a property of the family.  Both currently
# blocking endpoints have 2 and 3 accepted points respectively.
MIN_ACCEPTED_CONTINUATION_POINTS = 1

# MIN_STEP_REFINEMENTS -- three halvings, i.e. the failing step over eight.
#
# "The step was too long" and "there is nothing out there" are distinguished by
# shortening the step: a predictor closer to a point where the family certainly
# exists is strictly easier to correct.  The resolver's own retry ladder already
# halves down to its 5e-5 floor (2.5e-3 -> 7.8e-5, five halvings); the existence
# probe deliberately continues below that floor.  Three is the smallest number
# that makes "halve it, and halve it again" a plural claim.
MIN_STEP_REFINEMENTS = 3

# MIN_OUTWARD_PERSISTENCE_PROBES -- non-existence has to persist.
#
# One non-closing point is a point.  A frontier is a region: if the family stops
# existing at the frontier, it does not resume two steps further out.  Probing
# further out is also the cheapest available escape hatch, since a single closed
# orbit beyond the claimed frontier destroys the claim outright.
MIN_OUTWARD_PERSISTENCE_PROBES = 2

# ---------------------------------------------------------------------------
# Audit-corroboration window.
# ---------------------------------------------------------------------------
# AUDIT_LINE_REACH -- 0.02 in m1, 20 mass grid steps.
# How far along m1 an audit's SCAN LINE may sit and still be asked about this
# frontier.  The audits are m2 sweeps at fixed m1, and the family's lower
# frontier is a sloping curve m2 = F(m1): measured on the full-domain audit it
# runs from F(0.8925) ~ 0.730 to F(1.0425) ~ 0.795, so it moves by about 0.001 --
# one grid step -- per 0.0025 line spacing.  Comparisons are therefore made LINE
# BY LINE and never pooled across a 2-D box: pooling sixteen lines whose frontier
# differs by 0.02 would make every window contain both closing and non-closing
# probes, which would refuse every frontier there is and quietly turn this
# condition into a wall.  0.02 is the widest m1 offset at which the nearest line
# is still the same physics (about 8 grid steps of frontier motion); a nearer
# line is always preferred, and a line beyond this reach is simply not evidence.
AUDIT_LINE_REACH = 0.02

# AUDIT_OUTWARD_REACH -- 0.02 in m2, 20 mass grid steps.
# How far outward along a scan line the audits are asked about.  An order above
# the largest mass step the resolver ever requests (2.5e-3), so it covers the
# region the walk was actually heading into, and 20 grid steps, so it cannot be
# satisfied by a single lattice cell of silence.
AUDIT_OUTWARD_REACH = 0.02

# MIN_AUDIT_OUTWARD_PROBES -- the region must actually have been measured.
# Four probes outward of the frontier ON ONE SCAN LINE.  The full-domain audit
# spaces about 272 probes over the declared m2 width, so 0.0018 apart: this reach
# holds about eleven of them and four means the interval was sampled several
# times over.  It is deliberately more than the 35-line audit can put on one line
# (its m2 spacing is about 0.04, so at most one probe in reach), which fixes the
# asymmetry this whole module is built on: ANY committed audit may REFUTE a
# frontier with a single closing orbit, while corroborating one takes an audit
# that actually probed the interval densely.
MIN_AUDIT_OUTWARD_PROBES = 4

# MIN_AUDIT_FAILURE_DENSITY -- 1.0, every probe on the line.
# A converged probe outward of the claimed frontier IS a periodic orbit outward
# of the claimed frontier, so one of them refutes the claim.  Empirically
# attainable, not decorative: on the full-domain audit's m1 = 0.8925 line the
# lowest 17 probes (m2 0.70089 .. 0.72937) all fail and everything above closes,
# and on its m1 = 1.0425 line the lowest 53 all fail.  Anything below 1.0 would
# mean "mostly nothing exists out there", which is not a statement about a
# family.
MIN_AUDIT_FAILURE_DENSITY = 1.0

# MIN_MEASURED_CLOSURE_FRACTION -- 0.9 of the outward failures must carry a
# measured closure norm.
# A probe that timed out or raised is a statement about a CPU budget, not about
# the family: the full-domain audit records 23 TimeoutError and 43 "Required
# step" failures out of 2785, and those must not be counted as non-existence.
# Requiring nine in ten failures to carry a real residual keeps a window made of
# budget failures from corroborating anything.
MIN_MEASURED_CLOSURE_FRACTION = 0.9

# MIN_OUTWARD_TRAVEL -- one mass grid step.
# The walk must have gone somewhere before it stopped.  0.001 is the grid the
# census is defined on (scripts/assemble_critical_graph.MASS_GRID_STEP) and the
# finest distinction the sampling supports, so a walk that has not moved one
# grid step outward has not established that it was leaving anything.  The two
# blocking endpoints travelled 1.42e-3 and 2.19e-3 outward in m2.
MIN_OUTWARD_TRAVEL = 0.001

#: Mirrors scripts/assemble_critical_graph.MASS_GRID_STEP.
MASS_GRID_STEP = 0.001

#: Mirrors scripts/assemble_critical_graph.DOMAIN_TOLERANCE and
#: trace_label_invisible_continuous.DECLARED_DOMAIN.  A terminus this close to a
#: declared face is a domain exit, which is a different -- and better
#: established -- stop class.
DOMAIN_TOLERANCE = 0.0015
DECLARED_DOMAIN = {"m1": (0.8, 1.1), "m2": (0.7, 1.2)}

#: The name this module awards.  Never written by a producer's own verdict.
TERMINUS_KIND = "existence_boundary_terminus"

#: Evaluation order.  Cheapest and most decisive first, so the recorded refusal
#: names the condition a reviewer should look at.
CONDITION_ORDER = (
    "certified_departure",
    "inside_declared_domain",
    "outward_direction",
    "divergence",
    "step_refinement_invariance",
    "precision_invariance",
    "outward_persistence",
    "audit_corroboration",
)


def _number(value: Any) -> float | None:
    """A float, or None when the value is absent or not a number at all.

    Non-finite floats survive: an integrator that returned inf/nan MEASURED a
    blow-up, which is evidence.  A missing value measured nothing, which is not.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def closure_closes(closure: Any) -> bool:
    """True when this closure norm is a periodic orbit under the frozen gate."""
    value = _number(closure)
    return value is not None and math.isfinite(value) and value <= CLOSURE_GATE


def closure_is_divergent(closure: Any) -> bool:
    """True when this closure norm is a measured non-closure, orders off the gate.

    Non-finite counts (the flow blew up).  A finite residual must clear
    DIVERGENT_CLOSURE_FLOOR; between the gate and the floor is a corrector miss,
    which this returns False for, on purpose.
    """
    value = _number(closure)
    if value is None:
        return False
    if not math.isfinite(value):
        return True
    return value >= DIVERGENT_CLOSURE_FLOOR


def distance_to_declared_face(masses: Any) -> tuple[str, float] | None:
    """(face, distance) for the nearest declared face, or None if unreadable."""
    if not isinstance(masses, (list, tuple)) or len(masses) < 2:
        return None
    m1, m2 = _number(masses[0]), _number(masses[1])
    if m1 is None or m2 is None or not math.isfinite(m1) or not math.isfinite(m2):
        return None
    candidates = (
        (abs(m1 - DECLARED_DOMAIN["m1"][0]), "domain_m1_min"),
        (abs(m1 - DECLARED_DOMAIN["m1"][1]), "domain_m1_max"),
        (abs(m2 - DECLARED_DOMAIN["m2"][0]), "domain_m2_min"),
        (abs(m2 - DECLARED_DOMAIN["m2"][1]), "domain_m2_max"),
    )
    distance, face = min(candidates)
    return face, float(distance)


def inside_declared_domain(masses: Any) -> bool:
    if not isinstance(masses, (list, tuple)) or len(masses) < 2:
        return False
    m1, m2 = _number(masses[0]), _number(masses[1])
    if m1 is None or m2 is None or not math.isfinite(m1) or not math.isfinite(m2):
        return False
    return (
        DECLARED_DOMAIN["m1"][0] <= m1 <= DECLARED_DOMAIN["m1"][1]
        and DECLARED_DOMAIN["m2"][0] <= m2 <= DECLARED_DOMAIN["m2"][1]
    )


# ---------------------------------------------------------------------------
# Committed-audit corroboration.
#
# The audits are read as raw per-probe records -- m1, m2, ok, closure -- which
# is the same data scripts/audit_sign_topology.py wrote and the same data
# assemble_critical_graph.sign_topology_coverage reads.  Nothing here trusts an
# audit's own summary fields.
# ---------------------------------------------------------------------------


def audit_probes(payload: dict[str, Any]) -> list[dict[str, Any]]:
    probes = payload.get("probes")
    if not isinstance(probes, list):
        return []
    return [probe for probe in probes if isinstance(probe, dict)]


def load_audits(paths: list[Path] | tuple[Path, ...]) -> list[dict[str, Any]]:
    """Read committed sign-topology audits into {path, schema, probes} records.

    An unreadable or probe-less audit is kept, with an empty probe list, so that
    it is visible in the record as having corroborated nothing.
    """
    import json

    audits: list[dict[str, Any]] = []
    for path in paths:
        entry: dict[str, Any] = {"path": str(path), "schema": None, "probes": []}
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
            audits.append(entry)
            continue
        if isinstance(payload, dict):
            entry["schema"] = payload.get("schema")
            entry["probes"] = audit_probes(payload)
        audits.append(entry)
    return audits


def _nearest_converged(
    converged: list[dict[str, Any]], m2: float, sign: int
) -> float | None:
    """m2 of the closest probe, outward of m2, that DID close an orbit.

    Recorded on every window even when it refuses nothing, because it is the
    single most useful number for a reviewer: it says how far out the family was
    last seen to exist.
    """
    values = [
        value
        for value in (_number(probe.get("m2")) for probe in converged)
        if value is not None and math.isfinite(value)
    ]
    if not values:
        return None
    return float(min(values, key=lambda value: sign * (value - m2)))


def scan_lines(probes: list[dict[str, Any]]) -> dict[float, list[dict[str, Any]]]:
    """Group probes by their m1, which is how the audits actually fired them.

    scripts/audit_sign_topology.py sweeps m2 at fixed m1 and seeds each solve
    from its neighbour on the same line, so a line is one continuation march and
    the natural unit of comparison.  assemble_critical_graph.sign_topology_coverage
    reads the same artifacts the same way.
    """
    lines: dict[float, list[dict[str, Any]]] = {}
    for probe in probes:
        m1 = _number(probe.get("m1"))
        if m1 is None or not math.isfinite(m1):
            continue
        lines.setdefault(round(m1, 9), []).append(probe)
    return lines


def _summarize(probes: list[dict[str, Any]], m2: float, sign: int) -> dict[str, Any]:
    failures = [probe for probe in probes if not probe.get("ok")]
    converged = [probe for probe in probes if probe.get("ok")]
    closures = [
        value
        for value in (_number(probe.get("closure")) for probe in failures)
        if value is not None and math.isfinite(value)
    ]
    return {
        "probes": len(probes),
        "failures": len(failures),
        "converged": len(converged),
        "failure_density": (len(failures) / len(probes)) if probes else None,
        "failures_with_measured_closure": len(closures),
        "measured_closure_fraction": (len(closures) / len(failures)) if failures else None,
        "median_failed_closure": float(median(closures)) if closures else None,
        "min_failed_closure": float(min(closures)) if closures else None,
        "nearest_converged_m2": _nearest_converged(converged, m2, sign),
    }


def line_windows(
    line_probes: list[dict[str, Any]], m2: float, sign: int
) -> dict[str, Any]:
    """Split one scan line into what lies outward of m2 and what lies inward.

    ``sign`` is -1 for decreasing m2, +1 for increasing.  Outward is
    ``sign * (probe_m2 - m2) >= 0``, so a probe sitting exactly at the claimed
    frontier counts as outward: if it closed an orbit, the claim is already wrong.
    Both halves are summarized because an existence BOUNDARY needs both -- orbits
    absent outward, and orbits present inward.
    """
    outward: list[dict[str, Any]] = []
    inward: list[dict[str, Any]] = []
    for probe in line_probes:
        value = _number(probe.get("m2"))
        if value is None or not math.isfinite(value):
            continue
        offset = sign * (value - m2)
        if 0.0 <= offset <= AUDIT_OUTWARD_REACH:
            outward.append(probe)
        elif -AUDIT_OUTWARD_REACH <= offset < 0.0:
            inward.append(probe)
    return {
        "outward": _summarize(outward, m2, sign),
        "inward": _summarize(inward, m2, -sign),
    }


def _line_refutes(window: dict[str, Any]) -> tuple[bool, str]:
    """Does this line already disprove a frontier at the anchor?  Any audit may.

    This is the escape hatch, and it is deliberately cheap to trigger: one probe
    that closed a periodic orbit outward of the claimed frontier is enough, from
    any audit, however sparse.  Corroboration is the expensive direction.
    """
    outward = window.get("outward")
    if not isinstance(outward, dict):
        return False, ""
    converged = outward.get("converged")
    if isinstance(converged, int) and converged > 0:
        return True, (
            f"{converged} of {outward.get('probes')} probes outward of the "
            f"frontier CLOSED a periodic orbit, the nearest at m2 "
            f"{outward.get('nearest_converged_m2')}; the family exists out there"
        )
    return False, ""


def _line_qualifies(window: dict[str, Any]) -> tuple[bool, str]:
    """Does this scan line show a measured absence of orbits outward, and their
    presence inward?

    Reads the window with ``.get`` throughout: this same function re-derives a
    window a producer wrote into an artifact, and a key the producer left out must
    refuse rather than raise.
    """
    outward = window.get("outward")
    inward = window.get("inward")
    if not isinstance(outward, dict) or not isinstance(inward, dict):
        return False, "the line records no outward/inward summary"
    refutes, why = _line_refutes(window)
    if refutes:
        return False, why
    probes = outward.get("probes")
    if not isinstance(probes, int) or probes < MIN_AUDIT_OUTWARD_PROBES:
        return False, (
            f"only {probes} probes lie outward of the frontier on this line, "
            f"under the {MIN_AUDIT_OUTWARD_PROBES} needed for the interval to "
            "count as measured"
        )
    density = _number(outward.get("failure_density"))
    if density is None or density < MIN_AUDIT_FAILURE_DENSITY:
        return False, (
            f"the outward failure density is {density}, under the required "
            f"{MIN_AUDIT_FAILURE_DENSITY}"
        )
    fraction = _number(outward.get("measured_closure_fraction"))
    if fraction is None or fraction < MIN_MEASURED_CLOSURE_FRACTION:
        return False, (
            f"only {fraction} of the outward failures carry a measured closure "
            f"norm, under {MIN_MEASURED_CLOSURE_FRACTION}; a timeout or a solver "
            "exception is a statement about a CPU budget, not about the family"
        )
    closure = _number(outward.get("median_failed_closure"))
    if closure is None or closure < DIVERGENT_CLOSURE_FLOOR:
        return False, (
            f"the outward failures' median closure is {closure}, under the "
            f"{DIVERGENT_CLOSURE_FLOOR} divergence floor; that is an interval of "
            "corrector misses, not of absent orbits"
        )
    if not isinstance(inward.get("converged"), int) or inward["converged"] < 1:
        return False, (
            "no probe INWARD of the frontier closed an orbit on this line, so the "
            "audit does not locate the family's support here at all; an existence "
            "boundary needs orbits on one side of it"
        )
    return True, ""


def corroboration_measurement(
    audits: list[dict[str, Any]], m1: float, m2: float
) -> dict[str, Any]:
    """Ask every committed audit, on its nearest scan line, about (m1, m2).

    The outward side is DERIVED here rather than assumed: it is the side, if
    exactly one, that at least one audit corroborates and no audit refutes.
    Failures on both sides, or on neither, leave the frontier uncorroborated --
    which refuses, like everything else here.
    """
    sides: dict[str, Any] = {}
    for sign, name in ((-1, "decreasing_m2"), (1, "increasing_m2")):
        per_audit = []
        for audit in audits:
            probes = audit.get("probes") or []
            lines = scan_lines(probes)
            entry: dict[str, Any] = {
                "path": audit.get("path"),
                "schema": audit.get("schema"),
                "probe_count": len(probes),
                "scan_line_count": len(lines),
                "error": audit.get("error"),
            }
            if not lines:
                entry["nearest_line_m1"] = None
                entry["why_not"] = "this audit contributed no probes"
                entry["qualifies"] = False
                entry["refutes"] = False
                per_audit.append(entry)
                continue
            nearest = min(lines, key=lambda value: (abs(value - m1), value))
            entry["nearest_line_m1"] = nearest
            entry["line_distance_m1"] = abs(nearest - m1)
            if entry["line_distance_m1"] > AUDIT_LINE_REACH:
                entry["why_not"] = (
                    f"its nearest scan line m1={nearest} is "
                    f"{entry['line_distance_m1']:.3e} away, beyond the "
                    f"{AUDIT_LINE_REACH} line reach"
                )
                entry["qualifies"] = False
                entry["refutes"] = False
                per_audit.append(entry)
                continue
            window = line_windows(lines[nearest], m2, sign)
            entry.update(window)
            refutes, refutation = _line_refutes(window)
            qualifies, why = _line_qualifies(window)
            entry["refutes"] = refutes
            entry["refutation"] = refutation
            entry["qualifies"] = qualifies
            entry["why_not"] = why
            per_audit.append(entry)
        corroborating = [item["path"] for item in per_audit if item.get("qualifies")]
        refuting = [item["path"] for item in per_audit if item.get("refutes")]
        sides[name] = {
            "sign": sign,
            "audits": per_audit,
            "corroborating_audits": corroborating,
            "refuting_audits": refuting,
            "supported": bool(corroborating) and not refuting,
        }
    qualifying = [sides[name]["sign"] for name in sides if sides[name]["supported"]]
    return {
        "anchor_masses": [float(m1), float(m2)],
        "window": {
            "line_reach_m1": AUDIT_LINE_REACH,
            "outward_reach_m2": AUDIT_OUTWARD_REACH,
            "min_outward_probes_per_line": MIN_AUDIT_OUTWARD_PROBES,
        },
        "sides": sides,
        "audits_read": len(audits),
        "audit_probes_pooled": sum(len(audit.get("probes") or []) for audit in audits),
        "supported_outward_sign": qualifying[0] if len(qualifying) == 1 else None,
        "qualifying_sides": qualifying,
    }


# ---------------------------------------------------------------------------
# The conditions.  Each takes the measured payload recorded for it and returns
# (passed, refusal).  Each may only reject: there is no branch that turns a
# missing or unreadable measurement into a pass, and every bound is read from
# this module's constants rather than from the payload.
# ---------------------------------------------------------------------------


def _check_certified_departure(measured: dict[str, Any]) -> tuple[bool, str]:
    accepted = measured.get("accepted_points")
    if not isinstance(accepted, int) or accepted < MIN_ACCEPTED_CONTINUATION_POINTS:
        return False, (
            f"the walk accepted {accepted} continuation points past its seed, "
            f"under the {MIN_ACCEPTED_CONTINUATION_POINTS} required; a corrector "
            "that never worked here cannot testify about the family"
        )
    closure = _number(measured.get("last_accepted_closure"))
    if closure is None or not closure_closes(closure):
        return False, (
            f"the last accepted point's closure is {closure}, not inside the "
            f"{CLOSURE_GATE} gate, so the walk never stood on a certified orbit"
        )
    event = _number(measured.get("last_accepted_event"))
    if event is None or not math.isfinite(event) or abs(event) > EVENT_GATE:
        return False, (
            f"the last accepted point's event is {event}, outside the "
            f"{EVENT_GATE} gate, so it is not a certified critical point"
        )
    return True, ""


def _check_inside_declared_domain(measured: dict[str, Any]) -> tuple[bool, str]:
    masses = measured.get("frontier_masses")
    if not inside_declared_domain(masses):
        return False, (
            f"the frontier probe at {masses} is not inside the declared domain; "
            "a walk that left the box needs a certified domain crossing, which "
            "is a different stop class"
        )
    nearest = distance_to_declared_face(masses)
    if nearest is None:
        return False, "the frontier probe records no readable masses"
    face, distance = nearest
    if distance <= DOMAIN_TOLERANCE:
        return False, (
            f"the frontier probe sits {distance:.3e} from {face}, inside the "
            f"{DOMAIN_TOLERANCE} domain tolerance; that terminus is a declared "
            "domain exit, not a boundary of the family"
        )
    return True, ""


def _check_outward_direction(measured: dict[str, Any]) -> tuple[bool, str]:
    sign = measured.get("outward_sign")
    if sign not in (-1, 1):
        return False, f"the outward direction is {sign!r}, not -1 or +1"
    travel = _number(measured.get("outward_travel"))
    if travel is None or not math.isfinite(travel):
        return False, "the walk records no outward travel"
    if travel < MIN_OUTWARD_TRAVEL:
        return False, (
            f"the walk travelled {travel:.3e} outward in m2, under one grid step "
            f"({MIN_OUTWARD_TRAVEL}); it was stalling, not leaving"
        )
    if measured.get("monotone_outward") is not True:
        return False, (
            "the accepted points are not monotone outward in m2, so the walk was "
            "turning rather than heading out of the support"
        )
    return True, ""


def _check_divergence(measured: dict[str, Any]) -> tuple[bool, str]:
    closure = _number(measured.get("closure"))
    if closure is None:
        return False, (
            "the frontier probe recorded no closure norm, so nothing was "
            f"measured there ({measured.get('note') or 'no note'})"
        )
    if closure_closes(closure):
        return False, (
            f"the periodic orbit CLOSES at the frontier probe (closure "
            f"{closure:.3e} <= {CLOSURE_GATE}); the family exists here and the "
            "continuation failure is about the event zero set, not about "
            "existence"
        )
    if not closure_is_divergent(closure):
        return False, (
            f"the frontier probe's closure is {closure:.3e}, above the "
            f"{CLOSURE_GATE} gate but under the {DIVERGENT_CLOSURE_FLOOR} "
            "divergence floor; that is a marginal corrector miss"
        )
    return True, ""


def _check_step_refinement_invariance(measured: dict[str, Any]) -> tuple[bool, str]:
    ladder = measured.get("ladder")
    if not isinstance(ladder, list) or not ladder:
        return False, "no step-refinement ladder was recorded"
    steps: list[float] = []
    for entry in ladder:
        if not isinstance(entry, dict):
            return False, "a step-refinement entry is not a record"
        step = _number(entry.get("requested_step"))
        if step is None or not math.isfinite(step) or step <= 0.0:
            return False, f"a step-refinement entry has an unusable step {step!r}"
        if entry.get("closed") is True or closure_closes(entry.get("closure")):
            return False, (
                f"halving the mass step to {step:.3e} CLOSED the step; this was "
                "step control, not existence"
            )
        steps.append(step)
    largest, smallest = max(steps), min(steps)
    halvings = math.log2(largest / smallest) if smallest > 0.0 else 0.0
    if halvings + 1e-9 < MIN_STEP_REFINEMENTS:
        return False, (
            f"the step was only refined {halvings:.2f} halvings "
            f"({largest:.3e} -> {smallest:.3e}), under the "
            f"{MIN_STEP_REFINEMENTS} required"
        )
    if len(steps) < MIN_STEP_REFINEMENTS + 1:
        return False, (
            f"only {len(steps)} step attempts were recorded, under the "
            f"{MIN_STEP_REFINEMENTS + 1} needed to show a refinement sequence"
        )
    return True, ""


def _check_precision_invariance(measured: dict[str, Any]) -> tuple[bool, str]:
    for name in ("tightened_float64", "accelerated"):
        path = measured.get(name)
        if not isinstance(path, dict):
            return False, f"the {name} precision path recorded nothing"
        if path.get("available") is not True:
            return False, (
                f"the {name} precision path was unavailable "
                f"({path.get('note') or 'no note'}); a failure this module cannot "
                "cross-check at higher precision is refused, never assumed"
            )
        closure = _number(path.get("closure"))
        if closure is None:
            return False, (
                f"the {name} precision path recorded no closure norm "
                f"({path.get('note') or 'no note'})"
            )
        if path.get("closed") is True or closure_closes(closure):
            return False, (
                f"the {name} precision path CLOSED the orbit (closure "
                f"{closure:.3e}); the float64 failure was an arithmetic artifact"
            )
        if not closure_is_divergent(closure):
            return False, (
                f"the {name} precision path leaves closure {closure:.3e}, under "
                f"the {DIVERGENT_CLOSURE_FLOOR} divergence floor; the two "
                "arithmetics do not agree that nothing closes here"
            )
    return True, ""


def _check_outward_persistence(measured: dict[str, Any]) -> tuple[bool, str]:
    probes = measured.get("probes")
    if not isinstance(probes, list):
        return False, "no outward persistence probes were recorded"
    distances: list[float] = []
    for probe in probes:
        if not isinstance(probe, dict):
            return False, "an outward persistence probe is not a record"
        distance = _number(probe.get("distance_outward"))
        if distance is None or not math.isfinite(distance) or distance <= 0.0:
            return False, f"an outward probe has an unusable distance {distance!r}"
        closure = _number(probe.get("closure"))
        if closure is None:
            return False, (
                f"the outward probe at {distance:.3e} recorded no closure norm "
                f"({probe.get('note') or 'no note'})"
            )
        if probe.get("closed") is True or closure_closes(closure):
            return False, (
                f"a periodic orbit CLOSES {distance:.3e} beyond the claimed "
                f"frontier (closure {closure:.3e}); the family does not end here"
            )
        if not closure_is_divergent(closure):
            return False, (
                f"the outward probe at {distance:.3e} leaves closure "
                f"{closure:.3e}, under the {DIVERGENT_CLOSURE_FLOOR} divergence "
                "floor"
            )
        distances.append(distance)
    if len(set(round(d, 12) for d in distances)) < MIN_OUTWARD_PERSISTENCE_PROBES:
        return False, (
            f"only {len(distances)} distinct outward probes were recorded, under "
            f"the {MIN_OUTWARD_PERSISTENCE_PROBES} needed to show the absence "
            "persists"
        )
    return True, ""


def _check_audit_corroboration(measured: dict[str, Any]) -> tuple[bool, str]:
    """Re-derive the corroboration from the recorded per-line windows.

    Every clause is recomputed from the probe counts and closure statistics in
    the record, using this module's constants, so a producer cannot corroborate
    itself by writing ``supported: true``.
    """
    sign = measured.get("outward_sign")
    if sign not in (-1, 1):
        return False, f"the outward direction is {sign!r}, not -1 or +1"
    sides = measured.get("sides")
    if not isinstance(sides, dict):
        return False, "no committed-audit scan lines were recorded"
    name = "decreasing_m2" if sign < 0 else "increasing_m2"
    other_name = "increasing_m2" if sign < 0 else "decreasing_m2"
    side = sides.get(name)
    if not isinstance(side, dict):
        return False, f"the committed audits were not queried on the {name} side"
    audits = side.get("audits")
    if not isinstance(audits, list) or not audits:
        return False, "no committed audit was queried at all"
    corroborating: list[str] = []
    for entry in audits:
        if not isinstance(entry, dict):
            return False, "an audit entry is not a record"
        refutes, refutation = _line_refutes(entry)
        if refutes:
            return False, (
                f"{entry.get('path')} refutes a frontier here on its m1="
                f"{entry.get('nearest_line_m1')} scan line: {refutation}"
            )
        qualifies, _why = _line_qualifies(entry)
        if qualifies:
            corroborating.append(str(entry.get("path")))
    if not corroborating:
        reasons = "; ".join(
            f"{entry.get('path')}: {_line_qualifies(entry)[1]}"
            for entry in audits
            if isinstance(entry, dict)
        )
        return False, f"no committed audit corroborates a frontier here -- {reasons}"
    # The opposite side must NOT look like a frontier too.  If it does, the
    # audits are describing a gap in their own reach rather than a boundary of
    # the family, and nothing in them says which way the support lies.  The
    # inward-orbits clause of _line_qualifies already makes that nearly
    # impossible; this is kept because it costs nothing and can only reject.
    other = sides.get(other_name)
    other_audits = other.get("audits") if isinstance(other, dict) else None
    if isinstance(other_audits, list):
        for entry in other_audits:
            if isinstance(entry, dict) and _line_qualifies(entry)[0]:
                return False, (
                    f"{entry.get('path')} shows the SAME absence of orbits on the "
                    f"{other_name} side, so this is a hole in the audit's reach "
                    "rather than one boundary of the family"
                )
    return True, ""


CONDITIONS = {
    "certified_departure": _check_certified_departure,
    "inside_declared_domain": _check_inside_declared_domain,
    "outward_direction": _check_outward_direction,
    "divergence": _check_divergence,
    "step_refinement_invariance": _check_step_refinement_invariance,
    "precision_invariance": _check_precision_invariance,
    "outward_persistence": _check_outward_persistence,
    "audit_corroboration": _check_audit_corroboration,
}

#: Human-readable statement of what each condition can only ever do.
CONDITION_INTENT = {
    "certified_departure": (
        "the walk produced at least one gate-passing continuation point past "
        "its seed, so its corrector demonstrably worked on this branch"
    ),
    "inside_declared_domain": (
        "the frontier is strictly inside the declared mass box, so this is not "
        "an uncertified domain exit wearing a new name"
    ),
    "outward_direction": (
        "the walk was monotonically leaving the family's support, by at least "
        "one mass grid step, rather than stalling"
    ),
    "divergence": (
        "no periodic orbit closes at the frontier probe, and the residual is "
        "orders outside the frozen gate rather than marginally above it"
    ),
    "step_refinement_invariance": (
        "shortening the mass step, repeatedly, does not rescue the step"
    ),
    "precision_invariance": (
        "a tightened float64 solve and an independent accelerated integrator "
        "both agree that nothing closes there"
    ),
    "outward_persistence": (
        "probes further outward also fail to close, so the absence is a region "
        "rather than a point"
    ),
    "audit_corroboration": (
        "the committed sign-topology audits independently show every probe "
        "outward of this frontier failing to close, with order-one residuals"
    ),
}


def cross_check(conditions: dict[str, Any]) -> list[str]:
    """Objections to conditions that pass individually but describe two places.

    Each condition above judges its own measurement, so nothing in them notices
    if the corroboration window was anchored somewhere other than the frontier
    that is being claimed, or if the direction the walk reports disagrees with
    the direction the audits were asked about.  Either would let a genuine
    measurement of one point vouch for a different one, which is the same defect
    the assembler's both-ends walk check exists to catch.
    """
    objections: list[str] = []
    inside = conditions.get("inside_declared_domain")
    audit = conditions.get("audit_corroboration")
    direction = conditions.get("outward_direction")
    if not isinstance(inside, dict) or not isinstance(audit, dict):
        return objections
    frontier = inside.get("frontier_masses")
    anchor = audit.get("anchor_masses")
    if not isinstance(frontier, (list, tuple)) or len(frontier) < 2:
        return objections
    if not isinstance(anchor, (list, tuple)) or len(anchor) < 2:
        objections.append("the corroboration window records no anchor masses")
    else:
        drift = [
            abs((_number(anchor[index]) or 0.0) - (_number(frontier[index]) or 0.0))
            for index in range(2)
        ]
        if max(drift) > 0.0:
            objections.append(
                f"the corroboration window is anchored at {list(anchor)[:2]} but the "
                f"frontier is claimed at {list(frontier)[:2]}; the audits were asked "
                "about a different point"
            )
    if isinstance(direction, dict) and direction.get("outward_sign") != audit.get(
        "outward_sign"
    ):
        objections.append(
            f"the walk reports outward sign {direction.get('outward_sign')!r} while "
            f"the audits were queried with {audit.get('outward_sign')!r}"
        )
    return objections


def evaluate(conditions: dict[str, Any]) -> dict[str, Any]:
    """Award or refuse ``existence_boundary_terminus`` from measured conditions.

    ``conditions`` maps each name in CONDITION_ORDER to the measured payload for
    it.  Every condition is evaluated -- so the record carries all of the
    numbers a reviewer needs -- and the class is awarded only when all of them
    pass.  The refusal names the first failing condition in CONDITION_ORDER.
    """
    verdict: dict[str, Any] = {}
    refusals: list[str] = []
    for name in CONDITION_ORDER:
        checker = CONDITIONS[name]
        measured = conditions.get(name)
        if not isinstance(measured, dict):
            passed, why = False, "this condition recorded no measurement at all"
        else:
            passed, why = checker(measured)
        verdict[name] = {
            "passed": bool(passed),
            "intent": CONDITION_INTENT[name],
            "measured": measured,
            "refusal": why,
        }
        if not passed:
            refusals.append(f"{name}: {why}")
    crossed = cross_check(conditions)
    refusals.extend(f"cross_check: {item}" for item in crossed)
    awarded = not refusals
    return {
        "class": TERMINUS_KIND,
        "awarded": awarded,
        "refused_conditions": [name for name in CONDITION_ORDER if not verdict[name]["passed"]],
        "refusal_reason": refusals[0] if refusals else "",
        "all_refusals": refusals,
        "cross_checks": crossed,
        "conditions": verdict,
        "thresholds": {
            "closure_gate": CLOSURE_GATE,
            "event_gate": EVENT_GATE,
            "divergent_closure_floor": DIVERGENT_CLOSURE_FLOOR,
            "min_accepted_continuation_points": MIN_ACCEPTED_CONTINUATION_POINTS,
            "min_step_refinements": MIN_STEP_REFINEMENTS,
            "min_outward_persistence_probes": MIN_OUTWARD_PERSISTENCE_PROBES,
            "audit_line_reach_m1": AUDIT_LINE_REACH,
            "audit_outward_reach_m2": AUDIT_OUTWARD_REACH,
            "min_audit_outward_probes": MIN_AUDIT_OUTWARD_PROBES,
            "min_audit_failure_density": MIN_AUDIT_FAILURE_DENSITY,
            "min_measured_closure_fraction": MIN_MEASURED_CLOSURE_FRACTION,
            "min_outward_travel_m2": MIN_OUTWARD_TRAVEL,
            "domain_tolerance": DOMAIN_TOLERANCE,
        },
    }


def recheck(record: Any) -> list[str]:
    """Re-derive a recorded verdict from its own numbers; list every objection.

    A consumer must never bind a terminus on a producer's ``awarded`` flag.  This
    replays every condition against THIS module's constants using the measured
    payloads the producer wrote down, so a hand-edited verdict, a missing
    condition, or a threshold the producer relaxed on its own authority all
    show up as objections.  An empty list means the record's own numbers earn
    the class.
    """
    objections: list[str] = []
    if not isinstance(record, dict):
        return ["the existence-boundary record is not an object"]
    if str(record.get("class") or "") != TERMINUS_KIND:
        objections.append(
            f"the record claims class {record.get('class')!r}, not {TERMINUS_KIND!r}"
        )
    conditions = record.get("conditions")
    if not isinstance(conditions, dict):
        return objections + ["the record carries no conditions to re-derive"]
    measurements: dict[str, Any] = {}
    for name in CONDITION_ORDER:
        entry = conditions.get(name)
        if not isinstance(entry, dict):
            objections.append(f"{name}: not recorded")
            continue
        measured = entry.get("measured")
        if not isinstance(measured, dict):
            objections.append(f"{name}: recorded no measurement")
            continue
        measurements[name] = measured
        passed, why = CONDITIONS[name](measured)
        if not passed:
            objections.append(f"{name}: {why}")
        elif entry.get("passed") is not True:
            objections.append(
                f"{name}: the producer recorded passed={entry.get('passed')!r} "
                "while its own numbers pass; the record disagrees with itself"
            )
    objections.extend(f"cross_check: {item}" for item in cross_check(measurements))
    if record.get("awarded") is not True and not objections:
        objections.append(
            "every condition's numbers pass but the record was not awarded; the "
            "record disagrees with itself"
        )
    return objections
