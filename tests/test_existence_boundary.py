"""The fourth continuation stop class, tested from the refusing side first.

``threebody_atlas.existence_boundary`` decides whether a continuation failure is
the periodic family running out or the corrector running out.  Every condition it
applies may only REJECT, so the tests that matter are the ones that hand it a
nearly-passing record with one thing wrong and require a refusal naming that
thing.  One test also awards the class, because a predicate that can never pass
is a wall wearing a test's clothes, not a test.

The synthetic records here are FIXTURES, never evidence.  They exist so the
logic can be covered on a machine without JAX/Diffrax, where the resolver that
takes the real measurements cannot run at all.  The two integration tests at the
bottom read the committed audits and are the only ones that touch real numbers.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import pytest

from threebody_atlas import existence_boundary as eb


ROOT = Path(__file__).resolve().parents[1]
AUDIT_35LINE = ROOT / "research/evidence/V1_SIGN_TOPOLOGY_AUDIT_INDEPENDENT_35LINE_2026-08-17.json"
AUDIT_FULLDOMAIN = ROOT / "research/evidence/V1_SIGN_TOPOLOGY_AUDIT_FULLDOMAIN_2026-08-18.json"

# A synthetic scan line: orbits close above m2 = 0.7300 and nothing closes below
# it, which is the shape the full-domain audit actually exhibits at m1 = 0.8925
# (its lowest 17 probes, m2 0.70089 .. 0.72937, all fail with order-one closure).
SYNTHETIC_LINE_M1 = 0.8925
SYNTHETIC_FRONTIER_M2 = 0.7300


def _synthetic_probes(
    *,
    frontier: float = SYNTHETIC_FRONTIER_M2,
    m1: float = SYNTHETIC_LINE_M1,
    closure: float = 1.85,
    measured: bool = True,
) -> list[dict[str, object]]:
    probes = []
    value = 0.7008
    while value < 0.7700:
        if value < frontier:
            probes.append(
                {
                    "m1": m1,
                    "m2": value,
                    "ok": False,
                    "closure": closure if measured else None,
                    "note": (
                        f"closure {closure:.3e} > 1e-07"
                        if measured
                        else "TimeoutError: probe exceeded its budget"
                    ),
                }
            )
        else:
            probes.append({"m1": m1, "m2": value, "ok": True, "closure": 3.1e-12})
        value += 0.0018
    return probes


def _synthetic_audit(**kwargs: object) -> list[dict[str, object]]:
    return [
        {
            # A repository path, because the assembler additionally requires that
            # anything credited with corroborating a frontier be a file that can
            # be re-read.  The STATISTICS below are synthetic.
            "path": "research/evidence/V1_SIGN_TOPOLOGY_AUDIT_FULLDOMAIN_2026-08-18.json",
            "schema": "atlas.v1.sign-topology-audit/1",
            "probes": _synthetic_probes(**kwargs),  # type: ignore[arg-type]
        }
    ]


def _ladder(*, rescued_at: float | None = None) -> list[dict[str, object]]:
    """The step-retry ladder as the resolver records it: 2.5e-3 down to 2.0e-5."""
    entries: list[dict[str, object]] = []
    step = 2.5e-3
    for _ in range(8):
        if rescued_at is not None and math.isclose(step, rescued_at, rel_tol=1e-9):
            entries.append(
                {"requested_step": step, "closed": True, "closure": 5e-12, "event": 1e-9}
            )
            break
        entries.append(
            {
                "requested_step": step,
                "closed": False,
                "error": "event-normal corrector missed frozen gate after 5 iterations",
            }
        )
        step *= 0.5
    return entries


def _conditions(**overrides: object) -> dict[str, object]:
    """A record whose every measured number earns the class, before overrides."""
    frontier = [SYNTHETIC_LINE_M1, 0.7290]
    corroboration = eb.corroboration_measurement(
        _synthetic_audit(), frontier[0], frontier[1]
    )
    corroboration["outward_sign"] = -1
    conditions: dict[str, object] = {
        "certified_departure": {
            "accepted_points": 3,
            "last_accepted_closure": 4.2e-11,
            "last_accepted_event": 8.1e-9,
            "last_accepted_masses": [SYNTHETIC_LINE_M1, 0.7315, 1.0],
        },
        "inside_declared_domain": {"frontier_masses": list(frontier)},
        "outward_direction": {
            "axis": "m2",
            "outward_sign": -1,
            "outward_travel": 3.5e-3,
            "monotone_outward": True,
            "walk_m2": [0.7350, 0.7332, 0.7315],
            "per_step_outward": [1.8e-3, 1.7e-3],
            "tangent_m2": -0.981,
        },
        "divergence": {
            "closure": 1.85,
            "event": None,
            "solver_success": False,
            "distance_outward": 1.95e-5,
            "note": "",
        },
        "step_refinement_invariance": {"ladder": _ladder()},
        "precision_invariance": {
            "tightened_float64": {
                "available": True,
                "closed": False,
                "closure": 1.84,
                "note": "",
            },
            "accelerated": {
                "available": True,
                "closed": False,
                "closure": 1.86,
                "iterations": 12,
                "note": "no descent after 6 backtracks",
            },
        },
        "outward_persistence": {
            "probes": [
                {"distance_outward": 2.5e-3, "closed": False, "closure": 1.9, "note": ""},
                {"distance_outward": 5.0e-3, "closed": False, "closure": 2.1, "note": ""},
                {"distance_outward": 1.0e-2, "closed": False, "closure": 2.4, "note": ""},
            ]
        },
        "audit_corroboration": corroboration,
    }
    conditions.update(overrides)
    return conditions


def _refusal(verdict: dict[str, object]) -> str:
    return str(verdict["refusal_reason"])


def test_a_fully_measured_frontier_is_awarded() -> None:
    """The predicate must be able to pass, or it is not a test of anything.

    Every condition is satisfied by a measurement here, including corroboration
    computed by the real window code over a synthetic scan line shaped like the
    one the full-domain audit shows at m1 0.8925.
    """
    verdict = eb.evaluate(_conditions())
    assert verdict["awarded"] is True, verdict["all_refusals"]
    assert verdict["refused_conditions"] == []
    assert verdict["class"] == "existence_boundary_terminus"
    assert sorted(verdict["conditions"]) == sorted(eb.CONDITION_ORDER)
    assert eb.recheck({**verdict, "class": eb.TERMINUS_KIND}) == []


def test_a_marginal_residual_is_a_corrector_miss_and_is_refused() -> None:
    """3e-7 is above the 1e-7 gate and five orders below the divergence floor."""
    verdict = eb.evaluate(
        _conditions(
            divergence={
                "closure": 3e-7,
                "event": None,
                "solver_success": False,
                "distance_outward": 1.95e-5,
                "note": "",
            }
        )
    )
    assert verdict["awarded"] is False
    assert "divergence" in verdict["refused_conditions"]
    assert "marginal corrector miss" in _refusal(verdict)


def test_a_closing_orbit_at_the_frontier_probe_is_refused() -> None:
    """If the orbit closes there the family exists there, whatever else failed.

    This is the case that separates the end of a CURVE from the end of a FAMILY.
    A projection fold, a tangency, or a corrector that simply lost the event zero
    all leave a closed orbit sitting at the frontier probe, and none of them is a
    boundary of the family.
    """
    verdict = eb.evaluate(
        _conditions(
            divergence={
                "closure": 3.4e-12,
                "event": 4.1e-4,
                "solver_success": True,
                "distance_outward": 1.95e-5,
                "note": "",
            }
        )
    )
    assert verdict["awarded"] is False
    assert "divergence" in verdict["refused_conditions"]
    assert "CLOSES at the frontier probe" in _refusal(verdict)


def test_an_unmeasured_frontier_probe_is_refused() -> None:
    """A failure with no residual is a budget failure, not a measurement."""
    verdict = eb.evaluate(
        _conditions(
            divergence={
                "closure": None,
                "event": None,
                "solver_success": False,
                "distance_outward": 1.95e-5,
                "note": "TimeoutError: probe exceeded its budget",
            }
        )
    )
    assert verdict["awarded"] is False
    assert "recorded no closure norm" in _refusal(verdict)


def test_a_step_rescued_failure_is_refused() -> None:
    """A shorter step that closes means step control stopped the walk."""
    verdict = eb.evaluate(
        _conditions(step_refinement_invariance={"ladder": _ladder(rescued_at=3.125e-4)})
    )
    assert verdict["awarded"] is False
    assert "step_refinement_invariance" in verdict["refused_conditions"]
    assert "step control, not existence" in _refusal(verdict)


def test_too_few_halvings_is_refused() -> None:
    """Two attempts is not "halve it, and halve it again"."""
    verdict = eb.evaluate(
        _conditions(
            step_refinement_invariance={
                "ladder": [
                    {"requested_step": 2.5e-3, "closed": False, "error": "missed gate"},
                    {"requested_step": 1.25e-3, "closed": False, "error": "missed gate"},
                ]
            }
        )
    )
    assert verdict["awarded"] is False
    assert "step_refinement_invariance" in verdict["refused_conditions"]
    assert "refined" in _refusal(verdict)


def test_a_precision_path_that_closes_the_orbit_is_refused() -> None:
    verdict = eb.evaluate(
        _conditions(
            precision_invariance={
                "tightened_float64": {
                    "available": True,
                    "closed": False,
                    "closure": 1.84,
                    "note": "",
                },
                "accelerated": {
                    "available": True,
                    "closed": True,
                    "closure": 6.7e-9,
                    "iterations": 5,
                    "note": "",
                },
            }
        )
    )
    assert verdict["awarded"] is False
    assert "precision_invariance" in verdict["refused_conditions"]
    assert "arithmetic artifact" in _refusal(verdict)


def test_an_unavailable_precision_path_is_refused_not_assumed() -> None:
    """The condition the task calls out explicitly: no reachable precision path
    must REJECT rather than silently pass."""
    verdict = eb.evaluate(
        _conditions(
            precision_invariance={
                "tightened_float64": {
                    "available": True,
                    "closed": False,
                    "closure": 1.84,
                    "note": "",
                },
                "accelerated": {
                    "available": False,
                    "closed": False,
                    "closure": None,
                    "note": "RuntimeError: JAX + Diffrax are required",
                },
            }
        )
    )
    assert verdict["awarded"] is False
    assert "precision_invariance" in verdict["refused_conditions"]
    assert "unavailable" in _refusal(verdict)


def test_a_precision_path_that_only_half_agrees_is_refused() -> None:
    """A small-but-not-closing residual in the other arithmetic is disagreement.

    If Diffrax leaves 1e-9 where SciPy left 1.8, the SciPy residual was a
    quadrature artifact even though neither run formally closed the orbit.
    """
    verdict = eb.evaluate(
        _conditions(
            precision_invariance={
                "tightened_float64": {
                    "available": True,
                    "closed": False,
                    "closure": 1.84,
                    "note": "",
                },
                "accelerated": {
                    "available": True,
                    "closed": False,
                    "closure": 9.0e-7,
                    "iterations": 12,
                    "note": "",
                },
            }
        )
    )
    assert verdict["awarded"] is False
    assert "do not agree" in _refusal(verdict)


def test_an_orbit_closing_further_out_is_refused() -> None:
    """Non-existence has to persist; the family must not resume two steps on."""
    verdict = eb.evaluate(
        _conditions(
            outward_persistence={
                "probes": [
                    {"distance_outward": 2.5e-3, "closed": False, "closure": 1.9, "note": ""},
                    {"distance_outward": 5.0e-3, "closed": True, "closure": 2.2e-11, "note": ""},
                ]
            }
        )
    )
    assert verdict["awarded"] is False
    assert "outward_persistence" in verdict["refused_conditions"]
    assert "does not end here" in _refusal(verdict)


def test_a_single_outward_probe_is_refused() -> None:
    verdict = eb.evaluate(
        _conditions(
            outward_persistence={
                "probes": [
                    {"distance_outward": 2.5e-3, "closed": False, "closure": 1.9, "note": ""}
                ]
            }
        )
    )
    assert verdict["awarded"] is False
    assert "outward_persistence" in verdict["refused_conditions"]


def test_a_stalling_walk_is_refused() -> None:
    """Heading OUT means travelling at least one mass grid step outward."""
    verdict = eb.evaluate(
        _conditions(
            outward_direction={
                "axis": "m2",
                "outward_sign": -1,
                "outward_travel": 2.0e-4,
                "monotone_outward": True,
                "walk_m2": [0.7292, 0.7290],
                "per_step_outward": [2.0e-4],
                "tangent_m2": -0.98,
            }
        )
    )
    assert verdict["awarded"] is False
    assert "outward_direction" in verdict["refused_conditions"]
    assert "stalling" in _refusal(verdict)


def test_a_turning_walk_is_refused() -> None:
    verdict = eb.evaluate(
        _conditions(
            outward_direction={
                "axis": "m2",
                "outward_sign": -1,
                "outward_travel": 3.5e-3,
                "monotone_outward": False,
                "walk_m2": [0.7350, 0.7300, 0.7315],
                "per_step_outward": [5.0e-3, -1.5e-3],
                "tangent_m2": -0.98,
            }
        )
    )
    assert verdict["awarded"] is False
    assert "turning" in _refusal(verdict)


def test_a_walk_the_audits_say_is_heading_inward_is_refused() -> None:
    """Direction is derived from the audits too, and the two must agree."""
    conditions = _conditions()
    corroboration = dict(conditions["audit_corroboration"])  # type: ignore[arg-type]
    corroboration["outward_sign"] = 1
    verdict = eb.evaluate(
        _conditions(
            audit_corroboration=corroboration,
            outward_direction={
                "axis": "m2",
                "outward_sign": 1,
                "outward_travel": 3.5e-3,
                "monotone_outward": True,
                "walk_m2": [0.7250, 0.7270, 0.7290],
                "per_step_outward": [2.0e-3, 2.0e-3],
                "tangent_m2": 0.981,
            },
        )
    )
    assert verdict["awarded"] is False
    assert "audit_corroboration" in verdict["refused_conditions"]
    assert "CLOSED a periodic orbit" in _refusal(verdict)


def test_a_frontier_outside_the_audited_region_is_refused() -> None:
    """The out-of-region case: the audits closed orbits where the walk stopped.

    This is the shape of the two endpoints that actually block the release: the
    frontier they would claim sits well inside the region where the committed
    full-domain audit closes periodic orbits.
    """
    frontier = [SYNTHETIC_LINE_M1, 0.7500]
    corroboration = eb.corroboration_measurement(
        _synthetic_audit(), frontier[0], frontier[1]
    )
    corroboration["outward_sign"] = -1
    verdict = eb.evaluate(
        _conditions(
            audit_corroboration=corroboration,
            inside_declared_domain={"frontier_masses": frontier},
        )
    )
    assert verdict["awarded"] is False
    assert "audit_corroboration" in verdict["refused_conditions"]
    assert "refutes a frontier here" in _refusal(verdict)


def test_an_audit_of_timeouts_does_not_corroborate() -> None:
    """A CPU budget is not a statement about the three-body problem."""
    frontier = [SYNTHETIC_LINE_M1, 0.7290]
    corroboration = eb.corroboration_measurement(
        _synthetic_audit(measured=False), frontier[0], frontier[1]
    )
    corroboration["outward_sign"] = -1
    verdict = eb.evaluate(_conditions(audit_corroboration=corroboration))
    assert verdict["awarded"] is False
    assert "audit_corroboration" in verdict["refused_conditions"]
    assert "CPU budget" in _refusal(verdict)


def test_a_sparse_audit_cannot_corroborate_on_its_own() -> None:
    """Corroboration needs a densely probed interval; refutation needs one probe."""
    sparse = [
        {
            "path": "research/evidence/V1_SIGN_TOPOLOGY_AUDIT_INDEPENDENT_35LINE_2026-08-17.json",
            "schema": "atlas.v1.sign-topology-audit/1",
            "probes": [
                {"m1": SYNTHETIC_LINE_M1, "m2": 0.7284, "ok": False, "closure": 1.85},
                {"m1": SYNTHETIC_LINE_M1, "m2": 0.7684, "ok": True, "closure": 2e-12},
            ],
        }
    ]
    corroboration = eb.corroboration_measurement(sparse, SYNTHETIC_LINE_M1, 0.7290)
    corroboration["outward_sign"] = -1
    verdict = eb.evaluate(_conditions(audit_corroboration=corroboration))
    assert verdict["awarded"] is False
    assert "no committed audit corroborates" in _refusal(verdict)


def test_no_audits_at_all_is_refused() -> None:
    corroboration = eb.corroboration_measurement([], SYNTHETIC_LINE_M1, 0.7290)
    corroboration["outward_sign"] = -1
    verdict = eb.evaluate(_conditions(audit_corroboration=corroboration))
    assert verdict["awarded"] is False
    assert "audit_corroboration" in verdict["refused_conditions"]


def test_a_frontier_on_a_declared_face_is_a_domain_exit_not_this_class() -> None:
    frontier = [0.8, 0.7290]
    corroboration = eb.corroboration_measurement(
        _synthetic_audit(m1=0.8), frontier[0], frontier[1]
    )
    corroboration["outward_sign"] = -1
    verdict = eb.evaluate(
        _conditions(
            inside_declared_domain={"frontier_masses": frontier},
            audit_corroboration=corroboration,
        )
    )
    assert verdict["awarded"] is False
    assert "inside_declared_domain" in verdict["refused_conditions"]
    assert "declared domain exit" in _refusal(verdict)


def test_a_frontier_outside_the_declared_box_is_refused() -> None:
    verdict = eb.evaluate(
        _conditions(inside_declared_domain={"frontier_masses": [0.8925, 0.6980]})
    )
    assert verdict["awarded"] is False
    assert "not inside the declared domain" in _refusal(verdict)


def test_a_corrector_that_never_worked_here_cannot_testify() -> None:
    verdict = eb.evaluate(
        _conditions(
            certified_departure={
                "accepted_points": 0,
                "last_accepted_closure": None,
                "last_accepted_event": None,
                "last_accepted_masses": None,
            }
        )
    )
    assert verdict["awarded"] is False
    assert "certified_departure" in verdict["refused_conditions"]


def test_a_departure_from_an_uncertified_point_is_refused() -> None:
    verdict = eb.evaluate(
        _conditions(
            certified_departure={
                "accepted_points": 2,
                "last_accepted_closure": 4.0e-6,
                "last_accepted_event": 8.1e-9,
                "last_accepted_masses": [SYNTHETIC_LINE_M1, 0.7315, 1.0],
            }
        )
    )
    assert verdict["awarded"] is False
    assert "not inside the" in _refusal(verdict)


def test_every_missing_condition_refuses() -> None:
    """A condition that was never measured must refuse exactly like a failed one."""
    for name in eb.CONDITION_ORDER:
        conditions = _conditions()
        conditions.pop(name)
        verdict = eb.evaluate(conditions)
        assert verdict["awarded"] is False, name
        assert name in verdict["refused_conditions"], name


def test_a_window_anchored_somewhere_else_is_refused() -> None:
    """Every condition can pass on its own and still describe two places.

    A corroboration window measured where the audits DO show a frontier cannot
    vouch for a terminus somewhere else; that is the same defect as reporting one
    terminus's miss distance at the other end of the walk.
    """
    conditions = _conditions()
    corroboration = dict(conditions["audit_corroboration"])  # type: ignore[arg-type]
    corroboration["anchor_masses"] = [SYNTHETIC_LINE_M1, 0.7100]
    verdict = eb.evaluate(_conditions(audit_corroboration=corroboration))
    assert verdict["awarded"] is False
    assert verdict["refused_conditions"] == []
    assert "anchored at" in _refusal(verdict)
    assert eb.cross_check(_conditions(audit_corroboration=corroboration))


def test_a_direction_that_disagrees_with_the_queried_side_is_refused() -> None:
    conditions = _conditions()
    direction = dict(conditions["outward_direction"])  # type: ignore[arg-type]
    direction["outward_sign"] = 1
    direction["walk_m2"] = [0.7250, 0.7270, 0.7290]
    direction["per_step_outward"] = [2.0e-3, 2.0e-3]
    verdict = eb.evaluate(_conditions(outward_direction=direction))
    assert verdict["awarded"] is False
    assert "while the audits were queried with" in _refusal(verdict)


def test_recheck_refuses_a_hand_edited_verdict() -> None:
    """A consumer must never bind on the producer's own boolean."""
    verdict = eb.evaluate(
        _conditions(
            divergence={
                "closure": 3e-7,
                "event": None,
                "solver_success": False,
                "distance_outward": 1.95e-5,
                "note": "",
            }
        )
    )
    forged = json.loads(json.dumps(verdict))
    forged["awarded"] = True
    forged["refused_conditions"] = []
    forged["refusal_reason"] = ""
    forged["conditions"]["divergence"]["passed"] = True
    objections = eb.recheck(forged)
    assert objections
    assert any("marginal corrector miss" in item for item in objections)


def test_recheck_refuses_a_relaxed_threshold_written_into_the_record() -> None:
    """Thresholds are read from the library, never from the artifact."""
    verdict = eb.evaluate(_conditions())
    forged = json.loads(json.dumps(verdict))
    forged["thresholds"]["divergent_closure_floor"] = 1e-9
    forged["conditions"]["divergence"]["measured"]["closure"] = 5e-9
    objections = eb.recheck(forged)
    assert any("divergence" in item for item in objections)


def test_recheck_refuses_a_record_missing_a_condition() -> None:
    verdict = json.loads(json.dumps(eb.evaluate(_conditions())))
    verdict["conditions"].pop("audit_corroboration")
    assert any("audit_corroboration: not recorded" in item for item in eb.recheck(verdict))


def test_recheck_refuses_a_record_that_disagrees_with_itself() -> None:
    verdict = json.loads(json.dumps(eb.evaluate(_conditions())))
    verdict["conditions"]["divergence"]["passed"] = False
    assert any("disagrees with itself" in item for item in eb.recheck(verdict))


def test_the_divergence_floor_is_the_number_the_committed_audits_exhibit() -> None:
    """1e-2 is derived, not chosen: re-derive it from the artifacts here.

    If a future audit records a non-closing probe whose residual is not orders
    off the gate, this fails, and the right response is to look at that probe --
    not to lower the floor.
    """
    assert eb.DIVERGENT_CLOSURE_FLOOR >= 1e5 * eb.CLOSURE_GATE
    payload = json.loads(AUDIT_35LINE.read_text(encoding="utf-8"))
    closures = [
        float(probe["closure"])
        for probe in payload["probes"]
        if not probe.get("ok")
        and isinstance(probe.get("closure"), (int, float))
        and math.isfinite(float(probe["closure"]))
    ]
    assert len(closures) == 97
    assert min(closures) > eb.DIVERGENT_CLOSURE_FLOOR
    assert all(eb.closure_is_divergent(value) for value in closures)
    assert not eb.closure_is_divergent(3e-7)
    assert not eb.closure_is_divergent(eb.CLOSURE_GATE)
    assert eb.closure_closes(eb.CLOSURE_GATE)


def test_the_committed_audits_refuse_the_two_blocking_endpoints() -> None:
    """The honest scientific finding, pinned.

    CI run 32146630216 leaves minus_one component 0 (low) and component 1 (high)
    unresolved, each having walked about 2e-3 DOWNWARD in m2 before the
    event-normal corrector failed, at (0.891661, 0.751659) and (1.044088,
    0.855789).  The prose that motivates an existence-boundary class quotes the
    35-line audit's failed-probe MEDIAN m2 of 0.820, but a median is not a
    frontier: read per scan line, the full-domain audit closes periodic orbits
    on both sides of both points -- eleven and twelve of them within 0.02 in m2,
    the nearest at m2 0.74744 and 0.85383.  So the family demonstrably exists
    where those two walks stopped, and the corroboration condition refuses.

    This test exists so that a future change which starts awarding the class
    there has to explain itself against these numbers.
    """
    audits = eb.load_audits([AUDIT_35LINE, AUDIT_FULLDOMAIN])
    assert [len(audit["probes"]) for audit in audits] == [465, 33701]
    for m1, m2 in ((0.891080, 0.749227), (1.045815, 0.853982)):
        measurement = eb.corroboration_measurement(audits, m1, m2)
        measurement["outward_sign"] = -1
        assert measurement["supported_outward_sign"] is None
        passed, why = eb.CONDITIONS["audit_corroboration"](measurement)
        assert passed is False
        assert "CLOSED a periodic orbit" in why
        full = next(
            item
            for item in measurement["sides"]["decreasing_m2"]["audits"]
            if "FULLDOMAIN" in str(item["path"])
        )
        assert full["refutes"] is True
        assert full["outward"]["converged"] >= 8


def test_the_committed_audits_do_corroborate_below_the_measured_frontier() -> None:
    """And the same code awards where the artifacts really do show a frontier.

    On the full-domain audit's m1 = 0.8925 scan line the lowest 17 probes fail
    with order-one closure and everything above m2 0.7312 closes.  A terminus at
    m2 0.7290 on that line is corroborated, with the outward direction derived
    from the probes rather than assumed.
    """
    audits = eb.load_audits([AUDIT_35LINE, AUDIT_FULLDOMAIN])
    measurement = eb.corroboration_measurement(audits, 0.8925, 0.7290)
    assert measurement["supported_outward_sign"] == -1
    measurement["outward_sign"] = -1
    passed, why = eb.CONDITIONS["audit_corroboration"](measurement)
    assert passed is True, why
    measurement["outward_sign"] = 1
    passed, why = eb.CONDITIONS["audit_corroboration"](measurement)
    assert passed is False
    assert "CLOSED a periodic orbit" in why


# --------------------------------------------------------------------------
# The real measurement, on real dynamics: opt-in, because it is a ~90 s probe set
#
# Everything the resolver measures for this class except the accelerated
# cross-check is SciPy, so the probes themselves can be run without JAX or
# Diffrax.  This is the test that decides the science rather than the plumbing:
# it walks outward along the real minus_one component 0 arm -- the arm whose
# terminus blocks the release -- and asks whether a periodic orbit still closes
# out there.
# --------------------------------------------------------------------------
SUPPLEMENTAL = ROOT / "research/evidence/V1_SUPPLEMENTAL_EVENT_SIGN_ROOTS_2026-08-16.json"


@pytest.mark.skipif(
    not os.environ.get("ATLAS_EXISTENCE_PROBE_TESTS"),
    reason="set ATLAS_EXISTENCE_PROBE_TESTS=1 to run the ~90 s existence probe set",
)
def test_the_real_component_0_arm_is_not_at_an_existence_boundary() -> None:
    """The periodic family is still there where the blocking walk stopped.

    Seeds are chosen exactly as scripts/resolve_sampled_sweep_endpoints.py chooses
    them -- walk inward until a lattice point re-certifies against the frozen
    gates, then take a direction-only neighbour for the first secant -- and
    oriented OUTWARD, down the arm in m2, which is the direction the blocking walk
    was travelling when its corrector failed.

    Then the measurement runs for real.  On this arm the orbit closes at every
    probe: at the nearest predictor, and 2.5e-3, 5e-3 and 1e-2 further out, the
    last of which lands within 1e-4 of (0.891661, 0.751659) -- the exact point
    where CI run 32146630216 reported "event=inf" and gave up.  Closure there is
    of order 1e-10, five orders INSIDE the 1e-7 gate.  So that walk ran out of
    corrector, not out of family, and the class is refused on four independent
    conditions.

    If a future change makes this pass, something real has changed and it must be
    explained; the correct response is never to relax a condition.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    resolver = pytest.importorskip("resolve_sampled_sweep_endpoints")
    cont = pytest.importorskip("trace_label_invisible_continuous")

    rows = [
        row
        for row in json.loads(SUPPLEMENTAL.read_text(encoding="utf-8"))["roots"]
        if int(row.get("sweep_component", -1)) == 0
    ]
    rows.sort(key=lambda row: (float(row["masses"][0]), float(row["masses"][1])))
    certified = None
    certified_row = None
    refused = []
    for row in rows:
        try:
            certified = resolver.strict_supplemental(row)
        except (RuntimeError, ValueError, FloatingPointError) as exc:
            refused.append(str(exc))
            continue
        certified_row = row
        break
    assert certified is not None, refused
    # A neighbour further UP the arm orients the predictor outward, down the arm.
    neighbour = next(
        row
        for row in rows
        if float(row["masses"][0]) > float(certified_row["masses"][0])
    )
    previous = resolver._direction_only_seed(certified, neighbour)

    ladder = []
    step = 2.5e-3
    trial = step
    while trial >= resolver.MINIMUM_RETRY_STEP:
        ladder.append({"requested_step": float(trial), "closed": False, "error": "scripted"})
        trial *= 0.5
    for _ in range(resolver.EXISTENCE_EXTRA_REFINEMENTS):
        ladder.append(
            {
                "requested_step": float(trial),
                "closed": False,
                "error": "scripted",
                "below_retry_floor": True,
            }
        )
        trial *= 0.5

    seed = [float(value) for value in list(certified.masses2)[:2]]
    conditions, measurements = resolver._existence_conditions(
        previous.localized,
        certified.localized,
        failing_step=step,
        ladder=ladder,
        accepted=[cont._serialize_localized(certified.localized)],
        # One step inward of the certified seed, so the walk has a travelled
        # distance for the direction condition to measure.
        seed_masses=[seed[0] + 0.0021, seed[1] + 0.0088],
        audits=eb.load_audits([AUDIT_35LINE, AUDIT_FULLDOMAIN]),
    )
    assert measurements.get("tangent_error") is None
    assert measurements["mass_tangent"][1] < 0.0, "the probe must head outward in m2"

    probes = measurements["closure_probes"]
    assert probes["nearest"]["closed"] is True
    assert probes["nearest"]["closure"] < eb.CLOSURE_GATE
    assert probes["persistence_4x"]["closed"] is True
    assert probes["persistence_4x"]["masses"][1] < 0.7525
    assert conditions["precision_invariance"]["tightened_float64"]["closed"] is True

    verdict = eb.evaluate(conditions)
    assert verdict["awarded"] is False
    assert "divergence" in verdict["refused_conditions"]
    assert "outward_persistence" in verdict["refused_conditions"]
    assert "audit_corroboration" in verdict["refused_conditions"]
    # The conditions that describe the WALK all pass, which is the point: this is
    # a legitimate outward continuation on a certified branch, and it is refused
    # purely because the family it is following has not ended.
    assert verdict["conditions"]["certified_departure"]["passed"] is True
    assert verdict["conditions"]["outward_direction"]["passed"] is True
    assert verdict["conditions"]["step_refinement_invariance"]["passed"] is True
