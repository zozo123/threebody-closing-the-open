"""Pin the 2e-8 event-gate margin of the frozen 620-cell census.

The census reports ``max |event| = 1.9897980818583960e-08`` against a frozen
``2e-8`` gate.  That is 99.489% of the gate: the whole completeness claim has
0.511% of headroom.  These tests exist so nobody can move that number -- in
either direction -- without saying so out loud.

They deliberately assert exact equality on values read out of a frozen evidence
artifact.  If a future run genuinely improves (or degrades) the census, the fix
is to update these constants in the same commit that explains why, not to widen
the assertion.  Nothing here loosens a gate; the gate itself is asserted to be
exactly 2e-8 everywhere it appears.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROOTS = ROOT / "research/evidence/V1_HYBRID_CRITICAL_ROOTS_2026-08-15.json"
MARGIN_AUDIT = ROOT / "research/evidence/V1_EVENT_MARGIN_AUDIT_2026-08-16.json"

EVENT_GATE = 2e-8
CLOSURE_GATE = 1e-7

# Measured on the frozen artifact.  Do not relax; update deliberately.
PINNED_MAX_ABS_EVENT = 1.989798081858396e-08
PINNED_GATE_OCCUPANCY = 0.994899040929198
PINNED_ROOTS_ABOVE_1E_8 = 165
PINNED_ROOTS_ABOVE_95_PERCENT_OF_GATE = 7
PINNED_LOCALIZED_ROOTS = 620


def _roots() -> list[dict]:
    return json.loads(ROOTS.read_text(encoding="utf-8"))["roots"]


def test_event_gate_is_exactly_two_e_minus_eight_everywhere() -> None:
    payload = json.loads(ROOTS.read_text(encoding="utf-8"))
    assert payload["frozen_gates"]["event"] == EVENT_GATE
    assert payload["frozen_gates"]["closure"] == CLOSURE_GATE

    merge_source = (ROOT / "scripts/merge_hybrid_critical_roots.py").read_text()
    assert "EVENT_GATE = 2e-8" in merge_source
    assert "CLOSURE_GATE = 1e-7" in merge_source

    localize_source = (ROOT / "scripts/localize_full_critical_network.py").read_text()
    assert 'refusing to loosen the 2e-8 event gate' in localize_source
    assert 'refusing to loosen the 1e-7 closure gate' in localize_source


def test_frozen_census_margin_is_pinned() -> None:
    """The census passes the 2e-8 gate by 0.511%, and that must stay visible."""
    roots = _roots()
    events = sorted(abs(float(root["event"])) for root in roots)
    assert len(events) == PINNED_LOCALIZED_ROOTS
    assert events[-1] == PINNED_MAX_ABS_EVENT
    assert events[-1] <= EVENT_GATE

    occupancy = events[-1] / EVENT_GATE
    assert math.isclose(occupancy, PINNED_GATE_OCCUPANCY, rel_tol=0.0, abs_tol=1e-12)
    # Stated in the direction a reader cares about: there is well under 1% of
    # room left, so this census cannot absorb any integrator or tolerance change.
    assert 1.0 - occupancy < 0.006

    assert sum(1 for value in events if value > 1e-8) == PINNED_ROOTS_ABOVE_1E_8
    assert (
        sum(1 for value in events if value > 0.95 * EVENT_GATE)
        == PINNED_ROOTS_ABOVE_95_PERCENT_OF_GATE
    )


def test_frozen_census_distribution_is_pinned() -> None:
    """Median 3.3e-9 with a tail piled against the gate is the shape to preserve."""
    events = sorted(abs(float(root["event"])) for root in _roots())

    def quantile(q: float) -> float:
        pos = q * (len(events) - 1)
        lo = int(math.floor(pos))
        hi = min(lo + 1, len(events) - 1)
        return events[lo] * (1.0 - (pos - lo)) + events[hi] * (pos - lo)

    assert math.isclose(quantile(0.50), 3.285798e-09, rel_tol=1e-5)
    assert math.isclose(quantile(0.90), 1.578565e-08, rel_tol=1e-5)
    assert math.isclose(quantile(0.99), 1.904002e-08, rel_tol=1e-5)


def test_worst_offenders_are_float64_and_do_not_cluster_on_one_mechanism() -> None:
    """The tail is a numerical signature, not a physical one.

    All seven roots above 95% of the gate come from the float64 estimator, and
    they span more than one event mode.  If a future change makes the tail
    cluster on one mechanism that is a physics finding and this test should
    fail so somebody looks at it.
    """
    roots = sorted(_roots(), key=lambda r: -abs(float(r["event"])))
    worst = [r for r in roots if abs(float(r["event"])) > 0.95 * EVENT_GATE]
    assert len(worst) == PINNED_ROOTS_ABOVE_95_PERCENT_OF_GATE
    assert {r["estimator"] for r in worst} == {"float64"}
    assert len({r["event_mode"] for r in worst}) > 1


def test_margin_audit_artifact_agrees_with_the_frozen_census() -> None:
    """The audit artifact must be a faithful function of the census, not prose."""
    if not MARGIN_AUDIT.exists():  # pragma: no cover - artifact regenerated in CI
        return
    audit = json.loads(MARGIN_AUDIT.read_text(encoding="utf-8"))
    assert audit["schema"] == "atlas.v1.event-margin-audit/1"
    assert audit["frozen_gates"]["event"] == EVENT_GATE
    assert audit["localized_roots"] == PINNED_LOCALIZED_ROOTS
    distribution = audit["event_distribution"]
    assert distribution["max"] == PINNED_MAX_ABS_EVENT
    assert distribution["above_1e_8"] == PINNED_ROOTS_ABOVE_1E_8
    assert math.isclose(
        distribution["gate_occupancy_max"], PINNED_GATE_OCCUPANCY, abs_tol=1e-12
    )


def test_reevaluation_uncertainty_exceeding_the_gate_stays_disclosed() -> None:
    """The falsification must not be quietly dropped from the artifact.

    Re-evaluating the event functional at the *recorded* state, with the same
    locked dependencies, across the integrator tolerance ladder moves it by more
    than the 2e-8 gate for the probed cells.  A residual that cannot be measured
    to better than the gate is not a certificate, and the artifact has to keep
    saying so.
    """
    if not MARGIN_AUDIT.exists():  # pragma: no cover - artifact regenerated in CI
        return
    audit = json.loads(MARGIN_AUDIT.read_text(encoding="utf-8"))
    reevaluation = audit.get("reevaluation")
    assert reevaluation, "margin audit must carry the re-evaluation probe"
    assert reevaluation["probes"], "re-evaluation probe must not be empty"
    # The honest error bar on the published residual is at least as large as the
    # gate itself.  If this ever drops below 1.0 the census got genuinely more
    # reproducible and the claim in the manuscript must be revisited.
    assert reevaluation["evaluation_uncertainty_over_gate"] > 1.0
    exceeding = reevaluation["cells_exceeding_gate_at_some_tight_tolerance"]
    assert len(exceeding) > 0
    # As frozen: 20 of the 24 probed cells cross the gate at some tight
    # tolerance, and the affected set is not confined to the worst offenders --
    # evenly spaced control cells are in it too.  The four exceptions are cells
    # whose recorded residual is already ~1e-9 or smaller, i.e. cells where the
    # event functional happens to be cleanly evaluable.
    controls = set(reevaluation["probed_control_cells"])
    assert len(exceeding) >= 20
    assert controls & set(exceeding)


def test_cross_platform_discrepancy_stays_on_the_record() -> None:
    """A float64 residual that moves by more than its own gate across CPUs.

    ``reevaluated_at_accepting_tolerance`` is bit-for-bit the computation that
    produced the recorded float64 residual, on the recorded state, with the
    locked numpy/scipy.  Re-running it twice on one machine reproduces every
    digit, so the difference from the frozen value is a cross-architecture
    difference amplified by the ~8-digit cancellation in the event functionals.
    This test keeps that number, and the platform it was measured on, in the
    artifact.
    """
    if not MARGIN_AUDIT.exists():  # pragma: no cover - artifact regenerated in CI
        return
    reevaluation = json.loads(MARGIN_AUDIT.read_text(encoding="utf-8"))["reevaluation"]
    platform_block = reevaluation["reevaluation_platform"]
    for field in ("system", "machine", "python", "numpy", "scipy"):
        assert platform_block[field]
    assert reevaluation["comparable_float64_probes"] > 0
    assert reevaluation["max_cross_platform_discrepancy"] is not None
    assert reevaluation["median_cross_platform_discrepancy"] is not None
    # As frozen: median discrepancy 2.78e-8 (1.39x the gate), max 4.14e-7
    # (20.7x the gate), 10 of 22 comparable probes land above 2e-8.
    assert reevaluation["median_cross_platform_discrepancy"] > EVENT_GATE, (
        "The frozen audit was measured on Darwin/arm64 against a census produced on "
        "Linux CI runners.  If this assertion now fails because the audit was "
        "regenerated on the same architecture that produced the census, that is a "
        "finding, not a flake: record the architecture pair and update the pin."
    )
    assert len(reevaluation["float64_probes_exceeding_gate_on_reevaluation"]) >= 10
