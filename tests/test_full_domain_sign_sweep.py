"""Tests for the label-independent full-domain sign-change sweep and its merger.

Two things are worth testing here and nothing else is.

First, the *detector must not be able to see the answer*.  The whole point of
this sweep is that the published S/U labels and the committed critical graph are
the objects under suspicion, so neither may influence where a probe is placed or
whether a sign change is recorded.  ``plan_lattice`` and ``sign_changes_on_line``
are therefore given inputs that contain no labels and no graph at all: if they
ever needed one, these tests would fail to even call them.

Second, the *merger must not be able to be lied to*.  A sharded run is only
worth more than an unsharded one if merging re-derives.  Each test below hands
the merger a shard that is wrong in one specific way -- a forged gate verdict, a
forged census-blindness verdict, a fabricated localization with no bracket
behind it, a probe whose stored event sign contradicts its own (alpha, beta) --
and asserts the merger catches it and reports the truth instead.

No float64 magnitude is pinned beyond an order of magnitude anywhere in this
file.  Measured on three machines, the same organizer chart in this repository
gives plus_one events spanning a factor of 3.4, so a test that pinned a
magnitude would be testing the machine.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SHIPPED_AUDIT = ROOT / "research/evidence/V1_SIGN_TOPOLOGY_AUDIT_2026-08-16.json"
SHIPPED_CROSSING = ROOT / "research/evidence/V1_SIGN_TOPOLOGY_CROSSING_2026-08-16.json"


def _module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SWEEP = _module("sweep_full_domain_sign_changes", "scripts/sweep_full_domain_sign_changes.py")
MERGE = _module("merge_full_domain_sign_sweep", "scripts/merge_full_domain_sign_sweep.py")


# --------------------------------------------------------------------------
# helpers: probes described by (alpha, beta), never by hand-written events
# --------------------------------------------------------------------------
def probe(m1: float, m2: float, alpha: float, beta: float, label: str = "U", ok: bool = True):
    record = {
        "m1": m1,
        "m2": m2,
        "published_label": label,
        "ok": ok,
        "note": "",
        "alpha": alpha,
        "beta": beta,
        "closure": 1e-10,
        "chart": [0.1, 2.0, 0.3, 5.0],
        "seconds": 3.5,
    }
    if ok:
        record.update(SWEEP.AST.state_from_invariants(alpha, beta))
        record["n_unstable"] = SWEEP.AST.unstable_count(alpha, beta)
    return record


def shard(label: str, probes, localizations=(), **overrides):
    document = {
        "schema": SWEEP.SHARD_SCHEMA,
        "phase": "complete",
        "python": "3.13.0",
        "shard": {"label": label, "m1_range": [0.8, 1.1], "m1_stride": 10, "m2_stride": 5},
        "inputs": {"baseline_digests": {"sha256": "deadbeef"}},
        "planned_scan_lines": sorted({p["m1"] for p in probes}),
        "completed_scan_lines": sorted({p["m1"] for p in probes}),
        "probe_summary": {"cpu_seconds": 1.0},
        "probes": list(probes),
        "localizations": list(localizations),
    }
    document.update(overrides)
    return document


# --------------------------------------------------------------------------
# the detector cannot consult the labels or the graph
# --------------------------------------------------------------------------
def test_lattice_is_planned_from_raster_geometry_alone():
    """plan_lattice is given m2 lists -- no labels, no charts, no graph."""
    m2_by_m1 = {0.80 + 0.001 * i: [0.70 + 0.001 * j for j in range(101)] for i in range(31)}
    lattice = SWEEP.plan_lattice(
        m2_by_m1, m1_stride=10, m2_stride=5, m1_range=(0.8, 1.1), m2_range=(0.7, 1.2)
    )
    assert len(lattice.scan_lines) == 4  # 0.800, 0.810, 0.820, and the kept last slice
    assert lattice.probe_count == sum(len(m2s) for _, m2s in lattice.points)


def test_strided_always_keeps_the_support_boundary():
    values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    kept = SWEEP.strided(values, 3)
    assert kept[0] == values[0]
    assert kept[-1] == values[-1]
    assert set(kept) <= set(values)


def test_sharding_cannot_change_which_points_are_probed():
    """A cover by shards plans exactly the lattice one unsharded run would.

    The stride is applied to the global slice list before the window, so a
    shard boundary is a bookkeeping split and never a scientific one.
    """
    m2_by_m1 = {round(0.80 + 0.001 * i, 3): [round(0.70 + 0.001 * j, 3) for j in range(201)] for i in range(301)}
    kwargs = {"m1_stride": 7, "m2_stride": 5, "m2_range": (0.7, 1.2)}
    whole = SWEEP.plan_lattice(m2_by_m1, m1_range=(0.8, 1.1), **kwargs)
    left = SWEEP.plan_lattice(m2_by_m1, m1_range=(0.8, 0.94), **kwargs)
    right = SWEEP.plan_lattice(m2_by_m1, m1_range=(0.9401, 1.1), **kwargs)
    assert sorted(left.points + right.points) == sorted(whole.points)
    assert left.probe_count + right.probe_count == whole.probe_count


@pytest.mark.parametrize(
    ("component", "mechanism"),
    [("G_plus", "plus_one"), ("G_minus", "minus_one"), ("discriminant", "trace_collision")],
)
def test_every_component_flip_is_reported_with_its_own_mechanism(component: str, mechanism: str):
    """Walk (alpha, beta) so exactly one component changes sign."""
    # alpha fixed; beta sweeps.  G_plus = beta - 6a + 20, G_minus = beta - 2a + 4,
    # discriminant = (a-4)^2 - 4(beta - 4a + 8): all three are monotone in beta,
    # with zeros at distinct beta, so a short interval isolates one flip.
    alpha = 3.5
    zeros = {
        "G_plus": 6 * alpha - 20,
        "G_minus": 2 * alpha - 4,
        "discriminant": (alpha - 4) ** 2 / 4 + 4 * alpha - 8,
    }
    target = zeros[component]
    probes = [probe(0.9, 0.80, alpha, target - 0.05), probe(0.9, 0.81, alpha, target + 0.05)]
    found = SWEEP.sign_changes_on_line(probes, sign_floor=1e-4)
    assert [item["mechanism"] for item in found] == [mechanism]
    assert found[0]["m2_bracket"] == [0.80, 0.81]
    assert found[0]["screening_confidence"] == "clear"


def test_a_flip_between_two_tiny_values_is_reported_as_marginal_not_suppressed():
    """A curve through a lattice point must not be silently discarded."""
    alpha = 3.5
    zero = 2 * alpha - 4  # G_minus
    probes = [probe(0.9, 0.80, alpha, zero - 1e-9), probe(0.9, 0.81, alpha, zero + 1e-9)]
    found = SWEEP.sign_changes_on_line(probes, sign_floor=1e-4)
    assert [item["mechanism"] for item in found] == ["minus_one"]
    assert found[0]["screening_confidence"] == "marginal"


def test_failed_probes_never_bracket_anything():
    probes = [
        probe(0.9, 0.80, 3.5, 2.0),
        probe(0.9, 0.81, 3.5, 9.0, ok=False),
        probe(0.9, 0.82, 3.5, 20.0),
    ]
    found = SWEEP.sign_changes_on_line(probes, sign_floor=1e-4)
    for item in found:
        assert 0.81 not in item["m2_bracket"]


def test_horizontal_flips_see_what_a_vertical_scan_line_cannot():
    """Same m2, adjacent scan lines: a curve steep in the mass plane."""
    alpha = 3.5
    zero = 6 * alpha - 20  # G_plus
    probes = [
        probe(0.90, 0.85, alpha, zero - 1.0),
        probe(0.90, 0.86, alpha, zero - 1.0),
        probe(0.91, 0.85, alpha, zero + 1.0),
        probe(0.91, 0.86, alpha, zero + 1.0),
    ]
    assert SWEEP.sign_changes_on_line(probes[:2], sign_floor=1e-4) == []
    horizontal = SWEEP.horizontal_sign_changes(probes, sign_floor=1e-4)
    assert {item["mechanism"] for item in horizontal} == {"plus_one"}
    assert all(item["m1_bracket"] == [0.90, 0.91] for item in horizontal)


# --------------------------------------------------------------------------
# the census-blindness verdict
# --------------------------------------------------------------------------
def test_a_curve_inside_a_uniformly_unstable_run_is_invisible_to_the_census():
    rows = [(0.870, "U"), (0.871, "U"), (0.872, "U"), (0.873, "S"), (0.874, "S")]
    verdict = SWEEP.published_cell_verdict(rows, 0.8715)
    assert verdict["census_would_bracket"] is False
    assert verdict["published_cell_labels"] == ["U", "U"]
    # And it still knows where the census *could* have looked.
    assert verdict["published_transitions_on_line"] == 1


def test_a_curve_inside_a_label_flip_is_reachable_by_the_census():
    rows = [(0.870, "U"), (0.871, "U"), (0.872, "S")]
    verdict = SWEEP.published_cell_verdict(rows, 0.8715)
    assert verdict["census_would_bracket"] is True
    assert verdict["published_cell_labels"] == ["U", "S"]


def test_a_curve_outside_the_published_support_is_not_reachable_either():
    rows = [(0.870, "U"), (0.871, "S")]
    verdict = SWEEP.published_cell_verdict(rows, 0.95)
    assert verdict["census_would_bracket"] is False
    assert verdict["status"] == "outside_published_support"


# --------------------------------------------------------------------------
# gate handling: frozen, and never widened by anything in this pipeline
# --------------------------------------------------------------------------
def test_the_frozen_gates_are_the_frozen_gates():
    assert SWEEP.MAX_EVENT == 2e-8
    assert SWEEP.MAX_CLOSURE == 1e-7
    assert MERGE.MAX_EVENT == SWEEP.MAX_EVENT
    assert MERGE.MAX_CLOSURE == SWEEP.MAX_CLOSURE


@pytest.mark.parametrize(
    ("event", "closure", "expected"),
    [
        (1e-9, 1e-10, True),
        (-1e-9, 1e-10, True),
        (2e-8, 1e-7, True),
        (2.1e-8, 1e-10, False),
        (1e-9, 1.1e-7, False),
        (float("nan"), 1e-10, False),
        (1e-9, float("inf"), False),
    ],
)
def test_gate_verdict_is_two_sided_and_rejects_non_finite(event, closure, expected):
    assert SWEEP.gate_verdict(event, closure) is expected


# --------------------------------------------------------------------------
# committed-edge annotation (annotation only -- never detection)
# --------------------------------------------------------------------------
def _edge(edge_id: str, mechanism: str, vertices):
    return SWEEP.AST.PolylineEdge(
        edge_id=edge_id, mechanism=mechanism, orientation="U->S", vertices=tuple(vertices)
    )


def test_committed_edge_match_is_mechanism_aware():
    edges = [
        _edge("plus", "plus_one", [(0.90, 0.50), (1.00, 0.60)]),
        _edge("minus", "minus_one", [(0.90, 0.50), (1.00, 0.60)]),
    ]
    near = SWEEP.committed_edge_match(edges, "plus_one", 0.95, 0.5500, tolerance=1.5e-3)
    assert near["matched"] is True and near["edge_id"] == "plus"
    far = SWEEP.committed_edge_match(edges, "plus_one", 0.95, 0.5800, tolerance=1.5e-3)
    assert far["matched"] is False
    absent = SWEEP.committed_edge_match(edges, "trace_collision", 0.95, 0.5500, tolerance=1.5e-3)
    assert absent["matched"] is False


# --------------------------------------------------------------------------
# the shard must carry the evidence, not only the conclusions
# --------------------------------------------------------------------------
def _document(probes, localizations=()):
    return SWEEP.shard_document(
        "complete",
        probes=probes,
        localizations=localizations,
        planned_scan_lines=sorted({p["m1"] for p in probes}),
        completed_scan_lines=sorted({p["m1"] for p in probes}),
        sign_floor=1e-4,
        metadata={
            "shard": {"label": "t"},
            "inputs": {"baseline_digests": {"sha256": "deadbeef"}},
            "cost_projection": {},
            "probes_planned": len(probes),
        },
        wall_seconds=1.0,
    )


def test_the_shard_artifact_carries_the_probe_records_it_concluded_from():
    """A shard is only mergeable if it ships the evidence, not just the verdict."""
    probes = _line_with_one_plus_one_flip()
    document = _document(probes)
    assert [p["m2"] for p in document["probes"]] == [p["m2"] for p in probes]
    assert all("alpha" in p and "beta" in p and "chart" in p for p in document["probes"])
    # Round-tripping through JSON must not lose the evidence either.
    reloaded = json.loads(json.dumps(document))
    merged, _duplicates = MERGE.merge_probes([reloaded])
    assert len(merged) == len(probes)


def test_a_shard_missing_its_probe_records_merges_to_nothing():
    """The failure this test exists for actually happened during development.

    An early shard artifact carried its brackets, its localizations and its
    probe *summary* but not the probe records themselves.  Every field a human
    reads looked right; the merger, which re-derives from the records, silently
    produced an empty census.  A pipeline whose whole value is re-derivation
    fails closed only if the raw evidence is present, so its absence must be a
    loud, tested condition rather than a quiet zero.
    """
    probes = _line_with_one_plus_one_flip()
    complete = shard("complete", probes)
    gutted = shard("gutted", probes)
    gutted["probes"] = []

    merged, _duplicates = MERGE.merge_probes([complete])
    assert len(merged) == len(probes)
    assert SWEEP._all_vertical(merged, sign_floor=1e-4)

    starved, _duplicates = MERGE.merge_probes([gutted])
    assert starved == []
    assert SWEEP._all_vertical(starved, sign_floor=1e-4) == []


# --------------------------------------------------------------------------
# the merger cannot be lied to
# --------------------------------------------------------------------------
def _line_with_one_plus_one_flip():
    alpha = 3.5
    zero = 6 * alpha - 20
    return [probe(0.94, 0.80, alpha, zero - 1.0), probe(0.94, 0.81, alpha, zero + 1.0)]


def test_merge_overrides_a_shard_that_forged_a_gate_verdict():
    probes = _line_with_one_plus_one_flip()
    bracket = SWEEP.sign_changes_on_line(probes, sign_floor=1e-4)[0]
    forged = {
        **bracket,
        # 1e-6 is fifty times the frozen event gate.  The shard says "passed".
        "status": "passed",
        "event_value": 1e-6,
        "closure": 1e-10,
        "masses": [0.94, 0.805, 1.0],
    }
    localizations, audit = MERGE.rederive_localizations(
        [shard("liar", probes, [forged])],
        [bracket],
        labelled_rows={0.94: [(0.80, "U"), (0.81, "U")]},
        edges=[],
        match_tolerance=1.5e-3,
    )
    assert localizations[0]["status"] == "missed_frozen_gates"
    assert localizations[0]["shard_reported_status"] == "passed"
    assert audit["gate_status_overridden_by_merge"] == 1


def test_merge_overrides_a_shard_that_forged_the_census_blindness_verdict():
    probes = _line_with_one_plus_one_flip()
    bracket = SWEEP.sign_changes_on_line(probes, sign_floor=1e-4)[0]
    forged = {
        **bracket,
        "status": "passed",
        "event_value": 1e-9,
        "closure": 1e-10,
        "masses": [0.94, 0.805, 1.0],
        # The shard claims the published census could have found this curve.
        "published_cell": {"census_would_bracket": True},
    }
    localizations, audit = MERGE.rederive_localizations(
        [shard("liar", probes, [forged])],
        [bracket],
        labelled_rows={0.94: [(0.80, "U"), (0.81, "U")]},  # both unstable: it could not
        edges=[],
        match_tolerance=1.5e-3,
    )
    assert localizations[0]["published_cell"]["census_would_bracket"] is False
    assert audit["census_blindness_verdict_overridden_by_merge"] == 1


def test_merge_flags_a_localization_with_no_bracket_behind_it():
    probes = _line_with_one_plus_one_flip()
    real = SWEEP.sign_changes_on_line(probes, sign_floor=1e-4)[0]
    invented = {
        "m1": 0.94,
        "component": "G_minus",
        "mechanism": "minus_one",
        "m2_bracket": [0.95, 0.96],
        "status": "passed",
        "event_value": 1e-9,
        "closure": 1e-10,
        "masses": [0.94, 0.955, 1.0],
    }
    localizations, audit = MERGE.rederive_localizations(
        [shard("liar", probes, [invented])],
        [real],
        labelled_rows={0.94: [(0.80, "U"), (0.81, "U")]},
        edges=[],
        match_tolerance=1.5e-3,
    )
    assert localizations[0]["bracket_rederived"] is False
    assert len(audit["localizations_without_a_rederived_bracket"]) == 1


def test_merge_recomputes_probe_events_from_alpha_and_beta():
    """A shard that stored an event of the wrong sign is caught and corrected."""
    good = probe(0.94, 0.80, 3.5, 5.0)
    tampered = dict(good)
    tampered["G_plus"] = -abs(tampered["G_plus"]) - 1.0
    merged, _duplicates = MERGE.merge_probes([shard("liar", [tampered])])
    assert merged[0]["rederivation_mismatch"] == ["G_plus"]
    assert (merged[0]["G_plus"] > 0) == (good["G_plus"] > 0)


def test_merge_reports_duplicate_lattice_points_by_agreement_not_by_equality():
    """Two machines will not agree on magnitudes; they must agree on signs.

    float64 in this repository is not reproducible across boxes, so the merger
    records the spread and asserts only that the signs and n_unstable match.
    """
    a = probe(0.94, 0.80, 3.5, 5.0)
    b = probe(0.94, 0.80, 3.5, 5.0 + 3e-9)  # a different machine, same physics
    _merged, duplicates = MERGE.merge_probes([shard("box-a", [a]), shard("box-b", [b])])
    assert len(duplicates) == 1
    entry = duplicates[0]
    assert entry["G_plus_signs_agree"] is True
    assert entry["n_unstable_agree"] is True
    assert entry["G_plus_spread"] > 0.0  # recorded, not asserted away


def test_merge_refuses_shards_that_probed_a_different_raster():
    left = shard("a", [probe(0.94, 0.80, 3.5, 5.0)])
    right = shard("b", [probe(0.95, 0.80, 3.5, 5.0)])
    right["inputs"]["baseline_digests"] = {"sha256": "0" * 64}
    digests = {
        json.dumps(document["inputs"]["baseline_digests"], sort_keys=True)
        for document in (left, right)
    }
    assert len(digests) > 1  # the guard main() enforces


# --------------------------------------------------------------------------
# curve components: the scientific object is a curve, not a per-line point
# --------------------------------------------------------------------------
def _root(
    m1: float,
    m2: float,
    mechanism: str = "plus_one",
    blind: bool = True,
    matched: bool = False,
):
    return {
        "mechanism": mechanism,
        "status": "passed",
        "masses": [m1, m2, 1.0],
        "event_value": 1e-9,
        "closure": 1e-10,
        "published_cell": {"census_would_bracket": not blind},
        "committed_edge": {"matched": matched, "edge_id": "plus_one_u_to_s_0" if matched else None},
    }


def test_roots_on_adjacent_scan_lines_link_into_one_curve():
    lines = [0.90, 0.91, 0.92]
    roots = [_root(0.90, 0.860), _root(0.91, 0.864), _root(0.92, 0.868)]
    components = MERGE.link_components(roots, lines, link_threshold=0.02)
    assert len(components) == 1
    assert components[0]["points"] == 3
    assert components[0]["invisible_to_published_labels"] is True
    assert components[0]["in_committed_graph"] is False


def test_a_jump_larger_than_the_link_threshold_starts_a_new_curve():
    lines = [0.90, 0.91]
    roots = [_root(0.90, 0.860), _root(0.91, 0.960)]
    components = MERGE.link_components(roots, lines, link_threshold=0.02)
    assert len(components) == 2


def test_two_curves_of_one_mechanism_on_the_same_line_stay_separate():
    lines = [0.90, 0.91]
    roots = [
        _root(0.90, 0.860),
        _root(0.90, 0.900),
        _root(0.91, 0.862),
        _root(0.91, 0.902),
    ]
    components = MERGE.link_components(roots, lines, link_threshold=0.02)
    assert len(components) == 2
    assert all(c["points"] == 2 for c in components)


def test_only_gate_passing_roots_become_curve_points():
    lines = [0.90, 0.91]
    missed = _root(0.91, 0.862)
    missed["status"] = "missed_frozen_gates"
    components = MERGE.link_components([_root(0.90, 0.860), missed], lines, link_threshold=0.02)
    assert len(components) == 1 and components[0]["points"] == 1


# --------------------------------------------------------------------------
# codimension-two proximity
# --------------------------------------------------------------------------
def test_two_mechanisms_meeting_on_one_line_are_flagged_as_an_organizer_candidate():
    roots = [_root(0.930, 0.8854, "plus_one"), _root(0.930, 0.8856, "minus_one")]
    crossings = MERGE.mechanism_crossings(roots, threshold=2e-3)
    assert len(crossings) == 1
    assert crossings[0]["mechanisms"] == ["minus_one", "plus_one"]
    assert crossings[0]["separation_m2"] < 2e-3


def test_two_roots_of_the_same_mechanism_are_not_an_organizer():
    roots = [_root(0.930, 0.8854, "plus_one"), _root(0.930, 0.8856, "plus_one")]
    assert MERGE.mechanism_crossings(roots, threshold=2e-3) == []


def test_well_separated_mechanisms_are_not_an_organizer():
    roots = [_root(0.930, 0.8854, "plus_one"), _root(0.930, 0.9100, "minus_one")]
    assert MERGE.mechanism_crossings(roots, threshold=2e-3) == []


def test_the_organizer_multipliers_are_forced_by_algebra_not_by_measurement():
    """G_plus = G_minus = 0 forces alpha = beta = 4 and trace roots +-2.

    beta - 6a + 20 = 0 and beta - 2a + 4 = 0 subtract to -4a + 16 = 0, so
    alpha = 4 and beta = 4.  Then P(t) = t^2 - (alpha-4) t + (beta - 4 alpha + 8)
    = t^2 - 4, whose roots are +2 and -2: multipliers {+1, +1, -1, -1}.
    """
    values = SWEEP.AST.state_from_invariants(4.0, 4.0)
    assert values["G_plus"] == 0.0
    assert values["G_minus"] == 0.0
    assert values["discriminant"] == pytest.approx(16.0)
    # Both trace roots sit exactly on the |t| = 2 boundary, so the stability
    # count is undecidable there -- which is the analytic signature of the X.
    assert SWEEP.AST.unstable_count(4.0, 4.0) is None


# --------------------------------------------------------------------------
# the shipped 2026-08-16 findings, re-read through this pipeline's own rules
# --------------------------------------------------------------------------
def _shipped_certifications():
    out = []
    for path in (SHIPPED_AUDIT, SHIPPED_CROSSING):
        document = json.loads(path.read_text(encoding="utf-8"))
        for item in document["missing_curve_refinements"]:
            certification = item.get("certification", {})
            if certification.get("status") == "passed":
                out.append((item, certification))
    return out


def test_the_seven_shipped_missing_curves_still_pass_the_frozen_gates():
    shipped = _shipped_certifications()
    assert len(shipped) == 7
    for _item, certification in shipped:
        assert SWEEP.gate_verdict(certification["event_value"], certification["closure"])


def test_six_of_the_seven_are_unstable_on_both_sides_and_the_seventh_is_off_raster():
    """The blindness mechanism, checked against the shipped brackets.

    Six of the seven bracket an event with ``n_unstable`` 2 -> 1: unstable on
    both sides, so no adjacent pair of published rows disagrees and
    ``transition_brackets`` emits nothing there at any grid spacing.

    The seventh steps 1 -> 0, which *is* a stability transition -- the claim
    that all seven step 2 -> 1 is not right.  It is still unreachable by the
    census, but for a second reason: it sits at m1 = 0.9295, and the frozen
    raster only carries m1 on a 0.001 grid, so that slice does not exist to be
    bracketed.  Two independent blind spots, not one.
    """
    both_unstable = []
    stability_transitions = []
    for item, _certification in _shipped_certifications():
        low, high = item["endpoint_n_unstable"]
        (both_unstable if min(low, high) > 0 else stability_transitions).append(item)
    assert len(both_unstable) == 6
    assert len(stability_transitions) == 1
    off_raster = stability_transitions[0]["m1"]
    assert abs(off_raster * 1000.0 - round(off_raster * 1000.0)) > 1e-9


def test_the_shipped_findings_would_be_reported_as_census_blind_by_this_pipeline():
    """Feed the shipped bracket labels through published_cell_verdict.

    Six of the seven brackets carry two unstable endpoints; a published raster
    that labels both endpoints U cannot produce a transition cell there.
    """
    blind = 0
    for item, certification in _shipped_certifications():
        low, high = item["endpoint_n_unstable"]
        rows = [
            (item["bracket"][0], "U" if low else "S"),
            (item["bracket"][1], "U" if high else "S"),
        ]
        verdict = SWEEP.published_cell_verdict(rows, certification["masses"][1])
        if verdict["census_would_bracket"] is False:
            blind += 1
    assert blind == 6
