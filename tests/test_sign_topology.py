"""Tests for the sign-vector / planar-arrangement consistency checker.

The point of this file is the demonstrated kill.  A completeness checker that
has never rejected anything is worthless, so the synthetic arrangement below is
built with an exactly known critical set, verified to produce zero violations,
and then deliberately damaged four different ways -- edge deleted, edge
truncated so it dangles in the interior, edge displaced, edge mislabelled --
with an assertion that the checker catches each one and names the right
component.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REAL_GRAPH = ROOT / "research/evidence/V1_CRITICAL_GRAPH.json"
REAL_ROOTS = ROOT / "research/evidence/V1_HYBRID_CRITICAL_ROOTS_2026-08-15.json"
SUPPLEMENTAL_ROOTS = ROOT / "research/evidence/V1_SUPPLEMENTAL_EVENT_SIGN_ROOTS_2026-08-16.json"
SHIPPED_AUDIT = ROOT / "research/evidence/V1_SIGN_TOPOLOGY_AUDIT_2026-08-16.json"
SHIPPED_CROSSING = ROOT / "research/evidence/V1_SIGN_TOPOLOGY_CROSSING_2026-08-16.json"
RERUN_AUDIT = ROOT / "research/evidence/V1_SIGN_TOPOLOGY_AUDIT_2026-08-17.json"
RERUN_CROSSING = ROOT / "research/evidence/V1_SIGN_TOPOLOGY_CROSSING_2026-08-17.json"


def _all_roots() -> list[dict]:
    roots = list(json.loads(REAL_ROOTS.read_text())["roots"])
    if SUPPLEMENTAL_ROOTS.exists():
        roots.extend(json.loads(SUPPLEMENTAL_ROOTS.read_text())["roots"])
    return roots


def _module():
    spec = importlib.util.spec_from_file_location(
        "audit_sign_topology", ROOT / "scripts/audit_sign_topology.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


AST = _module()


# --------------------------------------------------------------------------
# A synthetic arrangement with an exactly known critical set.
#
# Take alpha = 3.5 everywhere and beta = v + 3 with
#     v(m1, m2) = 8 * (m2 - 0.50 - 0.05 * m1).
# Then the three reduced invariants reduce to closed form:
#     G_minus      = v                          zero on m2 = 0.50     + 0.05 m1
#     G_plus       = v + 2                      zero on m2 = 0.25     + 0.05 m1
#     discriminant = 12.25 - 4 v                zero on m2 = 0.8828125 + 0.05 m1
# and n_unstable walks 2 -> 1 -> 0 -> 2 across those three lines.
# --------------------------------------------------------------------------
LINES = {
    "plus_one": (0.25, 0.05),
    "minus_one": (0.50, 0.05),
    "trace_collision": (0.8828125, 0.05),
}
SCAN_LINES = (0.2, 0.4, 0.6, 0.8)
KWARGS = {"threshold": 0.01, "min_clearance": 0.01}


def _synthetic_probe(m1: float, m2: float):
    v = 8.0 * (m2 - 0.50 - 0.05 * m1)
    return AST.make_probe(m1, m2, alpha=3.5, beta=v + 3.0)


def _synthetic_edge(mechanism: str, *, m1_lo: float = 0.0, m1_hi: float = 1.0, shift: float = 0.0):
    intercept, slope = LINES[mechanism]
    vertices = tuple(
        (m1_lo + (m1_hi - m1_lo) * k / 20.0, intercept + shift + slope * (m1_lo + (m1_hi - m1_lo) * k / 20.0))
        for k in range(21)
    )
    return AST.PolylineEdge(
        edge_id=f"synthetic_{mechanism}",
        mechanism=mechanism,
        orientation="U->S",
        vertices=vertices,
    )


def _full_arrangement():
    return [_synthetic_edge(m) for m in LINES]


def _probe_field(edges):
    """Standoff probes either side of every present edge plus face midpoints."""
    out: dict[float, list] = {}
    for m1 in SCAN_LINES:
        m2s = AST.plan_line(edges, m1, m2_lo=0.02, m2_hi=0.99, standoff=0.02)
        out[m1] = [_synthetic_probe(m1, m2) for m2 in m2s]
    return out


def _kinds(report):
    return report["violation_counts"]


# --------------------------------------------------------------------------
# baseline: the checker must not cry wolf
# --------------------------------------------------------------------------
def test_synthetic_arrangement_is_clean():
    edges = _full_arrangement()
    probes = _probe_field(edges)
    assert sum(len(v) for v in probes.values()) >= 20
    report = AST.audit_probes(edges, probes, **KWARGS)
    assert report["violations"] == [], report["violations"]


def test_synthetic_probes_span_every_state():
    edges = _full_arrangement()
    probes = _probe_field(edges)
    states = {p.state(KWARGS["threshold"]) for line in probes.values() for p in line}
    states.discard(None)
    unstable = sorted({s[3] for s in states})
    assert unstable == [0, 1, 2]
    assert len(states) >= 4


# --------------------------------------------------------------------------
# demonstrated kills
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("missing", "component"),
    [
        ("plus_one", "G_plus"),
        ("minus_one", "G_minus"),
        ("trace_collision", "discriminant"),
    ],
)
def test_deleted_edge_is_caught(missing: str, component: str):
    """The headline kill: an entire critical curve removed from the graph."""
    full = _full_arrangement()
    probes = _probe_field(full)  # probes are placed as if the graph were complete
    damaged = [e for e in full if e.mechanism != missing]

    report = AST.audit_probes(damaged, probes, **KWARGS)
    missing_curves = [v for v in report["violations"] if v["kind"] == "missing_critical_curve"]
    assert missing_curves, f"deleting the {missing} curve went undetected"
    assert {v["component"] for v in missing_curves} == {component}

    intercept, slope = LINES[missing]
    for violation in missing_curves:
        lo, hi = sorted(violation["m2_bracket"])
        truth = intercept + slope * violation["m1"]
        assert lo < truth < hi, "reported bracket does not contain the deleted curve"

    # and the face test independently sees it as a state change across a path
    # that meets no committed edge
    assert _kinds(report).get("face_state_mismatch", 0) > 0


def test_truncated_edge_dangling_in_the_interior_is_caught():
    """The real defect shape: a curve that simply stops mid-domain."""
    full = _full_arrangement()
    probes = _probe_field(full)
    damaged = [
        _synthetic_edge("plus_one", m1_hi=0.5),
        _synthetic_edge("minus_one"),
        _synthetic_edge("trace_collision"),
    ]
    report = AST.audit_probes(damaged, probes, **KWARGS)
    missing_curves = [v for v in report["violations"] if v["kind"] == "missing_critical_curve"]
    assert missing_curves
    assert {v["component"] for v in missing_curves} == {"G_plus"}
    # only the scan lines beyond the truncation may complain
    assert all(v["m1"] > 0.5 for v in missing_curves)


def test_displaced_edge_is_caught():
    """Right mechanism, wrong geometry: the sign flip is not where the edge is."""
    full = _full_arrangement()
    probes = _probe_field(full)
    damaged = [
        _synthetic_edge("plus_one", shift=0.12),
        _synthetic_edge("minus_one"),
        _synthetic_edge("trace_collision"),
    ]
    report = AST.audit_probes(damaged, probes, **KWARGS)
    kinds = _kinds(report)
    assert kinds.get("missing_critical_curve", 0) > 0
    assert kinds.get("no_flip_across_edge", 0) > 0


def test_mislabelled_mechanism_is_caught():
    """Curve in the right place, wrong mechanism label."""
    full = _full_arrangement()
    probes = _probe_field(full)
    intercept, slope = LINES["minus_one"]
    liar = AST.PolylineEdge(
        edge_id="synthetic_liar",
        mechanism="trace_collision",
        orientation="U->S",
        vertices=tuple((k / 20.0, intercept + slope * (k / 20.0)) for k in range(21)),
    )
    damaged = [_synthetic_edge("plus_one"), liar, _synthetic_edge("trace_collision")]
    report = AST.audit_probes(damaged, probes, **KWARGS)
    kinds = _kinds(report)
    assert kinds.get("forbidden_component_flip", 0) > 0
    assert kinds.get("no_flip_across_edge", 0) > 0
    flips = [v for v in report["violations"] if v["kind"] == "forbidden_component_flip"]
    assert {v["component"] for v in flips} == {"G_minus"}


# --------------------------------------------------------------------------
# the checker must stay silent when the data cannot decide
# --------------------------------------------------------------------------
def test_undecidable_components_do_not_produce_violations():
    edges = _full_arrangement()
    probes = _probe_field(edges)
    damaged = [e for e in edges if e.mechanism != "plus_one"]
    # a threshold larger than every synthetic magnitude makes every sign
    # undecidable, so nothing may be reported
    huge = AST.audit_probes(damaged, probes, threshold=1e6, min_clearance=0.01)
    assert huge["violations"] == []


def test_failed_probes_are_excluded():
    edges = _full_arrangement()
    probes = {
        0.2: [AST.Probe(0.2, 0.1, ok=False, note="closure"), _synthetic_probe(0.2, 0.9)],
    }
    report = AST.audit_probes(edges, probes, **KWARGS)
    assert report["violations"] == []


# --------------------------------------------------------------------------
# exact piecewise-linear geometry
# --------------------------------------------------------------------------
def test_horizontal_crossings_count_a_non_monotone_edge_twice():
    vee = AST.PolylineEdge(
        edge_id="vee",
        mechanism="plus_one",
        orientation="U->S",
        vertices=((0.0, 1.0), (0.5, 0.0), (1.0, 1.0)),
    )
    assert vee.horizontal_crossings(0.5, 0.0, 1.0) == 2
    assert vee.horizontal_crossings(0.5, 0.0, 0.4) == 1
    assert vee.horizontal_crossings(1.5, 0.0, 1.0) == 0
    assert vee.horizontal_crossings(0.5, 2.0, 3.0) == 0, "outside the edge's m1 support"


def test_vertical_crossings_and_interpolation():
    edge = AST.PolylineEdge(
        edge_id="line",
        mechanism="minus_one",
        orientation="U->S",
        vertices=((0.0, 0.0), (1.0, 1.0)),
    )
    assert edge.m2_at(0.25) == pytest.approx(0.25)
    assert edge.m2_at(-0.1) is None
    assert edge.vertical_crossings(0.5, 0.0, 1.0) == 1
    assert edge.vertical_crossings(0.5, 0.6, 1.0) == 0
    assert edge.vertical_crossings(2.0, 0.0, 1.0) == 0


def test_lpath_crossings_uses_both_legs():
    edge = AST.PolylineEdge(
        edge_id="flat",
        mechanism="plus_one",
        orientation="U->S",
        vertices=((0.0, 0.5), (1.0, 0.5)),
    )
    below = AST.make_probe(0.1, 0.1, alpha=3.5, beta=3.0)
    above = AST.make_probe(0.9, 0.9, alpha=3.5, beta=3.0)
    met, clearance = AST.lpath_crossings([edge], below, above)
    assert [e.edge_id for e in met] == ["flat"]
    assert clearance == 0.0


# --------------------------------------------------------------------------
# the discrete state itself
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("t1", "t2", "expected"),
    [
        (0.5, -0.5, 0),  # both trace roots inside (-2, 2)
        (3.0, 0.5, 1),
        (3.0, -3.0, 2),
        (-3.0, -0.5, 1),
        (5.0, 4.0, 2),  # both roots beyond +2
    ],
)
def test_unstable_count_from_trace_roots(t1: float, t2: float, expected: int):
    a = t1 + t2
    b = t1 * t2
    alpha = a + 4.0
    beta = b + 4.0 * alpha - 8.0
    assert AST.unstable_count(alpha, beta) == expected


def test_unstable_count_is_undecidable_on_the_critical_set():
    # t1 = 2 exactly: a nontrivial multiplier sits at +1
    a, b = 2.0 + 0.5, 2.0 * 0.5
    alpha = a + 4.0
    beta = b + 4.0 * alpha - 8.0
    assert AST.unstable_count(alpha, beta) is None
    probe = AST.make_probe(0.0, 0.0, alpha, beta)
    assert probe.values["G_plus"] == pytest.approx(0.0, abs=1e-12)
    assert probe.state(1e-6) is None


def test_state_components_match_the_repository_event_definitions():
    """G_plus / G_minus must equal critical_manifold.event_value, not a lookalike."""
    from threebody_atlas.critical_manifold import event_value
    from threebody_atlas.reduced import ReducedFloquetResult

    import numpy as np

    alpha, beta = 5.734240765712390, 14.405444594279624
    disc = (alpha - 4.0) ** 2 - 4.0 * (beta - 4.0 * alpha + 8.0)
    result = ReducedFloquetResult(
        monodromy=np.eye(8),
        multipliers=np.ones(8),
        alpha=alpha,
        beta=beta,
        discriminant=disc,
        trace_roots=(0j, 0j),
        linearly_stable=None,
        stability_margin=0.0,
    )
    values = AST.state_from_invariants(alpha, beta)
    assert values["G_plus"] == pytest.approx(event_value(result, "plus_one"))
    assert values["G_minus"] == pytest.approx(event_value(result, "minus_one"))
    assert values["discriminant"] == pytest.approx(event_value(result, "trace_collision"))


# --------------------------------------------------------------------------
# the committed arrangement, read straight from the shipped evidence
# --------------------------------------------------------------------------
@pytest.mark.skipif(not REAL_GRAPH.exists(), reason="committed graph not present")
def test_committed_edges_are_graphs_over_m1():
    graph = json.loads(REAL_GRAPH.read_text())
    roots = _all_roots()
    assert AST.non_graph_edges(graph, roots) == []
    edges = AST.edges_from_graph(graph, roots)
    assert len(edges) == len(graph["edges"])
    for edge in edges:
        xs = [v[0] for v in edge.vertices]
        assert xs == sorted(xs)
        assert len(set(xs)) == len(xs)


@pytest.mark.skipif(not RERUN_AUDIT.exists() or not RERUN_CROSSING.exists(), reason="13-edge re-audit not present")
def test_thirteen_edge_reaudit_clears_the_release_conjuncts() -> None:
    """The 2026-08-17 re-audits are what the assembler now reads."""
    for path in (RERUN_AUDIT, RERUN_CROSSING):
        audit = json.loads(path.read_text())
        assert audit["schema"] == AST.SCHEMA
        assert audit["parameters"]["max_closure"] == 1e-7
        counts = audit.get("violation_counts") or {}
        assert int(counts.get("missing_critical_curve", 0)) == 0
        assert int(counts.get("forbidden_component_flip", 0)) == 0


@pytest.mark.skipif(not SHIPPED_AUDIT.exists(), reason="audit artifact not present")
def test_shipped_audit_finding_cannot_silently_disappear():
    """Lock in the falsification the shipped run produced.

    The audit found critical curves the graph does not contain.  If a later
    change makes that finding vanish, it must vanish because someone
    regenerated the artifact and looked at it, not because a checker quietly
    stopped firing.
    """
    audit = json.loads(SHIPPED_AUDIT.read_text())
    assert audit["schema"] == AST.SCHEMA
    assert audit["violation_counts"].get("missing_critical_curve", 0) > 0
    missing = [v for v in audit["violations"] if v["kind"] == "missing_critical_curve"]
    assert all(v["component"] in AST.COMPONENTS for v in missing)
    assert all(v["committed_edges_in_bracket"] == [] for v in missing)


@pytest.mark.skipif(not SHIPPED_AUDIT.exists(), reason="audit artifact not present")
def test_shipped_audit_used_the_frozen_gates():
    """No finding here may rest on a relaxed tolerance."""
    audit = json.loads(SHIPPED_AUDIT.read_text())
    assert audit["parameters"]["max_closure"] == 1e-7
    for refinement in audit.get("missing_curve_refinements", []):
        certification = refinement.get("certification")
        if not certification or "gates" not in certification:
            continue
        assert certification["gates"]["maximum_absolute_event"] == 2e-8
        assert certification["gates"]["maximum_periodic_closure"] == 1e-7
        if certification["status"] == "passed":
            assert abs(certification["event_value"]) <= 2e-8
            assert certification["closure"] <= 1e-7


@pytest.mark.skipif(not SHIPPED_CROSSING.exists(), reason="crossing artifact not present")
def test_two_uncommitted_curves_cross_where_the_graph_has_a_dangling_endpoint():
    """The plus_one and minus_one critical curves swap order near m1 = 0.9293.

    At m1 = 0.925 the minus_one zero lies below the plus_one zero; by
    m1 = 0.9295 the order has reversed.  Two continuous curves that swap order
    have crossed, and at the crossing P(+2) = P(-2) = 0 simultaneously -- a
    codimension-two organizer with reduced multipliers {+1, +1, -1, -1}.
    That crossing is mixed_principal_left (m1 = 0.929239).  Reconstructing
    polylines from sample cells alone left a one-step hole at 0.9295; the
    auditor now extends each incident edge to the organizer, so the
    committed plus_one sweep covers the reversal.
    """
    crossing = json.loads(SHIPPED_CROSSING.read_text())
    certified: dict[tuple[float, str], float] = {}
    for refinement in crossing["missing_curve_refinements"]:
        certification = refinement.get("certification", {})
        if certification.get("status") == "passed":
            certified[(refinement["m1"], refinement["event_mode"])] = certification["masses"][1]

    # both branches resolved on the same scan line, and very close together
    plus_at_9295 = certified[(0.9295, "plus_one")]
    minus_at_9295 = certified[(0.9295, "minus_one")]
    assert 0.0 < minus_at_9295 - plus_at_9295 < 1e-3

    # on the earlier scan line the order is the other way round
    minus_at_925 = certified[(0.925, "minus_one")]
    graph = json.loads(REAL_GRAPH.read_text())
    roots = _all_roots()
    edges = AST.edges_from_graph(graph, roots)
    plus_edge = next(e for e in edges if e.edge_id == "plus_one_u_to_s_0")
    plus_at_925 = plus_edge.m2_at(0.925)
    assert plus_at_925 is not None
    assert minus_at_925 < plus_at_925

    # the catalog plus_one arc still stops at the organizer; the event-sign
    # continuation, extended to mixed_principal_left, covers 0.9295
    assert plus_edge.m1_max < 0.9295
    sweep = next(e for e in edges if e.edge_id == "plus_one_sweep_component_10")
    assert sweep.m2_at(0.9295) is not None
    assert abs(sweep.m2_at(0.9295) - plus_at_9295) < 0.002


def test_auditor_extends_polylines_through_mixed_organizers() -> None:
    """A 0.001 hole next to a named mixed vertex is not a missing curve."""
    graph = {
        "edges": [
            {
                "id": "plus_one_left",
                "kind": "mechanism_polyline",
                "mechanism": "plus_one",
                "cell_ids": [0],
                "endpoints": {"start": {}, "end": {"node": "mixed_principal_left"}},
            },
            {
                "id": "plus_one_right",
                "kind": "mechanism_polyline",
                "mechanism": "plus_one",
                "cell_ids": [1],
                "endpoints": {"start": {"node": "mixed_principal_left"}, "end": {}},
            },
        ],
        "nodes": [
            {
                "id": "mixed_principal_left",
                "kind": "mixed_organizer",
                "masses": [0.929239, 0.885366, 1.0],
            }
        ],
    }
    roots = [
        {"cell_id": 0, "masses": [0.929, 0.88508, 1.0]},
        {"cell_id": 1, "masses": [0.930, 0.88627, 1.0]},
    ]
    edges = AST.edges_from_graph(graph, roots)
    right = next(e for e in edges if e.edge_id == "plus_one_right")
    assert right.m1_min == pytest.approx(0.929239)
    assert right.m2_at(0.9295) == pytest.approx(0.885676, abs=1e-4)


@pytest.mark.skipif(not SHIPPED_CROSSING.exists(), reason="crossing artifact not present")
def test_forbidden_component_flip_fired_on_real_data():
    """The mechanism-permission check is not decorative: it caught something."""
    crossing = json.loads(SHIPPED_CROSSING.read_text())
    flips = [v for v in crossing["violations"] if v["kind"] == "forbidden_component_flip"]
    assert flips
    assert all(v["component"] != AST.MECHANISM_COMPONENT[v["edge_mechanism"]] for v in flips)


@pytest.mark.skipif(not REAL_GRAPH.exists(), reason="committed graph not present")
def test_committed_edges_leave_the_declared_domain_uncovered():
    """Catalog polylines still stop short of m1 = 1.1; one sweep edge does not.

    The 620-cell census never reached the m1-max face.  The event-sign sweep
    did: minus_one_sweep_component_3 exits at (1.1, 1.092).  That is a fact
    about the shipped artifact, asserted so neither side can quietly change.
    """
    graph = json.loads(REAL_GRAPH.read_text())
    roots = _all_roots()
    edges = AST.edges_from_graph(graph, roots)
    declared_hi = float(graph["declared_mass_domain"]["m1"][1])
    catalog = [edge for edge in edges if "sweep_component" not in edge.edge_id]
    sweep = [edge for edge in edges if "sweep_component" in edge.edge_id]
    catalog_hi = max(edge.m1_max for edge in catalog)
    assert catalog_hi < declared_hi
    assert declared_hi - catalog_hi > 0.02
    assert sweep
    assert any(abs(edge.m1_max - declared_hi) < 1e-12 for edge in sweep)
