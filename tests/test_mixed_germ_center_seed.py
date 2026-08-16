"""The twelve headline mixed germs, and the centre seed they are launched from.

Two things are pinned here.

1. The CONSUMER ADAPTATION.  ``scripts/trace_canonical_mixed_germs.py`` used to
   read exactly one float64 mixed-vertex candidate key, ``direct_candidate``,
   which only ``scripts/locate_secondary_right_mixed.py`` ever writes.  The
   three headline junction screens write their candidate under
   ``direct_mixed_vertex_retry`` instead, so the headline organizers had no
   centre and no germs.  Both keys are now accepted, in a fixed priority order.

2. The INDEPENDENCE of that centre.  The candidate is produced by a float64
   pipeline that never reads the canonical BigFloat organizer, so the 1e-4
   agreement check inside ``center_points`` is a real cross-pipeline test and
   not a restatement of the seed.  The committed artifacts are checked to still
   have that shape: the recorded agreement is measured, small, and derived from
   a screen whose own provenance hash is pinned.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "research/evidence"
EVENT_GATE = 2e-8
CLOSURE_GATE = 1e-7
GERM_ATTACH_DISTANCE = 0.008

HEADLINE_NODES = {
    "mixed_principal_left": (
        "V1_MIXED_GERMS_PRINCIPAL_LEFT_2026-08-16.json",
        "V1_MIXED_DIRECT_RETRY_PRINCIPAL_LEFT_2026-08-16.json",
        "V1_MIXED_CANONICAL_PRINCIPAL_LEFT_2026-08-15.json",
        "V1_JUNCTION_PRINCIPAL_LEFT_2026-08-15.json",
    ),
    "mixed_secondary_left": (
        "V1_MIXED_GERMS_SECONDARY_LEFT_2026-08-16.json",
        "V1_MIXED_DIRECT_RETRY_SECONDARY_LEFT_2026-08-16.json",
        "V1_MIXED_CANONICAL_SECONDARY_LEFT_2026-08-15.json",
        "V1_JUNCTION_SECONDARY_LEFT_2026-08-15.json",
    ),
    "mixed_principal_right": (
        "V1_MIXED_GERMS_PRINCIPAL_RIGHT_2026-08-16.json",
        "V1_MIXED_DIRECT_RETRY_PRINCIPAL_RIGHT_2026-08-16.json",
        "V1_MIXED_CANONICAL_PRINCIPAL_RIGHT_2026-08-15.json",
        "V1_JUNCTION_PRINCIPAL_RIGHT_2026-08-15.json",
    ),
}


def _load(name: str) -> dict:
    return json.loads((EVIDENCE / name).read_text())


def _sha256(name: str) -> str:
    return hashlib.sha256((EVIDENCE / name).read_bytes()).hexdigest()


def _germ_module():
    """Import the germ tracer.  It pulls in JAX, so callers must skip first."""
    spec = importlib.util.spec_from_file_location(
        "trace_canonical_mixed_germs", ROOT / "scripts/trace_canonical_mixed_germs.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _assembler():
    spec = importlib.util.spec_from_file_location(
        "assemble_critical_graph", ROOT / "scripts/assemble_critical_graph.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _chart(extra: dict | None = None) -> dict:
    body = {"masses": [1.0, 1.0, 1.0], "x1": -0.4, "v1": 1.0, "v2": 0.4, "period": 8.0}
    body.update(extra or {})
    return body


# --------------------------------------------------------------------------
# 1. the consumer adaptation
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def germ_module():
    pytest.importorskip("jax")
    pytest.importorskip("diffrax")
    return _germ_module()


def test_direct_candidate_still_wins_when_both_keys_are_present(germ_module) -> None:
    screen = {
        "direct_candidate": _chart({"success": True, "x1": -0.1}),
        "direct_mixed_vertex_retry": _chart(
            {"status": "accepted_screening_candidate", "x1": -0.2}
        ),
    }
    key, candidate = germ_module.accepted_candidate(screen)
    assert key == "direct_candidate"
    assert candidate["x1"] == -0.1


def test_junction_retry_candidate_is_accepted_as_a_centre_seed(germ_module) -> None:
    screen = {
        "direct_mixed_vertex_retry": _chart({"status": "accepted_screening_candidate"})
    }
    key, candidate = germ_module.accepted_candidate(screen)
    assert key == "direct_mixed_vertex_retry"
    assert candidate["period"] == 8.0


def test_the_recorded_missing_dependency_failure_is_still_rejected(germ_module) -> None:
    """The pre-retry state of two headline screens must not become a centre."""
    screen = {
        "direct_mixed_vertex_retry": {
            "status": "not_accepted",
            "error": "RuntimeError: JAX + Diffrax are required; install the accelerated extra",
        }
    }
    with pytest.raises(SystemExit) as excinfo:
        germ_module.accepted_candidate(screen)
    assert "direct_mixed_vertex_retry" in str(excinfo.value)


def test_a_candidate_missing_chart_fields_is_rejected(germ_module) -> None:
    screen = {
        "direct_mixed_vertex_retry": {
            "status": "accepted_screening_candidate",
            "masses": [1.0, 1.0, 1.0],
            "x1": -0.4,
            "v1": 1.0,
            "v2": 0.4,
        }
    }
    with pytest.raises(SystemExit) as excinfo:
        germ_module.accepted_candidate(screen)
    assert "period" in str(excinfo.value)


def test_a_two_mass_candidate_is_rejected(germ_module) -> None:
    screen = {"direct_candidate": _chart({"success": True, "masses": [1.0, 1.0]})}
    with pytest.raises(SystemExit):
        germ_module.accepted_candidate(screen)


def test_a_screen_with_no_candidate_key_names_both_keys(germ_module) -> None:
    with pytest.raises(SystemExit) as excinfo:
        germ_module.accepted_candidate({"claim_status": "nothing here"})
    message = str(excinfo.value)
    assert "direct_candidate" in message and "direct_mixed_vertex_retry" in message


def test_centre_conditioning_is_ordered_and_never_touches_the_frozen_gates(
    germ_module,
) -> None:
    assert germ_module.CENTER_CLOSURE_SCALES[0] == 1e-6
    assert list(germ_module.CENTER_CLOSURE_SCALES) == sorted(
        germ_module.CENTER_CLOSURE_SCALES
    )
    assert germ_module.EVENT_GATE == EVENT_GATE
    assert germ_module.CLOSURE_GATE == CLOSURE_GATE


def test_solve_direct_vertex_rejects_a_nonpositive_closure_weight() -> None:
    pytest.importorskip("jax")
    pytest.importorskip("diffrax")
    from threebody_atlas.hybrid_vertices import solve_direct_vertex

    with pytest.raises(ValueError, match="closure_scale"):
        solve_direct_vertex(
            [-0.4, 1.0, 0.4, 8.0, 1.0, 1.0],
            "mixed_plus_minus_one",
            mass_bounds=((0.9, 1.1), (0.9, 1.1)),
            closure_scale=0.0,
        )


# --------------------------------------------------------------------------
# 2. the committed artifacts (no accelerated extra required)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("node", sorted(HEADLINE_NODES))
def test_retry_candidate_agrees_with_the_canonical_organizer(node: str) -> None:
    """The independence test, stated as a number.

    The retry candidate is seeded by the junction trace's own closest-to-(4,4)
    sample and bounded by the junction's coarse requested_center.  It never sees
    the BigFloat chart.  Its agreement with that chart is therefore evidence.
    """
    _germs, retry_name, canonical_name, junction_name = HEADLINE_NODES[node]
    retry = _load(retry_name)
    canonical = _load(canonical_name)
    assert retry["mixed_node"] == node
    assert retry["source_junction_sha256"] == _sha256(junction_name)
    assert retry["seed_provenance"] == "junction_trace_closest_to_alpha_beta_four"
    candidate = retry["direct_mixed_vertex_retry"]
    assert candidate["status"] == "accepted_screening_candidate"

    canonical_mass = [float(value) for value in canonical["masses"][:2]]
    seed_mass = [float(value) for value in retry["seed_masses"][:2]]
    candidate_mass = [float(value) for value in candidate["masses"][:2]]
    agreement = math.dist(candidate_mass, canonical_mass)
    seed_gap = math.dist(seed_mass, canonical_mass)
    assert agreement <= 1e-4
    # The solve did not start at the answer.  Its seed is a float64 trace sample
    # at least 1e-5 away in the mass plane, and the refinement closed that gap by
    # an order of magnitude or more.  That is what makes the 1e-4 check an
    # independent-agreement test rather than a restatement of the seed.
    assert seed_gap > 1e-5
    assert agreement < seed_gap / 10.0


@pytest.mark.parametrize("node", sorted(HEADLINE_NODES))
def test_headline_germ_artifact_is_bound_hashed_and_inside_the_frozen_gates(
    node: str,
) -> None:
    germs_name, retry_name, canonical_name, _junction = HEADLINE_NODES[node]
    payload = _load(germs_name)
    assert payload["schema"] == "atlas.v1.mixed-germs/1"
    assert payload["passed"] is True
    assert payload["mixed_node"] == node
    assert payload["center_seed_source"] == "direct_mixed_vertex_retry"
    assert payload["center_seed_independent_of_canonical"] is True
    assert payload["center_shift_from_canonical"] <= 1e-4
    assert payload["frozen_gates"] == {"event": EVENT_GATE, "closure": CLOSURE_GATE}
    assert payload["screen_sha256"] == _sha256(retry_name)
    assert payload["canonical_sha256"] == _sha256(canonical_name)
    assert payload["source_roots_sha256"] == _sha256(
        "V1_HYBRID_CRITICAL_ROOTS_2026-08-15.json"
    )

    germs = payload["germs"]
    assert {(row["event_mode"], row["direction"]) for row in germs} == {
        (mode, direction)
        for mode in ("plus_one", "minus_one")
        for direction in ("+", "-")
    }
    for row in germs:
        assert row["mixed_node"] == node
        assert row["status"] == "traced"
        assert row["canonical_bound"] is True
        assert row["canonical_bracketed"] is True
        assert abs(row["event"]) <= EVENT_GATE
        assert abs(row["closure"]) <= CLOSURE_GATE
        assert 0.0 < row["canonical_distance"] <= GERM_ATTACH_DISTANCE
        assert "stopped_reason" not in row


@pytest.mark.parametrize("node", sorted(HEADLINE_NODES))
def test_the_two_signed_germs_bracket_their_organizer(node: str) -> None:
    """A germ pair that leaves the vertex the same way is not a bracket."""
    germs_name, _retry, _canonical, _junction = HEADLINE_NODES[node]
    payload = _load(germs_name)
    centre = [float(value) for value in payload["float64_center_masses"][:2]]
    for mode in ("plus_one", "minus_one"):
        rows = {row["direction"]: row for row in payload["germs"] if row["event_mode"] == mode}
        assert set(rows) == {"+", "-"}
        plus = [float(rows["+"]["masses"][i]) - centre[i] for i in (0, 1)]
        minus = [float(rows["-"]["masses"][i]) - centre[i] for i in (0, 1)]
        assert plus[0] * minus[0] + plus[1] * minus[1] < 0.0
        assert payload["directional_audit"][mode]["opposite_mass_directions"] is True


def test_the_twelve_headline_germs_pass_the_assembler_uniform_validation() -> None:
    module = _assembler()
    mixed_node_ids = frozenset(module.BASE_MIXED_NODE_IDS)
    seen = set()
    for node, (germs_name, *_rest) in HEADLINE_NODES.items():
        for row in _load(germs_name)["germs"]:
            record = dict(row, source_artifact=str(EVIDENCE / germs_name))
            assert module.germ_rejections(record, mixed_node_ids) == []
            seen.add((node, row["event_mode"], row["direction"]))
    assert len(seen) == 12
    assert module.missing_mixed_germs(
        [
            dict(row, source_artifact=str(EVIDENCE / germs_name))
            for germs_name, *_rest in HEADLINE_NODES.values()
            for row in _load(germs_name)["germs"]
        ],
        mixed_node_ids,
    ) == []


def test_the_superseded_headline_germ_artifact_is_no_longer_an_assembler_input() -> None:
    """The old twelve are history, not evidence, and must stay unwired."""
    shell = (ROOT / "scripts/assemble_v1_critical_graph.sh").read_text()
    wired = [
        line.strip()
        for line in shell.splitlines()
        if line.strip().startswith("--germs")
    ]
    assert not any("V1_MIXED_GERMS_2026-08-15.json" in line for line in wired)
    for germs_name, *_rest in HEADLINE_NODES.values():
        assert any(germs_name in line for line in wired)
