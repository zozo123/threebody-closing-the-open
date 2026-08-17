from __future__ import annotations

import copy
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from threebody_atlas.graph_semantics import (
    SCHEMA_VERSION,
    ComparisonLevel,
    EvidenceIdentity,
    GraphSemanticsError,
    MechanismEdge,
    MechanismGraph,
    MechanismNode,
    canonical_form,
    canonical_sha256,
    compare_graphs,
    find_isomorphism,
    load_graph,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "research/evidence/V1_CRITICAL_GRAPH.json"
AUDIT = ROOT / "research/graph/V1_GRAPH_ISOMORPHISM_MUTATION_AUDIT_2026-08-16.json"


def shipped_graph() -> MechanismGraph:
    return load_graph(SOURCE, repository_root=ROOT)


def cloned(graph: MechanismGraph, mutation) -> MechanismGraph:
    document = copy.deepcopy(graph.model_dump(mode="json"))
    mutation(document)
    return MechanismGraph.model_validate(document, strict=False)


def synthetic_graph() -> MechanismGraph:
    evidence = EvidenceIdentity(
        status="validated",
        level="independent",
        passed=True,
        artifact_sha256s=("1" * 64,),
        payload_sha256="2" * 64,
    )
    return MechanismGraph(
        coordinate_system="test_mass_chart",
        coordinate_axes=("m1", "m2"),
        declared_domain={"m1": ("0", "2"), "m2": ("0", "2")},
        nodes=(
            MechanismNode(
                id="left",
                physical_class="organizer",
                mechanism="plus_one",
                sheet_id="sheet-a",
                coordinates=("1", "1"),
                local_sectors=("minus", "plus"),
                evidence=evidence,
            ),
            MechanismNode(
                id="right",
                physical_class="endpoint",
                mechanism="minus_one",
                sheet_id="sheet-a",
                coordinates=("1.5", "1"),
                evidence=evidence,
            ),
        ),
        edges=(
            MechanismEdge(
                id="wall",
                source="left",
                target="right",
                kind="mechanism_arc",
                mechanism="plus_one",
                orientation="U->S",
                sheet_id="sheet-a",
                source_sector="plus",
                target_sector="incoming",
                evidence=evidence,
            ),
        ),
    )


def test_generated_schema_and_mutation_audit_are_current():
    subprocess.run(
        [sys.executable, "scripts/build_graph_semantics_artifacts.py", "--check"],
        cwd=ROOT,
        check=True,
    )
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    assert audit["passed"] is True
    assert audit["case_count"] == len(audit["cases"]) == 12
    expected = {
        "added_edge",
        "removed_edge",
        "changed_endpoint",
        "changed_mechanism",
        "orientation_flip",
        "sheet_reassignment",
        "node_split",
        "node_merge",
        "evidence_changed",
    }
    observed = {
        case["expected_difference"]
        for case in audit["cases"]
        if case.get("expected_difference") is not None
    }
    assert observed == expected


def test_shipped_graph_adapter_freezes_sheet_domain_nodes_and_edges():
    graph = shipped_graph()
    assert graph.schema_version == SCHEMA_VERSION
    assert len(graph.nodes) == 14
    # TWO sweep polylines still have an unclassified lattice end; the adapter
    # does not invent nodes for those ends, so they are not edges in the
    # incidence graph.  It was three until plus_one component 12's tip was
    # re-localized at BigFloat: cell 10131 then re-certified under float64, the
    # component gained a second certifying root, its two sides stopped sharing a
    # seed, and both its termini resolved -- so that polyline joined the sheet
    # and the edge count went 10 -> 11.  This number tracks
    # unclassified_edge_endpoints and should fall again as those two close.
    assert len(graph.edges) == 11
    assert graph.coordinate_axes == ("m1", "m2", "m3")
    assert graph.declared_domain == {
        "m1": ("0.8", "1.1"),
        "m2": ("0.7", "1.2"),
        "m3": ("1", "1"),
    }
    assert len({node.sheet_id for node in graph.nodes}) == 1
    assert all(edge.evidence is not None for edge in graph.edges)
    assert all(edge.source_sector and edge.target_sector for edge in graph.edges)


def test_all_levels_ignore_ids_and_input_order():
    graph = shipped_graph()
    document = copy.deepcopy(graph.model_dump(mode="json"))
    random.Random(17).shuffle(document["nodes"])
    random.Random(18).shuffle(document["edges"])
    rename = {node["id"]: f"node-{index}" for index, node in enumerate(document["nodes"])}
    for node in document["nodes"]:
        node["id"] = rename[node["id"]]
    for index, edge in enumerate(document["edges"]):
        edge["id"] = f"edge-{index}"
        edge["source"] = rename[edge["source"]]
        edge["target"] = rename[edge["target"]]
    reconstructed = MechanismGraph.model_validate(document, strict=False)

    for level in ComparisonLevel:
        comparison = compare_graphs(graph, reconstructed, level)
        assert comparison.equivalent is True
        assert len(comparison.mapping) == len(graph.nodes)
        assert comparison.left_canonical_sha256 == comparison.right_canonical_sha256


def test_comparison_levels_are_nested_and_orientation_is_not_coordinate_noise():
    graph = synthetic_graph()
    flipped = cloned(
        graph,
        lambda document: document["edges"][0].update(orientation="S->U"),
    )
    assert compare_graphs(graph, flipped, ComparisonLevel.TOPOLOGY_ONLY).equivalent
    assert compare_graphs(graph, flipped, ComparisonLevel.MECHANISM_LABELED).equivalent
    oriented = compare_graphs(graph, flipped, ComparisonLevel.ORIENTED)
    assert not oriented.equivalent
    assert {difference.kind for difference in oriented.differences} == {"orientation_flip"}


def test_mechanism_mismatch_never_matches_through_coordinate_tolerance():
    graph = synthetic_graph()
    changed = cloned(
        graph,
        lambda document: document["nodes"][0].update(mechanism="minus_one"),
    )
    assert find_isomorphism(
        graph,
        changed,
        ComparisonLevel.SHEET_AWARE,
        coordinate_tolerance="1000",
    ) is None
    comparison = compare_graphs(
        graph,
        changed,
        ComparisonLevel.SHEET_AWARE,
        coordinate_tolerance="1000",
    )
    assert "changed_mechanism" in {difference.kind for difference in comparison.differences}


def test_different_sheets_never_collapse_even_at_identical_coordinates():
    graph = synthetic_graph()
    changed = cloned(
        graph,
        lambda document: document["nodes"][0].update(sheet_id="sheet-b"),
    )
    comparison = compare_graphs(
        graph,
        changed,
        ComparisonLevel.SHEET_AWARE,
        coordinate_tolerance="1000",
    )
    assert not comparison.equivalent
    assert "sheet_reassignment" in {difference.kind for difference in comparison.differences}


def test_coordinate_tolerance_applies_after_identity_and_is_decimal_exact():
    graph = synthetic_graph()
    shifted = cloned(
        graph,
        lambda document: document["nodes"][0]["coordinates"].__setitem__(
            0,
            "1.000000005",
        ),
    )
    within = compare_graphs(
        graph,
        shifted,
        ComparisonLevel.SHEET_AWARE,
        coordinate_tolerance="0.00000001",
    )
    outside = compare_graphs(
        graph,
        shifted,
        ComparisonLevel.SHEET_AWARE,
        coordinate_tolerance="0.000000001",
    )
    assert within.equivalent
    assert not outside.equivalent
    assert "coordinate_shift" in {difference.kind for difference in outside.differences}
    assert within.left_canonical_sha256 != within.right_canonical_sha256


def test_parallel_edge_multiplicity_is_semantic():
    graph = synthetic_graph()
    duplicated = cloned(
        graph,
        lambda document: document["edges"].append(
            {**copy.deepcopy(document["edges"][0]), "id": "parallel-wall"}
        ),
    )
    comparison = compare_graphs(graph, duplicated, ComparisonLevel.TOPOLOGY_ONLY)
    assert not comparison.equivalent
    assert {difference.kind for difference in comparison.differences} == {"added_edge"}


def _referenced_evidence_paths(document: dict) -> set[str]:
    paths: set[str] = set()
    for node in document.get("nodes", []):
        evidence = node.get("evidence")
        if isinstance(evidence, str) and evidence:
            paths.add(evidence)
    for edge in document.get("edges", []):
        endpoints = edge.get("endpoints") or {}
        for endpoint in endpoints.values():
            if not isinstance(endpoint, dict):
                continue
            binding = endpoint.get("binding_evidence")
            if isinstance(binding, str) and binding:
                paths.add(binding)
    return paths


def test_deleted_referenced_evidence_fails_closed(tmp_path: Path):
    document = json.loads(SOURCE.read_text(encoding="utf-8"))
    referenced = sorted(_referenced_evidence_paths(document))
    assert referenced
    for relative in referenced:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    graph_path = tmp_path / "mutated-critical-graph.json"
    graph_path.write_text(json.dumps(document), encoding="utf-8")
    present = load_graph(graph_path, repository_root=tmp_path)
    assert any(node.evidence and node.evidence.artifact_sha256s for node in present.nodes)

    deleted = tmp_path / referenced[0]
    deleted.unlink()
    with pytest.raises(GraphSemanticsError, match="missing or unreadable"):
        load_graph(graph_path, repository_root=tmp_path)


def test_renamed_binding_evidence_fails_closed(tmp_path: Path):
    document = json.loads(SOURCE.read_text(encoding="utf-8"))
    edge = next(
        item
        for item in document["edges"]
        if (item.get("endpoints") or {}).get("start", {}).get("binding_evidence")
    )
    original = edge["endpoints"]["start"]["binding_evidence"]
    edge["endpoints"]["start"]["binding_evidence"] = original.replace(
        ".json", ".renamed-away.json"
    )
    graph_path = tmp_path / "renamed-evidence-graph.json"
    graph_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(GraphSemanticsError, match="missing or unreadable"):
        load_graph(graph_path, repository_root=ROOT)
    # The original artifact still exists; only the referenced name changed.
    assert (ROOT / original).is_file()


def test_null_evidence_still_loads_without_a_digest():
    graph = shipped_graph()
    domain = next(node for node in graph.nodes if node.id.startswith("domain_"))
    assert domain.evidence is not None
    assert domain.evidence.artifact_sha256s == ()


def test_evidence_is_excluded_until_the_strictest_level():
    graph = synthetic_graph()
    changed = cloned(
        graph,
        lambda document: document["nodes"][0]["evidence"].update(
            artifact_sha256s=["f" * 64]
        ),
    )
    assert compare_graphs(graph, changed, ComparisonLevel.SHEET_AWARE).equivalent
    strict = compare_graphs(graph, changed, ComparisonLevel.EVIDENCE_EQUIVALENT)
    assert not strict.equivalent
    assert "evidence_changed" in {difference.kind for difference in strict.differences}


def test_local_sector_and_domain_face_are_sheet_aware_semantics():
    graph = synthetic_graph()
    changed_sector = cloned(
        graph,
        lambda document: document["nodes"][0].update(local_sectors=["minus"]),
    )
    assert compare_graphs(
        graph,
        changed_sector,
        ComparisonLevel.ORIENTED,
    ).equivalent
    assert not compare_graphs(
        graph,
        changed_sector,
        ComparisonLevel.SHEET_AWARE,
    ).equivalent


def test_invalid_graphs_fail_closed():
    graph = synthetic_graph()
    raw = graph.model_dump(mode="json")
    raw["edges"][0]["target"] = "missing"
    with pytest.raises(ValidationError, match="unknown endpoint"):
        MechanismGraph.model_validate(raw, strict=False)

    raw = graph.model_dump(mode="json")
    raw["nodes"].append(copy.deepcopy(raw["nodes"][0]))
    with pytest.raises(ValidationError, match="node ids must be unique"):
        MechanismGraph.model_validate(raw, strict=False)

    raw = graph.model_dump(mode="json")
    raw["nodes"][0]["coordinates"][0] = "1.0"
    with pytest.raises(ValidationError, match="not canonical"):
        MechanismGraph.model_validate(raw, strict=False)

    raw = graph.model_dump(mode="json")
    raw["nodes"][0]["unexpected"] = "hidden semantics"
    with pytest.raises(ValidationError, match="Extra inputs"):
        MechanismGraph.model_validate(raw, strict=False)


def test_domain_boundary_nodes_require_face_coordinates():
    graph = synthetic_graph()
    raw = graph.model_dump(mode="json")
    raw["nodes"][0]["domain_face"] = "domain_m1_min"
    with pytest.raises(ValidationError, match="needs a boundary_coordinate"):
        MechanismGraph.model_validate(raw, strict=False)


def test_pathological_symmetry_exceeding_bound_fails_instead_of_using_input_order():
    graph = synthetic_graph()
    raw = graph.model_dump(mode="json")
    template = copy.deepcopy(raw["nodes"][0])
    template["coordinates"] = None
    template["evidence"] = None
    raw["nodes"] = [{**copy.deepcopy(template), "id": f"isolated-{index}"} for index in range(11)]
    raw["edges"] = []
    symmetric = MechanismGraph.model_validate(raw, strict=False)
    with pytest.raises(GraphSemanticsError, match="fail-closed limit"):
        canonical_form(symmetric, ComparisonLevel.TOPOLOGY_ONLY)


def test_canonical_hashes_match_frozen_audit():
    graph = shipped_graph()
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    for level in ComparisonLevel:
        assert canonical_sha256(graph, level) == audit["baseline_canonical_sha256"][level.value]


def test_cli_compares_shipped_graph_without_hand_mapping(tmp_path: Path):
    output = tmp_path / "comparison.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/compare_mechanism_graphs.py",
            str(SOURCE),
            str(SOURCE),
            "--level",
            "evidence_equivalent",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
    )
    comparison = json.loads(output.read_text(encoding="utf-8"))
    assert comparison["equivalent"] is True
    assert len(comparison["mapping"]) == 14
