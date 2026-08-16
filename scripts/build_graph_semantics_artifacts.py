#!/usr/bin/env python3
"""Build the graph schema and executable isomorphism mutation audit."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Callable

from threebody_atlas.graph_semantics import (
    ComparisonLevel,
    MechanismGraph,
    canonical_sha256,
    compare_graphs,
    load_graph,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE_GRAPH = ROOT / "research/evidence/V1_CRITICAL_GRAPH.json"
SCHEMA = ROOT / "research/graph/V1_GRAPH_CANONICAL_SCHEMA.json"
AUDIT = ROOT / "research/graph/V1_GRAPH_ISOMORPHISM_MUTATION_AUDIT_2026-08-16.json"


def _clone(graph: MechanismGraph, mutation: Callable[[dict[str, Any]], None]) -> MechanismGraph:
    document = copy.deepcopy(graph.model_dump(mode="json"))
    mutation(document)
    return MechanismGraph.model_validate(document, strict=False)


def _difference_kinds(comparison: Any) -> list[str]:
    return sorted({difference.kind for difference in comparison.differences})


def _mutation_case(
    baseline: MechanismGraph,
    *,
    name: str,
    expected: str,
    level: ComparisonLevel,
    mutation: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    mutated = _clone(baseline, mutation)
    comparison = compare_graphs(baseline, mutated, level)
    observed = _difference_kinds(comparison)
    return {
        "name": name,
        "level": level.value,
        "expected_difference": expected,
        "observed_differences": observed,
        "equivalent": comparison.equivalent,
        "passed": not comparison.equivalent and expected in observed,
        "baseline_sha256": comparison.left_canonical_sha256,
        "mutated_sha256": comparison.right_canonical_sha256,
    }


def _renamed_and_reordered(graph: MechanismGraph) -> MechanismGraph:
    document = copy.deepcopy(graph.model_dump(mode="json"))
    random.Random(173).shuffle(document["nodes"])
    random.Random(174).shuffle(document["edges"])
    rename = {node["id"]: f"reconstruction-node-{index}" for index, node in enumerate(document["nodes"])}
    for node in document["nodes"]:
        node["id"] = rename[node["id"]]
    for index, edge in enumerate(document["edges"]):
        edge["id"] = f"reconstruction-edge-{index}"
        edge["source"] = rename[edge["source"]]
        edge["target"] = rename[edge["target"]]
    return MechanismGraph.model_validate(document, strict=False)


def _add_edge(document: dict[str, Any]) -> None:
    edge = copy.deepcopy(document["edges"][0])
    edge["id"] = "mutation-added-parallel-edge"
    document["edges"].append(edge)


def _remove_edge(document: dict[str, Any]) -> None:
    document["edges"].pop()


def _change_endpoint(document: dict[str, Any]) -> None:
    edge = document["edges"][0]
    edge["target"] = "mixed_secondary_left"


def _change_mechanism(document: dict[str, Any]) -> None:
    document["edges"][0]["mechanism"] = "plus_one"


def _flip_orientation(document: dict[str, Any]) -> None:
    edge = document["edges"][0]
    edge["orientation"] = "S->U" if edge["orientation"] == "U->S" else "U->S"


def _reassign_sheet(document: dict[str, Any]) -> None:
    node = next(node for node in document["nodes"] if node["id"] == "mixed_principal_left")
    node["sheet_id"] = "adversarial-nearby-sheet"


def _split_node(document: dict[str, Any]) -> None:
    original = next(node for node in document["nodes"] if node["id"] == "mixed_principal_left")
    split = copy.deepcopy(original)
    split["id"] = "mixed_principal_left_split"
    document["nodes"].append(split)
    edge = next(edge for edge in document["edges"] if edge["source"] == original["id"])
    edge["source"] = split["id"]


def _merge_nodes(document: dict[str, Any]) -> None:
    removed = "domain_m2_max_m1_1p071"
    survivor = "domain_m2_max_m1_1p053"
    for edge in document["edges"]:
        if edge["source"] == removed:
            edge["source"] = survivor
        if edge["target"] == removed:
            edge["target"] = survivor
    document["nodes"] = [node for node in document["nodes"] if node["id"] != removed]


def _change_evidence(document: dict[str, Any]) -> None:
    node = next(node for node in document["nodes"] if node["id"] == "mixed_principal_left")
    node["evidence"]["artifact_sha256s"] = ["0" * 64]


def _shift_coordinate(document: dict[str, Any], delta: str) -> None:
    node = next(node for node in document["nodes"] if node["id"] == "mixed_principal_left")
    from decimal import Decimal

    node["coordinates"][0] = format(Decimal(node["coordinates"][0]) + Decimal(delta), "f")


def build_audit() -> dict[str, Any]:
    baseline = load_graph(SOURCE_GRAPH, repository_root=ROOT)
    renamed = _renamed_and_reordered(baseline)

    order_results = {}
    for level in ComparisonLevel:
        comparison = compare_graphs(baseline, renamed, level)
        order_results[level.value] = {
            "equivalent": comparison.equivalent,
            "same_canonical_sha256": (
                comparison.left_canonical_sha256 == comparison.right_canonical_sha256
            ),
            "mapping_size": len(comparison.mapping),
        }
    order_case = {
        "name": "id-and-input-order-invariance",
        "passed": all(
            result["equivalent"] and result["same_canonical_sha256"]
            for result in order_results.values()
        ),
        "levels": order_results,
    }

    mutations = [
        _mutation_case(
            baseline,
            name="parallel-edge-added",
            expected="added_edge",
            level=ComparisonLevel.TOPOLOGY_ONLY,
            mutation=_add_edge,
        ),
        _mutation_case(
            baseline,
            name="edge-removed",
            expected="removed_edge",
            level=ComparisonLevel.TOPOLOGY_ONLY,
            mutation=_remove_edge,
        ),
        _mutation_case(
            baseline,
            name="endpoint-rewired",
            expected="changed_endpoint",
            level=ComparisonLevel.TOPOLOGY_ONLY,
            mutation=_change_endpoint,
        ),
        _mutation_case(
            baseline,
            name="active-mechanism-changed",
            expected="changed_mechanism",
            level=ComparisonLevel.MECHANISM_LABELED,
            mutation=_change_mechanism,
        ),
        _mutation_case(
            baseline,
            name="orientation-flipped",
            expected="orientation_flip",
            level=ComparisonLevel.ORIENTED,
            mutation=_flip_orientation,
        ),
        _mutation_case(
            baseline,
            name="node-sheet-reassigned",
            expected="sheet_reassignment",
            level=ComparisonLevel.SHEET_AWARE,
            mutation=_reassign_sheet,
        ),
        _mutation_case(
            baseline,
            name="node-split",
            expected="node_split",
            level=ComparisonLevel.TOPOLOGY_ONLY,
            mutation=_split_node,
        ),
        _mutation_case(
            baseline,
            name="nodes-merged",
            expected="node_merge",
            level=ComparisonLevel.TOPOLOGY_ONLY,
            mutation=_merge_nodes,
        ),
        _mutation_case(
            baseline,
            name="evidence-content-substituted",
            expected="evidence_changed",
            level=ComparisonLevel.EVIDENCE_EQUIVALENT,
            mutation=_change_evidence,
        ),
    ]

    near = _clone(baseline, lambda document: _shift_coordinate(document, "0.000000005"))
    within = compare_graphs(
        baseline,
        near,
        ComparisonLevel.SHEET_AWARE,
        coordinate_tolerance="0.00000001",
    )
    outside = compare_graphs(
        baseline,
        near,
        ComparisonLevel.SHEET_AWARE,
        coordinate_tolerance="0.000000001",
    )
    tolerance_case = {
        "name": "coordinate-tolerance-after-physical-identity",
        "passed": within.equivalent
        and not outside.equivalent
        and "coordinate_shift" in _difference_kinds(outside),
        "within_tolerance_equivalent": within.equivalent,
        "outside_tolerance_equivalent": outside.equivalent,
        "outside_differences": _difference_kinds(outside),
    }

    evidence_mutated = _clone(baseline, _change_evidence)
    sheet_comparison = compare_graphs(
        baseline,
        evidence_mutated,
        ComparisonLevel.SHEET_AWARE,
    )
    evidence_comparison = compare_graphs(
        baseline,
        evidence_mutated,
        ComparisonLevel.EVIDENCE_EQUIVALENT,
    )
    evidence_level_case = {
        "name": "evidence-excluded-from-pure-graph-levels",
        "passed": sheet_comparison.equivalent and not evidence_comparison.equivalent,
        "sheet_aware_equivalent": sheet_comparison.equivalent,
        "evidence_equivalent": evidence_comparison.equivalent,
    }

    schema_sha256 = hashlib.sha256(_render(build_schema()).encode("utf-8")).hexdigest()
    all_cases = [order_case, *mutations, tolerance_case, evidence_level_case]
    return {
        "schema": "atlas.graph-isomorphism-mutation-audit.v1",
        "generated_on": "2026-08-16",
        "source_graph": str(SOURCE_GRAPH.relative_to(ROOT)),
        "source_graph_sha256": hashlib.sha256(SOURCE_GRAPH.read_bytes()).hexdigest(),
        "canonical_schema_sha256": schema_sha256,
        "baseline_canonical_sha256": {
            level.value: canonical_sha256(baseline, level) for level in ComparisonLevel
        },
        "case_count": len(all_cases),
        "passed": all(case["passed"] for case in all_cases),
        "cases": all_cases,
    }


def build_schema() -> dict[str, Any]:
    schema = MechanismGraph.model_json_schema()
    schema["$id"] = "https://threebody-atlas.invalid/schema/atlas.mechanism-multigraph.v1.json"
    schema["title"] = "ATLAS canonical sheet-aware mechanism multigraph"
    return schema


def _render(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _check_or_write(path: Path, text: str, check: bool) -> None:
    if check:
        if not path.exists():
            raise SystemExit(f"missing generated artifact: {path.relative_to(ROOT)}")
        if path.read_text(encoding="utf-8") != text:
            raise SystemExit(f"stale generated artifact: {path.relative_to(ROOT)}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    schema = build_schema()
    audit = build_audit()
    if not audit["passed"]:
        failed = [case["name"] for case in audit["cases"] if not case["passed"]]
        raise SystemExit(f"graph semantic mutations escaped detection: {', '.join(failed)}")
    _check_or_write(SCHEMA, _render(schema), args.check)
    _check_or_write(AUDIT, _render(audit), args.check)
    verb = "verified" if args.check else "wrote"
    print(f"{verb} graph schema and {audit['case_count']}-case mutation audit")


if __name__ == "__main__":
    main()
