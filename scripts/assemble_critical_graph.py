#!/usr/bin/env python3
"""Assemble the v1 mechanism-resolved Floquet critical graph.

620 catalog S/U cells are samples supporting the graph. They are not 620 edges.
An edge is a mechanism-specific polyline carrying a list of source-cell ids.
Endpoints, mixed germs, and completeness must come from artifacts.
The assembler never invents a classification and never flips
release_ready without the mandatory artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import string
from collections import defaultdict
from pathlib import Path
from typing import Any


REQUIRED_HEADLINE_IDS = (
    "mixed_principal_left",
    "mixed_secondary_left",
    "mixed_principal_right",
    "headline_lower_plus_one",
    "headline_upper_collision",
)
MIXED_NODE_IDS = (
    "mixed_principal_left",
    "mixed_secondary_left",
    "mixed_principal_right",
)
REQUIRED_GERM_KEYS = (
    ("plus_one", "+"),
    ("plus_one", "-"),
    ("minus_one", "+"),
    ("minus_one", "-"),
)
LEFT_BIRTH_CLASSES = frozenset(
    {"projection_fold", "two_separate_arcs", "mixed_organizer"}
)
RIGHT_DEATH_CLASSES = frozenset({"mixed_organizer", "projection_fold", "domain_boundary"})
DAUGHTER_CLASSES = frozenset(
    {
        "reconnecting",
        "closed_loop",
        "distinct_branch",
        "obstruction",
        "no_branch_attachment",
        "falsified",
    }
)
MASS_JUMP = 0.025
M1_SLICE_GAP = 0.0015
GERM_ATTACH_DISTANCE = 0.008
DOMAIN_TOLERANCE = 0.0015
DECLARED_DOMAIN = {"m1": (0.8, 1.1), "m2": (0.7, 1.2)}
RELEASE_EVIDENCE_LEVELS = frozenset(
    {"independently_reproduced", "physical", "continuation", "definition"}
)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def node(node_id: str, kind: str, *, status: str, masses: list[Any] | None = None, **extra: Any) -> dict[str, Any]:
    record = {"id": node_id, "kind": kind, "status": status, "masses": masses}
    record.update(extra)
    return record


def headline_nodes(root: Path) -> list[dict[str, Any]]:
    mixed_left = load(root / "research/evidence/V1_MIXED_CANONICAL_PRINCIPAL_LEFT_2026-08-15.json")
    mixed_sec = load(root / "research/evidence/V1_MIXED_CANONICAL_SECONDARY_LEFT_2026-08-15.json")
    mixed_right = load(root / "research/evidence/V1_MIXED_CANONICAL_PRINCIPAL_RIGHT_2026-08-15.json")
    plus_one = load(root / "research/evidence/V1_CANONICAL_LOWER_PLUS_ONE_2026-08-15.json")
    collision = load(root / "research/evidence/V1_CANONICAL_UPPER_COLLISION_2026-08-15.json")
    return [
        node(
            "mixed_principal_left",
            "mixed_organizer",
            status="independently_reproduced",
            masses=mixed_left.get("masses"),
            mechanism="mixed_plus_one_minus_one",
            evidence="research/evidence/V1_MIXED_CANONICAL_PRINCIPAL_LEFT_2026-08-15.json",
            passed=mixed_left.get("passed"),
        ),
        node(
            "mixed_secondary_left",
            "mixed_organizer",
            status="independently_reproduced",
            masses=mixed_sec.get("masses"),
            mechanism="mixed_plus_one_minus_one",
            evidence="research/evidence/V1_MIXED_CANONICAL_SECONDARY_LEFT_2026-08-15.json",
            passed=mixed_sec.get("passed"),
        ),
        node(
            "mixed_principal_right",
            "mixed_organizer",
            status="independently_reproduced",
            masses=mixed_right.get("masses"),
            mechanism="mixed_plus_one_minus_one",
            evidence="research/evidence/V1_MIXED_CANONICAL_PRINCIPAL_RIGHT_2026-08-15.json",
            passed=mixed_right.get("passed"),
        ),
        node(
            "headline_lower_plus_one",
            "event_representative",
            status="independently_reproduced",
            masses=None,
            mechanism=plus_one.get("mechanism"),
            evidence="research/evidence/V1_CANONICAL_LOWER_PLUS_ONE_2026-08-15.json",
            passed=plus_one.get("passed"),
            bracket_m2=plus_one.get("critical_bracket_m2"),
        ),
        node(
            "headline_upper_collision",
            "event_representative",
            status="independently_reproduced",
            masses=None,
            mechanism=collision.get("mechanism"),
            evidence="research/evidence/V1_CANONICAL_UPPER_COLLISION_2026-08-15.json",
            passed=collision.get("passed"),
            bracket_m2=collision.get("critical_bracket_m2"),
        ),
    ]


def forbidden_class(value: Any) -> bool:
    text = str(value or "").lower().replace(" ", "_")
    return "newton" in text and "fail" in text


def load_classification(
    path: Path | None,
    *,
    node_id: str,
    default_kind: str,
    allowed: frozenset[str],
    missing_note: str,
) -> dict[str, Any]:
    if path is None:
        return node(
            node_id,
            default_kind,
            status="unresolved",
            masses=None,
            mechanism="unknown",
            note=missing_note,
            passed=False,
        )
    payload = load(path)
    raw_class = payload.get("class") or payload.get("mechanism") or payload.get("classification")
    if forbidden_class(raw_class) or forbidden_class(payload.get("status")) or forbidden_class(payload.get("error")):
        return node(
            node_id,
            default_kind,
            status="illegal",
            masses=payload.get("masses"),
            mechanism=str(raw_class),
            note="Newton-failed is not an allowed endpoint class",
            passed=False,
            evidence=str(path),
        )
    evidence_level = str(payload.get("evidence_level") or "screening")
    passed = (
        bool(payload.get("passed"))
        and str(raw_class) in allowed
        and evidence_level in RELEASE_EVIDENCE_LEVELS
    )
    return node(
        node_id,
        str(payload.get("kind") or default_kind),
        status="independently_reproduced" if passed else "unresolved",
        masses=payload.get("masses"),
        mechanism=str(raw_class) if raw_class is not None else "unknown",
        note=payload.get("note"),
        passed=passed,
        evidence=str(path),
        estimator=payload.get("estimator"),
        evidence_level=evidence_level,
        screening_passed=bool(payload.get("passed")),
        edge_endpoint_bindings=payload.get("edge_endpoint_bindings") or [],
    )


def polyline_edges(roots: list[dict[str, Any]], *, jump: float = MASS_JUMP) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for root in roots:
        grouped[
            (
                str(root.get("event_mode") or "unknown"),
                str(root.get("orientation") or "unknown"),
            )
        ].append(root)
    edges: list[dict[str, Any]] = []
    for (mode, orientation), items in grouped.items():
        by_slice: dict[float, list[dict[str, Any]]] = defaultdict(list)
        for item in items:
            masses = item.get("masses") or [0.0, 0.0]
            by_slice[round(float(masses[0]), 10)].append(item)
        parent = {int(item["cell_id"]): int(item["cell_id"]) for item in items}

        def find(cell: int, parent_map: dict[int, int] = parent) -> int:
            while parent_map[cell] != cell:
                parent_map[cell] = parent_map[parent_map[cell]]
                cell = parent_map[cell]
            return cell

        def union(
            left: int,
            right: int,
            parent_map: dict[int, int] = parent,
        ) -> None:
            a, b = find(left), find(right)
            if a != b:
                parent_map[max(a, b)] = min(a, b)

        slice_keys = sorted(by_slice)
        for left_key, right_key in zip(slice_keys, slice_keys[1:], strict=False):
            if right_key - left_key > M1_SLICE_GAP:
                continue
            candidates: list[tuple[float, int, int]] = []
            for left in by_slice[left_key]:
                for right in by_slice[right_key]:
                    lm, rm = left.get("masses") or [0.0, 0.0], right.get("masses") or [0.0, 0.0]
                    distance = (
                        (float(lm[0]) - float(rm[0])) ** 2
                        + (float(lm[1]) - float(rm[1])) ** 2
                    ) ** 0.5
                    if distance <= jump:
                        candidates.append((distance, int(left["cell_id"]), int(right["cell_id"])))
            used_left: set[int] = set()
            used_right: set[int] = set()
            for _distance, left_cell, right_cell in sorted(candidates):
                if left_cell in used_left or right_cell in used_right:
                    continue
                union(left_cell, right_cell)
                used_left.add(left_cell)
                used_right.add(right_cell)

        components: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for item in items:
            components[find(int(item["cell_id"]))].append(item)
        runs = sorted(
            components.values(),
            key=lambda run: min(int(item["cell_id"]) for item in run),
        )
        orientation_id = orientation.lower().replace("->", "_to_").replace("-", "_")
        for index, run in enumerate(runs):
            run.sort(
                key=lambda item: (
                    float((item.get("masses") or [0.0, 0.0])[0]),
                    float((item.get("masses") or [0.0, 0.0])[1]),
                    int(item["cell_id"]),
                )
            )
            cell_ids = [int(row["cell_id"]) for row in run]
            start_masses = [float(value) for value in (run[0].get("masses") or [])]
            end_masses = [float(value) for value in (run[-1].get("masses") or [])]
            widths = []
            for row in run:
                bracket = row.get("source_m2_bracket") or []
                if len(bracket) == 2:
                    widths.append(abs(float(bracket[1]) - float(bracket[0])))
            edges.append(
                {
                    "id": f"{mode}_{orientation_id}_{index}",
                    "kind": "mechanism_polyline",
                    "mechanism": mode,
                    "orientation": orientation,
                    "cell_ids": cell_ids,
                    "source_cell_count": len(cell_ids),
                    "estimators": sorted({str(row.get("estimator") or "float64") for row in run}),
                    "uncertainty": {
                        "max_abs_event": max(abs(float(row.get("event") or 0.0)) for row in run),
                        "max_closure": max(float(row.get("closure") or 0.0) for row in run),
                        "max_source_m2_bracket_width": max(widths) if widths else None,
                    },
                    "endpoints": {
                        "start": {"cell_id": cell_ids[0], "masses": start_masses, "node": None},
                        "end": {"cell_id": cell_ids[-1], "masses": end_masses, "node": None},
                        "classified": False,
                    },
                }
            )
    edges.sort(key=lambda edge: (str(edge["mechanism"]), edge["cell_ids"][0]))
    return edges


def mass_distance(left: list[Any] | None, right: list[Any] | None) -> float:
    if not left or not right or len(left) < 2 or len(right) < 2:
        return float("inf")
    return ((float(left[0]) - float(right[0])) ** 2 + (float(left[1]) - float(right[1])) ** 2) ** 0.5


def domain_node(masses: list[Any] | None) -> str | None:
    if not masses or len(masses) < 2:
        return None
    m1, m2 = float(masses[0]), float(masses[1])
    candidates = (
        (abs(m1 - DECLARED_DOMAIN["m1"][0]), "domain_m1_min"),
        (abs(m1 - DECLARED_DOMAIN["m1"][1]), "domain_m1_max"),
        (abs(m2 - DECLARED_DOMAIN["m2"][0]), "domain_m2_min"),
        (abs(m2 - DECLARED_DOMAIN["m2"][1]), "domain_m2_max"),
    )
    distance, node_id = min(candidates)
    return node_id if distance <= DOMAIN_TOLERANCE else None


def apply_classification_bindings(
    edges: list[dict[str, Any]], nodes: list[dict[str, Any]]
) -> list[str]:
    errors: list[str] = []
    existing_nodes = {str(item["id"]): item for item in nodes}
    for item in list(nodes):
        for binding in item.get("edge_endpoint_bindings") or []:
            side = str(binding.get("side") or "")
            if side not in {"start", "end"}:
                errors.append(f"{item['id']}: binding side must be start or end")
                continue
            try:
                cell_id = int(binding["cell_id"])
            except (KeyError, TypeError, ValueError):
                errors.append(f"{item['id']}: binding needs an integer cell_id")
                continue
            matches = [
                edge
                for edge in edges
                if int(edge["endpoints"][side]["cell_id"]) == cell_id
                and (
                    binding.get("mechanism") is None
                    or edge.get("mechanism") == binding.get("mechanism")
                )
                and (
                    binding.get("orientation") is None
                    or edge.get("orientation") == binding.get("orientation")
                )
            ]
            if len(matches) != 1:
                errors.append(
                    f"{item['id']}: binding cell={cell_id} side={side} matched {len(matches)} edges"
                )
                continue
            endpoint = matches[0]["endpoints"][side]
            if endpoint.get("node") or endpoint.get("reserved_for"):
                errors.append(f"{item['id']}: duplicate binding for cell={cell_id} side={side}")
                continue
            endpoint["attachment"] = (
                "verified_classification_artifact"
                if item.get("passed")
                else "unverified_classification_artifact"
            )
            endpoint["binding_class"] = item.get("mechanism")
            endpoint["binding_evidence"] = item.get("evidence")
            if item.get("passed"):
                attachment_node = str(binding.get("node_id") or item["id"])
                if not attachment_node:
                    errors.append(f"{item['id']}: binding node_id must be non-empty")
                    continue
                if attachment_node not in existing_nodes:
                    derived = node(
                        attachment_node,
                        str(binding.get("node_kind") or item.get("mechanism") or item.get("kind")),
                        status=str(item.get("status")),
                        masses=binding.get("masses"),
                        mechanism=item.get("mechanism"),
                        passed=True,
                        evidence=item.get("evidence"),
                        evidence_level=item.get("evidence_level"),
                        parent_classification=item["id"],
                    )
                    nodes.append(derived)
                    existing_nodes[attachment_node] = derived
                elif attachment_node != item["id"]:
                    existing = existing_nodes[attachment_node]
                    if existing.get("parent_classification") != item["id"]:
                        errors.append(
                            f"{item['id']}: binding node_id {attachment_node} conflicts with an existing node"
                        )
                        continue
                endpoint["node"] = attachment_node
            else:
                endpoint["reserved_for"] = item["id"]
    return errors


def attach_edge_endpoints(
    edges: list[dict[str, Any]],
    nodes: list[dict[str, Any]],
    germs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    binding_errors = apply_classification_bindings(edges, nodes)
    used_domains: set[str] = set()
    for edge in edges:
        for side in ("start", "end"):
            endpoint = edge["endpoints"][side]
            if endpoint.get("node") or endpoint.get("reserved_for"):
                continue
            masses = endpoint.get("masses")
            boundary = domain_node(masses)
            if boundary:
                endpoint["node"] = boundary
                endpoint["attachment"] = "declared_domain_boundary"
                used_domains.add(boundary)

    passed_nodes = {str(item["id"]) for item in nodes if item.get("passed")}
    candidates: list[tuple[float, str, str, int, str]] = []
    for edge in edges:
        for side in ("start", "end"):
            endpoint = edge["endpoints"][side]
            if endpoint.get("node") or endpoint.get("reserved_for"):
                continue
            for germ_index, germ in enumerate(germs):
                if not valid_germ(germ):
                    continue
                node_id = str(germ.get("mixed_node"))
                if node_id not in passed_nodes or germ.get("event_mode") != edge.get("mechanism"):
                    continue
                distance = mass_distance(endpoint.get("masses"), germ.get("masses"))
                if distance <= GERM_ATTACH_DISTANCE:
                    candidates.append((distance, edge["id"], side, germ_index, node_id))
    used_endpoints: set[tuple[str, str]] = set()
    used_germs: set[int] = set()
    edge_by_id = {str(edge["id"]): edge for edge in edges}
    for distance, edge_id, side, germ_index, node_id in sorted(candidates):
        endpoint_key = (edge_id, side)
        if endpoint_key in used_endpoints or germ_index in used_germs:
            continue
        endpoint = edge_by_id[edge_id]["endpoints"][side]
        endpoint["node"] = node_id
        endpoint["attachment"] = "continuation_germ"
        endpoint["distance_to_germ"] = distance
        endpoint["germ_direction"] = germs[germ_index].get("direction")
        endpoint["binding_evidence"] = germs[germ_index].get("source_artifact")
        used_endpoints.add(endpoint_key)
        used_germs.add(germ_index)

    existing = {str(item["id"]) for item in nodes}
    for node_id in sorted(used_domains):
        if node_id not in existing:
            nodes.append(
                node(
                    node_id,
                    "declared_domain_boundary",
                    status="frozen_domain",
                    masses=None,
                    passed=True,
                    evidence_level="definition",
                )
            )
    unclassified: list[dict[str, Any]] = []
    for edge in edges:
        for side in ("start", "end"):
            endpoint = edge["endpoints"][side]
            if not endpoint.get("node"):
                unclassified.append(
                    {
                        "edge": edge["id"],
                        "side": side,
                        "masses": endpoint.get("masses"),
                        "reserved_for": endpoint.get("reserved_for"),
                    }
                )
        edge["endpoints"]["classified"] = all(
            edge["endpoints"][side].get("node") for side in ("start", "end")
        )
    return unclassified, binding_errors


def germ_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(record.get("mixed_node") or record.get("node") or ""),
        str(record.get("event_mode") or record.get("mode") or ""),
        str(record.get("direction") or ""),
    )


def collect_germs(paths: list[Path]) -> list[dict[str, Any]]:
    germs: list[dict[str, Any]] = []
    for path in paths:
        payload = load(path)
        rows = payload.get("germs") or payload.get("results") or ([payload] if payload.get("event_mode") or payload.get("mode") else [])
        for row in rows:
            if isinstance(row, dict):
                germ = dict(row)
                germ["source_artifact"] = str(path)
                germ["source_schema"] = payload.get("schema")
                germs.append(germ)
    return germs


def valid_germ(record: dict[str, Any]) -> bool:
    masses = record.get("masses")
    return bool(
        record.get("status") in {"traced", "verified", "passed"}
        and record.get("mixed_node") in MIXED_NODE_IDS
        and record.get("event_mode") in {"plus_one", "minus_one"}
        and record.get("direction") in {"+", "-"}
        and isinstance(masses, list)
        and len(masses) >= 2
        and record.get("source_artifact")
    )


def missing_mixed_germs(germs: list[dict[str, Any]]) -> list[str]:
    have = {germ_key(row) for row in germs if valid_germ(row)}
    missing: list[str] = []
    for node_id in MIXED_NODE_IDS:
        for mode, direction in REQUIRED_GERM_KEYS:
            key = (node_id, mode, direction)
            if key in have:
                continue
            # A germ that immediately ends on another classified node may omit
            # the opposite unused direction only if that exact key is present
            # with ends_on set; absence is still missing.
            missing.append(f"{node_id}:{mode}:{direction}")
    return missing


def valid_completeness_certificate(record: dict[str, Any] | None) -> bool:
    """Verify the completeness freezer schema and canonical content digest."""
    if not isinstance(record, dict):
        return False
    digest = record.get("sha256_content")
    if (
        record.get("schema") != "atlas.v1.completeness-certificate/1"
        or record.get("passed") is not True
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in string.hexdigits for character in digest)
    ):
        return False
    canonical_record = dict(record)
    canonical_record.pop("sha256_content", None)
    canonical = json.dumps(canonical_record, sort_keys=True, separators=(",", ":"))
    expected = hashlib.sha256(canonical.encode()).hexdigest()
    return digest.lower() == expected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="research/evidence/V1_CRITICAL_GRAPH.json")
    parser.add_argument("--roots")
    parser.add_argument("--left-birth")
    parser.add_argument("--right-death")
    parser.add_argument("--daughter")
    parser.add_argument("--germs", action="append", default=[])
    parser.add_argument("--completeness")
    parser.add_argument("--al-screen")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]

    nodes = headline_nodes(root)
    nodes.append(
        load_classification(
            Path(args.left_birth) if args.left_birth else None,
            node_id="secondary_left_birth",
            default_kind="endpoint",
            allowed=LEFT_BIRTH_CLASSES,
            missing_note="Need event-specific G- geometry: projection fold with nonzero quadratic, or a recorded falsification.",
        )
    )
    nodes.append(
        load_classification(
            Path(args.right_death) if args.right_death else None,
            node_id="secondary_right_death",
            default_kind="endpoint",
            allowed=RIGHT_DEATH_CLASSES,
            missing_note="Allowed classes: mixed_organizer, projection_fold, domain_boundary. Newton-failed is forbidden.",
        )
    )
    daughter_node = load_classification(
        Path(args.daughter) if args.daughter else None,
        node_id="lower_plus_one_daughter",
        default_kind="branch",
        allowed=DAUGHTER_CLASSES,
        missing_note=(
            "Classify the lower +1 daughter as reconnecting, closed_loop, distinct_branch, "
            "obstruction, no_branch_attachment, or falsified."
        ),
    )
    nodes.append(daughter_node)
    daughter_status = {
        "required_for_v1_graph": True,
        "status": daughter_node["status"],
        "class": daughter_node.get("mechanism"),
    }

    roots: list[dict[str, Any]] = []
    if args.roots:
        payload = load(Path(args.roots))
        roots = [row for row in payload.get("roots", []) if row.get("status") == "ok" or row.get("passed") is True]

    germs = collect_germs([Path(path) for path in args.germs])
    missing_germs = missing_mixed_germs(germs)

    completeness = None
    if args.completeness:
        completeness = load(Path(args.completeness))
    elif args.al_screen:
        al = load(Path(args.al_screen))
        accepted = al.get("accepted_candidates", [])
        stable = [row for row in accepted if row.get("corrected", {}).get("screening_stable")]
        completeness = {
            "passed": False,
            "note": "AL pocket screen only; neck raster and vertex harvest still required",
            "attempted": len(al.get("attempted", [])),
            "accepted": len(accepted),
            "screening_stable": len(stable),
        }

    newton_failed = [
        item
        for item in roots
        if forbidden_class(item.get("status")) or forbidden_class(item.get("error"))
    ]
    edges = polyline_edges(roots) if roots else []
    unclassified_edge_endpoints, classification_binding_errors = attach_edge_endpoints(
        edges, nodes, germs
    )
    cell_ids = [int(root["cell_id"]) for root in roots]
    assigned = [cell for edge in edges for cell in edge["cell_ids"]]
    duplicates = sorted({cell for cell in assigned if assigned.count(cell) > 1})
    coverage = {
        "source_transition_cells": 620,
        "localized_roots": len(roots),
        "required_cells": 620,
        "complete": len(roots) == 620 and set(cell_ids) == set(range(620)),
        "edge_count": len(edges),
        "cells_on_edges": len(assigned),
        "duplicate_cell_ids": duplicates,
        "missing_mixed_germs": missing_germs,
        "newton_failed": len(newton_failed),
        "completeness_passed": valid_completeness_certificate(completeness),
        "unclassified_edge_endpoints": unclassified_edge_endpoints,
        "classification_binding_errors": classification_binding_errors,
        "edge_topology_complete": not unclassified_edge_endpoints
        and not classification_binding_errors,
    }

    missing_required = [
        node_id
        for node_id in REQUIRED_HEADLINE_IDS
        if not any(item["id"] == node_id and item.get("passed") for item in nodes)
    ]
    mandatory_unresolved = {
        item["id"] for item in nodes if item["status"] in {"unresolved", "illegal"}
    }
    unexplained = sorted(mandatory_unresolved)
    illegal = [item["id"] for item in nodes if item["status"] == "illegal"]
    organizer_count = sum(1 for item in nodes if item.get("kind") == "mixed_organizer" and item.get("passed"))

    release_ready = (
        not missing_required
        and not unexplained
        and not illegal
        and coverage["complete"]
        and coverage["edge_count"] >= 1
        and coverage["cells_on_edges"] == 620
        and not duplicates
        and not missing_germs
        and not newton_failed
        and not unclassified_edge_endpoints
        and not classification_binding_errors
        and coverage["completeness_passed"]
        and all(item.get("passed") for item in nodes if item["id"] in REQUIRED_HEADLINE_IDS)
    )
    graph = {
        "schema": "atlas.v1.critical-graph/2",
        "claim_status": (
            "release_ready complete mechanism-resolved Floquet critical graph on the connected family sheet"
            if release_ready
            else "partial graph: 620 cells are samples, not edges; unresolved gates are enumerated by the artifact"
        ),
        "release_ready": release_ready,
        "family_component": "one continuation-connected Li-Li-Liao catalog sheet",
        "source_transition_cells": 620,
        "localized_roots": len(roots),
        "nodes": nodes,
        "edges": edges,
        "mixed_germs": [
            {
                "mixed_node": row.get("mixed_node") or row.get("node"),
                "event_mode": row.get("event_mode") or row.get("mode"),
                "direction": row.get("direction"),
                "ends_on": row.get("ends_on"),
                "masses": row.get("masses"),
                "status": row.get("status"),
                "source_artifact": row.get("source_artifact"),
                "valid": valid_germ(row),
            }
            for row in germs
        ],
        "root_coverage": coverage,
        "unexplained_nodes": unexplained,
        "missing_required_nodes": missing_required,
        "organizer_count": organizer_count,
        "daughter_classification": daughter_status,
        "completeness": completeness,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(graph, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(out),
                "release_ready": release_ready,
                "unexplained_nodes": unexplained,
                "localized_roots": coverage["localized_roots"],
                "edge_count": coverage["edge_count"],
                "missing_mixed_germs": coverage["missing_mixed_germs"],
            },
            indent=2,
        )
    )
    if not release_ready:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
