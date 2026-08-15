#!/usr/bin/env python3
"""Assemble the v1 mechanism-resolved Floquet critical graph.

620 catalog S/U cells are samples supporting the graph. They are not 620 edges.
An edge is a mechanism-specific polyline carrying a list of source-cell ids.
Endpoints, mixed germs, and completeness must come from artifacts.
The lower +1 daughter is deferred unless it is shown to change this graph.
The assembler never invents a classification and never flips
release_ready without the mandatory artifacts.
"""
from __future__ import annotations

import argparse
import json
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
    {"projection_fold", "two_separate_arcs", "mixed_organizer", "domain_boundary"}
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
    passed = bool(payload.get("passed")) and str(raw_class) in allowed
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
    )


def polyline_edges(roots: list[dict[str, Any]], *, jump: float = MASS_JUMP) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for root in roots:
        grouped[str(root.get("event_mode") or "unknown")].append(root)
    edges: list[dict[str, Any]] = []
    for mode, items in grouped.items():
        def key(item: dict[str, Any]) -> tuple[float, float, int]:
            masses = item.get("masses") or [0.0, 0.0]
            try:
                return float(masses[0]), float(masses[1]), int(item["cell_id"])
            except (TypeError, ValueError, KeyError, IndexError):
                return 0.0, 0.0, int(item.get("cell_id", -1))

        items = sorted(items, key=key)
        runs: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        previous: dict[str, Any] | None = None
        for item in items:
            if previous is None:
                current = [item]
            else:
                pm = previous.get("masses") or [0.0, 0.0]
                nm = item.get("masses") or [0.0, 0.0]
                try:
                    dist = ((float(pm[0]) - float(nm[0])) ** 2 + (float(pm[1]) - float(nm[1])) ** 2) ** 0.5
                except (TypeError, ValueError, IndexError):
                    dist = jump + 1.0
                if dist > jump:
                    runs.append(current)
                    current = [item]
                else:
                    current.append(item)
            previous = item
        if current:
            runs.append(current)
        for index, run in enumerate(runs):
            cell_ids = [int(row["cell_id"]) for row in run]
            edges.append(
                {
                    "id": f"{mode}_{index}",
                    "kind": "mechanism_polyline",
                    "mechanism": mode,
                    "cell_ids": cell_ids,
                    "source_cell_count": len(cell_ids),
                    "estimators": sorted({str(row.get("estimator") or "float64") for row in run}),
                    "endpoints": {
                        "start_cell": cell_ids[0],
                        "end_cell": cell_ids[-1],
                        "classified": False,
                    },
                }
            )
    edges.sort(key=lambda edge: (str(edge["mechanism"]), edge["cell_ids"][0]))
    return edges


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
                germs.append(row)
    return germs


def missing_mixed_germs(germs: list[dict[str, Any]]) -> list[str]:
    have = {germ_key(row) for row in germs}
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
    if args.daughter:
        daughter_node = load_classification(
            Path(args.daughter),
            node_id="lower_plus_one_daughter",
            default_kind="branch",
            allowed=DAUGHTER_CLASSES,
            missing_note="Optional follow-up: not required to close the v1 critical graph.",
        )
        nodes.append(daughter_node)
        daughter_status = {
            "required_for_v1_graph": False,
            "status": daughter_node["status"],
            "class": daughter_node.get("mechanism"),
        }
    else:
        daughter_status = {
            "required_for_v1_graph": False,
            "status": "deferred_not_required_for_v1_graph",
            "class": None,
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
        "completeness_passed": bool(completeness and completeness.get("passed")),
    }

    missing_required = [
        node_id
        for node_id in REQUIRED_HEADLINE_IDS
        if not any(item["id"] == node_id and item.get("passed") for item in nodes)
    ]
    mandatory_unresolved = {
        item["id"]
        for item in nodes
        if item["id"] != "lower_plus_one_daughter" and item["status"] in {"unresolved", "illegal"}
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
        and coverage["completeness_passed"]
        and all(item.get("passed") for item in nodes if item["id"] in REQUIRED_HEADLINE_IDS)
    )
    graph = {
        "schema": "atlas.v1.critical-graph/2",
        "claim_status": (
            "release_ready complete mechanism-resolved Floquet critical graph on the connected family sheet"
            if release_ready
            else "partial graph: headline nodes frozen; 620 cells are samples, not edges; endpoints/germs/completeness still open"
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
                "status": row.get("status"),
            }
            for row in germs
        ],
        "root_coverage": coverage,
        "unexplained_nodes": unexplained,
        "missing_required_nodes": missing_required,
        "organizer_count": organizer_count,
        "daughter_classification": daughter_status,
        "completeness": completeness,
        "provisional_components": [
            "principal lower: +1 -> mixed -> -1 -> mixed -> +1",
            "principal upper: Delta=0 Hamiltonian-Hopf",
            "secondary lobe: unresolved left birth -> mixed -> unresolved right death",
        ],
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
