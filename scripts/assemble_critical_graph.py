#!/usr/bin/env python3
"""Assemble the v1 Floquet critical graph from frozen evidence.

This is the Gate B object.  It refuses release_ready until every required node
class is present and, if a 620-cell root file is supplied, every cell is
assigned.  Missing endpoints stay explicit.  Screening tracks are not promoted
to release edges.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_NODE_IDS = (
    "mixed_principal_left",
    "mixed_secondary_left",
    "mixed_principal_right",
    "headline_lower_plus_one",
    "headline_upper_collision",
)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def node(node_id: str, kind: str, *, status: str, masses: list[Any] | None = None, **extra: Any) -> dict[str, Any]:
    record = {"id": node_id, "kind": kind, "status": status, "masses": masses}
    record.update(extra)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="research/evidence/V1_CRITICAL_GRAPH.json")
    parser.add_argument("--roots")
    parser.add_argument("--al-screen")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]

    mixed_left = load(root / "research/evidence/V1_MIXED_CANONICAL_PRINCIPAL_LEFT_2026-08-15.json")
    mixed_sec = load(root / "research/evidence/V1_MIXED_CANONICAL_SECONDARY_LEFT_2026-08-15.json")
    mixed_right = load(root / "research/evidence/V1_MIXED_CANONICAL_PRINCIPAL_RIGHT_2026-08-15.json")
    plus_one = load(root / "research/evidence/V1_CANONICAL_LOWER_PLUS_ONE_2026-08-15.json")
    collision = load(root / "research/evidence/V1_CANONICAL_UPPER_COLLISION_2026-08-15.json")

    nodes = [
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
        node(
            "secondary_left_fold",
            "projection_fold",
            status="unresolved",
            masses=[0.995705, 0.97424, 1.0],
            mechanism="minus_one_m1_fold_candidate",
            note="Root-count screen exists; event-specific geometry and BigFloat nondegeneracy are still required.",
        ),
        node(
            "secondary_right_death",
            "endpoint",
            status="unresolved",
            masses=[1.04306, 1.04640, 1.0],
            mechanism="unknown",
            note="Allowed classes: mixed_organizer, projection_fold, domain_boundary. Newton-failed is forbidden.",
        ),
        node(
            "lower_plus_one_daughter",
            "branch",
            status="unresolved",
            masses=None,
            mechanism="physical_soft_plus_one_daughter_candidate",
            note="Float64 minus continuation exists; independent BigFloat of d0-minus is required before classification.",
        ),
    ]

    roots: list[dict[str, Any]] = []
    if args.roots:
        payload = load(Path(args.roots))
        roots = list(payload.get("roots", []))

    al_note = None
    if args.al_screen:
        al = load(Path(args.al_screen))
        accepted = al.get("accepted_candidates", [])
        stable = [row for row in accepted if row.get("corrected", {}).get("screening_stable")]
        al_note = {
            "attempted": len(al.get("attempted", [])),
            "accepted": len(accepted),
            "screening_stable": len(stable),
            "interpretation": (
                "off-grid proposals corrected onto the known sheet; no hidden stable pocket in this sample"
                if accepted and not stable
                else "inspect accepted stable points before changing the graph"
            ),
        }

    missing_required = [node_id for node_id in REQUIRED_NODE_IDS if not any(n["id"] == node_id and n.get("passed") for n in nodes)]
    unexplained = [n["id"] for n in nodes if n["status"] == "unresolved"]
    coverage = {
        "supplied_roots": len(roots),
        "required_cells": 620,
        "complete": len(roots) == 620 and {int(r["cell_id"]) for r in roots} == set(range(620)),
    }
    release_ready = (
        not missing_required
        and not unexplained
        and coverage["complete"]
        and all(n.get("passed") for n in nodes if n["id"] in REQUIRED_NODE_IDS)
    )
    graph = {
        "schema": "atlas.v1.critical-graph/1",
        "claim_status": (
            "release_ready connected Floquet critical graph"
            if release_ready
            else "partial graph: headline nodes frozen; edges/endpoints still open"
        ),
        "release_ready": release_ready,
        "family_component": "one continuation-connected Li-Li-Liao catalog sheet",
        "nodes": nodes,
        "edges": [],
        "root_coverage": coverage,
        "unexplained_nodes": unexplained,
        "missing_required_nodes": missing_required,
        "completeness_screen": al_note,
        "provisional_components": [
            "principal lower: +1 -> mixed -> -1 -> mixed -> +1",
            "principal upper: Delta=0 Hamiltonian-Hopf",
            "secondary lobe: unresolved left fold -> mixed -> unresolved right death",
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
                "supplied_roots": coverage["supplied_roots"],
            },
            indent=2,
        )
    )
    if not release_ready:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
