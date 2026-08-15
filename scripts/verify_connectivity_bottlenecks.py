#!/usr/bin/env python3
"""Forward/reverse shooting continuation across adversarial spanning-tree edges.

The full catalog mass-grid adjacency graph is connected.  This verifier attacks
two finite edge sets from its minimum spanning tree:

* balanced bottlenecks, chosen because deleting them separates macroscopic
  fractions of the sampled catalog;
* globally largest MST weights, chosen because they are the worst shooting-chart
  jumps anywhere in the spanning connection.

Each selected edge is only one 0.001 mass-grid step.  We correct both published
endpoints independently, walk the straight mass segment in warm-started shooting
substeps in both directions, and require the terminal orbit to match the
independently corrected opposite endpoint.

Passing these certificates is strong numerical evidence that the adversarial
links of the sampled spanning connection do not hide branch switches.  It is not
a rigorous proof about every unsampled point of the periodic-orbit moduli space.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from threebody_atlas.liao_family import FamilyPoint, correct_family_point


def from_record(record: dict) -> FamilyPoint:
    return FamilyPoint(
        masses=tuple(float(x) for x in record["masses"]),
        x1=float(record["x1"]),
        v1=float(record["v1"]),
        v2=float(record["v2"]),
        period=float(record["period"]),
        residual_norm=float("nan"),
        nfev=0,
        success=True,
    )


def params(p: FamilyPoint) -> np.ndarray:
    return np.asarray([p.x1, p.v1, p.v2, p.period], dtype=float)


def corrected(raw: FamilyPoint, max_residual: float) -> FamilyPoint:
    p = correct_family_point(raw.masses, tuple(params(raw)), max_nfev=80)
    if not p.success or p.residual_norm > max_residual:
        raise RuntimeError(f"endpoint correction failed at {raw.masses}: {p.residual_norm:.3e}")
    return p


def walk(
    start: FamilyPoint,
    target: FamilyPoint,
    *,
    substeps: int,
    max_residual: float,
) -> tuple[FamilyPoint, list[dict]]:
    m0 = np.asarray(start.masses, dtype=float)
    m1 = np.asarray(target.masses, dtype=float)
    current = start
    trajectory = []
    for k in range(1, substeps + 1):
        theta = k / substeps
        masses = tuple(float(x) for x in ((1.0 - theta) * m0 + theta * m1))
        point = correct_family_point(
            masses,
            (current.x1, current.v1, current.v2, current.period),
            max_nfev=80,
        )
        if not point.success or point.residual_norm > max_residual:
            raise RuntimeError(
                f"continuation failed theta={theta:.6f} masses={masses} "
                f"residual={point.residual_norm:.3e}"
            )
        trajectory.append(
            {
                "theta": theta,
                "masses": masses,
                "x1": point.x1,
                "v1": point.v1,
                "v2": point.v2,
                "period": point.period,
                "shooting_residual": point.residual_norm,
            }
        )
        current = point
    return current, trajectory


def normalized_chart_distance(a: FamilyPoint, b: FamilyPoint) -> float:
    scale = np.maximum(
        np.maximum(np.abs(params(a)), np.abs(params(b))),
        np.asarray([0.05, 0.5, 0.1, 1.0]),
    )
    return float(np.linalg.norm((params(a) - params(b)) / scale))


def certify(
    edge: dict,
    *,
    category: str,
    substeps: int,
    max_residual: float,
    match_tolerance: float,
) -> dict:
    raw_a, raw_b = from_record(edge["left"]), from_record(edge["right"])
    a, b = corrected(raw_a, max_residual), corrected(raw_b, max_residual)
    forward, forward_path = walk(a, b, substeps=substeps, max_residual=max_residual)
    reverse, reverse_path = walk(b, a, substeps=substeps, max_residual=max_residual)
    forward_match = normalized_chart_distance(forward, b)
    reverse_match = normalized_chart_distance(reverse, a)
    passed = forward_match <= match_tolerance and reverse_match <= match_tolerance
    if not passed:
        raise RuntimeError(
            "branch-hysteresis gate failed across adversarial MST edge: "
            f"forward={forward_match:.3e} reverse={reverse_match:.3e} "
            f"tol={match_tolerance:.3e}"
        )
    return {
        "category": category,
        "global_rank": edge.get("rank"),
        "graph_weight": edge["weight"],
        "smaller_partition_size": edge["smaller_partition_size"],
        "smaller_partition_fraction": edge["smaller_partition_fraction"],
        "left_index": edge["left"].get("index"),
        "right_index": edge["right"].get("index"),
        "left_masses": a.masses,
        "right_masses": b.masses,
        "left_corrected_residual": a.residual_norm,
        "right_corrected_residual": b.residual_norm,
        "forward_terminal_match": forward_match,
        "reverse_terminal_match": reverse_match,
        "substeps_each_direction": substeps,
        "passed": True,
        "forward_path": forward_path,
        "reverse_path": reverse_path,
    }


def edge_key(edge: dict) -> tuple[int, int]:
    a = int(edge["left"]["index"])
    b = int(edge["right"]["index"])
    return tuple(sorted((a, b)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("graph_json")
    parser.add_argument("output")
    parser.add_argument("--substeps", type=int, default=6)
    parser.add_argument("--max-residual", type=float, default=2e-7)
    parser.add_argument("--match-tolerance", type=float, default=2e-5)
    parser.add_argument(
        "--top-mst",
        type=int,
        default=20,
        help="also certify this many globally largest MST chart-jump edges",
    )
    args = parser.parse_args()
    if args.substeps < 1 or args.top_mst < 0:
        raise SystemExit("substeps must be positive and top-mst nonnegative")

    graph = json.loads(Path(args.graph_json).read_text(encoding="utf-8"))
    selected: list[tuple[str, dict]] = []
    used: set[tuple[int, int]] = set()

    for edge in graph["balanced_bottlenecks"]:
        key = edge_key(edge)
        if key not in used:
            used.add(key)
            selected.append(("balanced_bottleneck", edge))

    for edge in graph.get("global_top_mst_edges", [])[: args.top_mst]:
        key = edge_key(edge)
        if key not in used:
            used.add(key)
            selected.append(("global_top_mst", edge))

    certificates = []
    for i, (category, edge) in enumerate(selected, start=1):
        print(
            f"certifying {i}/{len(selected)} category={category} "
            f"weight={edge['weight']:.6g} "
            f"masses={edge['left']['masses']}->{edge['right']['masses']}",
            flush=True,
        )
        certificates.append(
            certify(
                edge,
                category=category,
                substeps=args.substeps,
                max_residual=args.max_residual,
                match_tolerance=args.match_tolerance,
            )
        )

    payload = {
        "claim_status": (
            "screening-supported bidirectional continuation certificates on the globally worst "
            "and macroscopic MST links; cross-chart/rank evidence remains part of the family gate"
        ),
        "mass_grid_adjacency_connected": bool(graph.get("mass_grid_adjacency_connected", False)),
        "catalog_rows": int(graph["rows"]),
        "mst_edges": int(graph.get("mst_edges", graph["rows"] - 1)),
        "requested_top_mst": args.top_mst,
        "certificate_count": len(certificates),
        "balanced_certificate_count": sum(
            x["category"] == "balanced_bottleneck" for x in certificates
        ),
        "global_top_certificate_count": sum(
            x["category"] == "global_top_mst" for x in certificates
        ),
        "all_passed": all(x["passed"] for x in certificates),
        "certificates": certificates,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "catalog_rows": payload["catalog_rows"],
                "mst_edges": payload["mst_edges"],
                "certificate_count": len(certificates),
                "balanced": payload["balanced_certificate_count"],
                "global_top": payload["global_top_certificate_count"],
                "all_passed": payload["all_passed"],
                "max_forward_match": max(
                    (x["forward_terminal_match"] for x in certificates), default=0.0
                ),
                "max_reverse_match": max(
                    (x["reverse_terminal_match"] for x in certificates), default=0.0
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
