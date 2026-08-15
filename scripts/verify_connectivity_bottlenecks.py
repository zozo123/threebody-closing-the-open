#!/usr/bin/env python3
"""Forward/reverse shooting continuation across macroscopic graph bottlenecks.

Each selected edge is only one 0.001 mass-grid step, but its removal is chosen to
separate a large fraction of the minimum-spanning-tree catalog.  We correct both
published endpoints independently, walk the straight mass segment in warm-started
shooting substeps in both directions, and require the terminal orbit to match the
independently corrected opposite endpoint.

Passing these certificates is strong numerical evidence of local branch
connectivity across the graph cut; it is not a rigorous global proof.
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


def walk(start: FamilyPoint, target: FamilyPoint, *, substeps: int, max_residual: float) -> tuple[FamilyPoint, list[dict]]:
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
                f"continuation failed theta={theta:.6f} masses={masses} residual={point.residual_norm:.3e}"
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
    scale = np.maximum(np.maximum(np.abs(params(a)), np.abs(params(b))), np.asarray([0.05, 0.5, 0.1, 1.0]))
    return float(np.linalg.norm((params(a) - params(b)) / scale))


def certify(edge: dict, *, substeps: int, max_residual: float, match_tolerance: float) -> dict:
    raw_a, raw_b = from_record(edge["left"]), from_record(edge["right"])
    a, b = corrected(raw_a, max_residual), corrected(raw_b, max_residual)
    forward, forward_path = walk(a, b, substeps=substeps, max_residual=max_residual)
    reverse, reverse_path = walk(b, a, substeps=substeps, max_residual=max_residual)
    forward_match = normalized_chart_distance(forward, b)
    reverse_match = normalized_chart_distance(reverse, a)
    passed = forward_match <= match_tolerance and reverse_match <= match_tolerance
    if not passed:
        raise RuntimeError(
            "branch-hysteresis gate failed across bottleneck: "
            f"forward={forward_match:.3e} reverse={reverse_match:.3e} tol={match_tolerance:.3e}"
        )
    return {
        "graph_weight": edge["weight"],
        "smaller_partition_size": edge["smaller_partition_size"],
        "smaller_partition_fraction": edge["smaller_partition_fraction"],
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("graph_json")
    parser.add_argument("output")
    parser.add_argument("--substeps", type=int, default=6)
    parser.add_argument("--max-residual", type=float, default=2e-7)
    parser.add_argument("--match-tolerance", type=float, default=2e-5)
    args = parser.parse_args()

    graph = json.loads(Path(args.graph_json).read_text(encoding="utf-8"))
    certificates = [
        certify(
            edge,
            substeps=args.substeps,
            max_residual=args.max_residual,
            match_tolerance=args.match_tolerance,
        )
        for edge in graph["balanced_bottlenecks"]
    ]
    payload = {
        "claim_status": "screening-supported local continuation certificates; independent precision checks still required",
        "certificate_count": len(certificates),
        "all_passed": all(x["passed"] for x in certificates),
        "certificates": certificates,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "certificate_count": len(certificates),
        "all_passed": payload["all_passed"],
        "max_forward_match": max((x["forward_terminal_match"] for x in certificates), default=0.0),
        "max_reverse_match": max((x["reverse_terminal_match"] for x in certificates), default=0.0),
    }, indent=2))


if __name__ == "__main__":
    main()
