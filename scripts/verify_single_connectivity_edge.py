#!/usr/bin/env python3
"""Adaptively falsify one globally worst MST continuation edge.

The full-catalog adjacency MST is only a discrete diagnostic.  A large chart jump
must therefore be attacked by actual branch-preserving shooting.  This driver
selects one ranked global MST edge and repeats the same bidirectional certificate
with progressively finer mass substeps.  Acceptance thresholds are never
loosened: only the predictor step size changes.

A failure is written to the JSON artifact before the process exits nonzero so a
hard edge cannot erase the evidence from easier retry levels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from verify_connectivity_bottlenecks import certify


def retry_schedule(initial: int, maximum: int) -> list[int]:
    if initial < 1 or maximum < initial:
        raise ValueError("require 1 <= initial-substeps <= max-substeps")
    values = []
    current = initial
    while current < maximum:
        values.append(current)
        current *= 2
    values.append(maximum)
    return sorted(set(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("graph_json")
    parser.add_argument("output")
    parser.add_argument("--global-rank", type=int, required=True)
    parser.add_argument("--initial-substeps", type=int, default=6)
    parser.add_argument("--max-substeps", type=int, default=48)
    parser.add_argument("--max-residual", type=float, default=2e-7)
    parser.add_argument("--match-tolerance", type=float, default=2e-5)
    args = parser.parse_args()

    graph = json.loads(Path(args.graph_json).read_text(encoding="utf-8"))
    edges = graph.get("global_top_mst_edges", [])
    if not 1 <= args.global_rank <= len(edges):
        raise SystemExit(
            f"global rank {args.global_rank} is outside available 1..{len(edges)}"
        )
    edge = edges[args.global_rank - 1]
    attempts = []
    certificate = None
    for substeps in retry_schedule(args.initial_substeps, args.max_substeps):
        print(
            f"global_rank={args.global_rank} trying substeps={substeps} "
            f"masses={edge['left']['masses']}->{edge['right']['masses']}",
            flush=True,
        )
        try:
            certificate = certify(
                edge,
                category="global_top_mst",
                substeps=substeps,
                max_residual=args.max_residual,
                match_tolerance=args.match_tolerance,
            )
        except RuntimeError as exc:
            attempts.append({"substeps": substeps, "passed": False, "error": str(exc)})
            continue
        attempts.append({"substeps": substeps, "passed": True})
        break

    payload = {
        "claim_status": (
            "adaptive float64 bidirectional certificate for one adversarial MST edge; "
            "the residual and terminal-match gates are fixed across retries"
        ),
        "catalog_rows": int(graph["rows"]),
        "mass_grid_adjacency_connected": bool(graph.get("mass_grid_adjacency_connected", False)),
        "global_rank": args.global_rank,
        "graph_weight": edge["weight"],
        "left": edge["left"],
        "right": edge["right"],
        "attempts": attempts,
        "passed": certificate is not None,
        "certificate": certificate,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "global_rank": args.global_rank,
                "weight": edge["weight"],
                "passed": payload["passed"],
                "attempts": attempts,
                "accepted_substeps": None
                if certificate is None
                else certificate["substeps_each_direction"],
            },
            indent=2,
        )
    )
    if certificate is None:
        raise SystemExit(
            f"global MST rank {args.global_rank} failed all adaptive substep retries"
        )


if __name__ == "__main__":
    main()
