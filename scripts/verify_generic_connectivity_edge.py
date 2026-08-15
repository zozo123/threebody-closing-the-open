#!/usr/bin/env python3
"""Cross-chart certification for one globally worst MST connectivity edge.

The Li shooting chart is an efficient coordinate system, but a global family
claim should not depend on that specialization. This driver takes an edge from
the frozen mass-grid MST, corrects both endpoints independently, converts them
to the generic 8D translation-reduced strict-periodic formulation, and walks
the same mass segment in both directions without imposing the Li collinearity
or velocity ansatz.

Only the number of mass substeps may increase. Closure, gauge, phase, and
terminal-match gates are fixed across retries. A pass is finite-path
cross-chart evidence; it is not by itself a theorem about the unsampled moduli
space.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from threebody_atlas.generic_periodic import GenericPeriodicPoint, correct_generic_periodic
from threebody_atlas.liao_family import correct_family_point, state_from_chart
from threebody_atlas.reduced import full_to_reduced


def retry_schedule(initial: int, maximum: int) -> list[int]:
    if initial < 2 or maximum < initial:
        raise ValueError("require 2 <= initial-substeps <= max-substeps")
    values: list[int] = []
    current = initial
    while current < maximum:
        values.append(current)
        current *= 2
    values.append(maximum)
    return sorted(set(values))


def endpoint_from_serialized(row: dict, max_residual: float) -> GenericPeriodicPoint:
    masses = tuple(float(x) for x in row["masses"])
    li = correct_family_point(
        masses,
        (float(row["x1"]), float(row["v1"]), float(row["v2"]), float(row["period"])),
        max_nfev=90,
    )
    if not li.success or li.residual_norm > max_residual:
        raise RuntimeError(f"Li warm-start endpoint correction failed: {li.residual_norm:.3e}")
    reduced = full_to_reduced(state_from_chart(li.masses, li.x1, li.v1, li.v2))
    generic = correct_generic_periodic(
        li.masses,
        reduced,
        li.period,
        reference_state=reduced,
        max_nfev=100,
        max_closure=max_residual,
        max_gauge=max_residual,
        max_phase=max_residual,
    )
    if not generic.success:
        raise RuntimeError(
            "generic endpoint correction failed: "
            f"closure={generic.closure_norm:.3e} gauge={generic.gauge_norm:.3e} "
            f"phase={generic.phase_residual:.3e}"
        )
    return generic


def distance(a: GenericPeriodicPoint, b: GenericPeriodicPoint) -> float:
    va, vb = a.vector, b.vector
    floors = np.asarray([0.2, 0.2, 0.5, 0.2, 0.5, 0.5, 0.5, 0.5, 1.0])
    scale = np.maximum(np.maximum(np.abs(va), np.abs(vb)), floors)
    return float(np.linalg.norm((va - vb) / scale))


def off_li_norm(point: GenericPeriodicPoint) -> float:
    z = np.asarray(point.state, dtype=float)
    return float(np.linalg.norm(z[[1, 4, 6]]))


def walk(
    start: GenericPeriodicPoint,
    target_masses: tuple[float, float, float],
    *,
    steps: int,
    max_residual: float,
) -> tuple[GenericPeriodicPoint, list[dict], dict[str, float]]:
    m0 = np.asarray(start.masses, dtype=float)
    m1 = np.asarray(target_masses, dtype=float)
    current = start
    path: list[dict] = []
    maxima = {
        "closure": float(current.closure_norm),
        "gauge": float(current.gauge_norm),
        "phase": abs(float(current.phase_residual)),
        "off_li": off_li_norm(current),
    }
    for k in range(1, steps + 1):
        theta = k / steps
        masses = tuple(float(x) for x in ((1.0 - theta) * m0 + theta * m1))
        reference = np.asarray(current.state, dtype=float)
        nxt = correct_generic_periodic(
            masses,
            reference,
            current.period,
            reference_state=reference,
            max_nfev=90,
            max_closure=max_residual,
            max_gauge=max_residual,
            max_phase=max_residual,
        )
        if not nxt.success:
            raise RuntimeError(
                f"generic continuation failed at theta={theta:.8f}: "
                f"closure={nxt.closure_norm:.3e} gauge={nxt.gauge_norm:.3e} "
                f"phase={nxt.phase_residual:.3e}"
            )
        current = nxt
        maxima["closure"] = max(maxima["closure"], float(current.closure_norm))
        maxima["gauge"] = max(maxima["gauge"], float(current.gauge_norm))
        maxima["phase"] = max(maxima["phase"], abs(float(current.phase_residual)))
        maxima["off_li"] = max(maxima["off_li"], off_li_norm(current))
        path.append(
            {
                "theta": theta,
                "masses": list(current.masses),
                "period": current.period,
                "closure_norm": current.closure_norm,
                "gauge_norm": current.gauge_norm,
                "phase_residual": current.phase_residual,
                "off_li_norm": off_li_norm(current),
                "nfev": current.nfev,
            }
        )
    return current, path, maxima


def certify(
    edge: dict,
    *,
    steps: int,
    max_residual: float,
    match_tolerance: float,
) -> dict:
    left = endpoint_from_serialized(edge["left"], max_residual)
    right = endpoint_from_serialized(edge["right"], max_residual)
    forward, forward_path, max_forward = walk(
        left, right.masses, steps=steps, max_residual=max_residual
    )
    reverse, reverse_path, max_reverse = walk(
        right, left.masses, steps=steps, max_residual=max_residual
    )
    forward_match = distance(forward, right)
    reverse_match = distance(reverse, left)
    if forward_match > match_tolerance or reverse_match > match_tolerance:
        raise RuntimeError(
            "generic terminal hysteresis gate failed: "
            f"forward={forward_match:.3e} reverse={reverse_match:.3e}"
        )
    return {
        "formulation": (
            "8D translation-reduced strict periodic single shooting with local scale, rotation, "
            "and time-phase gauges; Li chart used only for endpoint warm starts"
        ),
        "substeps_each_direction": steps,
        "left_endpoint": {
            "masses": list(left.masses),
            "closure_norm": left.closure_norm,
            "gauge_norm": left.gauge_norm,
            "phase_residual": left.phase_residual,
        },
        "right_endpoint": {
            "masses": list(right.masses),
            "closure_norm": right.closure_norm,
            "gauge_norm": right.gauge_norm,
            "phase_residual": right.phase_residual,
        },
        "forward_maxima": max_forward,
        "reverse_maxima": max_reverse,
        "forward_terminal_match": forward_match,
        "reverse_terminal_match": reverse_match,
        "forward_path": forward_path,
        "reverse_path": reverse_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("graph_json")
    parser.add_argument("output")
    parser.add_argument("--global-rank", type=int, required=True)
    parser.add_argument("--initial-substeps", type=int, default=12)
    parser.add_argument("--max-substeps", type=int, default=48)
    parser.add_argument("--max-residual", type=float, default=2e-7)
    parser.add_argument("--match-tolerance", type=float, default=2e-5)
    args = parser.parse_args()

    graph = json.loads(Path(args.graph_json).read_text(encoding="utf-8"))
    edges = graph.get("global_top_mst_edges", [])
    if not 1 <= args.global_rank <= len(edges):
        raise SystemExit(f"global rank {args.global_rank} outside 1..{len(edges)}")
    edge = edges[args.global_rank - 1]

    attempts: list[dict] = []
    certificate: dict | None = None
    for steps in retry_schedule(args.initial_substeps, args.max_substeps):
        try:
            certificate = certify(
                edge,
                steps=steps,
                max_residual=args.max_residual,
                match_tolerance=args.match_tolerance,
            )
        except RuntimeError as exc:
            attempts.append({"substeps": steps, "passed": False, "error": str(exc)})
            continue
        attempts.append({"substeps": steps, "passed": True})
        break

    payload = {
        "claim_status": (
            "cross-chart float64 bidirectional continuation evidence for one adversarial MST edge; "
            "all numerical gates are fixed across adaptive step retries"
        ),
        "global_rank": args.global_rank,
        "graph_weight": edge["weight"],
        "left": edge["left"],
        "right": edge["right"],
        "max_residual_gate": args.max_residual,
        "terminal_match_gate": args.match_tolerance,
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
        raise SystemExit(f"generic MST rank {args.global_rank} failed all fixed-gate retries")


if __name__ == "__main__":
    main()
