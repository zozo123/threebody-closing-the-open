#!/usr/bin/env python3
"""Cross-chart certification for one globally worst MST connectivity edge.

The Li shooting chart is an efficient coordinate system, but a global family
claim should not depend on that specialization. This driver takes an edge from
the frozen mass-grid MST, corrects both endpoints independently, converts them
to the generic 8D translation-reduced strict-periodic formulation, and walks
the same mass segment in both directions without imposing the Li collinearity
or velocity ansatz.

Only predictor quality, Newton work, and the number of mass substeps may
increase. Closure, gauge, phase, and terminal-match gates are fixed across all
retries. A pass is finite-path cross-chart evidence; it is not by itself a
theorem about the unsampled moduli space.
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
        max_nfev=180,
        max_closure=max_residual,
        max_gauge=max_residual,
        max_phase=max_residual,
        rtol=1e-10,
        atol=1e-12,
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


def correction_score(point: GenericPeriodicPoint) -> float:
    return float(
        point.closure_norm
        + point.gauge_norm
        + abs(point.phase_residual)
    )


def correct_with_predictor_ensemble(
    masses: tuple[float, float, float],
    predictors: list[tuple[str, np.ndarray]],
    reference_state: np.ndarray,
    *,
    max_residual: float,
) -> tuple[GenericPeriodicPoint, str, list[dict]]:
    """Try branch-informed predictors without changing any acceptance gate."""
    trials: list[tuple[str, GenericPeriodicPoint]] = []
    diagnostics: list[dict] = []
    for label, predictor in predictors:
        candidate = correct_generic_periodic(
            masses,
            predictor[:8],
            float(predictor[8]),
            reference_state=reference_state,
            max_nfev=220,
            max_closure=max_residual,
            max_gauge=max_residual,
            max_phase=max_residual,
            rtol=1e-10,
            atol=1e-12,
        )
        diagnostics.append(
            {
                "predictor": label,
                "success": bool(candidate.success),
                "closure_norm": candidate.closure_norm,
                "gauge_norm": candidate.gauge_norm,
                "phase_residual": candidate.phase_residual,
                "nfev": candidate.nfev,
            }
        )
        if candidate.success:
            trials.append((label, candidate))
    if not trials:
        best = min(
            diagnostics,
            key=lambda item: (
                item["closure_norm"]
                + item["gauge_norm"]
                + abs(item["phase_residual"])
            ),
        )
        raise RuntimeError(
            "generic corrector predictor ensemble missed fixed gates; "
            f"best={best}"
        )
    label, point = min(trials, key=lambda item: correction_score(item[1]))
    return point, label, diagnostics


def walk(
    start: GenericPeriodicPoint,
    target: GenericPeriodicPoint,
    *,
    steps: int,
    max_residual: float,
) -> tuple[GenericPeriodicPoint, list[dict], dict[str, float]]:
    m0 = np.asarray(start.masses, dtype=float)
    m1 = np.asarray(target.masses, dtype=float)
    start_vector = start.vector.copy()
    target_vector = target.vector.copy()
    previous: GenericPeriodicPoint | None = None
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

        # Endpoint interpolation uses only independently generic-corrected
        # endpoint solutions.  It imposes no Li ansatz on an interior point.
        endpoint_linear = (1.0 - theta) * start_vector + theta * target_vector
        predictors: list[tuple[str, np.ndarray]] = []
        if previous is not None:
            # Equal-mass-step secant predictor is the standard local continuation
            # predictor and is tried first after two accepted generic points.
            predictors.append(("secant", current.vector + (current.vector - previous.vector)))
        predictors.append(("generic_endpoint_linear", endpoint_linear))
        predictors.append(("previous_point", current.vector.copy()))

        nxt, predictor_used, diagnostics = correct_with_predictor_ensemble(
            masses,
            predictors,
            reference,
            max_residual=max_residual,
        )
        previous, current = current, nxt
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
                "predictor_used": predictor_used,
                "predictor_trials": diagnostics,
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
        left, right, steps=steps, max_residual=max_residual
    )
    reverse, reverse_path, max_reverse = walk(
        right, left, steps=steps, max_residual=max_residual
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
            "and time-phase gauges; Li chart used only for endpoint warm starts; interior predictor "
            "ensemble uses generic endpoint interpolation and accepted-point secants"
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
