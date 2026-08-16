#!/usr/bin/env python3
"""Resolve one coarse Floquet mechanism switch by event-specific continuation.

The global 620-cell audit found three places where adjacent S/U cells change
unique smooth event label.  A mechanism switch in a coarse grid is not itself a
codimension-two vertex: it can also be produced by a fold of one event curve,
separate nearby arcs, or multiple roots hidden inside a cell.

This driver attacks one small mass window.  It

1. classifies only the transition brackets in that window;
2. selects the nearest cross-mechanism adjacency edge;
3. seeds each mechanism from two same-event brackets on its own side;
4. traces both smooth event curves by six-variable pseudo-arclength *toward*
   the coarse junction;
5. screens the traces for projection folds; and
6. only then retries the direct mixed ``(alpha,beta)=(4,4)`` solve from the
   closest traced spectral seed.

All output is float64 screening evidence.  Any accepted organizer must still be
independently reproduced at BigFloat/canonical precision.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from threebody_atlas.boundary import evaluate
from threebody_atlas.critical_manifold import event_value, localize_critical_point, trace_augmented_critical
from threebody_atlas.hybrid_vertices import solve_direct_vertex
from threebody_atlas.liao_family import FamilyPoint

MODES = ("plus_one", "minus_one", "trace_collision")


def point(row: dict[str, str], side: str) -> FamilyPoint:
    return FamilyPoint(
        masses=(float(row["m1"]), float(row[f"{side}_m2"]), float(row["m3"])),
        x1=float(row[f"{side}_x1"]),
        v1=float(row[f"{side}_v1"]),
        v2=float(row[f"{side}_v2"]),
        period=float(row[f"{side}_period"]),
        residual_norm=float("nan"),
        nfev=0,
        success=True,
    )


def midpoint_m2(row: dict[str, str]) -> float:
    return 0.5 * (float(row["left_m2"]) + float(row["right_m2"]))


def orientation(row: dict[str, str]) -> str:
    return f'{row["left_label"]}->{row["right_label"]}'


def sign_changing_modes(row: dict[str, str]) -> tuple[list[str], dict[str, list[float]]]:
    left = evaluate(point(row, "left"))
    right = evaluate(point(row, "right"))
    values = {
        mode: [event_value(left.floquet, mode), event_value(right.floquet, mode)]
        for mode in MODES
    }
    modes = [
        mode
        for mode, (a, b) in values.items()
        if a == 0.0 or b == 0.0 or a * b < 0.0
    ]
    return modes, values


def localize(row: dict[str, str], mode: str):
    left, right = point(row, "left"), point(row, "right")
    stable = left if row["left_label"] == "S" else right
    unstable = left if row["left_label"] == "U" else right
    return localize_critical_point(
        stable,
        unstable,
        event_mode=mode,
        m2_tolerance=2e-9,
        event_tolerance=2e-8,
        max_iterations=40,
        max_closure=1e-7,
    )


def serialize_localized(p) -> dict[str, Any]:
    q, f = p.sample.point, p.sample.floquet
    return {
        "masses": [float(x) for x in q.masses],
        "x1": float(q.x1),
        "v1": float(q.v1),
        "v2": float(q.v2),
        "period": float(q.period),
        "shooting_residual": float(q.residual_norm),
        "event_mode": p.event_mode,
        "event_value": float(p.event_value),
        "alpha": float(f.alpha),
        "beta": float(f.beta),
        "discriminant": float(f.discriminant),
        "trace_roots": [[float(z.real), float(z.imag)] for z in f.trace_roots],
    }


def serialize_step(p) -> dict[str, Any]:
    q, f = p.sample.point, p.sample.floquet
    return {
        "masses": [float(x) for x in q.masses],
        "x1": float(q.x1),
        "v1": float(q.v1),
        "v2": float(q.v2),
        "period": float(q.period),
        "shooting_residual": float(q.residual_norm),
        "event_mode": p.event_mode,
        "event_value": float(p.event_value),
        "alpha": float(f.alpha),
        "beta": float(f.beta),
        "discriminant": float(f.discriminant),
        "trace_roots": [[float(z.real), float(z.imag)] for z in f.trace_roots],
        "scaled_tangent": [float(x) for x in p.tangent_scaled],
        "arclength_residual": float(p.arclength_residual),
        "normalized_step": float(p.normalized_step),
        "nfev": int(p.nfev),
    }


def distance_to_mixed(record: dict[str, Any]) -> float:
    return float(np.hypot(float(record["alpha"]) - 4.0, float(record["beta"]) - 4.0))


def fold_screens(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Find oriented tangent sign changes in the mass projection."""
    screens: list[dict[str, Any]] = []
    for parameter, index, transverse_index in (("m1", 4, 5), ("m2", 5, 4)):
        for i in range(1, len(points)):
            ta = points[i - 1].get("scaled_tangent")
            tb = points[i].get("scaled_tangent")
            if ta is None or tb is None:
                continue
            a, b = float(ta[index]), float(tb[index])
            if a == 0.0 or b == 0.0 or a * b < 0.0:
                transverse = min(abs(float(ta[transverse_index])), abs(float(tb[transverse_index])))
                screens.append(
                    {
                        "parameter": parameter,
                        "point_indices": [i - 1, i],
                        "tangent_values": [a, b],
                        "minimum_transverse_tangent": transverse,
                        "generic_screen": bool(transverse >= 1e-4),
                    }
                )
    return screens


def nearest_cross_mode_edge(
    records: list[dict[str, Any]],
    center: np.ndarray,
    link_threshold: float,
    *,
    allow_opposite_orientations: bool = False,
):
    candidates = []
    for i, a in enumerate(records):
        if a["mode"] is None:
            continue
        for j in range(i + 1, len(records)):
            b = records[j]
            if (
                b["mode"] is None
                or (
                    not allow_opposite_orientations
                    and a["orientation"] != b["orientation"]
                )
                or a["mode"] == b["mode"]
            ):
                continue
            if abs(a["m1"] - b["m1"]) > 0.0015:
                continue
            if abs(a["m2"] - b["m2"]) > link_threshold:
                continue
            midpoint = np.asarray([(a["m1"] + b["m1"]) / 2.0, (a["m2"] + b["m2"]) / 2.0])
            candidates.append((float(np.linalg.norm(midpoint - center)), i, j))
    if not candidates:
        raise RuntimeError("no adjacent unique cross-mechanism edge found in requested window")
    _, i, j = min(candidates)
    return i, j


def same_mode_far_seed(
    records: list[dict[str, Any]],
    near_index: int,
    other_index: int,
    *,
    link_threshold: float,
) -> int:
    near, other = records[near_index], records[other_index]
    direction = near["m1"] - other["m1"]
    candidates = []
    for i, candidate in enumerate(records):
        if i in (near_index, other_index):
            continue
        if candidate["mode"] != near["mode"] or candidate["orientation"] != near["orientation"]:
            continue
        dm1 = candidate["m1"] - near["m1"]
        dm2 = candidate["m2"] - near["m2"]
        if abs(dm1) > 0.0041 or abs(dm2) > max(0.03, 1.5 * link_threshold):
            continue
        away = bool(direction == 0.0 or dm1 * direction > 0.0)
        score = (0 if away else 1, abs(dm1) + 0.25 * abs(dm2), i)
        candidates.append((score, i))
    if not candidates:
        raise RuntimeError(f"no same-mode far seed found for {near['mode']} side")
    return min(candidates)[1]


def closest_mass_gap(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> dict[str, Any]:
    best = None
    for ia, pa in enumerate(a):
        ma = np.asarray(pa["masses"][:2], dtype=float)
        for ib, pb in enumerate(b):
            mb = np.asarray(pb["masses"][:2], dtype=float)
            gap = float(np.linalg.norm(ma - mb))
            if best is None or gap < best[0]:
                best = (gap, ia, ib, ma, mb)
    if best is None:
        return {"distance": None}
    return {
        "distance": best[0],
        "indices": [best[1], best[2]],
        "masses": [best[3].tolist(), best[4].tolist()],
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("brackets_tsv")
    parser.add_argument("output")
    parser.add_argument("--center-m1", type=float)
    parser.add_argument("--center-m2", type=float)
    parser.add_argument(
        "--center-json",
        help="passed organizer classification whose masses define the trace center",
    )
    parser.add_argument("--mixed-node")
    parser.add_argument("--allow-opposite-orientations", action="store_true")
    parser.add_argument("--skip-direct", action="store_true")
    parser.add_argument("--radius-m1", type=float, default=0.006)
    parser.add_argument("--radius-m2", type=float, default=0.035)
    parser.add_argument("--link-threshold", type=float, default=0.02)
    parser.add_argument("--steps", type=int, default=14)
    parser.add_argument("--arclength-step", type=float, default=1.5e-3)
    parser.add_argument("--direct-mass-padding", type=float, default=0.012)
    args = parser.parse_args()

    if args.center_json:
        center_record = json.loads(Path(args.center_json).read_text(encoding="utf-8"))
        center_masses = center_record.get("masses") or []
        if center_record.get("passed") is not True or len(center_masses) < 2:
            raise SystemExit("--center-json must be a passed organizer record with masses")
        center_m1, center_m2 = float(center_masses[0]), float(center_masses[1])
    elif args.center_m1 is not None and args.center_m2 is not None:
        center_m1, center_m2 = float(args.center_m1), float(args.center_m2)
    else:
        raise SystemExit("provide --center-json or both --center-m1 and --center-m2")
    center = np.asarray([center_m1, center_m2], dtype=float)
    with Path(args.brackets_tsv).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    window_rows = [
        row
        for row in rows
        if abs(float(row["m1"]) - center_m1) <= args.radius_m1
        and abs(midpoint_m2(row) - center_m2) <= args.radius_m2
    ]
    if len(window_rows) < 4:
        raise RuntimeError(f"junction window contains too few transition brackets: {len(window_rows)}")

    records: list[dict[str, Any]] = []
    for row in window_rows:
        modes, values = sign_changing_modes(row)
        records.append(
            {
                "m1": float(row["m1"]),
                "m2": midpoint_m2(row),
                "orientation": orientation(row),
                "modes": modes,
                "mode": modes[0] if len(modes) == 1 else None,
                "event_endpoint_values": values,
                "row": row,
            }
        )
    records.sort(key=lambda r: (r["m1"], r["m2"]))

    left_index, right_index = nearest_cross_mode_edge(
        records,
        center,
        args.link_threshold,
        allow_opposite_orientations=args.allow_opposite_orientations,
    )
    edge_records = [records[left_index], records[right_index]]
    traces = []
    for near_index, other_index in ((left_index, right_index), (right_index, left_index)):
        near = records[near_index]
        far_index = same_mode_far_seed(
            records,
            near_index,
            other_index,
            link_threshold=args.link_threshold,
        )
        far = records[far_index]
        first = localize(far["row"], near["mode"])
        second = localize(near["row"], near["mode"])
        trace = trace_augmented_critical(
            first,
            second,
            steps=args.steps,
            normalized_step=args.arclength_step,
        )
        trace_points = [serialize_step(p) for p in trace.points]
        traces.append(
            {
                "event_mode": near["mode"],
                "coarse_seed_cells": [
                    {"m1": far["m1"], "m2": far["m2"]},
                    {"m1": near["m1"], "m2": near["m2"]},
                ],
                "localized_seeds": [serialize_localized(first), serialize_localized(second)],
                "points": trace_points,
                "stopped_reason": trace.stopped_reason,
                "fold_screens": fold_screens(trace_points),
            }
        )
        write_json_atomic(
            Path(args.output),
            {
                "claim_status": "partial event-specific continuation checkpoint; classification forbidden",
                "requested_center": [center_m1, center_m2],
                "mixed_node": args.mixed_node,
                "traces": traces,
                "completed": False,
            },
        )

    combined = []
    for trace in traces:
        combined.extend(trace["localized_seeds"])
        combined.extend(trace["points"])
    closest_mixed = min(combined, key=distance_to_mixed)
    mixed_seed_distance = distance_to_mixed(closest_mixed)
    direct_result: dict[str, Any]
    edge_modes = {trace["event_mode"] for trace in traces}
    if edge_modes == {"plus_one", "minus_one"} and not args.skip_direct:
        seed = np.asarray(
            [
                closest_mixed["x1"],
                closest_mixed["v1"],
                closest_mixed["v2"],
                closest_mixed["period"],
                closest_mixed["masses"][0],
                closest_mixed["masses"][1],
            ],
            dtype=float,
        )
        try:
            direct = solve_direct_vertex(
                seed,
                "mixed_plus_minus_one",
                m3=float(closest_mixed["masses"][2]),
                mass_bounds=(
                    (center_m1 - args.direct_mass_padding, center_m1 + args.direct_mass_padding),
                    (center_m2 - args.direct_mass_padding, center_m2 + args.direct_mass_padding),
                ),
                max_nfev=60,
            )
            direct_result = {
                "status": "accepted_screening_candidate",
                "masses": [float(x) for x in direct.point.masses],
                "x1": float(direct.point.x1),
                "v1": float(direct.point.v1),
                "v2": float(direct.point.v2),
                "period": float(direct.point.period),
                "shooting_residual": float(direct.point.residual_norm),
                "alpha": float(direct.alpha),
                "beta": float(direct.beta),
                "event_values": [float(x) for x in direct.event_values],
                "invariant_error": float(direct.invariant_error),
                "nfev": int(direct.nfev),
            }
        except Exception as exc:
            direct_result = {"status": "not_accepted", "error": f"{type(exc).__name__}: {exc}"}
    elif args.skip_direct:
        direct_result = {
            "status": "skipped_by_request",
            "reason": "canonical organizer was supplied separately; this run certifies local continuation germs",
        }
    else:
        direct_result = {"status": "not_applicable", "edge_modes": sorted(edge_modes)}

    side_points = [trace["localized_seeds"] + trace["points"] for trace in traces]
    mass_gap = closest_mass_gap(side_points[0], side_points[1])
    any_fold = any(
        screen.get("generic_screen")
        for trace in traces
        for screen in trace["fold_screens"]
    )
    if direct_result["status"] == "accepted_screening_candidate":
        classification = "mixed_plus_minus_one_vertex_candidate"
    elif any_fold:
        classification = "event_curve_projection_fold_candidate"
    elif mass_gap.get("distance") is not None and mass_gap["distance"] <= 5e-4 and mixed_seed_distance <= 1e-2:
        classification = "mixed_vertex_approach_unresolved"
    else:
        classification = "separate_arcs_or_hidden_multiple_root_unresolved"

    payload = {
        "claim_status": "event-specific float64 organizer screen; independent BigFloat/canonical reproduction required",
        "requested_center": [center_m1, center_m2],
        "mixed_node": args.mixed_node,
        "window": [args.radius_m1, args.radius_m2],
        "window_transition_count": len(records),
        "coarse_cross_mode_edge": [
            {"m1": r["m1"], "m2": r["m2"], "orientation": r["orientation"], "mode": r["mode"]}
            for r in edge_records
        ],
        "traces": traces,
        "closest_trace_spectral_distance_to_mixed_vertex": mixed_seed_distance,
        "closest_cross_trace_mass_gap": mass_gap,
        "direct_mixed_vertex_retry": direct_result,
        "screening_classification": classification,
        "interpretation": (
            "The classification is a falsification-oriented screen. A direct mixed candidate requires "
            "independent reproduction; a fold screen requires event-curve curvature verification; "
            "an unresolved result must not be converted into an endpoint or nonexistence claim."
        ),
        "completed": True,
    }
    write_json_atomic(Path(args.output), payload)
    print(
        json.dumps(
            {
                "center": payload["requested_center"],
                "coarse_edge": payload["coarse_cross_mode_edge"],
                "trace_points": [len(t["points"]) for t in traces],
                "stopped_reasons": [t["stopped_reason"] for t in traces],
                "mixed_seed_distance": mixed_seed_distance,
                "mass_gap": mass_gap,
                "direct_status": direct_result["status"],
                "classification": classification,
            },
            indent=2,
        )
    )
    if any(not trace["points"] for trace in traces):
        raise SystemExit("one or more organizer traces produced zero accepted pseudo-arclength points")


if __name__ == "__main__":
    main()
