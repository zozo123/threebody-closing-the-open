#!/usr/bin/env python3
"""Audit every published S/U transition cell for smooth Floquet events.

The earlier local event-network audit deliberately inspected only two narrow
windows around the secondary lobe.  That was sufficient to falsify a
single-mechanism boundary model, but not sufficient for a completeness claim.

This driver evaluates all three smooth reduced-Floquet event functions at both
endpoints of *every* frozen S/U transition bracket:

    G+ = P(+2),  G- = P(-2),  GC = Delta.

It then reconstructs the coarse transition tracks, splits them into connected
single-mechanism segments, and reports every adjacency at which mechanism
identity changes or the coarse cell is ambiguous.  Only anomalous/junction
cells are localized exactly; the global pass therefore remains much cheaper
than refining all 620 cells while still exposing hidden multi-event structure.

Endpoint signs are a topology/completeness screen, not publication truth.  An
even number of same-event zeros can evade a sign test, so all final graph edges
still require branch continuation and the adversarial local searches defined in
research/V1_CLOSURE.md.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from threebody_atlas.boundary import evaluate
from threebody_atlas.critical_manifold import event_value, localize_critical_point
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


def endpoint_key(p: FamilyPoint) -> tuple[float, ...]:
    return (
        *p.masses,
        p.x1,
        p.v1,
        p.v2,
        p.period,
    )


def midpoint_m2(row: dict[str, str]) -> float:
    return 0.5 * (float(row["left_m2"]) + float(row["right_m2"]))


def orientation(row: dict[str, str]) -> str:
    return f'{row["left_label"]}->{row["right_label"]}'


def sign_modes(left, right) -> tuple[list[str], dict[str, list[float]]]:
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


def localized(row: dict[str, str], mode: str) -> dict[str, Any]:
    left_point = point(row, "left")
    right_point = point(row, "right")
    stable = left_point if row["left_label"] == "S" else right_point
    unstable = left_point if row["left_label"] == "U" else right_point
    try:
        critical = localize_critical_point(
            stable,
            unstable,
            event_mode=mode,
            m2_tolerance=2e-9,
            event_tolerance=2e-8,
            max_iterations=36,
            max_closure=1e-7,
        )
    except RuntimeError as exc:
        return {
            "status": "localization_failed",
            "event_mode": mode,
            "error": str(exc),
        }
    p, f = critical.sample.point, critical.sample.floquet
    return {
        "status": "localized",
        "event_mode": mode,
        "masses": [float(x) for x in p.masses],
        "x1": float(p.x1),
        "v1": float(p.v1),
        "v2": float(p.v2),
        "period": float(p.period),
        "shooting_residual": float(p.residual_norm),
        "event_value": float(critical.event_value),
        "source_width": float(critical.source_width),
        "alpha": float(f.alpha),
        "beta": float(f.beta),
        "discriminant": float(f.discriminant),
        "trace_roots": [[float(z.real), float(z.imag)] for z in f.trace_roots],
    }


def build_adjacency(records: list[dict[str, Any]], threshold: float) -> list[list[int]]:
    buckets: dict[tuple[str, float], list[int]] = defaultdict(list)
    for rec in records:
        buckets[(rec["orientation"], round(rec["m1"], 3))].append(rec["id"])

    adjacency: list[list[int]] = [[] for _ in records]
    for rec in records:
        for delta in (-0.001, 0.001):
            key = (rec["orientation"], round(rec["m1"] + delta, 3))
            for other_id in buckets.get(key, []):
                other = records[other_id]
                if abs(rec["m2"] - other["m2"]) <= threshold:
                    adjacency[rec["id"]].append(other_id)
    return adjacency


def connected_components(
    node_ids: set[int],
    adjacency: list[list[int]],
) -> list[list[int]]:
    unseen = set(node_ids)
    components: list[list[int]] = []
    while unseen:
        seed = unseen.pop()
        queue = deque([seed])
        component = [seed]
        while queue:
            u = queue.popleft()
            for v in adjacency[u]:
                if v in unseen:
                    unseen.remove(v)
                    queue.append(v)
                    component.append(v)
        components.append(component)
    components.sort(key=len, reverse=True)
    return components


def mode_label(rec: dict[str, Any]) -> str:
    modes = rec["sign_changing_modes"]
    if len(modes) == 1:
        return modes[0]
    if not modes:
        return "none"
    return "multi:" + "+".join(sorted(modes))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("brackets_tsv")
    parser.add_argument("output")
    parser.add_argument("--link-threshold", type=float, default=0.02)
    parser.add_argument("--min-track-points", type=int, default=10)
    parser.add_argument(
        "--max-localized-junction-cells",
        type=int,
        default=80,
        help="safety cap for exact localization work around anomalies/junctions",
    )
    args = parser.parse_args()
    if args.link_threshold <= 0.0 or args.min_track_points < 1:
        raise SystemExit("link threshold and min-track-points must be positive")

    with Path(args.brackets_tsv).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise RuntimeError("no transition brackets supplied")

    cache = {}
    records: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        left_point = point(row, "left")
        right_point = point(row, "right")
        for p in (left_point, right_point):
            key = endpoint_key(p)
            if key not in cache:
                cache[key] = evaluate(p)
        left = cache[endpoint_key(left_point)]
        right = cache[endpoint_key(right_point)]
        modes, values = sign_modes(left, right)
        records.append(
            {
                "id": i,
                "m1": float(row["m1"]),
                "m2": midpoint_m2(row),
                "m2_bracket": [float(row["left_m2"]), float(row["right_m2"])],
                "m3": float(row["m3"]),
                "orientation": orientation(row),
                "published_labels": [row["left_label"], row["right_label"]],
                "sign_changing_modes": modes,
                "event_endpoint_values": values,
                "row": row,
            }
        )

    adjacency = build_adjacency(records, args.link_threshold)
    all_components = connected_components(set(range(len(records))), adjacency)
    coarse_tracks = [c for c in all_components if len(c) >= args.min_track_points]

    track_payloads = []
    record_to_track: dict[int, int] = {}
    for track_id, component in enumerate(coarse_tracks):
        for node_id in component:
            record_to_track[node_id] = track_id
        ordered = sorted((records[i] for i in component), key=lambda r: (r["m1"], r["m2"]))

        # Connected components after deleting edges that change the unique
        # mechanism label.  Ambiguous cells remain their own labels and are
        # reported explicitly as junction/anomaly candidates.
        by_label: dict[str, set[int]] = defaultdict(set)
        for rec in ordered:
            by_label[mode_label(rec)].add(rec["id"])
        segments = []
        for label, ids in by_label.items():
            restricted = [[] for _ in records]
            for u in ids:
                restricted[u] = [v for v in adjacency[u] if v in ids]
            for component_ids in connected_components(set(ids), restricted):
                nodes = sorted((records[i] for i in component_ids), key=lambda r: (r["m1"], r["m2"]))
                segments.append(
                    {
                        "mode": label,
                        "points": len(nodes),
                        "m1_range": [nodes[0]["m1"], nodes[-1]["m1"]],
                        "m2_range": [min(n["m2"] for n in nodes), max(n["m2"] for n in nodes)],
                        "start_cell_id": nodes[0]["id"],
                        "end_cell_id": nodes[-1]["id"],
                    }
                )
        segments.sort(key=lambda s: (-s["points"], s["mode"], s["m1_range"]))
        track_payloads.append(
            {
                "track_id": track_id,
                "orientation": ordered[0]["orientation"],
                "points": len(ordered),
                "m1_range": [ordered[0]["m1"], ordered[-1]["m1"]],
                "m2_range": [min(n["m2"] for n in ordered), max(n["m2"] for n in ordered)],
                "mode_counts": {
                    label: sum(1 for rec in ordered if mode_label(rec) == label)
                    for label in sorted({mode_label(rec) for rec in ordered})
                },
                "mechanism_segments": segments,
            }
        )

    junction_edges = []
    seen_edges = set()
    for u, neighbors in enumerate(adjacency):
        if u not in record_to_track:
            continue
        for v in neighbors:
            if v not in record_to_track or record_to_track[u] != record_to_track[v]:
                continue
            edge = tuple(sorted((u, v)))
            if edge in seen_edges:
                continue
            seen_edges.add(edge)
            lu, lv = mode_label(records[u]), mode_label(records[v])
            if lu != lv:
                junction_edges.append(
                    {
                        "track_id": record_to_track[u],
                        "cell_ids": list(edge),
                        "masses_midpoint": [
                            [records[u]["m1"], records[u]["m2"]],
                            [records[v]["m1"], records[v]["m2"]],
                        ],
                        "labels": [lu, lv],
                    }
                )

    multi = [rec for rec in records if len(rec["sign_changing_modes"]) > 1]
    none = [rec for rec in records if not rec["sign_changing_modes"]]

    # Localize only cells that are directly implicated in ambiguity or a mode
    # change.  Exact smooth-branch continuation from these seeds belongs to the
    # next stage; this stage globally determines where that work is required.
    interesting_ids = {rec["id"] for rec in multi}
    interesting_ids.update(rec["id"] for rec in none)
    for edge in junction_edges:
        interesting_ids.update(edge["cell_ids"])
    ordered_interesting = sorted(
        interesting_ids,
        key=lambda i: (records[i]["m1"], records[i]["m2"], i),
    )
    if len(ordered_interesting) > args.max_localized_junction_cells:
        localized_ids = set(ordered_interesting[: args.max_localized_junction_cells])
        localization_truncated = True
    else:
        localized_ids = set(ordered_interesting)
        localization_truncated = False

    localizations = []
    for node_id in sorted(localized_ids):
        rec = records[node_id]
        for mode in rec["sign_changing_modes"]:
            result = localized(rec["row"], mode)
            result.update(
                {
                    "cell_id": node_id,
                    "track_id": record_to_track.get(node_id),
                    "coarse_midpoint": [rec["m1"], rec["m2"]],
                    "coarse_bracket": rec["m2_bracket"],
                }
            )
            localizations.append(result)

    global_counts = {mode: 0 for mode in MODES}
    for rec in records:
        for mode in rec["sign_changing_modes"]:
            global_counts[mode] += 1

    compact_records = []
    for rec in records:
        compact_records.append(
            {
                key: value
                for key, value in rec.items()
                if key != "row"
            }
        )

    payload = {
        "claim_status": (
            "global coarse event-sign topology screen; exact critical graph still requires "
            "smooth pseudo-arclength continuation and independent BigFloat/canonical verification"
        ),
        "transition_brackets": len(records),
        "unique_endpoint_floquet_evaluations": len(cache),
        "link_threshold": args.link_threshold,
        "macroscopic_track_count": len(coarse_tracks),
        "macroscopic_tracks": track_payloads,
        "global_sign_change_counts": global_counts,
        "single_event_cells": sum(len(rec["sign_changing_modes"]) == 1 for rec in records),
        "multi_event_cells": len(multi),
        "no_sign_change_cells": len(none),
        "junction_edge_count": len(junction_edges),
        "junction_edges": junction_edges,
        "localization_truncated": localization_truncated,
        "localized_junction_cells": len(localized_ids),
        "localizations": localizations,
        "cells": compact_records,
        "small_track_components": [len(c) for c in all_components if len(c) < args.min_track_points],
        "interpretation": (
            "Unique endpoint sign changes classify coarse spectral walls globally. Multi-event, "
            "no-sign-change, and cross-mode adjacency cells are explicit targets for local dense "
            "search/direct organizer solves. Endpoint signs cannot exclude an even number of hidden "
            "same-event roots inside a cell."
        ),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "transition_brackets": payload["transition_brackets"],
                "unique_endpoint_floquet_evaluations": payload[
                    "unique_endpoint_floquet_evaluations"
                ],
                "macroscopic_track_count": payload["macroscopic_track_count"],
                "global_sign_change_counts": global_counts,
                "single_event_cells": payload["single_event_cells"],
                "multi_event_cells": payload["multi_event_cells"],
                "no_sign_change_cells": payload["no_sign_change_cells"],
                "junction_edge_count": payload["junction_edge_count"],
                "localized_junction_cells": payload["localized_junction_cells"],
                "localization_truncated": payload["localization_truncated"],
                "tracks": [
                    {
                        "track_id": track["track_id"],
                        "orientation": track["orientation"],
                        "points": track["points"],
                        "m1_range": track["m1_range"],
                        "mode_counts": track["mode_counts"],
                        "segments": track["mechanism_segments"],
                    }
                    for track in track_payloads
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
