#!/usr/bin/env python3
"""Reconstruct coarse stability-boundary tracks and classify their Floquet events.

The frozen 0.001 mass grid contains 620 adjacent S/U transition brackets over
301 m1 slices.  Rather than assuming two boundaries, this script connects
same-orientation brackets across adjacent m1 slices by continuity in m2, emits
the resulting coarse tracks, and independently localizes representative smooth
Floquet events at the start/middle/end of each macroscopic track.

This is a topology *screen*.  Final geometry comes from pseudo-arclength
continuation and independent BigFloat verification.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict, deque
from pathlib import Path

from threebody_atlas.critical_manifold import localize_critical_point
from threebody_atlas.liao_family import FamilyPoint


def midpoint_m2(row: dict[str, str]) -> float:
    return 0.5 * (float(row["left_m2"]) + float(row["right_m2"]))


def orientation(row: dict[str, str]) -> str:
    return f'{row["left_label"]}->{row["right_label"]}'


def family(row: dict[str, str], side: str) -> FamilyPoint:
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


def localize(row: dict[str, str]):
    left, right = family(row, "left"), family(row, "right")
    stable = left if row["left_label"] == "S" else right
    unstable = left if row["left_label"] == "U" else right
    return localize_critical_point(
        stable,
        unstable,
        m2_tolerance=8e-9,
        event_tolerance=8e-8,
        max_iterations=28,
        max_closure=1e-7,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("brackets_tsv")
    parser.add_argument("output")
    parser.add_argument("--link-threshold", type=float, default=0.02)
    args = parser.parse_args()

    with Path(args.brackets_tsv).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    nodes = [
        {
            "id": i,
            "row": row,
            "m1": float(row["m1"]),
            "m2": midpoint_m2(row),
            "orientation": orientation(row),
        }
        for i, row in enumerate(rows)
    ]
    buckets: dict[tuple[str, float], list[int]] = defaultdict(list)
    for node in nodes:
        buckets[(node["orientation"], round(node["m1"], 3))].append(node["id"])

    adjacency: list[list[int]] = [[] for _ in nodes]
    for node in nodes:
        for delta in (-0.001, 0.001):
            key = (node["orientation"], round(node["m1"] + delta, 3))
            for other_id in buckets.get(key, []):
                other = nodes[other_id]
                if abs(node["m2"] - other["m2"]) <= args.link_threshold:
                    adjacency[node["id"]].append(other_id)

    unseen = set(range(len(nodes)))
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

    tracks = []
    for component in components:
        ordered = sorted((nodes[i] for i in component), key=lambda n: (n["m1"], n["m2"]))
        if len(ordered) < 10:
            continue
        sample_indices = sorted(set((0, len(ordered) // 2, len(ordered) - 1)))
        classified = []
        for idx in sample_indices:
            node = ordered[idx]
            critical = localize(node["row"])
            p, f = critical.sample.point, critical.sample.floquet
            classified.append(
                {
                    "track_index": idx,
                    "coarse_m1": node["m1"],
                    "coarse_m2": node["m2"],
                    "localized_masses": p.masses,
                    "event_mode": critical.event_mode,
                    "event_value": critical.event_value,
                    "source_bracket_width": critical.source_width,
                    "shooting_residual": p.residual_norm,
                    "alpha": f.alpha,
                    "beta": f.beta,
                    "discriminant": f.discriminant,
                    "trace_roots": [[z.real, z.imag] for z in f.trace_roots],
                }
            )
        event_modes = sorted({x["event_mode"] for x in classified})
        tracks.append(
            {
                "orientation": ordered[0]["orientation"],
                "points": len(ordered),
                "m1_range": [ordered[0]["m1"], ordered[-1]["m1"]],
                "m2_range": [min(x["m2"] for x in ordered), max(x["m2"] for x in ordered)],
                "coarse_start": [ordered[0]["m1"], ordered[0]["m2"]],
                "coarse_end": [ordered[-1]["m1"], ordered[-1]["m2"]],
                "representative_event_modes": event_modes,
                "representatives": classified,
            }
        )

    tracks.sort(key=lambda x: x["points"], reverse=True)
    payload = {
        "transition_brackets": len(rows),
        "link_threshold": args.link_threshold,
        "macroscopic_tracks": tracks,
        "small_components": [len(c) for c in components if len(c) < 10],
        "interpretation": (
            "coarse topology screen only; track endpoints that share an m1 range may join through "
            "turning points between grid slices and require pseudo-arclength continuation"
        ),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps([
        {
            "orientation": t["orientation"],
            "points": t["points"],
            "m1_range": t["m1_range"],
            "m2_range": t["m2_range"],
            "event_modes": t["representative_event_modes"],
        }
        for t in tracks
    ], indent=2))


if __name__ == "__main__":
    main()
