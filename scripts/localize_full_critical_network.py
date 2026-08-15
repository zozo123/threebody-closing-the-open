#!/usr/bin/env python3
"""Localize every published S/U transition cell at its unique smooth Floquet event.

This is the root-level edge substrate for the v1 critical graph.  The frozen
coarse audit established that all 620 adjacent S/U cells have exactly one
endpoint-sign-changing event among ``G+``, ``G-`` and ``Delta``.  Here each cell
is independently corrected and localized at fixed ``m1``; no mass-plane track
association is assumed during the solve.

The resulting points are still float64 structural evidence.  Organizers, folds,
and headline mechanisms remain gated by the independent Julia BigFloat and
canonical workflows.  This pass exists to replace coarse cell midpoints by
branch-preserving corrected event roots along every graph edge.
"""
from __future__ import annotations

import argparse
import csv
import json
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


def orientation(row: dict[str, str]) -> str:
    return f'{row["left_label"]}->{row["right_label"]}'


def solve_cell(cell_id: int, row: dict[str, str]) -> dict[str, Any]:
    left_point, right_point = point(row, "left"), point(row, "right")
    left, right = evaluate(left_point), evaluate(right_point)
    values = {
        mode: [event_value(left.floquet, mode), event_value(right.floquet, mode)]
        for mode in MODES
    }
    modes = [
        mode
        for mode, (a, b) in values.items()
        if a == 0.0 or b == 0.0 or a * b < 0.0
    ]
    if len(modes) != 1:
        raise RuntimeError(
            f"cell {cell_id} lost unique-event invariant: modes={modes} values={values}"
        )
    mode = modes[0]
    stable = left_point if row["left_label"] == "S" else right_point
    unstable = left_point if row["left_label"] == "U" else right_point
    critical = localize_critical_point(
        stable,
        unstable,
        event_mode=mode,
        # The scientific gate is the event residual below, not bracket width.
        # A 2e-9 width stop was coarse enough to return residuals above 2e-8
        # (for example cells 0 and 1 in run 31887432802), so refine the mass
        # bracket further instead of weakening the event acceptance criterion.
        m2_tolerance=1e-12,
        event_tolerance=2e-8,
        max_iterations=60,
        max_closure=1e-7,
    )
    q, f = critical.sample.point, critical.sample.floquet
    lo, hi = sorted((float(row["left_m2"]), float(row["right_m2"])))
    m2 = float(q.masses[1])
    if not (lo - 2e-9 <= m2 <= hi + 2e-9):
        raise RuntimeError(f"cell {cell_id} localized outside source bracket: {m2} not in [{lo},{hi}]")
    if q.residual_norm > 1e-7 or abs(critical.event_value) > 2e-8:
        raise RuntimeError(
            f"cell {cell_id} missed gates: closure={q.residual_norm:.3e} event={critical.event_value:.3e}"
        )
    return {
        "cell_id": cell_id,
        "orientation": orientation(row),
        "published_labels": [row["left_label"], row["right_label"]],
        "event_mode": mode,
        "source_m2_bracket": [lo, hi],
        "source_event_endpoint_values": values[mode],
        "masses": [float(x) for x in q.masses],
        "x1": float(q.x1),
        "v1": float(q.v1),
        "v2": float(q.v2),
        "period": float(q.period),
        "closure": float(q.residual_norm),
        "event": float(critical.event_value),
        "source_width": float(critical.source_width),
        "alpha": float(f.alpha),
        "beta": float(f.beta),
        "discriminant": float(f.discriminant),
        "plus_one_event": float(event_value(f, "plus_one")),
        "minus_one_event": float(event_value(f, "minus_one")),
        "trace_collision_event": float(event_value(f, "trace_collision")),
        "trace_roots": [[float(z.real), float(z.imag)] for z in f.trace_roots],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("brackets_tsv")
    parser.add_argument("output")
    parser.add_argument("--chunk-index", type=int, required=True)
    parser.add_argument("--chunks", type=int, required=True)
    args = parser.parse_args()
    if args.chunks < 1 or not (0 <= args.chunk_index < args.chunks):
        raise SystemExit("invalid chunk partition")

    with Path(args.brackets_tsv).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise RuntimeError("no transition brackets supplied")

    selected = [(i, row) for i, row in enumerate(rows) if i % args.chunks == args.chunk_index]
    roots = []
    for ordinal, (cell_id, row) in enumerate(selected, start=1):
        print(f"chunk={args.chunk_index}/{args.chunks} root={ordinal}/{len(selected)} cell={cell_id}")
        roots.append(solve_cell(cell_id, row))

    payload = {
        "claim_status": "float64 structural localization of every assigned unique-event S/U cell; independent headline verification remains separate",
        "chunk_index": args.chunk_index,
        "chunks": args.chunks,
        "total_input_cells": len(rows),
        "localized_cells": len(roots),
        "max_closure": max((x["closure"] for x in roots), default=0.0),
        "max_abs_event": max((abs(x["event"]) for x in roots), default=0.0),
        "roots": roots,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("chunk_index", "localized_cells", "max_closure", "max_abs_event")}, indent=2))


if __name__ == "__main__":
    main()
