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
from threebody_atlas.critical_manifold import (
    classify_localized_cell,
    event_value,
    localize_critical_point,
)
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


def solve_cell(
    cell_id: int,
    row: dict[str, str],
    *,
    max_iterations: int = 14,
    m2_tolerance: float = 1e-12,
    event_tolerance: float = 2e-8,
    max_closure: float = 1e-7,
) -> dict[str, Any]:
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
    lo, hi = sorted((float(row["left_m2"]), float(row["right_m2"])))
    base = {
        "cell_id": cell_id,
        "orientation": orientation(row),
        "published_labels": [row["left_label"], row["right_label"]],
        "source_m2_bracket": [lo, hi],
        "source_event_endpoint_values": values,
    }
    if len(modes) != 1:
        return {
            **base,
            "status": "lost_unique_event",
            "event_mode": None,
            "error": f"modes={modes}",
        }
    mode = modes[0]
    stable = left_point if row["left_label"] == "S" else right_point
    unstable = left_point if row["left_label"] == "U" else right_point
    try:
        critical = localize_critical_point(
            stable,
            unstable,
            event_mode=mode,
            # The scientific gate is the event residual below, not bracket width.
            # A 2e-9 width stop was coarse enough to return residuals above 2e-8
            # (for example cells 0 and 1 in run 31887432802), so refine the mass
            # bracket further instead of weakening the event acceptance criterion.
            m2_tolerance=m2_tolerance,
            event_tolerance=event_tolerance,
            max_iterations=max_iterations,
            max_closure=max_closure,
        )
    except Exception as exc:
        return {
            **base,
            "status": "localize_failed",
            "event_mode": mode,
            "error": f"{type(exc).__name__}: {exc}",
        }
    q, f = critical.sample.point, critical.sample.floquet
    m2 = float(q.masses[1])
    status = classify_localized_cell(
        closure=float(q.residual_norm),
        event=float(critical.event_value),
        m2=m2,
        lo=lo,
        hi=hi,
        max_closure=max_closure,
        event_tolerance=event_tolerance,
    )
    return {
        **base,
        "status": status,
        "event_mode": mode,
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
    parser.add_argument("--chunk-index", type=int, default=0)
    parser.add_argument("--chunks", type=int, default=1)
    parser.add_argument("--cell-ids", default="")
    parser.add_argument("--max-iterations", type=int, default=14)
    parser.add_argument("--m2-tolerance", type=float, default=1e-12)
    parser.add_argument("--event-tolerance", type=float, default=2e-8)
    parser.add_argument("--max-closure", type=float, default=1e-7)
    args = parser.parse_args()
    if args.chunks < 1 or not (0 <= args.chunk_index < args.chunks):
        raise SystemExit("invalid chunk partition")
    if args.max_iterations < 1:
        raise SystemExit("max-iterations must be positive")
    if args.event_tolerance > 2e-8:
        raise SystemExit("refusing to loosen the 2e-8 event gate")
    if args.max_closure > 1e-7:
        raise SystemExit("refusing to loosen the 1e-7 closure gate")
    if args.m2_tolerance <= 0.0:
        raise SystemExit("m2-tolerance must be positive")

    with Path(args.brackets_tsv).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise RuntimeError("no transition brackets supplied")

    if args.cell_ids.strip():
        wanted: list[int] = []
        for part in args.cell_ids.split(","):
            token = part.strip()
            if not token:
                continue
            wanted.append(int(token))
        bad = [cell for cell in wanted if not (0 <= cell < len(rows))]
        if bad:
            raise SystemExit(f"cell ids out of range: {bad[:20]}")
        selected = [(cell, rows[cell]) for cell in wanted]
    else:
        selected = [(i, row) for i, row in enumerate(rows) if i % args.chunks == args.chunk_index]
    attempts = []
    for ordinal, (cell_id, row) in enumerate(selected, start=1):
        print(f"chunk={args.chunk_index}/{args.chunks} root={ordinal}/{len(selected)} cell={cell_id}")
        attempts.append(
            solve_cell(
                cell_id,
                row,
                max_iterations=args.max_iterations,
                m2_tolerance=args.m2_tolerance,
                event_tolerance=args.event_tolerance,
                max_closure=args.max_closure,
            )
        )

    roots = [item for item in attempts if item.get("status") == "ok"]
    missed = [item for item in attempts if item.get("status") != "ok"]
    payload = {
        "claim_status": "float64 structural localization of every assigned unique-event S/U cell; independent headline verification remains separate",
        "chunk_index": args.chunk_index,
        "chunks": args.chunks,
        "solver": {
            "max_iterations": args.max_iterations,
            "m2_tolerance": args.m2_tolerance,
            "event_tolerance": args.event_tolerance,
            "max_closure": args.max_closure,
            "cell_ids": args.cell_ids,
        },
        "total_input_cells": len(rows),
        "attempted_cells": len(attempts),
        "localized_cells": len(roots),
        "missed_cells": len(missed),
        "missed_status_counts": {
            status: sum(1 for item in missed if item.get("status") == status)
            for status in sorted({str(item.get("status")) for item in missed})
        },
        "max_closure": max((float(x["closure"]) for x in roots), default=0.0),
        "max_abs_event": max((abs(float(x["event"])) for x in roots), default=0.0),
        "attempts": attempts,
        "roots": roots,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                k: payload[k]
                for k in (
                    "chunk_index",
                    "attempted_cells",
                    "localized_cells",
                    "missed_cells",
                    "missed_status_counts",
                    "max_closure",
                    "max_abs_event",
                )
            },
            indent=2,
        )
    )
    if missed:
        first = missed[0]
        raise SystemExit(
            f"{len(missed)} cell(s) missed gates; first cell={first.get('cell_id')} "
            f"status={first.get('status')} event={first.get('event')} "
            f"closure={first.get('closure')} error={first.get('error')}"
        )


if __name__ == "__main__":
    main()
