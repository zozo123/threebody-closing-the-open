#!/usr/bin/env python3
"""Merge float64 persist-all roots with independent Julia BigFloat cells.

Publication residual is the Julia event when a cell was escalated. Float64
roots are kept only when they already met the frozen 2e-8 / 1e-7 gates.
The merger never loosens those gates, never invents a missing cell, and keeps
the original published-cell orientation/bracket metadata when BigFloat replaces
a float64 miss.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


EVENT_GATE = 2e-8
CLOSURE_GATE = 1e-7
EXPECTED_CELLS = set(range(620))


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def as_float(value: Any) -> float:
    return float(value)


def julia_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if "results" in payload:
        return list(payload["results"])
    if "cell_id" in payload:
        return [payload]
    raise SystemExit(f"unrecognized Julia payload keys: {sorted(payload)[:12]}")


def from_python(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["estimator"] = "float64"
    out["source"] = "python_float64"
    out["passed"] = row.get("status") == "ok"
    return out


def from_julia(row: dict[str, Any], screening: dict[str, Any] | None) -> dict[str, Any]:
    cell_id = int(row["cell_id"])
    event = as_float(row.get("event_value", row.get("event")))
    closure = as_float(row.get("closure_norm", row.get("closure")))
    passed = bool(row.get("passed", abs(event) <= EVENT_GATE and closure <= CLOSURE_GATE))
    passed = passed and abs(event) <= EVENT_GATE and closure <= CLOSURE_GATE
    masses = row.get("masses")
    if masses is None:
        masses = [row.get("m1"), row.get("m2"), row.get("m3")]
    out = {
        "cell_id": cell_id,
        "status": "ok" if passed else "missed_gate",
        "event_mode": row.get("event_mode") or row.get("mode"),
        "event": event,
        "closure": closure,
        "masses": [as_float(x) for x in masses if x is not None],
        "x1": row.get("x1"),
        "v1": row.get("v1"),
        "v2": row.get("v2"),
        "period": row.get("period"),
        "alpha": row.get("alpha"),
        "beta": row.get("beta"),
        "discriminant": row.get("discriminant"),
        "estimator": "julia_bigfloat",
        "passed": passed,
        "source": "julia_bigfloat_vern9",
    }
    if screening:
        out["orientation"] = screening.get("orientation")
        out["published_labels"] = screening.get("published_labels")
        out["source_m2_bracket"] = screening.get("source_m2_bracket")
        out["screening"] = {
            "estimator": "python_float64",
            "status": screening.get("status"),
            "event_mode": screening.get("event_mode"),
            "event": screening.get("event"),
            "closure": screening.get("closure"),
            "error": screening.get("error"),
        }
        if screening.get("event_mode") and screening.get("event_mode") != out["event_mode"]:
            out["screening_event_mode_reclassified"] = True
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("python_roots")
    parser.add_argument("output")
    parser.add_argument("--julia", action="append", default=[], help="Julia cell/batch JSON (repeatable)")
    args = parser.parse_args()

    python_payload = load(Path(args.python_roots))
    python_rows = list(python_payload.get("roots") or [])
    attempts = list(python_payload.get("attempts") or python_rows)
    attempt_by_cell: dict[int, dict[str, Any]] = {}
    for row in attempts:
        cell_id = int(row["cell_id"])
        if cell_id in attempt_by_cell:
            raise SystemExit(f"duplicate float64 attempt for cell {cell_id}")
        attempt_by_cell[cell_id] = row
    if set(attempt_by_cell) != EXPECTED_CELLS:
        missing = sorted(EXPECTED_CELLS - set(attempt_by_cell))
        extra = sorted(set(attempt_by_cell) - EXPECTED_CELLS)
        raise SystemExit(f"float64 attempt coverage must be exactly 0..619; missing={missing[:20]} extra={extra[:20]}")

    merged: dict[int, dict[str, Any]] = {}
    for row in python_rows:
        if row.get("status") != "ok":
            continue
        item = from_python(row)
        if abs(as_float(item["event"])) > EVENT_GATE or as_float(item["closure"]) > CLOSURE_GATE:
            raise SystemExit(f"python root cell {item['cell_id']} fails frozen gates")
        cell_id = int(item["cell_id"])
        if cell_id in merged:
            raise SystemExit(f"duplicate float64 accepted root for cell {cell_id}")
        merged[cell_id] = item

    seen_julia: set[int] = set()
    julia_rows_seen = 0
    julia_passed = 0
    reclassified: list[int] = []
    for path in args.julia:
        payload = load(Path(path))
        for row in julia_rows(payload):
            julia_rows_seen += 1
            cell_id = int(row["cell_id"])
            if cell_id not in EXPECTED_CELLS:
                raise SystemExit(f"Julia cell out of range: {cell_id}")
            if cell_id in seen_julia:
                raise SystemExit(f"duplicate Julia result for cell {cell_id}")
            seen_julia.add(cell_id)
            screening = attempt_by_cell.get(cell_id)
            item = from_julia(row, screening)
            if not item["passed"]:
                continue
            julia_passed += 1
            if item.get("screening_event_mode_reclassified"):
                reclassified.append(cell_id)
            merged[cell_id] = item

    roots = [merged[cell] for cell in sorted(merged)]
    missing = sorted(EXPECTED_CELLS - set(merged))
    counts = Counter(root["estimator"] for root in roots)
    payload = {
        "schema": "atlas.v1.hybrid-critical-root-network/2",
        "claim_status": (
            "hybrid 620-cell localization: float64 prefilter plus independent Julia BigFloat escalation on misses"
            if not missing
            else "partial hybrid localization; remaining cells still require accepted high-precision localization"
        ),
        "source_transition_cells": 620,
        "localized_roots": len(roots),
        "missing_cells": missing,
        "complete": not missing,
        "estimator_counts": dict(sorted(counts.items())),
        "event_mode_counts": dict(sorted(Counter(root.get("event_mode") for root in roots).items())),
        "max_closure": max((as_float(root["closure"]) for root in roots), default=0.0),
        "max_abs_event": max((abs(as_float(root["event"])) for root in roots), default=0.0),
        "python_attempts": len(attempts),
        "python_ok": counts.get("float64", 0),
        "julia_rows_seen": julia_rows_seen,
        "julia_ok": julia_passed,
        "julia_event_mode_reclassifications": sorted(reclassified),
        "frozen_gates": {"event": EVENT_GATE, "closure": CLOSURE_GATE},
        "roots": roots,
    }
    if payload["max_abs_event"] > EVENT_GATE:
        raise SystemExit(f"merged event gate failed: {payload['max_abs_event']:.6e}")
    if payload["max_closure"] > CLOSURE_GATE:
        raise SystemExit(f"merged closure gate failed: {payload['max_closure']:.6e}")
    if len({int(root["cell_id"]) for root in roots}) != len(roots):
        raise SystemExit("merged roots contain duplicate cell ids")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "localized_roots": payload["localized_roots"],
        "missing_count": len(missing),
        "missing_cells": missing[:20],
        "estimator_counts": payload["estimator_counts"],
        "julia_event_mode_reclassifications": payload["julia_event_mode_reclassifications"],
        "max_closure": payload["max_closure"],
        "max_abs_event": payload["max_abs_event"],
    }, indent=2))
    if missing:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
