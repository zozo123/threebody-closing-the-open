#!/usr/bin/env python3
"""Merge float64 persist-all roots with independent Julia BigFloat cells.

Publication residual is the Julia event when a cell was escalated. Float64
roots are kept only when they already met the frozen 2e-8 / 1e-7 gates.
The merger never loosens those gates and never invents a missing cell.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


EVENT_GATE = 2e-8
CLOSURE_GATE = 1e-7


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


def from_julia(row: dict[str, Any]) -> dict[str, Any]:
    cell_id = int(row["cell_id"])
    event = as_float(row.get("event_value", row.get("event")))
    closure = as_float(row.get("closure_norm", row.get("closure")))
    passed = bool(row.get("passed", abs(event) <= EVENT_GATE and closure <= CLOSURE_GATE))
    if abs(event) > EVENT_GATE or closure > CLOSURE_GATE:
        passed = False
    masses = row.get("masses")
    if masses is None:
        masses = [row.get("m1"), row.get("m2"), row.get("m3")]
    return {
        "cell_id": cell_id,
        "status": "ok" if passed else "missed_event",
        "event_mode": row.get("event_mode") or row.get("mode"),
        "event": event,
        "closure": closure,
        "masses": [as_float(x) for x in masses if x is not None],
        "x1": row.get("x1"),
        "v1": row.get("v1"),
        "v2": row.get("v2"),
        "period": row.get("period"),
        "estimator": "julia_bigfloat",
        "passed": passed,
        "source": "julia",
    }


def from_python(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["estimator"] = "float64"
    out["source"] = "python_float64"
    out["passed"] = row.get("status") == "ok"
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("python_roots")
    parser.add_argument("output")
    parser.add_argument("--julia", action="append", default=[], help="Julia cell JSON (repeatable)")
    args = parser.parse_args()

    python_payload = load(Path(args.python_roots))
    python_rows = python_payload.get("roots") or []
    attempts = python_payload.get("attempts") or python_rows
    merged: dict[int, dict[str, Any]] = {}
    for row in python_rows:
        if row.get("status") != "ok":
            continue
        item = from_python(row)
        if abs(as_float(item["event"])) > EVENT_GATE or as_float(item["closure"]) > CLOSURE_GATE:
            raise SystemExit(f"python root cell {item['cell_id']} fails frozen gates")
        merged[int(item["cell_id"])] = item

    julia_count = 0
    for path in args.julia:
        payload = load(Path(path))
        for row in julia_rows(payload):
            item = from_julia(row)
            julia_count += 1
            cell_id = int(item["cell_id"])
            if not item["passed"]:
                continue
            merged[cell_id] = item

    roots = [merged[cell] for cell in sorted(merged)]
    missing = sorted(set(range(620)) - set(merged))
    payload = {
        "schema": "atlas.v1.hybrid-critical-root-network/1",
        "claim_status": (
            "hybrid 620-cell localization: float64 prefilter plus independent Julia BigFloat on misses"
            if not missing
            else "partial hybrid localization; remaining misses still require Julia or classification"
        ),
        "source_transition_cells": 620,
        "localized_roots": len(roots),
        "missing_cells": missing,
        "estimator_counts": dict(sorted(Counter(root["estimator"] for root in roots).items())),
        "event_mode_counts": dict(
            sorted(Counter(root.get("event_mode") for root in roots).items())
        ),
        "max_closure": max((as_float(root["closure"]) for root in roots), default=0.0),
        "max_abs_event": max((abs(as_float(root["event"])) for root in roots), default=0.0),
        "python_ok": sum(1 for root in roots if root["estimator"] == "float64"),
        "julia_ok": sum(1 for root in roots if root["estimator"] == "julia_bigfloat"),
        "julia_files": julia_count,
        "python_attempts": len(attempts),
        "roots": roots,
    }
    if payload["max_abs_event"] > EVENT_GATE:
        raise SystemExit(f"merged event gate failed: {payload['max_abs_event']:.6e}")
    if payload["max_closure"] > CLOSURE_GATE:
        raise SystemExit(f"merged closure gate failed: {payload['max_closure']:.6e}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "localized_roots": payload["localized_roots"],
                "missing_cells": payload["missing_cells"][:20],
                "missing_count": len(missing),
                "estimator_counts": payload["estimator_counts"],
                "max_closure": payload["max_closure"],
                "max_abs_event": payload["max_abs_event"],
            },
            indent=2,
        )
    )
    if missing:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
