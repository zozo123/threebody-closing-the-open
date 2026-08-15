#!/usr/bin/env python3
"""XOR-classify the secondary-right death from a wall-continue artifact.

Allowed classes: mixed_organizer, projection_fold, domain_boundary.
Newton-failed is never emitted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ALLOWED = frozenset({"mixed_organizer", "projection_fold", "domain_boundary"})


def classify(payload: dict[str, Any]) -> dict[str, Any]:
    direct = payload.get("direct_candidate")
    direct_error = payload.get("direct_error")
    approach = payload.get("approach") or []
    continuation_error = payload.get("continuation_error")
    closest = payload.get("closest_approach_mass_gap")

    mixed_ok = isinstance(direct, dict) and direct.get("success") is True
    if mixed_ok:
        return {
            "id": "secondary_right_death",
            "kind": "endpoint",
            "class": "mixed_organizer",
            "passed": False,
            "status": "candidate",
            "note": "Walls produced a simultaneous mixed seed; this is not a fourth organizer until BigFloat+physical bind.",
            "estimator": "float64_wall_screen",
            "closest_approach_mass_gap": closest,
            "requires_gate_a": True,
        }

    # Geometry-only classes. Continuation errors are diagnostics, not classes.
    gaps = [row.get("mass_gap") for row in approach if isinstance(row, dict) and "mass_gap" in row]
    if gaps and closest is not None and float(closest) <= 2e-3 and not mixed_ok:
        return {
            "id": "secondary_right_death",
            "kind": "endpoint",
            "class": "projection_fold",
            "passed": True,
            "status": "classified_from_geometry_screen",
            "note": "Walls approach closely without a surviving mixed solve; treat as a projection-fold death of the pair.",
            "estimator": "float64_wall_screen",
            "closest_approach_mass_gap": closest,
            "continuation_error": continuation_error,
            "direct_error": direct_error,
        }
    if approach:
        return {
            "id": "secondary_right_death",
            "kind": "endpoint",
            "class": "domain_boundary",
            "passed": True,
            "status": "classified_from_geometry_screen",
            "note": "Secondary walls were continued and did not produce a mixed organizer; recorded as a domain-boundary death of the island.",
            "estimator": "float64_wall_screen",
            "closest_approach_mass_gap": closest,
            "continuation_error": continuation_error,
            "direct_error": direct_error,
            "approach_steps": len(approach),
        }
    raise SystemExit("right-death artifact has no wall tape and no mixed candidate; refusing unknown")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("screen_json")
    parser.add_argument("output")
    args = parser.parse_args()
    payload = json.loads(Path(args.screen_json).read_text(encoding="utf-8"))
    record = classify(payload)
    if record["class"] not in ALLOWED:
        raise SystemExit(f"illegal class {record['class']}")
    if "newton" in str(record["class"]).lower():
        raise SystemExit("Newton-failed is not an allowed class")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"class": record["class"], "passed": record["passed"], "requires_gate_a": record.get("requires_gate_a")}, indent=2))


if __name__ == "__main__":
    main()
