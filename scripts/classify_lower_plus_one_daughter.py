#!/usr/bin/env python3
"""Freeze the lower +1 daughter class from continuation and independent roots.

The v1 graph only needs to decide whether the branch-switch screen changes the
Li-family critical graph. A finite float64 trace is insufficient by itself. A
``distinct_branch`` class requires a physical soft-direction trace, its frozen
reversing-symmetry identification, and at least one independently corrected
BigFloat generic orbit separated from the same-mass Li parent.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def classify(structural: dict[str, Any], independent: dict[str, Any]) -> dict[str, Any]:
    artifacts = structural.get("artifacts") or []
    physical = next((row for row in artifacts if row.get("label") == "d0-minus"), None)
    symmetry = structural.get("symmetry_and_branch_identity_screens") or {}
    results = independent.get("results") or []
    if physical is None or not results or independent.get("passed") is not True:
        raise SystemExit(
            "daughter closure requires the physical d0-minus trace and a passed independent artifact"
        )
    trace_points = int(physical.get("trace_point_count") or 0)
    trace_closure = float(physical.get("maximum_closure_norm") or float("inf"))
    reversing_distance = float(
        symmetry.get("direction0_plus_minus_reversing_symmetry_best_normalized_chart_distance")
        or float("inf")
    )
    if trace_points < 3 or trace_closure > 1e-7 or reversing_distance > 1e-5:
        raise SystemExit(
            "daughter structural gates failed: "
            f"points={trace_points} closure={trace_closure:.3e} symmetry={reversing_distance:.3e}"
        )
    for row in results:
        closure = abs(float(row.get("closure_norm", "inf")))
        gauges = abs(float(row.get("gauge_norm", "inf")))
        parent_distance = abs(float(row.get("parent_distance", 0.0)))
        off_li = abs(float(row.get("off_li_norm", 0.0)))
        if closure > 1e-7 or gauges > 1e-7:
            raise SystemExit(
                f"independent daughter residual gate failed: closure={closure:.3e} gauges={gauges:.3e}"
            )
        if parent_distance < 1e-4 or off_li < 1e-5:
            raise SystemExit(
                "independent daughter collapsed onto the Li chart: "
                f"parent_distance={parent_distance:.3e} off_li={off_li:.3e}"
            )
    return {
        "id": "lower_plus_one_daughter",
        "kind": "branch",
        "class": "distinct_branch",
        "passed": True,
        "status": "independently_reproduced",
        "evidence_level": "independently_reproduced",
        "estimator": independent.get("implementation"),
        "note": (
            "The physical soft direction defines a generic strict-periodic branch distinct from "
            "the same-mass Li parent; its reversing-symmetry partner is the same daughter."
        ),
        "trace_point_count": trace_points,
        "maximum_trace_closure": trace_closure,
        "reversing_symmetry_distance": reversing_distance,
        "independent_result_count": len(results),
        "known_limitation": "Global long-range daughter genealogy is outside this finite v1 critical-graph classification.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--structural", required=True)
    parser.add_argument("--independent", required=True)
    args = parser.parse_args()
    structural = json.loads(Path(args.structural).read_text(encoding="utf-8"))
    independent = json.loads(Path(args.independent).read_text(encoding="utf-8"))
    record = classify(structural, independent)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "class": record["class"],
                "passed": record["passed"],
                "independent_result_count": record["independent_result_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
