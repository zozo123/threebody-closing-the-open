#!/usr/bin/env python3
"""Merge independently completed stability-neck raster tiles.

The monolithic 21x131 scan repeatedly exceeded the one-hour hosted-runner
limit. Each tile contains one or more complete m1 lines; this merger verifies
that the tiles are complete, disjoint, and cover the exact declared grid before
it emits a completeness-admissible artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def frange(start: float, stop: float, step: float) -> list[float]:
    count = int(round((stop - start) / step))
    return [round(start + index * step, 10) for index in range(count + 1)]


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("fragments", nargs="+")
    parser.add_argument("--expected-m1-min", type=float, required=True)
    parser.add_argument("--expected-m1-max", type=float, required=True)
    parser.add_argument("--expected-m2-min", type=float, required=True)
    parser.add_argument("--expected-m2-max", type=float, required=True)
    parser.add_argument("--step", type=float, required=True)
    args = parser.parse_args()

    expected_m1 = frange(args.expected_m1_min, args.expected_m1_max, args.step)
    expected_keys = {round(value, 10) for value in expected_m1}
    summaries_by_m1: dict[float, dict[str, Any]] = {}
    samples: list[dict[str, Any]] = []
    sources: list[dict[str, str]] = []
    maximum_closure = 0.0

    for raw_path in args.fragments:
        path = Path(raw_path)
        payload = load(path)
        if payload.get("completed") is not True:
            raise SystemExit(f"incomplete neck tile: {path}")
        grid = payload.get("grid") or {}
        if float(grid.get("step", float("nan"))) != args.step:
            raise SystemExit(f"grid step mismatch in {path}")
        if [float(x) for x in grid.get("m2", [])] != [args.expected_m2_min, args.expected_m2_max]:
            raise SystemExit(f"m2 domain mismatch in {path}")
        tile_summaries = payload.get("line_summaries") or []
        tile_samples = payload.get("samples") or []
        if int(grid.get("samples", -1)) != len(tile_samples):
            raise SystemExit(f"sample-count mismatch in {path}")
        tile_m1_domain = [float(value) for value in grid.get("m1", [])]
        if len(tile_m1_domain) != 2:
            raise SystemExit(f"invalid m1 domain in {path}")
        tile_m1 = frange(tile_m1_domain[0], tile_m1_domain[1], args.step)
        tile_m2 = frange(args.expected_m2_min, args.expected_m2_max, args.step)
        summary_keys = {round(float(summary["m1"]), 10) for summary in tile_summaries}
        if summary_keys != set(tile_m1):
            raise SystemExit(f"line-summary coverage mismatch in {path}")
        if int(payload.get("completed_m1_lines", len(tile_summaries))) != len(tile_m1):
            raise SystemExit(f"completed-line count mismatch in {path}")
        if int(payload.get("expected_m1_lines", len(tile_m1))) != len(tile_m1):
            raise SystemExit(f"expected-line count mismatch in {path}")
        samples_by_m1: dict[float, list[float]] = {}
        for sample in tile_samples:
            key = round(float(sample["m1"]), 10)
            samples_by_m1.setdefault(key, []).append(round(float(sample["m2"]), 10))
        if set(samples_by_m1) != set(tile_m1):
            raise SystemExit(f"sample m1 coverage mismatch in {path}")
        for key, m2_values in samples_by_m1.items():
            if sorted(m2_values) != tile_m2:
                raise SystemExit(f"sample m2 coverage mismatch at m1={key} in {path}")
        for summary in tile_summaries:
            key = round(float(summary["m1"]), 10)
            if key in summaries_by_m1:
                raise SystemExit(f"duplicate m1 line {key} in {path}")
            summaries_by_m1[key] = summary
        samples.extend(tile_samples)
        maximum_closure = max(maximum_closure, float(payload.get("max_shooting_residual") or 0.0))
        sources.append(
            {
                "file": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )

    actual_keys = set(summaries_by_m1)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise SystemExit(f"neck-tile coverage mismatch: missing={missing} extra={extra}")
    expected_samples_per_line = len(frange(args.expected_m2_min, args.expected_m2_max, args.step))
    if len(samples) != len(expected_m1) * expected_samples_per_line:
        raise SystemExit(
            "merged sample-count mismatch: "
            f"got={len(samples)} expected={len(expected_m1) * expected_samples_per_line}"
        )

    summaries = [summaries_by_m1[key] for key in sorted(summaries_by_m1)]
    samples.sort(key=lambda row: (float(row["m1"]), float(row["m2"])))
    positive_gaps = [
        float(gap)
        for summary in summaries
        for gap in summary.get("interior_unstable_gaps", [])
        if float(gap) >= 0.0
    ]
    record: dict[str, Any] = {
        "schema": "atlas.v1.stability-neck-scan/2",
        "completed": True,
        "fragment_count": len(args.fragments),
        "grid": {
            "m1": [args.expected_m1_min, args.expected_m1_max],
            "m2": [args.expected_m2_min, args.expected_m2_max],
            "step": args.step,
            "samples": len(samples),
        },
        "max_shooting_residual": maximum_closure,
        "minimum_resolved_unstable_gap": min(positive_gaps) if positive_gaps else None,
        "any_vertical_merge": any(len(summary.get("stable_intervals", [])) <= 1 for summary in summaries),
        "line_summaries": summaries,
        "samples": samples,
        "source_fragments": sorted(sources, key=lambda row: row["file"]),
        "claim_status": (
            "completed sharded float64 subgrid topology screen; critical boundaries require "
            "continuation/BigFloat verification"
        ),
    }
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":"))
    record["sha256_content"] = hashlib.sha256(canonical.encode()).hexdigest()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "completed": True,
                "fragments": len(args.fragments),
                "samples": len(samples),
                "minimum_resolved_unstable_gap": record["minimum_resolved_unstable_gap"],
                "any_vertical_merge": record["any_vertical_merge"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
