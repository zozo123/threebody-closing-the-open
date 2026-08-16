#!/usr/bin/env python3
"""Ten-times-finer stability scan around the closest approach of two stable lobes.

The published 0.001 grid creates a one-cell adjacency between the main stable
region and the secondary lobe near m1=0.997--0.998, while vertical transition
brackets still show a narrow unstable gap.  This script removes that grid
aliasing ambiguity by warm-starting the periodic family on a 0.0001 local grid
and evaluating the full reduced Floquet criterion at every corrected orbit.

This remains float64 screening.  It is designed to determine topology targets
for continuation, not to substitute for BigFloat critical-point evidence.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from threebody_atlas.baseline import iter_baseline
from threebody_atlas.boundary import evaluate
from threebody_atlas.evidence_semantics import artifact_semantics
from threebody_atlas.liao_family import correct_family_point

sys.path.insert(0, str(Path(__file__).resolve().parent))

from neck_topology import summarize  # noqa: E402


def frange(start: float, stop: float, step: float) -> list[float]:
    count = int(round((stop - start) / step))
    return [round(start + i * step, 10) for i in range(count + 1)]


def payload_for(
    *,
    args: argparse.Namespace,
    m1_values: list[float],
    samples: list[dict],
    summaries: list[dict],
    max_closure: float,
    completed: bool,
) -> dict:
    positive_gaps = [
        gap
        for summary in summaries
        for gap in summary["interior_unstable_gaps"]
        if gap >= 0.0
    ]
    annotated, verdicts = summarize(
        summaries,
        m2_min=args.m2_min,
        m2_max=args.m2_max,
        step=args.step,
    )
    return {
        "schema": "atlas.v1.stability-neck-scan/3",
        "search_semantics": artifact_semantics(
            Path(__file__).resolve().parents[1], "local_neck_raster/v1"
        ),
        "completed": completed,
        "completed_m1_lines": len(summaries),
        "expected_m1_lines": len(m1_values),
        "grid": {
            "m1": [args.m1_min, args.m1_max],
            "m2": [args.m2_min, args.m2_max],
            "step": args.step,
            "samples": len(samples),
        },
        "max_shooting_residual": max_closure,
        "minimum_resolved_unstable_gap": min(positive_gaps) if positive_gaps else None,
        **verdicts,
        "line_summaries": annotated,
        "samples": samples,
        "claim_status": (
            "completed float64 subgrid topology screen; critical boundaries require "
            "continuation/BigFloat verification"
            if completed
            else "partial float64 subgrid topology checkpoint; not admissible as completeness evidence"
        ),
    }


def write_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("output")
    parser.add_argument("--m1-min", type=float, default=0.997)
    parser.add_argument("--m1-max", type=float, default=0.999)
    parser.add_argument("--m2-min", type=float, default=0.993)
    parser.add_argument("--m2-max", type=float, default=1.006)
    parser.add_argument("--step", type=float, default=0.0001)
    parser.add_argument("--max-residual", type=float, default=1e-7)
    args = parser.parse_args()

    baseline = list(iter_baseline(args.dataset))
    by_key = {(round(r.m1, 3), round(r.m2, 3)): r for r in baseline}
    m1_values = frange(args.m1_min, args.m1_max, args.step)
    m2_values = frange(args.m2_min, args.m2_max, args.step)
    output = Path(args.output)
    samples = []
    summaries = []
    max_closure = 0.0

    for m1 in m1_values:
        # Use the nearest frozen row at the left edge as the first predictor,
        # then march monotonically in m2 so every subsequent solve is a genuine
        # local continuation step of only 1e-4.
        anchor_key = (round(m1, 3), round(args.m2_min, 3))
        row = by_key.get(anchor_key)
        if row is None:
            candidates = [
                r for r in baseline
                if abs(r.m1 - m1) <= 0.0011 and abs(r.m2 - args.m2_min) <= 0.0011
            ]
            if not candidates:
                raise RuntimeError(f"no baseline anchor near m1={m1}")
            row = min(candidates, key=lambda r: abs(r.m1 - m1) + abs(r.m2 - args.m2_min))
        current = correct_family_point(
            (m1, args.m2_min, 1.0),
            (row.x1, row.v1, row.v2, row.period),
            max_nfev=70,
        )
        if not current.success or current.residual_norm > args.max_residual:
            raise RuntimeError(f"initial correction failed at {(m1,args.m2_min)}: {current.residual_norm:.3e}")

        line = []
        for j, m2 in enumerate(m2_values):
            if j:
                current = correct_family_point(
                    (m1, m2, 1.0),
                    (current.x1, current.v1, current.v2, current.period),
                    max_nfev=60,
                )
                if not current.success or current.residual_norm > args.max_residual:
                    raise RuntimeError(f"continuation failed at {(m1,m2)}: {current.residual_norm:.3e}")
            sample = evaluate(current)
            max_closure = max(max_closure, current.residual_norm)
            stable = bool(sample.score > 0.0)
            record = {
                "m1": m1,
                "m2": m2,
                "stable": stable,
                "score": sample.score,
                "shooting_residual": current.residual_norm,
                "alpha": sample.floquet.alpha,
                "beta": sample.floquet.beta,
                "discriminant": sample.floquet.discriminant,
                "trace_roots": [[z.real, z.imag] for z in sample.floquet.trace_roots],
            }
            samples.append(record)
            line.append(record)

        # Convert Boolean samples into stable intervals on this vertical line.
        intervals = []
        start = None
        for record in line:
            if record["stable"] and start is None:
                start = record["m2"]
            if not record["stable"] and start is not None:
                intervals.append([start, round(record["m2"] - args.step, 10)])
                start = None
        if start is not None:
            intervals.append([start, line[-1]["m2"]])
        gaps = []
        for left, right in zip(intervals, intervals[1:], strict=False):
            gaps.append(right[0] - left[1] - args.step)
        summaries.append(
            {
                "m1": m1,
                "stable_intervals": intervals,
                "interior_unstable_gaps": gaps,
            }
        )
        write_checkpoint(
            output,
            payload_for(
                args=args,
                m1_values=m1_values,
                samples=samples,
                summaries=summaries,
                max_closure=max_closure,
                completed=False,
            ),
        )

    payload = payload_for(
        args=args,
        m1_values=m1_values,
        samples=samples,
        summaries=summaries,
        max_closure=max_closure,
        completed=True,
    )
    write_checkpoint(output, payload)
    print(json.dumps({
        "samples": len(samples),
        "max_shooting_residual": max_closure,
        "minimum_resolved_unstable_gap": payload["minimum_resolved_unstable_gap"],
        "any_vertical_merge": payload["any_vertical_merge"],
        "any_boundary_truncated_merge_test": payload["any_boundary_truncated_merge_test"],
        "any_line_without_stable_sample": payload["any_line_without_stable_sample"],
        "any_stable_interval_touches_boundary": payload["any_stable_interval_touches_boundary"],
        "all_lines_separated": payload["all_lines_separated"],
        "merge_verdict_counts": payload["merge_verdict_counts"],
        "summaries": payload["line_summaries"],
    }, indent=2))


if __name__ == "__main__":
    main()
