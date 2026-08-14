#!/usr/bin/env python3
"""Trace lower/upper screening critical components from refined mass slices."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from threebody_atlas.boundary import evaluate
from threebody_atlas.critical_curve import trace_critical_curve
from threebody_atlas.liao_family import FamilyPoint


def family_point(data: dict) -> FamilyPoint:
    return FamilyPoint(
        masses=tuple(data["masses"]),
        x1=float(data["x1"]),
        v1=float(data["v1"]),
        v2=float(data["v2"]),
        period=float(data["period"]),
        residual_norm=float(data["shooting_residual"]),
        nfev=0,
        success=True,
    )


def critical_sample(record: dict):
    stable = record["stable_side"]
    unstable = record["unstable_side"]
    chosen = stable if abs(stable["stability_score"]) <= abs(unstable["stability_score"]) else unstable
    return evaluate(family_point(chosen))


def serialize_point(point) -> dict:
    p = point.sample.point
    return {
        "masses": p.masses,
        "x1": p.x1,
        "v1": p.v1,
        "v2": p.v2,
        "period": p.period,
        "shooting_residual": p.residual_norm,
        "stability_score": point.sample.score,
        "tangent": point.tangent,
        "normal_correction": point.correction_offset,
        "normal_bracket_width": point.bracket_width,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("refined_json")
    parser.add_argument("output")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--arclength-step", type=float, default=5e-4)
    args = parser.parse_args()

    payload = json.loads(Path(args.refined_json).read_text(encoding="utf-8"))
    grouped: dict[str, list[dict]] = {"U->S": [], "S->U": []}
    for record in payload["refined_brackets"]:
        orientation = "->".join(record["published_labels"])
        if orientation in grouped:
            grouped[orientation].append(record)

    components = {}
    for orientation, records in grouped.items():
        records.sort(key=lambda x: x["m1"])
        if len(records) < 2:
            continue
        first, second = critical_sample(records[0]), critical_sample(records[1])
        trace = trace_critical_curve(
            first,
            second,
            steps=args.steps,
            arclength_step=args.arclength_step,
            normal_half_width=7.5e-4,
            normal_tolerance=2e-7,
            score_tolerance=2e-7,
        )
        components[orientation] = {
            "seed_m1": [records[0]["m1"], records[1]["m1"]],
            "points": [serialize_point(p) for p in trace.points],
            "stopped_reason": trace.stopped_reason,
        }

    out = {
        "claim_status": "screening-only; critical curves require BigFloat verification",
        "components": components,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: len(v["points"]) for k, v in components.items()}, indent=2))


if __name__ == "__main__":
    main()
