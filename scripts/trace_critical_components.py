#!/usr/bin/env python3
"""Localize and trace smooth Floquet critical components from refined mass slices."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from threebody_atlas.critical_manifold import (
    localize_critical_point,
    trace_augmented_critical,
)
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


def localize_record(record: dict, event_mode=None):
    stable = family_point(record["stable_side"])
    unstable = family_point(record["unstable_side"])
    return localize_critical_point(
        stable,
        unstable,
        event_mode=event_mode,
        m2_tolerance=5e-9,
        event_tolerance=5e-8,
    )


def serialize_localized(point) -> dict:
    p, f = point.sample.point, point.sample.floquet
    return {
        "masses": p.masses,
        "x1": p.x1,
        "v1": p.v1,
        "v2": p.v2,
        "period": p.period,
        "shooting_residual": p.residual_norm,
        "event_mode": point.event_mode,
        "event_value": point.event_value,
        "source_bracket_width": point.source_width,
        "stability_score": point.sample.score,
        "alpha": f.alpha,
        "beta": f.beta,
        "discriminant": f.discriminant,
        "trace_roots": [[z.real, z.imag] for z in f.trace_roots],
    }


def serialize_point(point) -> dict:
    p, f = point.sample.point, point.sample.floquet
    return {
        "masses": p.masses,
        "x1": p.x1,
        "v1": p.v1,
        "v2": p.v2,
        "period": p.period,
        "shooting_residual": p.residual_norm,
        "event_mode": point.event_mode,
        "event_value": point.event_value,
        "stability_score": point.sample.score,
        "alpha": f.alpha,
        "beta": f.beta,
        "discriminant": f.discriminant,
        "trace_roots": [[z.real, z.imag] for z in f.trace_roots],
        "scaled_tangent": point.tangent_scaled,
        "arclength_residual": point.arclength_residual,
        "normalized_step": point.normalized_step,
        "nfev": point.nfev,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("refined_json")
    parser.add_argument("output")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--arclength-step", type=float, default=4e-3)
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

        first = localize_record(records[0])
        second = localize_record(records[1], event_mode=first.event_mode)
        trace = trace_augmented_critical(
            first,
            second,
            steps=args.steps,
            normalized_step=args.arclength_step,
        )
        components[orientation] = {
            "event_mode": first.event_mode,
            "seed_m1": [records[0]["m1"], records[1]["m1"]],
            "localized_seeds": [serialize_localized(first), serialize_localized(second)],
            "points": [serialize_point(p) for p in trace.points],
            "stopped_reason": trace.stopped_reason,
        }

    out = {
        "claim_status": (
            "screening-only augmented pseudo-arclength critical curves; "
            "independent BigFloat and canonical mechanism verification required"
        ),
        "components": components,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                k: {
                    "event_mode": v["event_mode"],
                    "points": len(v["points"]),
                    "stopped_reason": v["stopped_reason"],
                }
                for k, v in components.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
