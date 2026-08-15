#!/usr/bin/env python3
"""Trace smooth critical components directly from frozen published S/U brackets.

The published 0.001 mass-grid endpoints are used only to bracket a smooth
Floquet event. Residual values always come from SciPy DOP853 + the existing
variational path. The optional JAX hybrid backend supplies audited derivative
blocks and a nullspace tangent for pseudo-arclength continuation.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from threebody_atlas.critical_manifold import localize_critical_point, trace_augmented_critical
from threebody_atlas.hybrid_critical import trace_hybrid_critical
from threebody_atlas.liao_family import FamilyPoint


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


def localize(row: dict[str, str], event_mode=None):
    left, right = point(row, "left"), point(row, "right")
    stable = left if row["left_label"] == "S" else right
    unstable = left if row["left_label"] == "U" else right
    return localize_critical_point(
        stable,
        unstable,
        event_mode=event_mode,
        m2_tolerance=5e-9,
        event_tolerance=5e-8,
        max_iterations=32,
        max_closure=1e-7,
    )


def serialize_localized(p) -> dict:
    q, f = p.sample.point, p.sample.floquet
    return {
        "masses": q.masses,
        "x1": q.x1,
        "v1": q.v1,
        "v2": q.v2,
        "period": q.period,
        "shooting_residual": q.residual_norm,
        "event_mode": p.event_mode,
        "event_value": p.event_value,
        "source_bracket_width": p.source_width,
        "alpha": f.alpha,
        "beta": f.beta,
        "discriminant": f.discriminant,
        "trace_roots": [[z.real, z.imag] for z in f.trace_roots],
    }


def serialize_step(p, diagnostic=None) -> dict:
    q, f = p.sample.point, p.sample.floquet
    out = {
        "masses": q.masses,
        "x1": q.x1,
        "v1": q.v1,
        "v2": q.v2,
        "period": q.period,
        "shooting_residual": q.residual_norm,
        "event_mode": p.event_mode,
        "event_value": p.event_value,
        "alpha": f.alpha,
        "beta": f.beta,
        "discriminant": f.discriminant,
        "trace_roots": [[z.real, z.imag] for z in f.trace_roots],
        "scaled_tangent": p.tangent_scaled,
        "arclength_residual": p.arclength_residual,
        "normalized_step": p.normalized_step,
        "nfev": p.nfev,
    }
    if diagnostic is not None:
        out["jacobian_null_residual"] = diagnostic.null_residual
        out["jacobian_spectral_gap"] = diagnostic.spectral_gap
        out["jacobian_singular_values"] = diagnostic.singular_values
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("brackets_tsv")
    parser.add_argument("output")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--arclength-step", type=float, default=2e-3)
    parser.add_argument(
        "--derivatives",
        choices=("finite-difference", "jax-hybrid"),
        default="finite-difference",
    )
    args = parser.parse_args()

    with Path(args.brackets_tsv).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[orientation(row)].append(row)
    for records in grouped.values():
        records.sort(key=lambda r: (float(r["m1"]), float(r["left_m2"])))

    components = {}
    for orient in ("U->S", "S->U"):
        records = grouped.get(orient, [])
        if len(records) < 2:
            continue
        first = localize(records[0])
        second = localize(records[1], event_mode=first.event_mode)
        diagnostics = ()
        if args.derivatives == "jax-hybrid":
            trace, diagnostics = trace_hybrid_critical(
                first,
                second,
                steps=args.steps,
                normalized_step=args.arclength_step,
            )
        else:
            trace = trace_augmented_critical(
                first,
                second,
                steps=args.steps,
                normalized_step=args.arclength_step,
            )
        components[orient] = {
            "event_mode": first.event_mode,
            "derivative_backend": args.derivatives,
            "residual_backend": "SciPy DOP853 + analytic variational equations",
            "published_seed_slices": [float(records[0]["m1"]), float(records[1]["m1"])],
            "localized_seeds": [serialize_localized(first), serialize_localized(second)],
            "points": [
                serialize_step(p, diagnostics[i] if i < len(diagnostics) else None)
                for i, p in enumerate(trace.points)
            ],
            "stopped_reason": trace.stopped_reason,
        }

    if not components:
        raise RuntimeError("no critical components could be seeded from published transition brackets")
    if not all(component["points"] for component in components.values()):
        failures = {k: v["stopped_reason"] for k, v in components.items() if not v["points"]}
        raise RuntimeError(f"zero-step critical components are not accepted: {failures}")

    payload = {
        "claim_status": (
            "screening-supported augmented pseudo-arclength curves; SciPy residual values "
            "remain authoritative; independent BigFloat and canonical mechanism verification required"
        ),
        "components": components,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        key: {
            "event_mode": value["event_mode"],
            "derivative_backend": value["derivative_backend"],
            "points": len(value["points"]),
            "stopped_reason": value["stopped_reason"],
        }
        for key, value in components.items()
    }, indent=2))


if __name__ == "__main__":
    main()
