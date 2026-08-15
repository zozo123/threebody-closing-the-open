#!/usr/bin/env python3
"""Find physical-sheet preimages of the mixed Floquet vertex (alpha,beta)=(4,4).

Representative event localization shows several ``+1 <-> -1`` mechanism
switches on the sampled U->S stability network. A genuine switch on one smooth
critical sheet must pass through the universal trace-root vertex
``{t1,t2}={+2,-2}``, hence exactly ``(alpha,beta)=(4,4)``, unless the coarse S/U
cell contains multiple distinct critical zeros.

This driver intentionally separates discovery from correction:

1. A sparse SciPy scan corrects published transition-cell midpoints and ranks
   them only as candidate seeds by distance to ``(4,4)``.
2. Final screening solves are *direct six-variable augmented solves* in
   ``(x1,v1,v2,T,m1,m2)`` using :func:`solve_direct_vertex`.
3. SciPy/DOP853 remains the residual/value oracle while audited JAX/Diffrax x64
   supplies derivative blocks.

No nested mass-only optimizer is used for the final vertex solve. Results are
still float64 screening and require independent BigFloat/canonical reproduction
before they can become release claims.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from threebody_atlas.boundary import evaluate
from threebody_atlas.hybrid_vertices import solve_direct_vertex
from threebody_atlas.liao_family import correct_family_point

TARGET = np.asarray([4.0, 4.0], dtype=float)
SCALE = np.asarray([4.0, 16.0], dtype=float)


def midpoint(row: dict[str, str]) -> np.ndarray:
    return np.asarray(
        [float(row["m1"]), 0.5 * (float(row["left_m2"]) + float(row["right_m2"]))],
        dtype=float,
    )


def predictor(row: dict[str, str]) -> tuple[float, float, float, float]:
    return (
        0.5 * (float(row["left_x1"]) + float(row["right_x1"])),
        0.5 * (float(row["left_v1"]) + float(row["right_v1"])),
        0.5 * (float(row["left_v2"]) + float(row["right_v2"])),
        0.5 * (float(row["left_period"]) + float(row["right_period"])),
    )


def coarse_score(row: dict[str, str]) -> dict | None:
    pair = midpoint(row)
    point = correct_family_point(
        (float(pair[0]), float(pair[1]), float(row["m3"])),
        predictor(row),
        max_nfev=65,
    )
    if not point.success or point.residual_norm > 2e-7:
        return None
    sample = evaluate(point)
    vector = np.asarray([sample.floquet.alpha, sample.floquet.beta], dtype=float)
    return {
        "row": row,
        "pair": pair,
        "point": point,
        "alpha": float(sample.floquet.alpha),
        "beta": float(sample.floquet.beta),
        "distance": float(np.linalg.norm((vector - TARGET) / SCALE)),
    }


def separated_best(scans: list[dict], keep: int) -> list[dict]:
    selected: list[dict] = []
    for scan in sorted(scans, key=lambda x: x["distance"]):
        pair = scan["pair"]
        if all(np.linalg.norm(pair - other["pair"]) >= 0.015 for other in selected):
            selected.append(scan)
        if len(selected) >= keep:
            break
    return selected


def solve_seed(seed: dict, max_nfev: int) -> dict:
    point = seed["point"]
    m1, m2, m3 = point.masses
    y0 = np.asarray([point.x1, point.v1, point.v2, point.period, m1, m2], dtype=float)
    mass_bounds = (
        (max(0.75, m1 - 0.045), min(1.15, m1 + 0.045)),
        (max(0.65, m2 - 0.075), min(1.25, m2 + 0.075)),
    )
    result = solve_direct_vertex(
        y0,
        "mixed_plus_minus_one",
        m3=float(m3),
        mass_bounds=mass_bounds,
        max_nfev=max_nfev,
    )
    p = result.point
    return {
        "masses": [float(x) for x in p.masses],
        "x1": float(p.x1),
        "v1": float(p.v1),
        "v2": float(p.v2),
        "period": float(p.period),
        "shooting_residual": float(p.residual_norm),
        "alpha": float(result.alpha),
        "beta": float(result.beta),
        "discriminant": float(result.discriminant),
        "event_values": {
            "plus_one": float(result.event_values[0]),
            "minus_one": float(result.event_values[1]),
        },
        "mixed_vertex_error": float(result.invariant_error),
        "source_seed_mass": [float(x) for x in seed["pair"]],
        "source_seed_distance": float(seed["distance"]),
        "direct_nfev": int(result.nfev),
        "optimality": float(result.optimality),
        "cost": float(result.cost),
        "mass_bounds": [list(mass_bounds[0]), list(mass_bounds[1])],
    }


def dedupe(candidates: list[dict], tolerance: float = 5e-4) -> list[dict]:
    out: list[dict] = []
    for candidate in sorted(candidates, key=lambda x: x["mixed_vertex_error"]):
        pair = np.asarray(candidate["masses"][:2], dtype=float)
        if all(
            np.linalg.norm(pair - np.asarray(other["masses"][:2], dtype=float)) > tolerance
            for other in out
        ):
            out.append(candidate)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("brackets_tsv")
    parser.add_argument("output")
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--keep", type=int, default=10)
    parser.add_argument("--max-nfev", type=int, default=40)
    args = parser.parse_args()
    if args.stride < 1 or args.keep < 1 or args.max_nfev < 1:
        raise SystemExit("stride, keep, and max-nfev must be positive")

    with Path(args.brackets_tsv).open(encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle, delimiter="\t")
            if row["left_label"] == "U" and row["right_label"] == "S"
        ]
    rows.sort(key=lambda row: (float(row["m1"]), midpoint(row)[1]))

    sampled = rows[:: args.stride]
    # Mechanism switches are already seen on the principal lower track and near
    # the secondary lobe. Always retain those regions even when the sparse stride
    # would skip them.
    sampled.extend(
        row
        for row in rows
        if 0.985 <= float(row["m1"]) <= 1.055
        or float(row["m1"]) <= 0.83
        or float(row["m1"]) >= 1.055
    )
    unique: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in sampled:
        unique[(row["m1"], row["left_m2"], row["right_m2"])] = row

    scans: list[dict] = []
    scan_failures = 0
    for row in unique.values():
        try:
            score = coarse_score(row)
        except Exception:  # discovery scan: preserve failures in aggregate only
            score = None
        if score is None:
            scan_failures += 1
        else:
            scans.append(score)

    seeds = separated_best(scans, args.keep)
    raw: list[dict] = []
    solve_failures: list[dict] = []
    for seed in seeds:
        try:
            raw.append(solve_seed(seed, args.max_nfev))
        except Exception as exc:
            solve_failures.append(
                {
                    "source_seed_mass": [float(x) for x in seed["pair"]],
                    "source_seed_distance": float(seed["distance"]),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    candidates = dedupe(raw)
    payload = {
        "target_alpha_beta": TARGET.tolist(),
        "architecture": (
            "SciPy corrected-midpoint seed scan; direct six-variable augmented vertex solve; "
            "SciPy residual values; JAX/Diffrax x64 derivative oracle"
        ),
        "u_to_s_brackets": len(rows),
        "coarse_rows_attempted": len(unique),
        "coarse_scans": len(scans),
        "coarse_scan_failures": scan_failures,
        "optimizer_seeds": [
            {
                "mass_pair": [float(x) for x in seed["pair"]],
                "alpha": seed["alpha"],
                "beta": seed["beta"],
                "normalized_distance": seed["distance"],
            }
            for seed in seeds
        ],
        "direct_solve_failures": solve_failures,
        "mixed_vertex_candidates": candidates,
        "claim_status": (
            "float64 direct augmented mixed-vertex candidates; independent BigFloat/canonical "
            "reproduction required"
        ),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "coarse_scans": len(scans),
                "optimizer_seeds": len(seeds),
                "direct_failures": len(solve_failures),
                "mixed_vertex_candidates": [
                    {
                        "masses": c["masses"],
                        "error": c["mixed_vertex_error"],
                        "events": c["event_values"],
                    }
                    for c in candidates
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
