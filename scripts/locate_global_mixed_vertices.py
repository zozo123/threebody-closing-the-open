#!/usr/bin/env python3
"""Find preimages of the exact mixed Floquet vertex (alpha,beta)=(4,4).

Representative event localization shows several `+1 <-> -1` mechanism switches
on the sampled U->S stability network.  A genuine switch must pass through the
universal trace-root vertex {t1,t2}={+2,-2}, hence exactly `(alpha,beta)=(4,4)`,
unless a coarse S/U cell hides several unrelated event zeros.

This script first performs a sparse coarse scan of U->S transition cells to find
mass regions whose corrected midpoint orbit lies near `(4,4)`.  It then solves
`alpha=4, beta=4` in locally bounded mass boxes from the best separated seeds.
All output is float64 screening and requires independent BigFloat reproduction.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from threebody_atlas.boundary import evaluate
from threebody_atlas.liao_family import FamilyPoint, correct_family_point

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


def corrected_at(row: dict[str, str], pair: np.ndarray, guess=None):
    pguess = predictor(row) if guess is None else guess
    point = correct_family_point(
        (float(pair[0]), float(pair[1]), float(row["m3"])),
        pguess,
        max_nfev=65,
    )
    if not point.success or point.residual_norm > 2e-7:
        raise RuntimeError(f"periodic correction failed at {tuple(pair)}: {point.residual_norm:.3e}")
    return point, evaluate(point)


def coarse_score(row: dict[str, str]) -> dict | None:
    pair = midpoint(row)
    try:
        point, sample = corrected_at(row, pair)
    except RuntimeError:
        return None
    vector = np.asarray([sample.floquet.alpha, sample.floquet.beta])
    return {
        "row": row,
        "pair": pair,
        "guess": (point.x1, point.v1, point.v2, point.period),
        "alpha": sample.floquet.alpha,
        "beta": sample.floquet.beta,
        "distance": float(np.linalg.norm((vector - TARGET) / SCALE)),
    }


def solve_seed(seed: dict) -> dict | None:
    row = seed["row"]
    center = seed["pair"]
    anchor_guess = seed["guess"]
    # Local boxes are intentionally small enough to identify the switch near the
    # scanned critical branch rather than allowing all seeds to converge to the
    # same distant root.
    lower = np.asarray([max(0.8, center[0] - 0.035), max(0.7, center[1] - 0.06)])
    upper = np.asarray([min(1.1, center[0] + 0.035), min(1.2, center[1] + 0.06)])
    cache = {}

    def corrected(pair):
        key = (float(pair[0]), float(pair[1]))
        if key not in cache:
            cache[key] = corrected_at(row, np.asarray(key), guess=anchor_guess)
        return cache[key]

    def residual(pair):
        try:
            _, sample = corrected(pair)
        except RuntimeError:
            return np.asarray([100.0, 100.0]) + 10.0 * (pair - center)
        return (
            np.asarray([sample.floquet.alpha, sample.floquet.beta]) - TARGET
        ) / SCALE

    fit = least_squares(
        residual,
        center,
        bounds=(lower, upper),
        x_scale=np.asarray([0.02, 0.04]),
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
        max_nfev=24,
    )
    try:
        point, sample = corrected(fit.x)
    except RuntimeError:
        return None
    raw_error = float(
        np.linalg.norm(np.asarray([sample.floquet.alpha, sample.floquet.beta]) - TARGET)
    )
    if not fit.success or raw_error > 2e-4 or point.residual_norm > 2e-7:
        return None
    return {
        "masses": point.masses,
        "x1": point.x1,
        "v1": point.v1,
        "v2": point.v2,
        "period": point.period,
        "shooting_residual": point.residual_norm,
        "alpha": sample.floquet.alpha,
        "beta": sample.floquet.beta,
        "discriminant": sample.floquet.discriminant,
        "trace_roots": [[z.real, z.imag] for z in sample.floquet.trace_roots],
        "mixed_vertex_error": raw_error,
        "source_seed_mass": center.tolist(),
        "source_seed_distance": seed["distance"],
        "outer_nfev": int(fit.nfev),
    }


def separated_best(scans: list[dict], keep: int = 18) -> list[dict]:
    selected = []
    for scan in sorted(scans, key=lambda x: x["distance"]):
        pair = scan["pair"]
        if all(np.linalg.norm(pair - other["pair"]) >= 0.02 for other in selected):
            selected.append(scan)
        if len(selected) >= keep:
            break
    return selected


def dedupe(candidates: list[dict], tolerance: float = 5e-4) -> list[dict]:
    out = []
    for candidate in sorted(candidates, key=lambda x: x["mixed_vertex_error"]):
        pair = np.asarray(candidate["masses"][:2])
        if all(np.linalg.norm(pair - np.asarray(other["masses"][:2])) > tolerance for other in out):
            out.append(candidate)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("brackets_tsv")
    parser.add_argument("output")
    parser.add_argument("--stride", type=int, default=8)
    args = parser.parse_args()

    with Path(args.brackets_tsv).open(encoding="utf-8") as handle:
        rows = [
            row for row in csv.DictReader(handle, delimiter="\t")
            if row["left_label"] == "U" and row["right_label"] == "S"
        ]
    rows.sort(key=lambda row: (float(row["m1"]), midpoint(row)[1]))
    sampled = rows[:: args.stride]
    # Always include all U->S cells near the secondary lobe and domain endpoints.
    sampled.extend(
        row for row in rows
        if 0.99 <= float(row["m1"]) <= 1.05 or float(row["m1"]) <= 0.82 or float(row["m1"]) >= 1.06
    )
    unique = {}
    for row in sampled:
        unique[(row["m1"], row["left_m2"], row["right_m2"])] = row
    scans = [score for row in unique.values() if (score := coarse_score(row)) is not None]
    seeds = separated_best(scans)
    raw = [candidate for seed in seeds if (candidate := solve_seed(seed)) is not None]
    candidates = dedupe(raw)

    payload = {
        "target_alpha_beta": TARGET.tolist(),
        "u_to_s_brackets": len(rows),
        "coarse_scans": len(scans),
        "optimizer_seeds": [
            {
                "mass_pair": seed["pair"].tolist(),
                "alpha": seed["alpha"],
                "beta": seed["beta"],
                "normalized_distance": seed["distance"],
            }
            for seed in seeds
        ],
        "mixed_vertex_candidates": candidates,
        "claim_status": "float64 exact-invariant mixed-vertex candidates; independent BigFloat reproduction required",
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "coarse_scans": len(scans),
        "optimizer_seeds": len(seeds),
        "mixed_vertex_candidates": [
            {
                "masses": c["masses"],
                "error": c["mixed_vertex_error"],
                "source_seed_mass": c["source_seed_mass"],
            }
            for c in candidates
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
