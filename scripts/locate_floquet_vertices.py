#!/usr/bin/env python3
"""Locate exact codimension-two vertices of the reduced Floquet stability domain.

For the two nontrivial reciprocal multiplier pairs, spectral stability is
controlled by trace roots t1,t2.  The three vertices of the universal stable
trace-root domain are

  t1=t2=-2  -> (alpha,beta)=(0,-4),
  {t1,t2}={-2,+2} -> (alpha,beta)=(4,4),
  t1=t2=+2  -> (alpha,beta)=(8,28).

This script searches for preimages of those exact invariant vertices near the
birth and death of the secondary stable lobe.  Every mass-plane residual
evaluation first Newton-corrects a periodic orbit from a nearby published
transition anchor; candidates must satisfy both shooting closure and invariant
residual gates.

Float64 solutions are organizer *candidates*.  Any accepted candidate must be
reproduced by the independent BigFloat canonical path before publication.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from threebody_atlas.boundary import evaluate
from threebody_atlas.liao_family import FamilyPoint, correct_family_point


VERTICES = {
    "double_minus_one": (0.0, -4.0),
    "mixed_plus_minus_one": (4.0, 4.0),
    "double_plus_one": (8.0, 28.0),
}


@dataclass(frozen=True)
class SearchWindow:
    name: str
    m1_bounds: tuple[float, float]
    m2_bounds: tuple[float, float]
    seed_m1_bounds: tuple[float, float]


WINDOWS = (
    SearchWindow("secondary_birth", (0.994, 0.9985), (0.94, 1.02), (0.995, 0.997)),
    SearchWindow("secondary_death", (1.040, 1.0445), (1.02, 1.07), (1.041, 1.043)),
)


def midpoint(row: dict[str, str]) -> tuple[float, float]:
    return float(row["m1"]), 0.5 * (float(row["left_m2"]) + float(row["right_m2"]))


def anchor_point(row: dict[str, str]) -> FamilyPoint:
    # Average the two adjacent published periodic solutions only as a predictor;
    # the first operation at every trial mass is a full Newton correction.
    return FamilyPoint(
        masses=(
            float(row["m1"]),
            0.5 * (float(row["left_m2"]) + float(row["right_m2"])),
            float(row["m3"]),
        ),
        x1=0.5 * (float(row["left_x1"]) + float(row["right_x1"])),
        v1=0.5 * (float(row["left_v1"]) + float(row["right_v1"])),
        v2=0.5 * (float(row["left_v2"]) + float(row["right_v2"])),
        period=0.5 * (float(row["left_period"]) + float(row["right_period"])),
        residual_norm=float("nan"),
        nfev=0,
        success=True,
    )


def params(point: FamilyPoint) -> tuple[float, float, float, float]:
    return point.x1, point.v1, point.v2, point.period


def solve_from_seed(
    row: dict[str, str],
    window: SearchWindow,
    vertex_name: str,
    target: tuple[float, float],
) -> dict | None:
    anchor = anchor_point(row)
    m3 = anchor.masses[2]
    target_alpha, target_beta = target
    # Scale residuals so alpha and beta contribute comparably near all vertices.
    scale = np.asarray([4.0, 16.0], dtype=float)
    cache: dict[tuple[float, float], tuple[FamilyPoint, object]] = {}

    def corrected(pair: np.ndarray):
        key = (float(pair[0]), float(pair[1]))
        if key in cache:
            return cache[key]
        point = correct_family_point(
            (key[0], key[1], m3),
            params(anchor),
            max_nfev=70,
        )
        if not point.success or point.residual_norm > 2e-7:
            raise RuntimeError(f"periodic correction failed at {key}: {point.residual_norm:.3e}")
        sample = evaluate(point)
        cache[key] = (point, sample)
        return point, sample

    def residual(pair: np.ndarray) -> np.ndarray:
        try:
            _, sample = corrected(pair)
        except RuntimeError:
            # Smooth enough penalty to let bounded least_squares back away from a
            # bad basin without treating a failed shooting solve as evidence.
            center = np.asarray(midpoint(row))
            return np.asarray([100.0, 100.0]) + 10.0 * (pair - center)
        return np.asarray(
            [sample.floquet.alpha - target_alpha, sample.floquet.beta - target_beta],
            dtype=float,
        ) / scale

    start = np.asarray(midpoint(row), dtype=float)
    lower = np.asarray([window.m1_bounds[0], window.m2_bounds[0]], dtype=float)
    upper = np.asarray([window.m1_bounds[1], window.m2_bounds[1]], dtype=float)
    start = np.clip(start, lower + 1e-8, upper - 1e-8)
    fit = least_squares(
        residual,
        start,
        bounds=(lower, upper),
        x_scale=np.asarray([0.003, 0.03]),
        xtol=2e-10,
        ftol=2e-10,
        gtol=2e-10,
        max_nfev=24,
    )
    try:
        point, sample = corrected(fit.x)
    except RuntimeError:
        return None
    da = float(sample.floquet.alpha - target_alpha)
    db = float(sample.floquet.beta - target_beta)
    invariant_error = float(np.hypot(da, db))
    if not fit.success or point.residual_norm > 2e-7 or invariant_error > 2e-4:
        return None
    return {
        "window": window.name,
        "vertex": vertex_name,
        "target_alpha_beta": [target_alpha, target_beta],
        "source_seed": {
            "m1": float(row["m1"]),
            "m2_bracket": [float(row["left_m2"]), float(row["right_m2"])],
            "published_labels": [row["left_label"], row["right_label"]],
        },
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
        "invariant_error": invariant_error,
        "outer_nfev": int(fit.nfev),
    }


def deduplicate(candidates: list[dict], mass_tolerance: float = 2e-4) -> list[dict]:
    accepted: list[dict] = []
    for candidate in sorted(candidates, key=lambda c: c["invariant_error"]):
        pair = np.asarray(candidate["masses"][:2], dtype=float)
        duplicate = False
        for prior in accepted:
            if candidate["window"] != prior["window"] or candidate["vertex"] != prior["vertex"]:
                continue
            other = np.asarray(prior["masses"][:2], dtype=float)
            if np.linalg.norm(pair - other) <= mass_tolerance:
                duplicate = True
                break
        if not duplicate:
            accepted.append(candidate)
    return accepted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("brackets_tsv")
    parser.add_argument("output")
    args = parser.parse_args()

    with Path(args.brackets_tsv).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    raw_candidates = []
    attempts = 0
    for window in WINDOWS:
        seeds = [
            row for row in rows
            if window.seed_m1_bounds[0] <= float(row["m1"]) <= window.seed_m1_bounds[1]
            and window.m2_bounds[0] <= midpoint(row)[1] <= window.m2_bounds[1]
        ]
        # Prefer seeds nearest the organizer box center but keep several distinct
        # transition cells so the result is not tied to one Newton basin.
        center = np.asarray(
            [
                0.5 * sum(window.m1_bounds),
                0.5 * sum(window.m2_bounds),
            ]
        )
        seeds.sort(key=lambda row: np.linalg.norm(np.asarray(midpoint(row)) - center))
        seeds = seeds[:8]
        for vertex_name, target in VERTICES.items():
            for row in seeds:
                attempts += 1
                candidate = solve_from_seed(row, window, vertex_name, target)
                if candidate is not None:
                    raw_candidates.append(candidate)

    candidates = deduplicate(raw_candidates)
    payload = {
        "attempts": attempts,
        "accepted_raw": len(raw_candidates),
        "deduplicated_candidates": len(candidates),
        "vertices": {name: list(value) for name, value in VERTICES.items()},
        "candidates": candidates,
        "claim_status": "float64 codimension-two organizer candidates; BigFloat canonical reproduction required",
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "attempts": attempts,
        "accepted_raw": len(raw_candidates),
        "deduplicated_candidates": [
            {
                "window": c["window"],
                "vertex": c["vertex"],
                "masses": c["masses"],
                "invariant_error": c["invariant_error"],
            }
            for c in candidates
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
