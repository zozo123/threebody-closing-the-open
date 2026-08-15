#!/usr/bin/env python3
"""Locate the mass-plane fold that births the secondary stable lobe.

At m1=0.996 the two new secondary-lobe boundaries are both `-1` Floquet events,
while the pair is absent at m1=0.995.  For the smooth event function

    F_-(m1,m2) = P(-2) = beta - 2 alpha + 4,

a vertical-slice birth/annihilation occurs at a turning point of the event curve
where

    F_- = 0,        dF_-/dm2 = 0.

This script solves those two equations with periodic shooting nested inside,
then evaluates first/second finite derivatives to verify the local quadratic
fold geometry.  Float64 output is a fold candidate; BigFloat reproduction is
required for a release claim.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from threebody_atlas.boundary import evaluate
from threebody_atlas.critical_manifold import event_value
from threebody_atlas.liao_family import correct_family_point


def midpoint(row: dict[str, str]) -> float:
    return 0.5 * (float(row["left_m2"]) + float(row["right_m2"]))


def avg_params(row: dict[str, str]) -> np.ndarray:
    return np.asarray(
        [
            0.5 * (float(row["left_x1"]) + float(row["right_x1"])),
            0.5 * (float(row["left_v1"]) + float(row["right_v1"])),
            0.5 * (float(row["left_v2"]) + float(row["right_v2"])),
            0.5 * (float(row["left_period"]) + float(row["right_period"])),
        ],
        dtype=float,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("brackets_tsv")
    parser.add_argument("output")
    parser.add_argument("--derivative-step", type=float, default=2e-5)
    args = parser.parse_args()

    with Path(args.brackets_tsv).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    slice_rows = [row for row in rows if abs(float(row["m1"]) - 0.996) < 5e-7]
    low_candidates = [row for row in slice_rows if row["left_label"] == "U" and row["right_label"] == "S"]
    high_candidates = [row for row in slice_rows if row["left_label"] == "S" and row["right_label"] == "U"]
    if len(low_candidates) < 2 or len(high_candidates) < 2:
        raise RuntimeError("expected four transitions at m1=0.996")
    # The secondary stable interval is the first U->S followed by the first S->U.
    low = min(low_candidates, key=midpoint)
    high = min(high_candidates, key=midpoint)
    m2_low, m2_high = midpoint(low), midpoint(high)
    p_low, p_high = avg_params(low), avg_params(high)
    h = args.derivative_step
    cache = {}

    def guess_for(m2: float) -> tuple[float, float, float, float]:
        theta = (m2 - m2_low) / (m2_high - m2_low)
        p = (1.0 - theta) * p_low + theta * p_high
        return tuple(float(x) for x in p)

    def sample(pair: np.ndarray):
        key = (float(pair[0]), float(pair[1]))
        if key in cache:
            return cache[key]
        point = correct_family_point(
            (key[0], key[1], 1.0),
            guess_for(key[1]),
            max_nfev=70,
        )
        if not point.success or point.residual_norm > 2e-7:
            raise RuntimeError(f"periodic correction failed at {key}: {point.residual_norm:.3e}")
        result = evaluate(point)
        value = event_value(result.floquet, "minus_one")
        cache[key] = (point, result, value)
        return cache[key]

    def derivatives(pair: np.ndarray):
        p = np.asarray(pair, dtype=float)
        _, _, f0 = sample(p)
        _, _, fp = sample(p + np.asarray([0.0, h]))
        _, _, fm = sample(p - np.asarray([0.0, h]))
        dm2 = (fp - fm) / (2.0 * h)
        d2m2 = (fp - 2.0 * f0 + fm) / (h * h)
        _, _, f1p = sample(p + np.asarray([h, 0.0]))
        _, _, f1m = sample(p - np.asarray([h, 0.0]))
        dm1 = (f1p - f1m) / (2.0 * h)
        return f0, dm1, dm2, d2m2

    scale = np.asarray([1.0, 100.0], dtype=float)

    def residual(pair: np.ndarray) -> np.ndarray:
        try:
            f0, _, dm2, _ = derivatives(pair)
        except RuntimeError:
            return np.asarray([100.0, 100.0])
        return np.asarray([f0, dm2], dtype=float) / scale

    start = np.asarray([0.9955, 0.5 * (m2_low + m2_high)], dtype=float)
    lower = np.asarray([0.9945, m2_low - 0.02])
    upper = np.asarray([0.9965, m2_high + 0.02])
    fit = least_squares(
        residual,
        start,
        bounds=(lower, upper),
        x_scale=np.asarray([0.001, 0.02]),
        xtol=2e-10,
        ftol=2e-10,
        gtol=2e-10,
        max_nfev=28,
    )
    point, floquet, f0 = sample(fit.x)
    f0, dm1, dm2, d2m2 = derivatives(fit.x)
    if not fit.success or abs(f0) > 2e-5 or abs(dm2) > 2e-2:
        raise RuntimeError(
            f"minus-one fold solve failed gates: F={f0:.3e} dFdm2={dm2:.3e} message={fit.message}"
        )
    # Generic quadratic fold requires a transverse parameter derivative and a
    # nonzero second derivative in the turning direction.
    generic_fold = abs(dm1) > 1e-2 and abs(d2m2) > 1.0
    if not generic_fold:
        raise RuntimeError(
            f"candidate is not a resolved generic fold: dFdm1={dm1:.3e} d2Fdm2={d2m2:.3e}"
        )

    payload = {
        "event_mode": "minus_one",
        "equations": ["P(-2)=0", "dP(-2)/dm2=0"],
        "masses": point.masses,
        "x1": point.x1,
        "v1": point.v1,
        "v2": point.v2,
        "period": point.period,
        "shooting_residual": point.residual_norm,
        "event_value": f0,
        "d_event_dm1": dm1,
        "d_event_dm2": dm2,
        "d2_event_dm2": d2m2,
        "alpha": floquet.floquet.alpha,
        "beta": floquet.floquet.beta,
        "discriminant": floquet.floquet.discriminant,
        "trace_roots": [[z.real, z.imag] for z in floquet.floquet.trace_roots],
        "generic_quadratic_fold_screen": True,
        "derivative_step": h,
        "outer_nfev": int(fit.nfev),
        "source_secondary_brackets": {
            "lower": [float(low["left_m2"]), float(low["right_m2"])],
            "upper": [float(high["left_m2"]), float(high["right_m2"])],
        },
        "claim_status": "float64 mass-plane event-fold candidate; independent BigFloat reproduction required",
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "masses": point.masses,
        "event_value": f0,
        "d_event_dm1": dm1,
        "d_event_dm2": dm2,
        "d2_event_dm2": d2m2,
        "shooting_residual": point.residual_norm,
        "generic_quadratic_fold_screen": True,
    }, indent=2))


if __name__ == "__main__":
    main()
