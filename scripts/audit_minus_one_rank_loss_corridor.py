#!/usr/bin/env python3
"""Audit the Float64-dropped G- corridor with a cancellation-free event.

For a Floquet ``-1`` event, the physically defining statement is that the
monodromy has eigenvalue -1.  Equivalently,

    det(M + I) = 0.

Unlike ``G- = beta - 2*alpha + 4``, this direct rank-loss condition does not
subtract large trace invariants.  The four neutral symmetry multipliers near
+1 are harmless: in ``M+I`` they contribute factors near 2, not zeros.

This script uses the raw full-domain sweep only for *brackets and initial
shooting charts*.  On each low-m2 slice from m1=1.042 through 1.074 it:

1. re-corrects the periodic orbit at both m2 bracket endpoints;
2. verifies the direct determinant event changes sign there;
3. solves det(M+I)=0 with Brent on the periodically corrected family;
4. records sigma_min(M+I), closure, and the historical trace-polynomial G- at
   the rank-loss root.

The trace-polynomial residual is diagnostic only here.  A large Float64 G- at a
point where det(M+I)=0 and sigma_min(M+I) collapses is direct evidence of
cancellation in the trace estimator, not disappearance of the -1 critical
curve.  No gate is relaxed and this audit does not write the release graph.
"""
from __future__ import annotations

import argparse
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import brentq

from threebody_atlas.liao_family import FamilyPoint, correct_family_point, state_from_chart
from threebody_atlas.reduced import compute_reduced_floquet

CORRECT_RTOL = 2e-12
CORRECT_ATOL = 2e-14
FLOQUET_RTOL = 5e-13
FLOQUET_ATOL = 5e-15
MAX_CLOSURE = 1e-7
M1_START = 1.042
M1_STOP = 1.074
M1_STEP = 0.002


def load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path}: JSON root must be an object")
    return payload


def trace_minus_event(monodromy: np.ndarray) -> float:
    alpha = float(np.trace(monodromy))
    beta = float(0.5 * (alpha * alpha - np.trace(monodromy @ monodromy)))
    return float(beta - 2.0 * alpha + 4.0)


def rank_event(monodromy: np.ndarray) -> float:
    """Scaled direct characteristic event; zero iff M has multiplier -1."""
    sign, logabs = np.linalg.slogdet(monodromy + np.eye(monodromy.shape[0]))
    if sign == 0.0:
        return 0.0
    # Division by 16 matches the exact neutral-factor normalization when the
    # four symmetry multipliers are +1.  Only the zero and sign are used for
    # root-finding; scale is reported for comparison with historical G-.
    return float(sign * math.exp(float(logabs)) / 16.0)


def corrected_floquet(
    m1: float,
    m2: float,
    chart: tuple[float, float, float, float],
) -> tuple[FamilyPoint, np.ndarray, float, float, float]:
    masses = (float(m1), float(m2), 1.0)
    point = correct_family_point(
        masses,
        chart,
        max_nfev=100,
        screening_rtol=CORRECT_RTOL,
        screening_atol=CORRECT_ATOL,
    )
    if not point.success or point.residual_norm > MAX_CLOSURE:
        raise RuntimeError(
            f"periodic correction failed at ({m1:.6f},{m2:.12f}): "
            f"closure={point.residual_norm:.3e}"
        )
    floquet = compute_reduced_floquet(
        state_from_chart(point.masses, point.x1, point.v1, point.v2),
        np.asarray(point.masses, dtype=float),
        point.period,
        rtol=FLOQUET_RTOL,
        atol=FLOQUET_ATOL,
    )
    matrix = np.asarray(floquet.monodromy, dtype=float)
    direct = rank_event(matrix)
    sigma = float(np.linalg.svd(matrix + np.eye(8), compute_uv=False)[-1])
    trace = trace_minus_event(matrix)
    return point, matrix, direct, sigma, trace


def corridor_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in payload.get("localizations", []):
        if str(row.get("event_mode") or row.get("mechanism")) != "minus_one":
            continue
        masses = row.get("masses")
        bracket = row.get("m2_bracket")
        value_bracket = row.get("value_bracket")
        if not isinstance(masses, list) or len(masses) < 3:
            continue
        if not isinstance(bracket, list) or len(bracket) != 2:
            continue
        if not isinstance(value_bracket, list) or len(value_bracket) != 2:
            continue
        m1, m2 = float(masses[0]), float(masses[1])
        if not (M1_START - 1e-12 <= m1 <= M1_STOP + 1e-12 and m2 < 0.9):
            continue
        if str(row.get("status")) not in {"passed", "missed_frozen_gates"}:
            continue
        if any(row.get(key) is None for key in ("x1", "v1", "v2", "period")):
            continue
        rows.append(row)
    rows.sort(key=lambda row: float(row["masses"][0]))
    expected = [round(M1_START + M1_STEP * i, 3) for i in range(17)]
    observed = [round(float(row["masses"][0]), 3) for row in rows]
    if observed != expected:
        raise RuntimeError(f"expected contiguous low-minus corridor {expected}, got {observed}")
    return rows


def audit_row(row: dict[str, Any]) -> dict[str, Any]:
    m1 = float(row["masses"][0])
    raw_m2 = float(row["masses"][1])
    low, high = (float(x) for x in row["m2_bracket"])
    raw_values = [float(x) for x in row["value_bracket"]]
    if raw_values[0] * raw_values[1] >= 0.0:
        raise RuntimeError(f"m1={m1:.3f}: raw event bracket no longer changes sign")
    chart = tuple(float(row[key]) for key in ("x1", "v1", "v2", "period"))

    @lru_cache(maxsize=64)
    def evaluate(m2: float):
        point, matrix, direct, sigma, trace = corrected_floquet(m1, float(m2), chart)
        return point, direct, sigma, trace

    low_point, low_direct, low_sigma, low_trace = evaluate(low)
    high_point, high_direct, high_sigma, high_trace = evaluate(high)
    if low_direct == 0.0:
        root_m2 = low
    elif high_direct == 0.0:
        root_m2 = high
    else:
        if low_direct * high_direct >= 0.0:
            raise RuntimeError(
                f"m1={m1:.3f}: direct det(M+I) event does not bracket zero: "
                f"{low_direct:.6e}, {high_direct:.6e}"
            )
        root_m2 = float(
            brentq(
                lambda x: evaluate(float(x))[1],
                low,
                high,
                xtol=2e-11,
                rtol=2e-13,
                maxiter=40,
            )
        )
    root_point, root_direct, root_sigma, root_trace = evaluate(root_m2)
    endpoint_sigma = min(low_sigma, high_sigma)
    contrast = float(endpoint_sigma / max(root_sigma, np.finfo(float).tiny))
    return {
        "m1": m1,
        "raw_status": row.get("status"),
        "raw_localized_m2": raw_m2,
        "raw_trace_event": float(row.get("event_value") or 0.0),
        "raw_closure": float(row.get("closure") or 0.0),
        "raw_value_bracket": raw_values,
        "m2_bracket": [low, high],
        "direct_endpoint_events": [low_direct, high_direct],
        "direct_endpoint_sigma_min_M_plus_I": [low_sigma, high_sigma],
        "direct_endpoint_trace_G_minus": [low_trace, high_trace],
        "rank_loss_root": {
            "m2": root_m2,
            "delta_m2_from_raw_localizer": float(root_m2 - raw_m2),
            "periodic_closure": float(root_point.residual_norm),
            "det_M_plus_I_over_16": root_direct,
            "sigma_min_M_plus_I": root_sigma,
            "trace_G_minus_float64": root_trace,
            "sigma_contrast_to_nearest_bracket_endpoint": contrast,
            "chart": {
                "x1": float(root_point.x1),
                "v1": float(root_point.v1),
                "v2": float(root_point.v2),
                "period": float(root_point.period),
            },
        },
        "rank_loss_passed": bool(
            root_point.residual_norm <= MAX_CLOSURE
            and root_sigma <= 2e-8
            and contrast >= 100.0
            and abs(root_m2 - raw_m2) <= 5e-5
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep")
    parser.add_argument("output")
    args = parser.parse_args()

    rows = corridor_rows(load(Path(args.sweep)))
    results = []
    for index, row in enumerate(rows, start=1):
        result = audit_row(row)
        results.append(result)
        root = result["rank_loss_root"]
        print(
            json.dumps(
                {
                    "slice": f"{index}/{len(rows)}",
                    "m1": result["m1"],
                    "raw_status": result["raw_status"],
                    "raw_trace_event": result["raw_trace_event"],
                    "root_m2": root["m2"],
                    "delta_m2": root["delta_m2_from_raw_localizer"],
                    "sigma_min": root["sigma_min_M_plus_I"],
                    "trace_G_minus_at_rank_root": root["trace_G_minus_float64"],
                    "passed": result["rank_loss_passed"],
                }
            ),
            flush=True,
        )

    passed = all(item["rank_loss_passed"] for item in results)
    payload = {
        "schema": "atlas.v1.minus-one-rank-loss-audit/1",
        "claim": "the Float64-dropped 1.042..1.074 low-m2 corridor remains a direct -1 Floquet rank-loss curve",
        "method": {
            "semantic_event": "monodromy has multiplier -1",
            "direct_equation": "det(M + I) = 0",
            "conditioning_diagnostic": "sigma_min(M + I)",
            "historical_equivalent": "G_minus = beta - 2*alpha + 4 (reported but not used for root solve)",
            "periodic_family": "variational Newton correction in the published Li-Li-Liao chart",
        },
        "float64_tolerances": {
            "correct_rtol": CORRECT_RTOL,
            "correct_atol": CORRECT_ATOL,
            "floquet_rtol": FLOQUET_RTOL,
            "floquet_atol": FLOQUET_ATOL,
            "maximum_periodic_closure": MAX_CLOSURE,
        },
        "results": results,
        "summary": {
            "slices": len(results),
            "historical_float64_gate_misses": sum(
                1 for item in results if item["raw_status"] == "missed_frozen_gates"
            ),
            "rank_loss_passes": sum(1 for item in results if item["rank_loss_passed"]),
            "max_abs_delta_m2_from_raw_localizer": max(
                abs(item["rank_loss_root"]["delta_m2_from_raw_localizer"]) for item in results
            ),
            "max_sigma_min_M_plus_I": max(
                item["rank_loss_root"]["sigma_min_M_plus_I"] for item in results
            ),
            "max_abs_trace_G_minus_at_rank_root": max(
                abs(item["rank_loss_root"]["trace_G_minus_float64"]) for item in results
            ),
            "passed": passed,
        },
        "claim_status": (
            "direct_rank_loss_reconciles_the_low_minus_corridor"
            if passed
            else "rank_loss_audit_unresolved"
        ),
        "note": (
            "A pass is a cancellation-free Float64 structural witness for the -1 event. "
            "It does not replace the independent BigFloat lane and does not alter the frozen trace-event gate."
        ),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2), flush=True)
    raise SystemExit(0 if passed else 3)


if __name__ == "__main__":
    main()
