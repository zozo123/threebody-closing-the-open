#!/usr/bin/env python3
"""Cross-check label-invisible +1 roots through the rank jump of M-I.

A regular strict periodic orbit on the translation-reduced atlas sheet carries
two geometric neutral +1 directions (time/energy and rotation/angular momentum),
embedded in algebraic Jordan chains.  Thus M-I generically has two tiny
singular values.  A physical +1 Floquet event adds a nontrivial +1 direction;
the *third-smallest* singular value must collapse.

This diagnostic avoids the cancellation in G+ = beta - 6*alpha + 20.  It is a
structural root/conditioning cross-check only: the independent Julia BigFloat
lane remains the arithmetic release evidence.
"""
from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar

from threebody_atlas.floquet_rank_events import plus_one_rank_jump
from threebody_atlas.liao_family import correct_family_point, state_from_chart
from threebody_atlas.reduced import compute_reduced_floquet

CORRECT_RTOL = 2e-12
CORRECT_ATOL = 2e-14
FLOQUET_RTOL = 5e-13
FLOQUET_ATOL = 5e-15
MAX_CLOSURE = 1e-7


def load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path}: JSON root must be an object")
    return payload


def trace_plus_event(monodromy: np.ndarray) -> float:
    alpha = float(np.trace(monodromy))
    beta = float(0.5 * (alpha * alpha - np.trace(monodromy @ monodromy)))
    return float(beta - 6.0 * alpha + 20.0)


def plus_targets(payload: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for slc in payload.get("slices", []):
        for row in slc.get("brackets", []):
            if row.get("event_mode") != "plus_one":
                continue
            if row.get("reachable_by_published_label_pipeline") is not False:
                continue
            cert = row.get("certification")
            if not isinstance(cert, dict) or cert.get("status") != "passed":
                continue
            out.append(row)
    out.sort(key=lambda row: float(row["m1"]))
    if len(out) != 4:
        raise RuntimeError(f"expected four label-invisible plus-one roots, got {len(out)}")
    return out


def audit(row: dict[str, Any]) -> dict[str, Any]:
    cert = row["certification"]
    m1 = float(row["m1"])
    low, high = (float(x) for x in row["m2_bracket"])
    raw_events = [float(x) for x in row["event_values"]]
    raw_m2 = float(cert["masses"][1])
    chart = tuple(float(cert[key]) for key in ("x1", "v1", "v2", "period"))

    @lru_cache(maxsize=64)
    def evaluate(m2: float):
        point = correct_family_point(
            (m1, float(m2), 1.0),
            chart,
            max_nfev=100,
            screening_rtol=CORRECT_RTOL,
            screening_atol=CORRECT_ATOL,
        )
        if not point.success or point.residual_norm > MAX_CLOSURE:
            raise RuntimeError(
                f"periodic correction failed at m1={m1:.3f}, m2={m2:.12f}: "
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
        extra, singular = plus_one_rank_jump(matrix, neutral_geometric_nullity=2)
        return point, float(extra), singular, trace_plus_event(matrix)

    low_point, low_extra, low_singular, low_trace = evaluate(low)
    high_point, high_extra, high_singular, high_trace = evaluate(high)
    raw_point, raw_extra, raw_singular, raw_trace = evaluate(raw_m2)

    # The structural event is an unsigned rank minimum.  Search only inside the
    # already sign-bracketed G+ cell; the bracket supplies event identity and
    # prevents the optimizer from wandering to another singularity.
    result = minimize_scalar(
        lambda x: np.log10(max(evaluate(float(x))[1], np.finfo(float).tiny)),
        bounds=(low, high),
        method="bounded",
        options={"xatol": 2e-10, "maxiter": 40},
    )
    minimum_m2 = float(result.x)
    min_point, min_extra, min_singular, min_trace = evaluate(minimum_m2)
    endpoint_floor = min(low_extra, high_extra)
    contrast = float(endpoint_floor / max(min_extra, np.finfo(float).tiny))
    raw_contrast = float(endpoint_floor / max(raw_extra, np.finfo(float).tiny))

    passed = bool(
        result.success
        and min_point.residual_norm <= MAX_CLOSURE
        and min_extra <= 1e-4
        and contrast >= 100.0
        and abs(minimum_m2 - raw_m2) <= 5e-5
        and raw_extra <= 1e-4
        and raw_contrast >= 50.0
    )
    return {
        "m1": m1,
        "m2_bracket": [low, high],
        "raw_event_values": raw_events,
        "certified_m2": raw_m2,
        "certified_trace_G_plus": float(cert["event_value"]),
        "certified_closure": float(cert["closure"]),
        "rank_jump_at_certified_chart": {
            "third_smallest_sigma_M_minus_I": raw_extra,
            "singular_values_ascending": list(raw_singular),
            "tight_trace_G_plus": raw_trace,
            "tight_periodic_closure": float(raw_point.residual_norm),
            "contrast_to_nearest_bracket_endpoint": raw_contrast,
        },
        "rank_jump_bracket_endpoints": {
            "third_smallest_sigma_M_minus_I": [low_extra, high_extra],
            "tight_trace_G_plus": [low_trace, high_trace],
            "periodic_closure": [float(low_point.residual_norm), float(high_point.residual_norm)],
        },
        "rank_jump_minimum": {
            "m2": minimum_m2,
            "delta_m2_from_certification": float(minimum_m2 - raw_m2),
            "third_smallest_sigma_M_minus_I": min_extra,
            "singular_values_ascending": list(min_singular),
            "tight_trace_G_plus": min_trace,
            "periodic_closure": float(min_point.residual_norm),
            "contrast_to_nearest_bracket_endpoint": contrast,
            "optimizer_success": bool(result.success),
            "optimizer_nfev": int(result.nfev),
        },
        "rank_jump_passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("comparison")
    parser.add_argument("output")
    args = parser.parse_args()

    results = []
    for row in plus_targets(load(Path(args.comparison))):
        record = audit(row)
        results.append(record)
        print(
            json.dumps(
                {
                    "m1": record["m1"],
                    "certified_m2": record["certified_m2"],
                    "certified_extra_sigma": record["rank_jump_at_certified_chart"]["third_smallest_sigma_M_minus_I"],
                    "minimum_m2": record["rank_jump_minimum"]["m2"],
                    "minimum_extra_sigma": record["rank_jump_minimum"]["third_smallest_sigma_M_minus_I"],
                    "passed": record["rank_jump_passed"],
                }
            ),
            flush=True,
        )

    passed = all(item["rank_jump_passed"] for item in results)
    payload = {
        "schema": "atlas.v1.plus-one-rank-jump-audit/1",
        "claim": "four label-invisible G+ roots coincide with extra nullity of the reduced monodromy at +1",
        "method": {
            "generic_neutral_geometric_nullity": 2,
            "structural_event": "third-smallest singular value of M-I collapses",
            "historical_equivalent": "G_plus = beta - 6*alpha + 20 (reported but not used as optimizer objective)",
            "search_domain": "only within each already sign-bracketed G+ cell",
        },
        "results": results,
        "summary": {
            "roots": len(results),
            "rank_jump_passes": sum(1 for item in results if item["rank_jump_passed"]),
            "max_abs_delta_m2_from_certification": max(
                abs(item["rank_jump_minimum"]["delta_m2_from_certification"]) for item in results
            ),
            "max_certified_extra_sigma": max(
                item["rank_jump_at_certified_chart"]["third_smallest_sigma_M_minus_I"] for item in results
            ),
            "passed": passed,
        },
        "claim_status": (
            "direct_rank_jump_crosschecks_all_four_label_invisible_plus_one_roots"
            if passed
            else "plus_one_rank_jump_audit_unresolved"
        ),
        "note": "Structural Float64 cross-check only; Julia BigFloat remains required for the marginal root.",
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2), flush=True)
    raise SystemExit(0 if passed else 3)


if __name__ == "__main__":
    main()
