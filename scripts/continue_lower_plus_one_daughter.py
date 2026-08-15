#!/usr/bin/env python3
"""Continue one lower-``+1`` daughter candidate as a generic periodic branch.

Two amplitude-constrained branch-switch points are used only as initial seeds.
After that, continuation is pure pseudo-arclength in the generic strict-periodic
chart ``(z0,T,m2)``.  Every accepted point is compared with the independently
corrected Li parent at the same masses, so a solver drift back onto the parent
sheet is visible rather than silently counted as daughter continuation.

The output is screening evidence.  A genuine branch connection/reconnection is
not a release claim until representative daughter points are independently
reproduced at high precision.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from threebody_atlas.branch_switch import correct_generic_branch_amplitude
from threebody_atlas.canonical_jacobi import jacobi_to_full_com
from threebody_atlas.generic_branch import trace_generic_branch
from threebody_atlas.generic_periodic import GenericPeriodicPoint, correct_generic_periodic
from threebody_atlas.liao_family import correct_family_point, state_from_chart
from threebody_atlas.physical_floquet import compute_physical_floquet
from threebody_atlas.reduced import full_to_reduced


def load_lower(path: str | Path) -> dict[str, str]:
    with Path(path).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    matches = [row for row in rows if row["name"] == "lower_plus_one"]
    if len(matches) != 1:
        raise RuntimeError("expected exactly one lower_plus_one representative")
    return matches[0]


def normalized_distance(a_state, a_period, b_state, b_period) -> float:
    a = np.asarray((*a_state, a_period), dtype=float)
    b = np.asarray((*b_state, b_period), dtype=float)
    floors = np.asarray([0.2, 0.2, 0.5, 0.2, 0.5, 0.5, 0.5, 0.5, 1.0])
    scale = np.maximum(np.maximum(np.abs(a), np.abs(b)), floors)
    return float(np.linalg.norm((a - b) / scale))


def off_li_norm(state) -> float:
    z = np.asarray(state, dtype=float)
    return float(np.linalg.norm(z[[1, 4, 6]]))


def generic_parent_at_m2(
    *,
    m1: float,
    m2: float,
    m3: float,
    li_guess: tuple[float, float, float, float],
    phase_reference: np.ndarray,
    max_residual: float,
) -> tuple[GenericPeriodicPoint, tuple[float, float, float, float]]:
    li = correct_family_point((m1, m2, m3), li_guess, max_nfev=120)
    if not li.success or li.residual_norm > max_residual:
        raise RuntimeError(f"Li parent correction failed at m2={m2:.16g}: {li.residual_norm:.3e}")
    reduced = full_to_reduced(state_from_chart(li.masses, li.x1, li.v1, li.v2))
    generic = correct_generic_periodic(
        li.masses,
        reduced,
        li.period,
        reference_state=phase_reference,
        max_nfev=120,
        max_closure=max_residual,
    )
    if not generic.success:
        raise RuntimeError(
            f"generic parent alignment failed at m2={m2:.16g}: "
            f"closure={generic.closure_norm:.3e} gauge={generic.gauge_norm:.3e} "
            f"phase={generic.phase_residual:.3e}"
        )
    return generic, (li.x1, li.v1, li.v2, li.period)


def serialize_candidate(point, parent, distance: float) -> dict:
    return {
        "masses": [float(x) for x in point.masses],
        "state": [float(x) for x in point.state],
        "period": float(point.period),
        "closure_norm": float(point.closure_norm),
        "gauge_norm": float(point.gauge_norm),
        "phase_residual": float(point.phase_residual),
        "off_li_norm": off_li_norm(point.state),
        "parent_distance": float(distance),
        "parent_off_li_norm": off_li_norm(parent.state),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("representatives_tsv")
    parser.add_argument("output")
    parser.add_argument("--direction-index", type=int, choices=(0, 1), required=True)
    parser.add_argument("--sign", type=int, choices=(-1, 1), required=True)
    parser.add_argument("--seed-amplitudes", default="0.0005,0.0015")
    parser.add_argument("--steps", type=int, default=18)
    parser.add_argument("--arclength-step", type=float, default=1.5e-3)
    parser.add_argument("--m2-halfwidth", type=float, default=0.006)
    parser.add_argument("--m2-min", type=float, default=0.7)
    parser.add_argument("--m2-max", type=float, default=1.2)
    parser.add_argument("--max-residual", type=float, default=3e-7)
    parser.add_argument("--distinct-tolerance", type=float, default=2e-5)
    parser.add_argument("--reconnect-tolerance", type=float, default=2e-5)
    args = parser.parse_args()

    amplitudes = sorted(float(x) for x in args.seed_amplitudes.split(",") if x.strip())
    if len(amplitudes) != 2 or any(x <= 0.0 for x in amplitudes) or amplitudes[0] == amplitudes[1]:
        raise SystemExit("seed-amplitudes must contain exactly two distinct positive values")
    if args.steps < 1:
        raise SystemExit("steps must be positive")
    if not (0.0 < args.m2_min < args.m2_max):
        raise SystemExit("invalid m2 bounds")

    row = load_lower(args.representatives_tsv)
    masses = (float(row["m1"]), float(row["m2"]), float(row["m3"]))
    li_guess = (float(row["x1"]), float(row["v1"]), float(row["v2"]), float(row["period"]))
    li = correct_family_point(masses, li_guess, max_nfev=120)
    if not li.success or li.residual_norm > args.max_residual:
        raise RuntimeError(f"lower Li correction failed: {li.residual_norm:.3e}")

    full_state = state_from_chart(li.masses, li.x1, li.v1, li.v2)
    reduced_state = full_to_reduced(full_state)
    generic_reference = correct_generic_periodic(
        li.masses,
        reduced_state,
        li.period,
        reference_state=reduced_state,
        max_nfev=120,
        max_closure=args.max_residual,
    )
    if not generic_reference.success:
        raise RuntimeError("generic lower-parent alignment failed")
    phase_reference = np.asarray(generic_reference.state, dtype=float)

    physical = compute_physical_floquet(full_state, np.asarray(li.masses, dtype=float), li.period)
    _, singular_values, vh = np.linalg.svd(physical.matrix - np.eye(4))
    quotient_direction = vh[-1] if args.direction_index == 0 else vh[-2]
    jacobi_tangent = np.asarray(physical.lift_vector(quotient_direction).real, dtype=float)
    full_tangent = jacobi_to_full_com(jacobi_tangent, np.asarray(li.masses, dtype=float))
    direction = full_to_reduced(full_tangent)
    if float(np.linalg.norm(direction)) == 0.0:
        raise RuntimeError("lifted physical branch direction vanished")

    seed_points = []
    seed_records = []
    parent_guess = (li.x1, li.v1, li.v2, li.period)
    for amplitude in amplitudes:
        target = float(args.sign) * amplitude
        seed = correct_generic_branch_amplitude(
            m1=li.masses[0],
            m3=li.masses[2],
            reference_m2=li.masses[1],
            reference_state=phase_reference,
            reference_period=generic_reference.period,
            direction=direction,
            target_amplitude=target,
            m2_halfwidth=args.m2_halfwidth,
            max_nfev=180,
            max_closure=args.max_residual,
        )
        if not seed.success:
            raise RuntimeError(f"daughter seed failed at target amplitude {target}")
        parent, parent_guess = generic_parent_at_m2(
            m1=li.masses[0],
            m2=seed.masses[1],
            m3=li.masses[2],
            li_guess=parent_guess,
            phase_reference=phase_reference,
            max_residual=args.max_residual,
        )
        distance = normalized_distance(seed.state, seed.period, parent.state, parent.period)
        if distance <= args.distinct_tolerance:
            raise RuntimeError(
                f"daughter seed collapsed onto Li parent at target {target}: distance={distance:.3e}"
            )
        seed_points.append(seed)
        record = serialize_candidate(seed, parent, distance)
        record.update(
            {
                "target_amplitude": float(seed.target_amplitude),
                "achieved_amplitude": float(seed.achieved_amplitude),
                "amplitude_residual": float(seed.amplitude_residual),
            }
        )
        seed_records.append(record)

    trace = trace_generic_branch(
        seed_points[0],
        seed_points[1],
        reference_state=phase_reference,
        m1=li.masses[0],
        m3=li.masses[2],
        steps=args.steps,
        normalized_step=args.arclength_step,
        m2_bounds=(args.m2_min, args.m2_max),
    )

    traced_records = []
    parent_failures = []
    min_parent_distance = min(record["parent_distance"] for record in seed_records)
    reconnect_candidates = []
    for index, point in enumerate(trace.points):
        try:
            parent, parent_guess = generic_parent_at_m2(
                m1=li.masses[0],
                m2=point.masses[1],
                m3=li.masses[2],
                li_guess=parent_guess,
                phase_reference=phase_reference,
                max_residual=args.max_residual,
            )
            distance = normalized_distance(point.state, point.period, parent.state, parent.period)
            min_parent_distance = min(min_parent_distance, distance)
            record = serialize_candidate(point, parent, distance)
            record.update(
                {
                    "index": index,
                    "arclength_residual": float(point.arclength_residual),
                    "normalized_step": float(point.normalized_step),
                    "nfev": int(point.nfev),
                }
            )
            if distance <= args.reconnect_tolerance:
                reconnect_candidates.append(record)
            traced_records.append(record)
        except Exception as exc:
            parent_failures.append(
                {
                    "index": index,
                    "masses": [float(x) for x in point.masses],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            traced_records.append(
                {
                    "index": index,
                    "masses": [float(x) for x in point.masses],
                    "state": [float(x) for x in point.state],
                    "period": float(point.period),
                    "closure_norm": float(point.closure_norm),
                    "gauge_norm": float(point.gauge_norm),
                    "phase_residual": float(point.phase_residual),
                    "arclength_residual": float(point.arclength_residual),
                    "normalized_step": float(point.normalized_step),
                    "parent_comparison": "failed",
                }
            )

    independent_seeds = []
    if traced_records:
        for idx in sorted({0, len(traced_records) // 2, len(traced_records) - 1}):
            rec = traced_records[idx]
            if "state" in rec:
                independent_seeds.append(
                    {
                        "index": idx,
                        "masses": rec["masses"],
                        "state": rec["state"],
                        "period": rec["period"],
                    }
                )

    payload = {
        "claim_status": "float64 generic daughter continuation; independent reproduction required",
        "source": "frozen lower_plus_one critical representative",
        "direction_index": args.direction_index,
        "sign": args.sign,
        "physical_plus_one_event": float(physical.plus_one_event),
        "physical_singular_values_A_minus_I": [float(x) for x in singular_values],
        "li_source_masses": [float(x) for x in li.masses],
        "li_source_residual": float(li.residual_norm),
        "generic_parent_closure": float(generic_reference.closure_norm),
        "seed_points": seed_records,
        "trace_points": traced_records,
        "trace_point_count": len(trace.points),
        "stopped_reason": trace.stopped_reason,
        "parent_comparison_failures": parent_failures,
        "minimum_parent_distance": float(min_parent_distance),
        "reconnect_tolerance": args.reconnect_tolerance,
        "reconnect_candidate_count": len(reconnect_candidates),
        "reconnect_candidates": reconnect_candidates,
        "independent_reproduction_seeds": independent_seeds,
        "interpretation": (
            "A nonzero trace_point_count demonstrates continuation beyond amplitude-constrained seeds. "
            "A reconnect candidate is only a screening flag that the generic branch approaches the "
            "same-mass Li parent; independent high-precision reproduction is required before a "
            "branch-connection claim."
        ),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "direction_index": args.direction_index,
                "sign": args.sign,
                "trace_point_count": payload["trace_point_count"],
                "stopped_reason": payload["stopped_reason"],
                "minimum_parent_distance": payload["minimum_parent_distance"],
                "reconnect_candidate_count": payload["reconnect_candidate_count"],
            },
            indent=2,
        )
    )
    if not trace.points:
        raise SystemExit("daughter pseudo-arclength produced zero accepted continuation points")


if __name__ == "__main__":
    main()
