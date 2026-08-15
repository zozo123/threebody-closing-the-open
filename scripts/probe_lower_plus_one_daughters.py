#!/usr/bin/env python3
"""Test whether the lower ``+1`` physical mode produces a distinct periodic branch.

This is the concrete branch-switch falsification experiment for the principal
lower stability edge.  It starts from the frozen lower ``+1`` representative,
constructs the regular physical 4D Floquet quotient, and uses the smallest
singular directions of ``A-I`` as transverse branch seeds.

For each signed target amplitude, a generic 8D periodic solve is performed with
``m2`` free.  Any converged solution is then compared at the *same returned
masses* with the independently corrected Li-parent orbit in the same generic
gauge.  This distinction test is essential: a nonzero amplitude constraint can
otherwise be satisfied merely by sliding along the known parent sheet.

A reported distinct solution is only a float64 daughter candidate.  It must be
continued, symmetry/topology-classified, and independently reproduced before
it can support a branch-connection claim.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from threebody_atlas.branch_switch import correct_generic_branch_amplitude
from threebody_atlas.canonical_jacobi import jacobi_to_full_com
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


def generic_parent_at_m2(
    m1: float,
    m2: float,
    m3: float,
    li_guess: tuple[float, float, float, float],
    phase_reference: np.ndarray,
    *,
    max_residual: float,
) -> tuple[GenericPeriodicPoint, tuple[float, float, float, float]]:
    li = correct_family_point((m1, m2, m3), li_guess, max_nfev=100)
    if not li.success or li.residual_norm > max_residual:
        raise RuntimeError(f"Li parent correction failed at m2={m2:.16g}: {li.residual_norm:.3e}")
    reduced = full_to_reduced(state_from_chart(li.masses, li.x1, li.v1, li.v2))
    generic = correct_generic_periodic(
        li.masses,
        reduced,
        li.period,
        reference_state=phase_reference,
        max_nfev=100,
        max_closure=max_residual,
    )
    if not generic.success:
        raise RuntimeError(
            f"generic parent alignment failed at m2={m2:.16g}: "
            f"closure={generic.closure_norm:.3e} gauge={generic.gauge_norm:.3e} "
            f"phase={generic.phase_residual:.3e}"
        )
    return generic, (li.x1, li.v1, li.v2, li.period)


def normalized_distance(a_state, a_period, b_state, b_period) -> float:
    a = np.asarray((*a_state, a_period), dtype=float)
    b = np.asarray((*b_state, b_period), dtype=float)
    floors = np.asarray([0.2, 0.2, 0.5, 0.2, 0.5, 0.5, 0.5, 0.5, 1.0])
    scale = np.maximum(np.maximum(np.abs(a), np.abs(b)), floors)
    return float(np.linalg.norm((a - b) / scale))


def off_li_norm(state) -> float:
    z = np.asarray(state, dtype=float)
    # With q23=(1,0) fixed by the generic gauge, the Li ansatz additionally has
    # q13_y=v13_x=v23_x=0 in reduced coordinates.
    return float(np.linalg.norm(z[[1, 4, 6]]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("representatives_tsv")
    parser.add_argument("output")
    parser.add_argument("--amplitudes", default="0.0005,0.0015,0.004")
    parser.add_argument("--m2-halfwidth", type=float, default=0.004)
    parser.add_argument("--max-residual", type=float, default=3e-7)
    parser.add_argument("--distinct-tolerance", type=float, default=2e-5)
    args = parser.parse_args()
    amplitudes = [float(x) for x in args.amplitudes.split(",") if x.strip()]
    if not amplitudes or any(x <= 0.0 for x in amplitudes):
        raise SystemExit("amplitudes must contain positive values")

    row = load_lower(args.representatives_tsv)
    masses = (float(row["m1"]), float(row["m2"]), float(row["m3"]))
    li_guess = (
        float(row["x1"]),
        float(row["v1"]),
        float(row["v2"]),
        float(row["period"]),
    )
    li = correct_family_point(masses, li_guess, max_nfev=100)
    if not li.success or li.residual_norm > args.max_residual:
        raise RuntimeError(f"lower Li correction failed: {li.residual_norm:.3e}")
    full_state = state_from_chart(li.masses, li.x1, li.v1, li.v2)
    reduced_state = full_to_reduced(full_state)
    generic_reference = correct_generic_periodic(
        li.masses,
        reduced_state,
        li.period,
        reference_state=reduced_state,
        max_nfev=100,
        max_closure=args.max_residual,
    )
    if not generic_reference.success:
        raise RuntimeError(
            "generic lower-parent alignment failed: "
            f"closure={generic_reference.closure_norm:.3e} "
            f"gauge={generic_reference.gauge_norm:.3e} "
            f"phase={generic_reference.phase_residual:.3e}"
        )
    phase_reference = np.asarray(generic_reference.state, dtype=float)

    physical = compute_physical_floquet(
        state_from_chart(li.masses, li.x1, li.v1, li.v2),
        np.asarray(li.masses, dtype=float),
        li.period,
    )
    _, singular_values, vh = np.linalg.svd(physical.matrix - np.eye(4))
    quotient_directions = [vh[-1], vh[-2]]
    reduced_directions = []
    for vector in quotient_directions:
        jacobi_tangent = np.asarray(physical.lift_vector(vector).real, dtype=float)
        full_tangent = jacobi_to_full_com(jacobi_tangent, np.asarray(li.masses, dtype=float))
        reduced_tangent = full_to_reduced(full_tangent)
        if np.linalg.norm(reduced_tangent) == 0.0:
            raise RuntimeError("lifted physical branch direction vanished in reduced coordinates")
        reduced_directions.append(reduced_tangent)

    trials = []
    parent_guess = (li.x1, li.v1, li.v2, li.period)
    for direction_index, direction in enumerate(reduced_directions):
        for amplitude in amplitudes:
            for sign in (-1.0, 1.0):
                target = sign * amplitude
                try:
                    candidate = correct_generic_branch_amplitude(
                        m1=li.masses[0],
                        m3=li.masses[2],
                        reference_m2=li.masses[1],
                        reference_state=phase_reference,
                        reference_period=generic_reference.period,
                        direction=direction,
                        target_amplitude=target,
                        m2_halfwidth=args.m2_halfwidth,
                        max_nfev=160,
                        max_closure=args.max_residual,
                    )
                    if candidate.success:
                        parent, parent_guess = generic_parent_at_m2(
                            li.masses[0],
                            candidate.masses[1],
                            li.masses[2],
                            parent_guess,
                            phase_reference,
                            max_residual=args.max_residual,
                        )
                        distance = normalized_distance(
                            candidate.state,
                            candidate.period,
                            parent.state,
                            parent.period,
                        )
                    else:
                        parent = None
                        distance = float("nan")
                    trials.append(
                        {
                            "direction_index": direction_index,
                            "target_amplitude": target,
                            "success": bool(candidate.success),
                            "masses": [float(x) for x in candidate.masses],
                            "period": float(candidate.period),
                            "closure_norm": float(candidate.closure_norm),
                            "gauge_norm": float(candidate.gauge_norm),
                            "phase_residual": float(candidate.phase_residual),
                            "amplitude_residual": float(candidate.amplitude_residual),
                            "nfev": int(candidate.nfev),
                            "off_li_norm": off_li_norm(candidate.state),
                            "parent_distance": distance,
                            "distinct_from_li_parent": bool(
                                candidate.success
                                and np.isfinite(distance)
                                and distance > args.distinct_tolerance
                            ),
                            "parent_off_li_norm": (
                                off_li_norm(parent.state) if parent is not None else None
                            ),
                        }
                    )
                except Exception as exc:
                    trials.append(
                        {
                            "direction_index": direction_index,
                            "target_amplitude": target,
                            "success": False,
                            "error": f"{type(exc).__name__}: {exc}",
                            "distinct_from_li_parent": False,
                        }
                    )

    distinct = [trial for trial in trials if trial.get("distinct_from_li_parent")]
    payload = {
        "source": "frozen lower_plus_one critical-curve screening representative",
        "masses": [float(x) for x in li.masses],
        "source_screening_event": float(row["screening_event"]),
        "li_shooting_residual": float(li.residual_norm),
        "generic_parent_closure": float(generic_reference.closure_norm),
        "physical_plus_one_event": float(physical.plus_one_event),
        "physical_quotient_symplectic_defect": float(physical.quotient_symplectic_defect),
        "physical_quotient_leakage": float(physical.quotient_leakage),
        "physical_singular_values_A_minus_I": [float(x) for x in singular_values],
        "tested_amplitudes": amplitudes,
        "m2_halfwidth": args.m2_halfwidth,
        "trials": trials,
        "distinct_candidate_count": len(distinct),
        "distinct_candidates": distinct,
        "interpretation": (
            "nonzero distinct_candidate_count is a float64 generic-chart daughter hypothesis; "
            "zero is a negative local branch-switch result at these amplitudes, not a theorem "
            "that no daughter exists"
        ),
        "claim_status": "screening only; continuation and independent reproduction required",
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "physical_plus_one_event": payload["physical_plus_one_event"],
                "physical_singular_values_A_minus_I": payload[
                    "physical_singular_values_A_minus_I"
                ],
                "trials": len(trials),
                "successful_trials": sum(bool(t.get("success")) for t in trials),
                "distinct_candidate_count": len(distinct),
                "distinct_candidates": distinct,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
