#!/usr/bin/env python3
"""Replay the direct mixed-vertex solve of a committed junction screen.

``scripts/trace_junction_organizer.py`` ends with a direct
``(alpha, beta) = (4, 4)`` solve seeded by the traced point whose Floquet
invariants sit closest to the mixed vertex.  That final step needs the
``accelerated`` extra (JAX + Diffrax) for its analytic Jacobian.  Two of the
three committed headline junction screens recorded

    "direct_mixed_vertex_retry": {"status": "not_accepted",
      "error": "RuntimeError: JAX + Diffrax are required; install the accelerated extra"}

so their float64 vertex candidates are missing because of a MISSING DEPENDENCY,
not because the candidates do not exist.

This driver re-runs exactly that step, and nothing else, against an already
committed junction artifact.  It re-derives the seed with the same rule the
original run used -- ``min`` over ``localized_seeds + points`` of
``hypot(alpha - 4, beta - 4)`` -- and it re-derives the mass box from the same
``requested_center`` and padding.  It never reads a canonical BigFloat record,
so the candidate it produces stays an INDEPENDENT float64 screen that the
canonical organizer can then be checked against.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from threebody_atlas.hybrid_vertices import solve_direct_vertex


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distance_to_mixed(record: dict[str, Any]) -> float:
    return float(np.hypot(float(record["alpha"]) - 4.0, float(record["beta"]) - 4.0))


def combined_samples(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], set[str]]:
    """Every traced sample of the screen, in the original run's order."""
    combined: list[dict[str, Any]] = []
    modes: set[str] = set()
    for trace in payload.get("traces") or []:
        mode = trace.get("event_mode") or trace.get("mode")
        if mode:
            modes.add(str(mode))
        combined.extend(trace.get("localized_seeds") or [])
        combined.extend(trace.get("points") or [])
    return combined, modes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("junction_json")
    parser.add_argument("output")
    parser.add_argument("--mixed-node", required=True)
    parser.add_argument("--direct-mass-padding", type=float, default=0.012)
    parser.add_argument("--max-nfev", type=int, default=60)
    args = parser.parse_args()

    junction_path = Path(args.junction_json)
    payload = json.loads(junction_path.read_text(encoding="utf-8"))
    center = payload.get("requested_center") or []
    if len(center) < 2:
        raise SystemExit("junction artifact needs a two-component requested_center")
    center_m1, center_m2 = float(center[0]), float(center[1])

    combined, modes = combined_samples(payload)
    if not combined:
        raise SystemExit("junction artifact carries no traced samples")
    if modes != {"plus_one", "minus_one"}:
        raise SystemExit(f"a mixed vertex needs both smooth event traces, got {sorted(modes)}")

    closest = min(combined, key=distance_to_mixed)
    seed_distance = distance_to_mixed(closest)
    seed = np.asarray(
        [
            closest["x1"],
            closest["v1"],
            closest["v2"],
            closest["period"],
            closest["masses"][0],
            closest["masses"][1],
        ],
        dtype=float,
    )
    padding = float(args.direct_mass_padding)
    try:
        direct = solve_direct_vertex(
            seed,
            "mixed_plus_minus_one",
            m3=float(closest["masses"][2]),
            mass_bounds=(
                (center_m1 - padding, center_m1 + padding),
                (center_m2 - padding, center_m2 + padding),
            ),
            max_nfev=int(args.max_nfev),
        )
    except Exception as exc:  # noqa: BLE001 - the recorded failure mode is the point
        direct_result: dict[str, Any] = {
            "status": "not_accepted",
            "error": f"{type(exc).__name__}: {exc}",
        }
    else:
        direct_result = {
            "status": "accepted_screening_candidate",
            "masses": [float(value) for value in direct.point.masses],
            "x1": float(direct.point.x1),
            "v1": float(direct.point.v1),
            "v2": float(direct.point.v2),
            "period": float(direct.point.period),
            "shooting_residual": float(direct.point.residual_norm),
            "alpha": float(direct.alpha),
            "beta": float(direct.beta),
            "event_values": [float(value) for value in direct.event_values],
            "invariant_error": float(direct.invariant_error),
            "nfev": int(direct.nfev),
        }

    record = {
        "schema": "atlas.v1.direct-mixed-vertex-retry/1",
        "claim_status": (
            "float64 direct mixed-vertex screen replayed from a committed junction trace with the "
            "accelerated extra installed; independent BigFloat/canonical reproduction still required"
        ),
        "mixed_node": args.mixed_node,
        "source_junction": str(junction_path),
        "source_junction_sha256": sha256_file(junction_path),
        "requested_center": [center_m1, center_m2],
        "direct_mass_padding": padding,
        "max_nfev": int(args.max_nfev),
        "seed_sample_index": combined.index(closest),
        "seed_masses": [float(value) for value in closest["masses"]],
        "seed_spectral_distance_to_mixed_vertex": seed_distance,
        "seed_provenance": "junction_trace_closest_to_alpha_beta_four",
        "direct_mixed_vertex_retry": direct_result,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: record[k] for k in ("mixed_node", "seed_spectral_distance_to_mixed_vertex", "direct_mixed_vertex_retry")}, indent=2))
    if direct_result["status"] != "accepted_screening_candidate":
        raise SystemExit(f"direct mixed-vertex retry did not accept: {direct_result.get('error')}")


if __name__ == "__main__":
    main()
