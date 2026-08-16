#!/usr/bin/env python3
"""Launch the four local event-curve germs from one canonical mixed organizer.

The canonical BigFloat record supplies the organizer itself.  For each of the
two smooth event equations, a frozen localized catalog root supplies only the
orientation of the local tangent.  Two signed pseudo-arclength correctors are
then launched from the organizer.  The output is accepted only when both
directions are distinct in the mass plane and every new solution satisfies the
frozen event and periodic-closure gates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from threebody_atlas.boundary import BoundarySample
from threebody_atlas.critical_manifold import (
    LocalizedCriticalPoint,
    _flow_for_vector,
    advance_augmented_critical,
    event_value,
)
from threebody_atlas.hybrid_vertices import solve_direct_vertex
from threebody_atlas.liao_family import FamilyPoint

MODES = ("plus_one", "minus_one")
EVENT_GATE = 2e-8
CLOSURE_GATE = 1e-7


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def root_point(row: dict[str, Any]) -> FamilyPoint:
    return FamilyPoint(
        masses=tuple(float(value) for value in row["masses"]),
        x1=float(row["x1"]),
        v1=float(row["v1"]),
        v2=float(row["v2"]),
        period=float(row["period"]),
        residual_norm=float(row["closure"]),
        nfev=0,
        success=True,
    )


def localized(row: dict[str, Any], mode: str) -> LocalizedCriticalPoint:
    point = root_point(row)
    vector = np.asarray(
        [point.x1, point.v1, point.v2, point.period, point.masses[0], point.masses[1]],
        dtype=float,
    )
    closure_vector, floquet = _flow_for_vector(
        vector,
        m3=point.masses[2],
        rtol=5e-13,
        atol=5e-15,
    )
    closure = float(np.linalg.norm(closure_vector))
    point = FamilyPoint(
        masses=point.masses,
        x1=point.x1,
        v1=point.v1,
        v2=point.v2,
        period=point.period,
        residual_norm=closure,
        nfev=0,
        success=True,
    )
    sample = BoundarySample(point, floquet, 0.0)
    value = event_value(sample.floquet, mode)
    if closure > CLOSURE_GATE or abs(value) > EVENT_GATE:
        raise SystemExit(
            f"source cell {row.get('cell_id')} fails frozen gates after reevaluation: "
            f"closure={closure:.3e} event={value:.3e}"
        )
    return LocalizedCriticalPoint(
        sample=sample,
        event_mode=mode,
        event_value=float(value),
        source_width=float(row.get("source_width", 0.0)),
    )


def center_points(
    screen: dict[str, Any], canonical: dict[str, Any]
) -> tuple[dict[str, LocalizedCriticalPoint], float]:
    """Refine the Float64 center and bind it to the independent organizer."""
    if canonical.get("passed") is not True:
        raise SystemExit("canonical organizer record is not passed")
    candidate = screen.get("direct_candidate") or {}
    masses = canonical.get("masses") or []
    candidate_masses = candidate.get("masses") or []
    if candidate.get("success") is not True or len(candidate_masses) < 3:
        raise SystemExit("screen record needs a successful direct mixed candidate")
    if len(masses) < 3:
        raise SystemExit("canonical organizer record needs three masses")
    seed = np.asarray(
        [
            candidate["x1"],
            candidate["v1"],
            candidate["v2"],
            candidate["period"],
            candidate_masses[0],
            candidate_masses[1],
        ],
        dtype=float,
    )
    canonical_mass = np.asarray([float(masses[0]), float(masses[1])], dtype=float)
    candidate_mass = np.asarray(
        [float(candidate_masses[0]), float(candidate_masses[1])], dtype=float
    )
    initial_shift = float(np.linalg.norm(candidate_mass - canonical_mass))
    if initial_shift > 1e-4:
        raise SystemExit(
            f"Float64 mixed seed is not bound to the canonical organizer: {initial_shift:.3e}"
        )
    direct = solve_direct_vertex(
        seed,
        "mixed_plus_minus_one",
        m3=float(masses[2]),
        mass_bounds=(
            (float(masses[0]) - 0.002, float(masses[0]) + 0.002),
            (float(masses[1]) - 0.002, float(masses[1]) + 0.002),
        ),
        max_closure=CLOSURE_GATE,
        max_event=EVENT_GATE,
        max_invariant_error=2e-8,
        max_nfev=80,
        screening_rtol=5e-13,
        screening_atol=5e-15,
    )
    refined_mass = np.asarray(direct.point.masses[:2], dtype=float)
    refined_shift = float(np.linalg.norm(refined_mass - canonical_mass))
    if refined_shift > 1e-4:
        raise SystemExit(
            f"refined Float64 center left the canonical organizer: {refined_shift:.3e}"
        )
    closure_vector, floquet = _flow_for_vector(
        direct.vector,
        m3=float(masses[2]),
        rtol=5e-13,
        atol=5e-15,
    )
    closure = float(np.linalg.norm(closure_vector))
    center_point = FamilyPoint(
        masses=direct.point.masses,
        x1=direct.point.x1,
        v1=direct.point.v1,
        v2=direct.point.v2,
        period=direct.point.period,
        residual_norm=closure,
        nfev=direct.point.nfev,
        success=True,
    )
    if closure > CLOSURE_GATE:
        raise SystemExit(f"refined Float64 center closure gate failed: {closure:.3e}")
    sample = BoundarySample(center_point, floquet, 0.0)
    points: dict[str, LocalizedCriticalPoint] = {}
    for mode in MODES:
        value = event_value(sample.floquet, mode)
        if abs(value) > EVENT_GATE:
            raise SystemExit(
                f"refined Float64 center {mode} reevaluation fails event gate: {value:.3e}"
            )
        points[mode] = LocalizedCriticalPoint(
            sample=sample,
            event_mode=mode,
            event_value=float(value),
            source_width=0.0,
        )
    return points, refined_shift


def serialize_germ(
    point: Any,
    *,
    mixed_node: str,
    direction: str,
    source_cell_id: int,
    canonical_mass: np.ndarray,
    signed_step: float,
) -> dict[str, Any]:
    family = point.sample.point
    mass = np.asarray(family.masses[:2], dtype=float)
    return {
        "mixed_node": mixed_node,
        "event_mode": point.event_mode,
        "direction": direction,
        "status": "traced",
        "ends_on": mixed_node,
        "masses": [float(value) for value in family.masses],
        "x1": float(family.x1),
        "v1": float(family.v1),
        "v2": float(family.v2),
        "period": float(family.period),
        "closure": float(family.residual_norm),
        "event": float(point.event_value),
        "arclength_residual": float(point.arclength_residual),
        "scaled_tangent": [float(value) for value in point.tangent_scaled],
        "signed_arclength_step": float(signed_step),
        "source_cell_id": int(source_cell_id),
        "canonical_bound": True,
        "canonical_distance": float(np.linalg.norm(mass - canonical_mass)),
        "canonical_bracketed": True,
    }


def directional_audit(germs: list[dict[str, Any]], canonical_mass: np.ndarray) -> dict[str, Any]:
    audit: dict[str, Any] = {}
    for mode in MODES:
        rows = {row["direction"]: row for row in germs if row["event_mode"] == mode}
        if set(rows) != {"+", "-"}:
            raise SystemExit(f"{mode} does not have exactly two signed germs")
        plus = np.asarray(rows["+"]["masses"][:2], dtype=float) - canonical_mass
        minus = np.asarray(rows["-"]["masses"][:2], dtype=float) - canonical_mass
        dot = float(np.dot(plus, minus))
        distances = [float(np.linalg.norm(plus)), float(np.linalg.norm(minus))]
        if dot >= 0.0:
            raise SystemExit(f"{mode} signed correctors do not leave the organizer in opposite mass directions")
        audit[mode] = {
            "mass_displacement_dot_product": dot,
            "canonical_distances": distances,
            "opposite_mass_directions": True,
        }
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots_json")
    parser.add_argument("screen_json")
    parser.add_argument("canonical_json")
    parser.add_argument("output")
    parser.add_argument("--mixed-node", required=True)
    parser.add_argument("--cell", action="append", type=int, required=True)
    parser.add_argument("--step", type=float, default=7.5e-4)
    parser.add_argument("--max-canonical-distance", type=float, default=0.008)
    args = parser.parse_args()
    if args.step <= 0.0:
        raise SystemExit("--step must be positive")

    roots_path = Path(args.roots_json)
    screen_path = Path(args.screen_json)
    canonical_path = Path(args.canonical_json)
    roots_payload = json.loads(roots_path.read_text(encoding="utf-8"))
    screen = json.loads(screen_path.read_text(encoding="utf-8"))
    canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
    selected = {
        int(row["cell_id"]): row
        for row in roots_payload.get("roots", [])
        if int(row.get("cell_id", -1)) in set(args.cell)
    }
    if set(selected) != set(args.cell):
        raise SystemExit(
            f"missing requested source cells: {sorted(set(args.cell) - set(selected))}"
        )
    by_mode = {str(row.get("event_mode")): row for row in selected.values()}
    if set(by_mode) != set(MODES):
        raise SystemExit(f"source cells must supply one root for each of {MODES}: {sorted(by_mode)}")

    centers, center_shift = center_points(screen, canonical)
    center_mass = np.asarray(centers["plus_one"].sample.point.masses[:2], dtype=float)
    canonical_mass = np.asarray(
        [float(canonical["masses"][0]), float(canonical["masses"][1])],
        dtype=float,
    )
    germs: list[dict[str, Any]] = []
    for mode in MODES:
        source = localized(by_mode[mode], mode)
        center = centers[mode]
        for direction, signed_step in (("+", args.step), ("-", -args.step)):
            point = advance_augmented_critical(
                source,
                center,
                normalized_step=signed_step,
                max_closure=CLOSURE_GATE,
                max_event=EVENT_GATE,
                max_arc=5e-5,
                max_nfev=120,
                screening_rtol=5e-13,
                screening_atol=5e-15,
                use_jax_jacobian=True,
            )
            germ = serialize_germ(
                point,
                mixed_node=args.mixed_node,
                direction=direction,
                source_cell_id=int(by_mode[mode]["cell_id"]),
                canonical_mass=canonical_mass,
                signed_step=signed_step,
            )
            if germ["closure"] > CLOSURE_GATE or abs(germ["event"]) > EVENT_GATE:
                raise SystemExit(f"{mode}:{direction} misses the frozen numerical gates")
            if germ["canonical_distance"] > args.max_canonical_distance:
                raise SystemExit(
                    f"{mode}:{direction} leaves the local germ radius: "
                    f"{germ['canonical_distance']:.3e}"
                )
            germs.append(germ)

    audit = directional_audit(germs, center_mass)
    record = {
        "schema": "atlas.v1.mixed-germs/1",
        "claim_status": (
            "four canonical-centered float64 pseudo-arclength germs; organizer truth is supplied "
            "by the independently reproduced canonical artifact"
        ),
        "passed": True,
        "mixed_node": args.mixed_node,
        "canonical_masses": [float(value) for value in canonical_mass] + [1.0],
        "float64_center_masses": [float(value) for value in center_mass] + [1.0],
        "center_shift_from_canonical": center_shift,
        "frozen_gates": {"event": EVENT_GATE, "closure": CLOSURE_GATE},
        "source_roots_sha256": sha256_file(roots_path),
        "screen_sha256": sha256_file(screen_path),
        "canonical_sha256": sha256_file(canonical_path),
        "directional_audit": audit,
        "germs": germs,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "mixed_node": args.mixed_node,
                "germs": len(germs),
                "max_closure": max(row["closure"] for row in germs),
                "max_event": max(abs(row["event"]) for row in germs),
                "max_canonical_distance": max(row["canonical_distance"] for row in germs),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
