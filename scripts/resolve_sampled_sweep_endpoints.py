#!/usr/bin/env python3
"""Replace sampled 'interior termini' with continuous scientific classifications.

Two supplemental minus-one components currently end strictly inside the scanned
mass support (component 0 at its low-m1 end; component 1 at its high-m1 end).
The assembler labels those finite-lattice ends as passed nodes.  This script
continues *away* from the sampled segment using the variational predictor in
``trace_label_invisible_continuous.py`` and asks what the zero set actually does.

Accepted scientific stops:
  * tangent-matched overlap with a committed catalog critical root/arc;
  * declared-domain boundary;
  * closed loop (return to an earlier continuation point with tangent match).
A projection turn is recorded but never treated as a stop.  Corrector failure is
reported as unresolved, never promoted to an endpoint.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

import trace_label_invisible_continuous as cont
from threebody_atlas.critical_geometry import continuation_scales


CATALOG_MATCH_TOL = 7e-3
CATALOG_TANGENT_COS = 0.94
LOOP_MATCH_TOL = 5e-3
LOOP_MIN_SEPARATION = 12
ORGANIZER_MATCH_TOL = 8e-3
GRAPH_PATH = Path(__file__).resolve().parents[1] / "research/evidence/V1_CRITICAL_GRAPH.json"


def load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path}: JSON root must be an object")
    return payload


def vector(row: dict[str, Any]) -> np.ndarray:
    masses = row["masses"]
    return np.asarray(
        [float(row["x1"]), float(row["v1"]), float(row["v2"]), float(row["period"]), float(masses[0]), float(masses[1])],
        dtype=float,
    )


def segment_distance(a: np.ndarray, b: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    scale = continuation_scales(target)
    aa = (a - target) / scale
    bb = (b - target) / scale
    d = bb - aa
    denom = float(np.dot(d, d))
    if denom == 0.0:
        return float(np.linalg.norm(aa)), 0.0
    t = float(np.clip(-np.dot(aa, d) / denom, 0.0, 1.0))
    return float(np.linalg.norm(aa + t * d)), t


def tangent_mass(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = b[4:6] - a[4:6]
    n = float(np.linalg.norm(d))
    return d / n if n else np.zeros(2)


def strict_supplemental(row: dict[str, Any]) -> cont.StrictPoint:
    return cont._strict_localize(
        row,
        str(row["event_mode"]),
        source="research/evidence/V1_SUPPLEMENTAL_EVENT_SIGN_ROOTS_2026-08-16.json",
        source_id=f"supplemental-cell-{int(row['cell_id'])}",
    )


def _direction_only_seed(certified: cont.StrictPoint, neighbor: dict[str, Any]) -> cont.StrictPoint:
    """Copy a re-certified chart onto a neighbor's masses for the first secant.

    The neighbor is not a scientific seed.  Only its mass-plane location orients
    the variational predictor.  The first accepted step still has to pass the
    frozen gates on its own.
    """
    masses = neighbor.get("masses") or []
    if len(masses) < 2:
        raise RuntimeError("direction-only neighbor is missing masses")
    point = certified.localized.sample.point
    dummy_point = replace(
        point,
        masses=(float(masses[0]), float(masses[1]), float(point.masses[2])),
    )
    dummy_sample = replace(certified.localized.sample, point=dummy_point)
    dummy_localized = replace(certified.localized, sample=dummy_sample)
    return replace(certified, localized=dummy_localized)


def mixed_organizers(mode: str) -> list[dict[str, Any]]:
    """Passed mixed organizers that a matching-mechanism branch may terminate on."""
    if not GRAPH_PATH.is_file():
        return []
    graph = load(GRAPH_PATH)
    rows = []
    for node in graph.get("nodes", []):
        kind = str(node.get("kind") or "")
        mechanism = str(node.get("mechanism") or "")
        if not node.get("passed"):
            continue
        if kind != "mixed_organizer" and mechanism != "mixed_organizer":
            continue
        masses = node.get("masses") or []
        if len(masses) < 2:
            continue
        rows.append(
            {
                "id": str(node["id"]),
                "masses": [float(masses[0]), float(masses[1])],
            }
        )
    return rows


def nearest_organizer(
    point: np.ndarray,
    organizers: list[dict[str, Any]],
) -> dict[str, Any] | None:
    best = None
    for item in organizers:
        target = np.asarray(item["masses"], dtype=float)
        miss = float(np.linalg.norm(point[4:6] - target))
        if best is None or miss < best["miss_mass"]:
            best = {"id": item["id"], "miss_mass": miss, "masses": item["masses"]}
    return best


def catalog_candidates(payload: dict[str, Any], mode: str) -> list[dict[str, Any]]:
    rows = []
    for row in payload.get("roots", []):
        if row.get("status") != "ok" or not row.get("passed") or row.get("event_mode") != mode:
            continue
        rows.append(
            {
                "cell_id": int(row["cell_id"]),
                "orientation": row.get("orientation"),
                "target": vector(row),
            }
        )
    return rows


def nearest_catalog_segment(
    a: np.ndarray,
    b: np.ndarray,
    candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    seg_tangent = tangent_mass(a, b)
    best = None
    for item in candidates:
        miss, fraction = segment_distance(a, b, item["target"])
        # A catalog root has no stored branch tangent.  Estimate it from the
        # local mass secant only after a close 6-D chart match is found; the
        # final branch tangent check is done against neighboring catalog roots
        # below by the caller when possible.
        if best is None or miss < best["miss_scaled"]:
            best = {
                "cell_id": item["cell_id"],
                "orientation": item["orientation"],
                "miss_scaled": miss,
                "fraction": fraction,
                "segment_mass_tangent": seg_tangent,
                "target": item["target"],
            }
    return best


def catalog_neighbor_tangent(
    hit: dict[str, Any], candidates: list[dict[str, Any]]
) -> tuple[float, list[int]]:
    target = hit["target"]
    same = sorted(
        candidates,
        key=lambda item: abs(float(item["target"][4] - target[4])) + abs(float(item["target"][5] - target[5])),
    )
    neighbors = [item for item in same if item["cell_id"] != hit["cell_id"]][:4]
    best_cos = 0.0
    ids: list[int] = []
    for item in neighbors:
        d = item["target"][4:6] - target[4:6]
        n = float(np.linalg.norm(d))
        if n == 0.0:
            continue
        ids.append(item["cell_id"])
        cos = abs(float(np.dot(hit["segment_mass_tangent"], d / n)))
        best_cos = max(best_cos, cos)
    return best_cos, ids


def loop_hit(history: list[np.ndarray], a: np.ndarray, b: np.ndarray) -> dict[str, Any] | None:
    if len(history) <= LOOP_MIN_SEPARATION:
        return None
    seg_tangent = tangent_mass(a, b)
    for index, target in enumerate(history[:-LOOP_MIN_SEPARATION]):
        miss, fraction = segment_distance(a, b, target)
        if miss > LOOP_MATCH_TOL:
            continue
        before = history[max(0, index - 1)]
        after = history[min(len(history) - 1, index + 1)]
        old_tangent = tangent_mass(before, after)
        cosine = abs(float(np.dot(seg_tangent, old_tangent)))
        if cosine >= 0.94:
            return {
                "kind": "closed_loop",
                "history_index": index,
                "miss_scaled": miss,
                "fraction": fraction,
                "tangent_abs_cosine": cosine,
            }
    return None


def trace_endpoint(
    rows: list[dict[str, Any]],
    catalog: list[dict[str, Any]],
    *,
    outward_side: str,
    mass_step: float,
    max_steps: int,
    organizers: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    rows = sorted(rows, key=lambda row: (float(row["masses"][0]), float(row["masses"][1])))
    if outward_side == "low":
        ordered = list(rows)
    elif outward_side == "high":
        ordered = list(reversed(rows))
    else:
        raise ValueError(outward_side)
    # A stored lattice sample can fail the frozen event gate on re-correction
    # (float64 evaluation floor).  Walk inward until a point re-certifies, then
    # continue outward.  A second re-certified neighbor is preferred for the
    # predictor direction; otherwise the unused lattice neighbor supplies only
    # a mass-plane secant (its chart is not treated as a seed).
    current_row = None
    current = None
    previous_row = None
    previous = None
    seed_errors: list[str] = []
    for row in ordered:
        try:
            point = strict_supplemental(row)
        except (RuntimeError, ValueError, FloatingPointError) as exc:
            seed_errors.append(f"cell {row.get('cell_id')}: {exc}")
            continue
        if current is None:
            current_row, current = row, point
            continue
        previous_row, previous = row, point
        break
    if current is None or current_row is None:
        raise RuntimeError(
            f"component {rows[0].get('sweep_component')} {outward_side}: "
            f"no lattice seed re-certified; {seed_errors}"
        )
    direction_only = False
    if previous is None or previous_row is None:
        neighbor = next((row for row in ordered if row is not current_row), rows[0])
        previous_row = neighbor
        previous = _direction_only_seed(current, neighbor)
        direction_only = True
    prev, cur = previous.localized, current.localized
    history = [previous.vector, current.vector]
    accepted = []
    folds = []
    step = float(mass_step)
    stop = "max_steps_exhausted"
    terminal = None
    previous_dm1 = float(current.masses2[0] - previous.masses2[0])

    for index in range(max_steps):
        point = None
        last_error = None
        trial = step
        for _retry in range(8):
            try:
                point = cont._advance_variational(prev, cur, requested_step=trial)
                break
            except (RuntimeError, ValueError, FloatingPointError) as exc:
                last_error = exc
                trial *= 0.5
                if trial < 5e-5:
                    break
        if point is None:
            stop = f"continuation_failure_not_a_scientific_terminus: {last_error}"
            break

        a = np.asarray(cur.vector, dtype=float)
        b = point.vector
        dm1 = float(b[4] - a[4])
        if previous_dm1 != 0.0 and dm1 != 0.0 and previous_dm1 * dm1 < 0.0:
            folds.append(
                {
                    "kind": "m1_projection_turn_crossed",
                    "step_index": index,
                    "before_masses": [float(a[4]), float(a[5]), 1.0],
                    "after_masses": [float(b[4]), float(b[5]), 1.0],
                }
            )
        if dm1 != 0.0:
            previous_dm1 = dm1

        crossing = cont._domain_crossing(a, b)
        if crossing is not None:
            terminal = {"kind": "declared_domain_boundary", **crossing}
            stop = "declared_domain_boundary_reached"

        if terminal is None:
            hit = nearest_catalog_segment(a, b, catalog)
            if hit is not None and hit["miss_scaled"] <= CATALOG_MATCH_TOL:
                cosine, neighbor_ids = catalog_neighbor_tangent(hit, catalog)
                if cosine >= CATALOG_TANGENT_COS:
                    terminal = {
                        "kind": "existing_catalog_critical_curve",
                        "cell_id": hit["cell_id"],
                        "orientation": hit["orientation"],
                        "miss_scaled": float(hit["miss_scaled"]),
                        "segment_fraction": float(hit["fraction"]),
                        "tangent_abs_cosine": float(cosine),
                        "neighbor_cells_used_for_tangent": neighbor_ids,
                    }
                    stop = "existing_catalog_curve_reached"

        if terminal is None:
            hit = nearest_organizer(b, organizers or [])
            if hit is not None and hit["miss_mass"] <= ORGANIZER_MATCH_TOL:
                terminal = {
                    "kind": "mixed_organizer",
                    "node_id": hit["id"],
                    "miss_mass": float(hit["miss_mass"]),
                    "organizer_masses": hit["masses"],
                }
                stop = "mixed_organizer_reached"

        if terminal is None:
            loop = loop_hit(history, a, b)
            if loop is not None:
                terminal = loop
                stop = "closed_loop_reached"

        accepted.append(cont._serialize_variational(point))
        history.append(b)
        prev, cur = cur, point.localized
        step = min(float(mass_step), trial * 1.35)
        if terminal is not None:
            break

    passed = terminal is not None and terminal["kind"] in {
        "declared_domain_boundary",
        "existing_catalog_critical_curve",
        "mixed_organizer",
        "closed_loop",
    }
    return {
        "source_component": int(rows[0]["sweep_component"]),
        "outward_side": outward_side,
        # The seed pair identifies the COMPUTATION.  main() refuses to report two
        # sides that share one -- see the degeneracy note there.
        "seed_cells": {
            "current": current_row.get("cell_id"),
            "previous": previous_row.get("cell_id"),
            "direction_only_neighbor": bool(direction_only),
        },
        "seed_rows": [int(previous_row["cell_id"]), int(current_row["cell_id"])],
        "seed_previous": cont._serialize_localized(previous.localized),
        "seed_current": cont._serialize_localized(current.localized),
        "accepted_points": accepted,
        "projection_turns_crossed": folds,
        "terminal": terminal,
        "stopped_reason": stop,
        "scientific_endpoint_resolved": bool(passed),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("supplemental_roots")
    parser.add_argument("catalog_roots")
    parser.add_argument("output")
    parser.add_argument("--mass-step", type=float, default=0.0025)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument(
        "--components",
        help=(
            "comma-separated sweep_component indices to resolve.  Default: every "
            "component in the artifact that carries at least two roots.  This used "
            "to be the literal tuple (0, 1, 12), which meant the script could only "
            "ever resolve the first supplemental artifact and raised "
            "'components no longer contain enough roots' on any other."
        ),
    )
    args = parser.parse_args()

    cont.require_accelerated_x64()
    supplemental = load(Path(args.supplemental_roots))
    catalog_payload = load(Path(args.catalog_roots))
    minus_catalog = catalog_candidates(catalog_payload, "minus_one")
    plus_catalog = catalog_candidates(catalog_payload, "plus_one")
    minus_organizers = mixed_organizers("minus_one")
    plus_organizers = mixed_organizers("plus_one")

    by_component: dict[int, list[dict[str, Any]]] = {}
    for row in supplemental.get("roots", []):
        by_component.setdefault(int(row.get("sweep_component", -1)), []).append(row)
    if args.components:
        requested = [int(x) for x in args.components.split(",") if x.strip()]
        missing = [i for i in requested if len(by_component.get(i, [])) < 2]
        if missing:
            raise RuntimeError(
                f"supplemental components {missing} carry fewer than two roots"
            )
    else:
        # Every component with enough roots to have two distinguishable termini.
        requested = sorted(i for i, rows in by_component.items() if i >= 0 and len(rows) >= 2)
    if not requested:
        raise RuntimeError(
            "no sweep component in this artifact carries two or more roots, so it "
            "has no finite termini to continue"
        )

    # A component's mechanism comes from its own roots, not from a table keyed by
    # component index: indices are per-artifact and carry no cross-artifact meaning.
    def mechanism_of(rows: list[dict[str, Any]]) -> str:
        modes = {str(r.get("event_mode")) for r in rows}
        if len(modes) != 1:
            raise RuntimeError(f"component spans mixed mechanisms {sorted(modes)}")
        return modes.pop()

    jobs = []
    for component in requested:
        rows = by_component[component]
        mode = mechanism_of(rows)
        if mode == "minus_one":
            catalog, organizers = minus_catalog, minus_organizers
        elif mode == "plus_one":
            catalog, organizers = plus_catalog, plus_organizers
        else:
            raise RuntimeError(f"component {component} has unsupported mechanism {mode!r}")
        # Both finite termini of every component, low-m1 and high-m1 side.
        jobs.append((component, "low", catalog, organizers))
        jobs.append((component, "high", catalog, organizers))
    jobs = tuple(jobs)
    results = [
        trace_endpoint(
            by_component[component],
            catalog,
            outward_side=side,
            mass_step=args.mass_step,
            max_steps=args.max_steps,
            organizers=organizers,
        )
        for component, side, catalog, organizers in jobs
    ]
    # DEGENERACY REFUSAL.
    #
    # A low trace and a high trace are two different computations.  When a
    # component's requested tip fails the frozen event gate on re-correction,
    # the walk above starts from a point further in; that is legitimate only
    # while the two sides still begin from different seeds.  If both sides end
    # up on the SAME (seed, neighbour) pair they are literally one computation,
    # so at most one of them resolves its labelled terminus -- and nothing in
    # the run says which.  Both are therefore unresolved.
    #
    # plus_one component 12 is that case.  It has two roots; cell 10131
    # (m1 1.046) fails re-correction at event 2.919e-08 against the 2e-08 gate,
    # leaving cell 10132 (m1 1.048) as the only seed.  Both sides collapsed onto
    # it with the same direction-only neighbour and emitted byte-identical
    # results -- same node, same miss_mass to 18 digits, final_masses and steps
    # both null -- whose miss_mass was measured at the HIGH terminus.  Consumed
    # naively that binds a far endpoint on a near endpoint's evidence.
    by_seed: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for item in results:
        seed = item.get("seed_cells") or {}
        by_seed.setdefault(
            (item["source_component"], seed.get("current"), seed.get("previous")), []
        ).append(item)
    for (component, current_cell, previous_cell), group in by_seed.items():
        if len(group) < 2:
            continue
        sides = sorted(str(item["outward_side"]) for item in group)
        for item in group:
            item["scientific_endpoint_resolved"] = False
            item["stopped_reason"] = (
                "degenerate_sides_not_a_scientific_terminus: component "
                f"{component} sides {sides} share seed pair "
                f"(current={current_cell}, previous={previous_cell}); one "
                "computation cannot resolve two termini"
            )
            item["terminal"] = None

    passed = all(item["scientific_endpoint_resolved"] for item in results)
    payload = {
        "schema": "atlas.v1.sampled-endpoint-resolution/1",
        "claim": (
            "continuous classification of the finite-lattice endpoints of sweep "
            f"components {list(requested)} in {Path(args.supplemental_roots).name}"
        ),
        "components_resolved": list(requested),
        "frozen_gates": {
            "maximum_absolute_event": cont.EVENT_GATE,
            "maximum_periodic_closure": cont.CLOSURE_GATE,
        },
        "results": results,
        "all_sampled_endpoints_resolved": passed,
        "claim_status": "sampled_endpoints_replaced_by_continuous_scientific_termini" if passed else "unresolved_continuation_required",
    }

    def sanitize(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, list):
            return [sanitize(v) for v in value]
        if isinstance(value, dict):
            return {k: sanitize(v) for k, v in value.items()}
        return value

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sanitize(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"passed": passed, "results": [{"component": r["source_component"], "stop": r["stopped_reason"], "terminal": r["terminal"]} for r in results]}, indent=2))
    raise SystemExit(0 if passed else 3)


if __name__ == "__main__":
    main()
