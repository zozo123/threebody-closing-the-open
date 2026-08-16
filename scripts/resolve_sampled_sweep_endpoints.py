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
from pathlib import Path
from typing import Any

import numpy as np

import trace_label_invisible_continuous as cont
from threebody_atlas.critical_geometry import continuation_scales


CATALOG_MATCH_TOL = 7e-3
CATALOG_TANGENT_COS = 0.94
LOOP_MATCH_TOL = 5e-3
LOOP_MIN_SEPARATION = 12


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
) -> dict[str, Any]:
    rows = sorted(rows, key=lambda row: (float(row["masses"][0]), float(row["masses"][1])))
    if outward_side == "low":
        previous_row, current_row = rows[1], rows[0]
    elif outward_side == "high":
        previous_row, current_row = rows[-2], rows[-1]
    else:
        raise ValueError(outward_side)
    previous = strict_supplemental(previous_row)
    current = strict_supplemental(current_row)
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
        "closed_loop",
    }
    return {
        "source_component": int(rows[0]["sweep_component"]),
        "outward_side": outward_side,
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
    args = parser.parse_args()

    cont.require_accelerated_x64()
    supplemental = load(Path(args.supplemental_roots))
    catalog_payload = load(Path(args.catalog_roots))
    minus_catalog = catalog_candidates(catalog_payload, "minus_one")

    comp0 = [row for row in supplemental.get("roots", []) if int(row.get("sweep_component", -1)) == 0]
    comp1 = [row for row in supplemental.get("roots", []) if int(row.get("sweep_component", -1)) == 1]
    if len(comp0) < 2 or len(comp1) < 2:
        raise RuntimeError("supplemental components 0/1 no longer contain enough roots")

    results = [
        trace_endpoint(
            comp0,
            minus_catalog,
            outward_side="low",
            mass_step=args.mass_step,
            max_steps=args.max_steps,
        ),
        trace_endpoint(
            comp1,
            minus_catalog,
            outward_side="high",
            mass_step=args.mass_step,
            max_steps=args.max_steps,
        ),
    ]
    passed = all(item["scientific_endpoint_resolved"] for item in results)
    payload = {
        "schema": "atlas.v1.sampled-endpoint-resolution/1",
        "claim": "continuous classification of the two unmatched minus-one finite-lattice endpoints",
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
