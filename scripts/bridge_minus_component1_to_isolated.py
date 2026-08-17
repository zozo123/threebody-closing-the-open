#!/usr/bin/env python3
"""Test whether minus-one component 1 continues through the isolated component 5 root.

The event-sign raster has a 24-point G- component ending at
(m1,m2)=(1.042,0.8579757) and a single later G- detection at
(1.066,0.8421783), with no sign-change detections on the intervening sampled
vertical lines.  A sign raster cannot distinguish a broken curve from an
even-crossing/projection-turn corridor.  This script does not use raster
adjacency: it continuously advances the corrected G-=0 branch from the final
two component-1 points and asks whether the continuous six-dimensional shooting
chart passes through the independently re-corrected component-5 root with the
same local tangent.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

import trace_label_invisible_continuous as cont


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path}: JSON root must be an object")
    return value


def strict(row: dict[str, Any], label: str) -> cont.StrictPoint:
    return cont._strict_localize(
        row,
        str(row["event_mode"]),
        source="research/evidence/V1_SUPPLEMENTAL_EVENT_SIGN_ROOTS_2026-08-16.json",
        source_id=label,
    )


def serialize_target(state: dict[str, Any]) -> dict[str, Any]:
    point: cont.StrictPoint = state["point"]
    return {
        "source_id": point.source_id,
        "point": cont._serialize_localized(point.localized),
        "d_event_dm": [float(x) for x in state["d_event_dm"]],
        "variational_mass_tangent": [float(x) for x in state["variational_mass_tangent"]],
        "jax_mass_tangent": [float(x) for x in state["jax"]["mass_tangent"]],
        "variational_vs_jax_mass_tangent_abs_cosine": float(
            state["variational_vs_jax_mass_tangent_abs_cosine"]
        ),
        "jax_null_residual": float(state["jax"]["null_residual"]),
        "jax_relative_null_residual": float(state["jax"]["relative_null_residual"]),
        "jax_spectral_gap": float(state["jax"]["spectral_gap"]),
        "jax_diagnostics_passed": cont._jax_diagnostics_pass(state["jax"]),
        "best_segment_miss_scaled": float(state["best_segment_miss_scaled"]),
        "best_segment_tangent_abs_cosine": float(state["best_segment_tangent_abs_cosine"]),
        "best_segment_index": state["best_segment_index"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("supplemental_roots")
    parser.add_argument("output")
    parser.add_argument("--mass-step", type=float, default=0.0025)
    parser.add_argument("--max-steps", type=int, default=40)
    args = parser.parse_args()

    cont.require_accelerated_x64()
    payload = load(Path(args.supplemental_roots))
    comp1 = sorted(
        [
            row
            for row in payload.get("roots", [])
            if int(row.get("sweep_component", -1)) == 1
            and row.get("status") == "ok"
            and row.get("passed")
        ],
        key=lambda row: (float(row["masses"][0]), float(row["masses"][1])),
    )
    comp5 = [
        row
        for row in payload.get("roots", [])
        if int(row.get("sweep_component", -1)) == 5
        and row.get("status") == "ok"
        and row.get("passed")
    ]
    if len(comp1) < 2:
        raise RuntimeError(f"component 1 needs two roots, got {len(comp1)}")
    if len(comp5) != 1:
        raise RuntimeError(f"expected one isolated component-5 root, got {len(comp5)}")
    if any(str(row.get("event_mode")) != "minus_one" for row in [*comp1[-2:], *comp5]):
        raise RuntimeError("bridge seeds are no longer all minus_one")

    previous = strict(comp1[-2], f"component1-cell-{int(comp1[-2]['cell_id'])}")
    current = strict(comp1[-1], f"component1-cell-{int(comp1[-1]['cell_id'])}")
    target = strict(comp5[0], f"component5-cell-{int(comp5[0]['cell_id'])}")
    target_state = cont._target_state(target)

    prev = previous.localized
    cur = current.localized
    step = float(args.mass_step)
    accepted = []
    turns = []
    previous_dm1 = float(current.masses2[0] - previous.masses2[0])
    stopped_reason = "max_steps_exhausted"
    hit_step = None
    post_hit_points = 0

    for index in range(args.max_steps):
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
            stopped_reason = f"continuation_failure_not_a_scientific_terminus: {last_error}"
            break

        a = np.asarray(cur.vector, dtype=float)
        b = point.vector
        cont._update_target(target_state, a, b, index)
        dm1 = float(b[4] - a[4])
        if previous_dm1 and dm1 and previous_dm1 * dm1 < 0.0:
            turns.append(
                {
                    "step_index": index,
                    "kind": "m1_projection_turn_crossed",
                    "before_masses": [float(a[4]), float(a[5]), 1.0],
                    "after_masses": [float(b[4]), float(b[5]), 1.0],
                }
            )
        if dm1:
            previous_dm1 = dm1

        accepted.append(cont._serialize_variational(point))
        target_hit = bool(
            target_state["best_segment_miss_scaled"] <= 4e-3
            and target_state["best_segment_tangent_abs_cosine"] >= 0.95
            and target_state["variational_vs_jax_mass_tangent_abs_cosine"] >= 0.95
            and cont._jax_diagnostics_pass(target_state["jax"])
        )
        if target_hit and hit_step is None:
            hit_step = index
        elif hit_step is not None:
            post_hit_points += 1
            if post_hit_points >= 3:
                stopped_reason = "isolated_root_crossed_and_branch_continued_beyond"
                prev, cur = cur, point.localized
                break

        prev, cur = cur, point.localized
        step = min(float(args.mass_step), trial * 1.35)

    bridge_passed = hit_step is not None and post_hit_points >= 3
    result = {
        "schema": "atlas.v1.continuous-event-bridge/1",
        "claim": "minus-one supplemental component 1 continues through isolated component 5",
        "frozen_gates": {
            "maximum_absolute_event": cont.EVENT_GATE,
            "maximum_periodic_closure": cont.CLOSURE_GATE,
        },
        "seed_previous": {
            "cell_id": int(comp1[-2]["cell_id"]),
            "point": cont._serialize_localized(previous.localized),
        },
        "seed_current": {
            "cell_id": int(comp1[-1]["cell_id"]),
            "point": cont._serialize_localized(current.localized),
        },
        "isolated_target": serialize_target(target_state),
        "accepted_points": accepted,
        "projection_turns_crossed": turns,
        "target_hit_step": hit_step,
        "accepted_points_after_target": post_hit_points,
        "stopped_reason": stopped_reason,
        "continuous_bridge_passed": bridge_passed,
        "claim_status": (
            "component1_and_component5_are_one_continuous_minus_one_curve"
            if bridge_passed
            else "unresolved_component1_to_component5_bridge"
        ),
    }

    def sanitize(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, list):
            return [sanitize(x) for x in value]
        if isinstance(value, dict):
            return {k: sanitize(v) for k, v in value.items()}
        return value

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(sanitize(result), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "continuous_bridge_passed": bridge_passed,
                "accepted_points": len(accepted),
                "target_hit_step": hit_step,
                "post_hit_points": post_hit_points,
                "best_miss_scaled": result["isolated_target"]["best_segment_miss_scaled"],
                "best_tangent_abs_cosine": result["isolated_target"]["best_segment_tangent_abs_cosine"],
                "stopped_reason": stopped_reason,
            },
            indent=2,
        )
    )
    raise SystemExit(0 if bridge_passed else 3)


if __name__ == "__main__":
    main()
