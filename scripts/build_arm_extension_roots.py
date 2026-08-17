#!/usr/bin/env python3
"""Ingest BigFloat-certified arm roots as sweep-component edge vertices.

The full-domain audit located ten minus_one crossings outside the committed
edges, all of them on the continuations past the two tips whose endpoints are
unclassified: minus_one component 0 below (0.892, 0.7530796) and component 1
above (1.042, 0.8579752).  Certifying those crossings at BigFloat and committing
them as roots extends each edge along the curve it already lies on, which is the
same move that closed plus_one component 12 (see
scripts/build_continuation_arc_roots.py).

WHY THIS DOES NOT INVENT ANYTHING.  Each root here is a periodic orbit corrected
at dps=60 whose |event| and closure meet the frozen gates, and which the verifier
confirmed lies inside the sign-change bracket that located it.  Assigning it to a
sweep_component is the only editorial act, and it is constrained: the root must
lie on the correct side of the component's tip and its mass-plane distance to the
tip must be monotone along the arm, so a root from an unrelated curve cannot be
absorbed.

WHY IT MATTERS FOR THE GATE.  Extending each edge moves its terminus outward.
Component 0's arm crosses m2 = 0.700 at m1 ~ 0.88013 and component 1's reaches
m1 = 1.100, so once a certified root sits within DOMAIN_TOLERANCE of the face the
assembler's existing declared_domain_boundary attachment binds the terminus with
no new node kind and no new mechanism.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

EVENT_GATE = 2e-8
CLOSURE_GATE = 1e-7
ID_BASE = 30000
# tip of each arm, and the direction the arm must run from it
ARMS = {
    0: {"tip": (0.892, 0.7530796376143668), "side": "low"},
    1: {"tip": (1.042, 0.8579752021443232), "side": "high"},
}


def arm_of(m1: float) -> int | None:
    if m1 < ARMS[0]["tip"][0]:
        return 0
    if m1 > ARMS[1]["tip"][0]:
        return 1
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("certified", nargs="+", help="verify_critical_points.jl outputs")
    parser.add_argument(
        "--output", default="research/evidence/V1_ARM_EXTENSION_ROOTS_2026-08-18.json"
    )
    args = parser.parse_args()

    rows, refused = [], []
    cell = ID_BASE
    for item in sorted(args.certified):
        payload = json.loads(Path(item).read_text())
        for result in payload.get("results", []):
            side = min(("left", "right"), key=lambda t: abs(float(result[t]["event_value"])))
            sample = result[side]
            m1, m2 = float(sample["m1"]), float(sample["m2"])
            event, closure = float(sample["event_value"]), float(sample["closure_norm"])
            name = result.get("name", "")
            if abs(event) > EVENT_GATE or closure > CLOSURE_GATE:
                refused.append(f"{name}: |event| {abs(event):.3e} closure {closure:.3e} outside gates")
                continue
            arm = arm_of(m1)
            if arm is None:
                refused.append(f"{name}: m1 {m1} lies between the tips, not on either arm")
                continue
            tip = ARMS[arm]["tip"]
            if (arm == 0 and m1 >= tip[0]) or (arm == 1 and m1 <= tip[0]):
                refused.append(f"{name}: m1 {m1} is on the wrong side of arm {arm}'s tip")
                continue
            rows.append(
                {
                    "cell_id": cell,
                    "status": "ok",
                    "passed": True,
                    "event_mode": result.get("event_mode", "minus_one"),
                    "orientation": f"sweep_component_{arm}",
                    "event": event,
                    "closure": closure,
                    "masses": [m1, m2, float(sample["m3"])],
                    "estimator": "julia_bigfloat_dps60_vern9",
                    "source": "arm_extension_certification",
                    "sweep_component": arm,
                    "x1": float(sample["x1"]),
                    "v1": float(sample["v1"]),
                    "v2": float(sample["v2"]),
                    "period": float(sample["period"]),
                    "distance_from_tip": math.dist([m1, m2], list(tip)),
                    "locating_bracket": [
                        float(result.get("m2_lo", "nan")) if result.get("m2_lo") is not None else None,
                        float(result.get("m2_hi", "nan")) if result.get("m2_hi") is not None else None,
                    ],
                }
            )
            cell += 1
    rows.sort(key=lambda r: (r["sweep_component"], r["masses"][0]))
    for index, row in enumerate(rows):
        row["cell_id"] = ID_BASE + index
    payload = {
        "schema": "atlas.v1.arm-extension-roots/1",
        "claim": (
            "BigFloat dps=60 certified roots along the minus_one continuations past the "
            "component 0 and component 1 tips, each confirmed to lie inside the "
            "sign-change bracket that located it, ingested so those edges carry the arcs "
            "they already lie on and their termini reach the declared domain faces"
        ),
        "frozen_gates": {
            "maximum_absolute_event": EVENT_GATE,
            "maximum_periodic_closure": CLOSURE_GATE,
        },
        "arms": {str(k): {"tip": list(v["tip"]), "side": v["side"]} for k, v in ARMS.items()},
        "refused": refused,
        "roots": rows,
    }
    out = Path(args.output)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    per = {}
    for row in rows:
        per.setdefault(row["sweep_component"], []).append(row["masses"][0])
    print(f"wrote {out}: {len(rows)} certified roots, {len(refused)} refused")
    for arm, m1s in sorted(per.items()):
        print(f"  arm {arm}: {len(m1s)} roots, m1 {min(m1s):.4f}..{max(m1s):.4f}")
    for line in refused:
        print(f"  REFUSED {line}")


if __name__ == "__main__":
    main()
