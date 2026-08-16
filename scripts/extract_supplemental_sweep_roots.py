#!/usr/bin/env python3
"""Turn full-domain sign-sweep curve components into supplemental graph roots.

The 620 catalog S/U cells only sample label-flipping brackets.  Interior
2→1 / 1→2 event curves flip no published S/U label and are therefore absent
from the hybrid census.  The sign-topology audit refuses release_ready until
those curves sit on committed polylines.

This extractor does not invent roots.  It copies only vertices that already
appear on a sweep curve component which is not wholly inside the committed
graph, and it attaches the matching certified localization (frozen 2e-8 /
1e-7) when one exists.  Cell ids start at 10000 so they cannot collide with
catalog ids 0..619.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EVENT_GATE = 2e-8
CLOSURE_GATE = 1e-7
SUPPLEMENTAL_CELL_BASE = 10000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_census")
    parser.add_argument("output")
    args = parser.parse_args()
    payload = json.loads(Path(args.sweep_census).read_text(encoding="utf-8"))
    localizations = list(payload.get("localizations") or [])
    loc_index: dict[tuple[float, float, str], dict[str, Any]] = {}
    for row in localizations:
        if row.get("status") != "passed":
            continue
        masses = row.get("masses") or [row.get("m1"), row.get("m2")]
        if not masses or masses[0] is None or masses[1] is None:
            continue
        key = (round(float(masses[0]), 6), round(float(masses[1]), 6), str(row.get("mechanism")))
        loc_index[key] = row

    roots: list[dict[str, Any]] = []
    components_used: list[int] = []
    next_id = SUPPLEMENTAL_CELL_BASE
    for index, component in enumerate(payload.get("curve_components") or []):
        if component.get("in_committed_graph") and not component.get("partly_in_committed_graph"):
            continue
        vertices = [
            vertex
            for vertex in (component.get("vertices") or component.get("points") or [])
            if vertex.get("committed_edge_matched") is not True
        ]
        if not vertices:
            continue
        components_used.append(index)
        mechanism = str(component.get("mechanism"))
        for vertex in vertices:
            m1 = float(vertex.get("m1"))
            m2 = float(vertex.get("m2"))
            match = loc_index.get((round(m1, 6), round(m2, 6), mechanism))
            event = match.get("event_value", vertex.get("event_value")) if match else vertex.get("event_value")
            closure = match.get("closure", vertex.get("closure")) if match else vertex.get("closure")
            if event is None or closure is None:
                continue
            if abs(float(event)) > EVENT_GATE or float(closure) > CLOSURE_GATE:
                continue
            n_unstable = (match or {}).get("n_unstable_bracket") or []
            row = {
                "cell_id": next_id,
                "status": "ok",
                "passed": True,
                "event_mode": mechanism,
                "orientation": f"sweep_component_{index}",
                "event": float(event),
                "closure": float(closure),
                "masses": [m1, m2, 1.0],
                "estimator": "float64",
                "source": "full_domain_event_sign_sweep",
                "sweep_component": index,
                "n_unstable_bracket": n_unstable,
                "census_would_bracket": vertex.get("census_would_bracket"),
                "committed_edge_matched": vertex.get("committed_edge_matched"),
            }
            if match:
                for key in ("x1", "v1", "v2", "period"):
                    if match.get(key) is not None:
                        row[key] = match[key]
            roots.append(row)
            next_id += 1

    record = {
        "schema": "atlas.v1.supplemental-event-sign-roots/1",
        "claim_status": (
            "label-invisible and graph-absent event-sign roots certified at the frozen "
            "2e-8 / 1e-7 gates; they extend the critical graph beyond the 620 S/U cells"
        ),
        "source_sweep": str(args.sweep_census),
        "source_run": payload.get("run_id"),
        "frozen_gates": {"event": EVENT_GATE, "closure": CLOSURE_GATE},
        "components_used": components_used,
        "localized_roots": len(roots),
        "max_abs_event": max((abs(float(row["event"])) for row in roots), default=0.0),
        "max_closure": max((float(row["closure"]) for row in roots), default=0.0),
        "roots": roots,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "localized_roots": record["localized_roots"],
                "components_used": components_used,
                "max_abs_event": record["max_abs_event"],
                "max_closure": record["max_closure"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
