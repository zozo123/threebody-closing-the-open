#!/usr/bin/env python3
"""Produce V1_CONTINUATION_ARC_ROOTS from the endpoint-resolution walks.

WHY THIS EXISTS.  The 2026-08-17 consolidation audit (issue #212 section 6) found
that research/evidence/V1_CONTINUATION_ARC_ROOTS_2026-08-17.json -- an input of the
canonical invocation -- carried a schema emitted by no script in the repository: it
had been produced inline and could not be re-derived.  A committed evidence artifact
whose producer does not exist is unreviewable, which is the same defect class as an
unpinned producing script.  This script is the producer, and it must regenerate the
committed artifact byte-for-byte from the committed resolution evidence alone.

WHAT IT DOES.  scripts/resolve_sampled_sweep_endpoints.py records, for each resolved
outward continuation, every accepted point of the walk -- each one a corrected
periodic orbit with its own event and closure values.  Points inside the frozen
gates are genuine critical roots the walk traversed, so the sweep-component edge
should carry them as vertices rather than only its lattice samples.

Selection rules, each learned the hard way on 2026-08-17:
  * Only components for which BOTH termini resolved contribute (component 12).
    Components 0 and 1 contribute one accepted point each; ingesting those extends
    their edges without resolving the new terminus -- measured effect: the
    unclassified-endpoint count went 2 -> 3 before they were excluded.
  * Points coinciding with an already-committed root are dropped, so no m1 slice
    carries a repeated vertex.  The comp-12 high walk's last accepted point IS the
    committed continuation root (cell 10133); keeping both made the edge
    non-single-valued over m1 and tripped test_committed_edges_are_graphs_over_m1.
  * Cell ids are assigned sequentially from 20000 across ALL gate-passing accepted
    points (in file order) BEFORE filtering, so the committed ids 20002..20022 stay
    stable as long as the resolution artifact does.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

EVENT_GATE = 2e-8
CLOSURE_GATE = 1e-7
ID_BASE = 20000
KEEP_COMPONENTS = frozenset({12})
RESOLUTION = Path("research/evidence/V1_ENDPOINT_RESOLUTION_BIGFLOAT_TIPS_2026-08-17.json")
COMMITTED_ROOTS = (
    Path("research/evidence/V1_HYBRID_CRITICAL_ROOTS_2026-08-15.json"),
    Path("research/evidence/V1_SUPPLEMENTAL_EVENT_SIGN_ROOTS_BIGFLOAT_TIPS_2026-08-17.json"),
    Path("research/evidence/V1_PLUS_ONE_12_CONTINUATION_ROOT_2026-08-17.json"),
)
CLAIM = (
    "gate-passing roots along the resolved plus_one component 12 outward "
    "continuations, so that edge carries the arc its walks traversed rather than "
    "only its three lattice samples. Components 0 and 1 contribute a single "
    "accepted point each, which would extend those edges without resolving the new "
    "terminus, so they are excluded. Points coinciding with an already-committed "
    "root are dropped, so no m1 slice carries a repeated vertex."
)


def committed_points() -> set[tuple[str, float, float]]:
    seen: set[tuple[str, float, float]] = set()
    for path in COMMITTED_ROOTS:
        for root in json.loads(path.read_text())["roots"]:
            seen.add(
                (
                    root["event_mode"],
                    round(root["masses"][0], 12),
                    round(root["masses"][1], 12),
                )
            )
    return seen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="research/evidence/V1_CONTINUATION_ARC_ROOTS_2026-08-17.json",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="regenerate and fail unless byte-identical to the committed artifact",
    )
    args = parser.parse_args()

    resolution = json.loads(RESOLUTION.read_text())
    seen = committed_points()
    rows: list[dict] = []
    cell = ID_BASE
    for result in resolution["results"]:
        if not result.get("scientific_endpoint_resolved"):
            continue
        component = result["source_component"]
        for point in result.get("accepted_points") or []:
            event = float(point["event"])
            closure = float(point["closure"])
            if abs(event) > EVENT_GATE or closure > CLOSURE_GATE:
                continue
            cell_id = cell
            cell += 1
            if component not in KEEP_COMPONENTS:
                continue
            key = (
                point["event_mode"],
                round(float(point["masses"][0]), 12),
                round(float(point["masses"][1]), 12),
            )
            if key in seen:
                continue
            rows.append(
                {
                    "cell_id": cell_id,
                    "status": "ok",
                    "passed": True,
                    "event_mode": point["event_mode"],
                    "orientation": f"sweep_component_{component}",
                    "event": event,
                    "closure": closure,
                    "masses": [float(v) for v in point["masses"]],
                    "estimator": "float64_variational_continuation",
                    "source": "endpoint_resolution_accepted_point",
                    "sweep_component": component,
                    "x1": float(point["x1"]),
                    "v1": float(point["v1"]),
                    "v2": float(point["v2"]),
                    "period": float(point["period"]),
                    "provenance_walk": f"component_{component}_{result['outward_side']}",
                }
            )
    payload = {
        "schema": "atlas.v1.continuation-arc-roots/1",
        "claim": CLAIM,
        "frozen_gates": {
            "maximum_absolute_event": EVENT_GATE,
            "maximum_periodic_closure": CLOSURE_GATE,
        },
        "source_resolution": str(RESOLUTION),
        "roots": rows,
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    out = Path(args.output)
    if args.check:
        if out.read_text() != text:
            raise SystemExit(
                f"REGENERATION MISMATCH: {out} does not equal what this producer "
                "derives from the committed resolution evidence"
            )
        print(f"byte-identical: {out} ({len(rows)} roots)")
        return
    out.write_text(text)
    print(f"wrote {out} ({len(rows)} roots, ids {rows[0]['cell_id']}..{rows[-1]['cell_id']})")


if __name__ == "__main__":
    main()
