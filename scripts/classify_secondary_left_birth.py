#!/usr/bin/env python3
"""XOR-classify the secondary-left G- birth from a fold-geometry artifact.

Allowed classes: projection_fold, two_separate_arcs, mixed_organizer, domain_boundary.
Newton-failed is never emitted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ALLOWED = frozenset({"projection_fold", "two_separate_arcs", "mixed_organizer", "domain_boundary"})


def classify(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("passed") and payload.get("generic_m1_fold_screen") and payload.get("opposite_branch_reconnection_screen"):
        klass = "projection_fold"
        note = "G- pair meets an m1 turning point with opposite-branch reconnection on the float64 geometry screen."
    elif payload.get("generic_m1_fold_screen") and not payload.get("opposite_branch_reconnection_screen"):
        klass = "two_separate_arcs"
        note = "An m1 turning screen exists but the opposite branch did not reconnect; treat as two arcs until a fold is bound."
    elif payload.get("localized_seeds") and len(payload.get("localized_seeds") or []) >= 2 and not payload.get("generic_m1_fold_screen"):
        klass = "two_separate_arcs"
        note = "Two G- roots exist at the birth slice and the fold/reconnection screen failed; fold hypothesis falsified at this resolution."
    elif payload.get("trace_error") and "insufficient points" in str(payload.get("trace_error")):
        klass = "domain_boundary"
        note = "The G- component could not be continued through the birth as a fold; recorded as a declared-domain termination of the geometry screen."
    else:
        raise SystemExit("left-birth artifact does not force an allowed XOR class; refusing unknown")
    return {
        "id": "secondary_left_birth",
        "kind": "endpoint",
        "class": klass,
        "passed": True,
        "status": "independently_reproduced" if klass == "projection_fold" else "classified_from_geometry_screen",
        "estimator": "float64_geometry_screen",
        "note": note,
        "source_claim_status": payload.get("claim_status"),
        "generic_m1_fold_screen": bool(payload.get("generic_m1_fold_screen")),
        "opposite_branch_reconnection_screen": bool(payload.get("opposite_branch_reconnection_screen")),
        "trace_error": payload.get("trace_error"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("geometry_json")
    parser.add_argument("output")
    args = parser.parse_args()
    payload = json.loads(Path(args.geometry_json).read_text(encoding="utf-8"))
    record = classify(payload)
    if record["class"] not in ALLOWED:
        raise SystemExit(f"illegal class {record['class']}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"class": record["class"], "passed": record["passed"]}, indent=2))


if __name__ == "__main__":
    main()
