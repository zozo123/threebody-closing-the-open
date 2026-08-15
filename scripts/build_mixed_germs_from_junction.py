#!/usr/bin/env python3
"""Turn junction traces into four local mixed germs (G± × two directions)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


LABEL_TO_NODE = {
    "principal-lower-left": "mixed_principal_left",
    "principal_lower_left": "mixed_principal_left",
    "secondary-birth": "mixed_secondary_left",
    "secondary_birth": "mixed_secondary_left",
    "secondary-lower-switch": "mixed_secondary_left",
    "principal-lower-right": "mixed_principal_right",
    "principal_lower_right": "mixed_principal_right",
}


def split_trace(trace: dict[str, Any], mixed_node: str) -> list[dict[str, Any]]:
    mode = trace.get("event_mode") or trace.get("mode")
    points = list(trace.get("points") or [])
    seeds = list(trace.get("localized_seeds") or [])
    combined = seeds + points
    if not combined or not mode:
        return []
    # Use the first and last mass samples as the two local directions.
    first, last = combined[0], combined[-1]
    germs = []
    for direction, sample in (("+", first), ("-", last)):
        masses = sample.get("masses")
        germs.append(
            {
                "mixed_node": mixed_node,
                "event_mode": mode,
                "direction": direction,
                "status": "traced",
                "ends_on": mixed_node,
                "masses": masses,
                "stopped_reason": trace.get("stopped_reason"),
            }
        )
    return germs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--junction", action="append", default=[], help="junction-*.json artifacts")
    args = parser.parse_args()
    germs: list[dict[str, Any]] = []
    for path in args.junction:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        label = " ".join(
            [
                str(payload.get("label") or ""),
                str(payload.get("name") or ""),
                Path(path).stem,
                str((payload.get("requested_center") or {})),
            ]
        ).lower()
        mixed_node = str(payload.get("mixed_node") or "")
        for key, node in LABEL_TO_NODE.items():
            if key.replace("_", "-") in label.replace("_", "-"):
                mixed_node = node
                break
        if not mixed_node:
            center = payload.get("requested_center") or {}
            try:
                m1 = float(center.get("m1") if isinstance(center, dict) else center[0])
                if abs(m1 - 0.9295) < 0.01:
                    mixed_node = "mixed_principal_left"
                elif abs(m1 - 0.9965) < 0.01:
                    mixed_node = "mixed_secondary_left"
                elif abs(m1 - 1.0495) < 0.01:
                    mixed_node = "mixed_principal_right"
            except (TypeError, ValueError, KeyError, IndexError):
                mixed_node = mixed_node
        if not mixed_node:
            raise SystemExit(f"cannot map {path} to a mixed node")
        traces = payload.get("traces") or payload.get("germs") or []
        if not traces and payload.get("event_mode"):
            traces = [payload]
        for trace in traces:
            germs.extend(split_trace(trace, mixed_node))
    record = {
        "schema": "atlas.v1.mixed-germs/1",
        "claim_status": "float64 junction traces split into local G+/G- germs; not a substitute for BigFloat vertices",
        "germs": germs,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"germs": len(germs)}, indent=2))


if __name__ == "__main__":
    main()
