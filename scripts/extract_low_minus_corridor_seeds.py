#!/usr/bin/env python3
"""Extract the low-m2 G- corridor that Float64 intermittently drops.

The full-domain sign sweep contains a smooth sequence of vertical G- brackets
from the certified component-1 endpoint at m1=1.042 through m1=1.074. Most of
the intermediate localized charts have closure ~1e-10 but were excluded from
the supplemental graph because the cancellation-sensitive Float64 event
residual happened to exceed 2e-8. Preserve those charts as seeds for the
independent Julia BigFloat slice verifier; do not change any gate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


FIELDS = ("x1", "v1", "v2", "period")


def load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{path}: JSON root must be an object")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep")
    parser.add_argument("output")
    args = parser.parse_args()

    payload = load(Path(args.sweep))
    rows: list[dict[str, Any]] = []
    for row in payload.get("localizations", []):
        if str(row.get("event_mode") or row.get("mechanism")) != "minus_one":
            continue
        masses = row.get("masses")
        if not isinstance(masses, list) or len(masses) < 3:
            continue
        m1, m2, _m3 = (float(x) for x in masses[:3])
        if not (1.042 <= m1 <= 1.074 and m2 < 0.9):
            continue
        if any(row.get(key) is None for key in FIELDS):
            continue
        # The independent verifier sees both the already-passed anchor and the
        # Float64 gate misses. Localizer failures without a chart are excluded
        # rather than fabricated.
        status = str(row.get("status") or "")
        if status not in {"passed", "missed_frozen_gates"}:
            continue
        rows.append(row)

    rows.sort(key=lambda row: float(row["masses"][0]))
    observed = [round(float(row["masses"][0]), 3) for row in rows]
    expected = [round(1.042 + 0.002 * i, 3) for i in range(17)]
    if observed != expected:
        raise SystemExit(
            "low-minus corridor is not the expected contiguous 1.042..1.074 sequence: "
            f"observed={observed}"
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # verify_critical_points.jl deliberately accepts exactly these ten columns.
    lines = ["name\tevent_mode\tm1\tm2\tm3\tx1\tv1\tv2\tperiod\tscreening_event"]
    for row in rows:
        m1, m2, m3 = row["masses"][:3]
        lines.append(
            "\t".join(
                (
                    f"low-minus-m1-{float(m1):.3f}",
                    "minus_one",
                    repr(float(m1)),
                    repr(float(m2)),
                    repr(float(m3)),
                    repr(float(row["x1"])),
                    repr(float(row["v1"])),
                    repr(float(row["v2"])),
                    repr(float(row["period"])),
                    repr(float(row.get("event_value") or row.get("event") or 0.0)),
                )
            )
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")

    misses = [row for row in rows if row.get("status") == "missed_frozen_gates"]
    print(
        json.dumps(
            {
                "seeds": len(rows),
                "float64_gate_misses": len(misses),
                "m1_range": [observed[0], observed[-1]],
                "statuses": {
                    status: sum(1 for row in rows if row.get("status") == status)
                    for status in sorted({str(row.get("status")) for row in rows})
                },
                "max_abs_float64_event": max(
                    abs(float(row.get("event_value") or row.get("event") or 0.0))
                    for row in rows
                ),
                "max_closure": max(float(row.get("closure") or 0.0) for row in rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
