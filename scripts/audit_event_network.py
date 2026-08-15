#!/usr/bin/env python3
"""Resolve all smooth Floquet zeros hidden inside coarse S/U transition cells.

Near the secondary stable lobe, a 0.001 published cell can contain more than one
spectral critical event.  A Boolean S/U transition therefore cannot safely be
assigned to whichever scalar event happens to be closest to zero.  This audit
checks every smooth reduced-Floquet event equation

  P(+2)=0, P(-2)=0, Delta=0

at both endpoints of every transition bracket in selected m1 windows, and when
a sign change exists it localizes that specific zero independently by
branch-preserving shooting.

The output is a local event network from which codimension-two organizers can
be targeted without conflating event type with S/U orientation.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from threebody_atlas.boundary import evaluate
from threebody_atlas.critical_manifold import event_value, localize_critical_point
from threebody_atlas.liao_family import FamilyPoint

MODES = ("plus_one", "minus_one", "trace_collision")


def point(row: dict[str, str], side: str) -> FamilyPoint:
    return FamilyPoint(
        masses=(float(row["m1"]), float(row[f"{side}_m2"]), float(row["m3"])),
        x1=float(row[f"{side}_x1"]),
        v1=float(row[f"{side}_v1"]),
        v2=float(row[f"{side}_v2"]),
        period=float(row[f"{side}_period"]),
        residual_norm=float("nan"),
        nfev=0,
        success=True,
    )


def in_windows(m1: float) -> bool:
    return 0.994 <= m1 <= 0.999 or 1.040 <= m1 <= 1.045


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("brackets_tsv")
    parser.add_argument("output")
    args = parser.parse_args()

    with Path(args.brackets_tsv).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    records = []
    localized_events = []
    for row in rows:
        m1 = float(row["m1"])
        if not in_windows(m1):
            continue
        left_point = point(row, "left")
        right_point = point(row, "right")
        left = evaluate(left_point)
        right = evaluate(right_point)
        values = {
            mode: [event_value(left.floquet, mode), event_value(right.floquet, mode)]
            for mode in MODES
        }
        sign_modes = [
            mode for mode, (a, b) in values.items()
            if a == 0.0 or b == 0.0 or a * b < 0.0
        ]
        rec = {
            "m1": m1,
            "m2_bracket": [float(row["left_m2"]), float(row["right_m2"])],
            "published_labels": [row["left_label"], row["right_label"]],
            "event_endpoint_values": values,
            "sign_changing_modes": sign_modes,
        }
        records.append(rec)

        stable = left_point if row["left_label"] == "S" else right_point
        unstable = left_point if row["left_label"] == "U" else right_point
        for mode in sign_modes:
            try:
                critical = localize_critical_point(
                    stable,
                    unstable,
                    event_mode=mode,
                    m2_tolerance=2e-9,
                    event_tolerance=2e-8,
                    max_iterations=36,
                    max_closure=1e-7,
                )
            except RuntimeError as exc:
                localized_events.append(
                    {
                        "m1": m1,
                        "source_m2_bracket": rec["m2_bracket"],
                        "published_labels": rec["published_labels"],
                        "event_mode": mode,
                        "status": "localization_failed",
                        "error": str(exc),
                    }
                )
                continue
            p, f = critical.sample.point, critical.sample.floquet
            localized_events.append(
                {
                    "m1": m1,
                    "source_m2_bracket": rec["m2_bracket"],
                    "published_labels": rec["published_labels"],
                    "event_mode": mode,
                    "status": "localized",
                    "masses": p.masses,
                    "x1": p.x1,
                    "v1": p.v1,
                    "v2": p.v2,
                    "period": p.period,
                    "shooting_residual": p.residual_norm,
                    "event_value": critical.event_value,
                    "alpha": f.alpha,
                    "beta": f.beta,
                    "discriminant": f.discriminant,
                    "trace_roots": [[z.real, z.imag] for z in f.trace_roots],
                }
            )

    multi = [r for r in records if len(r["sign_changing_modes"]) > 1]
    none = [r for r in records if not r["sign_changing_modes"]]
    counts = {mode: 0 for mode in MODES}
    for event in localized_events:
        if event["status"] == "localized":
            counts[event["event_mode"]] += 1

    payload = {
        "windows": [[0.994, 0.999], [1.040, 1.045]],
        "transition_brackets_audited": len(records),
        "multi_event_brackets": len(multi),
        "no_smooth_sign_change_brackets": len(none),
        "localized_event_counts": counts,
        "brackets": records,
        "localized_events": localized_events,
        "interpretation": (
            "A coarse S/U cell may contain several spectral event zeros; event-network topology "
            "must be inferred from the localized smooth zeros, not from Boolean orientation alone."
        ),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "transition_brackets_audited": len(records),
        "multi_event_brackets": len(multi),
        "no_smooth_sign_change_brackets": len(none),
        "localized_event_counts": counts,
        "multi_event_examples": multi[:10],
    }, indent=2))


if __name__ == "__main__":
    main()
