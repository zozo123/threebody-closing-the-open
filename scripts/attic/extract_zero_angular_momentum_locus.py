#!/usr/bin/env python3
"""Extract and characterize the near-zero angular-momentum locus in the baseline."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from threebody_atlas.baseline import iter_baseline


def angular_momentum(row) -> float:
    return row.m1 * row.x1 * row.v1 + row.m2 * row.v2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("output_tsv")
    parser.add_argument("output_json")
    parser.add_argument("--tol", type=float, default=1e-12)
    args = parser.parse_args()

    rows = []
    for row in iter_baseline(args.dataset):
        L = angular_momentum(row)
        if abs(L) <= args.tol:
            rows.append((row, L))
    rows.sort(key=lambda item: (item[0].m2, item[0].m3, item[0].m1))

    Path(args.output_tsv).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output_tsv).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["index","m1","m2","m3","x1","v1","v2","period","L","label"])
        for row, L in rows:
            writer.writerow([
                row.index, row.m1, row.m2, row.m3, row.x1, row.v1, row.v2, row.period, L,
                row.published_stability,
            ])

    groups: dict[tuple[float,float], list[float]] = {}
    for row, _ in rows:
        groups.setdefault((row.m2,row.m3), []).append(row.m1)
    summary = {
        "count": len(rows),
        "tolerance": args.tol,
        "mass_line_groups": [
            {
                "m2": key[0], "m3": key[1], "count": len(vals),
                "m1_min": min(vals), "m1_max": max(vals),
            }
            for key, vals in sorted(groups.items(), key=lambda kv: -len(kv[1]))
        ],
        "interpretation": "candidate continuation bridge locus; zero angular momentum alone does not establish family identity",
    }
    Path(args.output_json).write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
