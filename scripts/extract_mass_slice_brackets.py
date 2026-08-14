#!/usr/bin/env python3
"""Extract published S/U transition brackets by m1 slice from the frozen baseline."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from threebody_atlas.baseline import BaselineRow, iter_baseline


def transition_brackets(rows: list[BaselineRow]) -> list[tuple[BaselineRow, BaselineRow]]:
    rows = sorted(rows, key=lambda r: r.m2)
    out: list[tuple[BaselineRow, BaselineRow]] = []
    for left, right in zip(rows, rows[1:], strict=False):
        if left.published_stability != right.published_stability:
            out.append((left, right))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("output")
    parser.add_argument("--m1-min", type=float, default=0.8)
    parser.add_argument("--m1-max", type=float, default=1.1)
    parser.add_argument("--stride", type=int, default=1, help="keep every Nth distinct m1 slice")
    args = parser.parse_args()

    grouped: dict[float, list[BaselineRow]] = defaultdict(list)
    for row in iter_baseline(args.dataset):
        if args.m1_min - 1e-12 <= row.m1 <= args.m1_max + 1e-12:
            grouped[row.m1].append(row)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with Path(args.output).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "m1",
                "m3",
                "left_m2",
                "left_label",
                "left_x1",
                "left_v1",
                "left_v2",
                "left_period",
                "right_m2",
                "right_label",
                "right_x1",
                "right_v1",
                "right_v2",
                "right_period",
            ]
        )
        for slice_index, m1 in enumerate(sorted(grouped)):
            if slice_index % args.stride:
                continue
            for left, right in transition_brackets(grouped[m1]):
                writer.writerow(
                    [
                        f"{m1:.15g}",
                        f"{left.m3:.15g}",
                        f"{left.m2:.15g}",
                        left.published_stability,
                        f"{left.x1:.17g}",
                        f"{left.v1:.17g}",
                        f"{left.v2:.17g}",
                        f"{left.period:.17g}",
                        f"{right.m2:.15g}",
                        right.published_stability,
                        f"{right.x1:.17g}",
                        f"{right.v1:.17g}",
                        f"{right.v2:.17g}",
                        f"{right.period:.17g}",
                    ]
                )
                total += 1
    print(f"m1_slices={len(grouped)} transition_brackets={total} output={args.output}")


if __name__ == "__main__":
    main()
