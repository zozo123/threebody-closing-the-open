#!/usr/bin/env python3
"""Recompute scale-invariant diagnostics for the Li--Li--Liao baseline.

This is a diagnostic audit, not a family classifier. Family identity is decided
by continuation connectivity, not clustering alone.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from threebody_atlas.baseline import BaselineRow, iter_baseline


def diagnostics(row: BaselineRow) -> dict[str, float]:
    m1, m2, m3 = row.m1, row.m2, row.m3
    v3 = -(m1 * row.v1 + m2 * row.v2) / m3
    kinetic = 0.5 * (m1 * row.v1**2 + m2 * row.v2**2 + m3 * v3**2)
    potential = -(
        m1 * m2 / abs(1.0 - row.x1)
        + m1 * m3 / abs(row.x1)
        + m2 * m3
    )
    energy = kinetic + potential
    angular_momentum = m1 * row.x1 * row.v1 + m2 * row.v2
    total_mass = m1 + m2 + m3
    l_si = angular_momentum * abs(energy) ** 0.5 / total_mass ** (13.0 / 6.0)
    # Word length is constant for this catalog; omit it because it rescales all rows equally.
    t_si = row.period * abs(energy) ** 1.5 / total_mass ** 2.5
    return {
        "energy": energy,
        "angular_momentum": angular_momentum,
        "L_si": l_si,
        "T_si": t_si,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("output_csv")
    parser.add_argument("output_summary")
    args = parser.parse_args()

    rows = list(iter_baseline(args.dataset))
    pairs = Counter((round(r.m1, 12), round(r.m2, 12), round(r.m3, 12)) for r in rows)
    duplicate_pairs = {k: v for k, v in pairs.items() if v > 1}

    records: list[dict[str, float | int | str]] = []
    for row in rows:
        record: dict[str, float | int | str] = {
            "index": row.index,
            "m1": row.m1,
            "m2": row.m2,
            "m3": row.m3,
            "x1": row.x1,
            "v1": row.v1,
            "v2": row.v2,
            "period": row.period,
            "published_stability": row.published_stability,
        }
        record.update(diagnostics(row))
        records.append(record)

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output_csv).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    t_values = np.asarray([float(r["T_si"]) for r in records])
    l_values = np.asarray([float(r["L_si"]) for r in records])
    corr = float(np.corrcoef(t_values, l_values)[0, 1])
    summary = {
        "rows": len(rows),
        "distinct_mass_triples": len(pairs),
        "duplicate_mass_triples": len(duplicate_pairs),
        "max_multiplicity": max(pairs.values()),
        "T_si_range": [float(t_values.min()), float(t_values.max())],
        "L_si_range": [float(l_values.min()), float(l_values.max())],
        "pearson_Tsi_Lsi": corr,
        "interpretation": (
            "diagnostic only; continuation connectivity is required to establish family identity"
        ),
    }
    Path(args.output_summary).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
