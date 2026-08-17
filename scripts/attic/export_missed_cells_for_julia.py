#!/usr/bin/env python3
"""Write published-cell TSV seeds for every float64 miss.

The Julia published-cell localizer reads both catalog endpoints and infers
the unique event independently. Python residuals are not imported as truth.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = [
    "cell_id",
    "m1",
    "m3",
    "left_m2",
    "left_x1",
    "left_v1",
    "left_v2",
    "left_period",
    "right_m2",
    "right_x1",
    "right_v1",
    "right_v2",
    "right_period",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("python_roots")
    parser.add_argument("brackets_tsv")
    parser.add_argument("output_tsv")
    args = parser.parse_args()

    payload = json.loads(Path(args.python_roots).read_text(encoding="utf-8"))
    attempts = payload.get("attempts") or payload.get("misses") or []
    missed_ids = sorted(
        {
            int(item["cell_id"])
            for item in attempts
            if item.get("status") and item.get("status") != "ok"
        }
    )
    if "misses" in payload and not missed_ids:
        missed_ids = sorted(int(item["cell_id"]) for item in payload["misses"])

    with Path(args.brackets_tsv).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    Path(args.output_tsv).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output_tsv).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for cell_id in missed_ids:
            if not (0 <= cell_id < len(rows)):
                raise SystemExit(f"missed cell {cell_id} is outside the published bracket table")
            writer.writerow({"cell_id": cell_id, **rows[cell_id]})
    print(json.dumps({"missed_cells": len(missed_ids), "output": args.output_tsv}, indent=2))


if __name__ == "__main__":
    main()
