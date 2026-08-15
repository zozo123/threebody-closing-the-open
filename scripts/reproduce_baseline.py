#!/usr/bin/env python3
"""Screen a deterministic row range from the published 135,445-orbit baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from threebody_atlas.baseline import load_range
from threebody_atlas.cli import screen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--stop", type=int, required=True)
    parser.add_argument("--floquet", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = load_range(args.dataset, args.start, args.stop)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    matches = 0
    classified = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            record = screen(row.candidate(), floquet=args.floquet)
            atlas_label = None
            if record.stability_class.value == "elliptic":
                atlas_label = "S"
            elif record.stability_class.value == "hyperbolic":
                atlas_label = "U"
            if atlas_label is not None:
                classified += 1
                matches += int(atlas_label == row.published_stability)
            payload = record.model_dump(mode="json")
            payload["baseline_row"] = row.index
            payload["published_stability"] = row.published_stability
            payload["screening_agrees"] = atlas_label == row.published_stability if atlas_label else None
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    summary = {
        "rows_requested": [args.start, args.stop],
        "rows_processed": len(rows),
        "classified": classified,
        "agreement_count": matches,
        "agreement_fraction": (matches / classified) if classified else None,
        "note": "Agreement is a float64 screening diagnostic, not independent publication verification.",
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
