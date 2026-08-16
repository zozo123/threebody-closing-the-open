#!/usr/bin/env python3
"""Compare two critical graphs using frozen semantic multigraph levels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from threebody_atlas.graph_semantics import ComparisonLevel, compare_graphs, load_graph

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument(
        "--level",
        choices=[level.value for level in ComparisonLevel],
        default=ComparisonLevel.SHEET_AWARE.value,
    )
    parser.add_argument("--coordinate-tolerance", default="0")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--expect-different",
        action="store_true",
        help="succeed only when the graphs differ at the requested level",
    )
    args = parser.parse_args()

    left = load_graph(args.left, repository_root=ROOT)
    right = load_graph(args.right, repository_root=ROOT)
    comparison = compare_graphs(
        left,
        right,
        args.level,
        args.coordinate_tolerance,
    )
    rendered = json.dumps(
        comparison.model_dump(mode="json"),
        indent=2,
        sort_keys=True,
    ) + "\n"
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")

    if args.expect_different:
        return 0 if not comparison.equivalent else 1
    return 0 if comparison.equivalent else 1


if __name__ == "__main__":
    raise SystemExit(main())
