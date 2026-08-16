#!/usr/bin/env python3
"""Detector: re-assemble the critical graph and check its structural invariants.

This is deliberately NOT a byte-diff against the committed graph.  A byte-diff
fires for every change, which makes it useless for telling one fault from
another: it cannot distinguish "a transition cell went missing" from "a comment
changed".  This probe re-derives the graph and then asks specific structural
questions whose answers are pinned to a baseline assembled from an unmutated
tree:

    exactly 620 localized roots, covering cell ids 0..619
    root_coverage.complete is true
    no duplicate cell ids on edges
    all 620 cells land on edges
    the edge count and per-edge source-cell counts are unchanged
    the edge orientations and mechanisms are unchanged
    newton_failed is zero
    the number of missing mixed germs has not grown
    the number of unclassified edge endpoints has not grown

``release_ready`` is deliberately NOT among them.  It is false today for honest
scientific reasons and this probe must not create any pressure on that bit.

Exit 0 clean, 1 when an invariant regressed, 2 when the assembler itself broke.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def assemble(repo_root: Path, output: Path) -> tuple[int, str]:
    env = dict(os.environ)
    env["PYTHON"] = sys.executable
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repo_root / "src"), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    result = subprocess.run(
        ["bash", str(repo_root / "scripts/assemble_v1_critical_graph.sh"), str(output)],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(repo_root),
        env=env,
    )
    return result.returncode, result.stdout + result.stderr


def fingerprint(graph: dict[str, Any]) -> dict[str, Any]:
    coverage = graph.get("root_coverage", {})
    edges = graph.get("edges", [])
    return {
        "localized_roots": graph.get("localized_roots"),
        "complete": coverage.get("complete"),
        "duplicate_cell_ids": coverage.get("duplicate_cell_ids"),
        "cells_on_edges": coverage.get("cells_on_edges"),
        "newton_failed": coverage.get("newton_failed"),
        "edge_count": coverage.get("edge_count"),
        "missing_mixed_germs": len(coverage.get("missing_mixed_germs") or []),
        "unclassified_edge_endpoints": len(coverage.get("unclassified_edge_endpoints") or []),
        # Lists, not tuples: this fingerprint round-trips through JSON, and a
        # tuple/list mismatch would otherwise read as a false regression.
        "edge_shape": sorted(
            [
                str(edge.get("mechanism")),
                str(edge.get("orientation")),
                int(edge.get("source_cell_count") or 0),
            ]
            for edge in edges
        ),
        "cell_ids_on_edges": sorted(
            cell for edge in edges for cell in (edge.get("cell_ids") or [])
        ),
    }


def check(current: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    if current["localized_roots"] != 620:
        problems.append(f"localized_roots {current['localized_roots']} != 620")
    if current["complete"] is not True:
        problems.append("root_coverage.complete is not true")
    if current["duplicate_cell_ids"]:
        problems.append(f"duplicate cell ids on edges: {current['duplicate_cell_ids']}")
    if current["cells_on_edges"] != 620:
        problems.append(f"cells_on_edges {current['cells_on_edges']} != 620")
    if current["newton_failed"]:
        problems.append(f"newton_failed {current['newton_failed']} != 0")
    if current["cell_ids_on_edges"] != list(range(620)):
        missing = sorted(set(range(620)) - set(current["cell_ids_on_edges"]))
        problems.append(
            f"edges no longer carry each cell exactly once (missing {missing[:5]}"
            f"{'...' if len(missing) > 5 else ''})"
        )
    for key in ("edge_count", "edge_shape"):
        if current[key] != baseline[key]:
            problems.append(f"{key} changed: {baseline[key]!r} -> {current[key]!r}")
    for key in ("missing_mixed_germs", "unclassified_edge_endpoints"):
        if current[key] > baseline[key]:
            problems.append(f"{key} grew: {baseline[key]} -> {current[key]}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--emit", help="write this tree's fingerprint here and exit 0")
    parser.add_argument("--baseline", help="fingerprint from an unmutated tree")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    with tempfile.TemporaryDirectory() as scratch:
        output = Path(scratch) / "graph.json"
        status, log = assemble(repo_root, output)
        # 0 == release_ready, 2 == assembled but legitimately not release_ready.
        if status not in (0, 2) or not output.is_file():
            print(f"assembler failed with exit {status}\n{log}")
            return 2
        graph = json.loads(output.read_text(encoding="utf-8"))

    current = fingerprint(graph)
    if args.emit:
        Path(args.emit).write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
        print(f"wrote graph fingerprint to {args.emit}")
        return 0

    if not args.baseline:
        parser.error("one of --emit or --baseline is required")
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    problems = check(current, baseline)
    print(
        json.dumps(
            {key: value for key, value in current.items() if key != "cell_ids_on_edges"},
            indent=2,
        )
    )
    if problems:
        print(f"\nGRAPH INVARIANTS REGRESSED ({len(problems)}):")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("\ngraph structural invariants unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
