#!/usr/bin/env python3
"""Concatenate catalog and supplemental roots for the sign-topology auditor.

The auditor rebuilds each committed polyline from cell_ids.  After the
graph gained supplemental vertices (cell ids >= 10000) those ids must
appear in --roots or edges_from_graph raises KeyError.  This helper does
not invent roots; it concatenates already-certified records.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def combine(paths: list[Path]) -> dict[str, Any]:
    roots: list[dict[str, Any]] = []
    sources: list[str] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        sources.append(str(path))
        roots.extend(payload.get("roots") or [])
    return {
        "schema": "atlas.v1.combined-critical-roots/1",
        "claim_status": (
            "concatenation of already-certified catalog and supplemental roots; "
            "no new localization"
        ),
        "sources": sources,
        "roots": roots,
        "n_roots": len(roots),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("inputs", nargs="+")
    args = parser.parse_args()
    record = combine([Path(path) for path in args.inputs])
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "n_roots": record["n_roots"], "sources": record["sources"]}))


if __name__ == "__main__":
    main()
