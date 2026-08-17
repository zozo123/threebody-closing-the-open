#!/usr/bin/env python3
"""Resolve WHICH root artifacts the committed graph was assembled from.

Cell ids in the supplemental space (>= 10000) are PER-RUN SEQUENTIAL INDICES.
They carry no cross-run meaning: cell 10000 is (0.892, 0.753080) in
V1_SUPPLEMENTAL_EVENT_SIGN_ROOTS_2026-08-16.json and (0.882, 0.707904) in the
2026-08-17 set.  So "load the newest supplemental artifact" is not a safe
default -- it resolves the graph's ids against different physical points and
every consumer silently agrees on a wrong answer.  Measured on the committed
graph: globbing for the newest artifact repoints 132 of its 132 supplemental
cells.

The only correct source of truth is the canonical assembly invocation,
scripts/assemble_v1_critical_graph.sh, which names --roots and every
--supplemental-roots the assembler actually used.  Parse that, so consumers
cannot drift from the assembler no matter how the artifacts are dated.
"""
from __future__ import annotations

import json
import re
import shlex
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "scripts" / "assemble_v1_critical_graph.sh"


def root_source_paths(script: Path = CANONICAL) -> list[Path]:
    """--roots plus every --supplemental-roots named by the canonical invocation."""
    text = script.read_text(encoding="utf-8")
    match = re.search(r"^EVIDENCE_ARGS=\((.*?)^\)", text, re.M | re.S)
    if match is None:
        raise RuntimeError(f"{script} has no EVIDENCE_ARGS array")
    body = match.group(1)
    for name, value in re.findall(r'^(\w+)="\$\{\1:-([^}]*)\}"', text, re.M):
        body = body.replace(f'"${name}"', value).replace(f"${name}", value)
    tokens = shlex.split(body)
    wanted = {"--roots", "--supplemental-roots"}
    out: list[Path] = []
    for index, token in enumerate(tokens[:-1]):
        if token in wanted:
            out.append(ROOT / tokens[index + 1])
    if not out:
        raise RuntimeError(f"{script} named no --roots")
    return out


def load_roots(script: Path = CANONICAL) -> list[dict[str, Any]]:
    """Every root the assembler saw, in the order the invocation names them."""
    roots: list[dict[str, Any]] = []
    seen: dict[int, Path] = {}
    for path in root_source_paths(script):
        for record in json.loads(path.read_text(encoding="utf-8"))["roots"]:
            cell = int(record["cell_id"])
            if cell in seen:
                raise RuntimeError(
                    f"cell_id {cell} appears in both {seen[cell].name} and "
                    f"{path.name}; the supplemental id space is per-run, so two "
                    "sources claiming one id means the invocation is inconsistent"
                )
            seen[cell] = path
            roots.append(record)
    return roots


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--emit-combined",
        help="write every root the assembler saw to this path, as a roots document",
    )
    args = parser.parse_args()
    paths = root_source_paths()
    roots = load_roots()
    if args.emit_combined:
        out = Path(args.emit_combined)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "schema": "atlas.v1.combined-graph-roots/1",
                    "claim_status": (
                        "convenience union of the root artifacts named by "
                        "scripts/assemble_v1_critical_graph.sh; not independent evidence"
                    ),
                    "sources": [str(p.relative_to(ROOT)) for p in paths],
                    "roots": roots,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    print(json.dumps({
        "sources": [str(p.relative_to(ROOT)) for p in paths],
        "n_roots": len(roots),
        "cell_id_range": [min(int(r["cell_id"]) for r in roots),
                          max(int(r["cell_id"]) for r in roots)],
    }, indent=2))
