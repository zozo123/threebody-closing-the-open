#!/usr/bin/env python3
"""Measure how tightly GERM_ATTACH_DISTANCE is pinned by the assembly it produces.

The assembler binds a mechanism-polyline edge end to a mixed organizer when the
mass distance between them is at most ``GERM_ATTACH_DISTANCE`` (0.008).  That
number is a modelling choice, not a measured quantity, so the honest question is
how much of it the current graph depends on: below some threshold an attachment
that the committed graph relies on is lost, and above some other threshold an
extra, unintended attachment appears.

This script re-runs the real assembler with only that constant varied and
bisects both edges of the window.  It is an audit tool: it writes no evidence
file, touches no numerical gate, and leaves the assembler's default unchanged.
The window it reports is quoted in research/DISCOVERY_RELEASE.json and in the
paper, so the prose cannot drift away from the code.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ASSEMBLER = ROOT / "scripts/assemble_critical_graph.py"
DEFAULT_INPUTS = [
    "--roots",
    "research/evidence/V1_HYBRID_CRITICAL_ROOTS_2026-08-15.json",
    "--germs",
    "research/evidence/V1_MIXED_GERMS_2026-08-15.json",
]


def _load_assembler() -> Any:
    spec = importlib.util.spec_from_file_location("_atlas_assembler_audit", ASSEMBLER)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"cannot import {ASSEMBLER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assemble_at(threshold: float, inputs: list[str]) -> dict[str, Any]:
    """Run the assembler with GERM_ATTACH_DISTANCE replaced by ``threshold``."""
    module = _load_assembler()
    module.GERM_ATTACH_DISTANCE = threshold
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "graph.json"
        argv = sys.argv
        sys.argv = ["assemble_critical_graph.py", "--output", str(out), *inputs]
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                module.main()
        except SystemExit:
            pass
        finally:
            sys.argv = argv
        return json.loads(out.read_text(encoding="utf-8"))


def germ_attachments(graph: dict[str, Any]) -> list[float]:
    return sorted(
        endpoint["distance_to_germ"]
        for edge in graph["edges"]
        for endpoint in (edge["endpoints"]["start"], edge["endpoints"]["end"])
        if endpoint.get("attachment") == "continuation_germ"
    )


def _bisect(inputs: list[str], keep: float, change: float, target: int) -> float:
    """Bisect between a threshold that yields ``target`` attachments and one that does not."""
    for _ in range(60):
        mid = (keep + change) / 2.0
        if mid in (keep, change):
            break
        if len(germ_attachments(assemble_at(mid, inputs))) == target:
            keep = mid
        else:
            change = mid
    return change


def window(inputs: list[str] | None = None) -> dict[str, Any]:
    inputs = list(inputs or DEFAULT_INPUTS)
    module = _load_assembler()
    default = float(module.GERM_ATTACH_DISTANCE)
    baseline = germ_attachments(assemble_at(default, inputs))
    target = len(baseline)
    # Below the largest distance the assembly actually uses, an attachment is lost.
    lower = max(baseline)
    # Above the next candidate distance, an extra attachment appears.
    upper = _bisect(inputs, keep=default, change=default * 4.0, target=target)
    return {
        "default_germ_attach_distance": default,
        "attachments_at_default": target,
        "attachment_distances": baseline,
        "admissible_window": [lower, upper],
        "window_ratio": upper / lower,
        "default_inside_window": lower <= default < upper,
        "interpretation": (
            f"The {target}-attachment assembly produced by the committed inputs is reproduced only "
            f"for GERM_ATTACH_DISTANCE in [{lower!r}, {upper!r}), a window {upper / lower:.4f}x wide. "
            f"The default {default} sits inside it. Below the window an attachment is lost; at or "
            f"above the upper end an additional attachment appears."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", default="research/evidence/V1_HYBRID_CRITICAL_ROOTS_2026-08-15.json")
    parser.add_argument("--germs", action="append", default=[])
    args = parser.parse_args()
    inputs = ["--roots", args.roots]
    for germ in args.germs or ["research/evidence/V1_MIXED_GERMS_2026-08-15.json"]:
        inputs += ["--germs", germ]
    print(json.dumps(window(inputs), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
