#!/usr/bin/env python3
"""Derive how much of the declared mass domain the completeness screen covers.

"Completeness" in this project is a *bounded* screen, not a statement about
the whole declared domain.  Its finest component is the stability-neck raster,
whose extent is fixed by the merge step of
``.github/workflows/stability-neck-scan.yml`` (that job refuses tiles that do
not tile the declared rectangle exactly).  The declared v1 mass domain is
``DECLARED_DOMAIN`` in ``scripts/assemble_critical_graph.py``.

Both numbers are read from those files rather than retyped, so the fraction
quoted in ``research/DISCOVERY_RELEASE.json`` and in the paper cannot drift
away from the artifact that produces it.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
NECK_WORKFLOW = ROOT / ".github/workflows/stability-neck-scan.yml"
ASSEMBLER = ROOT / "scripts/assemble_critical_graph.py"


def declared_domain(assembler: Path = ASSEMBLER) -> dict[str, tuple[float, float]]:
    """Read DECLARED_DOMAIN from the assembler without running it."""
    spec = importlib.util.spec_from_file_location("_atlas_assembler", assembler)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"cannot import {assembler}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    domain = module.DECLARED_DOMAIN
    return {key: (float(lo), float(hi)) for key, (lo, hi) in domain.items()}


def neck_raster(workflow: Path = NECK_WORKFLOW) -> dict[str, Any]:
    """Read the frozen neck-raster rectangle from the merge job's own flags."""
    text = workflow.read_text(encoding="utf-8")
    wanted = (
        "expected-m1-min",
        "expected-m1-max",
        "expected-m2-min",
        "expected-m2-max",
        "step",
    )
    found: dict[str, float] = {}
    for flag in wanted:
        match = re.search(rf"--{flag}\s+([0-9.eE+-]+)", text)
        if match is None:
            raise RuntimeError(f"{workflow} does not declare --{flag}")
        found[flag] = float(match.group(1))
    return {
        "m1": (found["expected-m1-min"], found["expected-m1-max"]),
        "m2": (found["expected-m2-min"], found["expected-m2-max"]),
        "step": found["step"],
        "source": workflow.relative_to(ROOT).as_posix(),
    }


def scope(
    *, assembler: Path = ASSEMBLER, workflow: Path = NECK_WORKFLOW
) -> dict[str, Any]:
    domain = declared_domain(assembler)
    raster = neck_raster(workflow)
    domain_area = (domain["m1"][1] - domain["m1"][0]) * (domain["m2"][1] - domain["m2"][0])
    raster_area = (raster["m1"][1] - raster["m1"][0]) * (raster["m2"][1] - raster["m2"][0])
    fraction = raster_area / domain_area
    step = raster["step"]
    m1_lines = int(round((raster["m1"][1] - raster["m1"][0]) / step)) + 1
    m2_lines = int(round((raster["m2"][1] - raster["m2"][0]) / step)) + 1
    return {
        "declared_domain": {"m1": list(domain["m1"]), "m2": list(domain["m2"])},
        "declared_domain_area": domain_area,
        "neck_raster": {"m1": list(raster["m1"]), "m2": list(raster["m2"]), "step": step},
        "neck_raster_area": raster_area,
        "neck_raster_samples": m1_lines * m2_lines,
        "area_fraction": fraction,
        "area_percent": 100.0 * fraction,
        "area_percent_rounded": f"{100.0 * fraction:.4f}",
        "sources": {
            "declared_domain": ASSEMBLER.relative_to(ROOT).as_posix(),
            "neck_raster": raster["source"],
        },
        "interpretation": (
            "The frozen neck raster resolves "
            f"{100.0 * fraction:.4f}% of the declared (m1,m2) area at step {step:g}. "
            "Bounded completeness is a statement about that rectangle plus the "
            "off-grid active-learning sample, not about the declared domain."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()
    record = scope()
    text = json.dumps(record, indent=2) + "\n"
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
