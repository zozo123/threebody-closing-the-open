#!/usr/bin/env python3
"""Derive how much of the declared mass domain the completeness screen covers.

"Completeness" in this project is a *bounded* screen, not a statement about
the whole declared domain.  It has exactly two inputs, and this module measures
how much of the declared domain each of them actually touches:

* the stability-neck raster, whose extent is fixed by the **merge** job of
  ``.github/workflows/stability-neck-scan.yml`` (that job refuses tiles that do
  not tile the declared rectangle exactly), and
* the off-grid active-learning pocket screen
  ``research/evidence/V1_AL_POCKET_SCREEN_2026-08-15.json``, whose twelve
  proposals are *not* a sample of the domain outside the raster: they fall in a
  single tiny pocket at the principal lower transition, and that artifact
  declares itself "AI proposals plus float64 screening only; not scientific
  discovery evidence".

The declared v1 mass domain is ``DECLARED_DOMAIN`` in
``scripts/assemble_critical_graph.py``.

Every number is read from those files rather than retyped, so the fractions
quoted in ``research/DISCOVERY_RELEASE.json``, ``README.md`` and the paper
cannot drift away from the artifacts that produce them.

Two traps this module exists to avoid:

1. ``--step`` appears twice in the neck workflow -- once in the *tile* job and
   once in the *merge* job.  A bare ``re.search`` finds the tile job's copy.
   The frozen raster is what the merge job accepted, so the flags are read only
   from the merge job's own argument block.
2. The merge flags describe what the workflow *asked* for.  ``scope()``
   re-reads the committed raster artifact and refuses to report a figure that
   disagrees with the rectangle, step or sample count actually on disk.
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
NECK_RASTER = ROOT / "research/evidence/V1_NECK_RASTER_2026-08-16.json"
AL_SCREEN = ROOT / "research/evidence/V1_AL_POCKET_SCREEN_2026-08-15.json"
MERGE_SCRIPT = "scripts/merge_stability_neck_scans.py"


def declared_domain(assembler: Path = ASSEMBLER) -> dict[str, tuple[float, float]]:
    """Read DECLARED_DOMAIN from the assembler without running it."""
    spec = importlib.util.spec_from_file_location("_atlas_assembler", assembler)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"cannot import {assembler}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    domain = module.DECLARED_DOMAIN
    return {key: (float(lo), float(hi)) for key, (lo, hi) in domain.items()}


def merge_argument_block(workflow: Path = NECK_WORKFLOW) -> str:
    """Return only the merge job's ``merge_stability_neck_scans.py`` invocation.

    The tile job runs ``scan_stability_neck.py`` with its own ``--step``, so a
    whole-file search for ``--step`` reads the wrong job.  The merge job is the
    one that verifies exact tile coverage and freezes the raster, so its
    argument block -- and nothing else -- defines the frozen rectangle.
    """
    lines = workflow.read_text(encoding="utf-8").splitlines()
    for start, line in enumerate(lines):
        # The script name also appears in this workflow's ``paths:`` triggers.
        # Only a real invocation carries the merge job's --expected-* flags.
        if MERGE_SCRIPT not in line:
            continue
        index = start
        block = [line]
        while block[-1].rstrip().endswith("\\") and index + 1 < len(lines):
            index += 1
            block.append(lines[index])
        text = "\n".join(block)
        if "--expected-m1-min" in text:
            return text
    raise RuntimeError(f"{workflow} has no {MERGE_SCRIPT} invocation with --expected-* flags")


def neck_raster(workflow: Path = NECK_WORKFLOW) -> dict[str, Any]:
    """Read the frozen neck-raster rectangle from the merge job's own flags."""
    text = merge_argument_block(workflow)
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
            raise RuntimeError(f"{workflow} merge job does not declare --{flag}")
        found[flag] = float(match.group(1))
    return {
        "m1": (found["expected-m1-min"], found["expected-m1-max"]),
        "m2": (found["expected-m2-min"], found["expected-m2-max"]),
        "step": found["step"],
        "source": workflow.relative_to(ROOT).as_posix(),
    }


def committed_raster(path: Path = NECK_RASTER) -> dict[str, Any]:
    """The frozen raster artifact the merge job actually produced."""
    record = json.loads(path.read_text(encoding="utf-8"))
    grid = record["grid"]
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "m1": (float(grid["m1"][0]), float(grid["m1"][1])),
        "m2": (float(grid["m2"][0]), float(grid["m2"][1])),
        "step": float(grid["step"]),
        "samples": int(grid["samples"]),
        "completed": bool(record.get("completed")),
        "all_lines_separated": bool(record.get("all_lines_separated")),
        "merge_verdict_counts": record.get("merge_verdict_counts"),
        "minimum_resolved_unstable_gap": record.get("minimum_resolved_unstable_gap"),
        "max_shooting_residual": record.get("max_shooting_residual"),
    }


def active_learning_pocket(path: Path = AL_SCREEN) -> dict[str, Any]:
    """Bounding box of the off-grid active-learning proposals.

    The twelve proposals are not spread over the declared domain; they sit in
    one pocket at the principal lower transition.  Reporting their extent is
    what keeps "completeness outside the raster rests on a 12-proposal off-grid
    sample" from reading as a sample *of the domain*.
    """
    record = json.loads(path.read_text(encoding="utf-8"))
    proposals = [row["proposal"] for row in record["attempted"]]
    m1 = [float(row["m1"]) for row in proposals]
    m2 = [float(row["m2"]) for row in proposals]
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "proposals": len(proposals),
        "m1": (min(m1), max(m1)),
        "m2": (min(m2), max(m2)),
        "claim_status": record.get("claim_status"),
    }


def scope(
    *,
    assembler: Path = ASSEMBLER,
    workflow: Path = NECK_WORKFLOW,
    raster_artifact: Path = NECK_RASTER,
    al_artifact: Path = AL_SCREEN,
) -> dict[str, Any]:
    domain = declared_domain(assembler)
    raster = neck_raster(workflow)
    frozen = committed_raster(raster_artifact)
    step = raster["step"]
    m1_lines = int(round((raster["m1"][1] - raster["m1"][0]) / step)) + 1
    m2_lines = int(round((raster["m2"][1] - raster["m2"][0]) / step)) + 1
    samples = m1_lines * m2_lines

    # The workflow flags say what was requested; the artifact says what exists.
    # A published coverage figure that disagrees with the frozen raster is worse
    # than no figure, so this refuses rather than reports.
    mismatches = [
        f"{name}: workflow {want!r} vs artifact {got!r}"
        for name, want, got in (
            ("m1", raster["m1"], frozen["m1"]),
            ("m2", raster["m2"], frozen["m2"]),
            ("step", step, frozen["step"]),
            ("samples", samples, frozen["samples"]),
        )
        if want != got
    ]
    if mismatches:
        raise RuntimeError(
            "frozen neck raster disagrees with the merge job that produced it: "
            + "; ".join(mismatches)
        )

    domain_area = (domain["m1"][1] - domain["m1"][0]) * (domain["m2"][1] - domain["m2"][0])
    raster_area = (raster["m1"][1] - raster["m1"][0]) * (raster["m2"][1] - raster["m2"][0])
    fraction = raster_area / domain_area

    pocket = active_learning_pocket(al_artifact)
    pocket_area = (pocket["m1"][1] - pocket["m1"][0]) * (pocket["m2"][1] - pocket["m2"][0])
    pocket_fraction = pocket_area / domain_area

    return {
        "declared_domain": {"m1": list(domain["m1"]), "m2": list(domain["m2"])},
        "declared_domain_area": domain_area,
        "neck_raster": {"m1": list(raster["m1"]), "m2": list(raster["m2"]), "step": step},
        "neck_raster_area": raster_area,
        "neck_raster_samples": samples,
        "neck_raster_artifact": frozen,
        "area_fraction": fraction,
        "area_percent": 100.0 * fraction,
        "area_percent_rounded": f"{100.0 * fraction:.4f}",
        "active_learning_pocket": {
            "proposals": pocket["proposals"],
            "m1": list(pocket["m1"]),
            "m2": list(pocket["m2"]),
            "area": pocket_area,
            "area_fraction": pocket_fraction,
            "area_percent_rounded": f"{100.0 * pocket_fraction:.6f}",
            "claim_status": pocket["claim_status"],
        },
        "sources": {
            "declared_domain": ASSEMBLER.relative_to(ROOT).as_posix(),
            "neck_raster_window": raster["source"],
            "neck_raster_artifact": frozen["path"],
            "active_learning": pocket["path"],
        },
        "interpretation": (
            "The frozen neck raster resolves "
            f"{100.0 * fraction:.4f}% of the declared (m1,m2) area at step {step:g} "
            f"({samples} corrected samples). The only other completeness input, the "
            f"{pocket['proposals']}-proposal off-grid active-learning screen, is not a sample of "
            "the domain outside that rectangle: its proposals lie in one pocket spanning "
            f"m1 in [{pocket['m1'][0]:.5f}, {pocket['m1'][1]:.5f}] x "
            f"m2 in [{pocket['m2'][0]:.5f}, {pocket['m2'][1]:.5f}], "
            f"{100.0 * pocket_fraction:.6f}% of the declared area, and that artifact declares "
            f"itself {pocket['claim_status']!r}. Bounded completeness is a statement about the "
            "rectangle and that pocket, not about the declared domain."
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
