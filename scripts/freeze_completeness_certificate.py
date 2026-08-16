#!/usr/bin/env python3
"""Freeze a bounded completeness certificate from AL + neck raster artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--al-screen", required=True)
    parser.add_argument("--neck-scan")
    args = parser.parse_args()
    al = json.loads(Path(args.al_screen).read_text(encoding="utf-8"))
    accepted = al.get("accepted_candidates", [])
    attempted = al.get("attempted", accepted)
    stable = [row for row in accepted if row.get("corrected", {}).get("screening_stable")]
    neck = None
    if args.neck_scan:
        neck = json.loads(Path(args.neck_scan).read_text(encoding="utf-8"))
    al_clean = bool(
        len(attempted) >= 12
        and len(accepted) == len(attempted)
        and not stable
        and all(
            row.get("shooting_success") is True
            and float(row.get("shooting_residual", float("inf"))) <= 1e-7
            for row in accepted
        )
    )
    neck_grid = neck.get("grid", {}) if isinstance(neck, dict) else {}
    neck_step = neck_grid.get("step")
    neck_gap = neck.get("minimum_resolved_unstable_gap") if isinstance(neck, dict) else None
    neck_done = bool(
        isinstance(neck, dict)
        and neck.get("completed") is True
        and neck_grid
        and neck_grid.get("samples")
        and neck.get("line_summaries")
        and float(neck.get("max_shooting_residual", float("inf"))) <= 1e-7
    )
    # A raster whose stable lobes run off the edge of the scan window cannot
    # decide the merge question at all: the wall that opens the second lobe may
    # simply lie outside the sampled box.  Such a raster is refused explicitly
    # rather than being scored as either merged or separated, and the flags must
    # be present -- a pre-truncation-analysis raster (schema/2) carries no
    # verdicts and therefore cannot freeze completeness.
    neck_truncated = neck.get("any_boundary_truncated_merge_test") if isinstance(neck, dict) else None
    neck_empty_lines = neck.get("any_line_without_stable_sample") if isinstance(neck, dict) else None
    neck_separated = neck.get("all_lines_separated") if isinstance(neck, dict) else None
    neck_clean = bool(
        neck_done
        and neck.get("any_vertical_merge") is False
        and neck_truncated is False
        and neck_empty_lines is False
        and neck_separated is True
        and neck_gap is not None
        and neck_step is not None
        and float(neck_gap) + 1e-12 >= float(neck_step)
    )
    passed = al_clean and neck_clean
    record: dict[str, Any] = {
        "schema": "atlas.v1.completeness-certificate/1",
        "passed": passed,
        "domain": {
            "catalog": "Li-Li-Liao unequal-mass non-hierarchical published grid",
            "neck": None if not neck else neck.get("grid"),
        },
        "resolution": None if not neck else neck.get("grid", {}).get("step"),
        "active_learning": {
            "attempted": len(attempted),
            "accepted": len(accepted),
            "screening_stable_hidden_pockets": len(stable),
            "interpretation": (
                "off-grid proposals corrected onto the known sheet; no hidden stable pocket in this sample"
                if al_clean
                else "inspect accepted stable points before freezing completeness"
            ),
        },
        "neck": None
        if not neck
        else {
            "samples": neck.get("grid", {}).get("samples"),
            "minimum_resolved_unstable_gap": neck.get("minimum_resolved_unstable_gap"),
            "any_vertical_merge": neck.get("any_vertical_merge"),
            "any_boundary_truncated_merge_test": neck_truncated,
            "any_line_without_stable_sample": neck_empty_lines,
            "any_stable_interval_touches_boundary": neck.get(
                "any_stable_interval_touches_boundary"
            ),
            "all_lines_separated": neck_separated,
            "merge_verdict_counts": neck.get("merge_verdict_counts"),
            "boundary_truncated_lines": neck.get("boundary_truncated_lines"),
            "max_shooting_residual": neck.get("max_shooting_residual"),
            "completed": neck.get("completed"),
            "topology_clean": neck_clean,
        },
        "note": (
            "Bounded completeness: no additional stability pocket found in the AL sample "
            "and the neck raster completed at the declared local resolution."
            if passed
            else (
                "Completeness not frozen: require a completed, closure-gated neck raster whose every "
                "line is separated by a resolved interior unstable gap of at least one grid step, with "
                "no interior vertical merge and no line whose merge verdict is limited by scan-window "
                "truncation, plus a clean AL pocket screen."
            )
        ),
        "sources": [
            {
                "role": "active_learning",
                "path": str(args.al_screen),
                "sha256": hashlib.sha256(Path(args.al_screen).read_bytes()).hexdigest(),
            },
            *(
                [
                    {
                        "role": "neck_scan",
                        "path": str(args.neck_scan),
                        "sha256": hashlib.sha256(Path(args.neck_scan).read_bytes()).hexdigest(),
                    }
                ]
                if args.neck_scan
                else []
            ),
        ],
    }
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":"))
    record["sha256_content"] = hashlib.sha256(canonical.encode()).hexdigest()
    text = json.dumps(record, indent=2) + "\n"
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(text, encoding="utf-8")
    print(
        json.dumps(
            {
                "passed": passed,
                "al_clean": al_clean,
                "neck_done": neck_done,
                "neck_clean": neck_clean,
                "neck_boundary_truncated_merge_test": neck_truncated,
                "neck_all_lines_separated": neck_separated,
            },
            indent=2,
        )
    )
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
