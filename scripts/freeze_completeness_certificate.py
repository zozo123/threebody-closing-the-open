#!/usr/bin/env python3
"""Freeze a bounded completeness certificate from AL + neck raster artifacts.

The record is built and sealed by :mod:`threebody_atlas.completeness`, which is
also what the assembler uses to *re-verify* it.  A sealed certificate is not
evidence on its own: it only means anything because the assembler re-reads the
source artifacts named here, re-hashes them, and re-derives the same numbers.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


try:  # pragma: no cover - exercised implicitly by both install layouts
    from threebody_atlas.completeness import (
        al_summary,
        build_record,
        neck_summary,
        seal,
        sha256_file,
        verify_certificate,
    )
except ModuleNotFoundError:  # running from a source checkout without an install
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from threebody_atlas.completeness import (
        al_summary,
        build_record,
        neck_summary,
        seal,
        sha256_file,
        verify_certificate,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--al-screen", required=True)
    parser.add_argument("--neck-scan")
    args = parser.parse_args()
    al_path = Path(args.al_screen)
    al = json.loads(al_path.read_text(encoding="utf-8"))
    neck = None
    neck_path = None
    if args.neck_scan:
        neck_path = Path(args.neck_scan)
        neck = json.loads(neck_path.read_text(encoding="utf-8"))

    sources: list[dict[str, Any]] = [
        {
            "role": "active_learning",
            "path": str(args.al_screen),
            "sha256": sha256_file(al_path),
        }
    ]
    if neck_path is not None:
        sources.append(
            {
                "role": "neck_scan",
                "path": str(args.neck_scan),
                "sha256": sha256_file(neck_path),
            }
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    record = seal(build_record(al, neck, sources))
    passed = record["passed"] is True

    verification_errors: list[str] = []
    if passed:
        # A certificate the assembler would reject must never leave this script
        # claiming to have passed.  This keeps freezer and verifier in lockstep:
        # if the sources cannot be re-read, re-hashed, and re-derived from where
        # this record says they are, the claim is downgraded before it is written.
        repo_root = Path(__file__).resolve().parents[1]
        passed, verification_errors = verify_certificate(
            record, repo_root=repo_root, certificate_path=output
        )
        if not passed:
            record["passed"] = False
            record["self_verification_errors"] = verification_errors
            record["note"] = (
                "Completeness not frozen: this record could not be re-verified against the "
                "artifacts it names. " + "; ".join(verification_errors)
            )
            record = seal(record)

    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    al_stats = al_summary(al)
    neck_stats = neck_summary(neck)
    print(
        json.dumps(
            {
                "passed": passed,
                "al_clean": al_stats["clean"],
                "neck_done": neck_stats["done"],
                "neck_clean": neck_stats["clean"],
                "neck_boundary_truncated_merge_test": neck_stats[
                    "any_boundary_truncated_merge_test"
                ],
                "neck_line_without_stable_sample": neck_stats[
                    "any_line_without_stable_sample"
                ],
                "neck_all_lines_separated": neck_stats["all_lines_separated"],
                "self_verification_errors": verification_errors,
            },
            indent=2,
        )
    )
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
