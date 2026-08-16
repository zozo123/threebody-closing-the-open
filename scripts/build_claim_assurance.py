#!/usr/bin/env python3
"""Build or verify the per-claim assurance matrix and weakest-link report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from threebody_atlas.assurance import (  # noqa: E402
    build_matrix,
    build_weakest_link_report,
    verify_committed_artifacts,
)

MANIFEST = ROOT / "research/DISCOVERY_RELEASE.json"
POLICY = ROOT / "research/ASSURANCE_DIMENSIONS.json"
MATRIX = ROOT / "research/evidence/V1_CLAIM_ASSURANCE_MATRIX.json"
REPORT = ROOT / "research/evidence/V1_WEAKEST_LINK_REPORT.json"


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"{path.relative_to(ROOT)} must contain a JSON object")
    return value


def _render(value: dict) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    manifest = _load(MANIFEST)
    policy = _load(POLICY)
    matrix = build_matrix(manifest, policy, ROOT)
    report = build_weakest_link_report(matrix)
    if args.check:
        verify_committed_artifacts(
            ROOT,
            manifest,
            policy,
            _load(MATRIX),
            _load(REPORT),
        )
        verb = "verified"
    else:
        MATRIX.write_text(_render(matrix), encoding="utf-8")
        REPORT.write_text(_render(report), encoding="utf-8")
        verb = "wrote"
    print(
        f"{verb} {matrix['claim_count']} claims x {matrix['dimension_count']} dimensions; "
        f"numerical_paper_ready={matrix['release_policy']['numerical_paper_ready']}; "
        f"theorem_grade_ready={matrix['release_policy']['theorem_grade_ready']}"
    )


if __name__ == "__main__":
    main()
