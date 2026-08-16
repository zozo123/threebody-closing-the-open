#!/usr/bin/env python3
"""Detector: verify one completeness certificate against the artifacts it names.

Exit 0 when the certificate verifies, 1 when it does not.  The point of having
this as a standalone probe is that ``scripts/mutation_harness.py`` can aim it at
a certificate whose *sources were altered after sealing*, or whose neck raster
was truncated and then re-sealed with ``passed`` forced back to true, and record
whether the source-digest check and the re-derivation actually fire.

A check nobody has watched reject anything is not evidence.  This is how we
watch.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from threebody_atlas.completeness import verify_certificate
except ModuleNotFoundError:  # running from a source checkout without an install
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from threebody_atlas.completeness import verify_certificate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("certificate")
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="root the recorded source paths must stay inside",
    )
    parser.add_argument(
        "--expect",
        choices=("pass", "fail"),
        default="pass",
        help="what a healthy tree should see; 'fail' inverts the exit status",
    )
    args = parser.parse_args()

    path = Path(args.certificate)
    if not path.is_file():
        print(f"certificate {path} does not exist")
        return 1
    record = json.loads(path.read_text(encoding="utf-8"))
    passed, errors = verify_certificate(
        record, repo_root=Path(args.repo_root), certificate_path=path
    )
    print(json.dumps({"certificate": str(path), "passed": passed, "errors": errors}, indent=2))
    if args.expect == "pass":
        return 0 if passed else 1
    return 0 if not passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
