#!/usr/bin/env python3
"""Build or verify the deterministic environment manifest and scientific SBOM."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from threebody_atlas.supply_chain import (
        build_environment_manifest,
        build_sbom,
        runtime_observation,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from threebody_atlas.supply_chain import (
        build_environment_manifest,
        build_sbom,
        runtime_observation,
    )


def _encoded(value: dict) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _write_or_check(path: Path, value: dict, *, check: bool) -> None:
    expected = _encoded(value)
    if check:
        if not path.is_file() or path.read_text(encoding="utf-8") != expected:
            raise SystemExit(
                f"stale supply-chain artifact: {path}; regenerate with "
                "scripts/build_supply_chain_manifest.py"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(expected, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output-dir", default="research/provenance")
    parser.add_argument("--runtime-output")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output_dir)
    manifest = build_environment_manifest(root)
    sbom = build_sbom(root, manifest)
    _write_or_check(output / "ENVIRONMENT_LOCK_MANIFEST.json", manifest, check=args.check)
    _write_or_check(output / "SCIENTIFIC_SBOM.json", sbom, check=args.check)
    if args.runtime_output:
        runtime_path = Path(args.runtime_output)
        runtime_path.parent.mkdir(parents=True, exist_ok=True)
        runtime_path.write_text(_encoded(runtime_observation()), encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest_sha256": manifest["manifest_sha256"],
                "actions": manifest["github_actions"]["count"],
                "components": len(sbom["components"]),
                "check": args.check,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
