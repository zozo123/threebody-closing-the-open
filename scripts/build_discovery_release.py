#!/usr/bin/env python3
"""Validate the open-problem gate and materialize a discovery release dossier."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from threebody_atlas.discovery import (  # noqa: E402
    DiscoveryValidationError,
    build_dossier,
    load_manifest,
    render_latex_claims,
    render_latex_status,
    render_summary,
    validate_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="research/DISCOVERY_RELEASE.json")
    parser.add_argument("--output-dir", default="artifacts/discovery-dossier")
    parser.add_argument("--require-solved", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--emit-paper-status")
    parser.add_argument("--emit-paper-claims")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = (ROOT / args.manifest).resolve()
    try:
        manifest = load_manifest(manifest_path)
        validate_manifest(manifest, ROOT, require_solved=args.require_solved)
    except (DiscoveryValidationError, ValueError, OSError) as exc:
        print(f"Discovery release validation failed:\n{exc}", file=sys.stderr)
        return 2

    if args.emit_paper_status:
        tex_path = (ROOT / args.emit_paper_status).resolve()
        tex_path.parent.mkdir(parents=True, exist_ok=True)
        tex_path.write_text(render_latex_status(manifest), encoding="utf-8")

    if args.emit_paper_claims:
        claims_path = (ROOT / args.emit_paper_claims).resolve()
        claims_path.parent.mkdir(parents=True, exist_ok=True)
        claims_path.write_text(render_latex_claims(manifest), encoding="utf-8")

    summary = render_summary(manifest)
    if os.getenv("GITHUB_STEP_SUMMARY"):
        with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a", encoding="utf-8") as handle:
            handle.write(summary)

    if not args.validate_only:
        output = build_dossier(manifest, manifest_path, ROOT, ROOT / args.output_dir)
        print(f"Discovery dossier: {output}")
    print(f"Scientific status: {manifest['status'].upper()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
