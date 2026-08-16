#!/usr/bin/env python3
"""Verify pinned GitHub Actions evidence against GitHub's immutable metadata.

The offline discovery manifest validator can prove repository-file hashes, but an
Actions artifact lives outside the git tree.  A solved release therefore performs
this online check before tagging: artifact id, workflow-run id, non-expiry,
GitHub-reported SHA-256 digest, successful run conclusion, and repository identity
must all agree with the frozen manifest.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "research" / "DISCOVERY_RELEASE.json"
API_ROOT = "https://api.github.com"


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _sha256(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    digest = value.lower().strip()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        return None
    return digest


def metadata_errors(
    record: dict[str, Any],
    artifact: dict[str, Any],
    run: dict[str, Any],
    repository: str,
) -> list[str]:
    """Return all provenance mismatches for one frozen Actions evidence record."""
    errors: list[str] = []
    evidence_id = str(record.get("id", "<unnamed>"))
    run_id = record.get("run_id")
    artifact_id = record.get("artifact_id")
    expected_sha = _sha256(record.get("sha256"))

    if not _positive_int(run_id):
        errors.append(f"{evidence_id}: run_id must be a positive integer")
    if not _positive_int(artifact_id):
        errors.append(f"{evidence_id}: artifact_id must be a positive integer")
    if expected_sha is None:
        errors.append(f"{evidence_id}: sha256 must be 64 lowercase/uppercase hexadecimal characters")

    if artifact.get("id") != artifact_id:
        errors.append(
            f"{evidence_id}: artifact id mismatch manifest={artifact_id} api={artifact.get('id')}"
        )
    if artifact.get("expired") is not False:
        errors.append(f"{evidence_id}: artifact is expired or expiry state is unknown")

    api_digest = artifact.get("digest")
    expected_api_digest = None if expected_sha is None else f"sha256:{expected_sha}"
    if api_digest != expected_api_digest:
        errors.append(
            f"{evidence_id}: artifact digest mismatch manifest={expected_api_digest} api={api_digest}"
        )

    artifact_run = artifact.get("workflow_run") or {}
    if artifact_run.get("id") != run_id:
        errors.append(
            f"{evidence_id}: artifact workflow-run mismatch "
            f"manifest={run_id} api={artifact_run.get('id')}"
        )
    if run.get("id") != run_id:
        errors.append(f"{evidence_id}: run id mismatch manifest={run_id} api={run.get('id')}")
    if run.get("status") != "completed" or run.get("conclusion") != "success":
        errors.append(
            f"{evidence_id}: evidence run must be completed/success, got "
            f"status={run.get('status')} conclusion={run.get('conclusion')}"
        )

    run_repository = run.get("repository") or {}
    api_repository = str(run_repository.get("full_name", ""))
    if api_repository.lower() != repository.lower():
        errors.append(
            f"{evidence_id}: repository mismatch manifest-context={repository} api={api_repository}"
        )

    repository_id = run_repository.get("id")
    head_repository_id = artifact_run.get("head_repository_id")
    if repository_id is not None and head_repository_id != repository_id:
        errors.append(
            f"{evidence_id}: artifact was not produced from the release repository "
            f"head_repository_id={head_repository_id} repository_id={repository_id}"
        )
    return errors


def _fetch_json(url: str, token: str) -> dict[str, Any]:
    request = Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "User-Agent": "threebody-atlas-release-verifier",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urlopen(request, timeout=30) as response:  # noqa: S310 - fixed GitHub API root
            payload = json.load(response)
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"GitHub API request failed for {url}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"GitHub API returned a non-object for {url}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--repository", default=os.getenv("GITHUB_REPOSITORY"))
    parser.add_argument("--api-root", default=API_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repository = args.repository
    token = os.getenv("GITHUB_TOKEN")
    if not repository:
        print("Actions evidence verification requires --repository or GITHUB_REPOSITORY", file=sys.stderr)
        return 2
    if not token:
        print("Actions evidence verification requires GITHUB_TOKEN", file=sys.stderr)
        return 2

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = [item for item in manifest.get("evidence", []) if item.get("kind") == "actions_artifact"]
    if not records:
        print("No Actions evidence records to verify.")
        return 0

    errors: list[str] = []
    api_root = str(args.api_root).rstrip("/")
    for record in records:
        evidence_id = str(record.get("id", "<unnamed>"))
        artifact_id = record.get("artifact_id")
        run_id = record.get("run_id")
        if not _positive_int(artifact_id) or not _positive_int(run_id):
            errors.extend(metadata_errors(record, {}, {}, repository))
            continue
        try:
            artifact = _fetch_json(
                f"{api_root}/repos/{repository}/actions/artifacts/{artifact_id}", token
            )
            run = _fetch_json(f"{api_root}/repos/{repository}/actions/runs/{run_id}", token)
        except RuntimeError as exc:
            errors.append(f"{evidence_id}: {exc}")
            continue
        errors.extend(metadata_errors(record, artifact, run, repository))

    if errors:
        print("Pinned Actions evidence verification failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 2

    print(f"Verified {len(records)} pinned Actions evidence records against GitHub metadata.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
