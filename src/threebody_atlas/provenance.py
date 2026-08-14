"""Release-manifest helpers."""
from __future__ import annotations

import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(artifact_paths: list[str | Path], metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    artifacts = []
    for item in artifact_paths:
        path = Path(item)
        artifacts.append({
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return {
        "manifest_version": "atlas.release-manifest.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_revision": os.getenv("GITHUB_SHA"),
        "run_id": os.getenv("GITHUB_RUN_ID"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "artifacts": artifacts,
        "metadata": metadata or {},
    }


def write_manifest(output: str | Path, artifacts: list[str | Path], metadata: dict[str, Any] | None = None) -> None:
    Path(output).write_text(json.dumps(build_manifest(artifacts, metadata), indent=2) + "\n", encoding="utf-8")
