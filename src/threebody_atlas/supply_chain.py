"""Deterministic supply-chain inventory and drift checks.

The scientific stack is more than the repository commit.  This module records
the immutable workflow actions, language lockfiles, package identities and
known external source pins that participate in reproducing evidence.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import re
import ssl
import subprocess
import sys
import tomllib
from collections import Counter
from pathlib import Path
from typing import Any


MANIFEST_SCHEMA = "atlas.v1.environment-lock-manifest/1"
SBOM_SCHEMA = "CycloneDX/1.6"
LOCKFILES = (
    "pyproject.toml",
    "uv.lock",
    "julia/Project.toml",
    "julia/Manifest.toml",
    "julia-latest/Project.toml",
)
_USES = re.compile(r"^\s*(?:-\s*)?uses:\s*([^\s#]+)")
_FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
_DOCKER_DIGEST = re.compile(r"^docker://[^\s@]+@sha256:[0-9a-f]{64}$")
_RUNNER = re.compile(r"^\s*runs-on:\s*(.+?)\s*(?:#.*)?$")
_CAPD_SHA = re.compile(r"git -C third_party/CAPD checkout ([0-9a-f]{40})")
_BASELINE_BLOB = re.compile(r'expected\s*=\s*"([0-9a-f]{40})"')


class SupplyChainError(ValueError):
    """Raised when a supply-chain input is moving or malformed."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_digest(record: dict[str, Any], *, omit: str) -> str:
    body = {key: value for key, value in record.items() if key != omit}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def workflow_files(repo_root: Path) -> list[Path]:
    github = repo_root / ".github"
    return sorted([*github.rglob("*.yml"), *github.rglob("*.yaml")])


def action_inventory(repo_root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    actions: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in workflow_files(repo_root):
        relative = str(path.relative_to(repo_root))
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            match = _USES.match(line)
            if match is None:
                continue
            spec = match.group(1)
            local = spec.startswith("./")
            docker = spec.startswith("docker://")
            repository = None
            reference = None
            immutable = local
            if docker:
                immutable = bool(_DOCKER_DIGEST.fullmatch(spec))
            elif not local:
                if "@" not in spec:
                    errors.append(f"{relative}:{line_number}: external action has no ref: {spec}")
                else:
                    repository, reference = spec.rsplit("@", 1)
                    immutable = bool(_FULL_SHA.fullmatch(reference))
            if not immutable:
                errors.append(
                    f"{relative}:{line_number}: action ref must be an immutable commit SHA: {spec}"
                )
            actions.append(
                {
                    "workflow": relative,
                    "line": line_number,
                    "uses": spec,
                    "repository": repository,
                    "commit": reference if reference and immutable else None,
                    "local": local,
                    "immutable": immutable,
                }
            )
    return actions, errors


def runner_inventory(repo_root: Path) -> dict[str, int]:
    runners: Counter[str] = Counter()
    for path in workflow_files(repo_root):
        for line in path.read_text(encoding="utf-8").splitlines():
            match = _RUNNER.match(line)
            if match:
                runners[match.group(1).strip()] += 1
    return dict(sorted(runners.items()))


def external_source_inventory(repo_root: Path) -> dict[str, Any]:
    capd: set[str] = set()
    baseline_blobs: set[str] = set()
    for path in workflow_files(repo_root):
        text = path.read_text(encoding="utf-8")
        capd.update(_CAPD_SHA.findall(text))
        baseline_blobs.update(_BASELINE_BLOB.findall(text))
    composite = repo_root / ".github/actions/atlas-uv-baseline/action.yml"
    if composite.is_file():
        baseline_blobs.update(_BASELINE_BLOB.findall(composite.read_text(encoding="utf-8")))
    return {
        "capd_commits": sorted(capd),
        "baseline_git_blobs": sorted(baseline_blobs),
    }


def _compat(project: dict[str, Any], name: str) -> Any:
    value = project.get("compat")
    return value.get(name) if isinstance(value, dict) else None


def build_environment_manifest(repo_root: Path) -> dict[str, Any]:
    actions, action_errors = action_inventory(repo_root)
    if action_errors:
        raise SupplyChainError("; ".join(action_errors))
    missing = [path for path in LOCKFILES if not (repo_root / path).is_file()]
    if missing:
        raise SupplyChainError("missing lock input(s): " + ", ".join(missing))

    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    uv_lock = tomllib.loads((repo_root / "uv.lock").read_text(encoding="utf-8"))
    julia_project = tomllib.loads(
        (repo_root / "julia/Project.toml").read_text(encoding="utf-8")
    )
    julia_manifest = tomllib.loads(
        (repo_root / "julia/Manifest.toml").read_text(encoding="utf-8")
    )
    julia_latest = tomllib.loads(
        (repo_root / "julia-latest/Project.toml").read_text(encoding="utf-8")
    )
    record: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "claim_status": (
            "language dependency locks and third-party Actions are immutable; "
            "GitHub-hosted runner images and apt packages remain runtime-observed, "
            "not image-digest-pinned"
        ),
        "lockfiles": [
            {"path": path, "sha256": sha256_file(repo_root / path)} for path in LOCKFILES
        ],
        "runtimes": {
            "python": {
                "project_requires": pyproject["project"]["requires-python"],
                "uv_lock_requires": uv_lock["requires-python"],
                "uv_required": pyproject["tool"]["uv"]["required-version"],
            },
            "julia_release": {
                "project_compat": _compat(julia_project, "julia"),
                "manifest_version": julia_manifest.get("julia_version"),
                "manifest_project_hash": julia_manifest.get("project_hash"),
            },
            "julia_adversarial_latest": {
                "project_compat": _compat(julia_latest, "julia"),
                "locked_manifest": False,
            },
        },
        "github_actions": {
            "all_external_refs_immutable": True,
            "count": len(actions),
            "steps": actions,
        },
        "runner_selectors": runner_inventory(repo_root),
        "external_sources": external_source_inventory(repo_root),
        "known_unfrozen_layers": [
            "ubuntu-latest hosted-runner image",
            "self-hosted runner OS/toolchain",
            "apt repository snapshot and package builds",
            "julia-latest adversarial Project.toml intentionally has no Manifest.toml",
        ],
    }
    record["manifest_sha256"] = canonical_digest(record, omit="manifest_sha256")
    return record


def _hashes_from_uv(package: dict[str, Any]) -> list[dict[str, str]]:
    found: set[str] = set()
    sdist = package.get("sdist")
    if isinstance(sdist, dict) and isinstance(sdist.get("hash"), str):
        found.add(sdist["hash"])
    for wheel in package.get("wheels", []):
        if isinstance(wheel, dict) and isinstance(wheel.get("hash"), str):
            found.add(wheel["hash"])
    return [
        {"alg": "SHA-256", "content": digest.removeprefix("sha256:")}
        for digest in sorted(found)
        if digest.startswith("sha256:")
    ]


def _python_components(repo_root: Path) -> list[dict[str, Any]]:
    lock = tomllib.loads((repo_root / "uv.lock").read_text(encoding="utf-8"))
    components: list[dict[str, Any]] = []
    for package in lock.get("package", []):
        if not isinstance(package, dict):
            continue
        source = package.get("source")
        if isinstance(source, dict) and source.get("editable") is not None:
            continue
        name = str(package["name"])
        version = str(package["version"])
        component: dict[str, Any] = {
            "type": "library",
            "name": name,
            "version": version,
            "purl": f"pkg:pypi/{name}@{version}",
            "properties": [{"name": "atlas:ecosystem", "value": "python/uv"}],
        }
        hashes = _hashes_from_uv(package)
        if hashes:
            component["hashes"] = hashes
        components.append(component)
    return components


def _julia_components(repo_root: Path) -> list[dict[str, Any]]:
    manifest = tomllib.loads((repo_root / "julia/Manifest.toml").read_text(encoding="utf-8"))
    components: list[dict[str, Any]] = []
    for name, raw_entries in manifest.get("deps", {}).items():
        entries = raw_entries if isinstance(raw_entries, list) else [raw_entries]
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            component: dict[str, Any] = {
                "type": "library",
                "name": str(name),
                "version": str(entry.get("version") or manifest.get("julia_version") or "stdlib"),
                "properties": [
                    {"name": "atlas:ecosystem", "value": "julia/manifest"},
                    {"name": "julia:uuid", "value": str(entry.get("uuid") or "")},
                ],
            }
            tree = entry.get("git-tree-sha1")
            if isinstance(tree, str):
                component["hashes"] = [{"alg": "SHA-1", "content": tree}]
            components.append(component)
    return components


def build_sbom(repo_root: Path, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    manifest = manifest or build_environment_manifest(repo_root)
    components = [*_python_components(repo_root), *_julia_components(repo_root)]
    components.sort(key=lambda item: (item["properties"][0]["value"], item["name"], item["version"]))
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "version": 1,
        "metadata": {
            "component": {
                "type": "application",
                "name": "threebody-atlas scientific computation environment",
                "version": "0.1.0",
            },
            "properties": [
                {
                    "name": "atlas:environment-manifest-sha256",
                    "value": manifest["manifest_sha256"],
                }
            ],
        },
        "components": components,
    }


def runtime_observation() -> dict[str, Any]:
    """Capture identities available only inside the executing environment."""
    numpy_config = None
    numpy_version = None
    try:
        import numpy as np

        numpy_version = np.__version__
        stream = io.StringIO()
        # Text output remains portable across NumPy releases whose structured
        # ``mode='dicts'`` support differs.
        from contextlib import redirect_stdout

        with redirect_stdout(stream):
            np.show_config()
        numpy_config = stream.getvalue()
    except (ImportError, TypeError):
        pass
    try:
        uv_version = subprocess.run(
            ["uv", "--version"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        uv_version = None
    return {
        "schema": "atlas.v1.runtime-environment-observation/1",
        "python": sys.version,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "libc": list(platform.libc_ver()),
        "openssl": ssl.OPENSSL_VERSION,
        "uv": uv_version,
        "numpy": {"version": numpy_version, "configuration": numpy_config},
        "github_runner": {
            "image_os": os.getenv("ImageOS"),
            "image_version": os.getenv("ImageVersion"),
            "runner_os": os.getenv("RUNNER_OS"),
            "runner_arch": os.getenv("RUNNER_ARCH"),
        },
    }
