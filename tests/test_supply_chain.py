from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from threebody_atlas.supply_chain import (
    SupplyChainError,
    action_inventory,
    build_environment_manifest,
    build_sbom,
    canonical_digest,
)


ROOT = Path(__file__).resolve().parents[1]


def test_every_third_party_action_is_pinned_to_a_commit() -> None:
    actions, errors = action_inventory(ROOT)
    assert errors == []
    external = [item for item in actions if not item["local"]]
    assert external
    assert all(item["immutable"] for item in external)
    assert all(
        len(item["commit"]) == 40
        or re.fullmatch(r"sha256:[0-9a-f]{64}", item["commit"])
        for item in external
    )


def test_moving_action_tag_is_rejected(tmp_path: Path) -> None:
    workflow = tmp_path / ".github/workflows/test.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text("steps:\n  - uses: actions/checkout@v7\n", encoding="utf-8")
    actions, errors = action_inventory(tmp_path)
    assert actions[0]["immutable"] is False
    assert errors == [
        ".github/workflows/test.yml:2: action ref must be an immutable commit SHA: "
        "actions/checkout@v7"
    ]


def test_immutable_docker_action_records_its_digest(tmp_path: Path) -> None:
    digest = "sha256:" + "a" * 64
    workflow = tmp_path / ".github/workflows/test.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text(
        f"steps:\n  - uses: docker://ghcr.io/example/action@{digest}\n",
        encoding="utf-8",
    )
    actions, errors = action_inventory(tmp_path)
    assert errors == []
    assert actions[0]["repository"] == "docker://ghcr.io/example/action"
    assert actions[0]["commit"] == digest
    assert actions[0]["immutable"] is True


def test_external_action_without_at_reports_one_error(tmp_path: Path) -> None:
    workflow = tmp_path / ".github/workflows/test.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text("steps:\n  - uses: actions/checkout\n", encoding="utf-8")
    actions, errors = action_inventory(tmp_path)
    assert actions[0]["immutable"] is False
    assert errors == [
        ".github/workflows/test.yml:2: external action has no ref: actions/checkout"
    ]


def test_quoted_uses_key_with_moving_tag_is_rejected(tmp_path: Path) -> None:
    workflow = tmp_path / ".github/workflows/test.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text(
        'jobs:\n  build:\n    steps:\n      - "uses": actions/checkout@v7\n',
        encoding="utf-8",
    )
    actions, errors = action_inventory(tmp_path)
    assert len(actions) == 1
    assert actions[0]["uses"] == "actions/checkout@v7"
    assert actions[0]["immutable"] is False
    assert errors == [
        ".github/workflows/test.yml:4: action ref must be an immutable commit SHA: "
        "actions/checkout@v7"
    ]


def test_single_quoted_uses_key_with_moving_tag_is_rejected(tmp_path: Path) -> None:
    workflow = tmp_path / ".github/workflows/test.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text(
        "jobs:\n  build:\n    steps:\n      - 'uses': actions/setup-python@v5\n",
        encoding="utf-8",
    )
    actions, errors = action_inventory(tmp_path)
    assert actions[0]["uses"] == "actions/setup-python@v5"
    assert actions[0]["immutable"] is False
    assert any("actions/setup-python@v5" in error for error in errors)


def test_unparseable_workflow_fails_closed(tmp_path: Path) -> None:
    workflow = tmp_path / ".github/workflows/test.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text("jobs: [\n  uses: actions/checkout@v7\n", encoding="utf-8")
    actions, errors = action_inventory(tmp_path)
    assert actions == []
    assert any("not parseable YAML" in error for error in errors)


def test_non_scalar_uses_value_fails_closed(tmp_path: Path) -> None:
    workflow = tmp_path / ".github/workflows/test.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text("steps:\n  - uses:\n      action: actions/checkout@v7\n", encoding="utf-8")
    actions, errors = action_inventory(tmp_path)
    assert any("uses value must be a scalar" in error for error in errors)


def test_committed_environment_manifest_is_exactly_reproducible() -> None:
    expected = build_environment_manifest(ROOT)
    committed = json.loads(
        (ROOT / "research/provenance/ENVIRONMENT_LOCK_MANIFEST.json").read_text()
    )
    assert committed == expected
    assert committed["manifest_sha256"] == canonical_digest(
        committed, omit="manifest_sha256"
    )
    assert committed["github_actions"]["all_external_refs_immutable"] is True


def test_committed_scientific_sbom_is_exactly_reproducible() -> None:
    manifest = build_environment_manifest(ROOT)
    expected = build_sbom(ROOT, manifest)
    committed = json.loads((ROOT / "research/provenance/SCIENTIFIC_SBOM.json").read_text())
    assert committed == expected
    ecosystems = {
        component["properties"][0]["value"] for component in committed["components"]
    }
    assert ecosystems == {"python/uv", "julia/manifest"}
    assert any(component["name"] == "numpy" for component in committed["components"])
    assert any(component["name"] == "OrdinaryDiffEqVerner" for component in committed["components"])


def test_lockfile_drift_changes_environment_identity(tmp_path: Path) -> None:
    import shutil

    for relative in (
        "pyproject.toml",
        "uv.lock",
        "julia/Project.toml",
        "julia/Manifest.toml",
        "julia-latest/Project.toml",
        ".github/workflows/ci.yml",
    ):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    before = build_environment_manifest(tmp_path)["manifest_sha256"]
    lock = tmp_path / "uv.lock"
    lock.write_text(lock.read_text() + "\n# dependency drift mutation\n", encoding="utf-8")
    after = build_environment_manifest(tmp_path)["manifest_sha256"]
    assert before != after


def test_manifest_build_refuses_any_moving_action(tmp_path: Path) -> None:
    import shutil

    for relative in (
        "pyproject.toml",
        "uv.lock",
        "julia/Project.toml",
        "julia/Manifest.toml",
        "julia-latest/Project.toml",
    ):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    workflow = tmp_path / ".github/workflows/bad.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text("steps:\n  - uses: actions/upload-artifact@v4\n", encoding="utf-8")
    with pytest.raises(SupplyChainError, match="immutable commit SHA"):
        build_environment_manifest(tmp_path)


def test_manifest_build_refuses_quoted_moving_action(tmp_path: Path) -> None:
    import shutil

    for relative in (
        "pyproject.toml",
        "uv.lock",
        "julia/Project.toml",
        "julia/Manifest.toml",
        "julia-latest/Project.toml",
    ):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    workflow = tmp_path / ".github/workflows/quoted.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text(
        'steps:\n  - "uses": actions/upload-artifact@v4\n',
        encoding="utf-8",
    )
    with pytest.raises(SupplyChainError, match="immutable commit SHA"):
        build_environment_manifest(tmp_path)
