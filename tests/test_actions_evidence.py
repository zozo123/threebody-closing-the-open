from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_actions_evidence.py"
SPEC = importlib.util.spec_from_file_location("verify_actions_evidence", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
metadata_errors = MODULE.metadata_errors


DIGEST = "a" * 64
REPOSITORY = "zozo123/threebody-closing-the-open"


def _record() -> dict:
    return {
        "id": "artifact-evidence",
        "kind": "actions_artifact",
        "run_id": 101,
        "artifact_id": 202,
        "sha256": DIGEST,
    }


def _artifact() -> dict:
    return {
        "id": 202,
        "expired": False,
        "digest": f"sha256:{DIGEST}",
        "workflow_run": {
            "id": 101,
            "head_repository_id": 303,
        },
    }


def _run() -> dict:
    return {
        "id": 101,
        "status": "completed",
        "conclusion": "success",
        "repository": {"id": 303, "full_name": REPOSITORY},
    }


def test_matching_actions_evidence_metadata_passes() -> None:
    assert metadata_errors(_record(), _artifact(), _run(), REPOSITORY) == []


def test_actions_evidence_rejects_digest_mismatch() -> None:
    artifact = _artifact()
    artifact["digest"] = "sha256:" + "b" * 64
    errors = metadata_errors(_record(), artifact, _run(), REPOSITORY)
    assert any("artifact digest mismatch" in error for error in errors)


def test_actions_evidence_rejects_expired_or_failed_source() -> None:
    artifact = _artifact()
    artifact["expired"] = True
    run = _run()
    run["conclusion"] = "failure"
    errors = metadata_errors(_record(), artifact, run, REPOSITORY)
    assert any("expired" in error for error in errors)
    assert any("completed/success" in error for error in errors)


def test_actions_evidence_rejects_wrong_run_or_repository() -> None:
    artifact = _artifact()
    artifact["workflow_run"]["id"] = 999
    run = _run()
    run["repository"] = {"id": 404, "full_name": "other/repository"}
    errors = metadata_errors(_record(), artifact, run, REPOSITORY)
    assert any("workflow-run mismatch" in error for error in errors)
    assert any("repository mismatch" in error for error in errors)
    assert any("release repository" in error for error in errors)


def test_actions_evidence_requires_integer_ids_and_hex_digest() -> None:
    record = _record()
    record["run_id"] = "101"
    record["artifact_id"] = 0
    record["sha256"] = "not-a-digest"
    errors = metadata_errors(record, {}, {}, REPOSITORY)
    assert any("run_id must be a positive integer" in error for error in errors)
    assert any("artifact_id must be a positive integer" in error for error in errors)
    assert any("sha256 must be" in error for error in errors)
