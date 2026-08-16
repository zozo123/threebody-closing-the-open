from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from threebody_atlas.evidence_semantics import (
    CERTIFICATE_CRITERION,
    FULL_EVENT_COVER_CRITERION,
    RELEASE_REQUIREMENT,
    SemanticContractError,
    artifact_semantics,
    build_certificate_semantics,
    classify_artifact,
    criterion_contract_digest,
    criterion_claims,
    load_registry,
    semantic_contract_digest,
    validate_registry,
    verify_certificate_semantics,
)


ROOT = Path(__file__).resolve().parents[1]


def test_committed_search_scope_registry_is_well_formed() -> None:
    registry = load_registry(ROOT)
    assert validate_registry(registry) == []
    assert CERTIFICATE_CRITERION in registry["criteria"]
    assert FULL_EVENT_COVER_CRITERION in registry["criteria"]
    assert RELEASE_REQUIREMENT in registry["requirements"]


def test_registry_requires_the_certificate_contract_entries() -> None:
    registry = load_registry(ROOT)
    missing_criterion = copy.deepcopy(registry)
    del missing_criterion["criteria"][CERTIFICATE_CRITERION]
    assert f"registry criteria must include {CERTIFICATE_CRITERION!r}" in validate_registry(
        missing_criterion
    )

    missing_requirement = copy.deepcopy(registry)
    del missing_requirement["requirements"][RELEASE_REQUIREMENT]
    assert (
        f"registry requirements must include {RELEASE_REQUIREMENT!r}"
        in validate_registry(missing_requirement)
    )

    missing_full = copy.deepcopy(registry)
    del missing_full["criteria"][FULL_EVENT_COVER_CRITERION]
    assert f"registry criteria must include {FULL_EVENT_COVER_CRITERION!r}" in validate_registry(
        missing_full
    )


def test_artifact_semantics_bind_the_full_criterion_contract() -> None:
    block = artifact_semantics(ROOT, "event_sign_brackets/v1")
    assert len(block["semantic_contract_sha256"]) == 64

    registry = load_registry(ROOT)
    changed = copy.deepcopy(registry)
    changed["criteria"]["event_sign_brackets/v1"]["description"] += " changed"
    expected_changed = criterion_contract_digest(changed, "event_sign_brackets/v1")
    assert block["semantic_contract_sha256"] != expected_changed


def test_historical_transition_scope_is_not_full_critical_set_scope() -> None:
    claims = criterion_claims(load_registry(ROOT), "published_label_brackets/v1")
    assert claims["enumerates_label_transition_roots"] is True
    assert claims["enumerates_full_critical_set"] is False
    assert claims["excludes_even_root_pairs"] is False
    assert claims["excludes_tangencies"] is False
    assert claims["bounded_resolution_only"] is True


def test_used_semantic_change_invalidates_contract_but_unrelated_addition_does_not() -> None:
    registry = load_registry(ROOT)
    criteria = [
        CERTIFICATE_CRITERION,
        "active_learning_pocket/v1",
        "local_neck_raster/v1",
    ]
    before = semantic_contract_digest(
        registry, criterion_ids=criteria, requirement_id=RELEASE_REQUIREMENT
    )

    unrelated = copy.deepcopy(registry)
    unrelated["criteria"]["unused_future_probe/v1"] = copy.deepcopy(
        unrelated["criteria"]["event_sign_brackets/v1"]
    )
    assert (
        semantic_contract_digest(
            unrelated, criterion_ids=criteria, requirement_id=RELEASE_REQUIREMENT
        )
        == before
    )

    breaking = copy.deepcopy(registry)
    breaking["criteria"]["active_learning_pocket/v1"]["description"] += " changed"
    assert (
        semantic_contract_digest(
            breaking, criterion_ids=criteria, requirement_id=RELEASE_REQUIREMENT
        )
        != before
    )


def test_full_event_cover_certificate_can_satisfy_release_requirement() -> None:
    registry = load_registry(ROOT)
    block = build_certificate_semantics(
        [], registry, criterion_id=FULL_EVENT_COVER_CRITERION
    )
    errors, report = verify_certificate_semantics(
        block=block,
        sources=[],
        payloads={},
        resolved_paths={},
        repo_root=ROOT,
        registry=registry,
    )
    assert errors == []
    assert report["criterion_id"] == FULL_EVENT_COVER_CRITERION
    assert report["release_scope_passed"] is True
    assert report["release_scope_errors"] == []

    bounded = build_certificate_semantics([], registry)
    bounded_errors, bounded_report = verify_certificate_semantics(
        block=bounded,
        sources=[],
        payloads={},
        resolved_paths={},
        repo_root=ROOT,
        registry=registry,
    )
    assert bounded_errors == []
    assert bounded_report["criterion_id"] == CERTIFICATE_CRITERION
    assert bounded_report["release_scope_passed"] is False


def test_unbound_new_artifact_without_semantics_is_rejected(tmp_path: Path) -> None:
    registry = load_registry(ROOT)
    path = tmp_path / "new_al.json"
    payload = {"attempted": [], "accepted_candidates": []}
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SemanticContractError, match="no search_semantics.criterion_id"):
        classify_artifact(
            role="active_learning",
            path=path,
            payload=payload,
            sha256="0" * 64,
            repo_root=ROOT,
            registry=registry,
        )


def test_declared_criterion_without_claim_scope_is_rejected(tmp_path: Path) -> None:
    registry = load_registry(ROOT)
    path = tmp_path / "partial.json"
    payload = {"search_semantics": {"criterion_id": "active_learning_pocket/v1"}}
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SemanticContractError, match="omits a matching claim_scope"):
        classify_artifact(
            role="active_learning",
            path=path,
            payload=payload,
            sha256="0" * 64,
            repo_root=ROOT,
            registry=registry,
        )


def test_frozen_binding_types_historical_artifact_without_search_semantics() -> None:
    registry = load_registry(ROOT)
    path = ROOT / "research/evidence/V1_AL_POCKET_SCREEN_2026-08-15.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "search_semantics" not in payload
    criterion_id, origin = classify_artifact(
        role="active_learning",
        path=path,
        payload=payload,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        repo_root=ROOT,
        registry=registry,
    )
    assert (criterion_id, origin) == ("active_learning_pocket/v1", "frozen_binding")
