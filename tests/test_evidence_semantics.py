from __future__ import annotations

import copy
from pathlib import Path

from threebody_atlas.evidence_semantics import (
    CERTIFICATE_CRITERION,
    RELEASE_REQUIREMENT,
    artifact_semantics,
    criterion_contract_digest,
    criterion_claims,
    load_registry,
    semantic_contract_digest,
    validate_registry,
)


ROOT = Path(__file__).resolve().parents[1]


def test_committed_search_scope_registry_is_well_formed() -> None:
    registry = load_registry(ROOT)
    assert validate_registry(registry) == []
    assert CERTIFICATE_CRITERION in registry["criteria"]
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
