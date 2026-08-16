"""Machine-checkable semantics for search and completeness evidence.

Content hashes answer whether bytes changed.  They do not answer whether an
artifact's *meaning* is strong enough for a downstream claim.  This module
binds every search criterion to an explicit claim scope and makes semantic
changes invalidate dependent certificates even when all file hashes still
match.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


REGISTRY_SCHEMA = "atlas.v1.search-scope-registry/1"
REGISTRY_PATH = Path("research/SEARCH_SCOPE_REGISTRY.json")
CERTIFICATE_CRITERION = "bounded_completeness_bundle/v1"
RELEASE_REQUIREMENT = "full_critical_set_release/v1"
CLAIM_KEYS = (
    "enumerates_label_transition_roots",
    "enumerates_full_critical_set",
    "excludes_even_root_pairs",
    "excludes_tangencies",
    "bounded_resolution_only",
)

_VERSIONED_ID = re.compile(r"^[a-z0-9][a-z0-9_-]*/v[1-9][0-9]*$")
_LEGACY_ROLE_CRITERIA = {
    "active_learning": "active_learning_pocket/v1",
    "neck_scan": "local_neck_raster/v1",
}


class SemanticContractError(ValueError):
    """Raised when evidence semantics are missing, malformed, or stale."""


def _canonical_digest(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def load_registry(repo_root: Path) -> dict[str, Any]:
    path = repo_root / REGISTRY_PATH
    try:
        registry = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SemanticContractError(f"search-scope registry is unreadable: {exc}") from exc
    errors = validate_registry(registry)
    if errors:
        raise SemanticContractError("invalid search-scope registry: " + "; ".join(errors))
    return registry


def _claims(value: Any, *, label: str) -> tuple[dict[str, bool] | None, list[str]]:
    if not isinstance(value, dict):
        return None, [f"{label} must be an object"]
    missing = sorted(set(CLAIM_KEYS) - set(value))
    extra = sorted(set(value) - set(CLAIM_KEYS))
    errors: list[str] = []
    if missing:
        errors.append(f"{label} is missing claim(s): {', '.join(missing)}")
    if extra:
        errors.append(f"{label} has unknown claim(s): {', '.join(extra)}")
    for key in CLAIM_KEYS:
        if key in value and not isinstance(value[key], bool):
            errors.append(f"{label}.{key} must be boolean")
    if errors:
        return None, errors
    return {key: value[key] for key in CLAIM_KEYS}, []


def validate_registry(registry: Any) -> list[str]:
    if not isinstance(registry, dict):
        return ["registry must be an object"]
    errors: list[str] = []
    if registry.get("schema") != REGISTRY_SCHEMA:
        errors.append(
            f"registry schema must be {REGISTRY_SCHEMA!r}, got {registry.get('schema')!r}"
        )
    criteria = registry.get("criteria")
    if not isinstance(criteria, dict) or not criteria:
        errors.append("registry criteria must be a non-empty object")
        criteria = {}
    for criterion_id, item in criteria.items():
        if not isinstance(criterion_id, str) or not _VERSIONED_ID.fullmatch(criterion_id):
            errors.append(f"criterion id {criterion_id!r} must end in an explicit /vN")
        if not isinstance(item, dict):
            errors.append(f"criterion {criterion_id!r} must be an object")
            continue
        _value, claim_errors = _claims(
            item.get("claim_scope"), label=f"criteria[{criterion_id!r}].claim_scope"
        )
        errors.extend(claim_errors)
    requirements = registry.get("requirements")
    if not isinstance(requirements, dict) or not requirements:
        errors.append("registry requirements must be a non-empty object")
        requirements = {}
    for requirement_id, item in requirements.items():
        if not isinstance(requirement_id, str) or not _VERSIONED_ID.fullmatch(requirement_id):
            errors.append(f"requirement id {requirement_id!r} must end in an explicit /vN")
        if not isinstance(item, dict):
            errors.append(f"requirement {requirement_id!r} must be an object")
            continue
        required = item.get("required_claims")
        if not isinstance(required, dict) or not required:
            errors.append(f"requirements[{requirement_id!r}].required_claims must be non-empty")
            continue
        for key, expected in required.items():
            if key not in CLAIM_KEYS:
                errors.append(f"requirement {requirement_id!r} has unknown claim {key!r}")
            if not isinstance(expected, bool):
                errors.append(f"requirement {requirement_id!r} claim {key!r} must be boolean")
    bindings = registry.get("artifact_bindings", [])
    if not isinstance(bindings, list):
        errors.append("registry artifact_bindings must be a list")
    else:
        for index, binding in enumerate(bindings):
            if not isinstance(binding, dict):
                errors.append(f"artifact_bindings[{index}] must be an object")
                continue
            if binding.get("criterion_id") not in criteria:
                errors.append(
                    f"artifact_bindings[{index}] names unknown criterion "
                    f"{binding.get('criterion_id')!r}"
                )
            digest = binding.get("sha256")
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                errors.append(f"artifact_bindings[{index}].sha256 must be lowercase sha256")
    return errors


def criterion_claims(registry: dict[str, Any], criterion_id: str) -> dict[str, bool]:
    item = registry["criteria"].get(criterion_id)
    if not isinstance(item, dict):
        raise SemanticContractError(f"unknown search criterion {criterion_id!r}")
    claims, errors = _claims(item.get("claim_scope"), label=f"criterion {criterion_id!r}")
    if errors or claims is None:
        raise SemanticContractError("; ".join(errors))
    return claims


def artifact_semantics(repo_root: Path, criterion_id: str) -> dict[str, Any]:
    """Return the canonical semantics block a new producer should emit."""
    registry = load_registry(repo_root)
    return {
        "criterion_id": criterion_id,
        "claim_scope": criterion_claims(registry, criterion_id),
    }


def _repository_relative(path: Path, repo_root: Path) -> str | None:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return None


def classify_artifact(
    *,
    role: str,
    path: Path,
    payload: dict[str, Any],
    sha256: str,
    repo_root: Path,
    registry: dict[str, Any],
) -> tuple[str, str]:
    """Return ``(criterion_id, origin)`` for a source artifact.

    New artifacts should declare ``search_semantics.criterion_id`` themselves.
    Frozen historical artifacts remain immutable and are typed by exact
    path+digest bindings in the registry.  Synthetic legacy fixtures may use a
    role-specific compatibility classification; the resulting certificate
    still records that inference and is never release-sufficient.
    """
    block = payload.get("search_semantics")
    declared = block.get("criterion_id") if isinstance(block, dict) else None
    relative = _repository_relative(path, repo_root)
    bound: str | None = None
    for binding in registry.get("artifact_bindings", []):
        if binding.get("path") == relative and binding.get("sha256") == sha256:
            bound = str(binding.get("criterion_id"))
            break
    if declared is not None and bound is not None and declared != bound:
        raise SemanticContractError(
            f"source {role!r} declares criterion {declared!r}, but its frozen binding says {bound!r}"
        )
    criterion_id = str(declared or bound or _LEGACY_ROLE_CRITERIA.get(role) or "")
    if not criterion_id:
        raise SemanticContractError(
            f"source {role!r} has no search_semantics.criterion_id or frozen registry binding"
        )
    expected = criterion_claims(registry, criterion_id)
    if isinstance(block, dict) and "claim_scope" in block:
        actual, errors = _claims(block.get("claim_scope"), label=f"source {role!r} claim_scope")
        if errors:
            raise SemanticContractError("; ".join(errors))
        if actual != expected:
            raise SemanticContractError(
                f"source {role!r} claim_scope disagrees with registry criterion {criterion_id!r}"
            )
    origin = "declared" if declared is not None else "frozen_binding" if bound else "legacy_role"
    return criterion_id, origin


def semantic_contract_digest(
    registry: dict[str, Any], *, criterion_ids: list[str], requirement_id: str
) -> str:
    contract = {
        "criteria": {
            criterion_id: registry["criteria"][criterion_id]
            for criterion_id in sorted(set(criterion_ids))
        },
        "requirement": {
            requirement_id: registry["requirements"][requirement_id]
        },
    }
    return _canonical_digest(contract)


def build_certificate_semantics(
    sources: list[dict[str, Any]], registry: dict[str, Any]
) -> dict[str, Any]:
    parent_criteria = {
        str(source["role"]): str(source["criterion_id"])
        for source in sources
        if source.get("role") and source.get("criterion_id")
    }
    criterion_ids = [CERTIFICATE_CRITERION, *parent_criteria.values()]
    return {
        "criterion_id": CERTIFICATE_CRITERION,
        "claim_scope": criterion_claims(registry, CERTIFICATE_CRITERION),
        "parent_criteria": parent_criteria,
        "release_requirement": RELEASE_REQUIREMENT,
        "semantic_contract_sha256": semantic_contract_digest(
            registry, criterion_ids=criterion_ids, requirement_id=RELEASE_REQUIREMENT
        ),
    }


def claims_satisfy(
    claim_scope: dict[str, bool], required_claims: dict[str, bool]
) -> tuple[bool, list[str]]:
    errors = [
        f"claim {key} must be {expected!r}, got {claim_scope.get(key)!r}"
        for key, expected in required_claims.items()
        if claim_scope.get(key) is not expected
    ]
    return not errors, errors


def verify_certificate_semantics(
    *,
    block: Any,
    sources: list[dict[str, Any]],
    payloads: dict[str, dict[str, Any]],
    resolved_paths: dict[str, Path],
    repo_root: Path,
    registry: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    if not isinstance(block, dict):
        return ["certificate is missing search_semantics"], {
            "release_scope_passed": False,
            "release_scope_errors": ["certificate is missing search_semantics"],
        }

    criterion_id = block.get("criterion_id")
    if criterion_id != CERTIFICATE_CRITERION:
        errors.append(
            f"certificate criterion_id must be {CERTIFICATE_CRITERION!r}, got {criterion_id!r}"
        )
    expected_claim_scope = criterion_claims(registry, CERTIFICATE_CRITERION)
    actual_claim_scope, claim_errors = _claims(
        block.get("claim_scope"), label="certificate search_semantics.claim_scope"
    )
    errors.extend(claim_errors)
    if actual_claim_scope is not None and actual_claim_scope != expected_claim_scope:
        errors.append("certificate claim_scope disagrees with the search-scope registry")

    actual_parents: dict[str, str] = {}
    for source in sources:
        role = str(source.get("role") or "")
        payload = payloads.get(role)
        path = resolved_paths.get(role)
        if payload is None or path is None:
            continue
        try:
            derived_criterion, derived_origin = classify_artifact(
                role=role,
                path=path,
                payload=payload,
                sha256=str(source.get("sha256") or ""),
                repo_root=repo_root,
                registry=registry,
            )
        except SemanticContractError as exc:
            errors.append(str(exc))
            continue
        actual_parents[role] = derived_criterion
        if source.get("criterion_id") != derived_criterion:
            errors.append(
                f"source {role!r} criterion_id says {source.get('criterion_id')!r}, "
                f"derived {derived_criterion!r} ({derived_origin})"
            )
    if block.get("parent_criteria") != actual_parents:
        errors.append(
            "certificate parent_criteria do not match the re-derived source semantics"
        )

    criterion_ids = [CERTIFICATE_CRITERION, *actual_parents.values()]
    expected_digest = semantic_contract_digest(
        registry, criterion_ids=criterion_ids, requirement_id=RELEASE_REQUIREMENT
    )
    if block.get("semantic_contract_sha256") != expected_digest:
        errors.append(
            "certificate semantic_contract_sha256 is stale for the current criterion definitions"
        )
    if block.get("release_requirement") != RELEASE_REQUIREMENT:
        errors.append(
            f"certificate release_requirement must be {RELEASE_REQUIREMENT!r}"
        )

    required = registry["requirements"][RELEASE_REQUIREMENT]["required_claims"]
    release_scope_passed, release_scope_errors = claims_satisfy(
        actual_claim_scope or {}, required
    )
    if errors:
        release_scope_passed = False
    return errors, {
        "criterion_id": criterion_id,
        "claim_scope": actual_claim_scope,
        "parent_criteria": actual_parents,
        "release_requirement": RELEASE_REQUIREMENT,
        "required_claims": required,
        "release_scope_passed": release_scope_passed,
        "release_scope_errors": release_scope_errors,
    }
