"""Machine-derived, multidimensional assurance for discovery claims.

Assurance is intentionally not a probability or a score.  Every claim keeps
every required dimension, its exact state, its evidence identities, and its
blocker.  Readiness is the conjunction of required cells, never an average.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from threebody_atlas.discovery import DiscoveryValidationError, validate_manifest


POLICY_SCHEMA = "atlas.claim-assurance-policy.v1"
MATRIX_SCHEMA = "atlas.claim-assurance-matrix.v1"
REPORT_SCHEMA = "atlas.weakest-link-report.v1"
STATUSES = {
    "pass",
    "fail",
    "not_applicable",
    "not_run",
    "infrastructure_blocked",
    "scientifically_unresolved",
}
SEVERITY = {
    "fail": 5,
    "scientifically_unresolved": 4,
    "infrastructure_blocked": 3,
    "not_run": 2,
    "not_applicable": 1,
    "pass": 0,
}
DIMENSION_IDS = (
    "source_data_lineage",
    "spec_conformance",
    "orbit_sheet_identity",
    "numerical_correction",
    "direct_continuation_topology",
    "physical_formulation_independence",
    "conditioning_margin",
    "truth_known_calibration",
    "mutation_metamorphic_coverage",
    "blind_n_version",
    "platform_systematics_envelope",
    "literature_novelty",
    "rigorous_certificate",
    "unresolved_contradictions",
)
PROFILE_IDS = ("numerical_paper", "theorem_grade")


class AssuranceError(ValueError):
    """Raised when assurance policy or artifacts are malformed or stale."""


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_policy(policy: Any) -> None:
    errors: list[str] = []
    if not isinstance(policy, dict):
        raise AssuranceError("assurance policy must be an object")
    if policy.get("schema") != POLICY_SCHEMA:
        errors.append(f"policy schema must be {POLICY_SCHEMA!r}")
    dimensions = policy.get("dimensions")
    if not isinstance(dimensions, list):
        errors.append("policy dimensions must be a list")
        dimensions = []
    ids = [item.get("id") for item in dimensions if isinstance(item, dict)]
    if tuple(ids) != DIMENSION_IDS:
        errors.append("policy dimensions must contain the frozen ordered dimension set")
    for item in dimensions:
        if not isinstance(item, dict):
            errors.append("every dimension must be an object")
            continue
        if not isinstance(item.get("description"), str) or not item["description"].strip():
            errors.append(f"dimension {item.get('id')!r} needs a description")
        evaluator = item.get("evaluator")
        if not isinstance(evaluator, dict) or not isinstance(evaluator.get("type"), str):
            errors.append(f"dimension {item.get('id')!r} needs an evaluator")
        if item.get("missing_status", "not_run") not in STATUSES:
            errors.append(f"dimension {item.get('id')!r} has invalid missing_status")
    profiles = policy.get("profiles")
    if not isinstance(profiles, dict) or tuple(profiles) != PROFILE_IDS:
        errors.append("policy profiles must be numerical_paper then theorem_grade")
    else:
        known = set(DIMENSION_IDS)
        for profile_id, profile in profiles.items():
            required = profile.get("required_dimensions") if isinstance(profile, dict) else None
            if not isinstance(required, list) or not required:
                errors.append(f"profile {profile_id!r} needs required_dimensions")
            elif not set(required) <= known:
                errors.append(f"profile {profile_id!r} names an unknown dimension")
    if errors:
        raise AssuranceError("; ".join(errors))


def _evidence_identity(record: dict[str, Any], root: Path) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "id": record.get("id"),
        "kind": record.get("kind"),
        "role": record.get("role"),
        "valid": False,
    }
    kind = record.get("kind")
    if kind in {"repository_file", "generated_artifact"}:
        relative = str(record.get("path") or "")
        path = (root / relative).resolve()
        identity["path"] = relative
        try:
            path.relative_to(root.resolve())
        except ValueError:
            identity["error"] = "path escapes repository root"
            return identity
        if not path.is_file():
            identity["error"] = "file is missing"
            return identity
        actual = sha256_file(path)
        declared = record.get("sha256")
        identity["sha256"] = actual
        if declared is not None and declared != actual:
            identity["error"] = "declared sha256 does not match file"
            identity["declared_sha256"] = declared
            return identity
        identity["valid"] = True
        return identity
    if kind == "actions_artifact":
        digest = str(record.get("sha256") or "")
        identity["sha256"] = digest
        identity["valid"] = len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)
        if not identity["valid"]:
            identity["error"] = "actions artifact has no valid sha256"
        return identity
    identity["error"] = f"unsupported evidence kind {kind!r}"
    return identity


def _cell(
    dimension: dict[str, Any],
    claim: dict[str, Any],
    manifest: dict[str, Any],
    root: Path,
    evidence_records: dict[str, dict[str, Any]],
    evidence_identities: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    evaluator = dimension["evaluator"]
    evaluator_type = evaluator["type"]
    missing = dimension.get("missing_status", "not_run")
    selected: list[str] = []
    status = missing
    detail = dimension.get("missing_detail", "required evidence is not recorded")

    if evaluator_type == "manifest_validation":
        selected = [
            evidence_id
            for evidence_id, record in evidence_records.items()
            if record.get("role") in set(evaluator.get("evidence_roles", []))
        ]
        try:
            validate_manifest(manifest, root)
        except (DiscoveryValidationError, OSError, ValueError) as exc:
            status = "fail"
            detail = f"discovery manifest validation failed: {str(exc).splitlines()[0]}"
        else:
            status = "pass"
            detail = "discovery manifest satisfies its executable specification"
    elif evaluator_type == "claim_evidence_presence":
        roles = set(evaluator.get("roles", []))
        ids = set(evaluator.get("ids", []))
        selected = [
            evidence_id
            for evidence_id in claim.get("evidence", [])
            if evidence_id in evidence_records
            and (evidence_records[evidence_id].get("role") in roles or evidence_id in ids)
        ]
        if selected:
            invalid = [item for item in selected if not evidence_identities[item]["valid"]]
            if invalid:
                status = "infrastructure_blocked"
                detail = "referenced evidence is missing or has a stale identity: " + ", ".join(
                    invalid
                )
            else:
                status = "pass"
                detail = "claim references immutable evidence for this dimension"
    elif evaluator_type == "global_artifact_boolean":
        roles = set(evaluator.get("roles", []))
        selected = [
            evidence_id
            for evidence_id, record in evidence_records.items()
            if record.get("role") in roles
        ]
        if selected:
            observations: list[bool] = []
            blocked: list[str] = []
            field = str(evaluator.get("field", "passed"))
            for evidence_id in selected:
                identity = evidence_identities[evidence_id]
                record = evidence_records[evidence_id]
                if not identity["valid"] or record.get("kind") != "repository_file":
                    blocked.append(evidence_id)
                    continue
                try:
                    payload = json.loads((root / str(record["path"])).read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    blocked.append(evidence_id)
                    continue
                if not isinstance(payload.get(field), bool):
                    blocked.append(evidence_id)
                    continue
                observations.append(payload[field])
            if blocked:
                status = "infrastructure_blocked"
                detail = "cannot re-derive artifact verdict(s): " + ", ".join(blocked)
            elif observations and all(observations):
                status = "pass"
                detail = f"all {len(observations)} bound artifact verdict(s) are true"
            elif observations:
                status = "fail"
                detail = "at least one bound artifact verdict is false"
    elif evaluator_type == "novelty_state":
        selected = [
            evidence_id
            for evidence_id, record in evidence_records.items()
            if record.get("role") == "novelty_audit"
        ]
        novelty = manifest.get("novelty", {})
        state = novelty.get("status")
        if state == "pass":
            status = "pass"
            detail = "novelty freeze is current and passed"
        elif state == "fail":
            status = "fail"
            detail = "novelty audit failed"
        else:
            status = "scientifically_unresolved"
            detail = f"novelty status is {state!r}, not 'pass'"
    elif evaluator_type == "no_blockers":
        blockers = manifest.get("blockers") or []
        if blockers:
            status = "scientifically_unresolved"
            detail = "open blocker(s): " + " | ".join(str(item) for item in blockers)
        else:
            status = "pass"
            detail = "the discovery manifest records no unresolved blockers"
    else:
        raise AssuranceError(f"unknown assurance evaluator {evaluator_type!r}")

    evidence = [evidence_identities[evidence_id] for evidence_id in sorted(set(selected))]
    return {
        "dimension": dimension["id"],
        "status": status,
        "detail": detail,
        "derivation_refs": [
            "discovery_manifest",
            f"claim:{claim['id']}",
            f"dimension:{dimension['id']}",
        ],
        "evidence": evidence,
    }


def _readiness(cells: list[dict[str, Any]], policy: dict[str, Any]) -> dict[str, Any]:
    by_dimension = {cell["dimension"]: cell for cell in cells}
    output: dict[str, Any] = {}
    for profile_id, profile in policy["profiles"].items():
        required = list(profile["required_dimensions"])
        blockers = [
            {
                "dimension": dimension_id,
                "status": by_dimension[dimension_id]["status"],
                "detail": by_dimension[dimension_id]["detail"],
            }
            for dimension_id in required
            if by_dimension[dimension_id]["status"] != "pass"
        ]
        output[profile_id] = {
            "ready": not blockers,
            "required_dimensions": required,
            "blockers": blockers,
        }
    return output


def build_matrix(manifest: dict[str, Any], policy: dict[str, Any], root: Path) -> dict[str, Any]:
    validate_policy(policy)
    records = {
        str(record["id"]): record
        for record in manifest.get("evidence", [])
        if isinstance(record, dict) and record.get("id")
    }
    identities = {
        evidence_id: _evidence_identity(record, root) for evidence_id, record in records.items()
    }
    rows: list[dict[str, Any]] = []
    for claim in manifest.get("claims", []):
        if not isinstance(claim, dict) or not claim.get("id"):
            continue
        cells = [
            _cell(dimension, claim, manifest, root, records, identities)
            for dimension in policy["dimensions"]
        ]
        rows.append(
            {
                "claim_id": claim["id"],
                "claim_status": claim.get("status"),
                "cells": cells,
                "readiness": _readiness(cells, policy),
            }
        )

    release_rows = [row for row in rows if row["claim_status"] == "release_claim"]
    derivation_sources = {
        "discovery_manifest": {
            "kind": "claim_registry",
            "sha256": canonical_sha256(manifest),
        },
        **{
            f"claim:{claim['id']}": {
                "kind": "claim",
                "sha256": canonical_sha256(claim),
            }
            for claim in manifest.get("claims", [])
            if isinstance(claim, dict) and claim.get("id")
        },
        **{
            f"dimension:{dimension['id']}": {
                "kind": "assurance_policy_dimension",
                "sha256": canonical_sha256(dimension),
            }
            for dimension in policy["dimensions"]
        },
    }
    views = {
        "headline_claims": [row["claim_id"] for row in release_rows],
        "edge_node_matrix": [
            claim_id
            for claim_id in policy.get("views", {}).get("edge_node_claims", [])
            if any(row["claim_id"] == claim_id for row in rows)
        ],
        "completeness_matrix": [
            claim_id
            for claim_id in policy.get("views", {}).get("completeness_claims", [])
            if any(row["claim_id"] == claim_id for row in rows)
        ],
        "numerical_paper_readiness": {
            row["claim_id"]: row["readiness"]["numerical_paper"] for row in rows
        },
        "theorem_grade_readiness": {
            row["claim_id"]: row["readiness"]["theorem_grade"] for row in rows
        },
        "independence_common_mode": {
            row["claim_id"]: [
                cell
                for cell in row["cells"]
                if cell["dimension"]
                in {
                    "physical_formulation_independence",
                    "blind_n_version",
                    "platform_systematics_envelope",
                }
            ]
            for row in rows
        },
    }
    matrix: dict[str, Any] = {
        "schema": MATRIX_SCHEMA,
        "source_manifest_sha256": canonical_sha256(manifest),
        "policy_sha256": canonical_sha256(policy),
        "claim_count": len(rows),
        "dimension_count": len(DIMENSION_IDS),
        "status_vocabulary": sorted(STATUSES),
        "derivation_sources": derivation_sources,
        "claims": rows,
        "views": views,
        "release_policy": {
            "numerical_paper_ready": bool(release_rows)
            and all(row["readiness"]["numerical_paper"]["ready"] for row in release_rows),
            "theorem_grade_ready": bool(release_rows)
            and all(row["readiness"]["theorem_grade"]["ready"] for row in release_rows),
            "release_claim_ids": [row["claim_id"] for row in release_rows],
            "rule": "every required cell must pass; states are never averaged",
        },
    }
    matrix["matrix_sha256"] = canonical_sha256(matrix)
    return matrix


def build_weakest_link_report(matrix: dict[str, Any]) -> dict[str, Any]:
    claims: list[dict[str, Any]] = []
    for row in matrix["claims"]:
        required = set(row["readiness"]["numerical_paper"]["required_dimensions"])
        blocked = [
            cell
            for cell in row["cells"]
            if cell["dimension"] in required and cell["status"] != "pass"
        ]
        weakest = max((SEVERITY[cell["status"]] for cell in blocked), default=0)
        claims.append(
            {
                "claim_id": row["claim_id"],
                "numerical_paper_ready": row["readiness"]["numerical_paper"]["ready"],
                "theorem_grade_ready": row["readiness"]["theorem_grade"]["ready"],
                "weakest_status": next(
                    (status for status, severity in SEVERITY.items() if severity == weakest),
                    "pass",
                ),
                "weakest_dimensions": [
                    {
                        "dimension": cell["dimension"],
                        "status": cell["status"],
                        "detail": cell["detail"],
                    }
                    for cell in blocked
                    if SEVERITY[cell["status"]] == weakest
                ],
            }
        )
    return {
        "schema": REPORT_SCHEMA,
        "matrix_sha256": matrix["matrix_sha256"],
        "release_policy": matrix["release_policy"],
        "claims": claims,
        "note": "No scalar confidence is computed; weakest links preserve dimension and state.",
    }


def validate_release_assurance(manifest: dict[str, Any], matrix: dict[str, Any]) -> None:
    errors: list[str] = []
    if matrix.get("schema") != MATRIX_SCHEMA:
        errors.append("claim assurance matrix has the wrong schema")
    if matrix.get("source_manifest_sha256") != canonical_sha256(manifest):
        errors.append("claim assurance matrix is stale for the discovery manifest")
    rows = {row.get("claim_id"): row for row in matrix.get("claims", []) if isinstance(row, dict)}
    for claim in manifest.get("claims", []):
        if claim.get("status") != "release_claim":
            continue
        row = rows.get(claim.get("id"))
        if row is None:
            errors.append(f"release claim {claim.get('id')} has no assurance row")
        elif row.get("readiness", {}).get("numerical_paper", {}).get("ready") is not True:
            errors.append(f"release claim {claim.get('id')} is not numerical-paper ready")
    if matrix.get("release_policy", {}).get("numerical_paper_ready") is not True:
        errors.append("claim assurance release policy is not numerical-paper ready")
    if errors:
        raise AssuranceError("; ".join(errors))


def verify_committed_artifacts(
    root: Path,
    manifest: dict[str, Any],
    policy: dict[str, Any],
    matrix: dict[str, Any],
    report: dict[str, Any],
) -> None:
    expected_matrix = build_matrix(manifest, policy, root)
    expected_report = build_weakest_link_report(expected_matrix)
    errors: list[str] = []
    if matrix != expected_matrix:
        errors.append("claim assurance matrix is stale")
    if report != expected_report:
        errors.append("weakest-link report is stale")
    if errors:
        raise AssuranceError("; ".join(errors))
