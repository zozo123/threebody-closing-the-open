"""Completeness-certificate contract shared by the freezer and the assembler.

A completeness certificate is a *claim* that the catalog sheet has no hidden
stability pocket.  Sealing that claim with a digest computed over the record
itself proves only that the record is internally consistent: any two-key dict
plus a self-consistent digest used to satisfy the old check, so anyone could
mint a passing certificate.

This module makes the certificate verifiable instead of self-sealed:

* the freezer builds the record from the AL pocket screen and the neck raster
  and records a sha256 for every source file it read;
* the assembler re-reads those source files, re-hashes them, and *re-derives*
  the substantive predicates.  A certificate whose sources are missing, whose
  source bytes changed, or whose recorded numbers disagree with the artifacts
  is rejected even when it is perfectly self-consistent and re-sealed.

Nothing here loosens a numerical gate: the shooting-residual gate stays at
1e-7 and the neck raster still has to resolve at least one full grid step.
"""
from __future__ import annotations

import hashlib
import json
import string
from pathlib import Path
from typing import Any


SCHEMA = "atlas.v1.completeness-certificate/2"
"""Schema 2 differs from 1 only in what a *verifier* must do: re-hash and
re-derive every source.  The version is bumped so that self-sealed schema-1
records can never be replayed against the strict verifier."""

REQUIRED_SOURCE_ROLES = ("active_learning", "neck_scan")
SHOOTING_RESIDUAL_GATE = 1e-7
MINIMUM_AL_ATTEMPTS = 12
GAP_SLACK = 1e-12


def content_digest(record: dict[str, Any]) -> str:
    """sha256 over the canonical record with ``sha256_content`` removed."""
    canonical_record = {key: value for key, value in record.items() if key != "sha256_content"}
    canonical = json.dumps(canonical_record, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def seal(record: dict[str, Any]) -> dict[str, Any]:
    record["sha256_content"] = content_digest(record)
    return record


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_hex_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in string.hexdigits for character in value)
    )


def al_summary(al: dict[str, Any] | None) -> dict[str, Any]:
    """Re-derive the active-learning pocket-screen predicate."""
    if not isinstance(al, dict):
        return {"attempted": 0, "accepted": 0, "screening_stable": 0, "clean": False}
    accepted = al.get("accepted_candidates") or []
    attempted = al.get("attempted", accepted) or []
    stable = [
        row
        for row in accepted
        if isinstance(row, dict) and (row.get("corrected") or {}).get("screening_stable")
    ]
    clean = bool(
        len(attempted) >= MINIMUM_AL_ATTEMPTS
        and len(accepted) == len(attempted)
        and not stable
        and all(
            isinstance(row, dict)
            and row.get("shooting_success") is True
            and float(row.get("shooting_residual", float("inf"))) <= SHOOTING_RESIDUAL_GATE
            for row in accepted
        )
    )
    return {
        "attempted": len(attempted),
        "accepted": len(accepted),
        "screening_stable": len(stable),
        "clean": clean,
    }


def neck_summary(neck: dict[str, Any] | None) -> dict[str, Any]:
    """Re-derive the stability-neck raster predicate."""
    grid = neck.get("grid", {}) if isinstance(neck, dict) else {}
    grid = grid if isinstance(grid, dict) else {}
    step = grid.get("step")
    gap = neck.get("minimum_resolved_unstable_gap") if isinstance(neck, dict) else None
    done = bool(
        isinstance(neck, dict)
        and neck.get("completed") is True
        and grid
        and grid.get("samples")
        and neck.get("line_summaries")
        and float(neck.get("max_shooting_residual", float("inf"))) <= SHOOTING_RESIDUAL_GATE
    )
    clean = bool(
        done
        and neck.get("any_vertical_merge") is False
        and gap is not None
        and step is not None
        and float(gap) + GAP_SLACK >= float(step)
    )
    return {
        "grid": grid or None,
        "step": step,
        "samples": grid.get("samples"),
        "minimum_resolved_unstable_gap": gap,
        "any_vertical_merge": neck.get("any_vertical_merge") if isinstance(neck, dict) else None,
        "max_shooting_residual": neck.get("max_shooting_residual") if isinstance(neck, dict) else None,
        "completed": neck.get("completed") if isinstance(neck, dict) else None,
        "done": done,
        "clean": clean,
    }


def build_record(
    al: dict[str, Any] | None,
    neck: dict[str, Any] | None,
    sources: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the unsealed certificate body from the two artifacts.

    The assembler re-derives every number in here from the same helpers, so the
    freezer and the verifier cannot drift apart.
    """
    al_stats = al_summary(al)
    neck_stats = neck_summary(neck)
    passed = bool(al_stats["clean"] and neck_stats["clean"])
    return {
        "schema": SCHEMA,
        "passed": passed,
        "domain": {
            "catalog": "Li-Li-Liao unequal-mass non-hierarchical published grid",
            "neck": neck_stats["grid"],
        },
        "resolution": neck_stats["step"],
        "active_learning": {
            "attempted": al_stats["attempted"],
            "accepted": al_stats["accepted"],
            "screening_stable_hidden_pockets": al_stats["screening_stable"],
            "interpretation": (
                "off-grid proposals corrected onto the known sheet; no hidden stable pocket in this sample"
                if al_stats["clean"]
                else "inspect accepted stable points before freezing completeness"
            ),
        },
        "neck": None
        if not isinstance(neck, dict)
        else {
            "samples": neck_stats["samples"],
            "minimum_resolved_unstable_gap": neck_stats["minimum_resolved_unstable_gap"],
            "any_vertical_merge": neck_stats["any_vertical_merge"],
            "max_shooting_residual": neck_stats["max_shooting_residual"],
            "completed": neck_stats["completed"],
            "topology_clean": neck_stats["clean"],
        },
        "note": (
            "Bounded completeness: no additional stability pocket found in the AL sample "
            "and the neck raster completed at the declared local resolution."
            if passed
            else (
                "Completeness not frozen: require a completed, closure-gated neck raster with no "
                "vertical merge and at least one resolved unstable-grid step, plus a clean AL pocket screen."
            )
        ),
        "sources": sources,
    }


def _contained(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def resolve_source_path(
    raw: Any, allowed_roots: list[Path]
) -> tuple[Path | None, str | None]:
    """Resolve a recorded source path inside one of the allowed roots.

    Relative paths are resolved against each allowed root in order (repository
    root first).  Absolute paths are accepted only when they land inside an
    allowed root.  ``..`` segments are refused outright, and the *resolved*
    real path must still be contained, so a symlink cannot point outside.
    """
    text = str(raw or "").strip()
    if not text:
        return None, "source entry has an empty path"
    candidate_path = Path(text)
    if any(part == ".." for part in candidate_path.parts):
        return None, f"source path {text!r} must not contain '..'"
    roots = [root.resolve() for root in allowed_roots]
    candidates = [candidate_path] if candidate_path.is_absolute() else [
        root / candidate_path for root in roots
    ]
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if not resolved.is_file():
            continue
        if not any(_contained(resolved, root) for root in roots):
            return None, (
                f"source path {text!r} resolves outside the repository and the certificate directory"
            )
        return resolved, None
    return None, f"source file {text!r} was not found under the allowed roots"


def _mismatch(label: str, recorded: Any, derived: Any) -> str:
    return f"certificate {label} says {recorded!r} but the artifact gives {derived!r}"


def _equal(recorded: Any, derived: Any) -> bool:
    if isinstance(recorded, bool) or isinstance(derived, bool):
        return recorded is derived
    if isinstance(recorded, (int, float)) and isinstance(derived, (int, float)):
        return float(recorded) == float(derived)
    return recorded == derived


def verify_certificate(
    record: dict[str, Any] | None,
    *,
    repo_root: Path,
    certificate_path: Path | None = None,
) -> tuple[bool, list[str]]:
    """Verify a sealed certificate against the artifacts it names.

    Returns ``(passed, errors)``.  ``passed`` is true only when the record is
    well-formed, self-consistent, references both required roles, every named
    source file still hashes to the recorded digest, and the substantive AL and
    neck predicates re-derived from those files reproduce the recorded numbers.
    """
    errors: list[str] = []
    if not isinstance(record, dict):
        return False, ["completeness certificate is missing"]

    if record.get("schema") != SCHEMA:
        errors.append(
            f"certificate schema must be {SCHEMA!r}, got {record.get('schema')!r}"
        )
    if record.get("passed") is not True:
        errors.append("certificate does not claim passed=true")
    digest = record.get("sha256_content")
    if not is_hex_digest(digest):
        errors.append("certificate sha256_content must be a 64-character hex digest")
    elif str(digest).lower() != content_digest(record):
        errors.append("certificate sha256_content does not match its own content")
    if errors:
        return False, errors

    sources = record.get("sources")
    if not isinstance(sources, list) or not sources:
        return False, ["certificate carries no sources to verify"]
    by_role: dict[str, list[dict[str, Any]]] = {}
    for entry in sources:
        if not isinstance(entry, dict):
            errors.append("every certificate source must be an object")
            continue
        by_role.setdefault(str(entry.get("role") or ""), []).append(entry)
    for role in REQUIRED_SOURCE_ROLES:
        found = by_role.get(role) or []
        if len(found) != 1:
            errors.append(
                f"certificate needs exactly one {role!r} source, found {len(found)}"
            )
    if errors:
        return False, errors

    allowed_roots = [repo_root]
    if certificate_path is not None:
        allowed_roots.append(Path(certificate_path).resolve().parent)

    payloads: dict[str, dict[str, Any]] = {}
    for entry in sources:
        role = str(entry.get("role") or "")
        recorded_digest = entry.get("sha256")
        if not is_hex_digest(recorded_digest):
            errors.append(f"source {role or '<no role>'} needs a 64-character hex sha256")
            continue
        resolved, problem = resolve_source_path(entry.get("path"), allowed_roots)
        if resolved is None:
            errors.append(f"source {role or '<no role>'}: {problem}")
            continue
        actual = sha256_file(resolved)
        if actual != str(recorded_digest).lower():
            errors.append(
                f"source {role or '<no role>'} {entry.get('path')!r} sha256 mismatch: "
                f"certificate={recorded_digest} file={actual}"
            )
            continue
        try:
            payload = json.loads(resolved.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            errors.append(f"source {role or '<no role>'} is not readable JSON: {exc}")
            continue
        if role in REQUIRED_SOURCE_ROLES:
            payloads[role] = payload
    if errors:
        return False, errors

    al_stats = al_summary(payloads.get("active_learning"))
    neck_stats = neck_summary(payloads.get("neck_scan"))
    if not al_stats["clean"]:
        errors.append("re-derived AL pocket screen does not support a completeness claim")
    if not neck_stats["clean"]:
        errors.append("re-derived neck raster does not support a completeness claim")

    recorded_al = record.get("active_learning")
    if not isinstance(recorded_al, dict):
        errors.append("certificate is missing its active_learning block")
    else:
        for label, recorded_key, derived_key in (
            ("active_learning.attempted", "attempted", "attempted"),
            ("active_learning.accepted", "accepted", "accepted"),
            (
                "active_learning.screening_stable_hidden_pockets",
                "screening_stable_hidden_pockets",
                "screening_stable",
            ),
        ):
            if not _equal(recorded_al.get(recorded_key), al_stats[derived_key]):
                errors.append(
                    _mismatch(label, recorded_al.get(recorded_key), al_stats[derived_key])
                )

    recorded_neck = record.get("neck")
    if not isinstance(recorded_neck, dict):
        errors.append("certificate is missing its neck block")
    else:
        for label, recorded_key, derived_key in (
            ("neck.samples", "samples", "samples"),
            (
                "neck.minimum_resolved_unstable_gap",
                "minimum_resolved_unstable_gap",
                "minimum_resolved_unstable_gap",
            ),
            ("neck.any_vertical_merge", "any_vertical_merge", "any_vertical_merge"),
            ("neck.max_shooting_residual", "max_shooting_residual", "max_shooting_residual"),
            ("neck.completed", "completed", "completed"),
            ("neck.topology_clean", "topology_clean", "clean"),
        ):
            if not _equal(recorded_neck.get(recorded_key), neck_stats[derived_key]):
                errors.append(
                    _mismatch(label, recorded_neck.get(recorded_key), neck_stats[derived_key])
                )

    if not _equal(record.get("resolution"), neck_stats["step"]):
        errors.append(_mismatch("resolution", record.get("resolution"), neck_stats["step"]))
    recorded_domain = record.get("domain")
    recorded_domain_neck = recorded_domain.get("neck") if isinstance(recorded_domain, dict) else None
    if recorded_domain_neck != neck_stats["grid"]:
        errors.append(_mismatch("domain.neck", recorded_domain_neck, neck_stats["grid"]))

    if errors:
        return False, errors
    return True, []


def verification_report(
    record: dict[str, Any] | None,
    *,
    repo_root: Path,
    certificate_path: Path | None = None,
) -> dict[str, Any]:
    """Verify and describe a certificate, for embedding in the graph artifact."""
    passed, errors = verify_certificate(
        record, repo_root=repo_root, certificate_path=certificate_path
    )
    sources_in_repository = None
    if isinstance(record, dict) and isinstance(record.get("sources"), list):
        allowed_roots = [repo_root]
        if certificate_path is not None:
            allowed_roots.append(Path(certificate_path).resolve().parent)
        located = []
        for entry in record["sources"]:
            if not isinstance(entry, dict):
                continue
            resolved, _problem = resolve_source_path(entry.get("path"), allowed_roots)
            located.append(
                resolved is not None and _contained(resolved, repo_root.resolve())
            )
        sources_in_repository = bool(located) and all(located)
    return {
        "schema": SCHEMA,
        "passed": passed,
        "errors": errors,
        "sources_in_repository": sources_in_repository,
        "certificate_path": None if certificate_path is None else str(certificate_path),
    }
