"""Fail-closed accounting for distributed scientific campaigns.

This module is intentionally independent of any one numerical solver.  It supplies
the safety boundary around sharded execution: content-addressed campaign identity,
exactly-once task accounting, cache validation, incident records, and atomic evidence
promotion.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import tempfile
from collections import defaultdict
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

IDENTITY_SCHEMA = "atlas.scientific-identity.v1"
PLAN_SCHEMA = "atlas.campaign-plan.v1"
RESULT_SCHEMA = "atlas.campaign-task-result.v1"
LEDGER_SCHEMA = "atlas.campaign-ledger.v1"
CACHE_SCHEMA = "atlas.scientific-cache-entry.v1"
INCIDENT_SCHEMA = "atlas.workflow-incident.v1"
PROMOTION_SCHEMA = "atlas.evidence-promotion-record.v1"
POINTER_SCHEMA = "atlas.evidence-promotion-pointer.v1"

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GIT_SHA = re.compile(r"[0-9a-f]{40}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/+@=-]*\Z")


class WorkflowIntegrityError(RuntimeError):
    """Raised when distributed workflow state cannot be trusted."""


class InjectedPromotionCrash(WorkflowIntegrityError):
    """Test-only exception raised at an explicit atomic-promotion fault point."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


def _canonical_bytes(value: BaseModel | dict[str, Any]) -> bytes:
    document = (
        value.model_dump(mode="json", exclude_none=True)
        if isinstance(value, BaseModel)
        else value
    )
    return json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: BaseModel | dict[str, Any] | bytes) -> str:
    payload = value if isinstance(value, bytes) else _canonical_bytes(value)
    return hashlib.sha256(payload).hexdigest()


def _validate_sha256(value: str, label: str) -> None:
    if not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be 64 lowercase hexadecimal digits")


def _validate_identifier(value: str, label: str) -> None:
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{label} has invalid syntax: {value!r}")


def _payload_set_digest(payloads: dict[str, str]) -> str | None:
    if not payloads:
        return None
    return _sha256({key: payloads[key] for key in sorted(payloads)})


class ScientificIdentity(_StrictModel):
    schema_version: Literal[IDENTITY_SCHEMA] = IDENTITY_SCHEMA
    implementation: str
    source_commit: str
    spec_sha256: str
    gate_manifest_sha256: str
    environment_lock_sha256: str
    task_parameters_sha256: str
    arithmetic: str
    precision: str
    platform: str
    input_artifacts: dict[str, str]

    @model_validator(mode="after")
    def validate_identity(self) -> ScientificIdentity:
        for label, value in (
            ("implementation", self.implementation),
            ("arithmetic", self.arithmetic),
            ("precision", self.precision),
            ("platform", self.platform),
        ):
            _validate_identifier(value, label)
        if not _GIT_SHA.fullmatch(self.source_commit):
            raise ValueError("source_commit must be an exact 40-character Git commit SHA")
        for label, digest in (
            ("spec_sha256", self.spec_sha256),
            ("gate_manifest_sha256", self.gate_manifest_sha256),
            ("environment_lock_sha256", self.environment_lock_sha256),
            ("task_parameters_sha256", self.task_parameters_sha256),
        ):
            _validate_sha256(digest, label)
        if not self.input_artifacts:
            raise ValueError("scientific identity must include at least one input artifact")
        for name, digest in self.input_artifacts.items():
            _validate_identifier(name, "input artifact name")
            _validate_sha256(digest, f"input artifact {name!r}")
        return self

    @property
    def campaign_id(self) -> str:
        return _sha256(self)


class CampaignPlan(_StrictModel):
    schema_version: Literal[PLAN_SCHEMA] = PLAN_SCHEMA
    identity: ScientificIdentity
    expected_task_ids: tuple[str, ...]
    affected_claims: tuple[str, ...]

    @model_validator(mode="after")
    def validate_plan(self) -> CampaignPlan:
        if not self.expected_task_ids:
            raise ValueError("campaign plan must contain at least one logical task")
        if tuple(sorted(set(self.expected_task_ids))) != self.expected_task_ids:
            raise ValueError("expected_task_ids must be unique and sorted")
        if not self.affected_claims:
            raise ValueError("campaign plan must name at least one affected claim")
        if tuple(sorted(set(self.affected_claims))) != self.affected_claims:
            raise ValueError("affected_claims must be unique and sorted")
        for task_id in self.expected_task_ids:
            _validate_identifier(task_id, "logical task id")
        for claim in self.affected_claims:
            _validate_identifier(claim, "affected claim")
        return self

    @property
    def campaign_id(self) -> str:
        return self.identity.campaign_id


class TaskStatus(StrEnum):
    SUCCESS = "success"
    FAILED = "failed"
    WORKER_TERMINATED = "worker_terminated"
    UPLOAD_FAILED = "upload_failed"
    TIMEOUT = "timeout"
    OOM = "oom"
    CANCELLED = "cancelled"


class TaskResult(_StrictModel):
    schema_version: Literal[RESULT_SCHEMA] = RESULT_SCHEMA
    campaign_id: str
    logical_task_id: str
    attempt_id: str
    status: TaskStatus
    payload_sha256: str | None = None
    diagnostic: str | None = None

    @model_validator(mode="after")
    def validate_result(self) -> TaskResult:
        _validate_sha256(self.campaign_id, "campaign_id")
        _validate_identifier(self.logical_task_id, "logical_task_id")
        _validate_identifier(self.attempt_id, "attempt_id")
        if self.status == TaskStatus.SUCCESS and self.payload_sha256 is None:
            raise ValueError("successful task result must carry payload_sha256")
        if self.payload_sha256 is not None:
            _validate_sha256(self.payload_sha256, "payload_sha256")
        if self.status != TaskStatus.SUCCESS and not self.diagnostic:
            raise ValueError("non-success task result must carry a diagnostic")
        return self


class IncidentKind(StrEnum):
    MISSING_TASK = "missing_task"
    FAILED_TASK = "failed_task"
    DUPLICATE_RESULT = "duplicate_result"
    CONFLICTING_DUPLICATE = "conflicting_duplicate"
    UNEXPECTED_TASK = "unexpected_task"
    STALE_CAMPAIGN = "stale_campaign"
    CORRUPT_RESULT = "corrupt_result"
    CACHE_IDENTITY_MISMATCH = "cache_identity_mismatch"
    PROMOTION_CONFLICT = "promotion_conflict"


class WorkflowIncident(_StrictModel):
    schema_version: Literal[INCIDENT_SCHEMA] = INCIDENT_SCHEMA
    kind: IncidentKind
    campaign_id: str
    affected_claims: tuple[str, ...]
    task_ids: tuple[str, ...] = ()
    rerun_required: bool
    detail: str

    @model_validator(mode="after")
    def validate_incident(self) -> WorkflowIncident:
        _validate_sha256(self.campaign_id, "incident campaign_id")
        if tuple(sorted(set(self.affected_claims))) != self.affected_claims:
            raise ValueError("incident affected_claims must be unique and sorted")
        if tuple(sorted(set(self.task_ids))) != self.task_ids:
            raise ValueError("incident task_ids must be unique and sorted")
        return self

    @property
    def incident_id(self) -> str:
        return _sha256(self)


class CampaignLedger(_StrictModel):
    schema_version: Literal[LEDGER_SCHEMA] = LEDGER_SCHEMA
    campaign_id: str
    affected_claims: tuple[str, ...]
    expected_task_ids: tuple[str, ...]
    completed_task_ids: tuple[str, ...]
    failed_task_ids: tuple[str, ...]
    missing_task_ids: tuple[str, ...]
    unexpected_task_ids: tuple[str, ...]
    duplicate_task_ids: tuple[str, ...]
    payloads: dict[str, str]
    payload_set_sha256: str | None
    release_eligible: bool
    incidents: tuple[WorkflowIncident, ...]

    @model_validator(mode="after")
    def validate_ledger(self) -> CampaignLedger:
        _validate_sha256(self.campaign_id, "ledger campaign_id")
        for task_id, digest in self.payloads.items():
            _validate_identifier(task_id, "payload task id")
            _validate_sha256(digest, f"payload {task_id!r}")
        computed_payload_set = _payload_set_digest(self.payloads)
        if self.payload_set_sha256 != computed_payload_set:
            raise ValueError("payload_set_sha256 does not match the payload map")
        for name in (
            "affected_claims",
            "expected_task_ids",
            "completed_task_ids",
            "failed_task_ids",
            "missing_task_ids",
            "unexpected_task_ids",
            "duplicate_task_ids",
        ):
            values = getattr(self, name)
            if tuple(sorted(set(values))) != values:
                raise ValueError(f"{name} must be unique and sorted")
        expected = set(self.expected_task_ids)
        completed = set(self.completed_task_ids)
        failed = set(self.failed_task_ids)
        missing = set(self.missing_task_ids)
        if completed | failed | missing != expected or any(
            first & second
            for first, second in (
                (completed, failed),
                (completed, missing),
                (failed, missing),
            )
        ):
            raise ValueError("completed/failed/missing task sets must partition expected tasks")
        if set(self.payloads) != completed:
            raise ValueError("payloads must exactly match completed_task_ids")
        should_release = (
            completed == expected
            and not self.failed_task_ids
            and not self.missing_task_ids
            and not self.unexpected_task_ids
            and not self.duplicate_task_ids
            and not self.incidents
            and set(self.payloads) == expected
        )
        if self.release_eligible != should_release:
            raise ValueError("release_eligible disagrees with exact accounting")
        return self

    def allowed_promotion_digests(self) -> frozenset[str]:
        allowed = set(self.payloads.values())
        if self.payload_set_sha256 is not None:
            allowed.add(self.payload_set_sha256)
        return frozenset(allowed)

    def payload_set_document_bytes(self) -> bytes:
        return _canonical_bytes({key: self.payloads[key] for key in sorted(self.payloads)})


def _incident(
    plan: CampaignPlan,
    kind: IncidentKind,
    detail: str,
    task_ids: tuple[str, ...] = (),
) -> WorkflowIncident:
    return WorkflowIncident(
        kind=kind,
        campaign_id=plan.campaign_id,
        affected_claims=plan.affected_claims,
        task_ids=tuple(sorted(set(task_ids))),
        rerun_required=True,
        detail=detail,
    )


def reduce_campaign(plan: CampaignPlan, results: list[TaskResult]) -> CampaignLedger:
    """Reduce results with exact task-set accounting and order-independent output."""

    expected = set(plan.expected_task_ids)
    grouped: dict[str, list[TaskResult]] = defaultdict(list)
    unexpected: set[str] = set()
    incidents: list[WorkflowIncident] = []

    for result in results:
        if result.campaign_id != plan.campaign_id:
            incidents.append(
                _incident(
                    plan,
                    IncidentKind.STALE_CAMPAIGN,
                    f"attempt {result.attempt_id} carries campaign {result.campaign_id}",
                    (result.logical_task_id,),
                )
            )
            unexpected.add(result.logical_task_id)
            continue
        if result.logical_task_id not in expected:
            incidents.append(
                _incident(
                    plan,
                    IncidentKind.UNEXPECTED_TASK,
                    f"attempt {result.attempt_id} is not in the frozen task plan",
                    (result.logical_task_id,),
                )
            )
            unexpected.add(result.logical_task_id)
            continue
        grouped[result.logical_task_id].append(result)

    completed: set[str] = set()
    failed: set[str] = set()
    missing: set[str] = set()
    duplicates: set[str] = set()
    payloads: dict[str, str] = {}

    for task_id in plan.expected_task_ids:
        task_results = sorted(grouped.get(task_id, []), key=lambda item: item.attempt_id)
        if not task_results:
            missing.add(task_id)
            incidents.append(
                _incident(
                    plan,
                    IncidentKind.MISSING_TASK,
                    "expected logical task produced no result",
                    (task_id,),
                )
            )
            continue
        if len(task_results) > 1:
            duplicates.add(task_id)
            identities = {(item.status, item.payload_sha256) for item in task_results}
            kind = (
                IncidentKind.CONFLICTING_DUPLICATE
                if len(identities) > 1
                else IncidentKind.DUPLICATE_RESULT
            )
            incidents.append(
                _incident(
                    plan,
                    kind,
                    f"logical task has {len(task_results)} attempts in reducer input",
                    (task_id,),
                )
            )
            failed.add(task_id)
            continue
        result = task_results[0]
        if result.status == TaskStatus.SUCCESS:
            completed.add(task_id)
            assert result.payload_sha256 is not None
            payloads[task_id] = result.payload_sha256
        else:
            failed.add(task_id)
            incidents.append(
                _incident(
                    plan,
                    IncidentKind.FAILED_TASK,
                    f"task ended with {result.status.value}: {result.diagnostic}",
                    (task_id,),
                )
            )

    incidents = sorted(
        incidents,
        key=lambda item: (item.kind.value, item.task_ids, item.detail, item.incident_id),
    )
    release_eligible = not incidents and completed == expected
    payload_set_sha256 = _payload_set_digest(payloads)
    return CampaignLedger(
        campaign_id=plan.campaign_id,
        affected_claims=plan.affected_claims,
        expected_task_ids=plan.expected_task_ids,
        completed_task_ids=tuple(sorted(completed)),
        failed_task_ids=tuple(sorted(failed)),
        missing_task_ids=tuple(sorted(missing)),
        unexpected_task_ids=tuple(sorted(unexpected)),
        duplicate_task_ids=tuple(sorted(duplicates)),
        payloads={key: payloads[key] for key in sorted(payloads)},
        payload_set_sha256=payload_set_sha256,
        release_eligible=release_eligible,
        incidents=tuple(incidents),
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise WorkflowIntegrityError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def load_task_result(text: str) -> TaskResult:
    """Strictly parse a task result; truncated/duplicate/nonstandard JSON fails."""

    try:
        document = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                WorkflowIntegrityError(f"non-standard JSON constant {value!r}")
            ),
        )
        return TaskResult.model_validate(document, strict=False)
    except WorkflowIntegrityError:
        raise
    except (json.JSONDecodeError, ValueError) as exc:
        raise WorkflowIntegrityError(f"invalid task result: {exc}") from exc


class CacheEntry(_StrictModel):
    schema_version: Literal[CACHE_SCHEMA] = CACHE_SCHEMA
    cache_key: str
    identity: ScientificIdentity
    payload_sha256: str

    @model_validator(mode="after")
    def validate_entry(self) -> CacheEntry:
        _validate_sha256(self.cache_key, "cache_key")
        _validate_sha256(self.payload_sha256, "cache payload_sha256")
        if self.cache_key != self.identity.campaign_id:
            raise ValueError("cache_key does not match embedded scientific identity")
        return self


class CacheValidation(_StrictModel):
    accepted: bool
    expected_cache_key: str
    observed_cache_key: str
    mismatched_fields: tuple[str, ...]
    incident: WorkflowIncident | None = None


def validate_cache_entry(
    expected: ScientificIdentity,
    entry: CacheEntry,
    *,
    affected_claims: tuple[str, ...],
) -> CacheValidation:
    expected_document = expected.model_dump(mode="json")
    observed_document = entry.identity.model_dump(mode="json")
    mismatches = tuple(
        sorted(
            key
            for key in set(expected_document) | set(observed_document)
            if expected_document.get(key) != observed_document.get(key)
        )
    )
    accepted = not mismatches and entry.cache_key == expected.campaign_id
    incident = None
    if not accepted:
        incident = WorkflowIncident(
            kind=IncidentKind.CACHE_IDENTITY_MISMATCH,
            campaign_id=expected.campaign_id,
            affected_claims=affected_claims,
            rerun_required=True,
            detail=f"cache identity mismatch in: {', '.join(mismatches) or 'cache_key'}",
        )
    return CacheValidation(
        accepted=accepted,
        expected_cache_key=expected.campaign_id,
        observed_cache_key=entry.cache_key,
        mismatched_fields=mismatches,
        incident=incident,
    )


class PromotionStage(StrEnum):
    CANDIDATE = "candidate"
    SCREENING = "screening"
    CORRECTION = "correction"
    INDEPENDENT = "independent"
    VALIDATED = "validated"


_STAGES = tuple(PromotionStage)


class PromotionRecord(_StrictModel):
    schema_version: Literal[PROMOTION_SCHEMA] = PROMOTION_SCHEMA
    sequence: int = Field(ge=1)
    stage: PromotionStage
    campaign_id: str
    campaign_payload_set_sha256: str
    artifact_sha256: str
    previous_record_sha256: str | None = None
    affected_claims: tuple[str, ...]

    @model_validator(mode="after")
    def validate_record(self) -> PromotionRecord:
        for label, digest in (
            ("promotion campaign_id", self.campaign_id),
            ("campaign_payload_set_sha256", self.campaign_payload_set_sha256),
            ("promotion artifact_sha256", self.artifact_sha256),
        ):
            _validate_sha256(digest, label)
        if self.previous_record_sha256 is not None:
            _validate_sha256(self.previous_record_sha256, "previous_record_sha256")
        if tuple(sorted(set(self.affected_claims))) != self.affected_claims:
            raise ValueError("promotion affected_claims must be unique and sorted")
        return self

    @property
    def record_sha256(self) -> str:
        return _sha256(self)


class PromotionPointer(_StrictModel):
    schema_version: Literal[POINTER_SCHEMA] = POINTER_SCHEMA
    sequence: int = Field(ge=1)
    record_sha256: str

    @model_validator(mode="after")
    def validate_pointer(self) -> PromotionPointer:
        _validate_sha256(self.record_sha256, "pointer record_sha256")
        return self


class PromotionFault(StrEnum):
    AFTER_ARTIFACT = "after_artifact"
    AFTER_RECORD = "after_record"
    BEFORE_POINTER_REPLACE = "before_pointer_replace"
    AFTER_POINTER_REPLACE = "after_pointer_replace"


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, payload: bytes, *, replace: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if replace:
            os.replace(temporary, path)
            _fsync_directory(path.parent)
        return temporary
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


class AtomicEvidenceStore:
    """Immutable blobs/records with one atomically replaced current pointer."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.artifacts = self.root / "artifacts"
        self.records = self.root / "records"
        self.incidents = self.root / "incidents"
        self.pointer_path = self.root / "CURRENT.json"
        self.lock_path = self.root / ".promotion.lock"

    def _raise_promotion_conflict(self, ledger: CampaignLedger, detail: str) -> None:
        incident = WorkflowIncident(
            kind=IncidentKind.PROMOTION_CONFLICT,
            campaign_id=ledger.campaign_id,
            affected_claims=ledger.affected_claims,
            rerun_required=True,
            detail=detail,
        )
        self._write_immutable(
            self.incidents / f"{incident.incident_id}.json",
            _canonical_bytes(incident) + b"\n",
        )
        raise WorkflowIntegrityError(f"{detail}; incident={incident.incident_id}")

    def _write_immutable(self, path: Path, payload: bytes) -> None:
        if path.exists():
            if path.read_bytes() != payload:
                raise WorkflowIntegrityError(f"immutable object collision at {path}")
            return
        temporary = _atomic_write(path, payload, replace=False)
        try:
            try:
                os.link(temporary, path)
            except FileExistsError:
                if path.read_bytes() != payload:
                    raise WorkflowIntegrityError(f"immutable object race at {path}") from None
            _fsync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)

    def _read_pointer(self) -> PromotionPointer | None:
        if not self.pointer_path.exists():
            return None
        try:
            return PromotionPointer.model_validate(
                json.loads(self.pointer_path.read_text(encoding="utf-8")),
                strict=False,
            )
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise WorkflowIntegrityError(f"invalid promotion pointer: {exc}") from exc

    def _read_record(self, digest: str) -> PromotionRecord:
        record_path = self.records / f"{digest}.json"
        if not record_path.is_file():
            raise WorkflowIntegrityError("promotion history references a missing record")
        raw = record_path.read_bytes()
        try:
            record = PromotionRecord.model_validate(json.loads(raw), strict=False)
        except (json.JSONDecodeError, ValueError) as exc:
            raise WorkflowIntegrityError(f"invalid promotion record: {exc}") from exc
        if record.record_sha256 != digest:
            raise WorkflowIntegrityError("promotion record digest does not match pointer")
        artifact_path = self.artifacts / f"{record.artifact_sha256}.bin"
        if not artifact_path.is_file() or hashlib.sha256(artifact_path.read_bytes()).hexdigest() != record.artifact_sha256:
            raise WorkflowIntegrityError("promoted artifact is missing or corrupt")
        return record

    def current(self) -> PromotionRecord | None:
        pointer = self._read_pointer()
        if pointer is None:
            return None
        head = self._read_record(pointer.record_sha256)
        if head.sequence != pointer.sequence:
            raise WorkflowIntegrityError("promotion record sequence does not match pointer")
        cursor = head
        seen = {cursor.record_sha256}
        while cursor.previous_record_sha256 is not None:
            previous = self._read_record(cursor.previous_record_sha256)
            if previous.record_sha256 in seen:
                raise WorkflowIntegrityError("promotion history contains a cycle")
            if previous.sequence != cursor.sequence - 1:
                raise WorkflowIntegrityError("promotion history sequence is discontinuous")
            if _STAGES.index(previous.stage) + 1 != _STAGES.index(cursor.stage):
                raise WorkflowIntegrityError("promotion history stage sequence is discontinuous")
            seen.add(previous.record_sha256)
            cursor = previous
        if cursor.sequence != 1 or cursor.stage != PromotionStage.CANDIDATE:
            raise WorkflowIntegrityError("promotion history has no valid candidate root")
        return head

    def promote(
        self,
        *,
        ledger: CampaignLedger,
        artifact: bytes,
        stage: PromotionStage | str,
        affected_claims: tuple[str, ...],
        expected_current_record_sha256: str | None,
        fault: PromotionFault | str | None = None,
    ) -> PromotionRecord:
        self.root.mkdir(parents=True, exist_ok=True)
        if not ledger.release_eligible or ledger.payload_set_sha256 is None:
            self._raise_promotion_conflict(
                ledger,
                "incomplete campaign ledger cannot promote evidence",
            )
        if tuple(sorted(set(affected_claims))) != ledger.affected_claims:
            self._raise_promotion_conflict(
                ledger,
                "promotion claims must exactly match the campaign ledger's affected claims",
            )
        stage = PromotionStage(stage)
        fault = PromotionFault(fault) if fault is not None else None
        with self.lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                current = self.current()
            except WorkflowIntegrityError as exc:
                self._raise_promotion_conflict(
                    ledger,
                    f"promotion store validation failed: {exc}",
                )
            current_sha = current.record_sha256 if current is not None else None
            if current_sha != expected_current_record_sha256:
                self._raise_promotion_conflict(
                    ledger,
                    f"promotion compare-and-swap conflict: expected "
                    f"{expected_current_record_sha256}, observed {current_sha}",
                )
            expected_stage = _STAGES[0] if current is None else (
                _STAGES[_STAGES.index(current.stage) + 1]
                if current.stage != _STAGES[-1]
                else None
            )
            if stage != expected_stage:
                self._raise_promotion_conflict(
                    ledger,
                    f"invalid promotion transition: current={current.stage.value if current else None} "
                    f"requested={stage.value} expected={expected_stage.value if expected_stage else None}",
                )

            artifact_sha = hashlib.sha256(artifact).hexdigest()
            if artifact_sha not in ledger.allowed_promotion_digests():
                self._raise_promotion_conflict(
                    ledger,
                    "promoted artifact is not a ledger payload or the committed payload-set digest",
                )
            self._write_immutable(self.artifacts / f"{artifact_sha}.bin", artifact)
            if fault == PromotionFault.AFTER_ARTIFACT:
                raise InjectedPromotionCrash(fault.value)

            record = PromotionRecord(
                sequence=1 if current is None else current.sequence + 1,
                stage=stage,
                campaign_id=ledger.campaign_id,
                campaign_payload_set_sha256=ledger.payload_set_sha256,
                artifact_sha256=artifact_sha,
                previous_record_sha256=current_sha,
                affected_claims=tuple(sorted(set(affected_claims))),
            )
            record_bytes = _canonical_bytes(record) + b"\n"
            self._write_immutable(
                self.records / f"{record.record_sha256}.json",
                record_bytes,
            )
            if fault == PromotionFault.AFTER_RECORD:
                raise InjectedPromotionCrash(fault.value)

            pointer = PromotionPointer(
                sequence=record.sequence,
                record_sha256=record.record_sha256,
            )
            pointer_bytes = _canonical_bytes(pointer) + b"\n"
            if fault == PromotionFault.BEFORE_POINTER_REPLACE:
                _atomic_write(self.pointer_path, pointer_bytes, replace=False)
                raise InjectedPromotionCrash(fault.value)
            _atomic_write(self.pointer_path, pointer_bytes)
            if fault == PromotionFault.AFTER_POINTER_REPLACE:
                raise InjectedPromotionCrash(fault.value)
            return record
