#!/usr/bin/env python3
"""Generate deterministic fail-closed workflow and cache-poisoning audits."""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from threebody_atlas.workflow_integrity import (
    AtomicEvidenceStore,
    CacheEntry,
    CampaignPlan,
    InjectedPromotionCrash,
    PromotionFault,
    PromotionStage,
    ScientificIdentity,
    TaskResult,
    TaskStatus,
    WorkflowIntegrityError,
    load_task_result,
    reduce_campaign,
    validate_cache_entry,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "research/workflow"
FAULT_MODEL = OUTPUT_DIR / "V1_WORKFLOW_FAULT_MODEL.json"
CHAOS_MATRIX = OUTPUT_DIR / "V1_SCIENTIFIC_CHAOS_MATRIX_2026-08-16.json"
RELEASE_AUDIT = OUTPUT_DIR / "V1_FAIL_CLOSED_RELEASE_AUDIT_2026-08-16.json"
CACHE_AUDIT = OUTPUT_DIR / "V1_CACHE_POISONING_AUDIT_2026-08-16.json"


def identity(**updates: Any) -> ScientificIdentity:
    document = {
        "implementation": "atlas-chaos-reference:v1",
        "source_commit": "1" * 40,
        "spec_sha256": "2" * 64,
        "gate_manifest_sha256": "3" * 64,
        "environment_lock_sha256": "4" * 64,
        "task_parameters_sha256": "5" * 64,
        "arithmetic": "julia-bigfloat",
        "precision": "bits:200",
        "platform": "linux-x86_64",
        "input_artifacts": {"baseline": "6" * 64, "roots": "7" * 64},
    }
    document.update(updates)
    return ScientificIdentity.model_validate(document, strict=False)


def campaign() -> CampaignPlan:
    return CampaignPlan(
        identity=identity(),
        expected_task_ids=("shard:000", "shard:001", "shard:002"),
        affected_claims=("critical-graph", "root-coverage"),
    )


def task_artifact(task_id: str) -> bytes:
    return task_id.encode("utf-8")


def result(
    plan: CampaignPlan,
    task_id: str,
    *,
    attempt: str = "attempt:1",
    status: TaskStatus = TaskStatus.SUCCESS,
    payload: str | None = None,
    campaign_id: str | None = None,
) -> TaskResult:
    if payload is None and status == TaskStatus.SUCCESS:
        payload = hashlib.sha256(task_artifact(task_id)).hexdigest()
    return TaskResult(
        campaign_id=campaign_id or plan.campaign_id,
        logical_task_id=task_id,
        attempt_id=attempt,
        status=status,
        payload_sha256=payload,
        diagnostic=None if status == TaskStatus.SUCCESS else f"injected {status.value}",
    )


def successes(plan: CampaignPlan) -> list[TaskResult]:
    return [result(plan, task_id) for task_id in plan.expected_task_ids]


def fault_model() -> dict[str, Any]:
    faults = [
        ("worker-termination", "worker_terminated result", "failed task + rerun incident"),
        ("network-upload-failure", "upload_failed result", "failed task + rerun incident"),
        ("truncated-json", "strict parser truncation", "reject artifact before reduction"),
        ("stale-cache", "changed scientific identity", "cache rejection + incident"),
        ("duplicate-worker-result", "same logical task twice", "duplicate incident; no double count"),
        ("missing-shard", "omit planned logical task", "explicit missing set; release false"),
        ("out-of-order-completion", "reverse result order", "identical deterministic ledger"),
        ("retry-after-partial", "failed and successful attempts together", "conflicting duplicate incident"),
        ("corrupted-digest", "invalid SHA-256 syntax", "schema rejection"),
        ("wrong-source-sha", "cache identity source mutation", "cache rejection"),
        ("wrong-spec-hash", "cache identity spec mutation", "cache rejection"),
        ("timeout-oom-cancel", "terminal task statuses", "failed task + rerun incident"),
        ("partial-promotion", "crash before pointer replace", "old pointer remains valid"),
        ("disk-full-write-failure", "injected failure before pointer replacement", "old pointer remains valid"),
        ("post-commit-crash", "crash after pointer replace", "new pointer is complete and valid"),
        ("racing-promoter", "stale compare-and-swap token", "promotion conflict"),
        ("malicious-schema-valid-cache", "internally valid stale entry", "expected identity mismatch"),
    ]
    return {
        "schema": "atlas.workflow-fault-model.v1",
        "generated_on": "2026-08-16",
        "claim_status": "generic workflow-integrity core; legacy scientific aggregators require migration",
        "fault_count": len(faults),
        "faults": [
            {
                "id": fault_id,
                "injection": injection,
                "only_allowed_response": response,
                "false_pass_allowed": False,
            }
            for fault_id, injection, response in faults
        ],
        "promotion_stages": [stage.value for stage in PromotionStage],
        "identity_dimensions": [
            "implementation",
            "source_commit",
            "spec_sha256",
            "gate_manifest_sha256",
            "environment_lock_sha256",
            "task_parameters_sha256",
            "arithmetic",
            "precision",
            "platform",
            "input_artifacts",
        ],
    }


def _ledger_case(name: str, plan: CampaignPlan, results: list[TaskResult], expected: str) -> dict[str, Any]:
    ledger = reduce_campaign(plan, results)
    incident_kinds = sorted({incident.kind.value for incident in ledger.incidents})
    return {
        "name": name,
        "expected_incident": expected,
        "release_eligible": ledger.release_eligible,
        "completed": list(ledger.completed_task_ids),
        "failed": list(ledger.failed_task_ids),
        "missing": list(ledger.missing_task_ids),
        "duplicates": list(ledger.duplicate_task_ids),
        "unexpected": list(ledger.unexpected_task_ids),
        "incident_kinds": incident_kinds,
        "incidents": [
            {
                **incident.model_dump(mode="json"),
                "incident_id": incident.incident_id,
            }
            for incident in ledger.incidents
        ],
        "passed": not ledger.release_eligible and expected in incident_kinds,
    }


def chaos_matrix() -> dict[str, Any]:
    plan = campaign()
    baseline_results = successes(plan)
    baseline = reduce_campaign(plan, baseline_results)
    cases = []

    reversed_ledger = reduce_campaign(plan, list(reversed(baseline_results)))
    cases.append(
        {
            "name": "delayed-out-of-order-completion",
            "release_eligible": reversed_ledger.release_eligible,
            "passed": reversed_ledger == baseline,
        }
    )
    cases.append(_ledger_case("missing-shard", plan, baseline_results[:-1], "missing_task"))
    for status in (
        TaskStatus.WORKER_TERMINATED,
        TaskStatus.UPLOAD_FAILED,
        TaskStatus.TIMEOUT,
        TaskStatus.OOM,
        TaskStatus.CANCELLED,
    ):
        mutated = list(baseline_results)
        mutated[1] = result(plan, "shard:001", status=status)
        cases.append(_ledger_case(status.value, plan, mutated, "failed_task"))

    duplicate = [*baseline_results, result(plan, "shard:001", attempt="attempt:2")]
    cases.append(_ledger_case("duplicate-result", plan, duplicate, "duplicate_result"))
    conflicting = [
        *baseline_results,
        result(plan, "shard:001", attempt="attempt:2", payload="f" * 64),
    ]
    cases.append(
        _ledger_case("conflicting-retry-after-partial", plan, conflicting, "conflicting_duplicate")
    )
    cases.append(
        _ledger_case(
            "unexpected-task",
            plan,
            [*baseline_results, result(plan, "shard:999", payload="a" * 64)],
            "unexpected_task",
        )
    )
    cases.append(
        _ledger_case(
            "stale-campaign-result",
            plan,
            [
                *baseline_results,
                result(
                    plan,
                    "shard:001",
                    attempt="stale:1",
                    campaign_id="e" * 64,
                ),
            ],
            "stale_campaign",
        )
    )

    valid_text = baseline_results[0].model_dump_json()
    for name, text, expected_fragment in (
        ("truncated-json", valid_text[:-5], "invalid task result"),
        ("duplicate-json-key", '{"campaign_id":"a","campaign_id":"b"}', "duplicate"),
        ("nonstandard-json-constant", '{"payload":NaN}', "non-standard"),
    ):
        try:
            load_task_result(text)
            passed = False
            diagnostic = "artifact was silently accepted"
        except WorkflowIntegrityError as exc:
            diagnostic = str(exc)
            passed = expected_fragment in diagnostic
        cases.append({"name": name, "passed": passed, "diagnostic": diagnostic})

    try:
        result(plan, "shard:000", payload="corrupt")
        digest_passed = False
        digest_diagnostic = "invalid digest accepted"
    except ValidationError as exc:
        digest_passed = True
        digest_diagnostic = str(exc)
    cases.append(
        {
            "name": "corrupted-payload-digest",
            "passed": digest_passed,
            "diagnostic": digest_diagnostic,
        }
    )

    return {
        "schema": "atlas.scientific-chaos-matrix.v1",
        "generated_on": "2026-08-16",
        "campaign_id": plan.campaign_id,
        "baseline_release_eligible": baseline.release_eligible,
        "baseline_payload_set_sha256": baseline.payload_set_sha256,
        "case_count": len(cases),
        "passed": baseline.release_eligible and all(case["passed"] for case in cases),
        "cases": cases,
    }


def release_audit() -> dict[str, Any]:
    plan = campaign()
    complete = reduce_campaign(plan, successes(plan))
    incomplete = reduce_campaign(plan, successes(plan)[:-1])
    cases: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory() as directory:
        store = AtomicEvidenceStore(Path(directory) / "incomplete")
        try:
            store.promote(
                ledger=incomplete,
                artifact=b"partial",
                stage=PromotionStage.CANDIDATE,
                affected_claims=incomplete.affected_claims,
                expected_current_record_sha256=None,
            )
            blocked = False
        except WorkflowIntegrityError:
            blocked = store.current() is None
        cases.append({"name": "incomplete-ledger-cannot-promote", "passed": blocked})

    with tempfile.TemporaryDirectory() as directory:
        store = AtomicEvidenceStore(Path(directory) / "skip")
        try:
            store.promote(
                ledger=complete,
                artifact=b"validated",
                stage=PromotionStage.VALIDATED,
                affected_claims=complete.affected_claims,
                expected_current_record_sha256=None,
            )
            blocked = False
        except WorkflowIntegrityError:
            blocked = store.current() is None
        cases.append({"name": "promotion-stage-skip-blocked", "passed": blocked})

    for fault in PromotionFault:
        with tempfile.TemporaryDirectory() as directory:
            store = AtomicEvidenceStore(Path(directory) / "fault")
            candidate = store.promote(
                ledger=complete,
                artifact=task_artifact("shard:000"),
                stage=PromotionStage.CANDIDATE,
                affected_claims=complete.affected_claims,
                expected_current_record_sha256=None,
            )
            try:
                store.promote(
                    ledger=complete,
                    artifact=task_artifact("shard:001"),
                    stage=PromotionStage.SCREENING,
                    affected_claims=complete.affected_claims,
                    expected_current_record_sha256=candidate.record_sha256,
                    fault=fault,
                )
                raised = False
            except InjectedPromotionCrash:
                raised = True
            current = store.current()
            expected_stage = (
                PromotionStage.SCREENING
                if fault == PromotionFault.AFTER_POINTER_REPLACE
                else PromotionStage.CANDIDATE
            )
            cases.append(
                {
                    "name": f"atomic-{fault.value}",
                    "passed": raised and current is not None and current.stage == expected_stage,
                    "current_stage": current.stage.value if current else None,
                    "expected_stage": expected_stage.value,
                }
            )

    with tempfile.TemporaryDirectory() as directory:
        store = AtomicEvidenceStore(Path(directory) / "race")
        candidate = store.promote(
            ledger=complete,
            artifact=task_artifact("shard:000"),
            stage=PromotionStage.CANDIDATE,
            affected_claims=complete.affected_claims,
            expected_current_record_sha256=None,
        )
        screening = store.promote(
            ledger=complete,
            artifact=task_artifact("shard:001"),
            stage=PromotionStage.SCREENING,
            affected_claims=complete.affected_claims,
            expected_current_record_sha256=candidate.record_sha256,
        )
        try:
            store.promote(
                ledger=complete,
                artifact=task_artifact("shard:002"),
                stage=PromotionStage.SCREENING,
                affected_claims=complete.affected_claims,
                expected_current_record_sha256=candidate.record_sha256,
            )
            blocked = False
        except WorkflowIntegrityError:
            blocked = store.current() == screening
        cases.append({"name": "racing-promoter-cas", "passed": blocked})

    with tempfile.TemporaryDirectory() as directory:
        store = AtomicEvidenceStore(Path(directory) / "corruption")
        record = store.promote(
            ledger=complete,
            artifact=task_artifact("shard:000"),
            stage=PromotionStage.CANDIDATE,
            affected_claims=complete.affected_claims,
            expected_current_record_sha256=None,
        )
        (store.artifacts / f"{record.artifact_sha256}.bin").write_bytes(b"corrupted")
        try:
            store.current()
            blocked = False
        except WorkflowIntegrityError:
            blocked = True
        cases.append({"name": "promoted-artifact-corruption", "passed": blocked})

    return {
        "schema": "atlas.fail-closed-release-audit.v1",
        "generated_on": "2026-08-16",
        "campaign_id": plan.campaign_id,
        "case_count": len(cases),
        "passed": all(case["passed"] for case in cases),
        "cases": cases,
    }


def cache_audit() -> dict[str, Any]:
    expected = identity()
    mutations = [
        ("source_commit", "a" * 40),
        ("spec_sha256", "a" * 64),
        ("gate_manifest_sha256", "b" * 64),
        ("environment_lock_sha256", "c" * 64),
        ("task_parameters_sha256", "d" * 64),
        ("arithmetic", "float64"),
        ("precision", "bits:53"),
        ("platform", "macos-arm64"),
        ("implementation", "other-solver:v2"),
        ("input_artifacts", {"baseline": "f" * 64, "roots": "7" * 64}),
    ]
    cases = []
    for field, value in mutations:
        stale_identity = identity(**{field: value})
        stale = CacheEntry(
            cache_key=stale_identity.campaign_id,
            identity=stale_identity,
            payload_sha256="a" * 64,
        )
        validation = validate_cache_entry(
            expected,
            stale,
            affected_claims=("critical-graph",),
        )
        cases.append(
            {
                "name": f"changed-{field}",
                "accepted": validation.accepted,
                "mismatched_fields": list(validation.mismatched_fields),
                "incident_kind": validation.incident.kind.value if validation.incident else None,
                "passed": not validation.accepted and field in validation.mismatched_fields,
            }
        )
    return {
        "schema": "atlas.cache-poisoning-audit.v1",
        "generated_on": "2026-08-16",
        "expected_cache_key": expected.campaign_id,
        "case_count": len(cases),
        "passed": all(case["passed"] for case in cases),
        "cases": cases,
    }


def _render(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _check_or_write(path: Path, payload: dict[str, Any], check: bool) -> None:
    rendered = _render(payload)
    if check:
        if not path.is_file():
            raise SystemExit(f"missing generated artifact: {path.relative_to(ROOT)}")
        if path.read_text(encoding="utf-8") != rendered:
            raise SystemExit(f"stale generated artifact: {path.relative_to(ROOT)}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    artifacts = [fault_model(), chaos_matrix(), release_audit(), cache_audit()]
    failed = [artifact["schema"] for artifact in artifacts[1:] if not artifact["passed"]]
    if failed:
        raise SystemExit(f"workflow fault audits failed: {', '.join(failed)}")
    for path, payload in zip(
        (FAULT_MODEL, CHAOS_MATRIX, RELEASE_AUDIT, CACHE_AUDIT),
        artifacts,
        strict=True,
    ):
        _check_or_write(path, payload, args.check)
    verb = "verified" if args.check else "wrote"
    print(
        f"{verb} {len(artifacts)} workflow-integrity artifacts with "
        f"{sum(item.get('case_count', 0) for item in artifacts)} executable cases"
    )


if __name__ == "__main__":
    main()
