from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from threebody_atlas.workflow_integrity import (
    AtomicEvidenceStore,
    CacheEntry,
    CampaignPlan,
    IncidentKind,
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


def test_generated_workflow_fault_artifacts_are_current():
    subprocess.run(
        [sys.executable, "scripts/build_workflow_chaos_artifacts.py", "--check"],
        cwd=ROOT,
        check=True,
    )
    expected = {
        "V1_SCIENTIFIC_CHAOS_MATRIX_2026-08-16.json": 15,
        "V1_FAIL_CLOSED_RELEASE_AUDIT_2026-08-16.json": 8,
        "V1_CACHE_POISONING_AUDIT_2026-08-16.json": 10,
    }
    for name, case_count in expected.items():
        artifact = json.loads((ROOT / "research/workflow" / name).read_text(encoding="utf-8"))
        assert artifact["passed"] is True
        assert artifact["case_count"] == case_count


def identity(**updates) -> ScientificIdentity:
    document = {
        "implementation": "atlas-solver:v1",
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


def plan(**identity_updates) -> CampaignPlan:
    return CampaignPlan(
        identity=identity(**identity_updates),
        expected_task_ids=("shard:000", "shard:001", "shard:002"),
        affected_claims=("critical-graph", "root-coverage"),
    )


def result(
    campaign: CampaignPlan,
    task_id: str,
    *,
    attempt: str = "attempt:1",
    status: TaskStatus = TaskStatus.SUCCESS,
    payload: str | None = None,
    diagnostic: str | None = None,
    campaign_id: str | None = None,
) -> TaskResult:
    if payload is None and status == TaskStatus.SUCCESS:
        payload = hashlib.sha256(task_id.encode("utf-8")).hexdigest()
    if diagnostic is None and status != TaskStatus.SUCCESS:
        diagnostic = f"injected {status.value}"
    return TaskResult(
        campaign_id=campaign_id or campaign.campaign_id,
        logical_task_id=task_id,
        attempt_id=attempt,
        status=status,
        payload_sha256=payload,
        diagnostic=diagnostic,
    )


def successful_results(campaign: CampaignPlan) -> list[TaskResult]:
    return [result(campaign, task_id) for task_id in campaign.expected_task_ids]


def complete_ledger(campaign: CampaignPlan | None = None):
    campaign = campaign or plan()
    return reduce_campaign(campaign, successful_results(campaign))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_commit", "a" * 40),
        ("spec_sha256", "a" * 64),
        ("gate_manifest_sha256", "b" * 64),
        ("environment_lock_sha256", "c" * 64),
        ("task_parameters_sha256", "d" * 64),
        ("arithmetic", "float64"),
        ("precision", "bits:53"),
        ("platform", "macos-arm64"),
        ("implementation", "other-solver:v2"),
        ("input_artifacts", {"baseline": "e" * 64, "roots": "7" * 64}),
    ],
)
def test_every_scientific_identity_dimension_changes_campaign_id(field: str, value):
    baseline = identity()
    changed = identity(**{field: value})
    assert changed.campaign_id != baseline.campaign_id


def test_campaign_plan_requires_sorted_unique_tasks_and_claims():
    with pytest.raises(ValidationError, match="expected_task_ids"):
        CampaignPlan(
            identity=identity(),
            expected_task_ids=("b", "a", "a"),
            affected_claims=("claim",),
        )
    with pytest.raises(ValidationError, match="affected_claims"):
        CampaignPlan(
            identity=identity(),
            expected_task_ids=("a",),
            affected_claims=("z", "a"),
        )


def test_complete_exactly_once_campaign_is_release_eligible():
    campaign = plan()
    ledger = reduce_campaign(campaign, successful_results(campaign))
    assert ledger.release_eligible is True
    assert ledger.completed_task_ids == campaign.expected_task_ids
    assert not ledger.failed_task_ids
    assert not ledger.missing_task_ids
    assert not ledger.incidents
    assert set(ledger.payloads) == set(campaign.expected_task_ids)
    assert ledger.payload_set_sha256 is not None


def test_reducer_is_independent_of_completion_order():
    campaign = plan()
    ordered = successful_results(campaign)
    forward = reduce_campaign(campaign, ordered)
    reverse = reduce_campaign(campaign, list(reversed(ordered)))
    shuffled = reduce_campaign(campaign, [ordered[1], ordered[2], ordered[0]])
    assert forward == reverse == shuffled


def test_missing_shard_is_not_interpreted_as_empty_scientific_result():
    campaign = plan()
    ledger = reduce_campaign(campaign, successful_results(campaign)[:-1])
    assert ledger.release_eligible is False
    assert ledger.missing_task_ids == ("shard:002",)
    assert {incident.kind for incident in ledger.incidents} == {IncidentKind.MISSING_TASK}


@pytest.mark.parametrize(
    "status",
    [
        TaskStatus.FAILED,
        TaskStatus.WORKER_TERMINATED,
        TaskStatus.UPLOAD_FAILED,
        TaskStatus.TIMEOUT,
        TaskStatus.OOM,
        TaskStatus.CANCELLED,
    ],
)
def test_worker_and_infrastructure_failures_block_release(status: TaskStatus):
    campaign = plan()
    results = successful_results(campaign)
    results[1] = result(campaign, "shard:001", status=status)
    ledger = reduce_campaign(campaign, results)
    assert ledger.release_eligible is False
    assert ledger.failed_task_ids == ("shard:001",)
    assert any(status.value in incident.detail for incident in ledger.incidents)


def test_identical_duplicate_attempts_fail_closed_instead_of_double_counting():
    campaign = plan()
    results = successful_results(campaign)
    results.append(result(campaign, "shard:001", attempt="attempt:2"))
    ledger = reduce_campaign(campaign, results)
    assert ledger.release_eligible is False
    assert ledger.duplicate_task_ids == ("shard:001",)
    assert ledger.failed_task_ids == ("shard:001",)
    assert IncidentKind.DUPLICATE_RESULT in {incident.kind for incident in ledger.incidents}


def test_conflicting_duplicate_attempts_are_distinguished():
    campaign = plan()
    results = successful_results(campaign)
    results.append(
        result(
            campaign,
            "shard:001",
            attempt="attempt:2",
            payload="f" * 64,
        )
    )
    ledger = reduce_campaign(campaign, results)
    assert IncidentKind.CONFLICTING_DUPLICATE in {
        incident.kind for incident in ledger.incidents
    }
    assert ledger.release_eligible is False


def test_unexpected_task_and_stale_campaign_are_incidents():
    campaign = plan()
    results = successful_results(campaign)
    results.extend(
        [
            result(campaign, "shard:999", payload="a" * 64),
            result(
                campaign,
                "shard:001",
                attempt="stale:1",
                campaign_id="f" * 64,
            ),
        ]
    )
    ledger = reduce_campaign(campaign, results)
    kinds = {incident.kind for incident in ledger.incidents}
    assert IncidentKind.UNEXPECTED_TASK in kinds
    assert IncidentKind.STALE_CAMPAIGN in kinds
    assert ledger.release_eligible is False


def test_payload_set_identity_changes_if_any_task_payload_changes():
    campaign = plan()
    baseline = complete_ledger(campaign)
    results = successful_results(campaign)
    results[0] = result(campaign, "shard:000", payload="a" * 64)
    changed = reduce_campaign(campaign, results)
    assert changed.release_eligible
    assert changed.payload_set_sha256 != baseline.payload_set_sha256


def test_incident_ids_and_ledgers_are_deterministic():
    campaign = plan()
    first = reduce_campaign(campaign, [])
    second = reduce_campaign(campaign, [])
    assert first == second
    assert [incident.incident_id for incident in first.incidents] == [
        incident.incident_id for incident in second.incidents
    ]


def test_strict_task_result_parser_rejects_truncation_duplicates_and_constants():
    campaign = plan()
    valid = result(campaign, "shard:000").model_dump_json()
    assert load_task_result(valid).logical_task_id == "shard:000"
    with pytest.raises(WorkflowIntegrityError, match="invalid task result"):
        load_task_result(valid[:-4])
    with pytest.raises(WorkflowIntegrityError, match="duplicate"):
        load_task_result('{"campaign_id":"x","campaign_id":"y"}')
    with pytest.raises(WorkflowIntegrityError, match="non-standard"):
        load_task_result('{"payload":NaN}')


def test_task_result_digest_and_success_contract_are_strict():
    campaign = plan()
    with pytest.raises(ValidationError, match="payload_sha256"):
        TaskResult(
            campaign_id=campaign.campaign_id,
            logical_task_id="shard:000",
            attempt_id="attempt:1",
            status=TaskStatus.SUCCESS,
        )
    with pytest.raises(ValidationError, match="lowercase hexadecimal"):
        result(campaign, "shard:000", payload="broken")


def test_matching_cache_identity_is_accepted():
    expected = identity()
    entry = CacheEntry(
        cache_key=expected.campaign_id,
        identity=expected,
        payload_sha256="a" * 64,
    )
    validation = validate_cache_entry(
        expected,
        entry,
        affected_claims=("critical-graph",),
    )
    assert validation.accepted is True
    assert not validation.mismatched_fields
    assert validation.incident is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_commit", "a" * 40),
        ("spec_sha256", "a" * 64),
        ("gate_manifest_sha256", "b" * 64),
        ("environment_lock_sha256", "c" * 64),
        ("task_parameters_sha256", "d" * 64),
        ("precision", "bits:53"),
        ("platform", "macos-arm64"),
        ("input_artifacts", {"baseline": "f" * 64, "roots": "7" * 64}),
    ],
)
def test_cache_poisoning_under_scientific_identity_changes_is_rejected(field: str, value):
    expected = identity()
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
    assert validation.accepted is False
    assert field in validation.mismatched_fields
    assert validation.incident is not None
    assert validation.incident.kind == IncidentKind.CACHE_IDENTITY_MISMATCH


def test_schema_valid_stale_cache_cannot_impersonate_expected_key():
    expected = identity()
    stale_identity = identity(source_commit="a" * 40)
    with pytest.raises(ValidationError, match="cache_key does not match"):
        CacheEntry(
            cache_key=expected.campaign_id,
            identity=stale_identity,
            payload_sha256="a" * 64,
        )


def test_atomic_promotion_advances_one_stage_and_preserves_chain(tmp_path: Path):
    ledger = complete_ledger()
    store = AtomicEvidenceStore(tmp_path / "store")
    candidate = store.promote(
        ledger=ledger,
        artifact=b"candidate evidence",
        stage=PromotionStage.CANDIDATE,
        affected_claims=ledger.affected_claims,
        expected_current_record_sha256=None,
    )
    screening = store.promote(
        ledger=ledger,
        artifact=b"screening evidence",
        stage=PromotionStage.SCREENING,
        affected_claims=ledger.affected_claims,
        expected_current_record_sha256=candidate.record_sha256,
    )
    assert screening.sequence == 2
    assert screening.previous_record_sha256 == candidate.record_sha256
    assert store.current() == screening


def test_incomplete_campaign_cannot_promote(tmp_path: Path):
    campaign = plan()
    incomplete = reduce_campaign(campaign, successful_results(campaign)[:-1])
    store = AtomicEvidenceStore(tmp_path / "store")
    with pytest.raises(WorkflowIntegrityError, match="incomplete campaign"):
        store.promote(
            ledger=incomplete,
            artifact=b"partial",
            stage=PromotionStage.CANDIDATE,
            affected_claims=incomplete.affected_claims,
            expected_current_record_sha256=None,
        )
    assert store.current() is None


def test_promotion_cannot_skip_stages_or_change_claim_scope(tmp_path: Path):
    ledger = complete_ledger()
    store = AtomicEvidenceStore(tmp_path / "store")
    with pytest.raises(WorkflowIntegrityError, match="invalid promotion transition"):
        store.promote(
            ledger=ledger,
            artifact=b"independent",
            stage=PromotionStage.INDEPENDENT,
            affected_claims=ledger.affected_claims,
            expected_current_record_sha256=None,
        )
    with pytest.raises(WorkflowIntegrityError, match="claims must exactly match"):
        store.promote(
            ledger=ledger,
            artifact=b"candidate",
            stage=PromotionStage.CANDIDATE,
            affected_claims=("other-claim",),
            expected_current_record_sha256=None,
        )


def test_promotion_compare_and_swap_rejects_racing_writer(tmp_path: Path):
    ledger = complete_ledger()
    store = AtomicEvidenceStore(tmp_path / "store")
    candidate = store.promote(
        ledger=ledger,
        artifact=b"candidate",
        stage=PromotionStage.CANDIDATE,
        affected_claims=ledger.affected_claims,
        expected_current_record_sha256=None,
    )
    screening = store.promote(
        ledger=ledger,
        artifact=b"screening-a",
        stage=PromotionStage.SCREENING,
        affected_claims=ledger.affected_claims,
        expected_current_record_sha256=candidate.record_sha256,
    )
    with pytest.raises(WorkflowIntegrityError, match="compare-and-swap conflict"):
        store.promote(
            ledger=ledger,
            artifact=b"screening-b",
            stage=PromotionStage.SCREENING,
            affected_claims=ledger.affected_claims,
            expected_current_record_sha256=candidate.record_sha256,
        )
    assert store.current() == screening
    incident_files = list(store.incidents.glob("*.json"))
    assert len(incident_files) == 1
    incident = json.loads(incident_files[0].read_text(encoding="utf-8"))
    assert incident["kind"] == "promotion_conflict"
    assert incident["affected_claims"] == list(ledger.affected_claims)
    assert incident["rerun_required"] is True


@pytest.mark.parametrize(
    "fault",
    [
        PromotionFault.AFTER_ARTIFACT,
        PromotionFault.AFTER_RECORD,
        PromotionFault.BEFORE_POINTER_REPLACE,
    ],
)
def test_crash_before_pointer_replace_leaves_old_state_valid(
    tmp_path: Path,
    fault: PromotionFault,
):
    ledger = complete_ledger()
    store = AtomicEvidenceStore(tmp_path / fault.value)
    candidate = store.promote(
        ledger=ledger,
        artifact=b"candidate",
        stage=PromotionStage.CANDIDATE,
        affected_claims=ledger.affected_claims,
        expected_current_record_sha256=None,
    )
    with pytest.raises(InjectedPromotionCrash, match=fault.value):
        store.promote(
            ledger=ledger,
            artifact=b"screening",
            stage=PromotionStage.SCREENING,
            affected_claims=ledger.affected_claims,
            expected_current_record_sha256=candidate.record_sha256,
            fault=fault,
        )
    assert store.current() == candidate


def test_crash_after_pointer_replace_leaves_new_state_valid(tmp_path: Path):
    ledger = complete_ledger()
    store = AtomicEvidenceStore(tmp_path / "store")
    with pytest.raises(InjectedPromotionCrash, match="after_pointer_replace"):
        store.promote(
            ledger=ledger,
            artifact=b"candidate",
            stage=PromotionStage.CANDIDATE,
            affected_claims=ledger.affected_claims,
            expected_current_record_sha256=None,
            fault=PromotionFault.AFTER_POINTER_REPLACE,
        )
    current = store.current()
    assert current is not None
    assert current.stage == PromotionStage.CANDIDATE


def test_promoted_artifact_corruption_is_detected(tmp_path: Path):
    ledger = complete_ledger()
    store = AtomicEvidenceStore(tmp_path / "store")
    record = store.promote(
        ledger=ledger,
        artifact=b"candidate",
        stage=PromotionStage.CANDIDATE,
        affected_claims=ledger.affected_claims,
        expected_current_record_sha256=None,
    )
    artifact = store.artifacts / f"{record.artifact_sha256}.bin"
    artifact.write_bytes(b"corrupted")
    with pytest.raises(WorkflowIntegrityError, match="missing or corrupt"):
        store.current()


def test_promotion_pointer_to_missing_record_is_detected(tmp_path: Path):
    store = AtomicEvidenceStore(tmp_path / "store")
    store.root.mkdir(parents=True)
    store.pointer_path.write_text(
        json.dumps(
            {
                "schema_version": "atlas.evidence-promotion-pointer.v1",
                "sequence": 1,
                "record_sha256": "a" * 64,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(WorkflowIntegrityError, match="missing record"):
        store.current()


def test_missing_promotion_history_record_is_detected(tmp_path: Path):
    ledger = complete_ledger()
    store = AtomicEvidenceStore(tmp_path / "store")
    candidate = store.promote(
        ledger=ledger,
        artifact=b"candidate",
        stage=PromotionStage.CANDIDATE,
        affected_claims=ledger.affected_claims,
        expected_current_record_sha256=None,
    )
    store.promote(
        ledger=ledger,
        artifact=b"screening",
        stage=PromotionStage.SCREENING,
        affected_claims=ledger.affected_claims,
        expected_current_record_sha256=candidate.record_sha256,
    )
    (store.records / f"{candidate.record_sha256}.json").unlink()
    with pytest.raises(WorkflowIntegrityError, match="missing record"):
        store.current()
