from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from threebody_atlas.assurance import (
    DIMENSION_IDS,
    MATRIX_SCHEMA,
    STATUSES,
    AssuranceError,
    build_matrix,
    build_weakest_link_report,
    validate_policy,
    validate_release_assurance,
    verify_committed_artifacts,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "research/DISCOVERY_RELEASE.json"
POLICY_PATH = ROOT / "research/ASSURANCE_DIMENSIONS.json"
MATRIX_PATH = ROOT / "research/evidence/V1_CLAIM_ASSURANCE_MATRIX.json"
REPORT_PATH = ROOT / "research/evidence/V1_WEAKEST_LINK_REPORT.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _cells(matrix: dict) -> dict[tuple[str, str], dict]:
    return {
        (row["claim_id"], cell["dimension"]): cell
        for row in matrix["claims"]
        for cell in row["cells"]
    }


def test_committed_assurance_artifacts_are_exactly_reproducible() -> None:
    manifest = _load(MANIFEST_PATH)
    policy = _load(POLICY_PATH)
    matrix = _load(MATRIX_PATH)
    report = _load(REPORT_PATH)
    verify_committed_artifacts(ROOT, manifest, policy, matrix, report)
    assert matrix["schema"] == MATRIX_SCHEMA
    assert matrix["claim_count"] == len(manifest["claims"]) == 7
    assert matrix["dimension_count"] == len(DIMENSION_IDS) == 14
    assert report == build_weakest_link_report(matrix)


def test_every_claim_has_every_dimension_and_typed_evidence_identities() -> None:
    matrix = _load(MATRIX_PATH)
    sources = matrix["derivation_sources"]
    for row in matrix["claims"]:
        assert tuple(cell["dimension"] for cell in row["cells"]) == DIMENSION_IDS
        assert all(cell["status"] in STATUSES for cell in row["cells"])
        for cell in row["cells"]:
            assert len(cell["derivation_refs"]) == 3
            assert all(reference in sources for reference in cell["derivation_refs"])
            assert all(
                len(sources[reference]["sha256"]) == 64 for reference in cell["derivation_refs"]
            )
            for identity in cell["evidence"]:
                assert identity["valid"] is True
                assert len(identity["sha256"]) == 64


def test_no_scalar_confidence_or_score_is_emitted() -> None:
    matrix = _load(MATRIX_PATH)
    report = _load(REPORT_PATH)

    def keys(value):
        if isinstance(value, dict):
            for key, child in value.items():
                yield key
                yield from keys(child)
        elif isinstance(value, list):
            for child in value:
                yield from keys(child)

    forbidden = {"score", "confidence", "probability", "aggregate_confidence"}
    assert forbidden.isdisjoint(keys(matrix))
    assert forbidden.isdisjoint(keys(report))
    assert isinstance(matrix["release_policy"]["numerical_paper_ready"], bool)
    assert isinstance(matrix["release_policy"]["theorem_grade_ready"], bool)


def test_current_open_campaign_is_explicitly_not_ready() -> None:
    matrix = _load(MATRIX_PATH)
    assert matrix["release_policy"]["numerical_paper_ready"] is False
    assert matrix["release_policy"]["theorem_grade_ready"] is False


def test_independence_dimensions_use_distinct_evidence_selectors() -> None:
    policy = _load(POLICY_PATH)
    evaluators = {
        item["id"]: item["evaluator"]
        for item in policy["dimensions"]
        if item["id"] in {"physical_formulation_independence", "blind_n_version"}
    }
    physical_sources = set(evaluators["physical_formulation_independence"]["roles"])
    physical_sources.update(evaluators["physical_formulation_independence"].get("ids", []))
    blind_sources = set(evaluators["blind_n_version"]["roles"])
    blind_sources.update(evaluators["blind_n_version"].get("ids", []))
    assert physical_sources.isdisjoint(blind_sources)


def test_malformed_assurance_policies_fail_closed() -> None:
    policy = _load(POLICY_PATH)
    mutations = []

    wrong_schema = copy.deepcopy(policy)
    wrong_schema["schema"] = "wrong"
    mutations.append(wrong_schema)

    reordered_dimensions = copy.deepcopy(policy)
    reordered_dimensions["dimensions"].reverse()
    mutations.append(reordered_dimensions)

    reordered_profiles = copy.deepcopy(policy)
    reordered_profiles["profiles"] = {
        "theorem_grade": reordered_profiles["profiles"]["theorem_grade"],
        "numerical_paper": reordered_profiles["profiles"]["numerical_paper"],
    }
    mutations.append(reordered_profiles)

    invalid_missing_status = copy.deepcopy(policy)
    invalid_missing_status["dimensions"][0]["missing_status"] = "greenish"
    mutations.append(invalid_missing_status)

    for mutation in mutations:
        with pytest.raises(AssuranceError):
            validate_policy(mutation)

    unknown_evaluator = copy.deepcopy(policy)
    unknown_evaluator["dimensions"][0]["evaluator"]["type"] = "unknown"
    with pytest.raises(AssuranceError, match="unknown assurance evaluator"):
        build_matrix(
            {
                "claims": [{"id": "synthetic", "status": "candidate"}],
                "evidence": [],
            },
            unknown_evaluator,
            ROOT,
        )


@pytest.mark.parametrize(
    ("relative", "expected_error"),
    [("../outside.json", "escapes"), ("missing.json", "missing")],
)
def test_evidence_paths_must_stay_inside_the_repository_and_exist(
    tmp_path: Path, relative: str, expected_error: str
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (tmp_path / "outside.json").write_text("{}\n", encoding="utf-8")
    policy = _load(POLICY_PATH)
    manifest = {
        "claims": [
            {
                "id": "synthetic",
                "status": "candidate",
                "statement": "fixture",
                "method": "fixture",
                "evidence": ["edge"],
            }
        ],
        "evidence": [
            {
                "id": "edge",
                "kind": "repository_file",
                "role": "critical_graph",
                "path": relative,
                "description": "fixture",
            }
        ],
        "blockers": [],
        "novelty": {"status": "pending"},
    }
    cell = _cells(build_matrix(manifest, policy, root))[
        ("synthetic", "direct_continuation_topology")
    ]
    assert cell["status"] == "infrastructure_blocked"
    assert expected_error in cell["evidence"][0]["error"]


def test_uppercase_actions_digest_is_normalized_before_validation(tmp_path: Path) -> None:
    policy = _load(POLICY_PATH)
    manifest = {
        "claims": [
            {
                "id": "synthetic",
                "status": "candidate",
                "statement": "fixture",
                "method": "fixture",
                "evidence": ["edge"],
            }
        ],
        "evidence": [
            {
                "id": "edge",
                "kind": "actions_artifact",
                "role": "critical_graph",
                "sha256": "A" * 64,
                "description": "fixture",
            }
        ],
        "blockers": [],
        "novelty": {"status": "pending"},
    }
    cell = _cells(build_matrix(manifest, policy, tmp_path))[
        ("synthetic", "direct_continuation_topology")
    ]
    assert cell["status"] == "pass"
    assert cell["evidence"][0]["sha256"] == "a" * 64


def test_novelty_cannot_pass_without_valid_audit_evidence() -> None:
    policy = _load(POLICY_PATH)
    manifest = _load(MANIFEST_PATH)
    manifest["novelty"]["status"] = "pass"
    manifest["evidence"] = [
        item for item in manifest["evidence"] if item["role"] != "novelty_audit"
    ]
    absent = build_matrix(manifest, policy, ROOT)
    assert all(
        _cells(absent)[(row["claim_id"], "literature_novelty")]["status"] == "not_run"
        for row in absent["claims"]
    )

    stale = _load(MANIFEST_PATH)
    stale["novelty"]["status"] = "pass"
    record = next(item for item in stale["evidence"] if item["role"] == "novelty_audit")
    record["path"] = "research/DOES_NOT_EXIST.md"
    stale_matrix = build_matrix(stale, policy, ROOT)
    assert all(
        _cells(stale_matrix)[(row["claim_id"], "literature_novelty")]["status"]
        == "infrastructure_blocked"
        for row in stale_matrix["claims"]
    )


def test_evidence_parent_mutation_stales_the_matrix_without_manual_edits() -> None:
    manifest = _load(MANIFEST_PATH)
    policy = _load(POLICY_PATH)
    committed_matrix = _load(MATRIX_PATH)
    committed_report = _load(REPORT_PATH)
    changed = copy.deepcopy(manifest)
    parent = next(item for item in changed["evidence"] if item["id"] == "family-bridge-artifact")
    parent["sha256"] = "f" * 64

    rebuilt = build_matrix(changed, policy, ROOT)
    before = _cells(committed_matrix)
    after = _cells(rebuilt)
    assert (
        before[("one-continuation-family", "direct_continuation_topology")]["evidence"]
        != after[("one-continuation-family", "direct_continuation_topology")]["evidence"]
    )
    assert all(before[key] == after[key] for key in before if key[0] != "one-continuation-family")
    with pytest.raises(AssuranceError, match="stale"):
        verify_committed_artifacts(
            ROOT,
            changed,
            policy,
            committed_matrix,
            committed_report,
        )


def test_novelty_downgrade_changes_only_the_literature_cells() -> None:
    manifest = _load(MANIFEST_PATH)
    policy = _load(POLICY_PATH)
    before = _cells(build_matrix(manifest, policy, ROOT))
    changed = copy.deepcopy(manifest)
    changed["novelty"]["status"] = "fail"
    after_matrix = build_matrix(changed, policy, ROOT)
    after = _cells(after_matrix)
    changed_dimensions = {
        dimension for key, cell in before.items() if cell != after[key] for dimension in [key[1]]
    }
    assert changed_dimensions == {"literature_novelty"}
    assert all(
        after[(row["claim_id"], "literature_novelty")]["status"] == "fail"
        for row in after_matrix["claims"]
    )


def test_unresolved_box_downgrades_only_the_contradiction_cells() -> None:
    manifest = _load(MANIFEST_PATH)
    policy = _load(POLICY_PATH)
    clear = copy.deepcopy(manifest)
    clear["blockers"] = []
    before = _cells(build_matrix(clear, policy, ROOT))
    clear["blockers"] = ["new unresolved box"]
    after_matrix = build_matrix(clear, policy, ROOT)
    after = _cells(after_matrix)
    changed_dimensions = {key[1] for key, cell in before.items() if cell != after[key]}
    assert changed_dimensions == {"unresolved_contradictions"}
    assert all(
        after[(row["claim_id"], "unresolved_contradictions")]["status"]
        == "scientifically_unresolved"
        for row in after_matrix["claims"]
    )


@pytest.mark.parametrize(
    ("role", "dimension"),
    [
        ("platform_systematics", "platform_systematics_envelope"),
        ("calibration_audit", "truth_known_calibration"),
    ],
)
def test_artifact_verdict_downgrade_is_derived_from_bound_bytes(
    tmp_path: Path, role: str, dimension: str
) -> None:
    policy = _load(POLICY_PATH)
    artifact = tmp_path / "platform.json"
    artifact.write_text('{"passed":true}\n', encoding="utf-8")
    manifest = {
        "claims": [
            {
                "id": "synthetic",
                "status": "candidate",
                "statement": "fixture",
                "method": "fixture",
                "evidence": [],
            }
        ],
        "evidence": [
            {
                "id": "platform",
                "kind": "repository_file",
                "role": role,
                "path": "platform.json",
                "description": "fixture",
            }
        ],
        "blockers": [],
        "novelty": {"status": "pending"},
    }
    passed = _cells(build_matrix(manifest, policy, tmp_path))[("synthetic", dimension)]
    artifact.write_text('{"passed":false}\n', encoding="utf-8")
    failed = _cells(build_matrix(manifest, policy, tmp_path))[("synthetic", dimension)]
    assert passed["status"] == "pass"
    assert failed["status"] == "fail"
    assert passed["evidence"][0]["sha256"] != failed["evidence"][0]["sha256"]


def test_release_gate_consumes_numerical_readiness_not_theorem_readiness() -> None:
    manifest = _load(MANIFEST_PATH)
    matrix = _load(MATRIX_PATH)
    with pytest.raises(AssuranceError, match="not numerical-paper ready"):
        validate_release_assurance(manifest, matrix)

    ready = copy.deepcopy(matrix)
    for row in ready["claims"]:
        if row["claim_status"] == "release_claim":
            row["readiness"]["numerical_paper"]["ready"] = True
            row["readiness"]["numerical_paper"]["blockers"] = []
    ready["release_policy"]["numerical_paper_ready"] = True
    ready["release_policy"]["theorem_grade_ready"] = False
    validate_release_assurance(manifest, ready)
