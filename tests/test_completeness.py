from __future__ import annotations

import json
from pathlib import Path

from threebody_atlas.completeness import (
    MINIMUM_AL_ATTEMPTS,
    SCHEMA,
    SHOOTING_RESIDUAL_GATE,
    al_summary,
    build_record,
    neck_summary,
    seal,
    sha256_file,
    verify_certificate,
)
from threebody_atlas.evidence_semantics import (
    build_certificate_semantics,
    criterion_claims,
    load_registry,
)


ROOT = Path(__file__).resolve().parents[1]


def _clean_al(count: int = MINIMUM_AL_ATTEMPTS) -> dict:
    rows = [
        {
            "shooting_success": True,
            "shooting_residual": 1e-12,
            "corrected": {"screening_stable": False},
        }
        for _ in range(count)
    ]
    return {"attempted": rows, "accepted_candidates": rows}


def _clean_neck() -> dict:
    return {
        "completed": True,
        "grid": {"m1": [0.997, 0.999], "m2": [0.993, 1.006], "step": 0.0001, "samples": 12},
        "minimum_resolved_unstable_gap": 0.0002,
        "any_vertical_merge": False,
        # A raster is only "clean" if it also answered the merge question inside
        # its own window: no line's verdict may be limited by scan-window
        # truncation, no line may lack a stable sample, and every line must
        # carry a separation witness.
        "any_boundary_truncated_merge_test": False,
        "any_line_without_stable_sample": False,
        "any_stable_interval_touches_boundary": False,
        "all_lines_separated": True,
        "merge_verdict_counts": {
            "separated": 1,
            "interior_merge": 0,
            "truncation_undecidable": 0,
            "no_stable_sample": 0,
        },
        "boundary_truncated_lines": [],
        "max_shooting_residual": 1e-9,
        "line_summaries": [{"m1": 0.997, "stable_intervals": [[0.994, 0.996]]}],
    }


def test_frozen_gates_are_not_loosened() -> None:
    assert SHOOTING_RESIDUAL_GATE == 1e-7
    assert MINIMUM_AL_ATTEMPTS == 12


def test_al_predicate_rejects_a_residual_above_the_frozen_gate() -> None:
    al = _clean_al()
    assert al_summary(al)["clean"] is True
    al["accepted_candidates"][0]["shooting_residual"] = 2e-7
    assert al_summary(al)["clean"] is False


def test_al_predicate_rejects_a_screening_stable_pocket() -> None:
    al = _clean_al()
    al["accepted_candidates"][0]["corrected"]["screening_stable"] = True
    summary = al_summary(al)
    assert summary["screening_stable"] == 1
    assert summary["clean"] is False


def test_neck_predicate_needs_a_fully_resolved_grid_step() -> None:
    neck = _clean_neck()
    assert neck_summary(neck)["clean"] is True
    neck["minimum_resolved_unstable_gap"] = neck["grid"]["step"] / 2
    assert neck_summary(neck)["clean"] is False


def _sealed(tmp_path: Path) -> tuple[Path, dict]:
    al_path = tmp_path / "al.json"
    neck_path = tmp_path / "neck.json"
    al_path.write_text(json.dumps(_clean_al()))
    neck_path.write_text(json.dumps(_clean_neck()))
    sources = [
        {
            "role": "active_learning",
            "path": str(al_path),
            "sha256": sha256_file(al_path),
            "criterion_id": "active_learning_pocket/v1",
            "semantic_origin": "legacy_role",
        },
        {
            "role": "neck_scan",
            "path": str(neck_path),
            "sha256": sha256_file(neck_path),
            "criterion_id": "local_neck_raster/v1",
            "semantic_origin": "legacy_role",
        },
    ]
    registry = load_registry(ROOT)
    record = seal(
        build_record(
            json.loads(al_path.read_text()),
            json.loads(neck_path.read_text()),
            sources,
            search_semantics=build_certificate_semantics(sources, registry),
        )
    )
    certificate = tmp_path / "comp.json"
    certificate.write_text(json.dumps(record, indent=2))
    return certificate, record


def test_sealed_record_verifies_against_its_sources(tmp_path) -> None:
    certificate, record = _sealed(tmp_path)
    assert record["schema"] == SCHEMA
    passed, errors = verify_certificate(record, repo_root=ROOT, certificate_path=certificate)
    assert (passed, errors) == (True, [])


def test_verification_fails_when_a_recorded_number_is_inflated(tmp_path) -> None:
    certificate, record = _sealed(tmp_path)
    record["active_learning"]["attempted"] = 999
    record = seal(record)
    passed, errors = verify_certificate(record, repo_root=ROOT, certificate_path=certificate)
    assert passed is False
    assert any("active_learning.attempted" in error for error in errors)


def test_verification_refuses_an_absolute_source_outside_the_allowed_roots(tmp_path) -> None:
    certificate, record = _sealed(tmp_path)
    outside = tmp_path.parent / "stray_al.json"
    outside.write_text(json.dumps(_clean_al()))
    for row in record["sources"]:
        if row["role"] == "active_learning":
            row["path"] = str(outside)
            row["sha256"] = sha256_file(outside)
    record = seal(record)
    passed, errors = verify_certificate(record, repo_root=ROOT, certificate_path=certificate)
    assert passed is False
    assert any("outside the repository" in error for error in errors)


def test_bounded_certificate_cannot_satisfy_full_critical_set_release(tmp_path) -> None:
    certificate, record = _sealed(tmp_path)
    from threebody_atlas.completeness import verification_report

    report = verification_report(record, repo_root=ROOT, certificate_path=certificate)
    assert report["passed"] is True
    assert report["release_scope_passed"] is False
    assert set(report["release_scope_errors"]) == {
        "claim enumerates_full_critical_set must be True, got False",
        "claim excludes_even_root_pairs must be True, got False",
        "claim excludes_tangencies must be True, got False",
        "claim bounded_resolution_only must be False, got True",
    }


def test_relabeling_a_parent_criterion_invalidates_a_resealed_certificate(tmp_path) -> None:
    certificate, record = _sealed(tmp_path)
    source = next(row for row in record["sources"] if row["role"] == "active_learning")
    source["criterion_id"] = "published_label_brackets/v1"
    record["search_semantics"]["parent_criteria"]["active_learning"] = source[
        "criterion_id"
    ]
    record = seal(record)
    passed, errors = verify_certificate(record, repo_root=ROOT, certificate_path=certificate)
    assert passed is False
    assert any("derived 'active_learning_pocket/v1'" in error for error in errors)


def test_claim_scope_cannot_be_strengthened_by_resealing(tmp_path) -> None:
    certificate, record = _sealed(tmp_path)
    record["search_semantics"]["claim_scope"] = criterion_claims(
        load_registry(ROOT), "validated_full_event_cover/v1"
    )
    record = seal(record)
    passed, errors = verify_certificate(record, repo_root=ROOT, certificate_path=certificate)
    assert passed is False
    assert "certificate claim_scope disagrees with the search-scope registry" in errors
