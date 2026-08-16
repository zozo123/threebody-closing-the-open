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
        # The flags above are re-derived from these lines, so the lines have to
        # earn them: two stable lobes, both strictly inside the m2 window, split
        # by a resolved interior unstable gap.
        "line_summaries": [
            {
                "m1": 0.997,
                "stable_intervals": [[0.994, 0.996], [0.998, 1.0]],
                "interior_unstable_gaps": [0.0019],
            }
        ],
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


def test_neck_predicate_refuses_a_raster_whose_flags_lie_about_its_own_lines() -> None:
    """The raster-level verdicts are recomputed, never taken as a self-report.

    Only the merger cross-checks declared-against-recomputed verdicts, and a
    certificate may name any neck JSON, including one the merger never saw.  A
    single stable interval that ends strictly inside the window is an *interior
    merge*, whatever the file claims about itself.
    """
    neck = _clean_neck()
    neck["line_summaries"] = [
        {"m1": 0.997, "stable_intervals": [[0.994, 0.996]], "interior_unstable_gaps": []}
    ]
    summary = neck_summary(neck)

    # The lines say interior_merge; the header still says "all separated".
    assert summary["all_lines_separated"] is False
    assert summary["any_vertical_merge"] is True
    assert summary["merge_verdict_counts"]["interior_merge"] == 1
    assert summary["clean"] is False
    assert any("all_lines_separated" in error for error in summary["verdict_errors"])
    assert any("any_vertical_merge" in error for error in summary["verdict_errors"])


def test_neck_predicate_refuses_a_raster_that_hides_a_truncated_line() -> None:
    """A lobe running off the window edge cannot be relabelled "separated"."""
    neck = _clean_neck()
    neck["line_summaries"] = [
        {"m1": 0.997, "stable_intervals": [[0.993, 0.996]], "interior_unstable_gaps": []}
    ]
    summary = neck_summary(neck)
    assert summary["any_boundary_truncated_merge_test"] is True
    assert summary["boundary_truncated_lines"] == [0.997]
    assert summary["clean"] is False
    assert any(
        "any_boundary_truncated_merge_test" in error for error in summary["verdict_errors"]
    )


def test_verification_refuses_a_certificate_frozen_over_a_lying_raster(tmp_path) -> None:
    """End to end: a self-consistent, correctly sealed certificate still fails.

    Every digest matches and the record is sealed over its own content; the only
    defect is that the neck raster's header contradicts its own line summaries.
    """
    al_path = tmp_path / "al.json"
    neck_path = tmp_path / "neck.json"
    neck = _clean_neck()
    neck["line_summaries"] = [
        {"m1": 0.997, "stable_intervals": [[0.994, 0.996]], "interior_unstable_gaps": []}
    ]
    al_path.write_text(json.dumps(_clean_al()))
    neck_path.write_text(json.dumps(neck))
    sources = [
        {"role": "active_learning", "path": str(al_path), "sha256": sha256_file(al_path)},
        {"role": "neck_scan", "path": str(neck_path), "sha256": sha256_file(neck_path)},
    ]
    # Freeze straight off the lying raster and seal it honestly.
    record = build_record(json.loads(al_path.read_text()), neck, sources)
    record["passed"] = True
    record = seal(record)
    certificate = tmp_path / "comp.json"
    certificate.write_text(json.dumps(record, indent=2))

    passed, errors = verify_certificate(record, repo_root=ROOT, certificate_path=certificate)
    assert passed is False
    assert any("contradicts its own line_summaries" in error for error in errors)


def _sealed(tmp_path: Path) -> tuple[Path, dict]:
    al_path = tmp_path / "al.json"
    neck_path = tmp_path / "neck.json"
    al_path.write_text(json.dumps(_clean_al()))
    neck_path.write_text(json.dumps(_clean_neck()))
    sources = [
        {"role": "active_learning", "path": str(al_path), "sha256": sha256_file(al_path)},
        {"role": "neck_scan", "path": str(neck_path), "sha256": sha256_file(neck_path)},
    ]
    record = seal(
        build_record(json.loads(al_path.read_text()), json.loads(neck_path.read_text()), sources)
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
