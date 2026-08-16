from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _run(script: str, args: list[str]) -> int:
    argv = sys.argv
    sys.argv = [script, *args]
    try:
        try:
            runpy.run_path(str(ROOT / "scripts" / script), run_name="__main__")
            return 0
        except SystemExit as exc:
            return int(exc.code or 0)
    finally:
        sys.argv = argv


def test_secondary_fold_selector_uses_adjacent_opposite_orientations() -> None:
    namespace = runpy.run_path(str(ROOT / "scripts/trace_secondary_minus_fold_geometry.py"))
    select = namespace["select_secondary_pair"]
    roots = [
        ("U->S", {"_cell_id": "392"}, "lower-secondary"),
        ("S->U", {"_cell_id": "393"}, "upper-secondary"),
        ("U->S", {"_cell_id": "394"}, "principal"),
    ]
    lower, upper = select(roots)
    assert lower[1]["_cell_id"] == "392"
    assert upper[1]["_cell_id"] == "393"


def test_left_birth_fold_screen_is_projection_fold(tmp_path) -> None:
    src = tmp_path / "geo.json"
    src.write_text(
        json.dumps(
            {
                "passed": True,
                "generic_m1_fold_screen": True,
                "opposite_branch_reconnection_screen": True,
                "localized_seeds": [{}, {}],
            }
        )
    )
    out = tmp_path / "left.json"
    assert _run("classify_secondary_left_birth.py", [str(src), str(out)]) == 0
    record = json.loads(out.read_text())
    assert record["class"] == "projection_fold"
    assert record["passed"] is False
    assert record["evidence_level"] == "screening"
    assert record["requires_independent_verification"] is True


def test_left_birth_failed_fold_is_two_arcs(tmp_path) -> None:
    src = tmp_path / "geo.json"
    src.write_text(
        json.dumps(
            {
                "passed": False,
                "generic_m1_fold_screen": False,
                "opposite_branch_reconnection_screen": False,
                "localized_seeds": [{}, {}],
            }
        )
    )
    out = tmp_path / "left.json"
    assert _run("classify_secondary_left_birth.py", [str(src), str(out)]) == 0
    record = json.loads(out.read_text())
    assert record["class"] == "two_separate_arcs"
    assert record["passed"] is False


def test_left_birth_requires_geometry_and_bigfloat_for_pass(tmp_path) -> None:
    geometry = tmp_path / "geometry.json"
    geometry.write_text(
        json.dumps(
            {
                "passed": True,
                "generic_m1_fold_screen": True,
                "opposite_branch_reconnection_screen": True,
                "edge_endpoint_bindings": [
                    {"cell_id": 392, "side": "start", "mechanism": "minus_one"},
                    {"cell_id": 393, "side": "start", "mechanism": "minus_one"},
                ],
            }
        )
    )
    bigfloat = tmp_path / "fold.json"
    bigfloat.write_text(
        json.dumps(
            {
                "passed": True,
                "implementation": "independent test fixture",
                "masses": ["0.9957", "0.9742", "1.0"],
                "closure_norm": "1e-18",
                "minus_one_event": "1e-12",
                "stationarity_stencil_audit": [
                    {"dGdm2": "1e-8", "dGdm1": "2", "d2Gdm22": "20"}
                ],
                "branch_curvature_audit": {
                    "relative_curvature_disagreement": "0.04",
                    "branches": [
                        {
                            "cell_id": 392,
                            "orientation": "U->S",
                            "source_m2_bracket": ["0.964", "0.965"],
                            "masses": ["0.996", "0.9645", "1.0"],
                            "closure_norm": "1e-18",
                            "minus_one_event": "1e-12",
                            "dm1_from_fold": "0.0003",
                            "dm2_from_fold": "-0.0097",
                            "secant_m1_curvature": "6.3",
                        },
                        {
                            "cell_id": 393,
                            "orientation": "S->U",
                            "source_m2_bracket": ["0.984", "0.985"],
                            "masses": ["0.996", "0.9841", "1.0"],
                            "closure_norm": "1e-18",
                            "minus_one_event": "1e-12",
                            "dm1_from_fold": "0.0003",
                            "dm2_from_fold": "0.0098",
                            "secant_m1_curvature": "6.1",
                        },
                    ],
                },
            }
        )
    )
    out = tmp_path / "left.json"
    assert (
        _run(
            "classify_secondary_left_birth.py",
            [str(geometry), str(out), "--bigfloat", str(bigfloat)],
        )
        == 0
    )
    record = json.loads(out.read_text())
    assert record["class"] == "projection_fold"
    assert record["passed"] is True
    assert record["evidence_level"] == "independently_reproduced"
    assert len(record["edge_endpoint_bindings"]) == 2


def test_right_death_interior_nonconvergence_is_only_a_fold_hypothesis(tmp_path) -> None:
    src = tmp_path / "right.json"
    src.write_text(
        json.dumps(
            {
                "direct_candidate": None,
                "closest_approach_mass_gap": 0.02,
                "approach": [{"step": 0, "mass_gap": 0.03}, {"step": 1, "mass_gap": 0.02}],
                "continuation_error": "RuntimeError: walls ended",
            }
        )
    )
    out = tmp_path / "class.json"
    assert _run("classify_secondary_right_death.py", [str(src), str(out)]) == 0
    record = json.loads(out.read_text())
    assert record["class"] == "projection_fold"
    assert record["passed"] is False
    assert record["evidence_level"] == "screening"
    assert "newton" not in record["class"]


def test_right_death_domain_boundary_requires_explicit_boundary_reach(tmp_path) -> None:
    src = tmp_path / "right.json"
    src.write_text(
        json.dumps(
            {
                "direct_candidate": None,
                "closest_approach_mass_gap": 0.02,
                "approach": [{"step": 0, "mass_gap": 0.02}],
                "reached_declared_domain_boundary": True,
            }
        )
    )
    out = tmp_path / "class.json"
    assert _run("classify_secondary_right_death.py", [str(src), str(out)]) == 0
    record = json.loads(out.read_text())
    assert record["class"] == "domain_boundary"
    assert record["passed"] is False


def test_right_death_canonical_mixed_record_closes_classification(tmp_path) -> None:
    screen = tmp_path / "screen.json"
    screen.write_text(
        json.dumps(
            {
                "direct_candidate": {
                    "success": True,
                    "masses": [1.0425, 1.04, 1.0],
                },
                "edge_endpoint_bindings": [
                    {"cell_id": 576, "side": "end", "mechanism": "plus_one"},
                    {"cell_id": 577, "side": "end", "mechanism": "minus_one"},
                ],
            }
        )
    )
    canonical = tmp_path / "canonical.json"
    canonical.write_text(
        json.dumps(
            {
                "passed": True,
                "implementation": "independent test fixture",
                "masses": ["1.04250001", "1.04000001", "1.0"],
                "relative_closure_norm": "1e-18",
                "relative_event_norm": "1e-12",
                "physical_plus_one_event": "1e-12",
                "physical_minus_one_event": "1e-12",
            }
        )
    )
    out = tmp_path / "class.json"
    assert (
        _run(
            "classify_secondary_right_death.py",
            [str(screen), str(out), "--canonical", str(canonical)],
        )
        == 0
    )
    record = json.loads(out.read_text())
    assert record["class"] == "mixed_organizer"
    assert record["passed"] is True
    assert record["evidence_level"] == "physical"
    assert len(record["edge_endpoint_bindings"]) == 2


def test_daughter_classification_requires_structural_and_independent_separation(tmp_path) -> None:
    structural = tmp_path / "structural.json"
    structural.write_text(
        json.dumps(
            {
                "artifacts": [
                    {
                        "label": "d0-minus",
                        "trace_point_count": 18,
                        "maximum_closure_norm": 5e-8,
                    }
                ],
                "symmetry_and_branch_identity_screens": {
                    "direction0_plus_minus_reversing_symmetry_best_normalized_chart_distance": 4e-7
                },
            }
        )
    )
    independent = tmp_path / "independent.json"
    independent.write_text(
        json.dumps(
            {
                "passed": True,
                "implementation": "independent test fixture",
                "results": [
                    {
                        "closure_norm": "1e-18",
                        "gauge_norm": "1e-18",
                        "parent_distance": "1e-3",
                        "off_li_norm": "1e-4",
                    }
                ],
            }
        )
    )
    out = tmp_path / "daughter.json"
    assert (
        _run(
            "classify_lower_plus_one_daughter.py",
            [
                str(out),
                "--structural",
                str(structural),
                "--independent",
                str(independent),
            ],
        )
        == 0
    )
    record = json.loads(out.read_text())
    assert record["class"] == "distinct_branch"
    assert record["passed"] is True
    assert record["evidence_level"] == "independently_reproduced"


def test_completeness_refuses_without_neck(tmp_path) -> None:
    al = ROOT / "research/evidence/V1_AL_POCKET_SCREEN_2026-08-15.json"
    out = tmp_path / "comp.json"
    assert _run("freeze_completeness_certificate.py", [str(out), "--al-screen", str(al)]) == 2
    record = json.loads(out.read_text())
    assert record["passed"] is False


def test_germ_builder_maps_frozen_principal_left_junction(tmp_path) -> None:
    out = tmp_path / "germs-test.json"
    src = ROOT / "research/evidence/V1_JUNCTION_PRINCIPAL_LEFT_2026-08-15.json"
    assert src.is_file()
    assert _run(
        "build_mixed_germs_from_junction.py",
        [str(out), "--junction", str(src)],
    ) == 0
    payload = json.loads(out.read_text())
    keys = {(g["mixed_node"], g["event_mode"], g["direction"]) for g in payload["germs"]}
    assert ("mixed_principal_left", "plus_one", "+") in keys
    assert ("mixed_principal_left", "minus_one", "-") in keys


def test_germ_builder_binds_new_mixed_endpoint_on_both_sides(tmp_path) -> None:
    junction = tmp_path / "junction-secondary-right.json"
    junction.write_text(
        json.dumps(
            {
                "mixed_node": "secondary_right_death",
                "traces": [
                    {
                        "event_mode": mode,
                        "localized_seeds": [
                            {"masses": [1.041, 1.044, 1.0]},
                            {"masses": [1.042, 1.045, 1.0]},
                        ],
                        "points": [
                            {"masses": [1.0426, 1.0460, 1.0]},
                            {"masses": [1.043, 1.047, 1.0]},
                        ],
                    }
                    for mode in ("plus_one", "minus_one")
                ],
            }
        )
    )
    canonical = tmp_path / "canonical.json"
    canonical.write_text(
        json.dumps({"passed": True, "masses": [1.0426, 1.0460, 1.0]})
    )
    out = tmp_path / "germs.json"
    assert _run(
        "build_mixed_germs_from_junction.py",
        [
            str(out),
            "--junction",
            str(junction),
            "--canonical",
            str(canonical),
            "--mixed-node",
            "secondary_right_death",
        ],
    ) == 0
    germs = json.loads(out.read_text())["germs"]
    keys = {(row["mixed_node"], row["event_mode"], row["direction"]) for row in germs}
    assert keys == {
        ("secondary_right_death", mode, direction)
        for mode in ("plus_one", "minus_one")
        for direction in ("+", "-")
    }
    assert all(row["canonical_bound"] is True for row in germs)


def test_canonical_germ_direction_audit_requires_opposite_mass_directions() -> None:
    namespace = runpy.run_path(
        str(ROOT / "scripts/trace_canonical_mixed_germs.py")
    )
    audit = namespace["directional_audit"]
    center = np.asarray([1.0, 1.0])
    germs = [
        {
            "event_mode": mode,
            "direction": direction,
            "masses": [1.0 + sign * 1e-4, 1.0 + sign * 2e-4, 1.0],
        }
        for mode in ("plus_one", "minus_one")
        for direction, sign in (("+", 1.0), ("-", -1.0))
    ]
    result = audit(germs, center)
    assert result["plus_one"]["opposite_mass_directions"] is True
    germs[-1]["masses"] = [1.0001, 1.0002, 1.0]
    with pytest.raises(SystemExit, match="opposite mass directions"):
        audit(germs, center)


def test_completeness_passes_with_neck_and_clean_al(tmp_path) -> None:
    al = ROOT / "research/evidence/V1_AL_POCKET_SCREEN_2026-08-15.json"
    neck = tmp_path / "neck.json"
    neck.write_text(
        json.dumps(
            {
                "completed": True,
                "grid": {"m1": [0.997, 0.999], "m2": [0.993, 1.006], "step": 0.0001, "samples": 12},
                "minimum_resolved_unstable_gap": 0.0002,
                "any_vertical_merge": False,
                "any_boundary_truncated_merge_test": False,
                "any_line_without_stable_sample": False,
                "any_stable_interval_touches_boundary": False,
                "all_lines_separated": True,
                "max_shooting_residual": 1e-9,
                "line_summaries": [
                    {
                        "m1": 0.997,
                        "stable_intervals": [[0.994, 0.996], [0.998, 1.0]],
                        "interior_unstable_gaps": [0.0019],
                        "merge_verdict": "separated",
                    }
                ],
            }
        )
    )
    out = tmp_path / "comp.json"
    assert _run("freeze_completeness_certificate.py", [str(out), "--al-screen", str(al), "--neck-scan", str(neck)]) == 0
    record = json.loads(out.read_text())
    assert record["passed"] is True
    assert record["active_learning"]["screening_stable_hidden_pockets"] == 0


def test_completeness_rejects_vertical_merge(tmp_path) -> None:
    al = ROOT / "research/evidence/V1_AL_POCKET_SCREEN_2026-08-15.json"
    neck = tmp_path / "neck.json"
    neck.write_text(
        json.dumps(
            {
                "completed": True,
                "grid": {"m1": [0.997, 0.997], "m2": [0.993, 1.006], "step": 0.0001, "samples": 131},
                "minimum_resolved_unstable_gap": None,
                "any_vertical_merge": True,
                "max_shooting_residual": 1e-9,
                "line_summaries": [{"m1": 0.997, "stable_intervals": [[0.993, 1.006]], "interior_unstable_gaps": []}],
            }
        )
    )
    out = tmp_path / "comp.json"
    assert _run("freeze_completeness_certificate.py", [str(out), "--al-screen", str(al), "--neck-scan", str(neck)]) == 2
    assert json.loads(out.read_text())["passed"] is False


def test_neck_tile_merger_requires_exact_complete_grid(tmp_path) -> None:
    fragments = []
    for m1 in (0.997, 0.9971):
        samples = [
            {"m1": m1, "m2": m2, "stable": False}
            for m2 in (0.993, 0.9931)
        ]
        fragment = tmp_path / f"tile-{m1}.json"
        fragment.write_text(
            json.dumps(
                {
                    "completed": True,
                    "grid": {"m1": [m1, m1], "m2": [0.993, 0.9931], "step": 0.0001, "samples": 2},
                    "max_shooting_residual": 1e-9,
                    "line_summaries": [{"m1": m1, "stable_intervals": [], "interior_unstable_gaps": []}],
                    "samples": samples,
                }
            )
        )
        fragments.append(fragment)
    output = tmp_path / "merged.json"
    args = [
        str(output),
        *(str(path) for path in fragments),
        "--expected-m1-min", "0.997",
        "--expected-m1-max", "0.9971",
        "--expected-m2-min", "0.993",
        "--expected-m2-max", "0.9931",
        "--step", "0.0001",
    ]
    assert _run("merge_stability_neck_scans.py", args) == 0
    merged = json.loads(output.read_text())
    assert merged["completed"] is True
    assert merged["fragment_count"] == 2
    assert merged["grid"]["samples"] == 4
