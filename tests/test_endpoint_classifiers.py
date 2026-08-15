from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path


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
    assert record["passed"] is True


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
    assert json.loads(out.read_text())["class"] == "two_separate_arcs"


def test_right_death_without_mixed_is_domain_boundary(tmp_path) -> None:
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
    assert record["class"] == "domain_boundary"
    assert "newton" not in record["class"]


def test_completeness_refuses_without_neck(tmp_path) -> None:
    al = ROOT / "research/evidence/V1_AL_POCKET_SCREEN_2026-08-15.json"
    out = tmp_path / "comp.json"
    assert _run("freeze_completeness_certificate.py", [str(out), "--al-screen", str(al)]) == 2
    record = json.loads(out.read_text())
    assert record["passed"] is False


def test_completeness_passes_with_neck_and_clean_al(tmp_path) -> None:
    al = ROOT / "research/evidence/V1_AL_POCKET_SCREEN_2026-08-15.json"
    neck = tmp_path / "neck.json"
    neck.write_text(
        json.dumps(
            {
                "grid": {"m1": [0.997, 0.999], "m2": [0.993, 1.006], "step": 0.0001, "samples": 12},
                "minimum_resolved_unstable_gap": 0.0002,
                "any_vertical_merge": False,
                "max_shooting_residual": 1e-9,
            }
        )
    )
    out = tmp_path / "comp.json"
    assert _run("freeze_completeness_certificate.py", [str(out), "--al-screen", str(al), "--neck-scan", str(neck)]) == 0
    record = json.loads(out.read_text())
    assert record["passed"] is True
    assert record["active_learning"]["screening_stable_hidden_pockets"] == 0
