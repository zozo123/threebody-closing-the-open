"""Scan-window truncation must be reported, never scored as a lobe merge.

The historical detector answered "did the two stable lobes merge?" with
``len(stable_intervals) <= 1``.  On the frozen raster from CI run 31932398616
that returned True for the four lines m1 = 0.9987 .. 0.9990 -- not because the
lobes merged there, but because the minus-one U->S wall that opens the second
lobe sits at m2 = 1.0080934 (m1 = 0.999), above the m2-max of 1.006 that the
raster actually scanned.  These tests pin the honest split of that verdict.
"""
from __future__ import annotations

import importlib.util
import json
import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("neck_topology", SCRIPTS / "neck_topology.py")
NECK = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(NECK)

AL_SCREEN = ROOT / "research/evidence/V1_AL_POCKET_SCREEN_2026-08-15.json"
STEP = 0.0001


def _exit_code(script: str, args: list[str]) -> object:
    """Run a script under ``__main__`` and return its raw SystemExit code."""
    argv = sys.argv
    sys.argv = [script, *args]
    try:
        try:
            runpy.run_path(str(SCRIPTS / script), run_name="__main__")
            return 0
        except SystemExit as exc:
            return exc.code if exc.code is not None else 0
    finally:
        sys.argv = argv


def _run(script: str, args: list[str]) -> int:
    code = _exit_code(script, args)
    assert isinstance(code, int), f"{script} aborted with a message: {code}"
    return code


def _line(m1: float, intervals: list[list[float]], gaps: list[float]) -> dict:
    return {"m1": m1, "stable_intervals": intervals, "interior_unstable_gaps": gaps}


def _annotate(intervals: list[list[float]], gaps: list[float], *, m2_min=0.993, m2_max=1.006):
    return NECK.annotate_line_summary(
        _line(0.999, intervals, gaps), m2_min=m2_min, m2_max=m2_max, step=STEP
    )


def test_single_interval_spanning_the_window_is_truncated_not_merged() -> None:
    line = _annotate([[0.993, 1.006]], [])
    assert line["merge_verdict"] == NECK.TRUNCATION_UNDECIDABLE
    assert line["truncated_at_m2_min"] is True
    assert line["truncated_at_m2_max"] is True
    assert line["separation_witnessed"] is False


def test_frozen_raster_line_that_lost_its_second_lobe_is_truncated_not_merged() -> None:
    # Exactly the m1=0.999 summary of the merged raster of run 31932398616.
    line = _annotate([[0.993, 1.0015]], [])
    assert line["merge_verdict"] == NECK.TRUNCATION_UNDECIDABLE
    assert line["truncated_at_m2_min"] is True
    assert line["truncated_at_m2_max"] is False


def test_widening_past_the_wall_turns_the_same_line_into_a_separation() -> None:
    # The minus-one U->S wall at m2=1.0080934 is inside a window that reaches
    # 1.012, so the second lobe is sampled and the interior gap is witnessed.
    line = _annotate([[0.993, 1.0015], [1.0081, 1.012]], [0.0065], m2_max=1.012)
    assert line["merge_verdict"] == NECK.SEPARATED
    assert line["separation_witnessed"] is True
    assert line["resolved_interior_gap_count"] == 1
    # The outer edges of both lobes still lie outside the window; that must stay
    # visible rather than being silently dropped.
    assert line["boundary_truncated"] is True


def test_interval_that_starts_and_ends_inside_the_window_is_a_real_merge() -> None:
    line = _annotate([[0.9945, 1.0055]], [])
    assert line["merge_verdict"] == NECK.INTERIOR_MERGE
    assert line["boundary_truncated"] is False


def test_one_grid_step_inside_the_edge_does_not_count_as_truncated() -> None:
    line = _annotate([[0.9931, 1.0059]], [])
    assert line["truncated_at_m2_min"] is False
    assert line["truncated_at_m2_max"] is False
    assert line["merge_verdict"] == NECK.INTERIOR_MERGE


def test_line_with_no_stable_sample_is_its_own_verdict() -> None:
    line = _annotate([], [])
    assert line["merge_verdict"] == NECK.NO_STABLE_SAMPLE
    assert line["boundary_truncated"] is False


def test_aggregate_keeps_truncation_out_of_any_vertical_merge() -> None:
    summaries, flags = NECK.summarize(
        [
            _line(0.997, [[0.993, 0.9951], [0.9993, 1.006]], [0.0041]),
            _line(0.999, [[0.993, 1.0015]], []),
        ],
        m2_min=0.993,
        m2_max=1.006,
        step=STEP,
    )
    assert [row["merge_verdict"] for row in summaries] == [
        NECK.SEPARATED,
        NECK.TRUNCATION_UNDECIDABLE,
    ]
    assert flags["any_vertical_merge"] is False
    assert flags["any_boundary_truncated_merge_test"] is True
    assert flags["any_stable_interval_touches_boundary"] is True
    assert flags["all_lines_separated"] is False
    assert flags["boundary_truncated_lines"] == [0.999]
    assert flags["merge_verdict_counts"]["truncation_undecidable"] == 1


def test_aggregate_flags_an_interior_merge_separately() -> None:
    _, flags = NECK.summarize(
        [_line(0.998, [[0.9945, 1.0055]], [])], m2_min=0.993, m2_max=1.006, step=STEP
    )
    assert flags["any_vertical_merge"] is True
    assert flags["any_boundary_truncated_merge_test"] is False
    assert flags["all_lines_separated"] is False


def test_aggregate_passes_only_when_every_line_is_separated() -> None:
    _, flags = NECK.summarize(
        [
            _line(0.997, [[0.993, 0.9951], [0.9993, 1.012]], [0.0041]),
            _line(0.999, [[0.993, 1.0015], [1.0081, 1.012]], [0.0065]),
        ],
        m2_min=0.993,
        m2_max=1.012,
        step=STEP,
    )
    assert flags["all_lines_separated"] is True
    assert flags["any_vertical_merge"] is False
    assert flags["any_boundary_truncated_merge_test"] is False
    assert flags["any_line_without_stable_sample"] is False


def test_annotation_is_idempotent() -> None:
    once = _annotate([[0.993, 1.0015]], [])
    twice = NECK.annotate_line_summary(once, m2_min=0.993, m2_max=1.006, step=STEP)
    assert once == twice


def _neck_payload(summaries: list[dict], *, m2_min: float, m2_max: float, gap: float) -> dict:
    annotated, flags = NECK.summarize(summaries, m2_min=m2_min, m2_max=m2_max, step=STEP)
    return {
        "schema": "atlas.v1.stability-neck-scan/3",
        "completed": True,
        "grid": {"m1": [0.997, 0.999], "m2": [m2_min, m2_max], "step": STEP, "samples": 42},
        "max_shooting_residual": 1e-9,
        "minimum_resolved_unstable_gap": gap,
        **flags,
        "line_summaries": annotated,
    }


def test_completeness_refuses_a_truncated_raster(tmp_path) -> None:
    neck = tmp_path / "neck.json"
    neck.write_text(
        json.dumps(
            _neck_payload(
                [
                    _line(0.997, [[0.993, 0.9951], [0.9993, 1.006]], [0.0041]),
                    _line(0.999, [[0.993, 1.0015]], []),
                ],
                m2_min=0.993,
                m2_max=1.006,
                gap=0.0041,
            )
        )
    )
    out = tmp_path / "comp.json"
    assert _run(
        "freeze_completeness_certificate.py",
        [str(out), "--al-screen", str(AL_SCREEN), "--neck-scan", str(neck)],
    ) == 2
    record = json.loads(out.read_text())
    assert record["passed"] is False
    assert record["neck"]["any_vertical_merge"] is False
    assert record["neck"]["any_boundary_truncated_merge_test"] is True
    assert record["neck"]["topology_clean"] is False


def test_completeness_accepts_a_raster_whose_lines_are_all_separated(tmp_path) -> None:
    neck = tmp_path / "neck.json"
    neck.write_text(
        json.dumps(
            _neck_payload(
                [
                    _line(0.997, [[0.993, 0.9951], [0.9993, 1.012]], [0.0041]),
                    _line(0.999, [[0.993, 1.0015], [1.0081, 1.012]], [0.0065]),
                ],
                m2_min=0.993,
                m2_max=1.012,
                gap=0.0041,
            )
        )
    )
    out = tmp_path / "comp.json"
    assert _run(
        "freeze_completeness_certificate.py",
        [str(out), "--al-screen", str(AL_SCREEN), "--neck-scan", str(neck)],
    ) == 0
    record = json.loads(out.read_text())
    assert record["passed"] is True
    assert record["neck"]["all_lines_separated"] is True
    # The lobes still leave the window; the certificate says so out loud.
    assert record["neck"]["any_stable_interval_touches_boundary"] is True


def test_completeness_refuses_a_raster_predating_the_truncation_analysis(tmp_path) -> None:
    neck = tmp_path / "neck.json"
    neck.write_text(
        json.dumps(
            {
                "schema": "atlas.v1.stability-neck-scan/2",
                "completed": True,
                "grid": {"m1": [0.997, 0.999], "m2": [0.993, 1.006], "step": STEP, "samples": 42},
                "max_shooting_residual": 1e-9,
                "minimum_resolved_unstable_gap": 0.0041,
                "any_vertical_merge": False,
                "line_summaries": [
                    _line(0.997, [[0.993, 0.9951], [0.9993, 1.006]], [0.0041]),
                ],
            }
        )
    )
    out = tmp_path / "comp.json"
    assert _run(
        "freeze_completeness_certificate.py",
        [str(out), "--al-screen", str(AL_SCREEN), "--neck-scan", str(neck)],
    ) == 2
    assert json.loads(out.read_text())["passed"] is False


def test_completeness_refuses_a_still_resolved_gap_below_one_grid_step(tmp_path) -> None:
    neck = tmp_path / "neck.json"
    neck.write_text(
        json.dumps(
            _neck_payload(
                [_line(0.997, [[0.993, 0.9951], [0.9993, 1.012]], [0.0041])],
                m2_min=0.993,
                m2_max=1.012,
                gap=0.0,
            )
        )
    )
    out = tmp_path / "comp.json"
    assert _run(
        "freeze_completeness_certificate.py",
        [str(out), "--al-screen", str(AL_SCREEN), "--neck-scan", str(neck)],
    ) == 2


def _tile(path: Path, m1: float, m2_values: list[float], summary: dict) -> Path:
    path.write_text(
        json.dumps(
            {
                "completed": True,
                "grid": {
                    "m1": [m1, m1],
                    "m2": [m2_values[0], m2_values[-1]],
                    "step": STEP,
                    "samples": len(m2_values),
                },
                "max_shooting_residual": 1e-9,
                "line_summaries": [summary],
                "samples": [{"m1": m1, "m2": m2, "stable": False} for m2 in m2_values],
            }
        )
    )
    return path


def test_merger_reannotates_truncation_from_the_declared_window(tmp_path) -> None:
    m2_values = [0.993, 0.9931, 0.9932]
    tile = _tile(
        tmp_path / "tile.json",
        0.997,
        m2_values,
        _line(0.997, [[0.993, 0.9932]], []),
    )
    output = tmp_path / "merged.json"
    assert _run(
        "merge_stability_neck_scans.py",
        [
            str(output),
            str(tile),
            "--expected-m1-min", "0.997",
            "--expected-m1-max", "0.997",
            "--expected-m2-min", "0.993",
            "--expected-m2-max", "0.9932",
            "--step", "0.0001",
        ],
    ) == 0
    merged = json.loads(output.read_text())
    assert merged["schema"] == "atlas.v1.stability-neck-scan/3"
    assert merged["any_vertical_merge"] is False
    assert merged["any_boundary_truncated_merge_test"] is True
    assert merged["all_lines_separated"] is False
    assert merged["line_summaries"][0]["merge_verdict"] == NECK.TRUNCATION_UNDECIDABLE


def test_merger_rejects_a_tile_whose_own_verdict_disagrees(tmp_path) -> None:
    m2_values = [0.993, 0.9931, 0.9932]
    summary = _line(0.997, [[0.993, 0.9932]], [])
    summary["merge_verdict"] = NECK.SEPARATED
    tile = _tile(tmp_path / "tile.json", 0.997, m2_values, summary)
    output = tmp_path / "merged.json"
    code = _exit_code(
        "merge_stability_neck_scans.py",
        [
            str(output),
            str(tile),
            "--expected-m1-min", "0.997",
            "--expected-m1-max", "0.997",
            "--expected-m2-min", "0.993",
            "--expected-m2-max", "0.9932",
            "--step", "0.0001",
        ],
    )
    assert isinstance(code, str) and "merge-verdict mismatch" in code
    assert not output.exists()
