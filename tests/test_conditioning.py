"""Conditioning metadata must accompany every published residual."""
from __future__ import annotations

import math

import numpy as np

from threebody_atlas.conditioning import (
    SolveConditioning,
    condition_report,
    conditioning_dict,
    scalar_condition_report,
    summarize_conditioning,
)


def test_condition_report_recovers_known_singular_values() -> None:
    jacobian = np.diag([1e3, 1e-3])
    residual = np.array([1e-9, 0.0])
    report = condition_report(jacobian, residual)
    assert report is not None
    assert math.isclose(report.sigma_max, 1e3, rel_tol=1e-12)
    assert math.isclose(report.sigma_min, 1e-3, rel_tol=1e-12)
    assert math.isclose(report.kappa_2, 1e6, rel_tol=1e-12)
    assert not report.rank_deficient
    # The whole point: a 1e-9 residual buys 1e-6 of forward accuracy here, not 1e-9.
    assert math.isclose(report.displacement_bound, 1e-6, rel_tol=1e-12)


def test_residual_alone_cannot_distinguish_two_very_different_solves() -> None:
    """Same residual, conditioning six orders apart, forward error six apart."""
    residual = np.array([1e-25, 0.0])
    benign = condition_report(np.diag([1.0, 1e-3]), residual)
    vicious = condition_report(np.diag([1.0, 1e-22]), residual)
    assert benign is not None and vicious is not None
    assert benign.residual_norm == vicious.residual_norm
    assert vicious.displacement_bound / benign.displacement_bound > 1e18


def test_rank_deficiency_is_flagged_and_bound_blows_up() -> None:
    """A numerically rank-deficient Jacobian must be labelled, not averaged away."""
    jacobian = np.array([[1.0, 2.0], [2.0, 4.0]])
    report = condition_report(jacobian, np.array([1e-12, 0.0]))
    assert report is not None
    assert report.rank_deficient
    assert report.numerical_rank == 1
    assert report.kappa_2 > 1e15
    assert report.displacement_bound > 1.0

    exact = condition_report(np.array([[0.0, 0.0], [0.0, 0.0]]), np.array([1e-12, 0.0]))
    assert exact is not None
    assert math.isinf(exact.kappa_2)
    assert math.isinf(exact.displacement_bound)


def test_rectangular_overdetermined_shooting_jacobian_shape_is_recorded() -> None:
    rng = np.random.default_rng(11)
    jacobian = rng.normal(size=(8, 4))
    report = condition_report(jacobian, np.zeros(8))
    assert report is not None
    assert (report.rows, report.cols) == (8, 4)
    assert len(report.singular_values) == 4
    assert report.numerical_rank == 4


def test_scalar_condition_report_converts_event_residual_to_parameter_error() -> None:
    report = scalar_condition_report(slope=-125.92548031431505, residual=1.989798081858396e-08)
    assert report.kappa_2 == 1.0
    assert math.isclose(report.displacement_bound, 1.5801e-10, rel_tol=1e-3)


def test_scalar_condition_report_with_zero_slope_is_infinite_not_zero() -> None:
    report = scalar_condition_report(slope=0.0, residual=1e-30)
    assert report.rank_deficient
    assert math.isinf(report.displacement_bound)


def test_condition_report_is_none_without_a_jacobian() -> None:
    assert condition_report(None, np.zeros(3)) is None
    assert conditioning_dict(None) is None


def test_summarize_accepts_objects_and_serialized_dicts_alike() -> None:
    reports: list[SolveConditioning | dict | None] = [
        condition_report(np.diag([1.0, 1e-2]), np.array([1e-10, 0.0])),
        conditioning_dict(condition_report(np.diag([1.0, 1e-4]), np.array([1e-10, 0.0]))),
        None,
    ]
    summary = summarize_conditioning(reports)
    assert summary["reported"] == 2
    assert summary["missing"] == 1
    assert math.isclose(summary["kappa_2_max"], 1e4, rel_tol=1e-9)
    assert math.isclose(summary["displacement_bound_max"], 1e-6, rel_tol=1e-9)


def test_merge_carries_julia_conditioning_through_to_the_published_root() -> None:
    """Conditioning must survive the float64/BigFloat merge, not be dropped there."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts/merge_hybrid_critical_roots.py"
    spec = importlib.util.spec_from_file_location("_merge_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Obviously synthetic BigFloat row, in the string form the Julia lane emits.
    julia_row = {
        "cell_id": 7,
        "event_mode": "plus_one",
        "passed": True,
        "event_value": "1.0e-30",
        "closure_norm": "2.0e-25",
        "m1": "0.9",
        "m2": "0.9",
        "m3": "1.0",
        "closure_conditioning": {
            "rows": 8,
            "cols": 4,
            "sigma_max": "1.0e3",
            "sigma_min": "1.0e-2",
            "kappa_2": "1.0e5",
            "residual_norm": "2.0e-25",
            "displacement_bound": "2.0e-23",
        },
        "event_conditioning": {
            "rows": 1,
            "cols": 1,
            "sigma_max": "50.0",
            "sigma_min": "50.0",
            "kappa_2": "1",
            "residual_norm": "1.0e-30",
            "displacement_bound": "2.0e-32",
        },
        "m2_uncertainty": "2.0e-32",
    }
    merged = module.from_julia(julia_row, None)
    assert merged["closure_conditioning"]["kappa_2"] == "1.0e5"
    assert merged["event_conditioning"]["displacement_bound"] == "2.0e-32"
    assert merged["m2_uncertainty"] == "2.0e-32"

    summary = summarize_conditioning([merged["closure_conditioning"]])
    assert math.isclose(summary["kappa_2_max"], 1.0e5, rel_tol=1e-12)
    assert math.isclose(summary["displacement_bound_max"], 2.0e-23, rel_tol=1e-12)


def test_family_point_correction_carries_its_own_conditioning() -> None:
    """The screening corrector must not hand back a bare residual norm."""
    from threebody_atlas.liao_family import correct_family_point

    point = correct_family_point(
        (0.8, 0.7557199219114411, 1.0),
        (-0.13492372038723308, 2.5161161461948803, 0.31611697369085825, 5.156231097524942),
        max_nfev=20,
    )
    assert point.conditioning is not None
    assert (point.conditioning.rows, point.conditioning.cols) == (8, 4)
    assert point.conditioning.sigma_min > 0.0
    assert point.conditioning.residual_norm == point.residual_norm
    assert (
        point.conditioning.displacement_bound
        >= point.residual_norm / point.conditioning.sigma_max
    )
