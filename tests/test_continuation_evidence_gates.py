"""Fail-closed checks for the #192 continuous-witness evidence lanes."""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _module(name: str, relative: str):
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CONT = _module(
    "trace_label_invisible_continuous_tested",
    "scripts/trace_label_invisible_continuous.py",
)
ENDS = _module(
    "resolve_sampled_sweep_endpoints_tested",
    "scripts/resolve_sampled_sweep_endpoints.py",
)


def test_independent_tangent_diagnostics_are_release_gates() -> None:
    assert CONT._jax_diagnostics_pass(
        {
            "relative_null_residual": CONT.MAX_JAX_RELATIVE_NULL_RESIDUAL,
            "spectral_gap": CONT.MIN_JAX_SPECTRAL_GAP,
        }
    )
    assert not CONT._jax_diagnostics_pass(
        {
            "relative_null_residual": 1.01 * CONT.MAX_JAX_RELATIVE_NULL_RESIDUAL,
            "spectral_gap": CONT.MIN_JAX_SPECTRAL_GAP,
        }
    )
    assert not CONT._jax_diagnostics_pass(
        {
            "relative_null_residual": CONT.MAX_JAX_RELATIVE_NULL_RESIDUAL,
            "spectral_gap": 0.99 * CONT.MIN_JAX_SPECTRAL_GAP,
        }
    )


@pytest.mark.parametrize("reference", ([1.0], [1.0, 2.0, 3.0], [1.0, float("nan")]))
def test_jax_tangent_rejects_a_non_mass_reference(reference) -> None:
    with pytest.raises(ValueError, match="two finite components"):
        CONT._jax_mass_tangent(None, reference)


def test_sample_proximity_is_not_a_scientific_terminus() -> None:
    assert "existing_catalog_critical_curve" not in ENDS.SCIENTIFIC_TERMINAL_KINDS
    assert "declared_domain_boundary" in ENDS.SCIENTIFIC_TERMINAL_KINDS
    assert "mixed_organizer" in ENDS.SCIENTIFIC_TERMINAL_KINDS


def test_direction_only_seed_orients_current_minus_previous_outward() -> None:
    current = np.asarray([1.048, 1.120])
    outward = np.asarray([-0.002, -0.014])
    previous = ENDS._direction_only_previous_mass(current, outward)
    actual_reference = current - previous
    assert np.dot(actual_reference, outward) > 0.0
    assert np.linalg.norm(actual_reference) == pytest.approx(1e-3)


def test_event_bracket_uses_secant_and_falls_back_to_bisection() -> None:
    def evaluation(lam: float, value: float):
        return CONT._NormalEvaluation(lam, value, None)

    left = evaluation(-2.0, -3.0)
    right = evaluation(4.0, 6.0)
    assert CONT._bracket_trial_lambda(left, right) == pytest.approx(0.0)

    flat_left = evaluation(-2.0, -1.0)
    flat_right = evaluation(4.0, -1.0)
    assert CONT._bracket_trial_lambda(flat_left, flat_right) == pytest.approx(1.0)


def test_event_bracket_selects_narrowest_sampled_sign_change() -> None:
    def evaluation(lam: float, value: float):
        return CONT._NormalEvaluation(lam, value, None)

    outer_left = evaluation(-2.0, -2.0)
    inner_left = evaluation(-0.1, -0.1)
    inner_right = evaluation(0.2, 0.2)
    outer_right = evaluation(3.0, 3.0)
    bracket = CONT._tightest_sign_bracket(
        [inner_right, outer_right, outer_left, inner_left]
    )
    assert bracket == (inner_left, inner_right)


def test_julia_lane_explicitly_enforces_the_same_event_gate() -> None:
    source = (ROOT / "julia/verify_critical_points.jl").read_text(encoding="utf-8")
    workflow = (ROOT / ".github/workflows/label-invisible-bigfloat.yml").read_text(
        encoding="utf-8"
    )
    assert 'event_gate = parse(BigFloat,"2e-8")' in source
    assert "independent critical event gate failed" in source
    assert "representative_passed_event_gate" in workflow


def test_continuation_reopens_signed_seed_cells_and_checkpoints_initialization() -> None:
    source = (ROOT / "scripts/trace_label_invisible_continuous.py").read_text(
        encoding="utf-8"
    )
    workflow = (ROOT / ".github/workflows/label-invisible-continuation.yml").read_text(
        encoding="utf-8"
    )
    assert "_precise_bracket_search(" in source
    assert "m2_bounds=(bounds[0], bounds[1])" in source
    assert "max_steps=32" in source
    assert "sys.excepthook = checkpoint_unhandled_exception" in source
    assert "failed_initialization_partial_continuation_evidence" in source
    assert "if: always()" in workflow


def test_continuation_covers_every_multi_point_supplemental_component() -> None:
    source = (ROOT / "scripts/trace_label_invisible_continuous.py").read_text(
        encoding="utf-8"
    )
    component_sets = set(
        re.findall(r"_mesh_rows\(\s*supplemental,\s*(\{[^}]+\})", source)
    )
    assert {"{0}", "{1}", "{3}", "{4}", "{10}", "{11, 12}"} <= component_sets
    assert 'branch_id="secondary_right_minus_to_domain"' in source
    assert 'branch_id="principal_right_minus_to_domain"' in source
