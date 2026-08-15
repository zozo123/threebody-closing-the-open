from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

import threebody_atlas.generic_branch as generic_branch


@dataclass(frozen=True)
class Point:
    masses: tuple[float, float, float]
    state: tuple[float, float, float, float, float, float, float, float]
    period: float


def point(m2: float) -> Point:
    state = (m2, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return Point((0.8, m2, 1.0), state, 1.0)


def fake_generic_residual(y, masses, reference_state, **_kwargs):
    """Analytic 1D branch: z0[0]=m2, other state fixed, T=1."""
    z0 = np.asarray(y[:8], dtype=float)
    period = float(y[8])
    reference = np.asarray(reference_state, dtype=float)
    m2 = float(masses[1])
    residual = np.zeros(11, dtype=float)
    residual[0] = z0[0] - m2
    residual[1:8] = z0[1:8] - reference[1:8]
    residual[8] = period - 1.0
    # Rows 9 and 10 deliberately duplicate exact first-integral/gauge
    # dependencies: the real generic corrector is also overdetermined.
    return residual


def test_advance_generic_branch_tracks_analytic_curve(monkeypatch):
    monkeypatch.setattr(generic_branch, "generic_periodic_residual", fake_generic_residual)
    reference = np.asarray(point(0.8).state, dtype=float)
    result = generic_branch.advance_generic_branch(
        point(0.80),
        point(0.81),
        reference_state=reference,
        m1=0.8,
        m3=1.0,
        normalized_step=0.01,
        m2_bounds=(0.7, 1.2),
        max_closure=1e-9,
        max_gauge=1e-9,
        max_phase=1e-9,
        max_arc=1e-7,
    )
    assert result.success
    assert result.masses[1] > 0.81
    assert result.state[0] == pytest.approx(result.masses[1], abs=1e-9)
    assert result.period == pytest.approx(1.0, abs=1e-9)
    assert result.closure_norm <= 1e-9
    assert abs(result.arclength_residual) <= 1e-7


def test_trace_generic_branch_accepts_multiple_steps(monkeypatch):
    monkeypatch.setattr(generic_branch, "generic_periodic_residual", fake_generic_residual)
    reference = np.asarray(point(0.8).state, dtype=float)
    trace = generic_branch.trace_generic_branch(
        point(0.80),
        point(0.81),
        reference_state=reference,
        m1=0.8,
        m3=1.0,
        steps=3,
        normalized_step=0.01,
        m2_bounds=(0.7, 1.2),
    )
    assert len(trace.points) == 3
    assert trace.stopped_reason == "requested_steps_completed"
    assert all(
        a.masses[1] < b.masses[1]
        for a, b in zip(trace.points, trace.points[1:], strict=True)
    )


def test_continuation_vector_rejects_bad_state():
    bad = Point((0.8, 0.8, 1.0), (0.0,) * 7, 1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="shape"):
        generic_branch.continuation_vector(bad)
