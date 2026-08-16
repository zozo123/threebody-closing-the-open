"""Tests for integrated variational mass sensitivities.

The load-bearing tests are the ones that would fail if ``df/dm``, ``dA/dm`` or
the chart's ``dc/dm`` gauge term carried a wrong mass coefficient.  Each of those
has a deliberately-wrong twin so the test is demonstrably able to fail.
"""
from __future__ import annotations

import numpy as np
import pytest

from threebody_atlas import mass_sensitivity as ms
from threebody_atlas.reduced import reduced_jacobian, reduced_rhs

# A generic (non-periodic) reduced state, well away from any close approach.
Z = np.array([-0.42, 0.17, 1.0, -0.23, 0.05, 1.21, -0.31, 0.44])
MASSES = (0.87, 1.13, 1.0)
# Short-flow chart point: not a periodic orbit, which is exactly the point --
# the flow sensitivity identity does not need one.
P_SHORT = np.array([-0.36, 1.25, 0.46, 0.35])

# Published cell root 392 (m1 = 0.996 secondary G- branch), used for the one
# end-to-end test against the finite differences this work replaces.
BRANCH_MASSES = (0.996, 0.9645756252404868, 1.0)
BRANCH_P = (-0.3548988720153132, 1.2574612854316434, 0.4601909497390701, 7.4381234411120865)


# --------------------------------------------------------------------------- #
# Pointwise algebra
# --------------------------------------------------------------------------- #
def test_reduced_field_matches_matrix_com_reduction():
    """The explicit relative field equals ``R f_12(P(m) z, m)`` from reduced.py."""
    f, a = ms.reduced_field(Z, MASSES)
    m = np.asarray(MASSES)
    assert np.allclose(f, reduced_rhs(0.0, Z, m), atol=1e-14)
    assert np.allclose(a, reduced_jacobian(Z, m), atol=1e-12)


def test_mass_partial_has_no_hidden_com_reduction_term():
    """``df/dm`` read off the explicit field matches differentiating the *matrix*
    reduction, whose reconstruction ``P(m)`` is itself mass dependent.

    This is the numerical form of the module-docstring claim that the COM
    reduction contributes nothing to the mass derivative.
    """
    h = 1e-6
    analytic = ms.field_mass_partial(Z, MASSES)
    for axis in range(2):
        hi, lo = list(MASSES), list(MASSES)
        hi[axis] += h
        lo[axis] -= h
        fd = (reduced_rhs(0.0, Z, np.asarray(hi)) - reduced_rhs(0.0, Z, np.asarray(lo))) / (2 * h)
        assert np.allclose(fd, analytic[:, axis], atol=1e-8)


def test_jacobian_mass_partial_matches_finite_difference():
    h = 1e-6
    analytic = ms.jacobian_mass_partial(Z, MASSES)
    for axis in range(2):
        hi, lo = list(MASSES), list(MASSES)
        hi[axis] += h
        lo[axis] -= h
        fd = (ms.reduced_field(Z, hi)[1] - ms.reduced_field(Z, lo)[1]) / (2 * h)
        assert np.allclose(fd, analytic[axis], atol=1e-8)


def test_kernel_hessian_contraction_matches_finite_difference():
    rng = np.random.default_rng(11)
    s = rng.normal(size=8)
    h = 1e-6
    fd = (ms.reduced_field(Z + h * s, MASSES)[1] - ms.reduced_field(Z - h * s, MASSES)[1]) / (2 * h)
    assert np.allclose(fd, ms.jacobian_state_directional(Z, MASSES, s), atol=1e-7)


def test_chart_mass_tangent_is_the_zero_momentum_gauge_term():
    """``dc/dm`` is nonzero and equals the finite difference of the chart map."""
    h = 1e-7
    analytic = ms.chart_mass_tangent(P_SHORT, MASSES)
    assert np.linalg.norm(analytic) > 1.0  # the gauge term is not negligible
    for axis in range(2):
        hi, lo = list(MASSES), list(MASSES)
        hi[axis] += h
        lo[axis] -= h
        fd = (ms.chart_state(P_SHORT, hi) - ms.chart_state(P_SHORT, lo)) / (2 * h)
        assert np.allclose(fd, analytic[:, axis], atol=1e-9)


def test_event_weight_reproduces_directional_derivative_of_the_event():
    rng = np.random.default_rng(3)
    mono = np.eye(8) + 0.1 * rng.normal(size=(8, 8))
    dm = rng.normal(size=(8, 8))
    h = 1e-7
    for mode in ms.EVENT_MODES:
        fd = (
            ms.event_from_monodromy(mono + h * dm, mode)
            - ms.event_from_monodromy(mono - h * dm, mode)
        ) / (2 * h)
        analytic = np.trace(ms.event_weight(mono, mode) @ dm)
        assert analytic == pytest.approx(fd, rel=1e-7, abs=1e-7)


# --------------------------------------------------------------------------- #
# Integrated sensitivities
# --------------------------------------------------------------------------- #
def _frozen_chart_flow(p, masses, rtol=1e-13, atol=1e-15):
    z_t, _, _, _, _ = ms.integrate_frozen_chart(p, masses, level=0, rtol=rtol, atol=atol)
    return z_t


def test_flow_sensitivity_matches_a_very_small_finite_difference():
    """S(T) from the sensitivity ODE against a central difference at h = 1e-7.

    This is the direct statement that the integrated sensitivity is the mass
    derivative of the flow, chart gauge term included.
    """
    _, _, s_t, _, _ = ms.integrate_frozen_chart(P_SHORT, MASSES, level=1)
    h = 1e-7
    for axis in range(2):
        hi, lo = list(MASSES), list(MASSES)
        hi[axis] += h
        lo[axis] -= h
        fd = (_frozen_chart_flow(P_SHORT, hi) - _frozen_chart_flow(P_SHORT, lo)) / (2 * h)
        assert np.allclose(fd, s_t[axis], rtol=1e-6, atol=1e-8)


def test_monodromy_sensitivity_matches_complex_step():
    """Psi(T) against a truncation-free complex step.

    The reduced RHS uses only ``x @ x`` and ``sqrt``, never ``abs`` or ``norm``,
    so it is complex-analytic and ``Im(M(m + i h))/h`` is the exact derivative.
    """
    _, mono, _, psi_t, _ = ms.integrate_frozen_chart(P_SHORT, MASSES, rtol=1e-13, atol=1e-15)
    step = 1e-30
    for axis in range(2):
        pert = [complex(x) for x in MASSES]
        pert[axis] += 1j * step
        _, mono_c = ms.flow_and_monodromy(P_SHORT, pert, rtol=1e-13, atol=1e-15)
        assert np.allclose(np.imag(mono_c) / step, psi_t[axis], rtol=1e-8, atol=1e-9)
        for mode in ms.EVENT_MODES:
            exact = float(np.imag(ms.event_from_monodromy(mono_c, mode)) / step)
            analytic = float(np.trace(ms.event_weight(mono, mode) @ psi_t[axis]))
            assert analytic == pytest.approx(exact, rel=1e-7, abs=1e-7)


# --------------------------------------------------------------------------- #
# Negative controls: these must FAIL when a mass coefficient is wrong
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("axis", [0, 1])
def test_a_wrong_mass_coefficient_in_dfdm_is_detected(monkeypatch, axis):
    """Scale one column of ``df/dm`` by 1.05 and the flow-sensitivity check breaks.

    A 5% coefficient error is the signature of confusing e.g. ``-g(d) - g(q1)``
    with ``-g(d) - m1 g(q1)`` at these mass values.  Without this control the
    agreement test above could be passing for the wrong reason.
    """
    truth = ms.field_mass_partial(Z, MASSES)
    assert np.linalg.norm(truth[:, axis]) > 0.1

    original = ms.field_mass_partial

    def wrong(z, masses):
        out = np.array(original(z, masses))
        out[:, axis] *= 1.05
        return out

    monkeypatch.setattr(ms, "field_mass_partial", wrong)
    _, _, s_bad, _, _ = ms.integrate_frozen_chart(P_SHORT, MASSES, level=1)
    monkeypatch.undo()
    _, _, s_good, _, _ = ms.integrate_frozen_chart(P_SHORT, MASSES, level=1)

    h = 1e-7
    hi, lo = list(MASSES), list(MASSES)
    hi[axis] += h
    lo[axis] -= h
    fd = (_frozen_chart_flow(P_SHORT, hi) - _frozen_chart_flow(P_SHORT, lo)) / (2 * h)
    assert np.allclose(fd, s_good[axis], rtol=1e-6, atol=1e-8)
    assert not np.allclose(fd, s_bad[axis], rtol=1e-6, atol=1e-8)


def test_dropping_the_chart_gauge_term_is_detected(monkeypatch):
    """Setting ``dc/dm = 0`` (the easy mistake) must break the same check."""
    monkeypatch.setattr(ms, "chart_mass_tangent", lambda p, masses: np.zeros((8, 2)))
    _, _, s_bad, _, _ = ms.integrate_frozen_chart(P_SHORT, MASSES, level=1)
    monkeypatch.undo()
    h = 1e-7
    for axis in range(2):
        hi, lo = list(MASSES), list(MASSES)
        hi[axis] += h
        lo[axis] -= h
        fd = (_frozen_chart_flow(P_SHORT, hi) - _frozen_chart_flow(P_SHORT, lo)) / (2 * h)
        assert not np.allclose(fd, s_bad[axis], rtol=1e-6, atol=1e-8)


def test_a_wrong_mass_coefficient_in_dAdm_is_detected(monkeypatch):
    """Scale ``dA/dm1`` by 1.05; the complex-step monodromy check must break."""
    original = ms.jacobian_mass_partial
    monkeypatch.setattr(
        ms, "jacobian_mass_partial", lambda z, m: np.array(original(z, m)) * np.array([[[1.05]], [[1.0]]])
    )
    _, _, _, psi_bad, _ = ms.integrate_frozen_chart(P_SHORT, MASSES, rtol=1e-13, atol=1e-15)
    monkeypatch.undo()
    step = 1e-30
    pert = [complex(x) for x in MASSES]
    pert[0] += 1j * step
    _, mono_c = ms.flow_and_monodromy(P_SHORT, pert, rtol=1e-13, atol=1e-15)
    assert not np.allclose(np.imag(mono_c) / step, psi_bad[0], rtol=1e-6, atol=1e-8)


# --------------------------------------------------------------------------- #
# Guards
# --------------------------------------------------------------------------- #
def test_corrected_sample_refuses_an_unconverged_node():
    """A stencil node that does not converge must raise, not return quietly.

    Silently accepting a stalled Newton is how a large-h mass stencil reports a
    derivative that is wrong by four orders of magnitude with no visible symptom.
    """
    with pytest.raises(RuntimeError):
        ms.corrected_sample(
            (BRANCH_MASSES[0] + 0.05, BRANCH_MASSES[1], 1.0), BRANCH_P, maxiter=6
        )


def test_unsupported_event_mode_is_rejected():
    with pytest.raises(ValueError):
        ms.event_weight(np.eye(8), "not_a_mode")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# End-to-end: the quantity the Julia stencils compute
# --------------------------------------------------------------------------- #
def test_total_event_mass_derivative_matches_richardson_stencil():
    """dG-/dm1 along the corrected family: sensitivities vs a corrected stencil.

    This is the whole point of the module -- it must agree with the finite
    difference it replaces, at a point where the derivative is O(10) rather than
    O(0) so the comparison has signal.
    """
    kw = dict(rtol=1e-12, atol=1e-14)
    corrected = ms.corrected_sample(BRANCH_MASSES, BRANCH_P, **kw)
    assert corrected.closure_norm < 1e-10
    sens = ms.mass_sensitivity(corrected.p, BRANCH_MASSES, **kw)
    # The 8x4 closure system is overdetermined but consistent; if it were not,
    # dp/dm -- and therefore every total derivative -- would be meaningless.
    assert sens.dp_lstsq_relative_residual < 1e-9
    assert sens.closure_jacobian_singular_values[-1] > 1e-3

    analytic = sens.d_events_dm["minus_one"][0]
    assert abs(analytic) > 1.0
    rich, _, _ = ms.richardson_event_derivative(
        BRANCH_MASSES, corrected.p, "minus_one", 0, 2e-4, **kw
    )
    assert rich == pytest.approx(analytic, rel=1e-5)
