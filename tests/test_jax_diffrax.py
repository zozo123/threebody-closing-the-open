from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("diffrax")

from threebody_atlas.jax_diffrax import chart_state_jax, reduced_rhs_jax, rhs_jacobian
from threebody_atlas.liao_family import state_from_chart
from threebody_atlas.reduced import full_to_reduced, reduced_jacobian, reduced_rhs


@pytest.fixture(autouse=True)
def require_x64():
    if not jax.config.x64_enabled:
        pytest.skip("JAX x64 is required for accelerated scientific screening")


def test_jax_reduced_rhs_matches_numpy_path() -> None:
    state = np.asarray([0.8, 0.2, -0.4, 1.1, 0.3, -0.7, 0.8, 0.1], dtype=float)
    masses = np.asarray([0.8, 0.91, 1.0], dtype=float)
    got = np.asarray(reduced_rhs_jax(0.0, state, masses), dtype=float)
    expected = reduced_rhs(0.0, state, masses)
    np.testing.assert_allclose(got, expected, rtol=2e-13, atol=2e-13)


def test_jax_local_jacobian_matches_analytic_variational_path() -> None:
    state = np.asarray([0.8, 0.2, -0.4, 1.1, 0.3, -0.7, 0.8, 0.1], dtype=float)
    masses = np.asarray([0.8, 0.91, 1.0], dtype=float)
    got = rhs_jacobian(state, masses)
    expected = reduced_jacobian(state, masses)
    np.testing.assert_allclose(got, expected, rtol=2e-11, atol=2e-11)


def test_jax_chart_matches_published_family_chart() -> None:
    y = np.asarray([-0.135, 2.51, 0.319, 5.2, 0.8, 0.758], dtype=float)
    m3 = 1.0
    full = state_from_chart((y[4], y[5], m3), y[0], y[1], y[2])
    expected = full_to_reduced(full)
    got = np.asarray(chart_state_jax(y, m3), dtype=float)
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=2e-15)
