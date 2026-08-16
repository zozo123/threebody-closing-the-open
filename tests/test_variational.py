from __future__ import annotations

import numpy as np

from threebody_atlas.dynamics import rhs
from threebody_atlas.variational import state_jacobian


def test_variational_jacobian_matches_centered_finite_difference_for_unequal_masses() -> None:
    masses = np.asarray([0.7, 1.1, 1.3], dtype=float)
    state = np.asarray(
        [
            -0.7,
            0.2,
            0.4,
            -0.6,
            1.1,
            0.9,
            0.3,
            -0.2,
            -0.1,
            0.4,
            0.2,
            -0.3,
        ],
        dtype=float,
    )
    analytic = state_jacobian(state, masses)
    numeric = np.empty_like(analytic)
    eps = 1e-7
    for column in range(state.size):
        delta = np.zeros_like(state)
        delta[column] = eps
        numeric[:, column] = (rhs(0.0, state + delta, masses) - rhs(0.0, state - delta, masses)) / (
            2.0 * eps
        )

    np.testing.assert_allclose(analytic, numeric, rtol=2e-7, atol=2e-8)
