import numpy as np

from threebody_atlas.cli import FIGURE_EIGHT
from threebody_atlas.dynamics import acceleration, center_of_mass, integrate_orbit


def test_figure_eight_closes_in_float64_screening():
    result = integrate_orbit(
        np.asarray(FIGURE_EIGHT.initial_state),
        np.asarray(FIGURE_EIGHT.masses),
        FIGURE_EIGHT.period,
        rtol=1e-11,
        atol=1e-13,
    )
    assert result.closure_norm < 2e-6
    assert abs(result.energy_final - result.energy_initial) < 1e-9
    assert abs(result.angular_momentum_final - result.angular_momentum_initial) < 1e-9


def test_figure_eight_center_of_mass_constraints():
    qcm, vcm = center_of_mass(
        np.asarray(FIGURE_EIGHT.initial_state), np.asarray(FIGURE_EIGHT.masses)
    )
    assert np.linalg.norm(qcm) < 1e-14
    assert np.linalg.norm(vcm) < 1e-14


def test_unequal_mass_acceleration_matches_newtonian_pair_sum_and_zero_net_force():
    positions = np.asarray([[-0.7, 0.2], [0.4, -0.6], [1.1, 0.9]], dtype=float)
    masses = np.asarray([0.7, 1.1, 1.3], dtype=float)
    got = acceleration(positions, masses)

    expected_body0 = np.zeros(2)
    for j in (1, 2):
        delta = positions[j] - positions[0]
        expected_body0 += masses[j] * delta / np.linalg.norm(delta) ** 3
    np.testing.assert_allclose(got[0], expected_body0, rtol=2e-15, atol=2e-15)

    # Internal Newtonian forces cancel pairwise even for unequal masses.
    net_force = np.sum(masses[:, None] * got, axis=0)
    np.testing.assert_allclose(net_force, 0.0, rtol=0.0, atol=2e-15)
