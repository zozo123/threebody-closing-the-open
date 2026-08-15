import numpy as np

from threebody_atlas.cli import FIGURE_EIGHT
from threebody_atlas.dynamics import center_of_mass, integrate_orbit


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
