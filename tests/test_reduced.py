import numpy as np

from threebody_atlas.cli import FIGURE_EIGHT
from threebody_atlas.reduced import (
    full_to_reduced,
    reconstruction_matrix,
    reduced_to_full,
    reduction_matrix,
)


def test_reduction_and_reconstruction_are_inverse_on_reduced_space():
    masses = np.array([0.8, 0.9, 1.0])
    assert np.allclose(reduction_matrix() @ reconstruction_matrix(masses), np.eye(8))


def test_figure_eight_roundtrip_on_com_zero_manifold():
    masses = np.asarray(FIGURE_EIGHT.masses)
    state = np.asarray(FIGURE_EIGHT.initial_state)
    rebuilt = reduced_to_full(full_to_reduced(state), masses)
    assert np.allclose(rebuilt, state, atol=1e-14)
