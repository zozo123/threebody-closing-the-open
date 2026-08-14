import mpmath as mp
import numpy as np

from threebody_atlas.cli import FIGURE_EIGHT
from threebody_atlas.high_precision_reduced import reduced_rhs_and_jacobian
from threebody_atlas.reduced import full_to_reduced, reduced_jacobian, reduced_rhs


def test_independent_mp_reduced_equations_match_float64_formulation():
    masses = np.asarray(FIGURE_EIGHT.masses, dtype=float)
    state = np.asarray(FIGURE_EIGHT.initial_state, dtype=float)
    z = full_to_reduced(state)
    mp_z = [mp.mpf(str(v)) for v in z]
    mp_m = [mp.mpf(str(v)) for v in masses]
    rhs_mp, jac_mp = reduced_rhs_and_jacobian(mp_z, mp_m)
    rhs_np = np.asarray([float(v) for v in rhs_mp])
    jac_np = np.asarray([[float(v) for v in row] for row in jac_mp])
    assert np.allclose(rhs_np, reduced_rhs(0.0, z, masses), rtol=1e-13, atol=1e-13)
    assert np.allclose(jac_np, reduced_jacobian(z, masses), rtol=1e-12, atol=1e-12)
