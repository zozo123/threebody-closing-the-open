import numpy as np

from threebody_atlas.generic_periodic import generic_periodic_residual


def test_equilateral_lagrange_orbit_closes_in_generic_gauge() -> None:
    root3 = np.sqrt(3.0)
    # Equal masses, unit side length.  The triangle rotates rigidly with
    # omega=sqrt(3), and the chosen phase has r2-r3=(1,0), exactly matching the
    # generic corrector's scale/rotation gauge without imposing a Li velocity
    # or collinearity ansatz.
    z0 = np.asarray(
        [
            0.5,
            root3 / 2.0,
            1.0,
            0.0,
            -1.5,
            root3 / 2.0,
            0.0,
            root3,
        ],
        dtype=float,
    )
    period = 2.0 * np.pi / root3
    residual = generic_periodic_residual(
        np.concatenate((z0, [period])),
        (1.0, 1.0, 1.0),
        z0,
        rtol=1e-11,
        atol=1e-13,
    )
    assert np.linalg.norm(residual[:8]) < 2e-9
    assert np.linalg.norm(residual[8:]) < 1e-12
