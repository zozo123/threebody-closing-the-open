"""Metamorphic physics tests: properties that hold whatever the answer is.

Every other numerical check in this repository is anchored to an artifact
somebody produced.  These are not.  They compare two runs of the shipped
dynamics against each other, so they stay meaningful even if every published
number turned out to be wrong -- and they are the detectors the mutation
harness (``scripts/mutation_harness.py``) points at a corrupted force law.

The residual actually achieved by each property is asserted here and printed by
``python -m threebody_atlas.metamorphic``.
"""
from __future__ import annotations

import numpy as np
import pytest

from threebody_atlas import metamorphic
from threebody_atlas.cli import FIGURE_EIGHT
from threebody_atlas.dynamics import total_energy


def _check(residuals: list[metamorphic.Residual]) -> None:
    assert residuals, "a metamorphic property produced no residuals at all"
    violated = [item.describe() for item in residuals if not item.passed]
    assert not violated, "metamorphic property violated:\n" + "\n".join(violated)


def test_permutation_covariance() -> None:
    """Relabel bodies and masses together; the trajectory must permute."""
    _check(metamorphic.permutation_residuals())


def test_translation_invariance() -> None:
    _check(metamorphic.translation_residuals())


def test_galilean_invariance() -> None:
    _check(metamorphic.galilean_residuals())


def test_rotation_covariance() -> None:
    _check(metamorphic.rotation_residuals())


def test_time_reversal() -> None:
    _check(metamorphic.time_reversal_residuals())


def test_newtonian_similarity_of_trajectories() -> None:
    _check(metamorphic.similarity_trajectory_residuals())


def test_newtonian_similarity_leaves_floquet_multipliers_invariant() -> None:
    """The sharp one: the reduced spectrum is dimensionless.

    Under r -> s r, t -> s^{3/2} t, v -> s^{-1/2} v the reduced monodromy is
    conjugated by diag(s I_4, s^{-1/2} I_4).  Multipliers, alpha, beta and the
    discriminant are therefore invariant, and the conjugacy itself is checked
    explicitly so that "the eigenvalues happen to line up" is not enough.
    """
    _check(
        metamorphic.similarity_multiplier_residuals(
            FIGURE_EIGHT.initial_state, FIGURE_EIGHT.masses, FIGURE_EIGHT.period
        )
    )


def test_tangent_map_is_covariant_across_the_three_formulations() -> None:
    """12D Cartesian vs 8D reduced vs canonical Jacobi: M_B ~ T M_A T^{-1}."""
    _check(
        metamorphic.coordinate_covariance_residuals(
            FIGURE_EIGHT.initial_state, FIGURE_EIGHT.masses, FIGURE_EIGHT.period
        )
    )


@pytest.fixture(scope="module")
def every_residual() -> list[metamorphic.Residual]:
    """The full aggregate, computed once (each property costs a few integrations)."""
    return metamorphic.all_residuals()


def test_every_metamorphic_residual_is_reported_and_bounded(
    every_residual: list[metamorphic.Residual],
) -> None:
    """The aggregate the harness and CI both consume."""
    assert len(every_residual) >= 40
    _check(every_residual)


@pytest.mark.parametrize(
    ("prop", "bound"),
    sorted(metamorphic.OBSERVED.items()),
)
def test_observed_residual_table_is_not_stale(
    prop: str, bound: float, every_residual: list[metamorphic.Residual]
) -> None:
    """The documented residuals must stay documented.

    ``metamorphic.OBSERVED`` records what this dynamics actually achieved.  If a
    property silently degrades by more than a decade the table is wrong and the
    tolerances above stop being "integrator noise" gates, so fail loudly rather
    than let the documentation rot.
    """
    residuals = [item for item in every_residual if item.prop.startswith(prop)]
    assert residuals, f"no residuals recorded for {prop!r}"
    worst = max(item.residual for item in residuals)
    assert worst <= 10.0 * bound, (
        f"{prop} residual {worst:.3e} is more than 10x the documented {bound:.1e}; "
        "update metamorphic.OBSERVED after understanding why"
    )


def test_conservation_laws_hold_on_the_metamorphic_fixture() -> None:
    """Energy and angular momentum on the same generic unequal-mass fixture.

    Conservation is not a metamorphic property -- it needs no second run -- but
    it is the cheapest independent detector of a broken force law, so it lives
    next to them and is exercised on the same state.
    """
    from threebody_atlas.dynamics import integrate_orbit

    state = np.asarray(metamorphic.GENERIC_STATE, dtype=float)
    masses = np.asarray(metamorphic.GENERIC_MASSES, dtype=float)
    orbit = integrate_orbit(state, masses, metamorphic.GENERIC_TIME, **metamorphic.TIGHT)
    energy_drift = abs(orbit.energy_final - orbit.energy_initial)
    momentum_drift = abs(
        orbit.angular_momentum_final - orbit.angular_momentum_initial
    )
    # Observed on this fixture: energy 2.1e-11, angular momentum 1.6e-14,
    # linear momentum 1.3e-14.  The bounds sit ~50x above each.
    assert energy_drift < 1e-9, f"energy drift {energy_drift:.3e}"
    assert momentum_drift < 1e-12, f"angular momentum drift {momentum_drift:.3e}"
    # Total linear momentum is exactly conserved by the equations of motion.
    linear = np.sum(masses[:, None] * orbit.final_state[6:].reshape(3, 2), axis=0)
    linear0 = np.sum(masses[:, None] * state[6:].reshape(3, 2), axis=0)
    assert np.max(np.abs(linear - linear0)) < 1e-12
    assert np.isfinite(total_energy(orbit.final_state, masses))
