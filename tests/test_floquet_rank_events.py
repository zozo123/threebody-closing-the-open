from __future__ import annotations

import numpy as np

from threebody_atlas.floquet_rank_events import (
    minus_one_rank_event,
    plus_one_rank_jump,
    rank_event_diagnostics,
)


def _jordan_plus_one() -> np.ndarray:
    # Two generic neutral +1 Jordan chains: algebraic multiplicity four but
    # geometric nullity two for M-I, matching the regular reduced orbit.
    return np.asarray(
        [
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def _synthetic_monodromy(lam: float, mu: float) -> np.ndarray:
    neutral = _jordan_plus_one()
    physical = np.diag([lam, 1.0 / lam, mu, 1.0 / mu])
    out = np.zeros((8, 8))
    out[:4, :4] = neutral
    out[4:, 4:] = physical
    return out


def test_minus_one_rank_event_vanishes_exactly_at_minus_one_pair() -> None:
    event, sigma, singular = minus_one_rank_event(_synthetic_monodromy(-1.0, 1.7))
    assert abs(event) < 1e-12
    assert sigma < 1e-12
    assert singular[0] < 1e-12


def test_minus_one_rank_event_is_not_confused_by_neutral_plus_one_chains() -> None:
    event, sigma, _singular = minus_one_rank_event(_synthetic_monodromy(1.3, 1.7))
    assert abs(event) > 1e-3
    assert sigma > 1e-3


def test_plus_one_rank_jump_skips_two_neutral_geometric_null_directions() -> None:
    generic, singular_generic = plus_one_rank_jump(_synthetic_monodromy(1.3, 1.7))
    critical, singular_critical = plus_one_rank_jump(_synthetic_monodromy(1.0, 1.7))
    assert singular_generic[0] < 1e-12
    assert singular_generic[1] < 1e-12
    assert generic > 1e-2
    assert critical < 1e-12
    assert singular_critical[2] < 1e-12


def test_plus_one_rank_jump_requires_explicit_valid_neutral_nullity() -> None:
    matrix = _synthetic_monodromy(1.3, 1.7)
    for invalid in (-1, 8):
        try:
            plus_one_rank_jump(matrix, neutral_geometric_nullity=invalid)
        except ValueError:
            pass
        else:  # pragma: no cover - defensive assertion
            raise AssertionError(f"neutral nullity {invalid} should fail")


def test_combined_diagnostics_are_consistent() -> None:
    matrix = _synthetic_monodromy(-1.0, 1.7)
    diag = rank_event_diagnostics(matrix)
    event, sigma, minus_singular = minus_one_rank_event(matrix)
    plus, plus_singular = plus_one_rank_jump(matrix)
    assert diag.minus_one_det_scaled == event
    assert diag.minus_one_sigma_min == sigma
    assert diag.minus_one_singular_values_ascending == minus_singular
    assert diag.plus_one_extra_sigma == plus
    assert diag.plus_one_singular_values_ascending == plus_singular
    assert diag.neutral_geometric_nullity == 2
