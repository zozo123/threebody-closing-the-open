"""Cancellation-free structural diagnostics for ±1 Floquet events.

The historical event polynomials are algebraically convenient but can lose many
digits because they subtract trace invariants of size O(1..10^2).  The defining
linear-algebra statements are simpler:

* ``-1`` event: ``M + I`` loses rank;
* ``+1`` event: ``M - I`` gains nullity beyond the neutral symmetry directions.

For the translation-reduced strict periodic three-body orbit used by the atlas,
the time/energy and rotation/angular-momentum neutral multipliers occur in two
Jordan chains at +1.  Generically ``M-I`` therefore has two tiny singular
values (geometric nullity two).  A physical +1 event introduces at least one
additional null direction, so the third-smallest singular value is the useful
rank-jump observable.

These quantities are *structural/conditioning diagnostics*, not replacement
publication gates.  Independent arbitrary-precision verification remains the
claim path.  They are especially useful for locating roots without catastrophic
trace-polynomial cancellation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class RankEventDiagnostics:
    minus_one_det_scaled: float
    minus_one_sigma_min: float
    plus_one_extra_sigma: float
    plus_one_singular_values_ascending: tuple[float, ...]
    minus_one_singular_values_ascending: tuple[float, ...]
    neutral_geometric_nullity: int


def _matrix(monodromy: Array) -> Array:
    matrix = np.asarray(monodromy, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("monodromy must be a square matrix")
    if matrix.shape[0] < 4:
        raise ValueError("monodromy dimension is too small for the Floquet diagnostics")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("monodromy contains non-finite values")
    return matrix


def minus_one_rank_event(monodromy: Array) -> tuple[float, float, tuple[float, ...]]:
    """Return direct ``det(M+I)`` event and rank-loss conditioning.

    The determinant is divided by 16 for comparability with the atlas's exact
    neutral-factor normalization: when the four algebraic neutral multipliers
    are exactly +1, ``det(M+I)/16`` equals the nontrivial ``G_minus`` factor.
    The zero set itself does not depend on this scale.
    """
    matrix = _matrix(monodromy)
    shifted = matrix + np.eye(matrix.shape[0])
    sign, logabs = np.linalg.slogdet(shifted)
    if sign == 0.0:
        direct = 0.0
    else:
        # Avoid overflow in diagnostics while preserving the sign.  Scientific
        # root localization only uses values in finite bracketing corridors.
        if float(logabs) > math.log(np.finfo(float).max):
            direct = math.copysign(float("inf"), float(sign))
        else:
            direct = float(sign * math.exp(float(logabs)) / 16.0)
    singular = tuple(float(x) for x in np.sort(np.linalg.svd(shifted, compute_uv=False)))
    return direct, singular[0], singular


def plus_one_rank_jump(
    monodromy: Array,
    *,
    neutral_geometric_nullity: int = 2,
) -> tuple[float, tuple[float, ...]]:
    """Return the first singular value beyond the neutral +1 null directions.

    On the regular translation-reduced atlas sheet, two symmetry generators give
    geometric nullity two for ``M-I``.  A physical +1 event drives the next
    singular value to zero.  The parameter is explicit so callers auditing a
    different reduction cannot silently inherit the atlas assumption.
    """
    matrix = _matrix(monodromy)
    n = int(neutral_geometric_nullity)
    if n < 0 or n >= matrix.shape[0]:
        raise ValueError("neutral_geometric_nullity must index a singular value")
    singular = tuple(
        float(x) for x in np.sort(np.linalg.svd(matrix - np.eye(matrix.shape[0]), compute_uv=False))
    )
    return singular[n], singular


def rank_event_diagnostics(
    monodromy: Array,
    *,
    neutral_geometric_nullity: int = 2,
) -> RankEventDiagnostics:
    direct_minus, sigma_minus, minus_singular = minus_one_rank_event(monodromy)
    extra_plus, plus_singular = plus_one_rank_jump(
        monodromy,
        neutral_geometric_nullity=neutral_geometric_nullity,
    )
    return RankEventDiagnostics(
        minus_one_det_scaled=direct_minus,
        minus_one_sigma_min=sigma_minus,
        plus_one_extra_sigma=extra_plus,
        plus_one_singular_values_ascending=plus_singular,
        minus_one_singular_values_ascending=minus_singular,
        neutral_geometric_nullity=int(neutral_geometric_nullity),
    )
