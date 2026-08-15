import numpy as np

from threebody_atlas.physical_floquet import (
    physical_trace_invariants,
    quotient_monodromy,
)


def _canonical_pair_rotation(matrix: np.ndarray, q: int, p: int, angle: float) -> None:
    c = np.cos(angle)
    s = np.sin(angle)
    matrix[q, q] = c
    matrix[q, p] = s
    matrix[p, q] = -s
    matrix[p, p] = c


def test_quotient_recovers_two_physical_elliptic_pairs() -> None:
    theta1 = 0.4
    theta2 = 1.1
    monodromy = np.eye(8)
    # Canonical ordering is q1,q2,q3,q4,p1,p2,p3,p4.  The first two q
    # directions span E; their conjugates are removed by E^omega.  The physical
    # quotient is therefore the two canonical pairs (q3,p3) and (q4,p4).
    _canonical_pair_rotation(monodromy, 2, 6, theta1)
    _canonical_pair_rotation(monodromy, 3, 7, theta2)
    e1 = np.eye(8)[0]
    e2 = np.eye(8)[1]

    matrix, form, basis, defect, leakage, isotropy, neutral_invariance = quotient_monodromy(
        monodromy,
        e1,
        e2,
    )
    assert matrix.shape == (4, 4)
    assert form.shape == (4, 4)
    assert basis.shape == (8, 4)
    assert defect < 1e-12
    assert leakage < 1e-12
    assert isotropy < 1e-12
    assert neutral_invariance < 1e-12

    multipliers = sorted(np.linalg.eigvals(matrix), key=lambda z: np.angle(z))
    expected = sorted(
        [
            np.exp(1j * theta1),
            np.exp(-1j * theta1),
            np.exp(1j * theta2),
            np.exp(-1j * theta2),
        ],
        key=lambda z: np.angle(z),
    )
    assert np.allclose(multipliers, expected, atol=2e-12, rtol=0.0)

    a, b, discriminant, roots = physical_trace_invariants(matrix)
    c1, c2 = np.cos(theta1), np.cos(theta2)
    assert abs(a - 2.0 * (c1 + c2)) < 2e-12
    assert abs(b - (2.0 + 4.0 * c1 * c2)) < 2e-12
    assert abs(discriminant - 4.0 * (c1 - c2) ** 2) < 2e-12
    assert np.allclose(sorted([root.real for root in roots]), sorted([2 * c1, 2 * c2]), atol=2e-12)


def test_mixed_plus_minus_one_vertex_is_exact_in_physical_invariants() -> None:
    monodromy = np.eye(8)
    _canonical_pair_rotation(monodromy, 2, 6, 0.0)
    _canonical_pair_rotation(monodromy, 3, 7, np.pi)
    matrix, *_ = quotient_monodromy(monodromy, np.eye(8)[0], np.eye(8)[1])
    a, b, discriminant, _ = physical_trace_invariants(matrix)

    plus_one = b - 2.0 * a + 2.0
    minus_one = b + 2.0 * a + 2.0
    assert abs(a) < 1e-12
    assert abs(b + 2.0) < 1e-12
    assert abs(plus_one) < 1e-12
    assert abs(minus_one) < 1e-12
    assert abs(discriminant - 16.0) < 1e-12
