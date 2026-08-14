from __future__ import annotations

import numpy as np

from threebody_atlas.canonical_jacobi import (
    full_to_jacobi,
    jacobi_to_full_com,
    rhs_and_jacobian,
    symplectic_matrix,
)


def test_jacobi_roundtrip_preserves_reduced_coordinates() -> None:
    masses = np.asarray([0.8, 0.756, 1.0])
    full = np.asarray(
        [0.1, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.0, -0.2, 0.0, -0.4088]
    )
    z = full_to_jacobi(full, masses)
    rebuilt = jacobi_to_full_com(z, masses)
    np.testing.assert_allclose(full_to_jacobi(rebuilt, masses), z, rtol=1e-13, atol=1e-13)


def test_canonical_linearization_is_hamiltonian() -> None:
    masses = np.asarray([0.8, 0.756, 1.0])
    z = np.asarray([0.9, 0.0, -0.4, 0.2, 0.1, 0.7, -0.2, -0.1])
    _, a = rhs_and_jacobian(z, masses)
    j = symplectic_matrix()
    # Hamiltonian linearization: A^T J + J A = 0.
    np.testing.assert_allclose(a.T @ j + j @ a, 0.0, rtol=1e-12, atol=1e-12)
