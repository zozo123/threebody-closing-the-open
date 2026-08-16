"""Metamorphic properties of the planar Newtonian three-body dynamics.

A metamorphic property is a relation between two runs of the *same* code that
must hold whatever the right answer is.  It needs no published reference orbit,
no oracle and no ground truth -- which is exactly why it survives when every
other check in this repository depends on an artifact somebody produced.

Every residual here is computed against the dynamics that actually ships in
:mod:`threebody_atlas` (``dynamics``, ``variational``, ``reduced``,
``canonical_jacobi``).  Nothing is reimplemented: a metamorphic test that
reimplements the physics only checks the reimplementation.

The functions return :class:`Residual` records rather than asserting, so that

* ``tests/test_metamorphic_physics.py`` can assert on them, and
* ``scripts/mutation_harness.py`` can use them as a *detector* -- run them
  against a deliberately corrupted copy of the source tree and see whether the
  property notices.

Tolerances.  Each residual carries the tolerance it is judged against.  The
tolerances are set roughly two orders of magnitude above the residual actually
observed on a DOP853 run at ``rtol=1e-12, atol=1e-14`` (recorded in
``OBSERVED`` below), so they are integrator-tolerance gates and not fitted
thresholds.  They are *not* project numerical gates and must never be confused
with the frozen |event| <= 2e-8 / closure <= 1e-7 gates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from . import canonical_jacobi as jacobi
from .dynamics import acceleration, integrate_orbit
from .reduced import compute_reduced_floquet, reconstruction_matrix, reduction_matrix
from .variational import compute_floquet

Array = np.ndarray

# ---------------------------------------------------------------------------
# Fixtures.  Deliberately generic: unequal masses, no reflection symmetry, no
# zero components, so that a sign flip or a swapped mass coefficient cannot
# cancel out by accident.
# ---------------------------------------------------------------------------
GENERIC_MASSES: tuple[float, float, float] = (0.83, 1.07, 1.0)
GENERIC_STATE: tuple[float, ...] = (
    -0.62, 0.31,
    0.55, -0.44,
    0.11, 0.72,
    0.34, -0.21,
    -0.18, 0.41,
    -0.15, -0.19,
)
GENERIC_TIME = 1.7

# Screening-grade integration is not good enough to separate a metamorphic
# violation from integrator noise, so these runs are tightened one decade
# beyond the library defaults.
TIGHT = {"rtol": 1e-12, "atol": 1e-14}

# Residuals observed locally (numpy 2.5.2 / scipy 1.18.0, DOP853, TIGHT):
OBSERVED = {
    "permutation": 7.7e-12,
    "translation": 1.2e-10,
    "galilean": 9.7e-11,
    "rotation": 1.5e-10,
    "time_reversal": 5.4e-11,
    "similarity_trajectory": 1.1e-10,
    "similarity_multipliers": 2.0e-8,
    # Dominated by the spectrum comparison between the reduced and the Jacobi
    # charts (8.5e-8); the matrix-level intertwining residuals are ~1e-9.
    "coordinate_covariance": 8.5e-8,
}

TRAJECTORY_TOLERANCE = 1e-8
MULTIPLIER_TOLERANCE = 1e-6
TANGENT_MAP_TOLERANCE = 1e-6


@dataclass(frozen=True)
class Residual:
    """One metamorphic check: what was compared, how far off it was."""

    prop: str
    case: str
    residual: float
    tolerance: float
    scale: float = 1.0

    @property
    def passed(self) -> bool:
        return bool(np.isfinite(self.residual) and self.residual <= self.tolerance)

    def describe(self) -> str:
        verdict = "ok " if self.passed else "FAIL"
        return (
            f"{verdict} {self.prop:<22} {self.case:<28} "
            f"residual={self.residual:.3e} tol={self.tolerance:.1e} scale={self.scale:.3e}"
        )


def _state(state: Sequence[float] | Array) -> Array:
    return np.asarray(state, dtype=float)


def _evolve(state: Sequence[float] | Array, masses: Sequence[float] | Array, span: float) -> Array:
    """Integrate the shipped 12D dynamics and return the final state."""
    return integrate_orbit(_state(state), np.asarray(masses, dtype=float), span, **TIGHT).final_state


def _blockwise(matrix2: Array) -> Array:
    """Lift a 2x2 planar map to the 12D state by applying it to every body vector."""
    out = np.zeros((12, 12), dtype=float)
    for block in range(6):
        out[2 * block : 2 * block + 2, 2 * block : 2 * block + 2] = matrix2
    return out


def permutation_matrix(order: Sequence[int]) -> Array:
    """12x12 relabelling matrix sending body ``order[k]`` to slot ``k``."""
    out = np.zeros((12, 12), dtype=float)
    for new, old in enumerate(order):
        out[2 * new : 2 * new + 2, 2 * old : 2 * old + 2] = np.eye(2)
        out[6 + 2 * new : 6 + 2 * new + 2, 6 + 2 * old : 6 + 2 * old + 2] = np.eye(2)
    return out


def rotation_matrix(angle: float) -> Array:
    cos, sin = np.cos(angle), np.sin(angle)
    return np.asarray([[cos, -sin], [sin, cos]], dtype=float)


# ---------------------------------------------------------------------------
# 1. Permutation covariance
# ---------------------------------------------------------------------------
PERMUTATIONS: tuple[tuple[int, int, int], ...] = (
    (1, 0, 2),
    (2, 0, 1),
    (0, 2, 1),
    (2, 1, 0),
    (1, 2, 0),
)


def permutation_residuals() -> list[Residual]:
    """Relabelling bodies (and their masses) must permute the trajectory.

    This is the property that pins the *pairing* of a mass with its body.  A
    force law that uses ``masses[i]`` where it should use ``masses[j]`` is
    invisible to energy conservation and to every symmetry below, but it breaks
    here as soon as the three masses are distinct.
    """
    masses = np.asarray(GENERIC_MASSES, dtype=float)
    base_state = _state(GENERIC_STATE)
    base = _evolve(base_state, masses, GENERIC_TIME)
    out: list[Residual] = []

    # Instantaneous acceleration first: exact, no integrator noise at all.
    positions = base_state[:6].reshape(3, 2)
    base_acc = acceleration(positions, masses)
    for order in PERMUTATIONS:
        permuted_acc = acceleration(positions[list(order)], masses[list(order)])
        out.append(
            Residual(
                "permutation",
                f"acceleration {order}",
                float(np.max(np.abs(permuted_acc - base_acc[list(order)]))),
                1e-13,
                float(np.max(np.abs(base_acc))),
            )
        )

    for order in PERMUTATIONS:
        matrix = permutation_matrix(order)
        got = _evolve(matrix @ base_state, masses[list(order)], GENERIC_TIME)
        out.append(
            Residual(
                "permutation",
                f"trajectory {order}",
                float(np.max(np.abs(got - matrix @ base))),
                TRAJECTORY_TOLERANCE,
                float(np.max(np.abs(base))),
            )
        )
    return out


# ---------------------------------------------------------------------------
# 2. Translation invariance
# ---------------------------------------------------------------------------
def translation_residuals() -> list[Residual]:
    """r_i -> r_i + c must translate the whole trajectory by c."""
    masses = np.asarray(GENERIC_MASSES, dtype=float)
    base_state = _state(GENERIC_STATE)
    base = _evolve(base_state, masses, GENERIC_TIME)
    out: list[Residual] = []
    for shift in ([0.3, -0.7], [10.0, 4.0]):
        moved = base_state.copy()
        moved[:6] += np.tile(shift, 3)
        expected = base.copy()
        expected[:6] += np.tile(shift, 3)
        out.append(
            Residual(
                "translation",
                f"c={shift}",
                float(np.max(np.abs(_evolve(moved, masses, GENERIC_TIME) - expected))),
                TRAJECTORY_TOLERANCE,
                float(np.max(np.abs(expected))),
            )
        )
    return out


# ---------------------------------------------------------------------------
# 3. Galilean invariance
# ---------------------------------------------------------------------------
def galilean_residuals() -> list[Residual]:
    """v_i -> v_i + u must add a uniform drift u*t to the positions."""
    masses = np.asarray(GENERIC_MASSES, dtype=float)
    base_state = _state(GENERIC_STATE)
    base = _evolve(base_state, masses, GENERIC_TIME)
    out: list[Residual] = []
    for boost in ([0.25, -0.4], [2.0, 1.0]):
        moved = base_state.copy()
        moved[6:] += np.tile(boost, 3)
        expected = base.copy()
        expected[:6] += np.tile(np.asarray(boost, dtype=float) * GENERIC_TIME, 3)
        expected[6:] += np.tile(boost, 3)
        out.append(
            Residual(
                "galilean",
                f"u={boost}",
                float(np.max(np.abs(_evolve(moved, masses, GENERIC_TIME) - expected))),
                TRAJECTORY_TOLERANCE,
                float(np.max(np.abs(expected))),
            )
        )
    return out


# ---------------------------------------------------------------------------
# 4. Rotation covariance
# ---------------------------------------------------------------------------
def rotation_residuals() -> list[Residual]:
    """(r_i, v_i) -> (R r_i, R v_i) must rotate the trajectory rigidly."""
    masses = np.asarray(GENERIC_MASSES, dtype=float)
    base_state = _state(GENERIC_STATE)
    base = _evolve(base_state, masses, GENERIC_TIME)
    out: list[Residual] = []
    for angle in (0.37, 2.4, -1.13):
        lifted = _blockwise(rotation_matrix(angle))
        got = _evolve(lifted @ base_state, masses, GENERIC_TIME)
        out.append(
            Residual(
                "rotation",
                f"theta={angle}",
                float(np.max(np.abs(got - lifted @ base))),
                TRAJECTORY_TOLERANCE,
                float(np.max(np.abs(base))),
            )
        )
    # A reflection is NOT a symmetry direction we rely on, but the planar
    # Newtonian field is equivariant under O(2), so include one to make sure the
    # covariance is not accidentally special to proper rotations.
    reflect = _blockwise(np.asarray([[1.0, 0.0], [0.0, -1.0]]))
    out.append(
        Residual(
            "rotation",
            "reflection y->-y",
            float(np.max(np.abs(_evolve(reflect @ base_state, masses, GENERIC_TIME) - reflect @ base))),
            TRAJECTORY_TOLERANCE,
            float(np.max(np.abs(base))),
        )
    )
    return out


# ---------------------------------------------------------------------------
# 5. Time reversal
# ---------------------------------------------------------------------------
def time_reversal_residuals() -> list[Residual]:
    """(r, v, t) -> (r, -v, -t): running the flipped end state back returns home."""
    masses = np.asarray(GENERIC_MASSES, dtype=float)
    base_state = _state(GENERIC_STATE)
    base = _evolve(base_state, masses, GENERIC_TIME)
    flipped = base.copy()
    flipped[6:] *= -1.0
    returned = _evolve(flipped, masses, GENERIC_TIME)
    returned[6:] *= -1.0
    return [
        Residual(
            "time_reversal",
            f"span={GENERIC_TIME}",
            float(np.max(np.abs(returned - base_state))),
            TRAJECTORY_TOLERANCE,
            float(np.max(np.abs(base_state))),
        )
    ]


# ---------------------------------------------------------------------------
# 6. Newtonian similarity
# ---------------------------------------------------------------------------
SIMILARITY_SCALES: tuple[float, ...] = (0.4, 2.0, 7.3)


def similarity_trajectory_residuals() -> list[Residual]:
    """r -> s r, t -> s^{3/2} t, v -> s^{-1/2} v maps solutions to solutions."""
    masses = np.asarray(GENERIC_MASSES, dtype=float)
    base_state = _state(GENERIC_STATE)
    base = _evolve(base_state, masses, GENERIC_TIME)
    out: list[Residual] = []
    for scale in SIMILARITY_SCALES:
        scaled = base_state.copy()
        scaled[:6] *= scale
        scaled[6:] *= scale**-0.5
        expected = base.copy()
        expected[:6] *= scale
        expected[6:] *= scale**-0.5
        got = _evolve(scaled, masses, GENERIC_TIME * scale**1.5)
        out.append(
            Residual(
                "similarity_trajectory",
                f"s={scale}",
                float(np.max(np.abs(got - expected))),
                TRAJECTORY_TOLERANCE,
                float(np.max(np.abs(expected))),
            )
        )
    return out


def _sorted_multipliers(matrix: Array) -> Array:
    values = np.linalg.eigvals(matrix)
    return np.asarray(sorted(values, key=lambda z: (round(float(np.real(z)), 9), float(np.imag(z)))))


def similarity_multiplier_residuals(
    state: Sequence[float] | Array,
    masses: Sequence[float] | Array,
    period: float,
) -> list[Residual]:
    """Floquet MULTIPLIERS are dimensionless: scaling must not move them.

    This is the sharpest available test of the reduced spectrum, because the
    monodromy matrix itself is *not* invariant -- it is conjugated by
    diag(s I_4, s^{-1/2} I_4) -- while its eigenvalues, the trace invariants
    alpha and beta, and the discriminant all are.  A reduced-space bug that
    mixes position and velocity blocks changes the spectrum here even though
    every trajectory-level property above still passes.
    """
    state = _state(state)
    masses = np.asarray(masses, dtype=float)
    base = compute_reduced_floquet(state, masses, period)
    base_multipliers = _sorted_multipliers(base.monodromy)
    out: list[Residual] = []
    for scale in SIMILARITY_SCALES:
        scaled = state.copy()
        scaled[:6] *= scale
        scaled[6:] *= scale**-0.5
        result = compute_reduced_floquet(scaled, masses, period * scale**1.5)
        out.append(
            Residual(
                "similarity_multipliers",
                f"s={scale} multipliers",
                float(np.max(np.abs(_sorted_multipliers(result.monodromy) - base_multipliers))),
                MULTIPLIER_TOLERANCE,
                float(np.max(np.abs(base_multipliers))),
            )
        )
        out.append(
            Residual(
                "similarity_multipliers",
                f"s={scale} alpha",
                abs(result.alpha - base.alpha),
                MULTIPLIER_TOLERANCE,
                abs(base.alpha),
            )
        )
        out.append(
            Residual(
                "similarity_multipliers",
                f"s={scale} beta",
                abs(result.beta - base.beta),
                MULTIPLIER_TOLERANCE,
                abs(base.beta),
            )
        )
        out.append(
            Residual(
                "similarity_multipliers",
                f"s={scale} discriminant",
                abs(result.discriminant - base.discriminant),
                MULTIPLIER_TOLERANCE,
                abs(base.discriminant),
            )
        )
        # The monodromy itself must be conjugate, not equal.  Checking the
        # conjugation explicitly is what distinguishes "the spectrum happens to
        # agree" from "the tangent map really transformed correctly".
        gauge = np.diag(np.concatenate((np.full(4, scale), np.full(4, scale**-0.5))))
        conjugated = gauge @ base.monodromy @ np.linalg.inv(gauge)
        out.append(
            Residual(
                "similarity_multipliers",
                f"s={scale} monodromy conjugacy",
                float(np.max(np.abs(result.monodromy - conjugated))),
                TANGENT_MAP_TOLERANCE * max(1.0, float(np.max(np.abs(conjugated)))),
                float(np.max(np.abs(conjugated))),
            )
        )
    return out


# ---------------------------------------------------------------------------
# 7. Coordinate covariance of the tangent map
# ---------------------------------------------------------------------------
def jacobi_transforms(masses: Sequence[float] | Array) -> tuple[Array, Array]:
    """Matrix forms of ``full_to_jacobi`` (8x12) and ``jacobi_to_full_com`` (12x8).

    Both maps are linear in the state at fixed masses, so they are recovered
    exactly by applying the shipped functions to basis vectors.  Building them
    this way -- rather than transcribing the algebra a second time -- keeps the
    covariance test honest about what the library actually does.
    """
    masses = np.asarray(masses, dtype=float)
    to_jacobi = np.column_stack(
        [jacobi.full_to_jacobi(np.eye(12)[index], masses) for index in range(12)]
    )
    from_jacobi = np.column_stack(
        [jacobi.jacobi_to_full_com(np.eye(8)[index], masses) for index in range(8)]
    )
    return to_jacobi, from_jacobi


def coordinate_covariance_residuals(
    state: Sequence[float] | Array,
    masses: Sequence[float] | Array,
    period: float,
) -> list[Residual]:
    """The same physical orbit, three formulations, one tangent map.

    Let ``M_12`` be the 12D Cartesian monodromy, ``M_red`` the 8D
    centre-of-mass-reduced monodromy and ``M_jac`` the canonical Jacobi
    monodromy.  With ``R``/``P`` the reduction/reconstruction pair and
    ``TJ``/``PJ`` the Jacobi pair, the required relations are

        M_red  =  R  M_12 P          (intertwining onto the reduced chart)
        M_jac  =  TJ M_12 PJ         (intertwining onto the Jacobi chart)
        M_jac  =  T  M_red T^{-1}    with T = TJ P, a genuine 8x8 similarity

    plus equality of the spectra.  The third relation is the ``M_B ~ T M_A
    T^{-1}`` the task asks for; the first two are the projections that make it
    meaningful.  ``T`` here has condition number 3 on the figure-eight, so the
    similarity carries no hidden amplification.
    """
    state = _state(state)
    masses = np.asarray(masses, dtype=float)
    full = compute_floquet(state, masses, period)
    reduced = compute_reduced_floquet(state, masses, period)
    canonical = jacobi.compute_canonical_floquet(state, masses, period)

    reduce_matrix = reduction_matrix()
    rebuild = reconstruction_matrix(masses)
    to_jacobi, from_jacobi = jacobi_transforms(masses)
    transfer = to_jacobi @ rebuild

    scale_red = float(np.max(np.abs(reduced.monodromy)))
    scale_jac = float(np.max(np.abs(canonical.monodromy)))
    out = [
        Residual(
            "coordinate_covariance",
            "TJ @ PJ == I_8",
            float(np.max(np.abs(to_jacobi @ from_jacobi - np.eye(8)))),
            1e-12,
            1.0,
        ),
        Residual(
            "coordinate_covariance",
            "R @ P == I_8",
            float(np.max(np.abs(reduce_matrix @ rebuild - np.eye(8)))),
            1e-12,
            1.0,
        ),
        Residual(
            "coordinate_covariance",
            "M_red == R M_12 P",
            float(np.max(np.abs(reduce_matrix @ full.monodromy @ rebuild - reduced.monodromy))),
            TANGENT_MAP_TOLERANCE * max(1.0, scale_red),
            scale_red,
        ),
        Residual(
            "coordinate_covariance",
            "M_jac == TJ M_12 PJ",
            float(np.max(np.abs(to_jacobi @ full.monodromy @ from_jacobi - canonical.monodromy))),
            TANGENT_MAP_TOLERANCE * max(1.0, scale_jac),
            scale_jac,
        ),
        Residual(
            "coordinate_covariance",
            "M_jac == T M_red T^-1",
            float(
                np.max(
                    np.abs(
                        transfer @ reduced.monodromy @ np.linalg.inv(transfer)
                        - canonical.monodromy
                    )
                )
            ),
            TANGENT_MAP_TOLERANCE * max(1.0, scale_jac),
            scale_jac,
        ),
        Residual(
            "coordinate_covariance",
            "spectrum(M_red) == spectrum(M_jac)",
            float(
                np.max(
                    np.abs(
                        _sorted_multipliers(reduced.monodromy)
                        - _sorted_multipliers(canonical.monodromy)
                    )
                )
            ),
            MULTIPLIER_TOLERANCE,
            1.0,
        ),
        Residual(
            "coordinate_covariance",
            "cond(T)",
            0.0 if np.linalg.cond(transfer) < 1e3 else float(np.linalg.cond(transfer)),
            1e-12,
            float(np.linalg.cond(transfer)),
        ),
    ]
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def all_residuals(
    floquet_state: Sequence[float] | Array | None = None,
    floquet_masses: Sequence[float] | Array | None = None,
    floquet_period: float | None = None,
) -> list[Residual]:
    """Every metamorphic residual, in report order.

    The Floquet-based properties need a genuine periodic orbit; the caller
    supplies one (the CLI figure-eight regression fixture by default, imported
    lazily so that this module does not depend on the CLI).
    """
    if floquet_state is None or floquet_masses is None or floquet_period is None:
        from .cli import FIGURE_EIGHT

        floquet_state = FIGURE_EIGHT.initial_state
        floquet_masses = FIGURE_EIGHT.masses
        floquet_period = FIGURE_EIGHT.period
    checks: list[Callable[[], list[Residual]]] = [
        permutation_residuals,
        translation_residuals,
        galilean_residuals,
        rotation_residuals,
        time_reversal_residuals,
        similarity_trajectory_residuals,
        lambda: similarity_multiplier_residuals(floquet_state, floquet_masses, floquet_period),
        lambda: coordinate_covariance_residuals(floquet_state, floquet_masses, floquet_period),
    ]
    out: list[Residual] = []
    for check in checks:
        out.extend(check())
    return out


def report(residuals: Sequence[Residual]) -> str:
    lines = [item.describe() for item in residuals]
    failed = [item for item in residuals if not item.passed]
    lines.append("")
    lines.append(f"{len(residuals)} metamorphic residuals, {len(failed)} violated")
    return "\n".join(lines)


def main() -> int:
    """Print the residual table; exit non-zero when a property is violated.

    ``python -m threebody_atlas.metamorphic`` is the form the mutation harness
    runs, because it needs a detector it can point at a corrupted copy of the
    source tree without dragging in the whole test session.
    """
    residuals = all_residuals()
    print(report(residuals))
    return 1 if any(not item.passed for item in residuals) else 0


if __name__ == "__main__":  # pragma: no cover - exercised via the mutation harness
    raise SystemExit(main())
