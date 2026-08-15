from __future__ import annotations

import numpy as np

from threebody_atlas.critical_geometry import (
    continuation_scales,
    critical_tangent,
    generic_projection_fold,
    projection_fold_indicator,
)


def test_critical_tangent_recovers_known_null_direction() -> None:
    tangent = np.asarray([1.0, -2.0, 0.5, 3.0, 0.2, -0.4])
    tangent /= np.linalg.norm(tangent)
    # Project random rows perpendicular to the known tangent; five independent
    # rows cut out a one-dimensional nullspace in six continuation variables.
    rng = np.random.default_rng(7)
    rows = []
    for _ in range(5):
        row = rng.normal(size=6)
        row -= np.dot(row, tangent) * tangent
        rows.append(row)
    jac = np.asarray(rows)
    got = critical_tangent(jac, reference=tangent)
    assert abs(np.dot(got.physical, tangent)) > 1.0 - 1e-11
    assert got.null_residual < 1e-11


def test_projection_fold_indicator_uses_mass_tangent_component() -> None:
    base = np.eye(6)[:5]
    # Null direction is the sixth coordinate: dm1=0, dm2!=0.
    got = critical_tangent(base)
    assert abs(projection_fold_indicator(got, "m1")) < 1e-14
    assert abs(projection_fold_indicator(got, "m2")) > 0.9


def test_generic_fold_screen_requires_oriented_sign_change() -> None:
    # Construct rank-five Jacobians with prescribed physical null directions.
    def from_tangent(t):
        t = np.asarray(t, dtype=float)
        t /= np.linalg.norm(t)
        rng = np.random.default_rng(int(abs(t[4]) * 1000) + 3)
        rows = []
        for _ in range(5):
            r = rng.normal(size=6)
            r -= np.dot(r, t) * t
            rows.append(r)
        return critical_tangent(np.asarray(rows), reference=t)

    before = from_tangent([0.1, 0.0, 0.0, 0.0, 0.2, 0.8])
    after = from_tangent([0.1, 0.0, 0.0, 0.0, -0.2, 0.8])
    assert generic_projection_fold(before, after, parameter="m1")


def test_continuation_scales_are_positive() -> None:
    scales = continuation_scales(np.zeros(6))
    assert np.all(scales > 0.0)
