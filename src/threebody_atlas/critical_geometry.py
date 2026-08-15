"""Differential geometry utilities for critical curves in continuation space.

A mass-plane stability boundary is not fundamentally a graph m2(m1).  It is a
one-dimensional curve embedded in the six-dimensional continuation chart

    y = (x1, v1, v2, T, m1, m2),

cut out by periodic closure plus one smooth Floquet event.  If J = dG/dy has
rank five, the curve tangent spans null(J).  Working with this nullspace makes
folds and branch geometry coordinate-aware and avoids nested finite differences
of a re-solved mass-plane graph.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class CriticalTangent:
    """Numerical tangent and conditioning diagnostics for a critical curve."""

    physical: Array
    scaled: Array
    singular_values: Array
    null_residual: float
    spectral_gap: float

    @property
    def dm1(self) -> float:
        return float(self.physical[4])

    @property
    def dm2(self) -> float:
        return float(self.physical[5])


def continuation_scales(y: Array) -> Array:
    """Use the same order-of-magnitude scaling as pseudo-arclength correction."""
    y = np.asarray(y, dtype=float)
    if y.shape != (6,):
        raise ValueError("continuation vector must have six components")
    floors = np.asarray([0.05, 0.5, 0.1, 1.0, 0.1, 0.1], dtype=float)
    return np.maximum(np.abs(y), floors)


def critical_tangent(
    jacobian: Array,
    *,
    scales: Array | None = None,
    reference: Array | None = None,
) -> CriticalTangent:
    """Extract the one-dimensional null direction of a critical residual Jacobian.

    ``jacobian`` is dG/dy.  We compute the SVD after changing variables to
    z=y/scales, i.e. dG/dz = (dG/dy) diag(scales).  This keeps the nullspace from
    being dominated by the very different natural magnitudes of T and masses.
    ``reference`` is a physical tangent used only to choose orientation.
    """
    j = np.asarray(jacobian, dtype=float)
    if j.ndim != 2 or j.shape[1] != 6:
        raise ValueError("critical Jacobian must have six columns")
    if scales is None:
        scale = np.ones(6, dtype=float)
    else:
        scale = np.asarray(scales, dtype=float)
        if scale.shape != (6,) or np.any(scale <= 0.0):
            raise ValueError("scales must contain six positive entries")

    j_scaled = j * scale[None, :]
    _u, singular, vh = np.linalg.svd(j_scaled, full_matrices=False)
    t_scaled = vh[-1].copy()
    t_physical = scale * t_scaled
    physical_norm = float(np.linalg.norm(t_physical))
    if physical_norm == 0.0:
        raise RuntimeError("critical null vector collapsed to zero")
    t_physical /= physical_norm
    t_scaled /= float(np.linalg.norm(t_scaled))

    if reference is not None:
        ref = np.asarray(reference, dtype=float)
        if ref.shape != (6,):
            raise ValueError("reference tangent must have six components")
        if float(np.dot(t_physical, ref)) < 0.0:
            t_physical *= -1.0
            t_scaled *= -1.0

    null_residual = float(np.linalg.norm(j @ t_physical))
    if singular.size >= 2:
        spectral_gap = float(singular[-2] / max(singular[-1], np.finfo(float).tiny))
    else:
        spectral_gap = float("inf")
    return CriticalTangent(
        physical=t_physical,
        scaled=t_scaled,
        singular_values=singular,
        null_residual=null_residual,
        spectral_gap=spectral_gap,
    )


def projection_fold_indicator(tangent: CriticalTangent, parameter: str = "m1") -> float:
    """Return the tangent component whose zero marks a projection fold."""
    if parameter == "m1":
        return tangent.dm1
    if parameter == "m2":
        return tangent.dm2
    raise ValueError("parameter must be 'm1' or 'm2'")


def generic_projection_fold(
    before: CriticalTangent,
    after: CriticalTangent,
    *,
    parameter: str = "m1",
    minimum_transverse_component: float = 1e-4,
) -> bool:
    """Screen a fold by an oriented sign change and nonzero transverse motion.

    This is a geometric *screen*, not a proof.  High-precision localization must
    additionally verify the critical equations and nondegenerate curvature.
    """
    a = projection_fold_indicator(before, parameter)
    b = projection_fold_indicator(after, parameter)
    if a == 0.0 or b == 0.0 or a * b > 0.0:
        return False
    transverse = (
        min(abs(before.dm2), abs(after.dm2))
        if parameter == "m1"
        else min(abs(before.dm1), abs(after.dm1))
    )
    return bool(transverse >= minimum_transverse_component)
