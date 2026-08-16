"""Conditioning metadata for every solve that produces a publication number.

A residual of 1e-25 means something entirely different when ``||J^+|| ~ 1e3``
than when it is ``~1e22``.  A solver residual is a *backward* error: it says the
computed point exactly solves a nearby problem.  What a reader needs is the
*forward* error -- how far the computed point can be from the true root -- and
that requires the conditioning of the Jacobian at the solution:

    ||dx||  <~  ||J^+|| ||F(x)||  =  ||F(x)|| / sigma_min(J).

So no solve in this package should report a residual without also reporting
``sigma_min``, ``sigma_max``, ``kappa_2 = sigma_max/sigma_min`` and the derived
displacement bound.  These are metadata, never gates: nothing here rejects a
point, and none of the frozen numerical gates are read or written from this
module.

Two shapes are supported.

``condition_report`` handles a general (possibly rectangular, possibly
rank-deficient) Jacobian via its singular values.  For an overdetermined
least-squares solve ``||J^+||_2 = 1/sigma_min`` still bounds the sensitivity of
the solution to a perturbation of the residual, so the same formula is used and
the report records the shape so a reader can tell which case they are in.

``scalar_condition_report`` handles the one-dimensional in-bracket solves used
by the critical-root census, where ``J`` degenerates to the scalar slope
``d(event)/d(parameter)``.  There ``kappa_2`` is identically 1 and the
informative number is the slope itself: an event residual of 2e-8 against a
slope of 1e2 localizes the parameter to 2e-10, while the same residual against a
slope of 1e-3 localizes nothing at all.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class SolveConditioning:
    """Singular-value conditioning of one solve, plus its forward-error bound."""

    rows: int
    cols: int
    sigma_max: float
    sigma_min: float
    kappa_2: float
    numerical_rank: int
    rank_deficient: bool
    residual_norm: float
    #: ``||J^+|| ||F||`` -- how far the computed point may sit from the true root.
    displacement_bound: float
    singular_values: tuple[float, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "cols": self.cols,
            "sigma_max": self.sigma_max,
            "sigma_min": self.sigma_min,
            "kappa_2": self.kappa_2,
            "numerical_rank": self.numerical_rank,
            "rank_deficient": self.rank_deficient,
            "residual_norm": self.residual_norm,
            "displacement_bound": self.displacement_bound,
            "singular_values": list(self.singular_values),
        }


def _finite(value: float) -> float:
    return float(value) if np.isfinite(value) else float("inf")


def condition_report(
    jacobian: Array | None,
    residual: Array | float | None,
    *,
    rank_tolerance: float | None = None,
) -> SolveConditioning | None:
    """Return SVD conditioning for ``jacobian`` with residual ``residual``.

    ``None`` is returned when no Jacobian is available, so callers can attach
    this metadata opportunistically without branching on solver internals.
    """
    if jacobian is None:
        return None
    j = np.asarray(jacobian, dtype=float)
    if j.ndim != 2 or j.size == 0:
        return None
    if not np.all(np.isfinite(j)):
        return None

    if residual is None:
        residual_norm = float("nan")
    elif np.isscalar(residual):
        residual_norm = abs(float(residual))
    else:
        residual_norm = float(np.linalg.norm(np.asarray(residual, dtype=float)))

    sigma = np.linalg.svd(j, compute_uv=False)
    sigma = np.asarray(sigma, dtype=float)
    rows, cols = j.shape
    sigma_max = float(sigma[0]) if sigma.size else 0.0
    sigma_min = float(sigma[-1]) if sigma.size else 0.0
    if rank_tolerance is None:
        rank_tolerance = sigma_max * max(rows, cols) * float(np.finfo(float).eps)
    numerical_rank = int(np.count_nonzero(sigma > rank_tolerance))
    full_rank = min(rows, cols)

    kappa = _finite(sigma_max / sigma_min) if sigma_min > 0.0 else float("inf")
    if sigma_min > 0.0 and np.isfinite(residual_norm):
        displacement = float(residual_norm / sigma_min)
    elif np.isnan(residual_norm):
        displacement = float("nan")
    else:
        displacement = float("inf")

    return SolveConditioning(
        rows=rows,
        cols=cols,
        sigma_max=sigma_max,
        sigma_min=sigma_min,
        kappa_2=kappa,
        numerical_rank=numerical_rank,
        rank_deficient=bool(numerical_rank < full_rank),
        residual_norm=residual_norm,
        displacement_bound=displacement,
        singular_values=tuple(float(x) for x in sigma),
    )


def scalar_condition_report(slope: float, residual: float) -> SolveConditioning:
    """Conditioning of a one-dimensional root solve with derivative ``slope``.

    Used by the critical-root census, where the accepted object is a scalar
    event functional localized in a single mass coordinate.  ``kappa_2`` is 1 by
    construction; ``displacement_bound = |residual| / |slope|`` is the parameter
    uncertainty the reported event residual actually buys.
    """
    magnitude = abs(float(slope))
    residual_norm = abs(float(residual))
    if magnitude > 0.0:
        displacement = residual_norm / magnitude
        kappa = 1.0
    else:
        displacement = float("inf")
        kappa = float("inf")
    return SolveConditioning(
        rows=1,
        cols=1,
        sigma_max=magnitude,
        sigma_min=magnitude,
        kappa_2=kappa,
        numerical_rank=1 if magnitude > 0.0 else 0,
        rank_deficient=magnitude <= 0.0,
        residual_norm=residual_norm,
        displacement_bound=displacement,
        singular_values=(magnitude,),
    )


def conditioning_dict(report: SolveConditioning | None) -> dict[str, Any] | None:
    return None if report is None else report.as_dict()


def summarize_conditioning(
    reports: list[SolveConditioning | dict[str, Any] | None],
) -> dict[str, Any]:
    """Aggregate conditioning across many solves for artifact-level reporting.

    Accepts either live ``SolveConditioning`` objects or the dictionaries they
    serialize to, so an assembler can summarize conditioning it only ever reads
    back out of a frozen JSON artifact.
    """
    present = [
        r.as_dict() if isinstance(r, SolveConditioning) else r
        for r in reports
        if r is not None
    ]
    if not present:
        return {"reported": 0, "missing": len(reports)}

    def column(name: str) -> list[float]:
        return sorted(
            float(r[name])
            for r in present
            if r.get(name) is not None and np.isfinite(float(r[name]))
        )

    kappas = column("kappa_2")
    sig_min = column("sigma_min")
    bounds = column("displacement_bound")

    def median(values: list[float]) -> float | None:
        return values[len(values) // 2] if values else None

    return {
        "reported": len(present),
        "missing": len(reports) - len(present),
        "kappa_2_median": median(kappas),
        "kappa_2_max": kappas[-1] if kappas else None,
        "sigma_min_min": sig_min[0] if sig_min else None,
        "sigma_min_median": median(sig_min),
        "displacement_bound_median": median(bounds),
        "displacement_bound_max": bounds[-1] if bounds else None,
        "rank_deficient_count": sum(1 for r in present if r.get("rank_deficient")),
    }
