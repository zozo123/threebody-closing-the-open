"""Re-derive published critical roots with the shipped Python dynamics.

Motivation.  Before this module, *nothing* in the repository connected the 620
published critical roots back to the dynamics in :mod:`threebody_atlas`.
``tests/test_float64_census.py`` checks the numbers a census *recorded* against
the frozen gates; ``tests/test_critical_graph.py`` feeds the roots file to the
assembler.  Both would keep passing if the force law itself were wrong, because
neither ever re-evaluates a root.  A grep for the roots file finds the assembler
and the graph tests, and nothing that integrates anything.

What this does.  For a sampled subset of the published roots it rebuilds the
full state from the recorded chart coordinates (x1, v1, v2, period, masses) and
recomputes the reduced Floquet trace invariants with the shipped SciPy DOP853
path.  The recorded ``alpha``/``beta``/``discriminant`` came from the *Julia
BigFloat* estimator, so agreement is a genuine cross-implementation audit, not
a tautology.

Precision, honestly stated.  Float64 CANNOT re-verify the frozen |event| <= 2e-8
gate at these points and this module does not pretend otherwise.  Measured over
31 roots sampled across the census (numpy 2.5.2 / scipy 1.18.0, DOP853 at
rtol=1e-12, atol=1e-14):

    max |alpha_float64 - alpha_bigfloat|        6.1e-07
    max |beta_float64  - beta_bigfloat|         1.3e-06
    max |disc_float64  - disc_bigfloat|         7.2e-06
    max |event_float64|                         2.5e-06   (frozen gate is 2e-8)
    max closure_float64                         3.5e-08   (frozen gate is 1e-7)

The event discrepancy is the expected conditioning of
``event = beta - 6 alpha + 20`` under an alpha carried to ~1e-7: it is a
statement about float64, not evidence against the roots.  The closure gate, by
contrast, IS reproduced in float64 and is asserted at its frozen value of 1e-7.
No gate is loosened anywhere here; the float64 bands below are new, additional
checks on a quantity nothing checked before.

Sensitivity.  These invariants are violently sensitive to the force law: a 1e-6
RELATIVE change in the gravitational constant moves ``event`` by up to 12.8 and
``alpha`` by up to 2.0 on the sampled roots -- six orders of magnitude above the
float64 agreement band.  That is what makes this a usable mutation detector.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .critical_manifold import event_value
from .dynamics import integrate_orbit
from .liao_family import state_from_chart
from .reduced import compute_reduced_floquet

Array = np.ndarray

# Integration settings.  One decade tighter than the library screening defaults,
# because the quantity being audited is a difference at the 1e-7 level.
AUDIT_RTOL = 1e-12
AUDIT_ATOL = 1e-14

# Frozen project gate.  Reproduced, never relaxed.
CLOSURE_GATE = 1e-7

# Float64 re-derivation bands.  These are NOT project gates: they are the
# measured disagreement between this float64 path and the Julia BigFloat
# estimator that produced the roots, rounded up ~50x for platform headroom.
ALPHA_BAND = 1e-4
BETA_BAND = 1e-4
DISCRIMINANT_BAND = 1e-3
FLOAT64_EVENT_BAND = 1e-4


@dataclass(frozen=True)
class RootAudit:
    cell_id: int
    event_mode: str
    alpha: float
    beta: float
    discriminant: float
    event: float
    closure: float
    recorded_alpha: float
    recorded_beta: float
    recorded_discriminant: float
    recorded_event: float
    alpha_error: float
    beta_error: float
    discriminant_error: float

    def failures(self) -> list[str]:
        problems: list[str] = []
        if not self.closure <= CLOSURE_GATE:
            problems.append(
                f"cell {self.cell_id}: float64 closure {self.closure:.3e} exceeds the "
                f"frozen gate {CLOSURE_GATE:.0e}"
            )
        for label, error, band in (
            ("alpha", self.alpha_error, ALPHA_BAND),
            ("beta", self.beta_error, BETA_BAND),
            ("discriminant", self.discriminant_error, DISCRIMINANT_BAND),
        ):
            if not error <= band:
                problems.append(
                    f"cell {self.cell_id}: float64 {label} differs from the recorded "
                    f"BigFloat value by {error:.3e} (band {band:.0e})"
                )
        if not abs(self.event) <= FLOAT64_EVENT_BAND:
            problems.append(
                f"cell {self.cell_id}: float64 |event| {abs(self.event):.3e} exceeds the "
                f"float64 re-derivation band {FLOAT64_EVENT_BAND:.0e}"
            )
        return problems


def load_roots(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        row
        for row in payload.get("roots", [])
        if row.get("status") == "ok" or row.get("passed") is True
    ]


def sample(roots: Sequence[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """Deterministically spread ``count`` roots across the census."""
    if count <= 0 or count >= len(roots):
        return list(roots)
    stride = len(roots) / float(count)
    return [roots[int(index * stride)] for index in range(count)]


def audit_root(row: dict[str, Any]) -> RootAudit:
    masses = tuple(float(value) for value in row["masses"])
    x1, v1, v2 = float(row["x1"]), float(row["v1"]), float(row["v2"])
    period = float(row["period"])
    state = state_from_chart(masses, x1, v1, v2)
    mass_vector = np.asarray(masses, dtype=float)
    floquet = compute_reduced_floquet(
        state, mass_vector, period, rtol=AUDIT_RTOL, atol=AUDIT_ATOL
    )
    orbit = integrate_orbit(state, mass_vector, period, rtol=AUDIT_RTOL, atol=AUDIT_ATOL)
    mode = str(row["event_mode"])
    recorded_alpha = float(row["alpha"])
    recorded_beta = float(row["beta"])
    recorded_discriminant = float(row["discriminant"])
    return RootAudit(
        cell_id=int(row["cell_id"]),
        event_mode=mode,
        alpha=float(floquet.alpha),
        beta=float(floquet.beta),
        discriminant=float(floquet.discriminant),
        event=float(event_value(floquet, mode)),  # type: ignore[arg-type]
        closure=float(orbit.closure_norm),
        recorded_alpha=recorded_alpha,
        recorded_beta=recorded_beta,
        recorded_discriminant=recorded_discriminant,
        recorded_event=float(row.get("event", float("nan"))),
        alpha_error=abs(float(floquet.alpha) - recorded_alpha),
        beta_error=abs(float(floquet.beta) - recorded_beta),
        discriminant_error=abs(float(floquet.discriminant) - recorded_discriminant),
    )


def audit(path: Path, *, count: int) -> list[RootAudit]:
    return [audit_root(row) for row in sample(load_roots(path), count)]


def audit_cells(path: Path, cell_ids: Sequence[int]) -> list[RootAudit]:
    """Audit exactly these cell ids.

    Index-based sampling shifts when a root is added or removed, which would
    make an artifact-level mutation look like a physics failure.  Pinning the
    cell ids keeps the differential comparison a statement about the dynamics.
    """
    wanted = list(cell_ids)
    by_id = {int(row["cell_id"]): row for row in load_roots(path)}
    return [audit_root(by_id[cell]) for cell in wanted if cell in by_id]


def to_json(audits: Sequence[RootAudit]) -> dict[str, Any]:
    return {
        "schema": "atlas.v1.root-physics-audit/1",
        "rtol": AUDIT_RTOL,
        "atol": AUDIT_ATOL,
        "closure_gate": CLOSURE_GATE,
        "audits": [asdict(item) for item in audits],
    }


def compare(
    current: Sequence[RootAudit], baseline: dict[str, Any], *, tolerance: float
) -> list[str]:
    """Differential check against a baseline emitted by an unmutated tree.

    This is the form the mutation harness uses.  It is far sharper than the
    absolute bands above: two runs of the same code on the same machine agree
    bit-for-bit, so any drift at all is a real change in the physics.
    """
    recorded = {int(row["cell_id"]): row for row in baseline.get("audits", [])}
    problems: list[str] = []
    for item in current:
        reference = recorded.get(item.cell_id)
        if reference is None:
            problems.append(f"cell {item.cell_id} is absent from the baseline audit")
            continue
        for field in ("alpha", "beta", "discriminant", "event", "closure"):
            drift = abs(getattr(item, field) - float(reference[field]))
            if drift > tolerance:
                problems.append(
                    f"cell {item.cell_id}: {field} drifted {drift:.6e} from the baseline "
                    f"(tolerance {tolerance:.0e})"
                )
    missing = sorted(set(recorded) - {item.cell_id for item in current})
    problems.extend(
        f"cell {cell} was audited in the baseline but is no longer in the roots file"
        for cell in missing
    )
    return problems


def baseline_cell_ids(baseline: dict[str, Any]) -> list[int]:
    return [int(row["cell_id"]) for row in baseline.get("audits", [])]
