"""Bracketing criteria for mass-slice critical-curve seeds.

WHY THIS MODULE EXISTS
----------------------
Every localized critical root in the v1 census reached the localizer through one
gate: ``scripts/extract_mass_slice_brackets.transition_brackets``, which emits a
1-D bracket for an adjacent pair of published baseline rows exactly when the
*published stability label* flips ``S <-> U``.  That criterion is structurally
incomplete, and no amount of raster refinement repairs it.

The reduced monodromy of a planar three-body periodic orbit has four
symmetry-forced unit multipliers; the remaining four are the roots of

    lambda + 1/lambda = t,   P(t) = t^2 - (alpha - 4) t + (beta - 4 alpha + 8).

The three codimension-one Floquet events are the zero sets of

    G_plus       = P(+2) = beta - 6 alpha + 20      (a nontrivial lambda = +1)
    G_minus      = P(-2) = beta - 2 alpha +  4      (a nontrivial lambda = -1)
    discriminant = (alpha - 4)^2 - 4 (beta - 4 alpha + 8)

which are polynomials in the smooth invariants (alpha, beta) and therefore
continuous along any smoothly continued family.  The published label, by
contrast, is the *thresholded* predicate

    S  <=>  n_unstable == 0,        n_unstable = #{ |lambda| > 1 } in {0, 1, 2}.

``n_unstable`` is not a function of the sign of a single event.  A ``G_plus``
zero at which the unstable dimension steps ``2 -> 1`` leaves ``n_unstable > 0``
on *both* sides, so the published label reads ``U`` on both sides and the label
criterion emits nothing -- at any grid resolution, because refining the grid
refines the two ``U`` samples into more ``U`` samples.  Such a curve is interior
to the unstable region: a genuine critical curve of the mass plane that is not a
stability boundary.

This is not hypothetical.  ``research/evidence/V1_SIGN_TOPOLOGY_AUDIT_2026-08-16.json``
and ``V1_SIGN_TOPOLOGY_CROSSING_2026-08-16.json`` report seven critical curves
absent from the committed graph, each localized by
``critical_manifold.localize_critical_point`` at the frozen gates
(``|event| <= 2e-8``, ``closure <= 1e-7``).  Six of the seven record endpoint
unstable dimensions ``2 -> 1``: interior crossings, invisible to any label rule.
The seventh, ``(0.9295, 0.8860337144)`` ``minus_one``, records ``1 -> 0`` and is
therefore a stability boundary; it went missing for the *second* reason below,
not the first.  The two losses are distinct and should not be conflated.

Second loss: ``critical_manifold.infer_event_mode`` returns exactly one event
mode per bracket.  A cell that two events cross yields one localized root, so one
of the two critical curves through that cell is discarded even though the cell
itself was sampled.  Near a codimension-two organizer, where a ``plus_one`` and a
``minus_one`` curve cross, this is what makes the reconstructed polylines swap
mechanism from one cell to the next.

So this module offers two named criteria:

``published_label_brackets``
    Bit-for-bit the historical criterion.  The 620-cell census and every
    artifact derived from it were produced under it and must stay reproducible,
    so it is preserved verbatim rather than "fixed" in place.

``event_sign_brackets``
    Brackets on a sign change of an event function itself, emitting one bracket
    per crossing event per cell.  A sign change of a continuous function on a
    path forces a zero on that path, so this criterion sees every transversal
    crossing of every one of the three events, whether or not the crossing also
    moves the stability label, and it does not collapse a two-event cell to a
    single root.  It contains the label criterion except on rows whose periodic
    closure cannot be certified, which bound no bracket here.

The price is arithmetic: the label criterion reads a published column, while the
event criterion needs one Newton correction plus one monodromy per baseline row.
That is the honest cost of not being blind.

Nothing here writes evidence or decides release readiness.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from .baseline import BaselineRow
from .reduced import ReducedFloquetResult

Criterion = Literal["published-label", "event-sign"]
CRITERIA: tuple[Criterion, ...] = ("published-label", "event-sign")

#: Event component name -> the ``critical_manifold.EventMode`` it certifies.
COMPONENT_EVENT_MODE: dict[str, str] = {
    "g_plus": "plus_one",
    "g_minus": "minus_one",
    "discriminant": "trace_collision",
}

EVENT_COMPONENTS: tuple[str, ...] = ("g_plus", "g_minus", "discriminant")

#: Frozen gates.  Never widened here.  A row that misses closure is dropped
#: from the criterion's domain, never accepted with a looser tolerance.
MAX_CLOSURE = 1e-7


# ---------------------------------------------------------------------------
# the historical criterion, preserved verbatim
# ---------------------------------------------------------------------------
def published_label_brackets(
    rows: Sequence[BaselineRow],
) -> list[tuple[BaselineRow, BaselineRow]]:
    """Bracket adjacent rows whose *published* S/U label flips.

    This is the criterion that produced the 620-cell census.  It is kept
    unchanged and separately named because every committed artifact derived from
    that census must remain reproducible.  It is blind, by construction, to any
    critical curve interior to the unstable region -- see the module docstring
    and ``research/BRACKET_CRITERION_BLINDNESS.md``.
    """
    ordered = sorted(rows, key=lambda r: r.m2)
    out: list[tuple[BaselineRow, BaselineRow]] = []
    for left, right in zip(ordered, ordered[1:], strict=False):
        if left.published_stability != right.published_stability:
            out.append((left, right))
    return out


# ---------------------------------------------------------------------------
# event-function state of one baseline row
# ---------------------------------------------------------------------------
def event_components(floquet: ReducedFloquetResult) -> dict[str, float]:
    """The three smooth codimension-one Floquet event functions."""
    return {
        "g_plus": float(floquet.beta - 6.0 * floquet.alpha + 20.0),
        "g_minus": float(floquet.beta - 2.0 * floquet.alpha + 4.0),
        "discriminant": float(floquet.discriminant),
    }


def unstable_count(alpha: float, beta: float, *, margin: float = 1e-6) -> int | None:
    """Number of reduced multiplier *pairs* off the unit circle, in {0, 1, 2}.

    Read off the trace polynomial rather than the raw eigenvalues, because the
    trace polynomial is the smooth object: with t = lambda + 1/lambda, a real
    |t| > 2 puts one pair off the circle and a complex conjugate pair of trace
    roots (negative discriminant, a Krein quartet) puts both off.  ``None`` means
    a threshold sits inside ``margin`` and float64 cannot decide.

    Deliberately the same predicate as ``scripts/audit_sign_topology.unstable_count``
    -- the falsifying audit and the replacement criterion must agree on what "the
    published label would have said" or the comparison in
    ``scripts/audit_bracket_criteria.py`` would be meaningless.  A test pins the
    two implementations together on a grid.
    """
    a = alpha - 4.0
    b = beta - 4.0 * alpha + 8.0
    disc = a * a - 4.0 * b
    if abs(disc) < margin:
        return None
    if disc < 0.0:
        return 2
    root = disc**0.5
    count = 0
    for t in ((a + root) / 2.0, (a - root) / 2.0):
        if abs(abs(t) - 2.0) < margin:
            return None
        if abs(t) > 2.0:
            count += 1
    return count


@dataclass(frozen=True)
class RowEvent:
    """One baseline row with its event functions evaluated.

    ``ok`` is False when the periodic closure could not be certified under
    ``MAX_CLOSURE``; such a row bounds no bracket, because a sign change is only
    meaningful along a family that actually closes.
    """

    row: BaselineRow
    ok: bool
    closure: float
    values: dict[str, float]
    n_unstable: int | None
    note: str = ""
    chart: tuple[float, float, float, float] | None = None

    @property
    def m2(self) -> float:
        return self.row.m2

    @property
    def screening_label(self) -> str | None:
        """S/U as recomputed here, for comparison with the published column."""
        if not self.ok or self.n_unstable is None:
            return None
        return "S" if self.n_unstable == 0 else "U"


@dataclass(frozen=True)
class EventBracket:
    """An adjacent published-row pair across which an event function changes sign."""

    left: RowEvent
    right: RowEvent
    component: str
    interior_to_unstable_region: bool

    @property
    def event_mode(self) -> str:
        return COMPONENT_EVENT_MODE[self.component]

    @property
    def label_flip(self) -> bool:
        """True when the published S/U label also flips across this bracket."""
        return self.left.row.published_stability != self.right.row.published_stability

    @property
    def m2_bracket(self) -> tuple[float, float]:
        return (self.left.m2, self.right.m2)

    @property
    def values(self) -> tuple[float, float]:
        return (self.left.values[self.component], self.right.values[self.component])


def row_event_from_invariants(
    row: BaselineRow,
    alpha: float,
    beta: float,
    *,
    closure: float = 0.0,
) -> RowEvent:
    """Build a ``RowEvent`` from stability invariants directly.

    For tests and for consumers that already hold (alpha, beta) -- the event
    functions are polynomials in the invariants, so no integration is needed to
    go from invariants to bracket decisions.
    """
    return RowEvent(
        row,
        ok=True,
        closure=float(closure),
        values={
            "g_plus": float(beta - 6.0 * alpha + 20.0),
            "g_minus": float(beta - 2.0 * alpha + 4.0),
            "discriminant": float((alpha - 4.0) ** 2 - 4.0 * (beta - 4.0 * alpha + 8.0)),
        },
        n_unstable=unstable_count(alpha, beta),
    )


def evaluate_row(
    row: BaselineRow,
    *,
    max_closure: float = MAX_CLOSURE,
    correct: bool = True,
    max_nfev: int = 60,
) -> RowEvent:
    """Evaluate the three event functions at one published baseline row.

    ``correct=True`` runs the repository's own variational Newton corrector on
    the published chart first, so the reported closure is a measured residual and
    not an assumption about how many digits the supplementary table printed.
    ``correct=False`` reads the published chart as given; it is faster, useful
    for reconnaissance, and reports ``closure`` as NaN because none was measured.

    Errors are captured, not raised: one non-converging row must not abort a
    slice sweep, and a row that fails is recorded as bounding no bracket.
    """
    from .boundary import evaluate  # noqa: PLC0415 - keep scipy off the import path
    from .liao_family import FamilyPoint, correct_family_point  # noqa: PLC0415

    nan = float("nan")
    empty = dict.fromkeys(EVENT_COMPONENTS, nan)
    try:
        if correct:
            point = correct_family_point(
                (row.m1, row.m2, row.m3),
                (row.x1, row.v1, row.v2, row.period),
                max_nfev=max_nfev,
            )
            if not point.success or point.residual_norm > max_closure:
                return RowEvent(
                    row,
                    ok=False,
                    closure=float(point.residual_norm),
                    values=empty,
                    n_unstable=None,
                    note=f"closure {point.residual_norm:.3e} > {max_closure:g}",
                )
            closure = float(point.residual_norm)
        else:
            point = FamilyPoint(
                masses=(row.m1, row.m2, row.m3),
                x1=row.x1,
                v1=row.v1,
                v2=row.v2,
                period=row.period,
                residual_norm=nan,
                nfev=0,
                success=True,
            )
            closure = nan
        floquet = evaluate(point).floquet
    except Exception as exc:  # noqa: BLE001 - a failed row is a result, not an abort
        return RowEvent(
            row,
            ok=False,
            closure=nan,
            values=empty,
            n_unstable=None,
            note=f"{type(exc).__name__}: {exc}",
        )
    return RowEvent(
        row,
        ok=True,
        closure=closure,
        values=event_components(floquet),
        n_unstable=unstable_count(floquet.alpha, floquet.beta),
        chart=(point.x1, point.v1, point.v2, point.period),
    )


def event_sign_brackets(
    states: Sequence[RowEvent],
    *,
    components: Sequence[str] = EVENT_COMPONENTS,
    sign_threshold: float = 0.0,
) -> list[EventBracket]:
    """Bracket adjacent rows across which an event function changes sign.

    One bracket is emitted per (adjacent pair, flipping component), so a pair
    straddling an organizer where two events vanish together yields two
    brackets -- which is the honest report: there are two curves there.

    ``sign_threshold`` screens out sign "changes" whose magnitudes are below the
    float64 event noise of the corrected chart on both sides; the default 0.0
    keeps every sign change, since a spurious bracket is cheap (the localizer
    rejects it at the frozen gates) while a missed one is exactly the failure
    this module exists to prevent.

    "Adjacent" means adjacent *in the supplied sequence*, so pass a contiguous
    run of published rows.  A sparse or strided sample still yields correct
    brackets, but wider ones, and it can merge two nearby crossings of the same
    event into a single interval across which the sign does not change -- the
    same even-crossing-count blindness every sign criterion has, now at the
    sampling density rather than at the label.  Rows that fail closure are
    skipped, which likewise widens the neighbouring interval.
    """
    for component in components:
        if component not in EVENT_COMPONENTS:
            raise ValueError(f"unknown event component: {component}")
    ordered = sorted(states, key=lambda s: s.m2)
    out: list[EventBracket] = []
    for left, right in zip(ordered, ordered[1:], strict=False):
        if not (left.ok and right.ok):
            continue
        for component in components:
            vl = left.values[component]
            vr = right.values[component]
            if max(abs(vl), abs(vr)) < sign_threshold:
                continue
            if vl == 0.0 or vr == 0.0 or (vl > 0.0) != (vr > 0.0):
                interior = (
                    left.n_unstable is not None
                    and right.n_unstable is not None
                    and left.n_unstable > 0
                    and right.n_unstable > 0
                )
                out.append(EventBracket(left, right, component, interior))
    return out


def label_invisible_brackets(brackets: Sequence[EventBracket]) -> list[EventBracket]:
    """The event brackets no S/U-label criterion can ever produce.

    Precisely: the published label does not flip across the pair, so
    ``published_label_brackets`` emits nothing there, yet an event function
    changes sign, so a critical curve crosses the pair.
    """
    return [b for b in brackets if not b.label_flip]
