#!/usr/bin/env python3
"""Sign-vector / planar-arrangement consistency audit of the mass-plane critical graph.

WHY THIS EXISTS
---------------
``research/evidence/V1_CRITICAL_GRAPH.json`` publishes seven ``mechanism_polyline``
edges reconstructed from 620 localized roots.  Those 620 roots all come from
*stability transition cells*: adjacent grid samples whose screening stability
label flips S<->U.  That construction can only ever see critical curves that are
also stability boundaries.  A critical curve that lives entirely inside the
unstable region -- a lambda=+1 crossing that takes the unstable dimension from 2
to 1, say -- produces no S/U bracket and is therefore structurally invisible to
the census, no matter how fine the grid.

The reduced Floquet data supplies a discrete state that closes that gap.  With

    P(t) = t^2 - a t + b,    a = alpha - 4,   b = beta - 4 alpha + 8

(the trace polynomial used throughout ``critical_manifold.py``), define

    G_plus       = P(+2) = beta - 6 alpha + 20     zero <=> a nontrivial lambda = +1
    G_minus      = P(-2) = beta - 2 alpha +  4     zero <=> a nontrivial lambda = -1
    discriminant = a^2 - 4 b                       zero <=> the two trace roots collide
    n_unstable   = #{ multipliers with |lambda| > 1 } in {0, 1, 2}

and set S(m1, m2) = (sgn G_plus, sgn G_minus, sgn discriminant, n_unstable).

All three scalars are polynomials in (alpha, beta), hence continuous along any
path on which the periodic-orbit family continues smoothly.  So a sign change of
a component *forces* a zero of that component on the path.  That gives a
completeness test no finite raster can give:

  * a path that meets the committed 1-complex in zero points but changes S must
    cross a critical curve that the graph does not contain;
  * crossing an edge labelled ``plus_one`` may flip only sgn G_plus, ``minus_one``
    only sgn G_minus, ``trace_collision`` only sgn discriminant;
  * consequently any closed loop must return S to its initial value.

The audit therefore probes S at a few hundred well-placed points, computes exact
piecewise-linear intersections of axis-aligned probe paths with the committed
polylines, and reports every inconsistency with coordinates and the offending
component.

WHAT THIS IS NOT
----------------
Screening arithmetic.  Probes use the float64 shooting corrector, whose event
noise is order 1e-5; probes are therefore held off the committed curves by a
standoff and a component is only assigned a sign when its magnitude exceeds
``--sign-threshold``.  A bracket found this way is a screening bracket, not yet
a root -- which is why every reported missing curve is then handed to
``critical_manifold.localize_critical_point``, the same localizer that produced
the 620 committed roots, at the unchanged gates ``|event| <= 2e-8`` and
``closure <= 1e-7``.  A certification that misses those gates is reported as a
miss; no tolerance is ever widened to make a finding land.  This script never
writes ``release_ready`` and never touches the critical graph.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import signal
import threading
from bisect import bisect_left
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

SCHEMA = "atlas.v1.sign-topology-audit/1"

COMPONENTS: tuple[str, ...] = ("G_plus", "G_minus", "discriminant")

#: Which sign component each mechanism is *permitted* to flip.
MECHANISM_COMPONENT: dict[str, str] = {
    "plus_one": "G_plus",
    "minus_one": "G_minus",
    "trace_collision": "discriminant",
}

#: Frozen gates.  Never loosened here; probes that miss closure are discarded,
#: not accepted with a wider tolerance.
MAX_CLOSURE = 1e-7

DEFAULT_SCAN_LINES = (
    0.820, 0.860, 0.900, 0.920, 0.940, 0.970, 1.000, 1.020,
    1.040, 1.046, 1.055, 1.065, 1.080, 1.095,
)


# --------------------------------------------------------------------------
# committed arrangement: exact piecewise-linear geometry
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class PolylineEdge:
    """One committed ``mechanism_polyline``, stored as a graph m2 = f(m1)."""

    edge_id: str
    mechanism: str
    orientation: str
    vertices: tuple[tuple[float, float], ...]

    @property
    def component(self) -> str | None:
        return MECHANISM_COMPONENT.get(self.mechanism)

    @property
    def m1_min(self) -> float:
        return self.vertices[0][0]

    @property
    def m1_max(self) -> float:
        return self.vertices[-1][0]

    @property
    def degenerate(self) -> bool:
        """A single-cell edge has no extent and cannot separate anything."""
        return self.m1_max <= self.m1_min

    def m2_at(self, m1: float) -> float | None:
        """Linear interpolation, or None where the committed edge does not exist."""
        if m1 < self.m1_min or m1 > self.m1_max:
            return None
        xs = [v[0] for v in self.vertices]
        idx = bisect_left(xs, m1)
        if idx == 0:
            return self.vertices[0][1]
        if idx >= len(xs):
            return self.vertices[-1][1]
        x0, y0 = self.vertices[idx - 1]
        x1, y1 = self.vertices[idx]
        if x1 == x0:
            return y1
        return y0 + (y1 - y0) * (m1 - x0) / (x1 - x0)

    def vertical_crossings(self, m1: float, m2_lo: float, m2_hi: float) -> int:
        """Intersections of the vertical segment {m1} x (m2_lo, m2_hi)."""
        y = self.m2_at(m1)
        if y is None:
            return 0
        lo, hi = (m2_lo, m2_hi) if m2_lo <= m2_hi else (m2_hi, m2_lo)
        return 1 if lo < y < hi else 0

    def horizontal_crossings(self, m2: float, m1_a: float, m1_b: float) -> int:
        """Intersections of the horizontal segment (m1_a, m1_b) x {m2}.

        Exact for the piecewise-linear edge: count sign changes of m2 - f(m1)
        over the edge vertices clipped to the segment, with the clipped
        endpoints evaluated by interpolation.
        """
        lo, hi = (m1_a, m1_b) if m1_a <= m1_b else (m1_b, m1_a)
        lo = max(lo, self.m1_min)
        hi = min(hi, self.m1_max)
        if lo >= hi:
            return 0
        xs = [lo]
        xs.extend(x for x, _ in self.vertices if lo < x < hi)
        xs.append(hi)
        values = []
        for x in xs:
            y = self.m2_at(x)
            assert y is not None
            values.append(m2 - y)
        crossings = 0
        for left, right in zip(values, values[1:], strict=False):
            if left == 0.0 or right == 0.0:
                # Path runs exactly through a vertex; caller screens these out
                # with the clearance test, but count it as a crossing so the
                # audit never *under*-counts committed structure.
                crossings += 1
            elif (left > 0.0) != (right > 0.0):
                crossings += 1
        return crossings

    def clearance_horizontal(self, m2: float, m1_a: float, m1_b: float) -> float:
        lo, hi = (m1_a, m1_b) if m1_a <= m1_b else (m1_b, m1_a)
        lo = max(lo, self.m1_min)
        hi = min(hi, self.m1_max)
        if lo >= hi:
            return math.inf
        xs = [lo]
        xs.extend(x for x, _ in self.vertices if lo < x < hi)
        xs.append(hi)
        return min(abs(m2 - (self.m2_at(x) or m2)) for x in xs)

    def clearance_vertical(self, m1: float, m2_a: float, m2_b: float) -> float:
        y = self.m2_at(m1)
        if y is None:
            return math.inf
        lo, hi = (m2_a, m2_b) if m2_a <= m2_b else (m2_b, m2_a)
        if lo <= y <= hi:
            return 0.0
        return min(abs(y - lo), abs(y - hi))


def edges_from_graph(graph: dict[str, Any], roots: Sequence[dict[str, Any]]) -> list[PolylineEdge]:
    """Rebuild each committed edge as a polyline from its ``cell_ids``."""
    by_cell = {int(r["cell_id"]): r for r in roots}
    out: list[PolylineEdge] = []
    for edge in graph.get("edges", ()):
        if edge.get("kind") != "mechanism_polyline":
            continue
        pts: list[tuple[float, float]] = []
        for cell in edge.get("cell_ids", ()):
            root = by_cell.get(int(cell))
            if root is None:
                raise KeyError(f"edge {edge['id']} references unknown cell {cell}")
            m1, m2 = float(root["masses"][0]), float(root["masses"][1])
            pts.append((m1, m2))
        pts.sort()
        deduped: list[tuple[float, float]] = []
        for point in pts:
            if deduped and point[0] == deduped[-1][0]:
                # Two roots of the same edge at one m1 slice would make the
                # edge a non-graph; keep both out of the interpolation and let
                # the caller see it via the recorded warning.
                continue
            deduped.append(point)
        out.append(
            PolylineEdge(
                edge_id=str(edge["id"]),
                mechanism=str(edge["mechanism"]),
                orientation=str(edge.get("orientation", "")),
                vertices=tuple(deduped),
            )
        )
    return out


def non_graph_edges(graph: dict[str, Any], roots: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Edges holding two or more roots at one m1 slice (not a graph over m1)."""
    by_cell = {int(r["cell_id"]): r for r in roots}
    flagged = []
    for edge in graph.get("edges", ()):
        seen: dict[float, int] = {}
        for cell in edge.get("cell_ids", ()):
            m1 = float(by_cell[int(cell)]["masses"][0])
            seen[m1] = seen.get(m1, 0) + 1
        repeats = sorted(m1 for m1, n in seen.items() if n > 1)
        if repeats:
            flagged.append({"edge_id": edge["id"], "repeated_m1": repeats})
    return flagged


# --------------------------------------------------------------------------
# discrete state
# --------------------------------------------------------------------------
def sign_of(value: float, threshold: float) -> int:
    """+1 / -1, or 0 meaning 'too close to zero for float64 screening to call'."""
    if not math.isfinite(value) or abs(value) < threshold:
        return 0
    return 1 if value > 0.0 else -1


def unstable_count(alpha: float, beta: float, *, margin: float = 1e-6) -> int | None:
    """Number of reduced multipliers with |lambda| > 1, or None if undecidable.

    Reciprocal pairs: t = lambda + 1/lambda.  A real |t| > 2 gives one multiplier
    off the unit circle; a complex conjugate pair of trace roots gives two.
    """
    a = alpha - 4.0
    b = beta - 4.0 * alpha + 8.0
    disc = a * a - 4.0 * b
    if abs(disc) < margin:
        return None
    if disc < 0.0:
        return 2
    root = math.sqrt(disc)
    count = 0
    for t in ((a + root) / 2.0, (a - root) / 2.0):
        if abs(abs(t) - 2.0) < margin:
            return None
        if abs(t) > 2.0:
            count += 1
    return count


@dataclass
class Probe:
    """One evaluated (m1, m2) sample of the reduced Floquet state."""

    m1: float
    m2: float
    ok: bool
    note: str = ""
    alpha: float = float("nan")
    beta: float = float("nan")
    values: dict[str, float] = field(default_factory=dict)
    n_unstable: int | None = None
    closure: float = float("nan")
    seconds: float = 0.0
    #: (x1, v1, v2, period) of the corrected orbit, so a downstream certifier can
    #: rebuild the FamilyPoint without paying for the shooting solve again.
    chart: tuple[float, float, float, float] | None = None

    def signs(self, threshold: float) -> dict[str, int]:
        return {c: sign_of(self.values.get(c, float("nan")), threshold) for c in COMPONENTS}

    def state(self, threshold: float) -> tuple[int, int, int, int] | None:
        """S, or None when any component is numerically undecidable."""
        if not self.ok or self.n_unstable is None:
            return None
        s = self.signs(threshold)
        if any(v == 0 for v in s.values()):
            return None
        return (s["G_plus"], s["G_minus"], s["discriminant"], self.n_unstable)

    def as_json(self) -> dict[str, Any]:
        return {
            "m1": self.m1,
            "m2": self.m2,
            "ok": self.ok,
            "note": self.note,
            "alpha": self.alpha,
            "beta": self.beta,
            "G_plus": self.values.get("G_plus"),
            "G_minus": self.values.get("G_minus"),
            "discriminant": self.values.get("discriminant"),
            "n_unstable": self.n_unstable,
            "closure": self.closure,
            "seconds": round(self.seconds, 3),
            "chart": list(self.chart) if self.chart else None,
        }


def state_from_invariants(alpha: float, beta: float) -> dict[str, float]:
    return {
        "G_plus": beta - 6.0 * alpha + 20.0,
        "G_minus": beta - 2.0 * alpha + 4.0,
        "discriminant": (alpha - 4.0) ** 2 - 4.0 * (beta - 4.0 * alpha + 8.0),
    }


def make_probe(m1: float, m2: float, alpha: float, beta: float, *, closure: float = 0.0) -> Probe:
    """Build a Probe from reduced invariants (used by the audit and by tests)."""
    return Probe(
        m1=m1,
        m2=m2,
        ok=True,
        alpha=alpha,
        beta=beta,
        values=state_from_invariants(alpha, beta),
        n_unstable=unstable_count(alpha, beta),
        closure=closure,
    )


# --------------------------------------------------------------------------
# consistency checks (pure: they take probes, not dynamics)
# --------------------------------------------------------------------------
def _committed_between(
    edges: Sequence[PolylineEdge], m1: float, m2_lo: float, m2_hi: float
) -> list[PolylineEdge]:
    return [e for e in edges if e.vertical_crossings(m1, m2_lo, m2_hi)]


def check_vertical_brackets(
    edges: Sequence[PolylineEdge],
    line_probes: Sequence[Probe],
    *,
    threshold: float,
) -> list[dict[str, Any]]:
    """Every sign flip between consecutive probes on a scan line must be explained.

    A flip of component ``c`` between m2_a and m2_b forces a zero of ``c`` in the
    open interval.  The committed graph must contain an edge of the matching
    mechanism there.  If it does not, a critical curve is missing and the
    interval is a bracket for it.
    """
    usable = [p for p in line_probes if p.ok]
    usable.sort(key=lambda p: p.m2)
    out: list[dict[str, Any]] = []
    for left, right in zip(usable, usable[1:], strict=False):
        ls, rs = left.signs(threshold), right.signs(threshold)
        crossed = _committed_between(edges, left.m1, left.m2, right.m2)
        for component in COMPONENTS:
            if ls[component] == 0 or rs[component] == 0 or ls[component] == rs[component]:
                continue
            matching = [e for e in crossed if e.component == component]
            if matching:
                continue
            out.append(
                {
                    "kind": "missing_critical_curve",
                    "component": component,
                    "mechanism": {v: k for k, v in MECHANISM_COMPONENT.items()}[component],
                    "m1": left.m1,
                    "m2_bracket": [left.m2, right.m2],
                    "value_bracket": [left.values[component], right.values[component]],
                    "n_unstable": [left.n_unstable, right.n_unstable],
                    "committed_edges_in_bracket": [e.edge_id for e in crossed],
                    "detail": (
                        f"sgn {component} flips between m2={left.m2:.6f} and m2={right.m2:.6f} at "
                        f"m1={left.m1:.6f}, but the committed graph has no "
                        f"{ {v: k for k, v in MECHANISM_COMPONENT.items()}[component] } edge there"
                    ),
                }
            )
    return out


def check_edge_transversals(
    edges: Sequence[PolylineEdge],
    line_probes: Sequence[Probe],
    *,
    threshold: float,
) -> list[dict[str, Any]]:
    """Crossing one committed edge may flip only that edge's own component."""
    usable = sorted((p for p in line_probes if p.ok), key=lambda p: p.m2)
    out: list[dict[str, Any]] = []
    for left, right in zip(usable, usable[1:], strict=False):
        crossed = _committed_between(edges, left.m1, left.m2, right.m2)
        if len(crossed) != 1:
            continue
        edge = crossed[0]
        ls, rs = left.signs(threshold), right.signs(threshold)
        for component in COMPONENTS:
            if ls[component] == 0 or rs[component] == 0:
                continue
            flipped = ls[component] != rs[component]
            if flipped and component != edge.component:
                out.append(
                    {
                        "kind": "forbidden_component_flip",
                        "component": component,
                        "edge_id": edge.edge_id,
                        "edge_mechanism": edge.mechanism,
                        "m1": left.m1,
                        "m2_bracket": [left.m2, right.m2],
                        "value_bracket": [left.values[component], right.values[component]],
                        "detail": (
                            f"crossing {edge.edge_id} ({edge.mechanism}) flips {component}, "
                            "which that mechanism may not flip"
                        ),
                    }
                )
            if not flipped and component == edge.component:
                out.append(
                    {
                        "kind": "no_flip_across_edge",
                        "component": component,
                        "edge_id": edge.edge_id,
                        "edge_mechanism": edge.mechanism,
                        "m1": left.m1,
                        "m2_bracket": [left.m2, right.m2],
                        "value_bracket": [left.values[component], right.values[component]],
                        "detail": (
                            f"{edge.edge_id} claims a {edge.mechanism} crossing but "
                            f"sgn {component} is unchanged across it"
                        ),
                    }
                )
    return out


def lpath_crossings(
    edges: Sequence[PolylineEdge], p: Probe, q: Probe
) -> tuple[list[PolylineEdge], float]:
    """Committed edges met by the L-path p -> (q.m1, p.m2) -> q, and clearance."""
    met: list[PolylineEdge] = []
    clearance = math.inf
    for edge in edges:
        n = edge.horizontal_crossings(p.m2, p.m1, q.m1)
        n += edge.vertical_crossings(q.m1, p.m2, q.m2)
        if n:
            met.extend([edge] * n)
        clearance = min(
            clearance,
            edge.clearance_horizontal(p.m2, p.m1, q.m1),
            edge.clearance_vertical(q.m1, p.m2, q.m2),
        )
    return met, clearance


def check_face_consistency(
    edges: Sequence[PolylineEdge],
    probes_a: Sequence[Probe],
    probes_b: Sequence[Probe],
    *,
    threshold: float,
    min_clearance: float,
) -> list[dict[str, Any]]:
    """Probes joined by a path disjoint from the committed 1-complex share S.

    This is the face test.  Zero intersections with every committed polyline
    means the two probes lie in the same face of the *claimed* arrangement, so
    S must agree.  Disagreement falsifies the claim that the seven polylines are
    the complete critical set.
    """
    out: list[dict[str, Any]] = []
    for p in probes_a:
        for q in probes_b:
            if not (p.ok and q.ok):
                continue
            sp, sq = p.state(threshold), q.state(threshold)
            if sp is None or sq is None or sp == sq:
                continue
            met, clearance = lpath_crossings(edges, p, q)
            if met or clearance < min_clearance:
                continue
            differing = [
                c for c in COMPONENTS if p.signs(threshold)[c] != q.signs(threshold)[c]
            ]
            out.append(
                {
                    "kind": "face_state_mismatch",
                    "components": differing or ["n_unstable"],
                    "from": [p.m1, p.m2],
                    "to": [q.m1, q.m2],
                    "state_from": list(sp),
                    "state_to": list(sq),
                    "path_clearance": clearance,
                    "detail": (
                        f"L-path ({p.m1:.4f},{p.m2:.4f}) -> ({q.m1:.4f},{p.m2:.4f}) -> "
                        f"({q.m1:.4f},{q.m2:.4f}) meets no committed edge yet S changes "
                        f"{sp} -> {sq}"
                    ),
                }
            )
    return out


def check_single_crossing_paths(
    edges: Sequence[PolylineEdge],
    probes_a: Sequence[Probe],
    probes_b: Sequence[Probe],
    *,
    threshold: float,
    min_clearance: float,
) -> list[dict[str, Any]]:
    """A path meeting exactly one committed edge may flip only that component."""
    out: list[dict[str, Any]] = []
    for p in probes_a:
        for q in probes_b:
            if not (p.ok and q.ok):
                continue
            sp, sq = p.state(threshold), q.state(threshold)
            if sp is None or sq is None:
                continue
            met, clearance = lpath_crossings(edges, p, q)
            if len(met) != 1 or clearance < min_clearance:
                continue
            edge = met[0]
            ps, qs = p.signs(threshold), q.signs(threshold)
            bad = [c for c in COMPONENTS if ps[c] != qs[c] and c != edge.component]
            if not bad:
                continue
            out.append(
                {
                    "kind": "forbidden_component_flip_on_path",
                    "components": bad,
                    "edge_id": edge.edge_id,
                    "edge_mechanism": edge.mechanism,
                    "from": [p.m1, p.m2],
                    "to": [q.m1, q.m2],
                    "state_from": list(sp),
                    "state_to": list(sq),
                    "path_clearance": clearance,
                    "detail": (
                        f"path meets only {edge.edge_id} ({edge.mechanism}) but flips {bad}"
                    ),
                }
            )
    return out


def audit_probes(
    edges: Sequence[PolylineEdge],
    probes_by_line: dict[float, list[Probe]],
    *,
    threshold: float,
    min_clearance: float,
) -> dict[str, Any]:
    """Run every check.  Pure -- takes probes, returns findings."""
    violations: list[dict[str, Any]] = []
    lines = sorted(probes_by_line)
    for m1 in lines:
        violations.extend(
            check_vertical_brackets(edges, probes_by_line[m1], threshold=threshold)
        )
        violations.extend(
            check_edge_transversals(edges, probes_by_line[m1], threshold=threshold)
        )
    for left, right in zip(lines, lines[1:], strict=False):
        violations.extend(
            check_face_consistency(
                edges,
                probes_by_line[left],
                probes_by_line[right],
                threshold=threshold,
                min_clearance=min_clearance,
            )
        )
        violations.extend(
            check_single_crossing_paths(
                edges,
                probes_by_line[left],
                probes_by_line[right],
                threshold=threshold,
                min_clearance=min_clearance,
            )
        )
    counts: dict[str, int] = {}
    for item in violations:
        counts[item["kind"]] = counts.get(item["kind"], 0) + 1
    return {"violations": violations, "violation_counts": counts}


# --------------------------------------------------------------------------
# probe placement and evaluation (the expensive half)
# --------------------------------------------------------------------------
def plan_line(
    edges: Sequence[PolylineEdge],
    m1: float,
    *,
    m2_lo: float,
    m2_hi: float,
    standoff: float,
    max_gap: float = 0.07,
) -> list[float]:
    """Standoff probes on both sides of every present edge, plus face interiors.

    ``max_gap`` matters more than it looks: a stretch of the scan line with no
    committed edge gets no probes at all from the edge-driven rules, and a
    stretch with no committed edge is precisely where an uncommitted critical
    curve would hide.  Subdividing keeps every reported bracket narrow.
    """
    present = sorted(
        (y, e.edge_id) for e in edges if not e.degenerate and (y := e.m2_at(m1)) is not None
    )
    cuts = [y for y, _ in present]
    wanted: list[float] = []
    boundaries = [m2_lo, *cuts, m2_hi]
    for y in cuts:
        wanted.extend([y - standoff, y + standoff])
    for lo, hi in zip(boundaries, boundaries[1:], strict=False):
        span = hi - lo
        if span <= 2.5 * standoff:
            continue
        pieces = max(2, math.ceil(span / max_gap))
        wanted.extend(lo + span * k / pieces for k in range(1, pieces))
    keep: list[float] = []
    for y in sorted(wanted):
        if y < m2_lo or y > m2_hi:
            continue
        # Never place a probe closer than half a standoff to a committed curve.
        if any(abs(y - c) < 0.5 * standoff for c in cuts):
            continue
        if keep and y - keep[-1] < 0.4 * standoff:
            continue
        keep.append(y)
    return keep


@contextmanager
def _cpu_budget(seconds: float):
    """Abort a probe that costs more CPU than it is worth.

    Probes are placed in regions nobody has integrated before.  A single orbit
    with a near-collision can make the adaptive integrator crawl for minutes,
    and one such probe can eat the whole audit.  A probe we could not afford is
    recorded as a failure, never as a pass.

    The budget is measured in *process CPU time*, not wall clock, so a busy
    machine shortens no probe: the audit's coverage must not depend on what else
    happens to be running.
    """
    if seconds <= 0 or threading.current_thread() is not threading.main_thread():
        yield
        return

    def _fire(_signum: int, _frame: Any) -> None:
        raise TimeoutError(f"probe exceeded {seconds:g}s CPU budget")

    previous = signal.signal(signal.SIGPROF, _fire)
    signal.setitimer(signal.ITIMER_PROF, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0.0)
        signal.signal(signal.SIGPROF, previous)


def evaluate_probes(
    m1: float,
    m2_values: Sequence[float],
    seeds: Sequence[dict[str, Any]],
    *,
    max_closure: float = MAX_CLOSURE,
    probe_budget: float = 45.0,
    verbose: bool = True,
) -> list[Probe]:
    """Correct the published family chart at each (m1, m2) and read off S.

    Reuses ``liao_family.correct_family_point`` and ``boundary.evaluate``; no
    dynamics are reimplemented here.  Probes are walked outward in m2 from the
    census root nearest the scan line so each solve is seeded by its neighbour,
    which both cuts the cost and keeps the walk on one continuation branch.
    """
    import time

    from threebody_atlas.boundary import evaluate
    from threebody_atlas.liao_family import correct_family_point

    if not m2_values:
        return []
    ordered = sorted(m2_values)
    middle = ordered[len(ordered) // 2]
    charted = [
        row
        for row in seeds
        if row.get("x1") is not None
        and row.get("v1") is not None
        and row.get("v2") is not None
        and row.get("period") is not None
    ]
    if not charted:
        raise RuntimeError(
            "no shooting-chart seeds (x1, v1, v2, period) available for this scan line"
        )
    anchor = min(
        charted,
        key=lambda r: (
            abs(float(r["masses"][0]) - m1),
            abs(float(r["masses"][1]) - middle),
        ),
    )
    anchor_guess = (
        float(anchor["x1"]),
        float(anchor["v1"]),
        float(anchor["v2"]),
        float(anchor["period"]),
    )
    anchor_m2 = float(anchor["masses"][1])
    split = min(range(len(ordered)), key=lambda i: abs(ordered[i] - anchor_m2))

    results: dict[float, Probe] = {}

    def solve(m2: float, guess: tuple[float, float, float, float]) -> tuple[Probe, tuple | None]:
        started = time.perf_counter()
        try:
            with _cpu_budget(probe_budget):
                point = correct_family_point((m1, m2, 1.0), guess)
                if not point.success or point.residual_norm > max_closure:
                    return (
                        Probe(
                            m1,
                            m2,
                            ok=False,
                            note=f"closure {point.residual_norm:.3e} > {max_closure:g}",
                            closure=point.residual_norm,
                            seconds=time.perf_counter() - started,
                        ),
                        None,
                    )
                floquet = evaluate(point).floquet
        except Exception as exc:  # noqa: BLE001 - screening must not abort the audit
            probe = Probe(m1, m2, ok=False, note=f"{type(exc).__name__}: {exc}")
            probe.seconds = time.perf_counter() - started
            return probe, None
        probe = make_probe(m1, m2, floquet.alpha, floquet.beta, closure=point.residual_norm)
        probe.seconds = time.perf_counter() - started
        probe.chart = (point.x1, point.v1, point.v2, point.period)
        return probe, probe.chart

    def walk(indices: Iterable[int]) -> None:
        guess = anchor_guess
        for i in indices:
            m2 = ordered[i]
            probe, nxt = solve(m2, guess)
            if not probe.ok and guess != anchor_guess and "TimeoutError" not in probe.note:
                # One retry from the census anchor: a marching seed can wander,
                # and a failed probe silently shrinks the audit's coverage.  A
                # probe that exhausted its budget is not retried -- that failure
                # is about the orbit, not the seed, and retrying doubles the bill.
                retry, nxt = solve(m2, anchor_guess)
                retry.seconds += probe.seconds
                probe = retry
            results[m2] = probe
            guess = nxt if nxt is not None else anchor_guess
            if verbose:
                if probe.ok:
                    print(
                        f"  probe m1={m1:.4f} m2={m2:.6f} "
                        f"G+={probe.values['G_plus']:+.4e} G-={probe.values['G_minus']:+.4e} "
                        f"D={probe.values['discriminant']:+.4e} n={probe.n_unstable} "
                        f"({probe.seconds:.1f}s)",
                        flush=True,
                    )
                else:
                    print(f"  probe m1={m1:.4f} m2={m2:.6f} FAILED: {probe.note}", flush=True)

    walk(range(split, len(ordered)))
    walk(range(split - 1, -1, -1))
    return [results[m2] for m2 in ordered]


def refine_missing_curve(
    m1: float,
    bracket: tuple[float, float],
    component: str,
    seeds: Sequence[dict[str, Any]],
    *,
    steps: int,
    probe_budget: float = 45.0,
    certify: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Bisect a missing-curve bracket, then hand it to the repository localizer.

    The bisection narrows *where* the uncommitted curve is.  The certification
    step then feeds the two bracket endpoints to
    ``critical_manifold.localize_critical_point`` -- the same localizer that
    produced the 620 committed roots -- at the frozen gates.  If that returns a
    root with ``|event| <= 2e-8`` and ``closure <= 1e-7``, the missing curve is
    not a screening artefact: it is a critical point of exactly the kind the
    graph catalogues, sitting where the graph has no edge.
    """
    mode = {v: k for k, v in MECHANISM_COMPONENT.items()}[component]
    lo, hi = bracket
    ends = evaluate_probes(m1, [lo, hi], seeds, probe_budget=probe_budget, verbose=False)
    if len(ends) != 2 or not all(p.ok for p in ends):
        return {"m1": m1, "bracket": [lo, hi], "component": component,
                "status": "endpoint_probe_failed"}
    lo_probe, hi_probe = ends
    lo_val, hi_val = lo_probe.values[component], hi_probe.values[component]
    if (lo_val > 0) == (hi_val > 0):
        return {"m1": m1, "bracket": [lo, hi], "component": component,
                "status": "bracket_not_reproduced"}
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        found = evaluate_probes(m1, [mid], seeds, probe_budget=probe_budget, verbose=False)
        if not found or not found[0].ok:
            break
        probe = found[0]
        value = probe.values[component]
        if (value > 0) == (lo_val > 0):
            lo, lo_val, lo_probe = mid, value, probe
        else:
            hi, hi_val, hi_probe = mid, value, probe

    result: dict[str, Any] = {
        "m1": m1,
        "component": component,
        "event_mode": mode,
        "bracket": [lo, hi],
        "width": hi - lo,
        "endpoint_values": [lo_val, hi_val],
        "endpoint_n_unstable": [lo_probe.n_unstable, hi_probe.n_unstable],
        "status": "refined",
        "arithmetic": "float64 screening",
    }
    if verbose:
        print(
            f"  refined missing {component} curve at m1={m1:.4f}: "
            f"m2 in [{lo:.8f}, {hi:.8f}] (n_unstable "
            f"{lo_probe.n_unstable} -> {hi_probe.n_unstable})",
            flush=True,
        )
    if certify:
        result["certification"] = certify_root(
            m1, lo_probe, hi_probe, mode, census=seeds, verbose=verbose
        )
    return result


def nearest_census_root(
    census: Sequence[dict[str, Any]], m1: float, m2: float
) -> dict[str, Any]:
    """Distance from a point to the closest committed root, for provenance."""
    best = min(
        census,
        key=lambda r: (float(r["masses"][0]) - m1) ** 2 + (float(r["masses"][1]) - m2) ** 2,
    )
    dm1 = float(best["masses"][0]) - m1
    dm2 = float(best["masses"][1]) - m2
    return {
        "cell_id": int(best["cell_id"]),
        "masses": [float(best["masses"][0]), float(best["masses"][1])],
        "event_mode": best.get("event_mode"),
        "distance": math.hypot(dm1, dm2),
        "same_m1_slice_distance_m2": abs(dm2) if abs(dm1) < 5e-4 else None,
    }


def certify_root(
    m1: float,
    lo_probe: Probe,
    hi_probe: Probe,
    mode: str,
    *,
    census: Sequence[dict[str, Any]] = (),
    verbose: bool = True,
) -> dict[str, Any]:
    """Localize a bracketed event with the repository's own certified localizer.

    Gates are the frozen ones and are never relaxed here: a certification that
    misses them is reported as a miss.
    """
    from threebody_atlas.critical_manifold import localize_critical_point
    from threebody_atlas.liao_family import FamilyPoint

    if lo_probe.chart is None or hi_probe.chart is None:
        return {"status": "no_chart"}

    def family(probe: Probe) -> FamilyPoint:
        x1, v1, v2, period = probe.chart  # type: ignore[misc]
        return FamilyPoint(
            masses=(m1, probe.m2, 1.0),
            x1=x1,
            v1=v1,
            v2=v2,
            period=period,
            residual_norm=probe.closure,
            nfev=0,
            success=True,
        )

    try:
        localized = localize_critical_point(
            family(lo_probe),
            family(hi_probe),
            event_mode=mode,  # type: ignore[arg-type]
            event_tolerance=2e-8,
            max_closure=MAX_CLOSURE,
            max_iterations=32,
        )
    except Exception as exc:  # noqa: BLE001 - a failed certification is a result
        return {"status": f"localizer_failed: {type(exc).__name__}: {exc}"}

    point = localized.sample.point
    passed = abs(localized.event_value) <= 2e-8 and point.residual_norm <= MAX_CLOSURE
    out = {
        "status": "passed" if passed else "missed_frozen_gates",
        "localizer": "threebody_atlas.critical_manifold.localize_critical_point",
        "event_mode": localized.event_mode,
        "masses": [point.masses[0], point.masses[1], point.masses[2]],
        "event_value": localized.event_value,
        "closure": point.residual_norm,
        "period": point.period,
        "x1": point.x1,
        "v1": point.v1,
        "v2": point.v2,
        "gates": {"maximum_absolute_event": 2e-8, "maximum_periodic_closure": MAX_CLOSURE},
    }
    if census:
        out["nearest_committed_root"] = nearest_census_root(
            census, point.masses[0], point.masses[1]
        )
    if verbose:
        print(
            f"    certification: {out['status']} at m2={point.masses[1]:.10f} "
            f"event={localized.event_value:.3e} closure={point.residual_norm:.3e}",
            flush=True,
        )
    return out


# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", default="research/evidence/V1_CRITICAL_GRAPH.json")
    parser.add_argument("--roots", default="research/evidence/V1_HYBRID_CRITICAL_ROOTS_2026-08-15.json")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--scan-lines",
        default=",".join(f"{x:g}" for x in DEFAULT_SCAN_LINES),
        help="comma-separated m1 values",
    )
    parser.add_argument("--standoff", type=float, default=4e-3)
    parser.add_argument("--max-gap", type=float, default=0.07)
    parser.add_argument(
        "--m2-range",
        default=None,
        help=(
            "restrict probes to lo,hi within the declared m2 domain.  Narrowing this "
            "shrinks coverage and is recorded in the artifact; it never widens a gate."
        ),
    )
    parser.add_argument("--sign-threshold", type=float, default=1e-3)
    parser.add_argument("--min-clearance", type=float, default=2e-3)
    parser.add_argument("--refine-steps", type=int, default=6)
    parser.add_argument("--max-refinements", type=int, default=6)
    parser.add_argument("--max-probes", type=int, default=400)
    parser.add_argument(
        "--probe-budget",
        type=float,
        default=45.0,
        help="process CPU seconds a single probe may consume before it is failed",
    )
    parser.add_argument("--no-certify", action="store_true")
    args = parser.parse_args()

    graph = json.loads(Path(args.graph).read_text(encoding="utf-8"))
    roots_doc = json.loads(Path(args.roots).read_text(encoding="utf-8"))
    roots = roots_doc["roots"]

    edges = edges_from_graph(graph, roots)
    domain = graph["declared_mass_domain"]
    m2_lo, m2_hi = float(domain["m2"][0]), float(domain["m2"][1])
    if args.m2_range:
        lo_text, hi_text = args.m2_range.split(",")
        m2_lo = max(m2_lo, float(lo_text))
        m2_hi = min(m2_hi, float(hi_text))
    m1_lo, m1_hi = float(domain["m1"][0]), float(domain["m1"][1])

    scan_lines = [float(x) for x in args.scan_lines.split(",") if x.strip()]
    scan_lines = [m1 for m1 in scan_lines if m1_lo <= m1 <= m1_hi]

    plan = {
        m1: plan_line(
            edges,
            m1,
            m2_lo=m2_lo,
            m2_hi=m2_hi,
            standoff=args.standoff,
            max_gap=args.max_gap,
        )
        for m1 in scan_lines
    }
    total = sum(len(v) for v in plan.values())
    if total > args.max_probes:
        raise SystemExit(f"probe plan of {total} exceeds --max-probes={args.max_probes}")
    print(f"probe plan: {total} probes on {len(scan_lines)} scan lines", flush=True)

    probes_by_line: dict[float, list[Probe]] = {}
    for m1 in scan_lines:
        print(f"scan line m1={m1}", flush=True)
        probes_by_line[m1] = evaluate_probes(
            m1, plan[m1], roots, probe_budget=args.probe_budget
        )

    report = audit_probes(
        edges,
        probes_by_line,
        threshold=args.sign_threshold,
        min_clearance=args.min_clearance,
    )

    refinements: list[dict[str, Any]] = []
    missing = [v for v in report["violations"] if v["kind"] == "missing_critical_curve"]
    seen_lines: set[float] = set()
    for item in missing:
        if len(refinements) >= args.max_refinements:
            break
        key = (item["m1"], item["component"])
        if key in seen_lines:
            continue
        seen_lines.add(key)
        refinements.append(
            refine_missing_curve(
                item["m1"],
                (item["m2_bracket"][0], item["m2_bracket"][1]),
                item["component"],
                roots,
                steps=args.refine_steps,
                probe_budget=args.probe_budget,
                certify=not args.no_certify,
            )
        )

    evaluated = [p for line in probes_by_line.values() for p in line]
    payload = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_revision": os.getenv("GITHUB_SHA"),
        "run_id": os.getenv("GITHUB_RUN_ID"),
        "python": platform.python_version(),
        "arithmetic": "float64 screening (scipy DOP853 + variational Newton)",
        "inputs": {
            "graph": args.graph,
            "graph_schema": graph.get("schema"),
            "graph_release_ready": graph.get("release_ready"),
            "roots": args.roots,
            "roots_schema": roots_doc.get("schema"),
        },
        "method": {
            "state": "S = (sgn G_plus, sgn G_minus, sgn discriminant, n_unstable)",
            "G_plus": "beta - 6*alpha + 20 = P(+2)",
            "G_minus": "beta - 2*alpha + 4 = P(-2)",
            "discriminant": "(alpha-4)^2 - 4*(beta - 4*alpha + 8)",
            "mechanism_component": MECHANISM_COMPONENT,
            "checks": [
                "missing_critical_curve: a sign flip between consecutive probes on a "
                "scan line with no committed edge of the matching mechanism between them",
                "forbidden_component_flip: crossing one committed edge flips a component "
                "that mechanism may not flip",
                "no_flip_across_edge: a committed edge across which its own component "
                "does not change sign",
                "face_state_mismatch: two probes joined by an axis-aligned path that "
                "meets no committed edge, yet with different S",
                "forbidden_component_flip_on_path: same, for paths meeting exactly one edge",
            ],
        },
        "parameters": {
            "scan_lines": scan_lines,
            "standoff": args.standoff,
            "max_gap": args.max_gap,
            "probed_m2_range": [m2_lo, m2_hi],
            "declared_m2_range": domain["m2"],
            "sign_threshold": args.sign_threshold,
            "min_clearance": args.min_clearance,
            "max_closure": MAX_CLOSURE,
            "refine_steps": args.refine_steps,
            "probe_cpu_budget_seconds": args.probe_budget,
        },
        "committed_edges": [
            {
                "edge_id": e.edge_id,
                "mechanism": e.mechanism,
                "orientation": e.orientation,
                "m1_range": [e.m1_min, e.m1_max],
                "vertices": len(e.vertices),
                "degenerate": e.degenerate,
            }
            for e in edges
        ],
        "non_graph_edges": non_graph_edges(graph, roots),
        "probe_summary": {
            "planned": total,
            "evaluated": len(evaluated),
            "converged": sum(1 for p in evaluated if p.ok),
            "failed": sum(1 for p in evaluated if not p.ok),
            "undecidable_state": sum(
                1 for p in evaluated if p.ok and p.state(args.sign_threshold) is None
            ),
            "cpu_seconds": round(sum(p.seconds for p in evaluated), 1),
        },
        "violation_counts": report["violation_counts"],
        "violations": report["violations"],
        "missing_curve_refinements": refinements,
        "probes": [p.as_json() for p in evaluated],
        "scope": (
            "This audit falsifies completeness where it fires; it cannot certify "
            "completeness where it does not.  A critical curve that both begins and "
            "ends between two adjacent scan lines, or that never separates two probes, "
            "is not sampled."
        ),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(
        {
            "output": str(out),
            "probes": payload["probe_summary"],
            "violation_counts": report["violation_counts"],
            "missing_curve_refinements": refinements,
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
