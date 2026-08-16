#!/usr/bin/env python3
"""Label-independent sweep for every sign change of the three critical events.

WHY THIS EXISTS
---------------
The 620 cells behind ``research/evidence/V1_CRITICAL_GRAPH.json`` all come from
``extract_mass_slice_brackets.transition_brackets``: adjacent rows of the frozen
Li--Li--Liao raster whose *published S/U label flips*.  That construction is
blind by design, not by resolution.  A critical curve interior to the unstable
region takes the unstable dimension 2 -> 1 without ever producing an S label, so
no pair of adjacent rows disagrees, so no bracket is emitted -- at any grid
spacing.  ``scripts/audit_sign_topology.py`` demonstrated this on 2026-08-16 by
finding seven such curves that pass the frozen gates; but it sampled only seven
scan lines chosen around the committed edges, so "seven" is a lower bound with
no denominator.

This script supplies the denominator.  It walks a lattice that is chosen from
the published raster geometry alone -- never from the published labels and never
from the committed graph -- evaluates

    G_plus       = beta - 6 alpha + 20 = P(+2)     zero <=> nontrivial lambda = +1
    G_minus      = beta - 2 alpha +  4 = P(-2)     zero <=> nontrivial lambda = -1
    discriminant = (alpha-4)^2 - 4(beta - 4 alpha + 8)   zero <=> trace roots collide

at every lattice point, enumerates every sign change of each component between
consecutive points, and hands each bracket to the repository's own
``critical_manifold.localize_critical_point`` at the frozen gates.  Each
localized root is then annotated with the question that matters:

    would the published S/U labels have bracketed this curve at all?

which is answered by looking up the two adjacent *published* rows that straddle
the localized m2 and comparing their labels.  ``census_would_bracket = false``
means the curve is structurally invisible to the census that produced the graph.

WHAT THIS IS AND IS NOT
-----------------------
It is a screening enumeration whose *certifications* are gated.  Probes are
float64; the gates ``|event| <= 2e-8`` and ``closure <= 1e-7`` are the frozen
ones and are never widened here.  A bracket whose localization misses a gate is
recorded as a miss, never as a find.  This script never writes ``release_ready``
and never touches the critical graph.

It cannot certify completeness.  A vertical scan line sees only an *odd* number
of zeros inside one lattice cell; two curves crossing the same cell cancel in
the sign test.  A curve that lives entirely between two scan lines is not
sampled at all.  The artifact records the lattice so the blind spots are
computable rather than rhetorical.

SEEDING
-------
Every lattice point is a published raster row, so each probe is corrected from
the published chart at *exactly* that mass pair.  That is why a probe here costs
~3.6 CPU s rather than the ~11 CPU s the seven-scan-line audit paid: the audit
had to march a seed in m2 from the nearest census root because its probe
placement was defined relative to the committed curves, not relative to the
raster.

SHARDING
--------
``--m1-range lo,hi`` restricts a run to one m1 window and writes one JSON.  The
lattice is selected from the *global* slice list before the window is applied,
so shard boundaries cannot change which lines get probed; a merge of any cover
of the domain is the same lattice as one unsharded run.
``scripts/merge_full_domain_sign_sweep.py`` re-derives the census from the shard
probe records rather than trusting the per-shard findings.

CHECKPOINTS
-----------
A partial artifact is rewritten atomically every ``--checkpoint-every`` probes,
at the end of every scan line, and after every certification, so a wall-clock
kill costs a handful of probes rather than a run.  ``--resume`` reloads it and
skips both completed scan lines and, within an unfinished line, the individual
lattice points already probed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

SHARD_SCHEMA = "atlas.v1.full-domain-sign-sweep-shard/1"

#: Frozen gates.  Never loosened; a localization that misses them is a miss.
MAX_EVENT = 2e-8
MAX_CLOSURE = 1e-7

#: Calibration measured on this repository with baseline-seeded probes.  Used
#: only to project cost before a run; it is never used as a result.
SECONDS_PER_PROBE = 3.6
SECONDS_PER_CERTIFICATION = 45.0

REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = REPO_ROOT / "scripts/audit_sign_topology.py"


def _load_audit_module() -> Any:
    """Reuse the audited sign-topology primitives instead of restating them.

    ``state_from_invariants``, ``unstable_count``, the committed-polyline
    geometry and the per-probe CPU budget all live in
    ``scripts/audit_sign_topology.py`` and are covered by
    ``tests/test_sign_topology.py``.  Re-typing them here would create a second
    definition of the event functions, which is exactly the failure mode this
    repository keeps finding.
    """
    spec = importlib.util.spec_from_file_location("_atlas_sign_topology", AUDIT_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - packaging error
        raise RuntimeError(f"cannot load {AUDIT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


AST = _load_audit_module()

COMPONENTS: tuple[str, ...] = AST.COMPONENTS
MECHANISM_COMPONENT: dict[str, str] = AST.MECHANISM_COMPONENT
COMPONENT_MECHANISM: dict[str, str] = {v: k for k, v in MECHANISM_COMPONENT.items()}


# --------------------------------------------------------------------------
# lattice planning (pure)
# --------------------------------------------------------------------------
def strided(values: Sequence[float], stride: int) -> list[float]:
    """Every ``stride``-th value, always keeping the first and the last.

    Keeping the last matters: the lattice must reach the edge of the published
    support, otherwise a curve that only exists near the support boundary is
    excluded by an off-by-one rather than by a stated scope.
    """
    if stride < 1:
        raise ValueError("stride must be >= 1")
    keep = list(values[::stride])
    if values and keep[-1] != values[-1]:
        keep.append(values[-1])
    return keep


@dataclass(frozen=True)
class Lattice:
    """The planned probe set, with the numbers needed to project its cost."""

    scan_lines: tuple[float, ...]
    points: tuple[tuple[float, tuple[float, ...]], ...]

    @property
    def probe_count(self) -> int:
        return sum(len(m2s) for _, m2s in self.points)

    def projected_cpu_seconds(self, expected_brackets: int) -> float:
        return (
            self.probe_count * SECONDS_PER_PROBE
            + expected_brackets * SECONDS_PER_CERTIFICATION
        )


def plan_lattice(
    m2_by_m1: dict[float, Sequence[float]],
    *,
    m1_stride: int,
    m2_stride: int,
    m1_range: tuple[float, float],
    m2_range: tuple[float, float],
) -> Lattice:
    """Choose scan lines and per-line m2 samples from the raster geometry alone.

    The stride is applied to the *global* slice list before ``m1_range`` is
    imposed, so a shard cover of the domain plans the same lattice a single run
    would.  No published label and no committed edge enters this function.
    """
    all_m1 = sorted(m2_by_m1)
    selected_m1 = [m for m in strided(all_m1, m1_stride) if m1_range[0] - 1e-12 <= m <= m1_range[1] + 1e-12]
    points: list[tuple[float, tuple[float, ...]]] = []
    for m1 in selected_m1:
        in_window = [m2 for m2 in sorted(m2_by_m1[m1]) if m2_range[0] - 1e-12 <= m2 <= m2_range[1] + 1e-12]
        m2s = strided(in_window, m2_stride)
        if len(m2s) >= 2:
            points.append((m1, tuple(m2s)))
    return Lattice(tuple(m1 for m1, _ in points), tuple(points))


# --------------------------------------------------------------------------
# sign-change enumeration (pure)
# --------------------------------------------------------------------------
def sign_changes_on_line(
    probes: Sequence[dict[str, Any]],
    *,
    sign_floor: float,
) -> list[dict[str, Any]]:
    """Every consecutive pair of converged probes across which a component flips.

    ``sign_floor`` does not gate detection -- it only labels the screening
    confidence.  Suppressing a flip because both endpoints are small would hide
    exactly the case where a critical curve passes through a lattice point, so
    every flip is reported and the certifier decides.
    """
    usable = sorted((p for p in probes if p.get("ok")), key=lambda p: p["m2"])
    out: list[dict[str, Any]] = []
    for left, right in zip(usable, usable[1:], strict=False):
        for component in COMPONENTS:
            vl, vr = left.get(component), right.get(component)
            if vl is None or vr is None:
                continue
            if not (math.isfinite(vl) and math.isfinite(vr)):
                continue
            if vl == 0.0 or vr == 0.0 or (vl > 0.0) != (vr > 0.0):
                out.append(
                    {
                        "m1": left["m1"],
                        "component": component,
                        "mechanism": COMPONENT_MECHANISM[component],
                        "m2_bracket": [left["m2"], right["m2"]],
                        "value_bracket": [vl, vr],
                        "n_unstable_bracket": [left.get("n_unstable"), right.get("n_unstable")],
                        "published_label_bracket": [
                            left.get("published_label"),
                            right.get("published_label"),
                        ],
                        "screening_confidence": (
                            "clear" if min(abs(vl), abs(vr)) > sign_floor else "marginal"
                        ),
                    }
                )
    return out


def horizontal_sign_changes(
    probes: Sequence[dict[str, Any]],
    *,
    sign_floor: float,
) -> list[dict[str, Any]]:
    """Component flips between adjacent scan lines at a shared m2.

    These cost nothing extra -- they reuse the vertical lattice -- and they see
    what a vertical scan line cannot: a critical curve steep enough in the mass
    plane to slip between two scan lines without separating any two probes of
    either one.  They are *not* localizable by
    ``localize_critical_point``, which varies m2 at fixed m1, so they are
    reported as structural evidence with no certification attached.
    """
    by_line: dict[float, dict[float, dict[str, Any]]] = defaultdict(dict)
    for probe in probes:
        if probe.get("ok"):
            by_line[probe["m1"]][probe["m2"]] = probe
    lines = sorted(by_line)
    out: list[dict[str, Any]] = []
    for left_m1, right_m1 in zip(lines, lines[1:], strict=False):
        shared = sorted(set(by_line[left_m1]) & set(by_line[right_m1]))
        for m2 in shared:
            left, right = by_line[left_m1][m2], by_line[right_m1][m2]
            for component in COMPONENTS:
                vl, vr = left.get(component), right.get(component)
                if vl is None or vr is None:
                    continue
                if not (math.isfinite(vl) and math.isfinite(vr)):
                    continue
                if vl == 0.0 or vr == 0.0 or (vl > 0.0) != (vr > 0.0):
                    out.append(
                        {
                            "m2": m2,
                            "component": component,
                            "mechanism": COMPONENT_MECHANISM[component],
                            "m1_bracket": [left_m1, right_m1],
                            "value_bracket": [vl, vr],
                            "n_unstable_bracket": [
                                left.get("n_unstable"),
                                right.get("n_unstable"),
                            ],
                            "screening_confidence": (
                                "clear" if min(abs(vl), abs(vr)) > sign_floor else "marginal"
                            ),
                        }
                    )
    return out


# --------------------------------------------------------------------------
# the census-blindness verdict (pure)
# --------------------------------------------------------------------------
def published_cell_verdict(
    labelled_rows: Sequence[tuple[float, str]], m2_star: float
) -> dict[str, Any]:
    """Would ``transition_brackets`` have emitted a cell containing ``m2_star``?

    ``labelled_rows`` is the full published (m2, label) list at this m1 -- the
    same rows ``extract_mass_slice_brackets`` consumes.  The census emits a
    bracket for a pair of adjacent rows exactly when their labels differ, so a
    critical curve is reachable by the census if and only if it falls inside
    such a pair.
    """
    rows = sorted(labelled_rows)
    if len(rows) < 2:
        return {"status": "no_published_rows", "census_would_bracket": None}
    if m2_star < rows[0][0] or m2_star > rows[-1][0]:
        return {
            "status": "outside_published_support",
            "census_would_bracket": False,
            "published_m2_support": [rows[0][0], rows[-1][0]],
        }
    index = 0
    for i in range(len(rows) - 1):
        if rows[i][0] <= m2_star <= rows[i + 1][0]:
            index = i
            break
    lo_m2, lo_label = rows[index]
    hi_m2, hi_label = rows[index + 1]
    flips = lo_label != hi_label
    transitions = [
        0.5 * (a[0] + b[0])
        for a, b in zip(rows, rows[1:], strict=False)
        if a[1] != b[1]
    ]
    nearest = min((abs(t - m2_star) for t in transitions), default=None)
    return {
        "status": "ok",
        "published_cell_m2": [lo_m2, hi_m2],
        "published_cell_labels": [lo_label, hi_label],
        "labels_flip": flips,
        "census_would_bracket": flips,
        "published_transitions_on_line": len(transitions),
        "distance_to_nearest_published_transition": nearest,
    }


def committed_edge_match(
    edges: Sequence[Any], mechanism: str, m1: float, m2_star: float, *, tolerance: float
) -> dict[str, Any]:
    """Nearest committed polyline of the same mechanism at this m1.

    Used only to *annotate* a localized root as already published or not.  It
    never participates in detection: a detector that consults the committed
    graph cannot falsify the committed graph.
    """
    candidates: list[tuple[float, Any, float]] = []
    for edge in edges:
        if edge.mechanism != mechanism or edge.degenerate:
            continue
        y = edge.m2_at(m1)
        if y is None:
            continue
        candidates.append((abs(y - m2_star), edge, y))
    if not candidates:
        return {"matched": False, "reason": "no committed edge of this mechanism at this m1"}
    distance, edge, y = min(candidates, key=lambda item: item[0])
    return {
        "matched": distance <= tolerance,
        "edge_id": edge.edge_id,
        "edge_orientation": edge.orientation,
        "edge_m2_at_m1": y,
        "distance_m2": distance,
        "match_tolerance_m2": tolerance,
    }


def gate_verdict(event_value: float, closure: float) -> bool:
    """The frozen gates, in one place, applied identically by sweep and merge."""
    return (
        math.isfinite(event_value)
        and math.isfinite(closure)
        and abs(event_value) <= MAX_EVENT
        and closure <= MAX_CLOSURE
    )


# --------------------------------------------------------------------------
# probing and certification (the expensive half)
# --------------------------------------------------------------------------
def probe_point(
    m1: float,
    m2: float,
    seed: tuple[float, float, float, float],
    published_label: str,
    *,
    probe_budget: float,
) -> dict[str, Any]:
    """Correct the published chart at (m1, m2) and read off the three events."""
    from threebody_atlas.boundary import evaluate
    from threebody_atlas.liao_family import correct_family_point

    started = time.process_time()
    record: dict[str, Any] = {
        "m1": m1,
        "m2": m2,
        "published_label": published_label,
        "ok": False,
        "note": "",
    }
    try:
        # audit_sign_topology._cpu_budget: a single near-collision orbit can make
        # the adaptive integrator crawl for minutes and eat a whole shard.  The
        # budget is process CPU time, so a busy machine shortens no probe.
        with AST._cpu_budget(probe_budget):
            point = correct_family_point((m1, m2, 1.0), seed)
            if not point.success or point.residual_norm > MAX_CLOSURE:
                record["note"] = f"closure {point.residual_norm:.3e} > {MAX_CLOSURE:g}"
                record["closure"] = point.residual_norm
                record["seconds"] = time.process_time() - started
                return record
            floquet = evaluate(point).floquet
    except Exception as exc:  # noqa: BLE001 - a failed probe is a recorded result
        record["note"] = f"{type(exc).__name__}: {exc}"
        record["seconds"] = time.process_time() - started
        return record
    alpha, beta = float(floquet.alpha), float(floquet.beta)
    record.update(AST.state_from_invariants(alpha, beta))
    record.update(
        {
            "ok": True,
            "alpha": alpha,
            "beta": beta,
            "n_unstable": AST.unstable_count(alpha, beta),
            "closure": point.residual_norm,
            "chart": [point.x1, point.v1, point.v2, point.period],
            "seconds": time.process_time() - started,
        }
    )
    return record


def certify_bracket(
    bracket: dict[str, Any],
    lo_probe: dict[str, Any],
    hi_probe: dict[str, Any],
    *,
    certify_budget: float,
) -> dict[str, Any]:
    """Localize one bracketed event with the repository's own localizer.

    Gates are the frozen ones.  A localization that misses them is reported as
    ``missed_frozen_gates``; nothing here widens a tolerance to make a finding
    land, and nothing here retries with a looser gate.
    """
    from threebody_atlas.critical_manifold import localize_critical_point
    from threebody_atlas.liao_family import FamilyPoint

    out = dict(bracket)
    started = time.process_time()

    def family(probe: dict[str, Any]) -> FamilyPoint:
        x1, v1, v2, period = probe["chart"]
        return FamilyPoint(
            masses=(probe["m1"], probe["m2"], 1.0),
            x1=x1,
            v1=v1,
            v2=v2,
            period=period,
            residual_norm=probe["closure"],
            nfev=0,
            success=True,
        )

    if lo_probe.get("chart") is None or hi_probe.get("chart") is None:
        out["status"] = "no_chart"
        return out
    try:
        with AST._cpu_budget(certify_budget):
            localized = localize_critical_point(
                family(lo_probe),
                family(hi_probe),
                event_mode=bracket["mechanism"],
                event_tolerance=MAX_EVENT,
                max_closure=MAX_CLOSURE,
                max_iterations=32,
            )
    except Exception as exc:  # noqa: BLE001 - a failed localization is a result
        out["status"] = f"localizer_failed: {type(exc).__name__}: {exc}"
        out["seconds"] = time.process_time() - started
        return out
    point = localized.sample.point
    out.update(
        {
            "status": "passed" if gate_verdict(localized.event_value, point.residual_norm) else "missed_frozen_gates",
            "localizer": "threebody_atlas.critical_manifold.localize_critical_point",
            "event_mode": localized.event_mode,
            "masses": [point.masses[0], point.masses[1], point.masses[2]],
            "event_value": float(localized.event_value),
            "closure": float(point.residual_norm),
            "period": point.period,
            "x1": point.x1,
            "v1": point.v1,
            "v2": point.v2,
            "gates": {"maximum_absolute_event": MAX_EVENT, "maximum_periodic_closure": MAX_CLOSURE},
            "seconds": time.process_time() - started,
        }
    )
    return out


# --------------------------------------------------------------------------
# baseline access
# --------------------------------------------------------------------------
def load_baseline(path: Path) -> tuple[dict[float, dict[float, Any]], dict[str, str]]:
    """Rows keyed by (m1, m2), plus content digests for provenance."""
    from threebody_atlas.baseline import iter_baseline

    raw = path.read_bytes()
    digests = {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "git_blob_sha1": hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest(),
        "bytes": str(len(raw)),
    }
    grid: dict[float, dict[float, Any]] = defaultdict(dict)
    for row in iter_baseline(path):
        grid[row.m1][row.m2] = row
    return dict(grid), digests


# --------------------------------------------------------------------------
# the shard artifact
# --------------------------------------------------------------------------
def shard_document(
    phase: str,
    *,
    probes: Sequence[dict[str, Any]],
    localizations: Sequence[dict[str, Any]],
    planned_scan_lines: Sequence[float],
    completed_scan_lines: Iterable[float],
    sign_floor: float,
    metadata: dict[str, Any],
    wall_seconds: float,
) -> dict[str, Any]:
    """Assemble the shard artifact.

    Built as a function rather than inline so the artifact's *shape* is
    testable.  During development this document carried its brackets, its
    localizations and a probe summary but not the probe records themselves: it
    read correctly to a human and merged to an empty census, because the merge
    step re-derives from the records and there were none.  Everything except
    ``probes`` here is a conclusion; ``probes`` is the evidence.
    """
    converged = [p for p in probes if p.get("ok")]
    return {
        "schema": SHARD_SCHEMA,
        "phase": phase,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_revision": os.getenv("GITHUB_SHA"),
        "run_id": os.getenv("GITHUB_RUN_ID"),
        "python": platform.python_version(),
        "arithmetic": "float64 screening (scipy DOP853 + variational Newton)",
        "shard": metadata["shard"],
        "inputs": metadata["inputs"],
        "method": {
            "detection": (
                "sign change of G_plus / G_minus / discriminant between consecutive "
                "lattice points; the lattice is chosen from the published raster "
                "geometry only, never from the published S/U labels and never from "
                "the committed critical graph"
            ),
            "G_plus": "beta - 6*alpha + 20 = P(+2)",
            "G_minus": "beta - 2*alpha + 4 = P(-2)",
            "discriminant": "(alpha-4)^2 - 4*(beta - 4*alpha + 8)",
            "certification": "threebody_atlas.critical_manifold.localize_critical_point",
            "census_blindness_test": (
                "census_would_bracket is true iff the two adjacent published raster "
                "rows straddling the localized m2 carry different S/U labels, which "
                "is exactly when extract_mass_slice_brackets.transition_brackets "
                "would have emitted a cell there"
            ),
        },
        "gates": {
            "maximum_absolute_event": MAX_EVENT,
            "maximum_periodic_closure": MAX_CLOSURE,
            "note": "frozen; a localization that misses a gate is recorded as a miss",
        },
        "cost_projection": metadata.get("cost_projection"),
        "planned_scan_lines": list(planned_scan_lines),
        "completed_scan_lines": sorted(completed_scan_lines),
        "probe_summary": {
            "planned": metadata.get("probes_planned"),
            "evaluated": len(probes),
            "converged": len(converged),
            "failed": len(probes) - len(converged),
            "undecidable_n_unstable": sum(1 for p in converged if p.get("n_unstable") is None),
            "cpu_seconds": round(sum(p.get("seconds", 0.0) for p in probes), 1),
        },
        "vertical_sign_changes": _all_vertical(probes, sign_floor=sign_floor),
        "horizontal_sign_changes": horizontal_sign_changes(probes, sign_floor=sign_floor),
        "localizations": list(localizations),
        "probes": list(probes),
        "wall_seconds": round(wall_seconds, 1),
        "scope": (
            "A vertical scan line detects only an odd number of zeros inside one "
            "lattice cell; a pair of curves crossing the same cell cancels.  A curve "
            "that both begins and ends between two scan lines is not sampled.  This "
            "artifact records the lattice so those blind spots are computable."
        ),
    }


# --------------------------------------------------------------------------
# checkpointing
# --------------------------------------------------------------------------
def write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", required=True, help="frozen Li-Li-Liao supplementary raster")
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default=None, help="default: <output>.partial")
    parser.add_argument("--graph", default="research/evidence/V1_CRITICAL_GRAPH.json")
    parser.add_argument("--roots", default="research/evidence/V1_HYBRID_CRITICAL_ROOTS_2026-08-15.json")
    parser.add_argument("--m1-range", default="0.8,1.1")
    parser.add_argument("--m2-range", default="0.7,1.2")
    parser.add_argument("--m1-stride", type=int, default=10, help="keep every Nth published m1 slice")
    parser.add_argument("--m2-stride", type=int, default=5, help="keep every Nth published m2 row")
    parser.add_argument("--shard-label", default="local")
    parser.add_argument("--sign-floor", type=float, default=1e-4)
    parser.add_argument("--match-tolerance", type=float, default=1.5e-3)
    parser.add_argument("--probe-budget", type=float, default=45.0)
    parser.add_argument("--certify-budget", type=float, default=600.0)
    parser.add_argument("--max-certifications", type=int, default=0, help="0 = no limit")
    parser.add_argument("--no-certify", action="store_true")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=8,
        help="rewrite the checkpoint after this many probes (0 = only at line ends)",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true", help="print the cost projection and exit")
    args = parser.parse_args()

    m1_lo, m1_hi = (float(x) for x in args.m1_range.split(","))
    m2_lo, m2_hi = (float(x) for x in args.m2_range.split(","))

    graph = json.loads(Path(args.graph).read_text(encoding="utf-8"))
    roots_doc = json.loads(Path(args.roots).read_text(encoding="utf-8"))
    edges = AST.edges_from_graph(graph, roots_doc["roots"])
    declared = graph["declared_mass_domain"]
    m1_lo = max(m1_lo, float(declared["m1"][0]))
    m1_hi = min(m1_hi, float(declared["m1"][1]))
    m2_lo = max(m2_lo, float(declared["m2"][0]))
    m2_hi = min(m2_hi, float(declared["m2"][1]))

    grid, digests = load_baseline(Path(args.baseline))
    m2_by_m1 = {m1: sorted(rows) for m1, rows in grid.items()}
    lattice = plan_lattice(
        m2_by_m1,
        m1_stride=args.m1_stride,
        m2_stride=args.m2_stride,
        m1_range=(m1_lo, m1_hi),
        m2_range=(m2_lo, m2_hi),
    )

    # Cost projection.  The bracket estimate comes from the only measurement we
    # have: the seven-scan-line audit found 3-4 sign changes per line.
    expected_brackets = 4 * len(lattice.scan_lines)
    projection = {
        "scan_lines": len(lattice.scan_lines),
        "probes_planned": lattice.probe_count,
        "seconds_per_probe_calibration": SECONDS_PER_PROBE,
        "seconds_per_certification_calibration": SECONDS_PER_CERTIFICATION,
        "expected_brackets": expected_brackets,
        "projected_cpu_seconds": round(lattice.projected_cpu_seconds(expected_brackets), 1),
        "projected_cpu_hours": round(lattice.projected_cpu_seconds(expected_brackets) / 3600.0, 2),
    }
    print(json.dumps({"cost_projection": projection}, indent=2), flush=True)
    if args.plan_only:
        return

    checkpoint = Path(args.checkpoint) if args.checkpoint else Path(str(args.output) + ".partial")

    probes: list[dict[str, Any]] = []
    localizations: list[dict[str, Any]] = []
    done_lines: set[float] = set()
    done_brackets: set[str] = set()
    done_points: set[tuple[float, float]] = set()
    if args.resume and checkpoint.exists():
        prior = json.loads(checkpoint.read_text(encoding="utf-8"))
        probes = list(prior.get("probes", ()))
        localizations = list(prior.get("localizations", ()))
        done_lines = {float(m1) for m1 in prior.get("completed_scan_lines", ())}
        done_brackets = {_bracket_key(item) for item in localizations}
        done_points = {(float(p["m1"]), float(p["m2"])) for p in probes}
        print(
            f"resumed from {checkpoint}: {len(probes)} probes, "
            f"{len(done_lines)} completed lines, {len(localizations)} localizations",
            flush=True,
        )

    started_wall = time.time()

    metadata = {
        "shard": {
            "label": args.shard_label,
            "m1_range": [m1_lo, m1_hi],
            "m2_range": [m2_lo, m2_hi],
            "m1_stride": args.m1_stride,
            "m2_stride": args.m2_stride,
            "sign_floor": args.sign_floor,
            "match_tolerance_m2": args.match_tolerance,
            "probe_cpu_budget_seconds": args.probe_budget,
            "certify_cpu_budget_seconds": args.certify_budget,
            "certified": not args.no_certify,
            "max_certifications": args.max_certifications,
            "checkpoint_every_probes": args.checkpoint_every,
        },
        "inputs": {
            "baseline": str(args.baseline),
            "baseline_digests": digests,
            "graph": args.graph,
            "graph_schema": graph.get("schema"),
            "graph_release_ready": graph.get("release_ready"),
            "roots": args.roots,
            "roots_schema": roots_doc.get("schema"),
        },
        "cost_projection": projection,
        "probes_planned": lattice.probe_count,
    }

    def payload(phase: str) -> dict[str, Any]:
        return shard_document(
            phase,
            probes=probes,
            localizations=localizations,
            planned_scan_lines=lattice.scan_lines,
            completed_scan_lines=done_lines,
            sign_floor=args.sign_floor,
            metadata=metadata,
            wall_seconds=time.time() - started_wall,
        )

    # ---------------- phase A: probe the lattice -------------------------
    for m1, m2s in lattice.points:
        if m1 in done_lines:
            continue
        pending = [m2 for m2 in m2s if (m1, m2) not in done_points]
        print(
            f"scan line m1={m1:.4f}: {len(pending)} probes "
            f"({len(m2s) - len(pending)} already on the checkpoint)",
            flush=True,
        )
        for index, m2 in enumerate(pending, start=1):
            row = grid[m1][m2]
            record = probe_point(
                m1,
                m2,
                (row.x1, row.v1, row.v2, row.period),
                row.published_stability,
                probe_budget=args.probe_budget,
            )
            probes.append(record)
            done_points.add((m1, m2))
            if args.checkpoint_every and index % args.checkpoint_every == 0:
                write_atomic(checkpoint, payload("probing"))
            if record["ok"]:
                print(
                    f"  m1={m1:.4f} m2={m2:.4f} [{record['published_label']}] "
                    f"G+={record['G_plus']:+.4e} G-={record['G_minus']:+.4e} "
                    f"D={record['discriminant']:+.4e} n={record['n_unstable']} "
                    f"({record['seconds']:.1f}s)",
                    flush=True,
                )
            else:
                print(f"  m1={m1:.4f} m2={m2:.4f} FAILED: {record['note']}", flush=True)
        done_lines.add(m1)
        write_atomic(checkpoint, payload("probing"))

    # ---------------- phase B: enumerate brackets (pure) ------------------
    brackets = _all_vertical(probes, sign_floor=args.sign_floor)
    print(
        f"\nlattice complete: {len(probes)} probes, {len(brackets)} vertical sign changes",
        flush=True,
    )

    # ---------------- phase C: certify each bracket ----------------------
    if not args.no_certify:
        by_key = {(p["m1"], p["m2"]): p for p in probes}
        labelled = {
            m1: [(m2, grid[m1][m2].published_stability) for m2 in m2_by_m1[m1]]
            for m1 in {b["m1"] for b in brackets}
        }
        for bracket in brackets:
            key = _bracket_key(bracket)
            if key in done_brackets:
                continue
            if args.max_certifications and len(localizations) >= args.max_certifications:
                print("certification budget exhausted; remaining brackets left uncertified", flush=True)
                break
            lo = by_key[(bracket["m1"], bracket["m2_bracket"][0])]
            hi = by_key[(bracket["m1"], bracket["m2_bracket"][1])]
            result = certify_bracket(bracket, lo, hi, certify_budget=args.certify_budget)
            if "masses" in result:
                m2_star = result["masses"][1]
                result["published_cell"] = published_cell_verdict(labelled[bracket["m1"]], m2_star)
                result["committed_edge"] = committed_edge_match(
                    edges,
                    bracket["mechanism"],
                    result["masses"][0],
                    m2_star,
                    tolerance=args.match_tolerance,
                )
            localizations.append(result)
            done_brackets.add(key)
            blind = (
                result.get("published_cell", {}).get("census_would_bracket") is False
                and result.get("status") == "passed"
            )
            print(
                f"  localize {bracket['mechanism']} at m1={bracket['m1']:.4f} -> "
                f"{result.get('status')} "
                f"m2={result.get('masses', [None, None])[1]} "
                f"event={result.get('event_value')} closure={result.get('closure')} "
                f"committed={result.get('committed_edge', {}).get('matched')} "
                f"census_blind={blind}",
                flush=True,
            )
            write_atomic(checkpoint, payload("certifying"))

    final = payload("complete")
    write_atomic(Path(args.output), final)
    passed = [item for item in localizations if item.get("status") == "passed"]
    blind = [
        item
        for item in passed
        if item.get("published_cell", {}).get("census_would_bracket") is False
    ]
    uncommitted = [item for item in passed if not item.get("committed_edge", {}).get("matched")]
    print(
        json.dumps(
            {
                "output": str(args.output),
                "probe_summary": final["probe_summary"],
                "vertical_sign_changes": len(brackets),
                "horizontal_sign_changes": len(final["horizontal_sign_changes"]),
                "localizations": len(localizations),
                "passed_frozen_gates": len(passed),
                "passed_and_not_in_committed_graph": len(uncommitted),
                "passed_and_invisible_to_published_labels": len(blind),
            },
            indent=2,
        )
    )


def _bracket_key(item: dict[str, Any]) -> str:
    return f"{item.get('m1')}|{item.get('component')}|{item.get('m2_bracket')}"


def _all_vertical(probes: Iterable[dict[str, Any]], *, sign_floor: float) -> list[dict[str, Any]]:
    by_line: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for probe in probes:
        by_line[probe["m1"]].append(probe)
    out: list[dict[str, Any]] = []
    for m1 in sorted(by_line):
        out.extend(sign_changes_on_line(by_line[m1], sign_floor=sign_floor))
    return out


if __name__ == "__main__":
    main()
