#!/usr/bin/env python3
"""Quantify the published-label bracket criterion against the event-sign criterion.

WHY THIS EXISTS
---------------
Gate B was falsified by ``scripts/audit_sign_topology.py``, which found seven
critical curves absent from the committed seven-polyline graph.  The root cause is
upstream of the graph: every census root reached the localizer through
``extract_mass_slice_brackets``, which brackets an adjacent published-row pair
only when the *published S/U label* flips.  A critical curve interior to the
unstable region flips no label -- the unstable dimension steps 2 -> 1, both sides
read ``U`` -- so it yields no bracket at any grid resolution.

``threebody_atlas.bracket_criteria`` now offers both criteria under explicit
names.  This script measures the difference on the real frozen baseline: for each
requested m1 slice it evaluates the three Floquet event functions at every
published row in an m2 window, then reports

  * how many brackets each criterion produces on exactly the same rows;
  * every bracket the event criterion finds that no label criterion can (the
    published label does not flip across it), with the unstable dimensions on
    both sides;
  * for each such bracket, optionally, a certification by the repository's own
    ``critical_manifold.localize_critical_point`` at the frozen gates
    ``|event| <= 2e-8`` and ``closure <= 1e-7``;
  * a cross-check against the seven curves already reported by the sign-topology
    audit, read out of the shipped evidence rather than transcribed;
  * for each certified label-invisible root, its distance to the nearest
    committed ``mechanism_polyline`` of the matching mechanism, so a root that
    merely re-finds a committed curve is not miscounted as new.

WHAT THIS IS NOT
----------------
A census.  It does not regenerate the 620-cell census, does not touch
``V1_CRITICAL_GRAPH.json``, and never writes ``release_ready``.  The census stands
as the frozen record of what the label criterion could see; this script measures
what it could not.

Gates are never widened.  A bracket whose certification misses the frozen gates
is reported as a miss.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from threebody_atlas.baseline import BaselineRow, iter_baseline  # noqa: E402
from threebody_atlas.bracket_criteria import (  # noqa: E402
    EVENT_COMPONENTS,
    MAX_CLOSURE,
    EventBracket,
    RowEvent,
    evaluate_row,
    event_sign_brackets,
    label_invisible_brackets,
    published_label_brackets,
)
from threebody_atlas.evidence_semantics import artifact_semantics  # noqa: E402

SCHEMA = "atlas.v1.bracket-criterion-comparison/1"
EVENT_TOLERANCE = 2e-8
ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SLICES = (0.900, 0.920, 0.925, 0.929, 0.930, 0.931, 0.940, 0.970, 1.000, 1.040)

#: The two shipped sign-topology artifacts whose certified roots we cross-check.
AUDIT_EVIDENCE = (
    "research/evidence/V1_SIGN_TOPOLOGY_AUDIT_2026-08-16.json",
    "research/evidence/V1_SIGN_TOPOLOGY_CROSSING_2026-08-16.json",
)


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------
def known_missing_curves(paths: Sequence[str]) -> list[dict[str, Any]]:
    """Read the already-certified missing curves out of the shipped evidence."""
    out: list[dict[str, Any]] = []
    for rel in paths:
        path = ROOT / rel
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for item in payload.get("missing_curve_refinements", []):
            cert = item.get("certification") or {}
            if cert.get("status") != "passed":
                continue
            out.append(
                {
                    "source": rel,
                    "m1": float(cert["masses"][0]),
                    "m2": float(cert["masses"][1]),
                    "event_mode": cert.get("event_mode") or item.get("event_mode"),
                    "endpoint_n_unstable": item.get("endpoint_n_unstable"),
                }
            )
    return out


def committed_edges() -> list[Any]:
    """Rebuild the committed polylines using the audit script's own geometry."""
    graph_path = ROOT / "research/evidence/V1_CRITICAL_GRAPH.json"
    roots_path = ROOT / "research/evidence/V1_HYBRID_CRITICAL_ROOTS_2026-08-15.json"
    if not (graph_path.exists() and roots_path.exists()):
        return []
    spec = importlib.util.spec_from_file_location(
        "audit_sign_topology_for_criteria", ROOT / "scripts/audit_sign_topology.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    roots = json.loads(roots_path.read_text(encoding="utf-8"))
    return module.edges_from_graph(graph, roots["roots"])


def nearest_committed_edge(edges: Sequence[Any], mechanism: str, m1: float, m2: float):
    """Vertical distance to the nearest committed edge of the same mechanism."""
    best: dict[str, Any] | None = None
    for edge in edges:
        if edge.mechanism != mechanism:
            continue
        y = edge.m2_at(m1)
        if y is None:
            continue
        gap = abs(y - m2)
        if best is None or gap < best["m2_gap"]:
            best = {"edge_id": edge.edge_id, "committed_m2": y, "m2_gap": gap}
    return best


# --------------------------------------------------------------------------
# per-slice comparison
# --------------------------------------------------------------------------
def bracket_record(m1: float, bracket: EventBracket) -> dict[str, Any]:
    left, right = bracket.values
    return {
        "m1": m1,
        "component": bracket.component,
        "event_mode": bracket.event_mode,
        "m2_bracket": list(bracket.m2_bracket),
        "published_labels": [
            bracket.left.row.published_stability,
            bracket.right.row.published_stability,
        ],
        "label_flip": bracket.label_flip,
        "interior_to_unstable_region": bracket.interior_to_unstable_region,
        "n_unstable": [bracket.left.n_unstable, bracket.right.n_unstable],
        "event_values": [left, right],
        "closure": [bracket.left.closure, bracket.right.closure],
    }


def historical_reachability(brackets: Sequence[EventBracket]) -> dict[int, dict[str, Any]]:
    """Decide, per event bracket, whether the historical pipeline could reach it.

    Two independent losses, not one:

    1. ``no_label_bracket`` -- the published label does not flip across the cell,
       so ``extract_mass_slice_brackets`` emitted nothing at all.  This is the
       blindness that falsified Gate B.

    2. ``shadowed_by_single_mode_selection`` -- the label *does* flip, so a cell
       existed, but two or three event functions change sign on that same cell
       and ``critical_manifold.infer_event_mode`` returns exactly one mode: the
       crossing whose endpoint magnitudes are smallest.  The census then contains
       one root for a cell that carries two critical curves.  This is what an
       ``X`` crossing of two mechanisms looks like from inside the old pipeline,
       and it is why the census's own cell at such a crossing is unclassifiable
       as an edge endpoint.

    The selection rule is mirrored from ``infer_event_mode`` deliberately: the
    claim "the historical pipeline would have missed this" is only honest if it
    is evaluated with the historical pipeline's own choice function.
    """
    by_cell: dict[tuple[float, float], list[int]] = {}
    for index, bracket in enumerate(brackets):
        by_cell.setdefault(bracket.m2_bracket, []).append(index)

    out: dict[int, dict[str, Any]] = {}
    for cell, indices in by_cell.items():
        label_flip = brackets[indices[0]].label_flip
        # infer_event_mode: among the modes that bracket zero, the one whose
        # larger endpoint magnitude is smallest.
        chosen = min(
            indices,
            key=lambda i: max(abs(v) for v in brackets[i].values),
        )
        for index in indices:
            if not label_flip:
                reason = "no_label_bracket"
            elif index != chosen:
                reason = "shadowed_by_single_mode_selection"
            else:
                reason = None
            out[index] = {
                "reachable_by_published_label_pipeline": reason is None,
                "unreachable_reason": reason,
                "events_crossing_this_cell": len(indices),
                "cell": list(cell),
            }
    return out


def certify(m1: float, bracket: EventBracket, *, verbose: bool) -> dict[str, Any]:
    """Localize a label-invisible bracket with the repository's own localizer.

    Same function, same frozen gates that produced the 620 committed roots.  Note
    that ``localize_critical_point`` names its endpoints ``stable``/``unstable``
    for historical reasons but only uses their m2 ordering and the event values,
    so a U/U bracket is a legitimate input once ``event_mode`` is stated.
    """
    from threebody_atlas.critical_manifold import localize_critical_point
    from threebody_atlas.liao_family import correct_family_point

    started = time.perf_counter()
    try:
        left = correct_family_point(
            (m1, bracket.left.m2, bracket.left.row.m3),
            bracket.left.chart
            or (
                bracket.left.row.x1,
                bracket.left.row.v1,
                bracket.left.row.v2,
                bracket.left.row.period,
            ),
            max_nfev=60,
        )
        right = correct_family_point(
            (m1, bracket.right.m2, bracket.right.row.m3),
            bracket.right.chart
            or (
                bracket.right.row.x1,
                bracket.right.row.v1,
                bracket.right.row.v2,
                bracket.right.row.period,
            ),
            max_nfev=60,
        )
        localized = localize_critical_point(
            left,
            right,
            event_mode=bracket.event_mode,  # type: ignore[arg-type]
            event_tolerance=EVENT_TOLERANCE,
            max_closure=MAX_CLOSURE,
        )
    except Exception as exc:  # noqa: BLE001 - a failed certification is a result
        return {
            "status": f"localizer_failed: {type(exc).__name__}: {exc}",
            "seconds": time.perf_counter() - started,
        }
    point = localized.sample.point
    passed = abs(localized.event_value) <= EVENT_TOLERANCE and point.residual_norm <= MAX_CLOSURE
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
        "gates": {"maximum_absolute_event": EVENT_TOLERANCE, "maximum_periodic_closure": MAX_CLOSURE},
        "seconds": time.perf_counter() - started,
    }
    if verbose:
        print(
            f"    certification {out['status']}: m2={point.masses[1]:.10f} "
            f"event={localized.event_value:.3e} closure={point.residual_norm:.3e} "
            f"({out['seconds']:.0f}s)",
            flush=True,
        )
    return out


def evaluate_slice(rows: Sequence[BaselineRow], *, jobs: int) -> list[RowEvent]:
    worker = partial(evaluate_row, correct=True)
    if jobs <= 1:
        return [worker(row) for row in rows]
    from multiprocessing import get_context

    with get_context("spawn").Pool(jobs) as pool:
        return pool.map(worker, rows, chunksize=1)


def compare_slice(
    m1: float,
    rows: Sequence[BaselineRow],
    *,
    jobs: int,
    do_certify: bool,
    verbose: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    states = evaluate_slice(rows, jobs=jobs)
    ok = [s for s in states if s.ok]
    label_pairs = published_label_brackets(list(rows))
    # The label criterion must be measured on the same rows, otherwise the
    # comparison silently credits it with brackets outside the evaluated window.
    label_cells = [(a.m2, b.m2) for a, b in label_pairs]
    events = event_sign_brackets(states)
    invisible = label_invisible_brackets(events)
    reach = historical_reachability(events)
    unreachable = [i for i, info in reach.items() if not info["reachable_by_published_label_pipeline"]]

    if verbose:
        print(
            f"  m1={m1:.4f}: rows={len(rows)} evaluated={len(ok)} "
            f"published_label_brackets={len(label_cells)} "
            f"event_sign_brackets={len(events)} label_invisible={len(invisible)} "
            f"unreachable_by_old_pipeline={len(unreachable)} "
            f"({time.perf_counter() - started:.0f}s)",
            flush=True,
        )

    records = [bracket_record(m1, b) for b in events]
    for index, record in enumerate(records):
        record.update(reach[index])
    if do_certify:
        for index in unreachable:
            records[index]["certification"] = certify(m1, events[index], verbose=verbose)

    # Every label bracket should also be an event bracket: a label flip means
    # n_unstable changed, which cannot happen without some event changing sign.
    event_cells = {tuple(r["m2_bracket"]) for r in records}
    unmatched = [cell for cell in label_cells if cell not in event_cells]

    return {
        "m1": m1,
        "rows_in_window": len(rows),
        "rows_evaluated": len(ok),
        "rows_failing_closure": [
            {"m2": s.m2, "closure": s.closure, "note": s.note} for s in states if not s.ok
        ],
        "published_label_bracket_count": len(label_cells),
        "published_label_brackets": [list(c) for c in label_cells],
        "event_sign_bracket_count": len(events),
        "label_invisible_bracket_count": len(invisible),
        "unreachable_by_published_label_pipeline_count": len(unreachable),
        "unreachable_reason_counts": dict(
            Counter(
                info["unreachable_reason"]
                for info in reach.values()
                if info["unreachable_reason"]
            )
        ),
        "cells_carrying_more_than_one_event": sorted(
            {
                tuple(info["cell"])
                for info in reach.values()
                if info["events_crossing_this_cell"] > 1
            }
        ),
        "label_brackets_without_an_event_sign_change": [list(c) for c in unmatched],
        # Sanity on the input column itself: does the published S/U label agree
        # with the label our own recomputed invariants imply?  A disagreement
        # would mean the census's blind spot is compounded by a mislabelled row,
        # so it is reported rather than assumed away.
        "rows_disagreeing_with_published_label": [
            {
                "m2": s.m2,
                "published": s.row.published_stability,
                "recomputed": s.screening_label,
                "n_unstable": s.n_unstable,
            }
            for s in states
            if s.ok and s.screening_label is not None
            and s.screening_label != s.row.published_stability
        ],
        "component_counts": dict(Counter(r["component"] for r in records)),
        "brackets": records,
        "seconds": time.perf_counter() - started,
    }


# --------------------------------------------------------------------------
# cross-check against the already-certified seven
# --------------------------------------------------------------------------
def cross_check(
    slices: Sequence[dict[str, Any]],
    known: Sequence[dict[str, Any]],
    *,
    m1_slack: float = 1.5e-3,
) -> list[dict[str, Any]]:
    """Match each already-certified missing curve to a label-invisible bracket.

    Three of the seven sit at m1 = 0.9295 / 0.9305, which are *not* baseline grid
    lines (the published m1 grid has spacing 0.001).  A criterion that runs on
    published rows therefore cannot reproduce them at their own m1; the honest
    check is whether the same curve, same mechanism, is caught on an adjacent
    on-grid slice at a nearby m2.  ``m1_slack`` is that adjacency, and every match
    records the m1 offset it used.
    """
    out = []
    for item in known:
        candidates = []
        for entry in slices:
            if abs(entry["m1"] - item["m1"]) > m1_slack:
                continue
            for record in entry["brackets"]:
                if record["reachable_by_published_label_pipeline"]:
                    continue
                if record["event_mode"] != item["event_mode"]:
                    continue
                lo, hi = record["m2_bracket"]
                # Allow one published cell of slack in m2: the curve moves with m1,
                # so an adjacent slice catches it in a neighbouring cell.
                if lo - 1.5e-3 <= item["m2"] <= hi + 1.5e-3:
                    candidates.append(record)
        matched = sorted(candidates, key=lambda r: abs(r["m1"] - item["m1"]))
        out.append(
            {
                **item,
                "on_published_m1_grid": any(
                    abs(item["m1"] - entry["m1"]) < 1e-9 for entry in slices
                ),
                "matched": bool(matched),
                "matches": [
                    {
                        "m1": r["m1"],
                        "m2_bracket": r["m2_bracket"],
                        "event_mode": r["event_mode"],
                        "n_unstable": r["n_unstable"],
                        "published_labels": r["published_labels"],
                        "m1_offset": r["m1"] - item["m1"],
                        "unreachable_reason": r["unreachable_reason"],
                        "certification_status": (r.get("certification") or {}).get("status"),
                        "certified_m2": ((r.get("certification") or {}).get("masses") or [None, None])[1],
                    }
                    for r in matched
                ],
            }
        )
    return out


def revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("dataset")
    parser.add_argument("output")
    parser.add_argument(
        "--m1",
        type=float,
        action="append",
        help=f"m1 slice to compare, repeatable (default: {DEFAULT_SLICES})",
    )
    parser.add_argument("--m2-min", type=float, default=0.75)
    parser.add_argument("--m2-max", type=float, default=1.05)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--no-certify",
        action="store_true",
        help="skip the localizer certification of label-invisible brackets",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    wanted = tuple(args.m1) if args.m1 else DEFAULT_SLICES
    grouped: dict[float, list[BaselineRow]] = {}
    for row in iter_baseline(args.dataset):
        match = next((m for m in wanted if abs(row.m1 - m) < 1e-9), None)
        if match is None:
            continue
        if args.m2_min - 1e-12 <= row.m2 <= args.m2_max + 1e-12:
            grouped.setdefault(match, []).append(row)
    missing = [m for m in wanted if m not in grouped]
    if missing:
        raise SystemExit(f"requested m1 slices absent from the baseline grid: {missing}")

    verbose = not args.quiet
    started = time.perf_counter()
    slices = []
    for m1 in sorted(grouped):
        slices.append(
            compare_slice(
                m1,
                sorted(grouped[m1], key=lambda r: r.m2),
                jobs=args.jobs,
                do_certify=not args.no_certify,
                verbose=verbose,
            )
        )

    edges = committed_edges()
    for entry in slices:
        for record in entry["brackets"]:
            cert = record.get("certification") or {}
            if cert.get("status") != "passed":
                continue
            record["nearest_committed_edge"] = nearest_committed_edge(
                edges, record["event_mode"], cert["masses"][0], cert["masses"][1]
            )

    known = known_missing_curves(AUDIT_EVIDENCE)
    checks = cross_check(slices, known)

    label_total = sum(e["published_label_bracket_count"] for e in slices)
    event_total = sum(e["event_sign_bracket_count"] for e in slices)
    invisible_total = sum(e["label_invisible_bracket_count"] for e in slices)
    unreachable_records = [
        r for e in slices for r in e["brackets"] if not r["reachable_by_published_label_pipeline"]
    ]
    certified = [
        r for r in unreachable_records if (r.get("certification") or {}).get("status") == "passed"
    ]
    missed = [
        r
        for r in unreachable_records
        if (r.get("certification") or {}).get("status") not in (None, "passed")
    ]
    reason_totals = Counter(r["unreachable_reason"] for r in unreachable_records)

    payload = {
        "schema": SCHEMA,
        "search_semantics": artifact_semantics(
            Path(__file__).resolve().parents[1], "event_sign_brackets/v1"
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_revision": revision(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "arithmetic": "float64 screening (scipy DOP853 + variational Newton)",
        "inputs": {
            "dataset": args.dataset,
            "audit_evidence": list(AUDIT_EVIDENCE),
            "committed_graph": "research/evidence/V1_CRITICAL_GRAPH.json",
        },
        "method": {
            "published_label_criterion": "bracket adjacent published rows whose S/U "
            "label flips (threebody_atlas.bracket_criteria.published_label_brackets); "
            "the criterion that produced the 620-cell census",
            "event_sign_criterion": "bracket adjacent published rows across which "
            "G_plus = beta-6*alpha+20, G_minus = beta-2*alpha+4 or "
            "discriminant = (alpha-4)^2-4*(beta-4*alpha+8) changes sign "
            "(threebody_atlas.bracket_criteria.event_sign_brackets)",
            "label_invisible": "an event-sign bracket across which the published label "
            "does not flip; no S/U-label criterion can produce it at any resolution",
            "unreachable_by_published_label_pipeline": "an event-sign bracket the "
            "historical pipeline could not have turned into a root, either because "
            "no label flipped on that cell (no_label_bracket) or because the cell "
            "carries more than one crossing event and infer_event_mode selects "
            "exactly one (shadowed_by_single_mode_selection)",
            "certification": "threebody_atlas.critical_manifold.localize_critical_point "
            "at the frozen gates |event| <= 2e-8 and closure <= 1e-7",
        },
        "parameters": {
            "m1_slices": list(wanted),
            "m2_window": [args.m2_min, args.m2_max],
            "components": list(EVENT_COMPONENTS),
            "jobs": args.jobs,
            "certified": not args.no_certify,
        },
        "totals": {
            "published_label_brackets": label_total,
            "event_sign_brackets": event_total,
            "label_invisible_brackets": invisible_total,
            "unreachable_by_published_label_pipeline": len(unreachable_records),
            "unreachable_reason_counts": dict(reason_totals),
            "unreachable_certified_at_frozen_gates": len(certified),
            "unreachable_missing_frozen_gates": len(missed),
            "label_brackets_without_an_event_sign_change": sum(
                len(e["label_brackets_without_an_event_sign_change"]) for e in slices
            ),
            "rows_failing_closure": sum(len(e["rows_failing_closure"]) for e in slices),
            "rows_disagreeing_with_published_label": sum(
                len(e["rows_disagreeing_with_published_label"]) for e in slices
            ),
            "wall_seconds": time.perf_counter() - started,
        },
        "slices": slices,
        "cross_check_against_sign_topology_audit": checks,
        "scope": "A comparison of two bracketing criteria on published baseline rows. "
        "It does not regenerate the 620-cell census, does not modify "
        "V1_CRITICAL_GRAPH.json, and never sets release_ready.  Where the "
        "event-sign criterion finds a label-invisible bracket that the localizer "
        "certifies, the label criterion provably could not have produced that "
        "root; where it finds none, nothing is certified about completeness.",
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": args.output,
                "totals": {k: v for k, v in payload["totals"].items() if k != "wall_seconds"},
                "cross_check_matched": sum(1 for c in checks if c["matched"]),
                "cross_check_total": len(checks),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
