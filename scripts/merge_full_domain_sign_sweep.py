#!/usr/bin/env python3
"""Merge sign-sweep shards by RE-DERIVING the census, not by trusting shards.

WHY THIS EXISTS
---------------
``scripts/sweep_full_domain_sign_changes.py`` runs one m1 window per process and
writes its own findings alongside its probes.  A merge step that concatenated
those findings would inherit every shard's arithmetic without checking it, and a
sharded run would then be *less* trustworthy than an unsharded one -- the
opposite of the point.

So this merger throws the shards' conclusions away and recomputes them from the
raw probe records:

  * G_plus / G_minus / discriminant / n_unstable are recomputed from each probe's
    (alpha, beta) and compared with what the shard stored;
  * every vertical and horizontal sign change is re-enumerated from the merged
    lattice, so a bracket a shard invented (or dropped) shows up as a
    disagreement rather than as a result;
  * every localization's pass/fail is re-decided from its event and closure
    against the frozen gates, so a shard cannot mark a gate-missing root
    ``passed``;
  * ``census_would_bracket`` is recomputed from the frozen published raster the
    merger loads itself, so the headline "how many curves the published S/U
    census cannot see" does not depend on a shard's bookkeeping;
  * localized points are linked across adjacent scan lines into curve
    components, because the scientific object is a curve and a per-line root is
    only a point on one.

Shards that overlap are welcome: a lattice point probed twice is reported with
its spread rather than silently averaged.  float64 results in this repository are
not reproducible across machines -- measured spreads of a factor of 3.4 on the
same organizer chart -- so the merger records agreement of *signs and gate
verdicts*, and never asserts agreement of magnitudes.

This script writes an evidence artifact.  It never sets ``release_ready`` and
never touches ``research/evidence/V1_CRITICAL_GRAPH.json``.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import platform
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

SCHEMA = "atlas.v1.full-domain-sign-sweep-census/1"

REPO_ROOT = Path(__file__).resolve().parents[1]
SWEEP_PATH = REPO_ROOT / "scripts/sweep_full_domain_sign_changes.py"


def _load_sweep_module() -> Any:
    spec = importlib.util.spec_from_file_location("_atlas_sign_sweep", SWEEP_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - packaging error
        raise RuntimeError(f"cannot load {SWEEP_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SWEEP = _load_sweep_module()
AST = SWEEP.AST
COMPONENTS: tuple[str, ...] = SWEEP.COMPONENTS
MAX_EVENT = SWEEP.MAX_EVENT
MAX_CLOSURE = SWEEP.MAX_CLOSURE


# --------------------------------------------------------------------------
# re-derivation of the per-probe state
# --------------------------------------------------------------------------
def rederive_probe(probe: dict[str, Any]) -> dict[str, Any]:
    """Recompute the three events and n_unstable from (alpha, beta).

    Returns a copy with re-derived values installed and the shard's own values
    kept under ``shard_reported`` so a disagreement is visible in the artifact.
    """
    out = dict(probe)
    if not probe.get("ok"):
        return out
    alpha, beta = float(probe["alpha"]), float(probe["beta"])
    derived = AST.state_from_invariants(alpha, beta)
    n = AST.unstable_count(alpha, beta)
    reported = {c: probe.get(c) for c in COMPONENTS}
    reported["n_unstable"] = probe.get("n_unstable")
    mismatch: list[str] = []
    for component in COMPONENTS:
        old = reported[component]
        new = derived[component]
        if old is None or not math.isfinite(float(old)):
            mismatch.append(component)
        elif (float(old) > 0.0) != (new > 0.0):
            mismatch.append(component)
    if reported["n_unstable"] != n:
        mismatch.append("n_unstable")
    out.update(derived)
    out["n_unstable"] = n
    if mismatch:
        out["rederivation_mismatch"] = mismatch
        out["shard_reported"] = reported
    return out


def merge_probes(shards: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """One record per lattice point; duplicates reported with their spread."""
    buckets: dict[tuple[float, float], list[dict[str, Any]]] = defaultdict(list)
    for shard in shards:
        label = shard.get("shard", {}).get("label", "?")
        for probe in shard.get("probes", ()):
            record = rederive_probe(probe)
            record["shard_label"] = label
            buckets[(float(probe["m1"]), float(probe["m2"]))].append(record)
    merged: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    for key in sorted(buckets):
        group = buckets[key]
        chosen = next((p for p in group if p.get("ok")), group[0])
        if len(group) > 1:
            ok_group = [p for p in group if p.get("ok")]
            entry: dict[str, Any] = {
                "m1": key[0],
                "m2": key[1],
                "shards": [p["shard_label"] for p in group],
                "converged": [bool(p.get("ok")) for p in group],
            }
            if len(ok_group) > 1:
                # Signs and n_unstable must agree; magnitudes need not, and are
                # reported as a spread rather than asserted equal.
                for component in COMPONENTS:
                    values = [float(p[component]) for p in ok_group]
                    entry[f"{component}_signs_agree"] = len({v > 0.0 for v in values}) == 1
                    entry[f"{component}_spread"] = max(values) - min(values)
                entry["n_unstable_agree"] = len({p.get("n_unstable") for p in ok_group}) == 1
            duplicates.append(entry)
        merged.append(chosen)
    return merged, duplicates


# --------------------------------------------------------------------------
# re-derivation of the findings
# --------------------------------------------------------------------------
def rederive_localizations(
    shards: Sequence[dict[str, Any]],
    brackets: Sequence[dict[str, Any]],
    *,
    labelled_rows: dict[float, list[tuple[float, str]]],
    edges: Sequence[Any],
    match_tolerance: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Re-decide gates, census blindness and committed-edge match for each root."""
    bracket_keys = {SWEEP._bracket_key(b) for b in brackets}
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    status_overrides = 0
    orphans: list[dict[str, Any]] = []
    verdict_overrides = 0
    for shard in shards:
        label = shard.get("shard", {}).get("label", "?")
        for item in shard.get("localizations", ()):
            key = SWEEP._bracket_key(item)
            if key in seen:
                continue
            seen.add(key)
            record = dict(item)
            record["shard_label"] = label
            record["bracket_rederived"] = key in bracket_keys
            if not record["bracket_rederived"]:
                orphans.append({"key": key, "shard": label, "mechanism": item.get("mechanism")})
            shard_status = item.get("status")
            if "event_value" in item and "closure" in item:
                passed = SWEEP.gate_verdict(float(item["event_value"]), float(item["closure"]))
                record["status"] = "passed" if passed else "missed_frozen_gates"
                if record["status"] != shard_status:
                    status_overrides += 1
                    record["shard_reported_status"] = shard_status
            if "masses" in item:
                m1 = float(item["masses"][0])
                m2_star = float(item["masses"][1])
                rows = labelled_rows.get(_nearest_key(labelled_rows, m1), [])
                verdict = SWEEP.published_cell_verdict(rows, m2_star)
                if item.get("published_cell", {}).get("census_would_bracket") != verdict.get(
                    "census_would_bracket"
                ):
                    verdict_overrides += 1
                    record["shard_reported_published_cell"] = item.get("published_cell")
                record["published_cell"] = verdict
                record["committed_edge"] = SWEEP.committed_edge_match(
                    edges,
                    str(item["mechanism"]),
                    m1,
                    m2_star,
                    tolerance=match_tolerance,
                )
            out.append(record)
    out.sort(key=lambda item: (item.get("mechanism", ""), item.get("m1", 0.0)))
    audit = {
        "localizations_rederived": len(out),
        "gate_status_overridden_by_merge": status_overrides,
        "census_blindness_verdict_overridden_by_merge": verdict_overrides,
        "localizations_without_a_rederived_bracket": orphans,
    }
    return out, audit


def _nearest_key(mapping: dict[float, Any], value: float) -> float:
    return min(mapping, key=lambda k: abs(k - value))


def link_components(
    localizations: Sequence[dict[str, Any]],
    scan_lines: Sequence[float],
    *,
    link_threshold: float,
) -> list[dict[str, Any]]:
    """Chain gate-passing roots of one mechanism across adjacent scan lines.

    A per-line root is a point; the scientific object is a curve.  Two roots of
    the same mechanism on adjacent *probed* lines are taken to lie on one curve
    when their m2 differ by at most ``link_threshold`` -- the same continuity
    rule ``scripts/audit_transition_topology.py`` uses on published brackets.
    Greedy nearest-neighbour chaining, so two curves that approach within the
    threshold are reported as one component and the ambiguity is recorded.
    """
    order = {m1: i for i, m1 in enumerate(sorted(scan_lines))}
    by_mechanism: dict[str, dict[float, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for item in localizations:
        if item.get("status") != "passed" or "masses" not in item:
            continue
        by_mechanism[str(item["mechanism"])][float(item["masses"][0])].append(item)

    components: list[dict[str, Any]] = []
    for mechanism, lines in sorted(by_mechanism.items()):
        open_chains: list[list[dict[str, Any]]] = []
        closed: list[list[dict[str, Any]]] = []
        previous_index: int | None = None
        for m1 in sorted(lines):
            index = order.get(m1)
            points = sorted(lines[m1], key=lambda it: float(it["masses"][1]))
            adjacent = previous_index is not None and index is not None and index - previous_index == 1
            if not adjacent:
                closed.extend(open_chains)
                open_chains = [[p] for p in points]
            else:
                available = list(open_chains)
                open_chains = []
                for point in points:
                    m2 = float(point["masses"][1])
                    best = None
                    best_distance = math.inf
                    for chain in available:
                        distance = abs(float(chain[-1]["masses"][1]) - m2)
                        if distance < best_distance:
                            best, best_distance = chain, distance
                    if best is not None and best_distance <= link_threshold:
                        available.remove(best)
                        best.append(point)
                        open_chains.append(best)
                    else:
                        open_chains.append([point])
                closed.extend(available)
            previous_index = index
        closed.extend(open_chains)
        for chain in closed:
            m1s = [float(p["masses"][0]) for p in chain]
            m2s = [float(p["masses"][1]) for p in chain]
            blind = [p for p in chain if p.get("published_cell", {}).get("census_would_bracket") is False]
            on_graph = [p for p in chain if p.get("committed_edge", {}).get("matched")]
            components.append(
                {
                    "mechanism": mechanism,
                    "points": len(chain),
                    "m1_range": [min(m1s), max(m1s)],
                    "m2_range": [min(m2s), max(m2s)],
                    "in_committed_graph": len(on_graph) == len(chain),
                    "partly_in_committed_graph": 0 < len(on_graph) < len(chain),
                    "committed_edge_ids": sorted({
                        str(p["committed_edge"].get("edge_id"))
                        for p in on_graph
                        if p.get("committed_edge", {}).get("edge_id")
                    }),
                    "points_invisible_to_published_labels": len(blind),
                    "invisible_to_published_labels": len(blind) == len(chain),
                    "vertices": [
                        {
                            "m1": float(p["masses"][0]),
                            "m2": float(p["masses"][1]),
                            "event_value": p.get("event_value"),
                            "closure": p.get("closure"),
                            "census_would_bracket": p.get("published_cell", {}).get(
                                "census_would_bracket"
                            ),
                            "committed_edge_matched": p.get("committed_edge", {}).get("matched"),
                        }
                        for p in sorted(chain, key=lambda it: float(it["masses"][0]))
                    ],
                }
            )
    components.sort(key=lambda c: (c["mechanism"], c["m1_range"][0], c["m2_range"][0]))
    return components


def mechanism_crossings(
    localizations: Sequence[dict[str, Any]], *, threshold: float
) -> list[dict[str, Any]]:
    """Scan lines where two different mechanisms localize to nearly the same m2.

    Where a plus_one and a minus_one curve meet, P(+2) = P(-2) = 0 together.
    Subtracting the two gives alpha = 4 and then beta = 4, so a = alpha - 4 = 0
    and b = beta - 4 alpha + 8 = -4, whence P(t) = t^2 - 4 and the reduced
    multipliers are exactly {+1, +1, -1, -1}.  That is a codimension-two
    organizer, and the local structure there is an X rather than an endpoint --
    which is why an endpoint classifier meeting one has nothing to say.

    This reports *proximity on a probed scan line*, not a localized organizer.
    Two curves passing within ``threshold`` is a candidate to be continued, not
    a certified intersection: a real crossing needs a two-parameter solve of
    G_plus = G_minus = closure = 0, which this sweep does not do.
    """
    by_line: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for item in localizations:
        if item.get("status") == "passed" and "masses" in item:
            by_line[float(item["masses"][0])].append(item)
    out: list[dict[str, Any]] = []
    for m1 in sorted(by_line):
        points = by_line[m1]
        for i, left in enumerate(points):
            for right in points[i + 1 :]:
                if left["mechanism"] == right["mechanism"]:
                    continue
                separation = abs(float(left["masses"][1]) - float(right["masses"][1]))
                if separation > threshold:
                    continue
                out.append(
                    {
                        "m1": m1,
                        "mechanisms": sorted([str(left["mechanism"]), str(right["mechanism"])]),
                        "m2": sorted([float(left["masses"][1]), float(right["masses"][1])]),
                        "separation_m2": separation,
                        "threshold_m2": threshold,
                        "both_invisible_to_published_labels": (
                            left.get("published_cell", {}).get("census_would_bracket") is False
                            and right.get("published_cell", {}).get("census_would_bracket") is False
                        ),
                        "both_absent_from_committed_graph": (
                            not left.get("committed_edge", {}).get("matched")
                            and not right.get("committed_edge", {}).get("matched")
                        ),
                        "interpretation": (
                            "candidate codimension-two organizer: where these two curves "
                            "meet, alpha = beta = 4 and the reduced multipliers are "
                            "{+1, +1, -1, -1}.  Proximity on one scan line, not a "
                            "localized intersection."
                        ),
                    }
                )
    return out


def coverage_report(
    probes: Sequence[dict[str, Any]],
    m2_by_m1: dict[float, Sequence[float]],
    domain: dict[str, Any],
) -> dict[str, Any]:
    """State the covered fraction of the declared domain without inflating it."""
    m1_lo, m1_hi = float(domain["m1"][0]), float(domain["m1"][1])
    m2_lo, m2_hi = float(domain["m2"][0]), float(domain["m2"][1])
    published_lines = [m1 for m1 in sorted(m2_by_m1) if m1_lo - 1e-12 <= m1 <= m1_hi + 1e-12]
    published_rows = sum(
        1
        for m1 in published_lines
        for m2 in m2_by_m1[m1]
        if m2_lo - 1e-12 <= m2 <= m2_hi + 1e-12
    )
    by_line: dict[float, list[float]] = defaultdict(list)
    for probe in probes:
        by_line[float(probe["m1"])].append(float(probe["m2"]))
    probed_lines = sorted(by_line)
    spans = []
    for m1 in probed_lines:
        m2s = sorted(by_line[m1])
        published = [m2 for m2 in m2_by_m1.get(m1, ()) if m2_lo - 1e-12 <= m2 <= m2_hi + 1e-12]
        spans.append(
            {
                "m1": m1,
                "probes": len(m2s),
                "probed_m2_range": [m2s[0], m2s[-1]],
                "published_m2_support": [published[0], published[-1]] if published else None,
                "max_probe_gap_m2": max(
                    (b - a for a, b in zip(m2s, m2s[1:], strict=False)), default=0.0
                ),
            }
        )
    return {
        "declared_domain": {"m1": [m1_lo, m1_hi], "m2": [m2_lo, m2_hi], "m3": domain.get("m3")},
        "published_m1_slices_in_domain": len(published_lines),
        "probed_m1_scan_lines": len(probed_lines),
        "m1_scan_line_fraction_of_published_slices": (
            len(probed_lines) / len(published_lines) if published_lines else None
        ),
        "published_raster_rows_in_domain": published_rows,
        "probes": len(probes),
        "probe_fraction_of_published_raster_rows": (
            len(probes) / published_rows if published_rows else None
        ),
        "per_line": spans,
        "honest_scope": (
            "The swept set is a finite union of 1-D scan lines and therefore has "
            "measure zero in the declared rectangle.  'Every sign change' means every "
            "sign change ON THIS LATTICE.  A critical curve confined between two "
            "adjacent scan lines is not sampled, and a lattice cell crossed by an even "
            "number of curves of one mechanism shows no sign change at all.  The "
            "published m2 support is narrower than the declared m2 interval at the "
            "extremes of m1, so no probe exists where the frozen raster lists no orbit."
        ),
    }


# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("shards", nargs="+", help="shard JSON files (or .partial checkpoints)")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--graph", default="research/evidence/V1_CRITICAL_GRAPH.json")
    parser.add_argument("--roots", default="research/evidence/V1_HYBRID_CRITICAL_ROOTS_2026-08-15.json")
    parser.add_argument("--sign-floor", type=float, default=1e-4)
    parser.add_argument("--match-tolerance", type=float, default=1.5e-3)
    parser.add_argument("--link-threshold", type=float, default=0.02)
    parser.add_argument(
        "--crossing-threshold",
        type=float,
        default=2e-3,
        help="m2 separation below which two mechanisms on one scan line are flagged",
    )
    args = parser.parse_args()

    shards: list[dict[str, Any]] = []
    shard_index: list[dict[str, Any]] = []
    for name in args.shards:
        path = Path(name)
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("schema") != SWEEP.SHARD_SCHEMA:
            raise SystemExit(f"{path}: unexpected schema {document.get('schema')!r}")
        shards.append(document)
        shard_index.append(
            {
                "path": str(path),
                "label": document.get("shard", {}).get("label"),
                "phase": document.get("phase"),
                "m1_range": document.get("shard", {}).get("m1_range"),
                "m1_stride": document.get("shard", {}).get("m1_stride"),
                "m2_stride": document.get("shard", {}).get("m2_stride"),
                "planned_scan_lines": len(document.get("planned_scan_lines", ())),
                "completed_scan_lines": len(document.get("completed_scan_lines", ())),
                "probes": len(document.get("probes", ())),
                "localizations": len(document.get("localizations", ())),
                "python": document.get("python"),
                "code_revision": document.get("code_revision"),
                "baseline_digests": document.get("inputs", {}).get("baseline_digests"),
                "cpu_seconds": document.get("probe_summary", {}).get("cpu_seconds"),
            }
        )

    digests = {
        json.dumps(entry["baseline_digests"], sort_keys=True) for entry in shard_index
    }
    if len(digests) > 1:
        raise SystemExit("shards used different baseline rasters; refusing to merge")

    graph = json.loads(Path(args.graph).read_text(encoding="utf-8"))
    roots_doc = json.loads(Path(args.roots).read_text(encoding="utf-8"))
    # Root sources come from the canonical assembly invocation, never from
    # --roots alone: the graph spans the 620-cell census AND the sweep-derived
    # supplemental roots (cell ids >= 10000), and supplemental ids are per-run
    # sequential indices, so reading the wrong artifact silently repoints every
    # one of them.  See scripts/graph_root_sources.py.
    import runpy as _runpy

    _resolver = _runpy.run_path(
        str(Path(__file__).resolve().parent / "graph_root_sources.py")
    )
    edges = AST.edges_from_graph(graph, _resolver["load_roots"]())
    grid, baseline_digests = SWEEP.load_baseline(Path(args.baseline))
    if len(digests) == 1 and next(iter(digests)) != "null":
        recorded = json.loads(next(iter(digests)))
        if recorded.get("sha256") != baseline_digests["sha256"]:
            raise SystemExit(
                "the raster this merge loaded is not the raster the shards probed: "
                f"{baseline_digests['sha256']} vs {recorded.get('sha256')}"
            )
    m2_by_m1 = {m1: sorted(rows) for m1, rows in grid.items()}
    labelled_rows = {
        m1: [(m2, grid[m1][m2].published_stability) for m2 in m2_by_m1[m1]] for m1 in m2_by_m1
    }

    probes, duplicates = merge_probes(shards)
    scan_lines = sorted({float(p["m1"]) for p in probes})
    vertical = SWEEP._all_vertical(probes, sign_floor=args.sign_floor)
    horizontal = SWEEP.horizontal_sign_changes(probes, sign_floor=args.sign_floor)
    localizations, audit = rederive_localizations(
        shards,
        vertical,
        labelled_rows=labelled_rows,
        edges=edges,
        match_tolerance=args.match_tolerance,
    )
    localized_keys = {SWEEP._bracket_key(item) for item in localizations}
    uncertified = [b for b in vertical if SWEEP._bracket_key(b) not in localized_keys]
    components = link_components(localizations, scan_lines, link_threshold=args.link_threshold)
    crossings = mechanism_crossings(localizations, threshold=args.crossing_threshold)

    passed = [item for item in localizations if item.get("status") == "passed"]
    blind = [item for item in passed if item.get("published_cell", {}).get("census_would_bracket") is False]
    uncommitted = [item for item in passed if not item.get("committed_edge", {}).get("matched")]
    blind_and_uncommitted = [
        item
        for item in passed
        if item.get("published_cell", {}).get("census_would_bracket") is False
        and not item.get("committed_edge", {}).get("matched")
    ]

    headline = {
        "vertical_sign_changes_on_the_lattice": len(vertical),
        "horizontal_sign_changes_between_scan_lines": len(horizontal),
        "brackets_certified": len(localizations),
        "brackets_left_uncertified": len(uncertified),
        "roots_passing_frozen_gates": len(passed),
        "roots_missing_frozen_gates": sum(
            1 for item in localizations if item.get("status") == "missed_frozen_gates"
        ),
        "localizer_failures": sum(
            1 for item in localizations if str(item.get("status", "")).startswith("localizer_failed")
        ),
        "roots_absent_from_the_committed_graph": len(uncommitted),
        "roots_invisible_to_the_published_S_U_labels": len(blind),
        "roots_both_absent_and_invisible": len(blind_and_uncommitted),
        "curve_components": len(components),
        "curve_components_absent_from_the_committed_graph": sum(
            1 for c in components if not c["in_committed_graph"]
        ),
        "curve_components_entirely_invisible_to_published_labels": sum(
            1 for c in components if c["invisible_to_published_labels"]
        ),
        "candidate_codimension_two_organizers": len(crossings),
        "by_mechanism": {
            mechanism: {
                "roots_passing_gates": sum(1 for item in passed if item.get("mechanism") == mechanism),
                "roots_invisible_to_published_labels": sum(
                    1 for item in blind if item.get("mechanism") == mechanism
                ),
                "components": sum(1 for c in components if c["mechanism"] == mechanism),
            }
            for mechanism in sorted(SWEEP.MECHANISM_COMPONENT)
        },
    }

    payload = {
        "schema": SCHEMA,
        "claim_status": "screening census; falsifies completeness where it fires, certifies nothing where it does not",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_revision": os.getenv("GITHUB_SHA"),
        "run_id": os.getenv("GITHUB_RUN_ID"),
        "python": platform.python_version(),
        "merge_policy": (
            "Shard conclusions are discarded and recomputed from the raw probe records: "
            "the three event scalars and n_unstable from (alpha, beta); every vertical and "
            "horizontal sign change from the merged lattice; every pass/fail from the event "
            "and closure against the frozen gates; every census-blindness verdict from the "
            "frozen raster this merge loaded itself."
        ),
        "inputs": {
            "baseline": str(args.baseline),
            "baseline_digests": baseline_digests,
            "graph": args.graph,
            "graph_schema": graph.get("schema"),
            "graph_release_ready": graph.get("release_ready"),
            "roots": args.roots,
            "roots_schema": roots_doc.get("schema"),
            "shards": shard_index,
        },
        "gates": {
            "maximum_absolute_event": MAX_EVENT,
            "maximum_periodic_closure": MAX_CLOSURE,
            "note": "frozen; re-applied by this merge from the recorded numbers",
        },
        "reproducibility_note": (
            "float64 results here are not reproducible across machines; a re-run may move "
            "an event magnitude by a factor of several.  This artifact therefore records "
            "agreement of signs, n_unstable and gate verdicts, and reports magnitudes "
            "only as measured values, never as reproducible constants."
        ),
        "headline": headline,
        "coverage": coverage_report(probes, m2_by_m1, graph["declared_mass_domain"]),
        "rederivation_audit": {
            **audit,
            "probes_merged": len(probes),
            "probes_with_rederivation_mismatch": [
                {"m1": p["m1"], "m2": p["m2"], "mismatch": p["rederivation_mismatch"]}
                for p in probes
                if p.get("rederivation_mismatch")
            ],
            "duplicate_lattice_points": duplicates,
            "duplicate_lattice_points_with_disagreeing_signs": [
                d
                for d in duplicates
                if any(d.get(f"{c}_signs_agree") is False for c in COMPONENTS)
                or d.get("n_unstable_agree") is False
            ],
        },
        "curve_components": components,
        "candidate_codimension_two_organizers": crossings,
        "localizations": localizations,
        "uncertified_vertical_sign_changes": uncertified,
        "vertical_sign_changes": vertical,
        "horizontal_sign_changes": horizontal,
        "probes": probes,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "headline": headline, "coverage": {
        k: v for k, v in payload["coverage"].items() if k != "per_line"
    }}, indent=2))


if __name__ == "__main__":
    main()
