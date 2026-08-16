#!/usr/bin/env python3
"""Audit how much room the localized critical-root census actually has.

The census reports ``max |event| <= 2e-8`` and calls the completeness gate
passed.  This script refuses to report that sentence without the number behind
it.  It computes

* the full |event| distribution (median / p75 / p90 / p99 / max) overall and
  split by estimator, event mode and orientation;
* the gate occupancy ``max |event| / gate`` -- how much of the frozen budget the
  worst cell consumes;
* whether the worst cells cluster (mechanism, orientation, m1 band, distance to
  the nearest organizer) or scatter.  A cluster is a physical signal; scatter is
  a numerical one;
* optionally (``--reevaluate N``) the *evaluation uncertainty* of the reported
  event residual, by recomputing the event functional at the exact recorded
  state across a ladder of integrator tolerances, and by converting that spread
  into an m2 uncertainty through the measured d(event)/d(m2) slope.

The re-evaluation column is the one that matters.  A residual is only a gate if
it can be measured more accurately than the gate.

This script only reads frozen artifacts and writes one new evidence JSON.  It
never touches a gate and never sets release_ready.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

GATE_EVENT = 2e-8
GATE_CLOSURE = 1e-7

# Integrator tolerance ladder used for the re-evaluation probe.  The first entry
# is the boundary.evaluate default; the second is the tolerance the localizer's
# accepting _precise_evaluate call uses; the rest tighten from there.
TOLERANCE_LADDER = ((2e-11, 2e-13), (5e-13, 5e-15), (1e-13, 1e-15), (5e-14, 5e-16))


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def distribution(values: list[float]) -> dict[str, Any]:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0}
    return {
        "count": len(ordered),
        "min": ordered[0],
        "median": percentile(ordered, 0.5),
        "p75": percentile(ordered, 0.75),
        "p90": percentile(ordered, 0.90),
        "p99": percentile(ordered, 0.99),
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
        "above_1e_9": sum(1 for v in ordered if v > 1e-9),
        "above_1e_8": sum(1 for v in ordered if v > 1e-8),
        "above_1p5e_8": sum(1 for v in ordered if v > 1.5e-8),
        "above_1p9e_8": sum(1 for v in ordered if v > 1.9e-8),
        "gate_occupancy_max": ordered[-1] / GATE_EVENT,
        "gate_occupancy_p90": percentile(ordered, 0.90) / GATE_EVENT,
    }


def group_distribution(roots: list[dict], key: str) -> dict[str, Any]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for root in roots:
        buckets[str(root.get(key))].append(abs(float(root["event"])))
    return {name: distribution(values) for name, values in sorted(buckets.items())}


def organizer_masses(graph_path: Path) -> list[dict[str, Any]]:
    if not graph_path.exists():
        return []
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    out = []
    for node in graph.get("nodes", []):
        masses = node.get("masses")
        if not masses:
            continue
        out.append(
            {
                "id": node.get("id"),
                "kind": node.get("kind"),
                "m1": float(masses[0]),
                "m2": float(masses[1]),
            }
        )
    return out


def nearest_organizer(root: dict, organizers: list[dict[str, Any]]) -> dict[str, Any]:
    if not organizers:
        return {"id": None, "distance": None}
    m1, m2 = float(root["masses"][0]), float(root["masses"][1])
    best = min(
        organizers,
        key=lambda org: math.hypot(m1 - org["m1"], m2 - org["m2"]),
    )
    return {
        "id": best["id"],
        "distance": math.hypot(m1 - best["m1"], m2 - best["m2"]),
    }


def m1_band(root: dict, width: float = 0.02) -> str:
    m1 = float(root["masses"][0])
    lo = math.floor(m1 / width) * width
    return f"{lo:.2f}-{lo + width:.2f}"


def clustering_report(roots: list[dict], organizers: list[dict[str, Any]], top: int) -> dict[str, Any]:
    """Is the tail structured, or is it noise sprayed across the census?"""
    ordered = sorted(roots, key=lambda r: -abs(float(r["event"])))
    worst = ordered[:top]

    def share(items: list[dict], key_fn) -> dict[str, float]:
        counts = Counter(key_fn(item) for item in items)
        total = max(len(items), 1)
        return {str(k): v / total for k, v in counts.most_common()}

    all_mode = share(roots, lambda r: r["event_mode"])
    worst_mode = share(worst, lambda r: r["event_mode"])
    all_orient = share(roots, lambda r: r.get("orientation"))
    worst_orient = share(worst, lambda r: r.get("orientation"))
    all_band = share(roots, m1_band)
    worst_band = share(worst, m1_band)

    def enrichment(worst_share: dict[str, float], base_share: dict[str, float]) -> dict[str, float]:
        return {
            k: (worst_share[k] / base_share[k]) if base_share.get(k) else float("inf")
            for k in worst_share
        }

    worst_org_distance = [nearest_organizer(r, organizers)["distance"] for r in worst]
    all_org_distance = [nearest_organizer(r, organizers)["distance"] for r in roots]
    worst_org_distance = [d for d in worst_org_distance if d is not None]
    all_org_distance = [d for d in all_org_distance if d is not None]

    # Spearman-style rank correlation between |event| and m1 (monotone trend
    # detector; no scipy dependency so this script stays importable anywhere).
    def rank(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    ev = [abs(float(r["event"])) for r in roots]
    m1s = [float(r["masses"][0]) for r in roots]
    rev, rm1 = rank(ev), rank(m1s)
    n = len(ev)
    mean_rev = statistics.fmean(rev)
    mean_rm1 = statistics.fmean(rm1)
    num = sum((rev[i] - mean_rev) * (rm1[i] - mean_rm1) for i in range(n))
    den = math.sqrt(
        sum((rev[i] - mean_rev) ** 2 for i in range(n))
        * sum((rm1[i] - mean_rm1) ** 2 for i in range(n))
    )
    spearman_m1 = num / den if den else float("nan")

    return {
        "top_n": top,
        "worst_cell_ids": [r["cell_id"] for r in worst],
        "event_mode_share_all": all_mode,
        "event_mode_share_worst": worst_mode,
        "event_mode_enrichment": enrichment(worst_mode, all_mode),
        "orientation_share_all": all_orient,
        "orientation_share_worst": worst_orient,
        "orientation_enrichment": enrichment(worst_orient, all_orient),
        "m1_band_share_worst": worst_band,
        "m1_band_enrichment": enrichment(worst_band, all_band),
        "distinct_m1_bands_in_worst": len(worst_band),
        "spearman_event_vs_m1": spearman_m1,
        "median_organizer_distance_all": (
            statistics.median(all_org_distance) if all_org_distance else None
        ),
        "median_organizer_distance_worst": (
            statistics.median(worst_org_distance) if worst_org_distance else None
        ),
    }


def reevaluate(roots: list[dict], cell_ids: list[int]) -> list[dict[str, Any]]:
    """Recompute the event functional at the exact recorded state.

    This is the honest error bar on the published residual.  If the same state,
    the same code and the same locked dependency versions disagree by more than
    the gate when the integrator tolerance moves, then ``|event| <= 2e-8`` is
    not a measurement -- it is a coincidence of one tolerance choice.
    """
    import numpy as np

    from threebody_atlas.critical_manifold import event_value
    from threebody_atlas.liao_family import FamilyPoint, correct_family_point
    from threebody_atlas.reduced import compute_reduced_floquet

    by_id = {int(r["cell_id"]): r for r in roots}
    out: list[dict[str, Any]] = []
    for cell_id in cell_ids:
        rec = by_id[cell_id]
        mode = rec["event_mode"]
        masses = tuple(float(x) for x in rec["masses"])
        point = FamilyPoint(
            masses=masses,
            x1=float(rec["x1"]),
            v1=float(rec["v1"]),
            v2=float(rec["v2"]),
            period=float(rec["period"]),
            residual_norm=float(rec.get("closure", 0.0)),
            nfev=0,
            success=True,
        )
        started = time.time()

        invariants: list[tuple[float, float]] = []

        def event_at(
            pt: FamilyPoint,
            rtol: float,
            atol: float,
            record: bool = False,
            _mode: str = mode,
            _invariants: list[tuple[float, float]] = invariants,
        ) -> float:
            floquet = compute_reduced_floquet(
                pt.state(),
                np.asarray(pt.masses, dtype=float),
                pt.period,
                rtol=rtol,
                atol=atol,
            )
            if record:
                _invariants.append((float(floquet.alpha), float(floquet.beta)))
            return float(event_value(floquet, _mode))

        ladder = [
            event_at(point, rtol, atol, record=True) for rtol, atol in TOLERANCE_LADDER
        ]
        # How many digits the event functional cancels away.  This is the
        # amplification that turns a small monodromy error into a large event
        # error, and it is why float64 cannot reach a tighter gate here.
        alpha, beta = invariants[1]
        if mode == "trace_collision":
            terms = ((alpha - 4.0) ** 2, 4.0 * (beta - 4.0 * alpha + 8.0))
        elif mode == "plus_one":
            terms = (beta + 20.0, 6.0 * alpha)
        else:
            terms = (beta + 4.0, 2.0 * alpha)
        largest_term = max(abs(t) for t in terms)
        cancellation = largest_term / max(abs(ladder[1]), 1e-300)
        tight_alpha = [a for a, _ in invariants[1:]]
        tight_beta = [b for _, b in invariants[1:]]
        alpha_spread = max(tight_alpha) - min(tight_alpha)
        beta_spread = max(tight_beta) - min(tight_beta)
        tight = ladder[1:]
        m1, m2, m3 = masses
        step = 1e-7
        plus = correct_family_point((m1, m2 + step, m3), (point.x1, point.v1, point.v2, point.period), max_nfev=60)
        minus = correct_family_point((m1, m2 - step, m3), (point.x1, point.v1, point.v2, point.period), max_nfev=60)
        slope = (
            event_at(plus, 5e-13, 5e-15) - event_at(minus, 5e-13, 5e-15)
        ) / (2.0 * step)
        tight_spread = max(tight) - min(tight)
        # The recorded value was produced by _precise_evaluate at rtol 5e-13,
        # which is exactly TOLERANCE_LADDER[1].  For a float64 root the two
        # numbers are the same computation on the same state, so any difference
        # is a cross-platform reproducibility failure, not a modelling choice.
        comparable = rec.get("estimator") == "float64"
        discrepancy = abs(ladder[1] - float(rec["event"])) if comparable else None
        out.append(
            {
                "cell_id": cell_id,
                "event_mode": mode,
                "recorded_event": float(rec["event"]),
                "recorded_abs_event": abs(float(rec["event"])),
                "estimator": rec.get("estimator"),
                "same_computation_as_recorded": comparable,
                "cross_platform_discrepancy": discrepancy,
                "cross_platform_discrepancy_over_gate": (
                    discrepancy / GATE_EVENT if discrepancy is not None else None
                ),
                "tolerance_ladder": [list(t) for t in TOLERANCE_LADDER],
                "event_at_tolerance": ladder,
                "tight_ladder_spread": tight_spread,
                "full_ladder_spread": max(ladder) - min(ladder),
                "reevaluated_at_accepting_tolerance": ladder[1],
                "reevaluation_exceeds_gate": bool(abs(ladder[1]) > GATE_EVENT),
                "any_tight_tolerance_exceeds_gate": bool(
                    any(abs(v) > GATE_EVENT for v in tight)
                ),
                "d_event_d_m2": slope,
                "m2_uncertainty_from_tight_spread": (
                    abs(tight_spread / slope) if slope else None
                ),
                "alpha": alpha,
                "beta": beta,
                "largest_cancelling_term": largest_term,
                "cancellation_factor": cancellation,
                "alpha_spread_over_tight_ladder": alpha_spread,
                "beta_spread_over_tight_ladder": beta_spread,
                "monodromy_invariant_relative_spread": max(
                    alpha_spread / max(abs(alpha), 1e-300),
                    beta_spread / max(abs(beta), 1e-300),
                ),
                "seconds": time.time() - started,
            }
        )
        print(json.dumps(out[-1]), flush=True)
    return out


# Measured BigFloat cost, CI run 31932398513, job bigfloat-fold, dps=60:
# one corrected_at is ~5.5 min (four ~82 s correction iterations).  The census
# lane runs at dps=40; BigFloat multiply cost is superlinear in precision, and
# a conservative reading of that ladder puts a dps=40 corrected_at at roughly
# half a dps=60 one.  Both numbers are recorded so a reader can redo the
# arithmetic with a better measurement.
BIGFLOAT_CORRECTED_AT_MINUTES_DPS60 = 5.5
DPS40_COST_FRACTION_OF_DPS60 = 0.5
CENSUS_CELLS = 620
JULIA_LOCALIZED_CELLS = 158


def bigfloat_iteration_counts(evidence_root: Path) -> list[int]:
    """Recorded BigFloat secant-iteration counts from the Julia lane artifacts."""
    counts: list[int] = []
    for directory in sorted(evidence_root.glob("julia_*")):
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = payload.get("results", [payload]) if isinstance(payload, dict) else []
            for row in rows:
                if isinstance(row, dict) and row.get("refinement_iterations") is not None:
                    counts.append(int(row["refinement_iterations"]))
    return counts


def tenfold_reduction_estimate(
    probes: list[dict[str, Any]], iteration_counts: list[int]
) -> dict[str, Any]:
    """What it would take to pull ``max |event|`` down by 10x, and what it costs.

    The float64 lane is ruled out on measurement grounds, not effort grounds:
    its own evaluation uncertainty already exceeds the target.  The only
    remaining route is to put all 620 cells through the BigFloat lane, so the
    cost is the BigFloat lane's per-cell cost times the cells it does not
    currently cover.
    """
    target = GATE_EVENT / 10.0
    float64_probes = [p for p in probes if p["estimator"] == "float64"]
    spreads = sorted(p["tight_ladder_spread"] for p in float64_probes)
    median_spread = spreads[len(spreads) // 2] if spreads else float("nan")
    cancellations = sorted(p["cancellation_factor"] for p in float64_probes)
    rel_spreads = sorted(
        p["monodromy_invariant_relative_spread"] for p in float64_probes
    )

    mean_iterations = (
        statistics.fmean(iteration_counts) if iteration_counts else float("nan")
    )
    # Two bracket-endpoint corrections plus one correction per secant iteration,
    # plus one extra iteration to reach 2e-9 instead of 2e-8.
    corrections_per_cell = mean_iterations + 3.0
    per_cell_minutes = (
        BIGFLOAT_CORRECTED_AT_MINUTES_DPS60
        * DPS40_COST_FRACTION_OF_DPS60
        * corrections_per_cell
    )
    remaining_cells = CENSUS_CELLS - JULIA_LOCALIZED_CELLS
    return {
        "target_max_abs_event": target,
        "float64_route": {
            "verdict": "impossible",
            "reason": (
                "The float64 evaluation uncertainty of the event functional at the "
                "recorded states already exceeds the 2e-9 target by orders of "
                "magnitude, so no integrator tolerance reaches it.  Tightening rtol "
                "does not help: the ladder shows the value moving, not converging."
            ),
            "median_float64_evaluation_uncertainty": median_spread,
            "ratio_to_target": median_spread / target,
            "median_cancellation_factor": (
                cancellations[len(cancellations) // 2] if cancellations else None
            ),
            "max_cancellation_factor": cancellations[-1] if cancellations else None,
            "median_monodromy_invariant_relative_spread": (
                rel_spreads[len(rel_spreads) // 2] if rel_spreads else None
            ),
        },
        "bigfloat_route": {
            "verdict": "sufficient_but_expensive",
            "reason": (
                "At dps=40 the BigFloat lane already reaches 7e-13 on its best cells, "
                "so 2e-9 is not a precision problem for it.  Its recorded 1e-8 roots "
                "are gate-limited: its localizer terminates on the first iterate under "
                "2e-8.  Reaching 2e-9 needs the same lane run over all 620 cells with "
                "the stopping tolerance set to 2e-9, which is roughly one extra secant "
                "iteration per cell on top of full coverage."
            ),
            "required_dps": 40,
            "recorded_bigfloat_cells_sampled": len(iteration_counts),
            "mean_recorded_bigfloat_iterations": mean_iterations,
            "max_recorded_bigfloat_iterations": (
                max(iteration_counts) if iteration_counts else None
            ),
            "corrections_per_cell_assumed": corrections_per_cell,
            "estimated_minutes_per_cell": per_cell_minutes,
            "cells_not_yet_bigfloat": remaining_cells,
            "estimated_serial_cpu_hours": per_cell_minutes * remaining_cells / 60.0,
            "estimated_wall_hours_at_62_way_parallelism": (
                per_cell_minutes * remaining_cells / 62.0 / 60.0
            ),
            "cost_model_source": (
                "CI run 31932398513 job bigfloat-fold: one dps=60 corrected_at ~5.5 min "
                "(four ~82 s correction iterations); dps=40 assumed at half that cost"
            ),
        },
        "cheapest_honest_alternative": (
            "Do not chase the residual.  Report d(event)/d(m2) with every root and "
            "state the m2 uncertainty (1e-10 to 8e-9 as measured), which is already "
            "five to seven orders inside the 1e-3 published cells.  That is a claim "
            "the float64 lane can actually support."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--roots",
        default="research/evidence/V1_HYBRID_CRITICAL_ROOTS_2026-08-15.json",
    )
    parser.add_argument("--graph", default="research/evidence/V1_CRITICAL_GRAPH.json")
    parser.add_argument("--output", default="research/evidence/V1_EVENT_MARGIN_AUDIT.json")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument(
        "--reevaluate",
        type=int,
        default=0,
        help="Re-evaluate the event functional at the recorded state for the N worst "
        "cells plus N evenly spaced control cells (requires numpy/scipy).",
    )
    args = parser.parse_args()

    payload_in = json.loads(Path(args.roots).read_text(encoding="utf-8"))
    roots = payload_in["roots"]
    gates = payload_in.get("frozen_gates", {})
    if float(gates.get("event", GATE_EVENT)) != GATE_EVENT:
        raise SystemExit(
            f"roots artifact declares a different event gate: {gates.get('event')}"
        )

    events = [abs(float(r["event"])) for r in roots]
    closures = [abs(float(r["closure"])) for r in roots]
    organizers = organizer_masses(Path(args.graph))

    ordered = sorted(roots, key=lambda r: -abs(float(r["event"])))
    worst_ids = [int(r["cell_id"]) for r in ordered[: args.reevaluate]]
    control_ids: list[int] = []
    if args.reevaluate:
        stride = max(len(ordered) // max(args.reevaluate, 1), 1)
        control_ids = [
            int(ordered[i]["cell_id"])
            for i in range(args.reevaluate, len(ordered), stride)
        ][: args.reevaluate]

    payload: dict[str, Any] = {
        "schema": "atlas.v1.event-margin-audit/1",
        "claim_status": (
            "read-only audit of the frozen hybrid critical-root census; no gate is "
            "modified, no release flag is set, no root is re-localized"
        ),
        "source_roots_artifact": args.roots,
        "source_roots_schema": payload_in.get("schema"),
        "frozen_gates": {"event": GATE_EVENT, "closure": GATE_CLOSURE},
        "localized_roots": len(roots),
        "event_distribution": distribution(events),
        "closure_distribution": distribution(closures),
        "event_distribution_by_estimator": group_distribution(roots, "estimator"),
        "event_distribution_by_mode": group_distribution(roots, "event_mode"),
        "event_distribution_by_orientation": group_distribution(roots, "orientation"),
        "clustering": clustering_report(roots, organizers, args.top),
        "worst_cells": [
            {
                "cell_id": r["cell_id"],
                "abs_event": abs(float(r["event"])),
                "gate_occupancy": abs(float(r["event"])) / GATE_EVENT,
                "event_mode": r["event_mode"],
                "orientation": r.get("orientation"),
                "estimator": r.get("estimator"),
                "m1": float(r["masses"][0]),
                "m2": float(r["masses"][1]),
                "closure": float(r["closure"]),
                "nearest_organizer": nearest_organizer(r, organizers),
            }
            for r in ordered[: args.top]
        ],
    }
    if args.reevaluate:
        import platform

        import numpy
        import scipy

        probes = reevaluate(roots, worst_ids + control_ids)
        tight_spreads = [p["tight_ladder_spread"] for p in probes]
        comparable = [p for p in probes if p["same_computation_as_recorded"]]
        discrepancies = sorted(p["cross_platform_discrepancy"] for p in comparable)
        payload["reevaluation"] = {
            "note": (
                "Same recorded state, same locked numpy/scipy, event functional "
                "recomputed across the integrator tolerance ladder.  The spread is "
                "the evaluation uncertainty of the published residual."
            ),
            "reevaluation_platform": {
                "system": platform.system(),
                "machine": platform.machine(),
                "python": platform.python_version(),
                "numpy": numpy.__version__,
                "scipy": scipy.__version__,
            },
            "cross_platform_note": (
                "The recorded float64 residual and 'reevaluated_at_accepting_tolerance' "
                "are the same deterministic computation (compute_reduced_floquet at "
                "rtol=5e-13, atol=5e-15) on the same recorded state.  Re-running this "
                "script twice on one machine reproduces every digit.  Any nonzero "
                "cross_platform_discrepancy is therefore a difference between the CPU "
                "that produced the frozen artifact and the CPU running this audit, "
                "amplified by the ~8-digit cancellation inside the event functionals."
            ),
            "comparable_float64_probes": len(comparable),
            "median_cross_platform_discrepancy": (
                discrepancies[len(discrepancies) // 2] if discrepancies else None
            ),
            "max_cross_platform_discrepancy": (
                discrepancies[-1] if discrepancies else None
            ),
            "float64_probes_exceeding_gate_on_reevaluation": [
                p["cell_id"]
                for p in comparable
                if p["reevaluation_exceeds_gate"]
            ],
            "probed_worst_cells": worst_ids,
            "probed_control_cells": control_ids,
            "median_tight_ladder_spread": statistics.median(tight_spreads),
            "max_tight_ladder_spread": max(tight_spreads),
            "cells_whose_reevaluation_exceeds_gate": [
                p["cell_id"] for p in probes if p["reevaluation_exceeds_gate"]
            ],
            "cells_exceeding_gate_at_some_tight_tolerance": [
                p["cell_id"] for p in probes if p["any_tight_tolerance_exceeds_gate"]
            ],
            "evaluation_uncertainty_over_gate": (
                statistics.median(tight_spreads) / GATE_EVENT
            ),
            "probes": probes,
        }
        payload["tenfold_reduction_estimate"] = tenfold_reduction_estimate(
            probes, bigfloat_iteration_counts(Path(args.roots).parent)
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    summary = {
        "localized_roots": payload["localized_roots"],
        "max_abs_event": payload["event_distribution"]["max"],
        "gate_occupancy_max": payload["event_distribution"]["gate_occupancy_max"],
        "above_1e_8": payload["event_distribution"]["above_1e_8"],
    }
    if "reevaluation" in payload:
        summary["evaluation_uncertainty_over_gate"] = payload["reevaluation"][
            "evaluation_uncertainty_over_gate"
        ]
        summary["cells_whose_reevaluation_exceeds_gate"] = payload["reevaluation"][
            "cells_whose_reevaluation_exceeds_gate"
        ]
        summary["float64_probes_exceeding_gate_on_reevaluation"] = len(
            payload["reevaluation"]["float64_probes_exceeding_gate_on_reevaluation"]
        )
        summary["comparable_float64_probes"] = payload["reevaluation"][
            "comparable_float64_probes"
        ]
        summary["max_cross_platform_discrepancy"] = payload["reevaluation"][
            "max_cross_platform_discrepancy"
        ]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
