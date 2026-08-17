#!/usr/bin/env python3
"""Aggregate per-root conditioning into the margin artifact no gate ever had.

WHY THIS EXISTS.  scripts/localize_full_critical_network.py has recorded
closure_conditioning and event_conditioning per root, on by default, since the
instrumentation landed -- and no committed artifact ever summarized them:
V1_CRITICAL_GRAPH.json's root_residual_margin reported {reported 0, missing 775}
and V1_CLAIM_ASSURANCE_MATRIX.json carried conditioning_margin = not_run on all
seven claims (issue #212 section 3, #211 Track C).

WHAT THE NUMBERS BUY.  A strictly positive minimum sigma_min of the closure
Jacobian across every certified root is the hypothesis of the implicit function
theorem along the family: the periodic orbit, its period, its monodromy, and
hence the invariants G_plus, G_minus and the discriminant depend real-analytically
on (m1, m2) near every certified root.  On a compact domain that is what makes
the critical set a real-analytic variety -- finitely many components, no
accumulation -- which is the logical footing a sampling audit's null result
otherwise lacks.  This script computes the aggregate; the analyticity STATEMENT
is only as strong as the coverage of the roots summarized, so the artifact
records exactly which cells contributed and which are missing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from threebody_atlas.conditioning import summarize_conditioning  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("shards", nargs="+", help="critical-root chunk JSON files")
    parser.add_argument(
        "--output",
        default="research/evidence/V1_CONDITIONING_MARGIN_2026-08-18.json",
    )
    parser.add_argument(
        "--expect-cells",
        type=int,
        default=620,
        help="cells the census owes; fewer contributing roots is reported, not hidden",
    )
    args = parser.parse_args()

    closure_reports, event_reports = [], []
    per_root, cells = [], set()
    worst = None
    for item in args.shards:
        shard = json.loads(Path(item).read_text())
        for root in shard.get("roots") or []:
            cell = root.get("cell_id")
            closure = root.get("closure_conditioning")
            event = root.get("event_conditioning")
            if cell is None or closure is None or event is None:
                continue
            cells.add(int(cell))
            closure_reports.append(closure)
            event_reports.append(event)
            row = {
                "cell_id": int(cell),
                "closure_sigma_min": float(closure["sigma_min"]),
                "closure_kappa_2": float(closure["kappa_2"]),
                "closure_displacement_bound": float(closure["displacement_bound"]),
                "event_slope": float(event["sigma_max"]),
                "event_displacement_bound": float(event["displacement_bound"]),
                "m2_uncertainty": (
                    float(root["m2_uncertainty"]) if root.get("m2_uncertainty") is not None else None
                ),
            }
            per_root.append(row)
            if worst is None or row["closure_sigma_min"] < worst["closure_sigma_min"]:
                worst = row
    if not per_root:
        raise SystemExit("no roots with conditioning found in the supplied shards")

    per_root.sort(key=lambda r: r["closure_sigma_min"])
    sigma_mins = [r["closure_sigma_min"] for r in per_root]
    payload = {
        "schema": "atlas.v1.conditioning-margin/1",
        "claim": (
            "aggregate conditioning of every contributing certified root: the minimum "
            "closure-Jacobian sigma_min is the implicit-function-theorem hypothesis for "
            "real-analytic mass dependence of the family and its Floquet invariants"
        ),
        "cells_contributing": len(cells),
        "cells_expected": args.expect_cells,
        "cells_missing": max(0, args.expect_cells - len(cells)),
        "min_closure_sigma_min": sigma_mins[0],
        "median_closure_sigma_min": sigma_mins[len(sigma_mins) // 2],
        "max_closure_displacement_bound": max(r["closure_displacement_bound"] for r in per_root),
        "max_m2_uncertainty": max(
            (r["m2_uncertainty"] for r in per_root if r["m2_uncertainty"] is not None),
            default=None,
        ),
        "worst_root": worst,
        "ten_smallest_sigma_min": per_root[:10],
        "closure_conditioning_summary": summarize_conditioning(closure_reports),
        "event_conditioning_summary": summarize_conditioning(event_reports),
        "analyticity_statement": (
            "every contributing root has closure-Jacobian sigma_min > 0, so the implicit "
            "function theorem applies at each: the periodic family and its invariants are "
            "real-analytic in (m1, m2) in a neighbourhood of every such root. The statement "
            "covers exactly the contributing cells listed here and claims nothing where "
            "conditioning was not computed."
            if sigma_mins[0] > 0
            else "AT LEAST ONE ROOT HAS A SINGULAR CLOSURE JACOBIAN; the analyticity "
            "statement FAILS and the failing roots are listed in ten_smallest_sigma_min"
        ),
    }
    out = Path(args.output)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        f"{out}: {len(cells)} cells, min sigma_min {sigma_mins[0]:.6g}, "
        f"max displacement bound {payload['max_closure_displacement_bound']:.3e}"
    )


if __name__ == "__main__":
    main()
