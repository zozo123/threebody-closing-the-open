#!/usr/bin/env python3
"""Re-derive published critical roots with the shipped dynamics.

Two modes:

    --emit BASELINE.json     write the audit of an unmutated tree
    --compare BASELINE.json  fail when this tree's audit drifts from that one

With neither, the audit is judged against the absolute bands documented in
``threebody_atlas.root_audit`` (frozen closure gate 1e-7, plus the measured
float64-vs-BigFloat agreement bands for alpha/beta/discriminant).

Exit status: 0 clean, 1 the audit found a problem.  ``scripts/mutation_harness.py``
uses the ``--compare`` form as its physics detector, which is why the drift
tolerance is a knob rather than a constant.

This writes nothing into research/evidence/ -- it is a checker, not a producer.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from threebody_atlas.root_audit import audit, audit_cells, baseline_cell_ids, compare, to_json
except ModuleNotFoundError:  # running from a source checkout without an install
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from threebody_atlas.root_audit import audit, audit_cells, baseline_cell_ids, compare, to_json


DEFAULT_ROOTS = "research/evidence/V1_HYBRID_CRITICAL_ROOTS_2026-08-15.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", default=DEFAULT_ROOTS)
    parser.add_argument("--count", type=int, default=8, help="roots sampled across the census")
    parser.add_argument("--emit", help="write the audit here and exit 0")
    parser.add_argument("--compare", help="compare against a previously emitted audit")
    parser.add_argument("--tolerance", type=float, default=1e-9)
    args = parser.parse_args()

    baseline: dict | None = None
    if args.compare:
        baseline = json.loads(Path(args.compare).read_text(encoding="utf-8"))
        # Audit exactly the cells the baseline audited.  Index sampling would
        # shift if a root were added or removed, turning an artifact-level fault
        # into a spurious physics failure.
        audits = audit_cells(Path(args.roots), baseline_cell_ids(baseline))
    else:
        audits = audit(Path(args.roots), count=args.count)
    payload = to_json(audits)

    if args.emit:
        out = Path(args.emit)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote baseline audit of {len(audits)} roots to {out}")
        return 0

    if baseline is not None:
        problems = compare(audits, baseline, tolerance=args.tolerance)
    else:
        problems = [problem for item in audits for problem in item.failures()]

    for item in audits:
        print(
            f"cell {item.cell_id:>3} {item.event_mode:<15} "
            f"d_alpha={item.alpha_error:.3e} d_beta={item.beta_error:.3e} "
            f"d_disc={item.discriminant_error:.3e} closure={item.closure:.3e} "
            f"|event_f64|={abs(item.event):.3e}"
        )
    if problems:
        print(f"\nROOT PHYSICS AUDIT FAILED ({len(problems)} problems):")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(f"\nroot physics audit clean over {len(audits)} published roots")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
