#!/usr/bin/env python3
"""Seed BigFloat certification of the two sweep-component arms.

The full-domain audit (issue #211 Track B1) located ten minus_one curve crossings
outside the committed edges, and they are not scattered: four descend below
minus_one component 0's low tip at (0.892, 0.7530796) and six continue above
component 1's high tip at (1.042, 0.8579752).  Zero narrow detections lie in the
interior 0.892 <= m1 <= 1.042.  So the graph is complete in the interior at that
resolution and the only missing curves are the continuations past the two tips
whose endpoints are unclassified -- the blockers themselves.

Each arm runs off the declared domain: component 0 crosses m2 = 0.700 between
m1 0.880 and 0.8825 (independently confirmed -- probes at m1 0.880 reach m2
0.70449 and find nothing, because the curve has already left), and component 1
reaches m1 = 1.100.  Certifying roots along each arm and ingesting them as edge
vertices extends the edges to the domain faces, where the assembler's existing
declared_domain_boundary attachment binds them.

This script converts NARROWED audit brackets into seeds for
julia/verify_critical_points.jl.  It refuses a bracket wider than the verifier's
max-shift guard, because seeding BigFloat with a midpoint that could be further
from the true root than the guard allows produces a refusal at best and a
plausible wrong root at worst.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# 12 columns: the 10-column contract plus the sign-change bracket that located
# the root.  julia/verify_critical_points.jl uses that bracket as a MEMBERSHIP
# guard -- the certified root must lie inside it -- which supersedes the fixed
# max-shift radius and is strictly stronger, because a radius admits any root
# within it while membership admits only the located one.
HEADER = ("name", "event_mode", "m1", "m2", "m3", "x1", "v1", "v2", "period",
          "screening_event", "m2_lo", "m2_hi")
# A bracket wider than this is not a located root, it is a coverage gap: the
# 2026-08-18 audit's wide brackets (3.6e-2 at m1 1.015, 5.0e-2 at 1.065,
# 1.5e-1 at 1.095) are regions where converged probes are absent, and all three
# mechanisms "flip" across them simultaneously, which no single curve does.
MAX_BRACKET = 5e-3
ARMS = {0: ("low", 0.892), 1: ("high", 1.042)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("narrowed", help="JSON list of {m1,mech,lo,hi} narrowed brackets")
    parser.add_argument("audits", nargs="+", help="audit artifacts supplying converged charts")
    parser.add_argument("--output-dir", default="experiments/arm_seeds")
    parser.add_argument("--emit-matrix")
    args = parser.parse_args()

    charts: list[dict] = []
    for item in args.audits:
        payload = json.loads(Path(item).read_text())
        for probe in payload.get("probes") or []:
            if isinstance(probe, dict) and probe.get("ok") and probe.get("chart"):
                charts.append(probe)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names, refused = [], []
    for row in json.loads(Path(args.narrowed).read_text()):
        m1, lo, hi = float(row["m1"]), float(row["lo"]), float(row["hi"])
        width = hi - lo
        if width > MAX_BRACKET:
            refused.append(
                f"m1={m1}: bracket width {width:.3e} exceeds {MAX_BRACKET} -- a coverage "
                "gap, not a located root"
            )
            continue
        m2 = (lo + hi) / 2.0
        near = min(
            (p for p in charts if abs(float(p["m1"]) - m1) < 1e-9),
            key=lambda p: abs(float(p["m2"]) - m2),
            default=None,
        )
        if near is None:
            refused.append(f"m1={m1}: no converged chart on this line")
            continue
        chart = [float(v) for v in near["chart"]]
        arm = 0 if m1 < 0.892 else 1
        name = f"arm{arm}_m1_{m1:.4f}".replace(".", "p")
        (out_dir / f"{name}.tsv").write_text(
            "\t".join(HEADER) + "\n"
            + "\t".join([
                name, str(row.get("mech", "minus_one")), repr(m1), repr(m2), repr(1.0),
                repr(chart[0]), repr(chart[1]), repr(chart[2]), repr(chart[3]),
                repr(float(near.get("G_minus") or 0.0)), repr(lo), repr(hi),
            ]) + "\n"
        )
        names.append(name)
        print(f"  {name:22} m1={m1:.4f} m2={m2:.7f} bracket={width:.2e} "
              f"chart_dm={math.dist([m1, m2], [float(near['m1']), float(near['m2'])]):.2e}")
    for line in refused:
        print(f"  REFUSED {line}")
    if args.emit_matrix:
        with open(args.emit_matrix, "a", encoding="utf-8") as handle:
            handle.write(f"matrix={json.dumps(names)}\n")
    print(f"{len(names)} seed(s), {len(refused)} refused")


if __name__ == "__main__":
    main()
