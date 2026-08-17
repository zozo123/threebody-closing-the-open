#!/usr/bin/env python3
"""Seed a BigFloat ladder beyond the two tips float64 cannot re-certify.

release_ready rests on exactly two unclassified lattice ends:

    minus_one_sweep_component_0 start   cell 10000 (0.892, 0.7530796)
    minus_one_sweep_component_1 end     cell 10042 (1.042, 0.8579752)

Both are BigFloat-verified critical roots that float64 cannot re-certify, and
float64 continuation past them returns event=inf.  Rather than continue, localize
the curve directly in BigFloat at m1 values beyond each tip.

The seed m2 at each rung is a LINEAR EXTRAPOLATION from the two nearest certified
roots on that arm.  That is a guess, and it is meant to be: julia/verify_critical_points.jl
brackets around the seed, refines the event edge, and REFUSES a root that moved
more than its max-shift gate.  A bad extrapolation therefore fails loudly instead
of producing a plausible wrong root.  Each rung is emitted as its own seed file so
one rung's failure does not take the arm with it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOTS = Path("research/evidence/V1_SUPPLEMENTAL_EVENT_SIGN_ROOTS_BIGFLOAT_TIPS_2026-08-17.json")
DOMAIN_M1 = (0.8, 1.1)
HEADER = ("name", "event_mode", "m1", "m2", "m3", "x1", "v1", "v2", "period", "screening_event")
ARMS = ((0, "low"), (1, "high"))


def arm_rungs(rows: list[dict], component: int, side: str, rungs: int, step: float):
    pts = sorted((r["masses"][0], r) for r in rows if r.get("sweep_component") == component)
    if len(pts) < 2:
        raise SystemExit(f"component {component} has fewer than two roots")
    tip, inward = (pts[0][1], pts[1][1]) if side == "low" else (pts[-1][1], pts[-2][1])
    m1_t, m2_t = tip["masses"][0], tip["masses"][1]
    m1_i, m2_i = inward["masses"][0], inward["masses"][1]
    slope = (m2_t - m2_i) / (m1_t - m1_i)
    out = []
    for k in range(1, rungs + 1):
        delta = (-step if side == "low" else step) * k
        m1 = m1_t + delta
        if not (DOMAIN_M1[0] <= m1 <= DOMAIN_M1[1]):
            break
        out.append(
            {
                "name": f"comp{component}_{side}_k{k:02d}",
                "event_mode": tip["event_mode"],
                "m1": m1,
                "m2": m2_t + slope * delta,
                "m3": tip["masses"][2],
                "x1": tip["x1"], "v1": tip["v1"], "v2": tip["v2"],
                "period": tip["period"], "screening_event": tip["event"],
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rungs", type=int, default=8)
    parser.add_argument("--step", type=float, default=0.001)
    parser.add_argument("--output-dir", default="experiments/tip_ladder")
    parser.add_argument("--emit-matrix", help="append matrix=<json> for GITHUB_OUTPUT")
    args = parser.parse_args()

    rows = json.loads(ROOTS.read_text())["roots"]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = []
    for component, side in ARMS:
        for seed in arm_rungs(rows, component, side, args.rungs, args.step):
            line = "\t".join(
                seed[k] if isinstance(seed[k], str) else repr(seed[k]) for k in HEADER
            )
            (out_dir / f"{seed['name']}.tsv").write_text("\t".join(HEADER) + "\n" + line + "\n")
            names.append(seed["name"])
            print(f"{seed['name']:22} m1={seed['m1']:.4f} m2_seed={seed['m2']:.7f}")
    if args.emit_matrix:
        with open(args.emit_matrix, "a", encoding="utf-8") as handle:
            handle.write(f"matrix={json.dumps(names)}\n")
    print(f"{len(names)} rungs")


if __name__ == "__main__":
    main()
