#!/usr/bin/env python3
"""Turn sign-topology audit output into independent BigFloat localization seeds.

WHY THIS EXISTS.  Three attempts to reach the tips float64 cannot certify all used
CONTINUATION -- carry a chart forward one step at a time -- and all three failed in
different ways: reusing the tip's chart went stale ("shooting produced non-positive
period"), not extrapolating m2 left the true root outside a 2e-5 bracket, and a
fixed m1 step is the wrong parameterization for a curve of slope 4.187.  Each step's
error compounds into the next, so the whole approach is fragile by construction.

scripts/audit_sign_topology.py already does something better, and independently at
every point: it solves for a converged periodic orbit (closure ~1e-10) and detects
where the mechanism's invariant changes sign.  Its output therefore carries, per
located curve crossing, exactly what a BigFloat localization needs:

    a fixed m1, an m2 bracket around the sign change, and a converged chart
    (x1, v1, v2, period) measured AT that mass point rather than extrapolated to it.

So localize each point independently.  No chart is carried, no step size has to be
tuned, nothing compounds, and every seed is embarrassingly parallel -- which is what
lets a fleet actually help.  This is the same shape as the run that certified the
three tips in the first place (CI 32037241295): seed BigFloat from a good chart at
the point of interest, and let it bracket and refine.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

HEADER = ("name", "event_mode", "m1", "m2", "m3", "x1", "v1", "v2", "period", "screening_event")
COMPONENT_MODE = {"G_plus": "plus_one", "G_minus": "minus_one", "discriminant": "trace_collision"}


def nearest_probe(probes: list[dict], m1: float, m2: float) -> dict | None:
    """The converged probe closest to a target point, for its chart."""
    best, best_d = None, math.inf
    for probe in probes:
        if not probe.get("ok") or not probe.get("chart"):
            continue
        try:
            d = math.dist([float(probe["m1"]), float(probe["m2"])], [m1, m2])
        except (TypeError, ValueError, KeyError):
            continue
        if d < best_d:
            best, best_d = probe, d
    return best


def seeds_from_audit(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    probes = [p for p in (payload.get("probes") or []) if isinstance(p, dict)]
    out = []
    for violation in payload.get("violations") or []:
        if violation.get("kind") != "missing_critical_curve":
            continue
        bracket = violation.get("m2_bracket") or []
        if len(bracket) != 2:
            continue
        m1 = float(violation["m1"])
        # midpoint of the sign-change bracket is the best available estimate of
        # where the invariant actually vanishes on this line
        m2 = (float(bracket[0]) + float(bracket[1])) / 2.0
        probe = nearest_probe(probes, m1, m2)
        if probe is None:
            continue
        chart = [float(v) for v in probe["chart"]]
        mode = COMPONENT_MODE.get(str(violation.get("component")), violation.get("mechanism"))
        if not mode:
            continue
        out.append(
            {
                "name": f"seed_{mode}_m1_{m1:.6f}".replace(".", "p"),
                "event_mode": mode,
                "m1": m1,
                "m2": m2,
                "m3": 1.0,
                "x1": chart[0], "v1": chart[1], "v2": chart[2], "period": chart[3],
                # the screening value is provenance, not a gate input
                "screening_event": float(probe.get("G_minus") if mode == "minus_one"
                                         else probe.get("G_plus") or 0.0),
                "_bracket": [float(bracket[0]), float(bracket[1])],
                "_chart_from": [float(probe["m1"]), float(probe["m2"])],
                "_chart_closure": float(probe.get("closure") or 0.0),
                "_source": str(path),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audits", nargs="+", help="sign-topology audit artifact(s)")
    parser.add_argument("--output-dir", default="experiments/audit_seeds")
    parser.add_argument("--emit-matrix", help="append matrix=<json> for GITHUB_OUTPUT")
    parser.add_argument("--manifest", help="write a provenance manifest here")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds: list[dict] = []
    for item in args.audits:
        path = Path(item)
        if not path.is_file():
            print(f"skip (unreadable): {path}")
            continue
        found = seeds_from_audit(path)
        print(f"{path.name}: {len(found)} located curve crossing(s)")
        seeds.extend(found)

    names = []
    for seed in seeds:
        line = "\t".join(
            seed[k] if isinstance(seed[k], str) else repr(seed[k]) for k in HEADER
        )
        (out_dir / f"{seed['name']}.tsv").write_text("\t".join(HEADER) + "\n" + line + "\n")
        names.append(seed["name"])
        print(f"  {seed['name']:34} m1={seed['m1']:.6f} m2={seed['m2']:.7f} "
              f"bracket_width={seed['_bracket'][1]-seed['_bracket'][0]:.3e} "
              f"chart_from_dm={math.dist([seed['m1'], seed['m2']], seed['_chart_from']):.3e} "
              f"chart_closure={seed['_chart_closure']:.2e}")
    if args.manifest:
        Path(args.manifest).write_text(json.dumps({"seeds": seeds}, indent=2, sort_keys=True) + "\n")
    if args.emit_matrix:
        with open(args.emit_matrix, "a", encoding="utf-8") as handle:
            handle.write(f"matrix={json.dumps(names)}\n")
    print(f"{len(names)} seed(s)")


if __name__ == "__main__":
    main()
