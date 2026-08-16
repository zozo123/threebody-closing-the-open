#!/usr/bin/env python3
"""Audit whether the recorded critical-root certificates are reproducible in float64.

Motivation
----------
The census records, per root, a closure norm and a Floquet event value, gated at
1e-7 and 2e-8.  Both numbers are produced by the same float64 integration that
produced the orbit.  This script asks the independent question: **take the
recorded chart and masses exactly as written, integrate them at a converged
tolerance, and see whether the recorded numbers come back.**

Why they might not.  With ``alpha = tr M`` and ``beta = (alpha^2 - tr M^2)/2``,
the event functions are

    G+ = beta - 6 alpha + 20,  G- = beta - 2 alpha + 4,
    Delta = (alpha-4)^2 - 4(beta - 4 alpha + 8)

and ``beta`` is extracted by cancellation from ``tr(M^2)``, whose float64
round-off is of order ``eps * ||M||^2``.  For these orbits ``||M||`` runs from
about 1e3 to 2.4e4, so that floor runs from 2e-10 to 1.3e-7 -- and ``Delta``
multiplies it by a further 4.  Wherever the floor exceeds the 2e-8 event gate,
the gate is being applied to a number float64 cannot compute that precisely, and
"|event| = 1.99e-8 against a gate of 2e-8" is not a 0.5% margin.

This script does not modify any evidence and refuses to write into
``research/evidence``.  It uses ``critical_manifold._flow_for_vector`` -- the
census's own code path -- so a disagreement cannot be blamed on a second
implementation.

Usage
-----
    PYTHONPATH=src python scripts/audit_event_conditioning.py \
        research/evidence/V1_HYBRID_CRITICAL_ROOTS_2026-08-15.json \
        --output /tmp/event_conditioning.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scipy.integrate import solve_ivp  # noqa: E402

from threebody_atlas.critical_manifold import _flow_for_vector, event_value  # noqa: E402
from threebody_atlas.liao_family import state_from_chart  # noqa: E402
from threebody_atlas.reduced import (  # noqa: E402
    full_to_reduced,
    reduced_jacobian,
    reduced_rhs,
    stability_invariants,
)

EPS = float(np.finfo(float).eps)

#: `first_step` values used by the step-sequence jitter probe.  A converged
#: quantity cannot depend on these; a round-off-dominated one does.
JITTER_FIRST_STEPS = (None, 1e-4, 3e-4, 1e-3, 3e-3)


def roundoff_floor(monodromy_norm: float, mode: str) -> float:
    """Float64 round-off floor of the event value.

    ``beta = (alpha^2 - tr M^2)/2`` is a cancellation against ``tr(M^2)``, whose
    entries are of order ``||M||^2``, so ``beta`` carries an absolute error of
    order ``eps * ||M||^2``.  ``Delta = (alpha-4)^2 - 4(beta - 4 alpha + 8)``
    multiplies that by 4; ``G+`` and ``G-`` inherit it unscaled.
    """
    floor = EPS * monodromy_norm * monodromy_norm
    return 4.0 * floor if mode == "trace_collision" else floor


def _event_with_first_step(root, first_step, *, rtol, atol) -> float:
    """Integrate at a FIXED tolerance, varying only the integrator's opening step.

    `first_step` cannot change a quantity that the tolerance has actually
    resolved -- the adaptive controller converges to its own step sequence
    either way, and truncation error is bounded by `rtol` regardless.  Any
    spread it produces is round-off, full stop.  This is the sharpest available
    evidence because it removes every confound: same machine, same build, same
    method, same tolerance, same chart.
    """
    masses = tuple(float(v) for v in root["masses"])
    m = np.asarray(masses, dtype=float)
    period = float(root["period"])
    z0 = full_to_reduced(
        state_from_chart(masses, float(root["x1"]), float(root["v1"]), float(root["v2"]))
    )
    y0 = np.concatenate((z0, np.eye(8).ravel()))

    def augmented(t, y):
        return np.concatenate(
            (reduced_rhs(t, y[:8], m), (reduced_jacobian(y[:8], m) @ y[8:].reshape(8, 8)).ravel())
        )

    kwargs = {"method": "DOP853", "rtol": rtol, "atol": atol}
    if first_step is not None:
        kwargs["first_step"] = first_step
    sol = solve_ivp(augmented, (0.0, period), y0, **kwargs)
    if not sol.success:
        raise RuntimeError(sol.message)
    mono = sol.y[8:, -1].reshape(8, 8)
    return float(event_value(stability_invariants(mono), root["event_mode"]))


def audit_root(root, *, rtol, atol, coarse_rtol, coarse_atol, with_coarse=True, jitter=False):
    m1, m2, m3 = (float(v) for v in root["masses"])
    y = np.array(
        [
            float(root["x1"]), float(root["v1"]), float(root["v2"]),
            float(root["period"]), m1, m2,
        ]
    )
    mode = root["event_mode"]
    closure, floquet = _flow_for_vector(y, m3=m3, rtol=rtol, atol=atol)
    norm_m = float(np.linalg.norm(floquet.monodromy))
    ev = float(event_value(floquet, mode))
    if with_coarse:
        _, floquet_c = _flow_for_vector(y, m3=m3, rtol=coarse_rtol, atol=coarse_atol)
        ev_c = float(event_value(floquet_c, mode))
    else:
        ev_c = float("nan")
    jitter_spread = float("nan")
    if jitter:
        vals = [
            _event_with_first_step(root, fs, rtol=rtol, atol=atol) for fs in JITTER_FIRST_STEPS
        ]
        jitter_spread = float(max(vals) - min(vals))
    floor = roundoff_floor(norm_m, mode)
    return {
        "cell_id": root["cell_id"],
        "estimator": root["estimator"],
        "event_mode": mode,
        "monodromy_norm": norm_m,
        "roundoff_floor_estimate": floor,
        "recorded_event": float(root["event"]),
        "recomputed_event": ev,
        "recomputed_event_coarse": ev_c,
        "event_discrepancy": abs(ev - float(root["event"])),
        "tolerance_spread": abs(ev - ev_c),
        "first_step_jitter_spread": jitter_spread,
        "recorded_closure": float(root["closure"]),
        "recomputed_closure": float(np.linalg.norm(closure)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("census", type=Path)
    ap.add_argument("--rtol", type=float, default=1e-13)
    ap.add_argument("--atol", type=float, default=1e-15)
    ap.add_argument("--coarse-rtol", type=float, default=5e-13)
    ap.add_argument("--coarse-atol", type=float, default=5e-15)
    ap.add_argument("--estimator", default="float64")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sample", type=int, default=0,
                    help="audit a seeded random subset of this size.  Prefer this over --stride: "
                         "cell ids advance in steps of ~4 per m1 slice, so a small stride aliases "
                         "onto one track and can miss an entire event mode.")
    ap.add_argument("--seed", type=int, default=20260816)
    ap.add_argument("--stride", type=int, default=1,
                    help="audit every k-th root.  Beware aliasing; see --sample.")
    ap.add_argument("--no-coarse", action="store_true",
                    help="skip the second, coarser integration (halves the cost)")
    ap.add_argument("--jitter", action="store_true",
                    help="also vary the integrator's first_step at FIXED tolerance; "
                         "the resulting spread is pure round-off (5x the cost)")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    if args.output is not None and "research/evidence" in args.output.as_posix():
        raise SystemExit("refusing to write into research/evidence: this is an audit, not a producer")

    census = json.loads(args.census.read_text())
    gates = census["frozen_gates"]
    roots = [r for r in census["roots"] if r["status"] == "ok"]
    if args.estimator != "all":
        roots = [r for r in roots if r["estimator"] == args.estimator]
    if args.stride > 1:
        roots = roots[:: args.stride]
    if args.sample and args.sample < len(roots):
        rng = np.random.default_rng(args.seed)
        idx = np.sort(rng.choice(len(roots), size=args.sample, replace=False))
        roots = [roots[i] for i in idx]
    if args.limit:
        roots = roots[: args.limit]

    records = []
    for i, r in enumerate(roots, 1):
        rec = audit_root(
            r, rtol=args.rtol, atol=args.atol,
            coarse_rtol=args.coarse_rtol, coarse_atol=args.coarse_atol,
            with_coarse=not args.no_coarse, jitter=args.jitter,
        )
        records.append(rec)
        if i % 25 == 0:
            print(f"  ... {i}/{len(roots)}", flush=True)

    over_event = [r for r in records if r["event_discrepancy"] > gates["event"]]
    floor_over_gate = [r for r in records if r["roundoff_floor_estimate"] > gates["event"]]
    closure_over_gate = [r for r in records if r["recomputed_closure"] > gates["closure"]]
    closure_optimistic = [
        r for r in records if r["recomputed_closure"] > 10.0 * max(r["recorded_closure"], 1e-300)
    ]
    jitter_over_gate = [
        r for r in records
        if r["first_step_jitter_spread"] == r["first_step_jitter_spread"]
        and r["first_step_jitter_spread"] > gates["event"]
    ]

    summary = {
        "census": args.census.as_posix(),
        "estimator_filter": args.estimator,
        "stride": args.stride,
        "sample": args.sample,
        "seed": args.seed,
        "event_mode_counts": {
            m: sum(1 for r in roots if r["event_mode"] == m)
            for m in ("plus_one", "minus_one", "trace_collision")
        },
        "frozen_gates": gates,
        "roots_audited": len(records),
        "recorded_event_not_reproducible_beyond_gate": len(over_event),
        "roundoff_floor_exceeds_event_gate": len(floor_over_gate),
        "recomputed_closure_exceeds_closure_gate": len(closure_over_gate),
        "recorded_closure_optimistic_by_more_than_10x": len(closure_optimistic),
        # None, not 0, when --jitter was not requested: "0 findings" and "not
        # measured" must never look the same in a summary.
        "first_step_jitter_spread_exceeds_event_gate": len(jitter_over_gate) if args.jitter else None,
        "max_event_discrepancy": max((r["event_discrepancy"] for r in records), default=0.0),
        "max_monodromy_norm": max((r["monodromy_norm"] for r in records), default=0.0),
        "max_recomputed_closure": max((r["recomputed_closure"] for r in records), default=0.0),
    }
    print(json.dumps(summary, indent=2))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"summary": summary, "roots": records}, indent=2) + "\n")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
