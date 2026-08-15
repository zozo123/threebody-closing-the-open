#!/usr/bin/env python3
"""Search the public continuation sheet for fixed-mass shooting singularities.

A disconnected/two-sheet interpretation can hide behind branch switching only
where local uniqueness is lost or Newton jumps basins.  This script attacks the
first possibility.  On a deterministic mass-grid subsample plus the entire
m2=m3=1 zero-angular-momentum spine, it integrates the analytic variational
system at the published orbit and computes the 8x4 shooting Jacobian with
respect to (x1,v1,v2,T).

We report both raw and dimensionless singular spectra.  Full column rank is a
local implicit-function diagnostic, not a rigorous global proof; the separate
forward/reverse continuation certificates attack branch switching directly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from threebody_atlas.baseline import iter_baseline
from threebody_atlas.liao_family import _flow_and_shooting_jacobian, state_from_chart
from threebody_atlas.reduced import full_to_reduced


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("output")
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--rtol", type=float, default=3e-10)
    parser.add_argument("--atol", type=float, default=3e-12)
    args = parser.parse_args()
    if args.stride < 1:
        raise SystemExit("stride must be positive")

    rows = list(iter_baseline(args.dataset))
    m1_values = sorted({round(r.m1, 12) for r in rows})
    m2_values = sorted({round(r.m2, 12) for r in rows})
    m1_index = {m: i for i, m in enumerate(m1_values)}
    m2_index = {m: i for i, m in enumerate(m2_values)}

    selected = []
    seen: set[int] = set()
    for row in rows:
        i = m1_index[round(row.m1, 12)]
        j = m2_index[round(row.m2, 12)]
        grid_sample = i % args.stride == 0 and j % args.stride == 0
        zero_l_spine = abs(row.m2 - 1.0) <= 5e-13 and abs(row.m3 - 1.0) <= 5e-13
        if grid_sample or zero_l_spine:
            if row.index not in seen:
                selected.append(row)
                seen.add(row.index)

    records = []
    worst_ratio = None
    worst_sigma = None
    worst_closure = None
    for k, row in enumerate(selected, start=1):
        masses = (row.m1, row.m2, row.m3)
        p = np.asarray([row.x1, row.v1, row.v2, row.period], dtype=float)
        closure, jac = _flow_and_shooting_jacobian(
            masses,
            p,
            rtol=args.rtol,
            atol=args.atol,
        )
        closure_norm = float(np.linalg.norm(closure))
        raw_s = np.linalg.svd(jac, compute_uv=False)

        # Dimensionless sensitivity: scale parameter increments by characteristic
        # orbit values and closure rows by characteristic reduced-state values.
        p_scale = np.maximum(np.abs(p), np.asarray([0.05, 0.5, 0.1, 1.0]))
        z0 = full_to_reduced(state_from_chart(masses, row.x1, row.v1, row.v2))
        row_scale = np.maximum(np.abs(z0), 1.0)
        scaled = (jac * p_scale[np.newaxis, :]) / row_scale[:, np.newaxis]
        s = np.linalg.svd(scaled, compute_uv=False)
        ratio = float(s[-1] / s[0]) if s[0] > 0 else 0.0
        rec = {
            "index": row.index,
            "masses": masses,
            "published_stability": row.published_stability,
            "closure_norm": closure_norm,
            "raw_singular_values": [float(x) for x in raw_s],
            "scaled_singular_values": [float(x) for x in s],
            "scaled_rank_ratio": ratio,
        }
        records.append(rec)
        if worst_ratio is None or ratio < worst_ratio[0]:
            worst_ratio = (ratio, rec)
        if worst_sigma is None or float(s[-1]) < worst_sigma[0]:
            worst_sigma = (float(s[-1]), rec)
        if worst_closure is None or closure_norm > worst_closure[0]:
            worst_closure = (closure_norm, rec)
        if k % 100 == 0:
            print(f"sampled={k}/{len(selected)} current_min_rank_ratio={worst_ratio[0]:.3e}")

    ratios = np.asarray([r["scaled_rank_ratio"] for r in records])
    sigma_min = np.asarray([r["scaled_singular_values"][-1] for r in records])
    closures = np.asarray([r["closure_norm"] for r in records])
    payload = {
        "sample_count": len(records),
        "grid_stride": args.stride,
        "includes_full_zero_angular_momentum_spine": True,
        "rank_ratio_quantiles": {
            str(q): float(np.quantile(ratios, q)) for q in (0.0, 0.001, 0.01, 0.1, 0.5, 1.0)
        },
        "scaled_sigma_min_quantiles": {
            str(q): float(np.quantile(sigma_min, q)) for q in (0.0, 0.001, 0.01, 0.1, 0.5, 1.0)
        },
        "closure_quantiles": {
            str(q): float(np.quantile(closures, q)) for q in (0.0, 0.5, 0.9, 0.99, 1.0)
        },
        "worst_rank_ratio": worst_ratio[1],
        "worst_scaled_sigma_min": worst_sigma[1],
        "worst_closure": worst_closure[1],
        "near_rank_loss_counts": {
            "ratio_lt_1e-8": int(np.sum(ratios < 1e-8)),
            "ratio_lt_1e-6": int(np.sum(ratios < 1e-6)),
            "ratio_lt_1e-4": int(np.sum(ratios < 1e-4)),
        },
        "interpretation": (
            "local implicit-function diagnostic only; absence of sampled rank loss supports "
            "but does not by itself prove global family connectivity"
        ),
        "records": records,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "sample_count": len(records),
        "min_scaled_rank_ratio": float(np.min(ratios)),
        "min_scaled_sigma_min": float(np.min(sigma_min)),
        "max_closure": float(np.max(closures)),
        "near_rank_loss_counts": payload["near_rank_loss_counts"],
    }, indent=2))


if __name__ == "__main__":
    main()
