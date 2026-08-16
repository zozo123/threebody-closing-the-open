#!/usr/bin/env python3
"""Train a surrogate, propose off-grid masses, then submit them to physics screening."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from threebody_atlas.active_learning import AtlasSurrogate
from threebody_atlas.baseline import iter_baseline
from threebody_atlas.boundary import stability_score
from threebody_atlas.evidence_semantics import artifact_semantics
from threebody_atlas.liao_family import correct_family_point
from threebody_atlas.reduced import compute_reduced_floquet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--max-training", type=int, default=5000)
    parser.add_argument("--trees", type=int, default=64)
    parser.add_argument("--candidates", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for row in iter_baseline(args.dataset):
        rows.append(row)
        if len(rows) >= args.max_training:
            break
    if len(rows) < 100:
        raise SystemExit("insufficient baseline rows")

    masses = np.asarray([[r.m1, r.m2, r.m3] for r in rows], dtype=float)
    chart = np.asarray([[r.x1, r.v1, r.v2, r.period] for r in rows], dtype=float)
    stable = np.asarray([r.published_stability == "S" for r in rows], dtype=int)
    surrogate = AtlasSurrogate(n_estimators=args.trees).fit(masses, chart, stable)

    # v1 deliberately searches a narrow off-grid neighborhood around the first
    # published m1=0.8 stable island.  Later rounds construct this pool from
    # uncertainty maps over the full mass simplex.
    m1_values = np.linspace(0.80005, 0.80295, 16)
    m2_values = np.linspace(0.75005, 0.76995, 100)
    pool = np.asarray([(m1, m2, 1.0) for m1 in m1_values for m2 in m2_values])
    proposals = surrogate.propose(pool)
    proposals.sort(key=lambda p: p.acquisition_score, reverse=True)

    accepted = []
    attempted = []
    for proposal in proposals[: max(args.candidates * 5, args.candidates)]:
        point = correct_family_point(
            (proposal.m1, proposal.m2, proposal.m3),
            (proposal.x1, proposal.v1, proposal.v2, proposal.period),
            max_nfev=25,
        )
        item = {
            "proposal": proposal.__dict__,
            "shooting_success": point.success,
            "shooting_residual": point.residual_norm,
        }
        if point.success and point.residual_norm < 1e-7:
            floquet = compute_reduced_floquet(point.state(), point.masses, point.period)
            item["corrected"] = {
                "masses": point.masses,
                "x1": point.x1,
                "v1": point.v1,
                "v2": point.v2,
                "period": point.period,
                "stability_score": stability_score(floquet),
                "screening_stable": floquet.linearly_stable,
                "alpha": floquet.alpha,
                "beta": floquet.beta,
                "discriminant": floquet.discriminant,
            }
            accepted.append(item)
        attempted.append(item)
        if len(accepted) >= args.candidates:
            break

    payload = {
        "search_semantics": artifact_semantics(
            Path(__file__).resolve().parents[1], "active_learning_pocket/v1"
        ),
        "training_rows": len(rows),
        "reported_stable_fraction_in_training": float(stable.mean()),
        "pool_size": len(pool),
        "attempted": attempted,
        "accepted_candidates": accepted,
        "claim_status": "AI proposals plus float64 screening only; not scientific discovery evidence",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "training_rows": len(rows),
        "pool_size": len(pool),
        "attempted": len(attempted),
        "accepted": len(accepted),
    }, indent=2))


if __name__ == "__main__":
    main()
