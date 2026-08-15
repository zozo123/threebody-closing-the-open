#!/usr/bin/env python3
"""Re-audit the first-stage low shooting-rank candidates after orbit correction.

The global rank census intentionally evaluated the published rounded catalog
parameters directly.  It found no sampled scaled rank ratio below 1e-6, but 28
of 647 deterministic samples were below 1e-4 and the worst rounded orbit had a
closure of order 5e-6.  This second stage asks the only scientifically useful
question: do those low ratios survive after tight branch-preserving periodic
correction?

The candidate indices are frozen from workflow run 31874535579, artifact
9244540103 (ZIP SHA-256
``e4640c2275d86b883b67052c5ca5efb57b62b0d4e8dc0008bd48d67f53fcfe30``).
Every candidate is corrected at fixed masses before the same dimensionless
shooting singular spectrum is recomputed.

A surviving ratio below the hard suspicion threshold is not automatically a
moduli-space branch point; it is a target for path-diverse/generic continuation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from threebody_atlas.baseline import iter_baseline
from threebody_atlas.liao_family import (
    _flow_and_shooting_jacobian,
    correct_family_point,
    state_from_chart,
)
from threebody_atlas.reduced import full_to_reduced

SOURCE_RUN = 31874535579
SOURCE_ARTIFACT = 9244540103
SOURCE_ARTIFACT_SHA256 = "e4640c2275d86b883b67052c5ca5efb57b62b0d4e8dc0008bd48d67f53fcfe30"
CANDIDATE_INDICES = (
    16014,
    34410,
    7698,
    24958,
    11,
    44191,
    34430,
    16034,
    24978,
    7718,
    44211,
    94275,
    34450,
    31,
    54195,
    16054,
    44231,
    84255,
    24998,
    64215,
    34470,
    7738,
    74235,
    54215,
    94295,
    51,
    44251,
    16074,
)


def spectrum(masses, x1: float, v1: float, v2: float, period: float, *, rtol: float, atol: float):
    p = np.asarray([x1, v1, v2, period], dtype=float)
    closure, jac = _flow_and_shooting_jacobian(masses, p, rtol=rtol, atol=atol)
    p_scale = np.maximum(np.abs(p), np.asarray([0.05, 0.5, 0.1, 1.0]))
    z0 = full_to_reduced(state_from_chart(masses, x1, v1, v2))
    row_scale = np.maximum(np.abs(z0), 1.0)
    scaled = (jac * p_scale[np.newaxis, :]) / row_scale[:, np.newaxis]
    singular = np.linalg.svd(scaled, compute_uv=False)
    ratio = float(singular[-1] / singular[0]) if singular[0] > 0 else 0.0
    return float(np.linalg.norm(closure)), singular, ratio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("output")
    parser.add_argument("--correct-rtol", type=float, default=2e-12)
    parser.add_argument("--correct-atol", type=float, default=2e-14)
    parser.add_argument("--jac-rtol", type=float, default=5e-13)
    parser.add_argument("--jac-atol", type=float, default=5e-15)
    parser.add_argument("--max-corrected-closure", type=float, default=5e-8)
    parser.add_argument("--suspicion-ratio", type=float, default=1e-6)
    args = parser.parse_args()

    by_index = {row.index: row for row in iter_baseline(args.dataset)}
    missing = [index for index in CANDIDATE_INDICES if index not in by_index]
    if missing:
        raise RuntimeError(f"frozen candidate indices missing from baseline: {missing}")

    records = []
    for index in CANDIDATE_INDICES:
        row = by_index[index]
        masses = (row.m1, row.m2, row.m3)
        raw_closure, raw_singular, raw_ratio = spectrum(
            masses,
            row.x1,
            row.v1,
            row.v2,
            row.period,
            rtol=args.correct_rtol,
            atol=args.correct_atol,
        )
        corrected = correct_family_point(
            masses,
            (row.x1, row.v1, row.v2, row.period),
            max_nfev=120,
            screening_rtol=args.correct_rtol,
            screening_atol=args.correct_atol,
        )
        if not corrected.success:
            raise RuntimeError(f"periodic correction failed for catalog index {index}")
        corrected_closure, corrected_singular, corrected_ratio = spectrum(
            masses,
            corrected.x1,
            corrected.v1,
            corrected.v2,
            corrected.period,
            rtol=args.jac_rtol,
            atol=args.jac_atol,
        )
        if corrected_closure > args.max_corrected_closure:
            raise RuntimeError(
                f"corrected closure gate failed for index {index}: {corrected_closure:.3e}"
            )
        rec = {
            "index": index,
            "masses": [float(x) for x in masses],
            "published_stability": row.published_stability,
            "raw_tight_closure": raw_closure,
            "raw_tight_scaled_singular_values": [float(x) for x in raw_singular],
            "raw_tight_scaled_rank_ratio": raw_ratio,
            "corrected_chart": {
                "x1": float(corrected.x1),
                "v1": float(corrected.v1),
                "v2": float(corrected.v2),
                "period": float(corrected.period),
            },
            "corrected_closure": corrected_closure,
            "corrected_scaled_singular_values": [float(x) for x in corrected_singular],
            "corrected_scaled_rank_ratio": corrected_ratio,
            "ratio_change_factor": (
                float(corrected_ratio / raw_ratio) if raw_ratio > 0.0 else None
            ),
            "suspicious_after_correction": corrected_ratio < args.suspicion_ratio,
        }
        records.append(rec)
        print(
            f"index={index} masses={masses} raw_ratio={raw_ratio:.3e} "
            f"corrected_ratio={corrected_ratio:.3e} closure={corrected_closure:.3e}",
            flush=True,
        )

    ratios = np.asarray([r["corrected_scaled_rank_ratio"] for r in records], dtype=float)
    closures = np.asarray([r["corrected_closure"] for r in records], dtype=float)
    suspicious = [r for r in records if r["suspicious_after_correction"]]
    payload = {
        "source_first_stage": {
            "workflow_run": SOURCE_RUN,
            "artifact_id": SOURCE_ARTIFACT,
            "artifact_zip_sha256": SOURCE_ARTIFACT_SHA256,
            "selection": "all 28 first-stage samples with scaled_rank_ratio < 1e-4",
        },
        "candidate_count": len(records),
        "correction_tolerances": {
            "rtol": args.correct_rtol,
            "atol": args.correct_atol,
            "max_corrected_closure": args.max_corrected_closure,
        },
        "jacobian_tolerances": {"rtol": args.jac_rtol, "atol": args.jac_atol},
        "suspicion_ratio": args.suspicion_ratio,
        "summary": {
            "min_corrected_rank_ratio": float(np.min(ratios)),
            "median_corrected_rank_ratio": float(np.median(ratios)),
            "max_corrected_closure": float(np.max(closures)),
            "surviving_suspicious_count": len(suspicious),
            "surviving_suspicious_indices": [r["index"] for r in suspicious],
        },
        "records": records,
        "interpretation": (
            "Second-stage local implicit-function diagnostic on corrected periodic orbits. "
            "No surviving sub-threshold candidates supports local regularity on the tested "
            "locations but does not prove global continuation connectedness."
        ),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2), flush=True)

    # A surviving very-low ratio is valuable scientific evidence and should make
    # the workflow visibly fail so it cannot be silently treated as a clean rank gate.
    if suspicious:
        raise SystemExit(
            "one or more corrected shooting candidates remain below the suspicion threshold"
        )


if __name__ == "__main__":
    main()
