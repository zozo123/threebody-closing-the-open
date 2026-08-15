#!/usr/bin/env python3
"""Test whether apparent invariant branches are folds of one 2D orbit sheet.

The unequal-mass catalog is parameterized by two masses (m1,m2), with m3=1.
The 2023 two-family interpretation is based on the projection of those orbits to
scale-invariant period/angular-momentum coordinates. A connected 2D manifold
can look multi-valued under such a projection whenever the projection Jacobian
loses rank. This script computes that 2x2 Jacobian by centered mass-grid finite
differences, inventories its fold/near-fold locus, and gives explicit
non-injectivity certificates: far-separated mass pairs with almost identical
invariant coordinates.

This does not prove family connectivity; it tests a concrete alternative
explanation for multiple functional branches in invariant space.
"""
from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from threebody_atlas.baseline import BaselineRow, iter_baseline


def invariants(row: BaselineRow) -> tuple[float, float]:
    m1, m2, m3 = row.m1, row.m2, row.m3
    v3 = -(m1 * row.v1 + m2 * row.v2) / m3
    kinetic = 0.5 * (m1 * row.v1**2 + m2 * row.v2**2 + m3 * v3**2)
    potential = -(m1 * m2 / abs(1.0 - row.x1) + m1 * m3 / abs(row.x1) + m2 * m3)
    energy = kinetic + potential
    angular = m1 * row.x1 * row.v1 + m2 * row.v2
    mt = m1 + m2 + m3
    # The topological word length is constant for the catalog, so omitting it
    # only rescales the T coordinate by one constant and cannot create/remove folds.
    tsi = row.period * abs(energy) ** 1.5 / mt**2.5
    lsi = angular * abs(energy) ** 0.5 / mt ** (13.0 / 6.0)
    return tsi, lsi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("output")
    parser.add_argument("--step", type=float, default=0.001)
    parser.add_argument(
        "--fold-ratio",
        type=float,
        default=0.03,
        help="dimensionless smallest/largest singular-value threshold",
    )
    parser.add_argument(
        "--far-mass-distance",
        type=float,
        default=0.05,
        help="minimum Euclidean distance in (m1,m2) for non-injectivity certificates",
    )
    args = parser.parse_args()

    rows = list(iter_baseline(args.dataset))
    key = {(round(r.m1, 6), round(r.m2, 6)): r for r in rows}
    inv = {k: np.asarray(invariants(r), dtype=float) for k, r in key.items()}
    all_keys = list(inv)
    all_values = np.asarray([inv[k] for k in all_keys])
    masses = np.asarray(all_keys, dtype=float)
    out_mean = np.mean(all_values, axis=0)
    out_scale = np.std(all_values, axis=0)
    out_scale = np.maximum(out_scale, np.finfo(float).eps)
    normalized_values = (all_values - out_mean) / out_scale
    h = args.step

    records = []
    by_key = {}
    for k, row in key.items():
        m1, m2 = k
        km = (round(m1 - h, 6), m2)
        kp = (round(m1 + h, 6), m2)
        lm = (m1, round(m2 - h, 6))
        lp = (m1, round(m2 + h, 6))
        if not all(x in inv for x in (km, kp, lm, lp)):
            continue
        d1 = (inv[kp] - inv[km]) / (2 * h) / out_scale
        d2 = (inv[lp] - inv[lm]) / (2 * h) / out_scale
        jac = np.column_stack((d1, d2))
        singular = np.linalg.svd(jac, compute_uv=False)
        ratio = float(singular[-1] / singular[0]) if singular[0] > 0 else 0.0
        determinant = float(np.linalg.det(jac))
        rec = {
            "m1": row.m1,
            "m2": row.m2,
            "m3": row.m3,
            "T_si": float(inv[k][0]),
            "L_si": float(inv[k][1]),
            "normalized_projection_det": determinant,
            "normalized_singular_values": [float(x) for x in singular],
            "rank_ratio": ratio,
        }
        records.append(rec)
        by_key[k] = rec

    ratios = np.asarray([r["rank_ratio"] for r in records])
    dets = np.asarray([r["normalized_projection_det"] for r in records])
    fold_keys = {
        (round(r["m1"], 6), round(r["m2"], 6))
        for r in records
        if r["rank_ratio"] <= args.fold_ratio
    }

    # A sign change of det between adjacent interior cells is a discrete fold
    # indicator even when the exact zero lies between grid nodes.
    sign_change_edges = []
    sign_fold_keys = set()
    for k, rec in by_key.items():
        for delta in ((h, 0.0), (0.0, h)):
            neighbor = (round(k[0] + delta[0], 6), round(k[1] + delta[1], 6))
            if neighbor not in by_key:
                continue
            a = rec["normalized_projection_det"]
            b = by_key[neighbor]["normalized_projection_det"]
            if a == 0.0 or b == 0.0 or a * b < 0.0:
                sign_change_edges.append(
                    {
                        "left": [k[0], k[1]],
                        "right": [neighbor[0], neighbor[1]],
                        "det_left": a,
                        "det_right": b,
                    }
                )
                fold_keys.add(k)
                fold_keys.add(neighbor)
                sign_fold_keys.add(k)
                sign_fold_keys.add(neighbor)

    def components_of(nodes: set[tuple[float, float]]) -> list[list[tuple[float, float]]]:
        unseen = set(nodes)
        components = []
        while unseen:
            seed = unseen.pop()
            queue = deque([seed])
            component = [seed]
            while queue:
                u = queue.popleft()
                for delta in ((h, 0.0), (-h, 0.0), (0.0, h), (0.0, -h)):
                    v = (round(u[0] + delta[0], 6), round(u[1] + delta[1], 6))
                    if v in unseen:
                        unseen.remove(v)
                        queue.append(v)
                        component.append(v)
            components.append(component)
        components.sort(key=len, reverse=True)
        return components

    components = components_of(fold_keys)
    sign_components = components_of(sign_fold_keys)

    # Explicit non-injectivity certificates. Query a small nearest-neighbor set in
    # standardized invariant space and ask whether a near match is far away in
    # mass space. This attacks the use of (T_si,L_si) as a family identifier.
    tree = cKDTree(normalized_values)
    distances, neighbors = tree.query(normalized_values, k=40)
    far_matches = []
    for i in range(len(rows)):
        for distance, j in zip(distances[i, 1:], neighbors[i, 1:], strict=True):
            mass_distance = float(np.linalg.norm(masses[i] - masses[int(j)]))
            if mass_distance >= args.far_mass_distance:
                far_matches.append(
                    {
                        "standardized_invariant_distance": float(distance),
                        "mass_distance": mass_distance,
                        "left": {
                            "index": rows[i].index,
                            "masses": [rows[i].m1, rows[i].m2, rows[i].m3],
                            "T_si": float(all_values[i, 0]),
                            "L_si": float(all_values[i, 1]),
                        },
                        "right": {
                            "index": rows[int(j)].index,
                            "masses": [rows[int(j)].m1, rows[int(j)].m2, rows[int(j)].m3],
                            "T_si": float(all_values[int(j), 0]),
                            "L_si": float(all_values[int(j), 1]),
                        },
                    }
                )
                break
    far_matches.sort(key=lambda x: x["standardized_invariant_distance"])
    match_distances = np.asarray(
        [x["standardized_invariant_distance"] for x in far_matches], dtype=float
    )

    payload = {
        "rows": len(rows),
        "interior_jacobians": len(records),
        "output_scales": {"T_si_std": float(out_scale[0]), "L_si_std": float(out_scale[1])},
        "rank_ratio_quantiles": {
            str(q): float(np.quantile(ratios, q)) for q in (0.0, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0)
        },
        "determinant_quantiles": {
            str(q): float(np.quantile(dets, q)) for q in (0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0)
        },
        "determinant_sign_counts": {
            "negative": int(np.sum(dets < 0)),
            "zero": int(np.sum(dets == 0)),
            "positive": int(np.sum(dets > 0)),
        },
        "determinant_sign_change_edges": len(sign_change_edges),
        "fold_rank_ratio_threshold": args.fold_ratio,
        "fold_candidate_nodes": len(fold_keys),
        "fold_component_sizes": [len(c) for c in components[:20]],
        "sign_fold_nodes": len(sign_fold_keys),
        "sign_fold_component_sizes": [len(c) for c in sign_components[:20]],
        "largest_sign_fold_component_bbox": None
        if not sign_components
        else {
            "m1": [min(x[0] for x in sign_components[0]), max(x[0] for x in sign_components[0])],
            "m2": [min(x[1] for x in sign_components[0]), max(x[1] for x in sign_components[0])],
        },
        "far_mass_distance_threshold": args.far_mass_distance,
        "far_mass_invariant_neighbor_count": len(far_matches),
        "far_mass_near_match_counts": {
            "distance_lt_1e-4": int(np.sum(match_distances < 1e-4)),
            "distance_lt_3e-4": int(np.sum(match_distances < 3e-4)),
            "distance_lt_1e-3": int(np.sum(match_distances < 1e-3)),
            "distance_lt_3e-3": int(np.sum(match_distances < 3e-3)),
        },
        "best_noninjectivity_certificates": far_matches[:50],
        "lowest_rank_ratio_points": sorted(records, key=lambda r: r["rank_ratio"])[:50],
        "sign_change_examples": sign_change_edges[:100],
        "interpretation": (
            "A nonempty projection fold set and far-separated masses with nearly identical "
            "(T_si,L_si) are compatible with one connected 2D continuation sheet appearing "
            "as multiple functional branches after projection. Dynamical-family identity still "
            "requires continuation/rank evidence in shooting space."
        ),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "interior_jacobians": len(records),
                "min_rank_ratio": float(np.min(ratios)),
                "det_sign_counts": payload["determinant_sign_counts"],
                "sign_change_edges": len(sign_change_edges),
                "sign_fold_nodes": len(sign_fold_keys),
                "sign_fold_components": payload["sign_fold_component_sizes"][:10],
                "far_mass_invariant_neighbor_count": len(far_matches),
                "near_match_counts": payload["far_mass_near_match_counts"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
