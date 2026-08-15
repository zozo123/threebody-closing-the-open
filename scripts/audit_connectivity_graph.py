#!/usr/bin/env python3
"""Find macroscopic bottlenecks in the published shooting-chart adjacency graph.

This is a *diagnostic* for continuation connectivity, not a proof.  Every public
mass-grid row is a node.  Existing +/-0.001 mass-grid neighbors are connected by
an edge weighted by the scale-normalized change in (x1,v1,v2,T).  A minimum
spanning tree exposes the weakest links needed to connect the sampled catalog.

If a catalog is made from two macroscopically separated solution sheets, one
expects a large-weight MST edge whose removal separates two large subtrees.  We
emit those balanced bottleneck edges so that a real shooting continuation can
attack them directly in both directions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

from threebody_atlas.baseline import iter_baseline


def chart(row) -> np.ndarray:
    return np.asarray([row.x1, row.v1, row.v2, row.period], dtype=float)


def serialize_row(row) -> dict:
    return {
        "index": row.index,
        "masses": [row.m1, row.m2, row.m3],
        "x1": row.x1,
        "v1": row.v1,
        "v2": row.v2,
        "period": row.period,
        "published_stability": row.published_stability,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("output")
    args = parser.parse_args()

    rows = list(iter_baseline(args.dataset))
    n = len(rows)
    X = np.asarray([chart(r) for r in rows])
    key = {(round(r.m1 * 1000), round(r.m2 * 1000), round(r.m3 * 1000)): i for i, r in enumerate(rows)}

    pairs: list[tuple[int, int]] = []
    raw_deltas: list[np.ndarray] = []
    for i, r in enumerate(rows):
        k1, k2, k3 = round(r.m1 * 1000), round(r.m2 * 1000), round(r.m3 * 1000)
        for delta in ((1, 0), (0, 1)):
            j = key.get((k1 + delta[0], k2 + delta[1], k3))
            if j is not None:
                pairs.append((i, j))
                raw_deltas.append(np.abs(X[j] - X[i]))

    raw = np.asarray(raw_deltas)
    scales = np.median(raw, axis=0)
    scales = np.maximum(scales, np.finfo(float).eps)
    weights = np.asarray([np.linalg.norm((X[j] - X[i]) / scales) for i, j in pairs])

    rr: list[int] = []
    cc: list[int] = []
    dd: list[float] = []
    for (i, j), w in zip(pairs, weights, strict=True):
        rr.extend((i, j))
        cc.extend((j, i))
        dd.extend((float(w), float(w)))
    graph = coo_matrix((dd, (rr, cc)), shape=(n, n)).tocsr()
    mst = minimum_spanning_tree(graph).tocoo()
    if len(mst.data) != n - 1:
        raise RuntimeError(f"mass-grid adjacency graph is disconnected: MST edges={len(mst.data)} n={n}")

    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for i, j, w in zip(mst.row, mst.col, mst.data, strict=True):
        a, b, wf = int(i), int(j), float(w)
        adjacency[a].append((b, wf))
        adjacency[b].append((a, wf))

    parent = np.full(n, -1, dtype=int)
    parent_weight = np.zeros(n, dtype=float)
    order = [0]
    parent[0] = 0
    for u in order:
        for v, w in adjacency[u]:
            if parent[v] == -1:
                parent[v] = u
                parent_weight[v] = w
                order.append(v)
    if len(order) != n:
        raise RuntimeError("MST traversal did not reach every row")

    subtree = np.ones(n, dtype=int)
    for u in reversed(order[1:]):
        subtree[parent[u]] += subtree[u]

    cuts = []
    for v in range(1, n):
        u = int(parent[v])
        small = min(int(subtree[v]), n - int(subtree[v]))
        cuts.append((float(parent_weight[v]), small, u, v))
    cuts.sort(reverse=True)

    fractions = (0.01, 0.05, 0.10, 0.25, 0.40)
    selected = []
    used: set[tuple[int, int]] = set()
    for fraction in fractions:
        minimum = int(np.ceil(fraction * n))
        eligible = [c for c in cuts if c[1] >= minimum]
        if not eligible:
            continue
        w, small, u, v = eligible[0]
        edge_key = tuple(sorted((u, v)))
        if edge_key in used:
            continue
        used.add(edge_key)
        selected.append(
            {
                "minimum_split_fraction": fraction,
                "weight": w,
                "smaller_partition_size": small,
                "smaller_partition_fraction": small / n,
                "left": serialize_row(rows[u]),
                "right": serialize_row(rows[v]),
            }
        )

    global_top = [
        {
            "weight": w,
            "smaller_partition_size": small,
            "left_masses": [rows[u].m1, rows[u].m2, rows[u].m3],
            "right_masses": [rows[v].m1, rows[v].m2, rows[v].m3],
        }
        for w, small, u, v in cuts[:20]
    ]
    payload = {
        "rows": n,
        "adjacency_edges": len(pairs),
        "chart_delta_scales": {
            "x1": float(scales[0]),
            "v1": float(scales[1]),
            "v2": float(scales[2]),
            "period": float(scales[3]),
        },
        "adjacency_weight_quantiles": {
            str(q): float(np.quantile(weights, q)) for q in (0.5, 0.9, 0.99, 0.999, 1.0)
        },
        "mst_weight_quantiles": {
            str(q): float(np.quantile(mst.data, q)) for q in (0.5, 0.9, 0.99, 0.999, 1.0)
        },
        "global_top_mst_edges": global_top,
        "balanced_bottlenecks": selected,
        "interpretation": (
            "diagnostic only: macroscopic family identity requires actual continuation across "
            "the emitted balanced bottleneck edges"
        ),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "rows": n,
        "adjacency_edges": len(pairs),
        "mst_max": float(np.max(mst.data)),
        "balanced_bottlenecks": [
            {"weight": x["weight"], "split": x["smaller_partition_size"]} for x in selected
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
