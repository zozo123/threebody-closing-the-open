#!/usr/bin/env python3
"""Test whether apparent invariant branches are folds of one 2D orbit sheet.

The unequal-mass catalog is parameterized by two masses (m1,m2), with m3=1.
The 2023 two-family interpretation is based on the projection of those orbits to
scale-invariant period/angular-momentum coordinates.  A connected 2D manifold
can look multi-valued under such a projection whenever the projection Jacobian
loses rank.  This script computes that 2x2 Jacobian by centered mass-grid finite
differences and inventories its fold/near-fold locus.

This does not prove family connectivity; it tests a concrete alternative
explanation for multiple functional branches in invariant space.
"""
from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np

from threebody_atlas.baseline import BaselineRow, iter_baseline


def invariants(row: BaselineRow) -> tuple[float, float]:
    m1,m2,m3=row.m1,row.m2,row.m3
    v3=-(m1*row.v1+m2*row.v2)/m3
    kinetic=0.5*(m1*row.v1**2+m2*row.v2**2+m3*v3**2)
    potential=-(m1*m2/abs(1.0-row.x1)+m1*m3/abs(row.x1)+m2*m3)
    energy=kinetic+potential
    angular=m1*row.x1*row.v1+m2*row.v2
    mt=m1+m2+m3
    # The topological word length is constant for the catalog, so omitting it
    # only rescales the T coordinate by one constant and cannot create/remove folds.
    Tsi=row.period*abs(energy)**1.5/mt**2.5
    Lsi=angular*abs(energy)**0.5/mt**(13.0/6.0)
    return Tsi,Lsi


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("output")
    parser.add_argument("--step",type=float,default=0.001)
    parser.add_argument("--fold-ratio",type=float,default=0.03,
                        help="dimensionless smallest/largest singular-value threshold")
    args=parser.parse_args()

    rows=list(iter_baseline(args.dataset))
    key={(round(r.m1,6),round(r.m2,6)):r for r in rows}
    inv={k:np.asarray(invariants(r),dtype=float) for k,r in key.items()}
    all_values=np.asarray(list(inv.values()))
    out_scale=np.std(all_values,axis=0)
    out_scale=np.maximum(out_scale,np.finfo(float).eps)
    h=args.step

    records=[]
    by_key={}
    for k,row in key.items():
        m1,m2=k
        km=(round(m1-h,6),m2); kp=(round(m1+h,6),m2)
        lm=(m1,round(m2-h,6)); lp=(m1,round(m2+h,6))
        if not all(x in inv for x in (km,kp,lm,lp)):
            continue
        d1=(inv[kp]-inv[km])/(2*h)/out_scale
        d2=(inv[lp]-inv[lm])/(2*h)/out_scale
        J=np.column_stack((d1,d2))
        s=np.linalg.svd(J,compute_uv=False)
        ratio=float(s[-1]/s[0]) if s[0]>0 else 0.0
        det=float(np.linalg.det(J))
        rec={
            "m1":row.m1,"m2":row.m2,"m3":row.m3,
            "T_si":float(inv[k][0]),"L_si":float(inv[k][1]),
            "normalized_projection_det":det,
            "normalized_singular_values":[float(x) for x in s],
            "rank_ratio":ratio,
        }
        records.append(rec); by_key[k]=rec

    ratios=np.asarray([r["rank_ratio"] for r in records])
    dets=np.asarray([r["normalized_projection_det"] for r in records])
    fold_keys={
        (round(r["m1"],6),round(r["m2"],6))
        for r in records if r["rank_ratio"] <= args.fold_ratio
    }

    # A sign change of det between adjacent interior cells is a discrete fold
    # indicator even when the exact zero lies between grid nodes.
    sign_change_edges=[]
    for k,r in by_key.items():
        for delta in ((h,0.0),(0.0,h)):
            n=(round(k[0]+delta[0],6),round(k[1]+delta[1],6))
            if n not in by_key:
                continue
            a=r["normalized_projection_det"]
            b=by_key[n]["normalized_projection_det"]
            if a==0.0 or b==0.0 or a*b<0.0:
                sign_change_edges.append({
                    "left":[k[0],k[1]],"right":[n[0],n[1]],
                    "det_left":a,"det_right":b,
                })
                fold_keys.add(k); fold_keys.add(n)

    # Connected components of the fold candidate set on the mass grid.
    unseen=set(fold_keys); components=[]
    while unseen:
        seed=unseen.pop(); q=deque([seed]); comp=[seed]
        while q:
            u=q.popleft()
            for d in ((h,0.0),(-h,0.0),(0.0,h),(0.0,-h)):
                v=(round(u[0]+d[0],6),round(u[1]+d[1],6))
                if v in unseen:
                    unseen.remove(v); q.append(v); comp.append(v)
        components.append(comp)
    components.sort(key=len,reverse=True)

    payload={
        "rows":len(rows),
        "interior_jacobians":len(records),
        "output_scales":{"T_si_std":float(out_scale[0]),"L_si_std":float(out_scale[1])},
        "rank_ratio_quantiles":{
            str(q):float(np.quantile(ratios,q)) for q in (0.0,0.001,0.01,0.05,0.1,0.5,1.0)
        },
        "determinant_quantiles":{
            str(q):float(np.quantile(dets,q)) for q in (0.0,0.01,0.1,0.5,0.9,0.99,1.0)
        },
        "determinant_sign_counts":{
            "negative":int(np.sum(dets<0)),"zero":int(np.sum(dets==0)),"positive":int(np.sum(dets>0)),
        },
        "determinant_sign_change_edges":len(sign_change_edges),
        "fold_rank_ratio_threshold":args.fold_ratio,
        "fold_candidate_nodes":len(fold_keys),
        "fold_component_sizes":[len(c) for c in components[:20]],
        "largest_fold_component_bbox":None if not components else {
            "m1":[min(x[0] for x in components[0]),max(x[0] for x in components[0])],
            "m2":[min(x[1] for x in components[0]),max(x[1] for x in components[0])],
        },
        "lowest_rank_ratio_points":sorted(records,key=lambda r:r["rank_ratio"])[:50],
        "sign_change_examples":sign_change_edges[:100],
        "interpretation":(
            "A nonempty projection fold set is compatible with one connected 2D continuation sheet "
            "appearing as multiple functional branches in (T_si,L_si). Dynamical-family identity "
            "still requires continuation/rank evidence in shooting space."
        ),
    }
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({
        "interior_jacobians":len(records),
        "min_rank_ratio":float(np.min(ratios)),
        "det_sign_counts":payload["determinant_sign_counts"],
        "sign_change_edges":len(sign_change_edges),
        "fold_candidate_nodes":len(fold_keys),
        "largest_fold_components":payload["fold_component_sizes"][:10],
    },indent=2))


if __name__=="__main__":
    main()
