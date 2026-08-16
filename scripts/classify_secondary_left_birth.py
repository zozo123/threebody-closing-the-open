#!/usr/bin/env python3
"""XOR-classify the secondary-left G- birth from a fold-geometry artifact.

Allowed classes: projection_fold, two_separate_arcs, mixed_organizer.
Newton-failed is never emitted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ALLOWED = frozenset({"projection_fold", "two_separate_arcs", "mixed_organizer"})


def classify(
    payload: dict[str, Any], bigfloat: dict[str, Any] | None = None
) -> dict[str, Any]:
    if bigfloat is not None:
        if not (
            payload.get("passed")
            and payload.get("generic_m1_fold_screen")
            and payload.get("opposite_branch_reconnection_screen")
            and bigfloat.get("passed") is True
        ):
            raise SystemExit(
                "fold closure requires passed bidirectional geometry and passed independent BigFloat verification"
            )
        closure = abs(float(bigfloat.get("closure_norm", "inf")))
        event = abs(float(bigfloat.get("minus_one_event", "inf")))
        audit = bigfloat.get("stationarity_stencil_audit") or []
        branch_audit = bigfloat.get("branch_curvature_audit") or {}
        branches = branch_audit.get("branches") or []
        if closure > 1e-7 or event > 2e-8 or not audit:
            raise SystemExit(
                f"independent fold fails frozen gates: closure={closure:.3e} event={event:.3e}"
            )
        finest = audit[-1]
        stationarity = abs(float(finest.get("dGdm2", "inf")))
        transverse = abs(float(finest.get("dGdm1", 0.0)))
        if stationarity > 1e-6 or transverse < 1.0:
            raise SystemExit(
                "independent fold nondegeneracy gates failed: "
                f"dGdm2={stationarity:.3e} dGdm1={transverse:.3e}"
            )
        if len(branches) != 2:
            raise SystemExit("independent fold needs exactly two corrected branch roots")
        if {int(row.get("cell_id", -1)) for row in branches} != {392, 393}:
            raise SystemExit("fold branches are not bound to source cells 392 and 393")
        if {str(row.get("orientation")) for row in branches} != {"U->S", "S->U"}:
            raise SystemExit("fold branches do not retain the two source orientations")
        dm2_signs: set[int] = set()
        for branch in branches:
            branch_closure = abs(float(branch.get("closure_norm", "inf")))
            branch_event = abs(float(branch.get("minus_one_event", "inf")))
            bracket = branch.get("source_m2_bracket") or []
            masses = branch.get("masses") or []
            if branch_closure > 1e-7 or branch_event > 2e-8:
                raise SystemExit(
                    "independent branch misses frozen gates: "
                    f"closure={branch_closure:.3e} event={branch_event:.3e}"
                )
            if len(bracket) != 2 or len(masses) < 2:
                raise SystemExit("independent branch lacks its source bracket or masses")
            m2 = float(masses[1])
            lo, hi = (float(value) for value in bracket)
            if not lo - 2e-9 <= m2 <= hi + 2e-9:
                raise SystemExit("independent branch root left its published source bracket")
            dm1 = float(branch.get("dm1_from_fold", "-inf"))
            dm2 = float(branch.get("dm2_from_fold", 0.0))
            curvature = float(branch.get("secant_m1_curvature", 0.0))
            if dm1 <= 0.0 or curvature < 0.1 or dm2 == 0.0:
                raise SystemExit(
                    "independent branch does not certify a newborn quadratic fold arc"
                )
            dm2_signs.add(1 if dm2 > 0.0 else -1)
        disagreement = float(
            branch_audit.get("relative_curvature_disagreement", "inf")
        )
        if dm2_signs != {-1, 1} or disagreement > 0.15:
            raise SystemExit(
                "two independent branches do not straddle one consistent fold: "
                f"signs={sorted(dm2_signs)} relative_disagreement={disagreement:.3e}"
            )
        return {
            "id": "secondary_left_birth",
            "kind": "endpoint",
            "class": "projection_fold",
            "passed": True,
            "status": "independently_reproduced",
            "estimator": bigfloat.get("implementation"),
            "evidence_level": "independently_reproduced",
            "requires_independent_verification": False,
            "note": (
                "The adjacent secondary G- walls reconnect through a nondegenerate "
                "m1-projection fold, independently anchored to source cells 392/393."
            ),
            "masses": bigfloat.get("masses"),
            "closure_norm": bigfloat.get("closure_norm"),
            "minus_one_event": bigfloat.get("minus_one_event"),
            "stationarity_stencil_audit": audit,
            "branch_curvature_audit": branch_audit,
            "edge_endpoint_bindings": payload.get("edge_endpoint_bindings") or [],
        }

    if payload.get("passed") and payload.get("generic_m1_fold_screen") and payload.get("opposite_branch_reconnection_screen"):
        klass = "projection_fold"
        note = "G- pair meets an m1 turning point with opposite-branch reconnection on the float64 geometry screen."
    elif payload.get("generic_m1_fold_screen") and not payload.get("opposite_branch_reconnection_screen"):
        klass = "two_separate_arcs"
        note = "An m1 turning screen exists but the opposite branch did not reconnect; treat as two arcs until a fold is bound."
    elif payload.get("localized_seeds") and len(payload.get("localized_seeds") or []) >= 2 and not payload.get("generic_m1_fold_screen"):
        klass = "two_separate_arcs"
        note = (
            "Two G- roots exist at the birth slice, but the fold/reconnection screen did not pass. "
            "This is a two-arc screening hypothesis, not a falsification or endpoint proof."
        )
    else:
        raise SystemExit("left-birth artifact does not force an allowed XOR class; refusing unknown")
    return {
        "id": "secondary_left_birth",
        "kind": "endpoint",
        "class": klass,
        "passed": False,
        "status": "screening_classification",
        "estimator": "float64_geometry_screen",
        "evidence_level": "screening",
        "requires_independent_verification": True,
        "note": note,
        "source_claim_status": payload.get("claim_status"),
        "generic_m1_fold_screen": bool(payload.get("generic_m1_fold_screen")),
        "opposite_branch_reconnection_screen": bool(payload.get("opposite_branch_reconnection_screen")),
        "trace_error": payload.get("trace_error"),
        "edge_endpoint_bindings": payload.get("edge_endpoint_bindings") or [],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("geometry_json")
    parser.add_argument("output")
    parser.add_argument("--bigfloat")
    args = parser.parse_args()
    payload = json.loads(Path(args.geometry_json).read_text(encoding="utf-8"))
    bigfloat = (
        json.loads(Path(args.bigfloat).read_text(encoding="utf-8"))
        if args.bigfloat
        else None
    )
    record = classify(payload, bigfloat)
    if record["class"] not in ALLOWED:
        raise SystemExit(f"illegal class {record['class']}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "class": record["class"],
                "passed": record["passed"],
                "requires_independent_verification": record["requires_independent_verification"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
