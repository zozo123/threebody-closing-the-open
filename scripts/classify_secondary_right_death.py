#!/usr/bin/env python3
"""XOR-classify the secondary-right death from a wall-continue artifact.

Allowed classes: mixed_organizer, projection_fold, domain_boundary.
Newton-failed is never emitted.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ALLOWED = frozenset({"mixed_organizer", "projection_fold", "domain_boundary"})


def classify(
    payload: dict[str, Any], canonical: dict[str, Any] | None = None
) -> dict[str, Any]:
    direct = payload.get("direct_candidate")
    direct_error = payload.get("direct_error")
    approach = payload.get("approach") or []
    continuation_error = payload.get("continuation_error")
    closest = payload.get("closest_approach_mass_gap")

    mixed_ok = isinstance(direct, dict) and direct.get("success") is True
    if canonical is not None:
        if not mixed_ok or canonical.get("passed") is not True:
            raise SystemExit("canonical mixed verification requires a successful float64 seed and passed record")
        masses = canonical.get("masses") or []
        direct_masses = direct.get("masses") or []
        if len(masses) < 2 or len(direct_masses) < 2:
            raise SystemExit("canonical mixed verification is missing masses")
        mass_shift = math.hypot(
            float(masses[0]) - float(direct_masses[0]),
            float(masses[1]) - float(direct_masses[1]),
        )
        relative_closure = abs(float(canonical.get("relative_closure_norm", "inf")))
        relative_event = abs(float(canonical.get("relative_event_norm", "inf")))
        physical_plus = abs(float(canonical.get("physical_plus_one_event", "inf")))
        physical_minus = abs(float(canonical.get("physical_minus_one_event", "inf")))
        if mass_shift > 1e-4:
            raise SystemExit(f"canonical mixed root left the 1e-4 seed neighborhood: {mass_shift:.3e}")
        if relative_closure > 1e-7 or relative_event > 2e-8:
            raise SystemExit(
                f"canonical mixed root fails frozen gates: closure={relative_closure:.3e} "
                f"event={relative_event:.3e}"
            )
        if max(physical_plus, physical_minus) > 1e-8:
            raise SystemExit(
                f"physical quotient fails mixed-event gate: G+={physical_plus:.3e} G-={physical_minus:.3e}"
            )
        return {
            "id": "secondary_right_death",
            "kind": "endpoint",
            "class": "mixed_organizer",
            "passed": True,
            "status": "independently_reproduced",
            "note": "The two secondary walls end at an independently reproduced physical mixed (+1,-1) organizer.",
            "estimator": canonical.get("implementation"),
            "evidence_level": "physical",
            "masses": masses,
            "mass_shift_from_screen": mass_shift,
            "relative_closure_norm": canonical.get("relative_closure_norm"),
            "relative_event_norm": canonical.get("relative_event_norm"),
            "physical_plus_one_event": canonical.get("physical_plus_one_event"),
            "physical_minus_one_event": canonical.get("physical_minus_one_event"),
            "edge_endpoint_bindings": payload.get("edge_endpoint_bindings") or [],
        }

    if mixed_ok:
        return {
            "id": "secondary_right_death",
            "kind": "endpoint",
            "class": "mixed_organizer",
            "passed": False,
            "status": "candidate",
            "note": "Walls produced a simultaneous mixed seed; this is not a fourth organizer until BigFloat+physical bind.",
            "estimator": "float64_wall_screen",
            "evidence_level": "screening",
            "closest_approach_mass_gap": closest,
            "requires_gate_a": True,
            "edge_endpoint_bindings": payload.get("edge_endpoint_bindings") or [],
        }

    # Geometry-only classes. Continuation errors are diagnostics, not classes.
    gaps = [row.get("mass_gap") for row in approach if isinstance(row, dict) and "mass_gap" in row]
    if gaps and closest is not None and float(closest) <= 2e-3 and not mixed_ok:
        return {
            "id": "secondary_right_death",
            "kind": "endpoint",
            "class": "projection_fold",
            "passed": False,
            "status": "screening_classification",
            "note": (
                "Walls approach closely without a surviving mixed solve; this is a projection-fold "
                "hypothesis pending separate continuation and independent nondegeneracy evidence."
            ),
            "estimator": "float64_wall_screen",
            "evidence_level": "screening",
            "requires_independent_verification": True,
            "closest_approach_mass_gap": closest,
            "continuation_error": continuation_error,
            "direct_error": direct_error,
            "edge_endpoint_bindings": payload.get("edge_endpoint_bindings") or [],
        }
    if approach:
        reached_boundary = payload.get("reached_declared_domain_boundary") is True
        return {
            "id": "secondary_right_death",
            "kind": "endpoint",
            "class": "domain_boundary" if reached_boundary else "projection_fold",
            "passed": False,
            "status": "screening_classification",
            "note": (
                "The wall tape reached the declared mass-domain boundary; that attachment still "
                "requires a verified terminal continuation."
                if reached_boundary
                else "The walls ended in the interior without a mixed root. Nonconvergence is not "
                "an endpoint class; retain a separate-fold hypothesis pending verified continuation."
            ),
            "estimator": "float64_wall_screen",
            "evidence_level": "screening",
            "requires_independent_verification": True,
            "closest_approach_mass_gap": closest,
            "continuation_error": continuation_error,
            "direct_error": direct_error,
            "approach_steps": len(approach),
            "edge_endpoint_bindings": payload.get("edge_endpoint_bindings") or [],
        }
    raise SystemExit("right-death artifact has no wall tape and no mixed candidate; refusing unknown")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("screen_json")
    parser.add_argument("output")
    parser.add_argument("--canonical")
    args = parser.parse_args()
    payload = json.loads(Path(args.screen_json).read_text(encoding="utf-8"))
    canonical = (
        json.loads(Path(args.canonical).read_text(encoding="utf-8"))
        if args.canonical
        else None
    )
    record = classify(payload, canonical)
    if record["class"] not in ALLOWED:
        raise SystemExit(f"illegal class {record['class']}")
    if "newton" in str(record["class"]).lower():
        raise SystemExit("Newton-failed is not an allowed class")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"class": record["class"], "passed": record["passed"], "requires_gate_a": record.get("requires_gate_a")}, indent=2))


if __name__ == "__main__":
    main()
