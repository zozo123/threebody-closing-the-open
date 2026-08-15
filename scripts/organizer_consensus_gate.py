#!/usr/bin/env python3
"""Require independent agreement before promoting mixed-organizer evidence.

This gate is intentionally stricter about provenance than about scientific
promotion. It combines three *different* lanes:

1. frozen Julia BigFloat + Vern9 mixed-organizer localization;
2. latest Julia/BifurcationKit independent Float64 correction;
3. CAPD rigorous full-period flow/C1 variational enclosure at the frozen seed.

Passing this script means the organizer is multi-formulation supported. It does
NOT by itself establish the final organizer certificate because the CAPD lane
currently validates the seed flow/monodromy rather than an interval-Newton root
of the full organizer equations, and canonical physical/Jordan checks remain a
separate release requirement.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

LABELS = (
    "principal-lower-left",
    "secondary-lower-switch",
    "principal-lower-right",
)


def _files(root: Path, label: str, token: str) -> list[Path]:
    return sorted(p for p in root.rglob("*.json") if label in p.name and token in p.name)


def _one(root: Path, label: str, token: str) -> Path:
    matches = _files(root, label, token)
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one {token} JSON for {label}, found {matches}")
    return matches[0]


def _f(value) -> float:
    return float(str(value))


def _bigfloat_record(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    results = data.get("results", [])
    if len(results) != 1:
        raise RuntimeError(f"{path}: expected one BigFloat result")
    root = results[0]["root"]
    gp = _f(root["plus_one_event"])
    gm = _f(root["minus_one_event"])
    alpha = _f(root["alpha"])
    beta = _f(root["beta"])
    closure = _f(root["closure_norm"])
    return {
        "path": str(path),
        "m1": _f(root["m1"]),
        "m2": _f(root["m2"]),
        "closure": closure,
        "event_norm": math.hypot(gp, gm),
        "spectral_vertex_error": math.hypot(alpha - 4.0, beta - 4.0),
        "alpha": alpha,
        "beta": beta,
        "plus_one_event": gp,
        "minus_one_event": gm,
    }


def _bk_record(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    masses = data["masses"]
    return {
        "path": str(path),
        "passed": bool(data.get("passed", False)),
        "m1": _f(masses[0]),
        "m2": _f(masses[1]),
        "closure": _f(data["closure_norm"]),
        "event_norm": _f(data["event_norm"]),
        "spectral_vertex_error": _f(data["spectral_vertex_error"]),
        "mass_shift_from_frozen_seed": _f(data["mass_shift_from_frozen_seed"]),
        "bifurcationkit_version": str(data.get("bifurcationkit_version", "unknown")),
        "julia_version": str(data.get("julia_version", "unknown")),
    }


def _capd_record(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "path": str(path),
        "passed": bool(data.get("passed", False)),
        "finite_enclosure": bool(data.get("finite_enclosure", False)),
        "collision_excluded_at_final_point_enclosure": bool(
            data.get("collision_excluded_at_final_point_enclosure", False)
        ),
        "collision_excluded_at_final_box_enclosure": bool(
            data.get("collision_excluded_at_final_box_enclosure", False)
        ),
        "max_point_monodromy_interval_width": _f(data["max_point_monodromy_interval_width"]),
        "max_box_monodromy_interval_width": _f(data["max_box_monodromy_interval_width"]),
        "proof_scope": str(data.get("proof_scope", "")),
    }


def evaluate(root: Path) -> dict:
    candidates = {}
    overall = True
    for label in LABELS:
        bf = _bigfloat_record(_one(root, label, "bigfloat"))
        bk = _bk_record(_one(root, label, "latest-bk"))
        capd = _capd_record(_one(root, label, "capd-validated-flow"))

        mass_disagreement = math.hypot(bf["m1"] - bk["m1"], bf["m2"] - bk["m2"])
        gates = {
            "bigfloat_closure": bf["closure"] <= 1e-18,
            "bigfloat_event_norm": bf["event_norm"] <= 1e-12,
            "bigfloat_vertex_error": bf["spectral_vertex_error"] <= 1e-10,
            "latest_bk_internal_gate": bk["passed"],
            "latest_bk_closure": bk["closure"] <= 2e-8,
            "latest_bk_event_norm": bk["event_norm"] <= 2e-6,
            "latest_bk_vertex_error": bk["spectral_vertex_error"] <= 5e-6,
            "bigfloat_vs_bk_mass_agreement": mass_disagreement <= 5e-4,
            "capd_validated_flow_pass": capd["passed"],
            "capd_finite_enclosure": capd["finite_enclosure"],
            "capd_final_collision_exclusion": (
                capd["collision_excluded_at_final_point_enclosure"]
                and capd["collision_excluded_at_final_box_enclosure"]
            ),
        }
        passed = all(gates.values())
        overall = overall and passed
        candidates[label] = {
            "passed": passed,
            "gates": gates,
            "bigfloat": bf,
            "latest_bifurcationkit": bk,
            "capd_validated_flow": capd,
            "bigfloat_vs_bk_mass_distance": mass_disagreement,
        }

    return {
        "claim_status": "cross_formulation_consensus_supported" if overall else "consensus_failed",
        "passed": overall,
        "promotion_boundary": (
            "Passing establishes agreement between frozen BigFloat localization, a latest-stack "
            "BifurcationKit correction, and rigorous CAPD seed-flow/variational enclosures. It is "
            "not a release claim: CAPD interval-Newton organizer existence, canonical physical/Jordan/"
            "Krein classification, event-arc continuation, and the remaining closure gates are separate."
        ),
        "candidates": candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="directory containing downloaded evidence artifacts")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = evaluate(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v["passed"] for k, v in result["candidates"].items()}, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
