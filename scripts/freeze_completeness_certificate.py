#!/usr/bin/env python3
"""Freeze a bounded completeness certificate from AL + neck raster artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--al-screen", required=True)
    parser.add_argument("--neck-scan")
    args = parser.parse_args()
    al = json.loads(Path(args.al_screen).read_text(encoding="utf-8"))
    accepted = al.get("accepted_candidates", [])
    attempted = al.get("attempted", accepted)
    stable = [row for row in accepted if row.get("corrected", {}).get("screening_stable")]
    neck = None
    if args.neck_scan:
        neck = json.loads(Path(args.neck_scan).read_text(encoding="utf-8"))
    al_clean = bool(accepted) and not stable
    neck_done = isinstance(neck, dict) and neck.get("grid") is not None
    passed = al_clean and neck_done
    record: dict[str, Any] = {
        "schema": "atlas.v1.completeness-certificate/1",
        "passed": passed,
        "domain": {
            "catalog": "Li-Li-Liao unequal-mass non-hierarchical published grid",
            "neck": None if not neck else neck.get("grid"),
        },
        "resolution": None if not neck else neck.get("grid", {}).get("step"),
        "active_learning": {
            "attempted": len(attempted),
            "accepted": len(accepted),
            "screening_stable_hidden_pockets": len(stable),
            "interpretation": (
                "off-grid proposals corrected onto the known sheet; no hidden stable pocket in this sample"
                if al_clean
                else "inspect accepted stable points before freezing completeness"
            ),
        },
        "neck": None
        if not neck
        else {
            "samples": neck.get("grid", {}).get("samples"),
            "minimum_resolved_unstable_gap": neck.get("minimum_resolved_unstable_gap"),
            "any_vertical_merge": neck.get("any_vertical_merge"),
            "max_shooting_residual": neck.get("max_shooting_residual"),
        },
        "note": (
            "Bounded completeness: no additional stability pocket found in the AL sample "
            "and the neck raster completed at the declared local resolution."
            if passed
            else "Completeness not frozen: need a completed neck raster plus a clean AL pocket screen."
        ),
    }
    text = json.dumps(record, indent=2) + "\n"
    record["sha256_self"] = hashlib.sha256(text.encode()).hexdigest()
    text = json.dumps(record, indent=2) + "\n"
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(text, encoding="utf-8")
    print(json.dumps({"passed": passed, "al_clean": al_clean, "neck_done": neck_done}, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
