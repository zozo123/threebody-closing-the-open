"""Command-line entry points used both locally and by GitHub Actions."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from .dynamics import integrate_orbit
from .schema import OrbitCandidate, VerificationRecord, VerificationStatus
from .variational import compute_floquet


FIGURE_EIGHT = OrbitCandidate(
    source="standard equal-mass figure-eight regression fixture",
    masses=(1.0, 1.0, 1.0),
    period=6.32591398,
    initial_state=(
        0.97000436, -0.24308753,
        -0.97000436, 0.24308753,
        0.0, 0.0,
        0.4662036850, 0.4323657300,
        0.4662036850, 0.4323657300,
        -0.9324073700, -0.8647314600,
    ),
)


def screen(candidate: OrbitCandidate, *, floquet: bool = False) -> VerificationRecord:
    state = np.asarray(candidate.initial_state, dtype=float)
    masses = np.asarray(candidate.masses, dtype=float)
    orbit = integrate_orbit(state, masses, candidate.period)
    multipliers: list[tuple[float, float]] = []
    symplectic_defect = None
    if floquet:
        result = compute_floquet(state, masses, candidate.period)
        multipliers = [(float(z.real), float(z.imag)) for z in result.multipliers]
        symplectic_defect = result.symplectic_defect
    return VerificationRecord(
        status=VerificationStatus.SCREENED,
        candidate=candidate,
        closure_norm=orbit.closure_norm,
        energy_defect=abs(orbit.energy_final - orbit.energy_initial),
        angular_momentum_defect=abs(
            orbit.angular_momentum_final - orbit.angular_momentum_initial
        ),
        symplectic_defect=symplectic_defect,
        floquet_multipliers=multipliers,
        arithmetic="IEEE-754 float64 / scipy DOP853 (screening only)",
        precision_digits=15,
        code_revision=os.getenv("GITHUB_SHA"),
        notes=["This record is screening evidence, not a high-precision publication claim."],
    )


def _write(record: VerificationRecord, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(record.model_dump_json(indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "closure_norm": record.closure_norm,
        "energy_defect": record.energy_defect,
        "symplectic_defect": record.symplectic_defect,
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(prog="threebody-atlas")
    sub = parser.add_subparsers(dest="command", required=True)
    smoke = sub.add_parser("smoke", help="run deterministic figure-eight regression")
    smoke.add_argument("--floquet", action="store_true")
    smoke.add_argument("--output", type=Path, default=Path("artifacts/smoke.json"))
    verify = sub.add_parser("screen-json", help="screen one OrbitCandidate JSON record")
    verify.add_argument("input", type=Path)
    verify.add_argument("--floquet", action="store_true")
    verify.add_argument("--output", type=Path, default=Path("artifacts/screen.json"))
    args = parser.parse_args()

    if args.command == "smoke":
        _write(screen(FIGURE_EIGHT, floquet=args.floquet), args.output)
    elif args.command == "screen-json":
        candidate = OrbitCandidate.model_validate_json(args.input.read_text(encoding="utf-8"))
        _write(screen(candidate, floquet=args.floquet), args.output)


if __name__ == "__main__":
    main()
