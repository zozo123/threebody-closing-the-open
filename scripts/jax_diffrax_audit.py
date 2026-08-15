#!/usr/bin/env python3
"""Cross-check adaptive JAX/Diffrax closure sensitivities against SciPy.

This audit is intentionally adversarial. JAX derivatives are not allowed to steer
continuation merely because autodiff produced numbers. We first require agreement
with the existing reduced dynamics, an independently integrated SciPy closure,
and converged central finite differences of that SciPy closure map.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from threebody_atlas.jax_diffrax import adaptive_closure_and_jacobian
from threebody_atlas.liao_family import state_from_chart
from threebody_atlas.reduced import full_to_reduced, reduced_rhs


def read_points(path: Path) -> list[tuple[str, np.ndarray, float]]:
    out: list[tuple[str, np.ndarray, float]] = []
    with path.open(encoding="utf-8") as handle:
        rows = [line for line in handle if line.strip() and not line.startswith("#")]
    reader = csv.reader(rows, delimiter="\t")
    for f in reader:
        edge, side = f[0], f[1]
        if side != "stable":
            continue
        m1, m2, m3 = map(float, f[2:5])
        x1, v1, v2, period = map(float, f[5:9])
        y = np.asarray([x1, v1, v2, period, m1, m2], dtype=float)
        out.append((edge, y, m3))
    return out


def scipy_closure(y: np.ndarray, m3: float) -> np.ndarray:
    x1, v1, v2, period, m1, m2 = [float(x) for x in y]
    masses = np.asarray([m1, m2, m3], dtype=float)
    full = state_from_chart((m1, m2, m3), x1, v1, v2)
    z0 = full_to_reduced(full)
    sol = solve_ivp(
        lambda t, z: reduced_rhs(t, z, masses),
        (0.0, period),
        z0,
        method="DOP853",
        rtol=2e-12,
        atol=2e-14,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y[:, -1] - z0


def central_jacobian(y: np.ndarray, m3: float, rel_step: float) -> np.ndarray:
    jac = np.empty((8, 6), dtype=float)
    for k in range(6):
        h = rel_step * max(abs(float(y[k])), 1.0)
        yp = y.copy()
        ym = y.copy()
        yp[k] += h
        ym[k] -= h
        jac[:, k] = (scipy_closure(yp, m3) - scipy_closure(ym, m3)) / (2.0 * h)
    return jac


def rel_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("seeds", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--require-eligible", action="store_true")
    args = parser.parse_args()

    import diffrax
    import jax

    if not jax.config.x64_enabled:
        raise SystemExit("JAX x64 is required; set JAX_ENABLE_X64=1")

    records = []
    all_eligible = True
    for name, y, m3 in read_points(args.seeds):
        scipy_c = scipy_closure(y, m3)
        jax_c, jax_j = adaptive_closure_and_jacobian(
            y,
            m3=m3,
            rtol=1e-10,
            atol=1e-12,
            max_steps=1 << 18,
        )
        fd_coarse = central_jacobian(y, m3, 2e-6)
        fd_fine = central_jacobian(y, m3, 1e-6)
        fd_self_rel = rel_error(fd_fine, fd_coarse)
        jax_rel = rel_error(jax_j, fd_fine)
        closure_diff = float(np.linalg.norm(jax_c - scipy_c))
        allowed_jac_rel = max(0.03, 8.0 * fd_self_rel)
        eligible = bool(
            np.isfinite(jax_j).all()
            and closure_diff <= 5e-7
            and fd_self_rel <= 0.02
            and jax_rel <= allowed_jac_rel
        )
        all_eligible &= eligible
        records.append(
            {
                "edge": name,
                "y": y.tolist(),
                "scipy_closure_norm": float(np.linalg.norm(scipy_c)),
                "jax_closure_norm": float(np.linalg.norm(jax_c)),
                "closure_difference_norm": closure_diff,
                "fd_self_relative_error": fd_self_rel,
                "jax_vs_scipy_fd_relative_error": jax_rel,
                "allowed_jacobian_relative_error": allowed_jac_rel,
                "eligible_for_continuation": eligible,
            }
        )
        print(
            name,
            "closure_diff=", closure_diff,
            "fd_self_rel=", fd_self_rel,
            "jax_rel=", jax_rel,
            "eligible=", eligible,
        )

    payload = {
        "implementation": "JAX x64 + Diffrax adaptive Dopri8 vs SciPy DOP853",
        "jax_version": jax.__version__,
        "diffrax_version": getattr(diffrax, "__version__", "unknown"),
        "all_eligible_for_continuation": all_eligible,
        "records": records,
        "claim_status": "derivative QA only; not publication evidence",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.require_eligible and not all_eligible:
        raise SystemExit("JAX/Diffrax derivative audit did not pass continuation gates")


if __name__ == "__main__":
    main()
