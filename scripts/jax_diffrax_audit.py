#!/usr/bin/env python3
"""Cross-check adaptive JAX/Diffrax continuation derivatives against SciPy.

This audit is intentionally adversarial. JAX derivatives are not allowed to steer
continuation merely because autodiff produced numbers. We require agreement with
an independently integrated SciPy closure, converged central finite differences
of that closure, and SciPy variational finite differences of the smooth Floquet
critical event.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from threebody_atlas.critical_manifold import event_value
from threebody_atlas.jax_diffrax import adaptive_closure_and_jacobian, adaptive_event_and_gradient
from threebody_atlas.liao_family import state_from_chart
from threebody_atlas.reduced import (
    full_to_reduced,
    reduced_jacobian,
    reduced_rhs,
    stability_invariants,
)

EVENT_BY_EDGE = {"lower": "plus_one", "upper": "trace_collision"}


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


def initial_reduced(y: np.ndarray, m3: float) -> tuple[np.ndarray, np.ndarray, float]:
    x1, v1, v2, period, m1, m2 = [float(x) for x in y]
    masses = np.asarray([m1, m2, m3], dtype=float)
    full = state_from_chart((m1, m2, m3), x1, v1, v2)
    return full_to_reduced(full), masses, period


def scipy_closure(y: np.ndarray, m3: float) -> np.ndarray:
    z0, masses, period = initial_reduced(y, m3)
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


def scipy_event(y: np.ndarray, m3: float, mode: str) -> float:
    z0, masses, period = initial_reduced(y, m3)
    u0 = np.concatenate((z0, np.eye(8).ravel()))

    def augmented(t: float, u: np.ndarray) -> np.ndarray:
        z = u[:8]
        phi = u[8:].reshape(8, 8)
        return np.concatenate((
            reduced_rhs(t, z, masses),
            (reduced_jacobian(z, masses) @ phi).ravel(),
        ))

    sol = solve_ivp(
        augmented,
        (0.0, period),
        u0,
        method="DOP853",
        rtol=8e-12,
        atol=8e-14,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    floquet = stability_invariants(sol.y[8:, -1].reshape(8, 8))
    return float(event_value(floquet, mode))


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


def central_event_gradient(y: np.ndarray, m3: float, mode: str, rel_step: float) -> np.ndarray:
    grad = np.empty(6, dtype=float)
    for k in range(6):
        h = rel_step * max(abs(float(y[k])), 1.0)
        yp = y.copy()
        ym = y.copy()
        yp[k] += h
        ym[k] -= h
        grad[k] = (scipy_event(yp, m3, mode) - scipy_event(ym, m3, mode)) / (2.0 * h)
    return grad


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
        mode = EVENT_BY_EDGE[name]
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

        scipy_event_value = scipy_event(y, m3, mode)
        jax_event_value, jax_event_grad = adaptive_event_and_gradient(
            y,
            mode,
            m3=m3,
            rtol=5e-10,
            atol=5e-12,
            max_steps=1 << 18,
        )
        event_fd_coarse = central_event_gradient(y, m3, mode, 4e-6)
        event_fd_fine = central_event_gradient(y, m3, mode, 2e-6)
        event_fd_self_rel = rel_error(event_fd_fine, event_fd_coarse)
        event_grad_rel = rel_error(jax_event_grad, event_fd_fine)
        event_value_diff = abs(jax_event_value - scipy_event_value)
        allowed_event_grad_rel = max(0.08, 10.0 * event_fd_self_rel)

        eligible = bool(
            np.isfinite(jax_j).all()
            and np.isfinite(jax_event_grad).all()
            and closure_diff <= 5e-7
            and fd_self_rel <= 0.02
            and jax_rel <= allowed_jac_rel
            and event_value_diff <= 2e-4
            and event_fd_self_rel <= 0.05
            and event_grad_rel <= allowed_event_grad_rel
        )
        all_eligible &= eligible
        records.append(
            {
                "edge": name,
                "event_mode": mode,
                "y": y.tolist(),
                "scipy_closure_norm": float(np.linalg.norm(scipy_c)),
                "jax_closure_norm": float(np.linalg.norm(jax_c)),
                "closure_difference_norm": closure_diff,
                "closure_fd_self_relative_error": fd_self_rel,
                "jax_vs_scipy_closure_fd_relative_error": jax_rel,
                "allowed_closure_jacobian_relative_error": allowed_jac_rel,
                "scipy_event_value": scipy_event_value,
                "jax_event_value": jax_event_value,
                "event_value_difference": event_value_diff,
                "event_fd_self_relative_error": event_fd_self_rel,
                "jax_vs_scipy_event_gradient_relative_error": event_grad_rel,
                "allowed_event_gradient_relative_error": allowed_event_grad_rel,
                "eligible_for_continuation": eligible,
            }
        )
        print(
            name,
            "closure_diff=", closure_diff,
            "closure_jac_rel=", jax_rel,
            "event_diff=", event_value_diff,
            "event_grad_rel=", event_grad_rel,
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
