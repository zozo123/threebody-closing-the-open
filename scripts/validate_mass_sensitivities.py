#!/usr/bin/env python3
"""Validate integrated mass sensitivities against the stencils they replace.

Three independent checks, run at several parameter points including the
secondary G-=0 fold root:

1. COMPLEX STEP on the frozen-chart event derivative.  The reduced field is
   complex-analytic (no ``abs``/``norm`` anywhere in the RHS), so perturbing a
   mass by ``i*1e-30`` gives a truncation-free derivative.  This validates
   ``df/dm``, ``dA/dm``, ``D_z A . S``, ``dc/dm`` and the ``S``/``Psi``
   integration to round-off.  It cannot validate ``dp/dm``, because the implicit
   correction solve is not complex-analytic.

2. RICHARDSON-EXTRAPOLATED CENTRAL DIFFERENCE of the *corrected* event -- the
   exact quantity ``five_point_m1``/``five_point_m2`` compute in
   ``julia/verify_secondary_minus_fold.jl``, only in float64 and with each node
   an independent Gauss--Newton chart correction.  This is the check that can
   falsify the derivation as a whole.

3. CHAIN-RULE CROSS CHECK.  ``dG/dm`` rebuilt as ``dG/dm|_p + dG/dp . dp/dm``
   with every partial from a complex step (and the exact ``tr(W A M)`` for the
   period).  It shares only ``dp/dm`` with the sensitivity path, so it would
   disagree if the ``Psi``-with-total-initial-condition construction were wrong.

4. FINITE DIFFERENCE OF ``dp/dm`` ITSELF.  The corrected chart point is O(1) and
   resolved to ~1e-13, so differencing it is far better conditioned than
   differencing the event -- this is the sharp test of the implicit step.

5. WALL CLOCK for both paths.

Prints a JSON report.  With ``--output PATH`` it also writes it; PATH must not
be under ``research/evidence`` -- this script is a development instrument, not
an evidence producer.

Environment used for the numbers in the branch report: throwaway
``python3 -m venv`` with numpy 2.5.2 / scipy 1.18.0, ``PYTHONPATH=src``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from threebody_atlas import mass_sensitivity as ms  # noqa: E402

FOLD_M1 = 0.99570497401902286347943897515365414417943247746666915723061112683929439
FOLD_M2 = 0.97426043269252856357695668982517342681864426504710985325848382038181221
FOLD_P = (-0.35850520191051544, 1.2496554002627902, 0.45738583902640956, 7.455322355760655)

# Two published cell roots that bracket the fold in m2 at m1=0.996, from
# experiments/secondary_minus_fold_branches_bigfloat.tsv.
BRANCH_LOWER = (0.996, 0.9645756252404868, 1.0,
                (-0.3548988720153132, 1.2574612854316434, 0.4601909497390701, 7.4381234411120865))
BRANCH_UPPER = (0.996, 0.9841352872863689, 1.0,
                (-0.3624263504926571, 1.240999023346825, 0.4548704409731832, 7.475981586658505))

POINTS = [
    ("fold_root", (FOLD_M1, FOLD_M2, 1.0), FOLD_P),
    ("fold_root_offset_m1_+1e-3", (FOLD_M1 + 1e-3, FOLD_M2, 1.0), FOLD_P),
    ("fold_root_offset_m2_-2e-3", (FOLD_M1, FOLD_M2 - 2e-3, 1.0), FOLD_P),
    ("branch_lower_cell392", (BRANCH_LOWER[0], BRANCH_LOWER[1], 1.0), BRANCH_LOWER[3]),
    ("branch_upper_cell393", (BRANCH_UPPER[0], BRANCH_UPPER[1], 1.0), BRANCH_UPPER[3]),
]


def complex_step_frozen(p, masses, step=1e-30, *, rtol, atol):
    out = {mode: [] for mode in ms.EVENT_MODES}
    for axis in range(2):
        pert = [complex(x) for x in masses]
        pert[axis] += 1j * step
        _, mono = ms.flow_and_monodromy(p, pert, rtol=rtol, atol=atol)
        for mode in ms.EVENT_MODES:
            out[mode].append(float(np.imag(ms.event_from_monodromy(mono, mode)) / step))
    return {mode: np.asarray(v) for mode, v in out.items()}


def complex_step_chart(p, masses, index, step=1e-30, *, rtol, atol):
    """Frozen-mass event derivative with respect to chart component ``index``."""
    q = [complex(x) for x in p]
    q[index] += 1j * step
    _, mono = ms.flow_and_monodromy(q, [complex(m) for m in masses], rtol=rtol, atol=atol)
    return {
        mode: float(np.imag(ms.event_from_monodromy(mono, mode)) / step)
        for mode in ms.EVENT_MODES
    }


def chain_rule_cross_check(p, masses, dp_dm, *, rtol, atol):
    """Rebuild ``dEvent/dm`` from complex-step partials plus ``dp/dm``.

    This shares only ``dp/dm`` with ``mass_sensitivity``: every partial here is a
    truncation-free complex step (or, for the period, the exact ``tr(W A M)``).
    If the Psi-with-total-initial-condition construction were wrong, this route
    and that one would disagree.
    """
    z_t, mono = ms.flow_and_monodromy(p, masses, rtol=rtol, atol=atol)
    _, a_end = ms.reduced_field(z_t, masses)
    dm_partial = complex_step_frozen(p, masses, rtol=rtol, atol=atol)
    dp_partial = [complex_step_chart(p, masses, k, rtol=rtol, atol=atol) for k in range(3)]
    out = {}
    for mode in ms.EVENT_MODES:
        w = ms.event_weight(mono, mode)
        d_dt = float(np.trace(w @ (a_end @ mono)))
        total = []
        for axis in range(2):
            value = dm_partial[mode][axis] + d_dt * dp_dm[3][axis]
            for k in range(3):
                value += dp_partial[k][mode] * dp_dm[k][axis]
            total.append(float(value))
        out[mode] = np.asarray(total)
    return out


def dp_dm_finite_difference(masses, p, h, *, rtol, atol):
    """Central difference of the corrected chart point itself.

    ``p*(m)`` is resolved to ~1e-13 by the Gauss--Newton corrector and is O(1),
    so this finite difference is far better conditioned than differencing the
    event.  It is the sharp test of the implicit-function step, equation (8).
    """
    cols = []
    for axis in range(2):
        vals = []
        for sign in (+1.0, -1.0):
            m = list(masses)
            m[axis] += sign * h
            vals.append(ms.corrected_sample(m, p, rtol=rtol, atol=atol).p)
        cols.append((vals[0] - vals[1]) / (2.0 * h))
    return np.stack(cols, axis=1)


def _rel(a, b):
    scale = max(abs(a), abs(b), 1e-300)
    return abs(a - b) / scale


def run_point(name, masses, guess, *, rtol, atol, fd_steps, dp_steps, modes):
    t0 = time.perf_counter()
    corrected = ms.corrected_sample(masses, guess, rtol=rtol, atol=atol)
    correct_wall = time.perf_counter() - t0
    p = corrected.p

    t0 = time.perf_counter()
    sens = ms.mass_sensitivity(p, masses, rtol=rtol, atol=atol)
    sens_wall = time.perf_counter() - t0

    frozen, _ = ms.frozen_chart_mass_derivative(p, masses, rtol=rtol, atol=atol)
    cstep = complex_step_frozen(p, masses, rtol=rtol, atol=atol)

    record = {
        "name": name,
        "masses": [float(m) for m in masses],
        "chart": [float(x) for x in p],
        "closure_norm": corrected.closure_norm,
        "correction_iterations": corrected.iterations,
        "dp_lstsq_residual": sens.dp_lstsq_residual,
        "events": sens.events,
        "dp_dm": sens.dp_dm.tolist(),
        "wall_seconds": {
            "one_chart_correction": correct_wall,
            "sensitivity_both_masses": sens_wall,
        },
        "complex_step_frozen_chart": {},
        "finite_difference": {},
    }

    for mode in ms.EVENT_MODES:
        record["complex_step_frozen_chart"][mode] = {
            "analytic": frozen[mode].tolist(),
            "complex_step": cstep[mode].tolist(),
            "relative_error": [_rel(frozen[mode][i], cstep[mode][i]) for i in range(2)],
        }

    chain = chain_rule_cross_check(p, masses, sens.dp_dm, rtol=rtol, atol=atol)
    record["chain_rule_cross_check"] = {
        mode: {
            "sensitivity_path": sens.d_events_dm[mode].tolist(),
            "complex_step_chain": chain[mode].tolist(),
            "relative_error": [
                _rel(sens.d_events_dm[mode][i], chain[mode][i]) for i in range(2)
            ],
        }
        for mode in ms.EVENT_MODES
    }

    dp_fd = {}
    for h in dp_steps:
        fd = dp_dm_finite_difference(masses, p, h, rtol=rtol, atol=atol)
        dp_fd[f"h={h:g}"] = {
            "finite_difference": fd.tolist(),
            "max_abs_error": float(np.max(np.abs(fd - sens.dp_dm))),
            "max_rel_error": float(np.max(np.abs(fd - sens.dp_dm) / np.abs(sens.dp_dm))),
        }
    record["dp_dm_finite_difference"] = dp_fd

    fd_walls = []
    for mode in modes:
        entry = {"analytic": sens.d_events_dm[mode].tolist(), "axes": []}
        for axis in range(2):
            steps = []
            for h in fd_steps:
                t0 = time.perf_counter()
                rich, d1, d2 = ms.richardson_event_derivative(
                    masses, p, mode, axis, h, rtol=rtol, atol=atol
                )
                fd_walls.append(time.perf_counter() - t0)
                steps.append(
                    {
                        "h": h,
                        "central_h": d1,
                        "central_h_over_2": d2,
                        "richardson": rich,
                        "richardson_minus_analytic": rich - sens.d_events_dm[mode][axis],
                        "central_h_over_2_minus_analytic": d2 - sens.d_events_dm[mode][axis],
                    }
                )
            entry["axes"].append({"axis": "m1" if axis == 0 else "m2", "steps": steps})
        record["finite_difference"][mode] = entry
    if fd_walls:
        record["wall_seconds"]["one_richardson_pair_4_corrections"] = float(np.mean(fd_walls))
    return record


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rtol", type=float, default=1e-13)
    ap.add_argument("--atol", type=float, default=1e-15)
    ap.add_argument("--fd-steps", type=float, nargs="*", default=[1e-3, 2e-4])
    ap.add_argument("--dp-steps", type=float, nargs="*", default=[2e-4])
    ap.add_argument("--modes", nargs="*", default=["minus_one"])
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    if args.output is not None and "research/evidence" in args.output.as_posix():
        raise SystemExit("refusing to write into research/evidence: this is not an evidence producer")

    records = []
    for name, masses, guess in POINTS:
        if args.only and name not in args.only:
            continue
        rec = run_point(
            name, masses, guess,
            rtol=args.rtol, atol=args.atol, fd_steps=args.fd_steps,
            dp_steps=args.dp_steps, modes=args.modes,
        )
        records.append(rec)
        print(json.dumps(rec, indent=2), flush=True)

    report = {
        "implementation": "integrated variational mass sensitivities (float64 lane)",
        "rtol": args.rtol,
        "atol": args.atol,
        "points": records,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
