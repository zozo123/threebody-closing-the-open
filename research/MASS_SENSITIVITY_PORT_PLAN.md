# Replacing nested mass finite differences with integrated variational sensitivities

Status: **Python float64 lane implemented and validated.  Julia port NOT started
(deliberately: a fold verification is running against
`julia/verify_secondary_minus_fold.jl` right now and that file is untouched).**

Implementation: `src/threebody_atlas/mass_sensitivity.py` (derivation in the
module docstring).  Validation instrument:
`scripts/validate_mass_sensitivities.py`.  Tests: `tests/test_mass_sensitivity.py`.

---

## 1. What is being replaced

`julia/verify_secondary_minus_fold.jl` obtains every mass derivative by
differencing *complete independent periodic corrections*:

| routine | nodes | what it costs at dps=60 |
| --- | --- | --- |
| `five_point_m2(h)` / `five_point_m1(h)` | 4 corrections each | ~27 min per stencil |
| `fold_vector` | 1 `five_point_m2` | ~27 min |
| `fold_mass_jacobian` | 2 outer corrections + 2 `fold_vector` | dominates a ~92 min Newton step |
| `audit_fold(hs=[8e-5,4e-5,2e-5,1e-5])` | 20 distinct corrections after `CORRECTED_CACHE` hits | ~154 min measured |

(20 nodes at the nominal 5.5 min is 110 min against the 154 min the orchestrator
measured, so some nodes cost more than the nominal four Newton iterations.  The
projections below use the measured 154 min, i.e. ~28 correction-equivalents.)

One BigFloat `corrected_at` is ~5.5 min (four ~82 s Newton iterations).  This is
the finite differencing that killed three CI runs on the 120 min job wall.

## 2. What replaces it

For `zdot = f(z, m)` the mass sensitivity `S_i = dz/dm_i` obeys

```
S_i'   = D_z f . S_i + df/dm_i,                      S_i(0) = dz0/dm_i
Psi_i' = D_z f . Psi_i + [(D_z D_z f . S_i) + d(D_z f)/dm_i] Phi,  Psi_i(0) = 0
```

integrated alongside the trajectory and the fundamental matrix.  `Psi_i(T)` is
`dM/dm_i`, and every event derivative is then a single trace,
`dEvent/dm_i = tr(W dM/dm_i)` with `W = (alpha-2) I - M` for `G-`,
`(alpha-6) I - M` for `G+`, `(8-2 alpha) I + 4 M` for `Delta`.

Two subtleties, both handled and both pinned by negative-control tests:

* The COM reduction's mass-dependent reconstruction contributes **nothing** to
  `df/dm`: `(dP/dm) z` is a pure translation of all three bodies and the 12D
  Jacobian annihilates translations.  `test_mass_partial_has_no_hidden_com_reduction_term`
  checks this against `reduced.reduced_rhs`, which carries `P(m)` explicitly.
* The Li--Li--Liao chart's `v3 = -(m1 v1 + m2 v2)/m3` **is** mass dependent, so
  `dz0/dm != 0`.  `test_dropping_the_chart_gauge_term_is_detected` fails if that
  term is dropped.

The verifier's derivative is a *total* derivative along the corrected family, so
`dp/dm` comes from implicit differentiation of the (overdetermined but
consistent) 8x4 closure system, and the period contributes
`A(z(T)) M dT/dm` to `dM/dm`.

## 3. Validation actually performed (float64, this branch)

Five parameter points, all corrected first with the float64 Gauss--Newton chart
corrector (closure 3e-13 … 6e-12):

| point | masses (m1, m2) | `dG-/dm1` | `dG-/dm2` |
| --- | --- | --- | --- |
| `fold_root` | 0.9957049740, 0.9742604327 | 28.04556482 | 1.7e-07 (zero) |
| `fold_root_offset_m1_+1e-3` | 0.9967049740, 0.9742604327 | 31.43452486 | -1.19241597 |
| `fold_root_offset_m2_-2e-3` | 0.9957049740, 0.9722604327 | 30.38445987 | 0.39216189 |
| `branch_lower_cell392` | 0.996, 0.9645756252 | 39.88094515 | 2.45177095 |
| `branch_upper_cell393` | 0.996, 0.9841352873 | 16.73931325 | -0.99010697 |

At the frozen fold root
`m1=0.995704974019022863…`, `m2=0.974260432692528563…`, `m3=1`:

| check | what it exercises | agreement |
| --- | --- | --- |
| complex step `Im(M(m + i·1e-30))/1e-30` vs `Psi(T)` | `df/dm`, `dA/dm`, `D_zA·S`, `dc/dm`, the whole S/Psi integration — truncation-free | **1e-11 to 6e-10 relative**, all three event modes, both masses |
| least-squares residual of `J_p dp/dm = -dF/dm` | consistency of the implicit step | **1.3e-12 relative**; `J_p` singular values `[1.54e3, 3.69e1, 3.46e0, 1.46e-1]`, full rank 4 |
| central difference of the corrected chart `p*(m)` | `dp/dm` itself | **7.2e-7 relative** (finite-difference limited) |
| complex-step chain rule `dG/dm = ∂G/∂m + ∂G/∂p · dp/dm` | the Psi-with-total-initial-condition construction, independently | **1.8e-8 to 7.3e-7 relative** |
| Richardson-extrapolated corrected central difference — *the thing being replaced* | everything, end to end | **~1e-7 relative** on `dG-/dm1`, noise-floor limited |

Across all five points the worst numbers were: complex-step frozen-chart
agreement **2.8e-9** relative, chain-rule agreement **1.7e-6**, `dp/dm`
finite-difference agreement **8.1e-7**, Richardson-vs-analytic **6.3e-6**
relative (worst case is the smallest derivative, `dG-/dm2 = 0.392`, where the
absolute residual is 2.5e-6).  On `dG-/dm1`, which is O(20-40) everywhere, the
Richardson agreement is **1.6e-8 to 1.8e-7** relative.

Nothing disagreed beyond finite-difference truncation/round-off.  The
disagreements do not shrink with `h`, and they change sign between step sizes,
which is the signature of a float64 noise floor rather than a derivation error.

A sixth check, Clairaut symmetry of the mass Hessian: each row is one central
difference of the *exact* gradient, so `d2G-/dm1 dm2` is obtained twice along
different axes.  At the fold root they agree to **8.5e-6** relative at `h=5e-4`
and **3.9e-6** at `h=2e-4` -- improving like the outer difference's own `O(h^2)`,
which is what a correct construction looks like.

Values at the fold root (float64, DOP853 rtol 1e-13 / atol 1e-15):

```
G-          =  4.5e-10          (Julia event gate 1e-12 at dps=60)
dG-/dm2     =  1.7e-07          (stationarity; zero to the float64 floor)
dG-/dm1     =  28.0455648       (transversality gate |.| >= 1 -> passes with margin 28x)
d2G-/dm2^2  = -173.103          (converged over h = 1e-3, 5e-4, 2e-4)
m1''        =  6.1722           (= -d2G-/dm2^2 / (dG-/dm1))
```

Both newborn branch roots were then re-solved to `G- = 0` at `m1 = 0.996` using
the *exact* `dG-/dm2` as the Newton slope -- two iterations each, inside their
published brackets, and with no derivative stencil at all
(`solve_branch_event` currently spends two extra corrected samples per iteration
just to estimate that slope):

```
cell392 (U->S)  m2* = 0.9645748736853440   secant curvature 2 dm1/dm2^2 = 6.289859
cell393 (S->U)  m2* = 0.9841356178020694   secant curvature 2 dm1/dm2^2 = 6.050618
relative curvature disagreement = 0.0380     (Julia gate <= 0.15)  PASS
both curvatures >= 0.1                       (Julia gate)          PASS
```

The local curvature computed from exact derivatives, **6.1722**, lies between the
two secants.  That is an independent float64 corroboration of the fold geometry
the Julia verifier is trying to establish, obtained in ~3 minutes instead of
~2.5 hours.

**It is a corroboration, not a substitute.** It is float64, it does not satisfy
any release gate, and section 8 shows exactly why float64 cannot be trusted for
this class of quantity in general.  The BigFloat verifier remains required.

## 4. Measured cost, honestly

Same machine, same orbit (period 7.4553), DOP853 rtol 1e-13 / atol 1e-15, best
of three:

| operation | wall | in units of one chart correction |
| --- | --- | --- |
| one chart correction (4 Gauss--Newton iterations) | 2.44 s | 1.00 |
| `mass_sensitivity` — **both** mass derivatives of **all three** event modes | 4.39 s | **1.80** |
| one five-point mass stencil (4 corrected nodes) — one scalar derivative | 11.78 s | 4.83 |

So per scalar derivative the speedup is **2.7x**, and for the pair
`(dG/dm1, dG/dm2)` that `audit_fold` actually needs it is **5.4x** at a single
step size.  The large win comes from deleting the step-size ladder, which exists
only because the derivative is approximate.

An RHS-flop count gives 1.63 correction-equivalents for the sensitivity pass
against the measured 1.80, so the ratio should carry to BigFloat, where
per-operation cost is uniform and numpy's small-matrix overhead disappears.

### Projected `audit_fold` cost after the port

Using the brief's measured BigFloat numbers (5.5 min per correction, 154 min for
`audit_fold`, i.e. ~28 correction-equivalents):

| variant | correction-equivalents | projected wall |
| --- | --- | --- |
| current `audit_fold` | ~28 | **~154 min (measured)** |
| gated content only (`dG-/dm2`, `dG-/dm1`), one sensitivity call | 1.8 | **~10 min** — *15x* |
| + repeat at a tighter ODE tolerance instead of a step-size ladder | 3.6 | ~20 min — *7.7x* |
| + the reported (ungated) `d2G-/dm2^2` via one difference of exact gradients | 3.6 + 2 nodes + 3.6 | ~50 min — *3.1x* |
| ...with the `dp/dm` predictor warm-starting those two nodes (see §6) | ~8.2 | ~45 min — *3.4x* |

Note what disappears: `audit_fold`'s two stencil-convergence gates
(`abs(last.dGdm2 - prev.dGdm2) <= 5e-6`, `relative_change(last.dGdm1, prev.dGdm1) <= 0.05`)
exist purely to certify that the *finite difference* has converged in `h`.  With
an integrated sensitivity there is no `h`.  They must be replaced by an
ODE-tolerance convergence gate (recompute at `tol` and `tol/100`, require
agreement), **not** deleted.  A gate that becomes vacuous must be replaced by a
gate on the thing that actually limits the new computation.

`solve_fold` benefits too: the fold system is `F = (G-, dG-/dm2)` and its 2x2
mass Jacobian's **top row is free** (one sensitivity call gives `dG-/dm1` and
`dG-/dm2`); its bottom row is one central difference of the exact gradient in
`m2`, which yields `d2G-/dm1dm2` and `d2G-/dm2^2` together.  That is ~6
correction-equivalents (~33 min) against the measured ~92 min per Newton step,
and it removes the nested `inner_h`/`outer_h` pair entirely.

## 5. Julia port plan (do not start while a fold verification is in flight)

The Julia side already has everything except the mass partials.
`julia/verify_reduced.jl:reduced_rhs_jacobian` is the exact analogue of
`mass_sensitivity.reduced_field`, block for block.

1. **`julia/mass_sensitivity.jl` (new file, included by `verify_critical_points.jl`).**
   * `force_hessian_contract(x, s)` — equation (7).  Ten lines, mirrors
     `force_and_derivative`.
   * `reduced_mass_partials(z, masses)` returning `df/dm` (8x2), `dA/dm`
     (2 of 8x8).  Read straight off `reduced_rhs_jacobian`'s existing
     `addblock!` calls with the mass coefficients differentiated.
   * `augmented_sensitivity!(du, u, masses, t)` for a 216-component state
     `(z, Phi, S, Psi)` and an 88-component variant `(z, Phi, S)`.
   * `chart_mass_tangent(masses, p)` — equation (4); note it is the *same*
     `v3` gauge already encoded in `chart_tangent`.
2. **`mass_sensitivity(masses, p; tol)`** doing the two passes and the
   least-squares implicit step.  Julia's `\` on an 8x4 already does the
   least-squares solve, exactly as `correct_chart` uses it.
3. **Diagnostics that must be emitted, because they are the new checks:**
   * relative least-squares residual of the `dp/dm` solve (must be at BigFloat
     round-off; it is 1.3e-12 in float64),
   * the four singular values of `J_p` (smallest must not collapse),
   * the sensitivity recomputed at `tol/100`, with a convergence gate replacing
     the deleted stencil-convergence gates.
4. **Only then** rewrite `audit_fold` / `fold_mass_jacobian`.  Keep
   `five_point_m2`/`five_point_m1` in the file as a *cross-check path* runnable
   at one step size, so the first port run can prove agreement at dps=60 before
   the stencils stop being the primary.

### Bit-precision note

`Psi` is ~1e5 while the total `dG/dm` is ~1e2, so the total derivative loses
about three decimal digits to cancellation.  At dps=60 that is irrelevant; at
float64 it is exactly what limits the agreements in §3 to ~1e-7.  The Julia port
should not inherit float64's tolerance choices.

## 6. Follow-on that this unlocks (not done here)

* **Predictor warm start.** `dp/dm` gives a first-order predictor
  `p(m + dm) ~ p(m) + (dp/dm) dm` for any remaining corrected node.  At
  `dm = 1e-5` the predictor residual is `O(dm^2 |d2p/dm2|)`, so a node should
  need one Newton iteration instead of four.  Every surviving corrected node in
  the fold pipeline gets ~4x cheaper.  This is projected, not measured.
* **Adjoint form.** Only `tr(W Psi_i(T))` is needed, not `Psi_i(T)`.  A backward
  adjoint pass would replace the 128-component `Psi` block with an 8x8 costate,
  cutting the sensitivity pass to roughly the cost of one plain correction.
  Worth it only if the direct form turns out to be the bottleneck.
* **Second-order sensitivities.** `d2G/dm2^2` currently still needs two corrected
  nodes.  A full second-order variational system would remove them, at the cost
  of a third-derivative tensor of the Newtonian kernel.

## 7. A test of the polyline assumption that this makes cheap

The brief names the single largest unverified assumption in the critical graph:
`polyline_edges()` chains consecutive `m1` slices under `MASS_JUMP`, and an
adversarial audit found `edge_count` stays 7 for `MASS_JUMP` anywhere in
`[0.0112, 0.096]`, so the reconstruction performs essentially no branch
discrimination.  "Nearby roots really are one continuous critical curve" is
assumed, not tested.

Exact mass sensitivities give a direct test.  On a critical curve `G = 0` the
analytic tangent is

    dm2/dm1 = -(dG/dm1) / (dG/dm2)

so for two consecutive census roots that genuinely lie on ONE smooth curve, the
secant slope must equal the midpoint of the two endpoint tangents to `O(h^2)`.
If a chained pair actually straddles two branches, the secant is unrelated to
either tangent and the test fails loudly.

Run on the 19 consecutive `minus_one`, `U->S` census roots spanning
`m1 = 1.031 … 1.049` (cells 534…592, slice spacing `h = 0.001`), all 18 chained
pairs agree:

```
tangent slope falls monotonically 1.96597 -> 1.71769
secant vs midpoint-of-tangents relative disagreement: 8.5e-6 … 5.4e-5
```

which is exactly the `O(h^2)` midpoint-rule error.  **On this arm the chaining
assumption survives.**  The test has real discriminating power -- a mis-chain
would miss by O(1), not by 3e-5 -- and it costs one correction plus one
sensitivity per root, ~7 s each in float64.  Running it over all 620 census
roots is roughly 50 minutes of float64 and would convert the graph's largest
assumption into a measurement.  That is the highest-value follow-on this branch
enables and it was **not** run here.

## 8. FALSIFIED: float64 cannot evaluate these events to the frozen 2e-8 gate

Full write-up: **`research/EVENT_GATE_CONDITIONING.md`**.  Audit script:
`scripts/audit_event_conditioning.py`.

`beta = (alpha^2 - tr M^2)/2` is a cancellation, so the float64 event floor is
`eps * ||M||^2` (times 4 for `trace_collision`).  `||M||` across the census runs
7e2 … 2.4e4, putting that floor at 2e-10 … 5e-7 against a frozen 2e-8 gate.
Using the census's own `_flow_for_vector` at its own recorded charts, high-`||M||`
roots recorded at `|event| ~ 1.8e-8` recompute to `1e-7 … 2.5e-6` and do **not**
converge as the tolerance is tightened.  Separately, the recorded closure norms
are optimistic by 30-100x because they were measured at `screening_rtol`; one
sampled root's converged closure, 1.27e-7, is **above** the frozen 1e-7 gate.

On a seeded random sample of 160 of the 620 roots: **61%** of recorded event
values are not reproducible within the gate, **100%** of recorded closure norms
are optimistic by >10x, and 4 roots have a converged closure above the frozen
1e-7 gate.  The failure rate is monotone in `||M||` (37% below 2e3, 100% above
5e3).  The sharpest evidence is a jitter probe: fixing everything and varying
only the integrator's `first_step` moves one root's event by 1.35e-6, 67x the
gate.

This does not challenge the root locations (an event error of 1e-6 displaces a
root by 2e-9 … 1e-8 in `m2` at the worst cells, against 1e-3 cells).  It
challenges the certificates, and it means "max |event| = 1.9898e-8 against a
2e-8 gate, a 0.5% margin" is a statement about a number whose own uncertainty is
10-100x the gate.

**Good news for this branch's own lane:** `minus_one` is the one event mode where
not a single sampled root failed to reproduce (0 of 40).  The secondary `G- = 0`
fold lives entirely in `minus_one`, so the float64 corroboration in §3 stands and
`dG-/dm2` is safe as a Newton slope.

## 9. Scope discipline

* No numerical gate was loosened.  `mass_sensitivity.corrected_sample` adds a
  `max_closure = 1e-9` guard, which is two orders **stricter** than the project's
  frozen 1e-7 closure gate.
* Nothing in `research/evidence/` was written or read-modified, and
  `scripts/validate_mass_sensitivities.py` refuses an `--output` path under
  `research/evidence`.
* `julia/verify_secondary_minus_fold.jl` is unmodified.
* Environment: throwaway `python3 -m venv` (numpy 2.5.2, scipy 1.18.0,
  pytest 8.x, ruff), run with `PYTHONPATH=src`.  The repo's `uv` pin was not
  touched.
