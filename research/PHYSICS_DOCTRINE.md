# ATLAS physics doctrine

Status: durable repo artifact. Principles, equations, references and attack patterns only.
No result claims live here; results live in `research/RESULT_LEDGER.md` and
`research/evidence/`.

This document states what this project reasons *from*. Every equation below was checked
against `src/threebody_atlas` and, where the check produced a number, the number and the
script that produced it are reported. Where the code and the usual textbook convention
differ, the difference is called out rather than smoothed over.

Companion documents, not superseded by this one:

- `research/OPEN_PROBLEM.md` — the frozen question and the SOLVED conditions.
- `research/PROTOCOL.md` — the evidence ladder as process.
- `research/FLOQUET_EVENT_GEOMETRY.md` — the `(alpha,beta)` trace-invariant geometry.
- `research/STRONGEST_HAMMERS_2026-08-15.md` — the three verification lanes.

Nothing here is copied from any book. Works are cited by author and title; the ideas are
restated in our own words.

**Read Section 4.3 and Section 6.1 first if you are short of time.** Writing this document
required actually running the attacks it recommends, and one of them came back positive:
the frozen `2e-8` event gate is, for the `G+` and `Delta` mechanisms, below the
chart-to-chart reproducibility of the quantity it gates. That is a falsification of a
statement this project has been making, it is reported where it belongs rather than
softened, and it changes no gate.

---

## 1. The governing structure

### 1.1 Equations of motion

Three point masses `m_1, m_2, m_3 > 0` move in the plane under Newtonian gravity. With
`G = 1` (the code's default `g=1.0` in `dynamics.acceleration`), positions `r_i in R^2`:

```
   d^2 r_i / dt^2  =  sum_{j != i}  m_j (r_j - r_i) / |r_j - r_i|^3            (1)
```

The screening state vector is 12-dimensional and, in `dynamics.unpack_state`, laid out as

```
   y = (r_1, r_2, r_3, v_1, v_2, v_3) in R^12,
   y[0:6]  -> positions, reshaped (3,2)
   y[6:12] -> velocities, reshaped (3,2)
```

Equation (1) is a second-order ODE with a singular right-hand side; `acceleration` raises
`FloatingPointError("binary collision")` at `r_ij = 0` rather than returning a large number.
That refusal is doctrine, not defensive coding: near-collision arcs are where every
error estimate in this project silently degrades, and a validated-numerics lane must
*reject* a box that approaches a binary collision rather than integrate through it.

### 1.2 Hamiltonian form and the ten classical integrals

With momenta `p_i = m_i v_i`,

```
   H(r,p) = sum_i |p_i|^2 / (2 m_i)  -  sum_{i<j} m_i m_j / |r_i - r_j|        (2)
```

verified against `dynamics.total_energy` (kinetic `0.5 * sum m_i |v_i|^2`, potential
`- sum_{i<j} m_i m_j / r_ij`).

The classical integrals of the planar problem, all present in the code:

```
   H                                   energy                (total_energy)
   L = sum_i m_i (x_i vy_i - y_i vx_i) angular momentum      (angular_momentum)
   P = sum_i m_i v_i in R^2            linear momentum       (center_of_mass, 2nd return)
   C = (sum_i m_i r_i) / M             centre of mass        (center_of_mass, 1st return)
```

with `M = m_1+m_2+m_3`, and the two further integrals `C - (P/M) t`. That is
`1 + 1 + 2 + 2 = 6` independent integrals in the planar case (ten in the spatial case).
Bruns and Poincare (Section 8) say there are no further ones of the classical kinds.

### 1.3 Reduction, and why it must precede the numerics

Two reduced charts are in the repo and they are used for different jobs.

**(a) The screening chart, `reduced.py` — 8D, relative, non-canonical.**

```
   z = (r_1 - r_3, r_2 - r_3, v_1 - v_3, v_2 - v_3) in R^8                     (3)
```

`reduction_matrix()` is the constant `8 x 12` matrix `R` implementing (3);
`reconstruction_matrix(m)` is the `12 x 8` matrix `P` that lifts back to the unique
`COM = 0, P_tot = 0` representative. Measured (`scratchpad/structure.py`):

```
   || R P - I_8 ||_inf = 0.000e+00   exactly, for every mass triple tested
   rank(P) = rank(R) = 8
   the COM and total-momentum functionals vanish on image(P) to 1.11e-16
```

This chart is **not canonical**. No symplectic form is claimed for it and none should be.
It is used because trace invariants are similarity invariants and therefore survive any
invertible change of chart (Section 6.1) — but Krein signatures and symplectic defects do
*not* survive, so they may never be read off this chart.

**(b) The publication chart, `canonical_jacobi.py` — 8D, canonical.**

```
   rho   = r_2 - r_1
   lam   = r_3 - (m_1 r_1 + m_2 r_2)/m_12,          m_12 = m_1 + m_2
   p_rho = mu_12 (v_2 - v_1),                       mu_12 = m_1 m_2 / m_12
   p_lam = mu_3 (v_3 - v_12),                       mu_3  = m_3 m_12 / M,
                                                    v_12  = (m_1 v_1 + m_2 v_2)/m_12
   z = (rho, lam, p_rho, p_lam) in R^8                                         (4)
```

The reduced Hamiltonian, with `x_13 = lam + a rho`, `x_23 = lam - b rho`,
`a = m_2/m_12`, `b = m_1/m_12` (these are exactly `r_3 - r_1` and `r_3 - r_2`):

```
   H = |p_rho|^2/(2 mu_12) + |p_lam|^2/(2 mu_3)
       - m_1 m_2 / |rho| - m_1 m_3 / |x_13| - m_2 m_3 / |x_23|                 (5)
```

`rhs_and_jacobian` implements `zdot = J grad H` with

```
   J = [[ 0_4,  I_4 ],
        [-I_4,  0_4 ]]                                                          (6)
```

(`canonical_jacobi.symplectic_matrix`). Term-by-term differentiation of (5) reproduces the
code's `grad_rho = m_1 m_2 g(rho) + m_1 m_3 a g(x_13) - m_2 m_3 b g(x_23)` and
`grad_lam = m_1 m_3 g(x_13) + m_2 m_3 g(x_23)` with `g(x) = x/|x|^3`. Checked.

**Why the reduction must happen before the numerics, not after.**

Three separate reasons, in increasing order of severity:

1. *The corrector is otherwise singular.* Periodic closure `Phi_T(y) - y = 0` on the raw
   12D state has a symmetry-generated kernel: translations (2), boosts (2), and the flow
   direction. A Newton or Gauss-Newton corrector on a residual with a structural null
   space is rank-deficient by construction, and the step it takes is decided by whatever
   regularisation the least-squares backend happens to apply. `liao_family` therefore
   builds its residual and its exact shooting Jacobian in the *reduced* coordinates (3),
   which remove the four translational/momentum directions exactly, before any Newton
   step is taken.
2. *Closure stops meaning what you think it means.* If `P_tot != 0`, the centre of mass
   drifts by `(P/M) T` over one period and `||y(T) - y(0)||` never reaches zero for a
   genuinely periodic relative motion. Measuring a "closure residual" that contains a
   secular symmetry drift makes the residual a function of the gauge, not of the orbit.
3. *The trivial multipliers are defective.* The monodromy of an autonomous Hamiltonian
   system has `lambda = 1` with a Jordan block (flow direction plus the `dT/dE`
   generalised direction), and the rotational symmetry adds a second. A Jordan block at
   `lambda = 1` perturbs like `sqrt(eps)`, so numerically computed "trivial" multipliers
   sit far from 1. Measured on frozen census root cell 497 at `rtol=5e-13`
   (`scratchpad/verify3.py`), the canonical 8x8 monodromy's four unit multipliers sit at

   ```
        |lambda - 1| = 2.30e-4, 2.30e-4, 6.45e-4, 6.45e-4
   ```

   Deleting "the four eigenvalues nearest 1" from an 8x8 spectrum is therefore a
   `1e-3`-accurate operation dressed up as an exact one. Do not do it. Two honest
   alternatives are in the repo and are described in Section 2.3.

### 1.4 The continuation chart

The published Li--Li--Liao baseline points are parameterised by
(`liao_family.state_from_chart`)

```
   r_1 = (x1, 0),  r_2 = (1, 0),  r_3 = (0, 0)
   v_1 = (0, v1),  v_2 = (0, v2),  v_3 = (0, -(m_1 v1 + m_2 v2)/m_3)           (7)
```

so the total momentum is zero by construction, and the shooting unknowns at fixed masses
are `(x1, v1, v2, T)`. The full continuation chart used by `critical_manifold` and
`critical_geometry` is

```
   y = (x1, v1, v2, T, m_1, m_2) in R^6,     m_3 = 1 fixed.                    (8)
```

Note what (7) does and does not fix: it fixes the momentum gauge and the rotational and
reflection gauges (all bodies on the x-axis, all velocities on the y-axis), but not the
translational gauge — the centre of mass of (7) is generally nonzero. This is harmless
because `P_tot = 0` makes it constant, but it means a 12D closure norm computed from (7)
is not the quantity the corrector minimises. Measured on cell 497: reduced closure
`1.31e-10` (the recorded value) versus 12D closure `3.22e-09` at `rtol=1e-12`.

A consequence worth stating: (7) also makes the section a *time-reversal-symmetric*
section, so the discrete symmetries "reverse all velocities" and "reflect `y -> -y`" act
identically on it. Both are exactly bit-preserving in this chart, which is why they are
worthless as metamorphic tests here (Section 6.1).

---

## 2. Floquet and spectral theory for periodic Hamiltonian systems

### 2.1 The reciprocal characteristic polynomial

Let `M` be the linearised return map of the periodic orbit on the physical transverse
quotient (Section 2.3), a `4 x 4` real symplectic matrix. Because `M` is symplectic its
spectrum is closed under `lambda -> 1/lambda`; because it is real it is closed under
`lambda -> conj(lambda)`. Hence multipliers come in the quadruples
`{lambda, 1/lambda, conj(lambda), 1/conj(lambda)}`, degenerating to reciprocal real pairs
or to conjugate unit-circle pairs. The characteristic polynomial is *reciprocal*:

```
   p(lambda) = lambda^4 - a lambda^3 + b lambda^2 - a lambda + 1               (9)

   a = tr M,      b = e_2(M) = ( (tr M)^2 - tr(M^2) ) / 2
```

Verified against `physical_floquet.physical_trace_invariants` line for line. The sign
convention matters and differs from several textbook presentations: here the coefficient
of `lambda^3` is `-a` with `a = +tr M`, so `a` is the trace itself and not its negation.

Substituting `t = lambda + 1/lambda` and dividing (9) by `lambda^2`:

```
   p(lambda)/lambda^2 = t^2 - a t + (b - 2) = 0                               (10)
```

The two roots `t_1, t_2` of (10) are the *trace roots*, one per reciprocal pair.

### 2.2 The three invariants and what they are

```
   G+    = p(+1) = b - 2a + 2       nontrivial pair at lambda = +1
   G-    = p(-1) = b + 2a + 2       nontrivial pair at lambda = -1
   Delta = disc(10) = a^2 - 4b + 8  collision of the two trace roots           (11)
```

All three verified against `physical_floquet.compute_physical_floquet`
(`plus = b - 2a + 2`, `minus = b + 2a + 2`, `discriminant = a*a - 4*b + 8`).

Equivalent product forms, verified numerically to `5.7e-14` over 20000 random `(t_1,t_2)`
(`scratchpad/structure.py`):

```
   G+    = (2 - t_1)(2 - t_2)
   G-    = (2 + t_1)(2 + t_2)
   Delta = (t_1 - t_2)^2                                                      (12)
```

**Naming hazard — read this before touching any event code.** The repo carries *two*
different meanings for the letters `alpha, beta`, and they differ by an affine shift:

| symbol | `reduced.py`, `critical_manifold.py`, `FLOQUET_EVENT_GEOMETRY.md` | `physical_floquet.py`, this document |
|---|---|---|
| chart | 8x8 relative monodromy `M_8` | 4x4 physical quotient `M_4` |
| first invariant | `alpha = tr M_8` | `a = tr M_4` |
| second invariant | `beta = e_2(M_8)` | `b = e_2(M_4)` |

They are related by algebraically stripping four unit multipliers:

```
   a = alpha - 4,        b = beta - 4 alpha + 10                              (13)
```

and therefore

```
   G+    = beta - 6 alpha + 20
   G-    = beta - 2 alpha + 4
   Delta = (alpha - 4)^2 - 4(beta - 4 alpha + 8) = alpha^2 + 8 alpha - 16 - 4 beta   (14)
```

which is exactly `critical_manifold.event_value`. Confirmed algebraically and numerically.
When reading a number called `alpha` in this repository, establish which chart produced it
before doing anything with it.

**Structural identity worth carrying in your head.** The map `(a,b) -> (G+, G-)` is an
affine bijection, so `Delta` is a polynomial in `G+` and `G-`:

```
   a = (G- - G+)/4,   b = (G+ + G-)/2 - 2
   Delta = (G- - G+)^2 / 16  -  2 (G+ + G-)  +  16                            (15)
```

verified to `4.3e-14` over 20000 random `(a,b)`. The three "independent" event functions
therefore live on a two-dimensional invariant space and `Delta = 0` is a **parabola** in
the `(G+, G-)` chart:

```
   Delta = 0   <=>   (G- - G+)^2 = 32 (G+ + G-) - 256                         (16)
```

In `(G+, G-)` coordinates the whole universal picture is: two orthogonal straight walls
(the coordinate axes) plus one parabola. The three exact vertices of
`FLOQUET_EVENT_GEOMETRY.md` become, in this chart (all verified):

| vertex | trace roots | `(alpha,beta)` | `(a,b)` | `(G+, G-, Delta)` |
|---|---|---|---|---|
| double `-1` | `(-2,-2)` | `(0,-4)` | `(-4, 6)` | `(16, 0, 0)` |
| mixed | `(-2,+2)` | `(4,4)` | `(0,-2)` | `(0, 0, 16)` |
| double `+1` | `(+2,+2)` | `(8,28)` | `(4, 6)` | `(0, 16, 0)` |

and this yields a fact with direct numerical consequences:

> Setting `G- = 0` in (16) gives `G+^2 - 32 G+ + 256 = 0`, a **double** root at `G+ = 16`.
> The `Delta = 0` wall is **tangent** to the `G- = 0` wall at the double-`-1` vertex, and
> by the `G+ <-> G-` symmetry of (16), tangent to `G+ = 0` at the double-`+1` vertex.
> By contrast `G+ = 0` and `G- = 0` are the coordinate axes and meet the mixed organiser
> **transversally**, with `Delta = 16 != 0` there.

Operationally: the mixed organiser `(alpha,beta) = (4,4)` is a well-conditioned
codimension-two intersection; the two double vertices are ill-conditioned tangential ones.
Any numerical claim that a `Delta` branch "meets" or "crosses" a `G+`/`G-` branch near a
double vertex is a near-tangency problem and must carry an explicit conditioning estimate.
Do not treat the three vertices as numerically equivalent objects.

### 2.3 Removing the trivial multipliers: two honest routes and one dishonest one

At a regular strict periodic orbit of the translation-reduced system, two commuting
symmetries survive: time translation (generator `X_H`) and planar rotation (generator
`X_L`). They span a two-dimensional isotropic subspace `E`, and `M E = E` exactly, since
`M X_H = X_H` (the flow direction) and `M X_L = X_L` (the rotation generator evaluated at
a fixed point of the return map). The physical transverse map acts on the four-dimensional
symplectic quotient `E^omega / E`.

- **Route A (`physical_floquet.quotient_monodromy`): build the quotient.** Choose the
  Euclidean-orthogonal complement `W` of `E` inside `E^omega`, set `A = Q^T M Q` with `Q`
  spanning `W`, and carry the induced form `Omega = Q^T J Q`. The construction reports its
  own defects: `quotient_symplectic_defect`, `quotient_leakage`
  (`||M Q - span(E,W) proj||`), `reciprocal_pairing_error`, `neutral_isotropy_defect`,
  `neutral_invariance_defect`. This is the publication route because those defects *are*
  the error bar.
- **Route B (`reduced.stability_invariants`): strip algebraically.** Use (13). This is
  exact in exact arithmetic — the trivial multipliers contribute exactly `4` to the trace
  whether or not they are semisimple, and their perturbations `1 +/- sqrt(eps)` cancel to
  first order in the trace — so the strip is second-order accurate even with a defective
  block. Measured: two *different* 8D translation reductions (relative chart (3) and
  canonical Jacobi (4)) agree under Route B to `1.4e-9` on cell 497.
- **Route C: delete the four eigenvalues nearest 1.** Forbidden. See the measured
  `|lambda - 1| ~ 6e-4` in Section 1.3. `physical_floquet`'s module docstring says the same
  thing.

Route B does **not** verify its own hypothesis. `stability_invariants` never checks that
the spectrum contains `1` with algebraic multiplicity four; it assumes it. That assumption
is correct mathematically, but the routine reports no defect of any kind, so a Route-B
`alpha, beta` carries no attached error estimate. Route A reports five. Section 4 is about
what happens when you gate a Route-B number at `2e-8`.

### 2.4 Unit-circle geometry

For a reciprocal pair with trace root `t`:

```
   t real, |t| < 2   ->  lambda = exp(+/- i theta), theta = arccos(t/2)   on the circle
   t = +/- 2         ->  lambda = +/- 1, double
   t real, |t| > 2   ->  lambda real reciprocal pair, one outside          unstable
   t complex         ->  the two pairs have merged into a reciprocal
                         quartet {lambda, conj lambda, 1/lambda, 1/conj lambda}  unstable
```

`reduced.stability_invariants` and `boundary.stability_score` implement exactly this:
`score = min(Delta, 2 - |t_1|, 2 - |t_2|)`, positive iff `Delta > 0` and both `|t_i| < 2`.
That is correct, and it matters that they use `|t_i| < 2` rather than the sign of the
three event functions — see Section 6.2 for why the sign vector alone is *not* a complete
stability invariant.

### 2.5 Krein signature

For a simple unit-circle multiplier `lambda`, `lambda != +/- 1`, with eigenvector `v`,
define the Krein form

```
   kappa(v) = -i * conj(v)^T Omega v                                          (17)
```

(`canonical_jacobi.krein_form` with `Omega = J`; `physical_floquet.physical_krein_form`
with `Omega = Q^T J Q`). It is real and nonzero for a simple unit-circle mode off `+/-1`,
its magnitude depends on the normalisation of `v` and is meaningless, and its **sign** is
invariant under any real change of basis of the quotient: with `Q' = Q S`, `S` real,
`Omega' = S^T Omega S` and `v' = S^{-1} v`, so `-i conj(v')^T Omega' v' = -i conj(v)^T Omega v`.
Measured on cell 497 (`scratchpad/verify_doctrine.py`): the conjugate pair
`lambda = 0.72072 +/- 0.69323 i` carries `kappa = -3.305e-1` and `+3.305e-1` — equal and
opposite, as required.

**Krein's theorem** (Krein; independently Gelfand and Lidskii; exposition and Hamiltonian
consequences in Moser, and in Yakubovich and Starzhinskii's monograph on linear
differential equations with periodic coefficients): a unit-circle multiplier whose Krein
form is *definite* on its eigenspace cannot leave the unit circle under small Hamiltonian
perturbation. Multipliers can therefore only leave the circle by colliding, and only a
collision of **opposite** Krein signature (or a collision at `+/-1`, which is signature
ambiguous) can produce instability.

This gives the mechanism taxonomy that the whole project rests on:

| event | what collides | what happens | name |
|---|---|---|---|
| `G+ = 0` | a reciprocal pair reaches `lambda = +1` | pair leaves along the real axis | `+1` crossing |
| `G- = 0` | a reciprocal pair reaches `lambda = -1` | pair leaves along the negative real axis | `-1` crossing |
| `Delta = 0`, `|t| < 2`, opposite Krein | two simple unit-circle pairs | quartet leaves the circle | Hamiltonian--Hopf (Krein collision) |
| `Delta = 0`, `|t| < 2`, same Krein | two simple unit-circle pairs | they pass through and stay on the circle | Krein-safe passing |
| `Delta = 0`, `|t| > 2` | two real reciprocal pairs | collision on the real axis | not a stability event |

**A Hamiltonian--Hopf is not a `+1` or `-1` crossing.** At `G+ = 0` or `G- = 0` a *single*
reciprocal pair degenerates and the linearised return map acquires a `+1` or `-1`
eigenvalue: the appropriate reduced object is a one-dimensional amplitude equation and the
generic outcome is a branch event (Section 3). At an opposite-Krein `Delta = 0` collision
*no* multiplier equals `+/-1`, the reduced object is two-dimensional (a `1:1` resonant
normal form), and the generic outcome is a complex quartet — spectral instability with no
new same-period branch. Conflating them is a category error, not a rounding issue.

**The transversality shortcut, and its price.** For a one-parameter family, a *transversal*
sign change of `Delta` at a collision with `|Re t| < 2` already forces opposite Krein
signature, because same-signature pairs cannot leave the circle and therefore cannot make
`Delta` change sign. So the checkable trio is:

1. `|Re t| < 2` at the event (unit-circle, not real-axis, collision);
2. `Delta` changes sign *transversally* (a tangential touch is a different codimension);
3. the collision is not at `t = +/- 2` (which is a double vertex, Section 2.2).

That trio is weaker than direct canonical Krein evidence and should be labelled as such.
Measured on the frozen census (`scratchpad/structure.py`):

```
   trace_collision roots                                        254 / 620
   with |Re t| < 2 at the recorded (alpha,beta)                 254 / 254
   with a recorded published-bracket sign change witness        151 / 254
      (the other 103 are BigFloat-escalated rows which do not
       carry source_event_endpoint_values at all)
   with ANY recorded Krein signature                              0 / 254
   of all 620 roots, records carrying a Krein signature            0 / 620
```

The repo already enforces the strict rule where it matters — the headline principal upper
collision has an independent `opposite_krein_hamiltonian_hopf_collision` record with the
canonical spectrum. The doctrine is that the *census* label `trace_collision` means
"`Delta = 0`" and nothing more, and must not be rendered as "Hamiltonian--Hopf" anywhere
in a manuscript without the Krein data or the explicit trio above.

---

## 3. Bifurcation and normal-form discipline

### 3.0 The rule

> **A bifurcation may be NAMED only after the coefficients of its reduced normal form have
> been COMPUTED and shown to be nonzero (or shown to vanish, when a symmetry forces it).
> A picture of two arcs meeting is not a name.**

`branch_switch.py` already states the operational half of this: a physical multiplier at
`+1` is a *necessary local signal* for a same-period bifurcation and is not by itself
evidence for a new periodic family.

### 3.1 Projection fold of an event arc

The object: the set `{G-(y) = 0}` intersected with periodic closure, viewed in the
`(m_1, m_2)` plane. A *projection fold* is a point where this one-dimensional set is
tangent to the `m_2` direction, i.e. where `m_1` is stationary along the arc. Write
`v = m_2` (the direction in which `G-` is stationary) and `w = m_1` (the unfolding
direction). Then at `(v_*, w_*)`:

```
   G-(v_*, w_*)          = 0                     the event holds
   D_v G-(v_*, w_*)      = 0                     stationarity
   D_v^2 G-(v_*, w_*)   != 0                     nondegenerate curvature
   D_w G-(v_*, w_*)     != 0                     transversal unfolding                (18)
```

Lyapunov--Schmidt is trivial here because the event is already scalar once closure is
solved; the remaining content is the Morse lemma with parameters. Taylor-expanding under
(18) — writing the coefficients `A` and `B` rather than `alpha, beta`, which this document
has already spent for the trace invariants:

```
   0 = A (w - w_*) + B (v - v_*)^2 + O(3),
   A = D_w G-  != 0,     B = (1/2) D_v^2 G-  != 0                            (19)
```

so locally `w - w_* = -(B/A)(v - v_*)^2`: two roots on one side of `w_*`, none on the
other. That — and only that — licenses the phrase "the two observed `-1` walls are one
folded arc". `A != 0` is what makes `w` a genuine unfolding parameter; `B != 0` is what
makes the fold quadratic rather than a higher-order degeneracy (a cusp needs `B = 0` and a
second parameter).

The repo's implementation of (18), verified in `scripts/classify_secondary_left_birth.py`
and `julia/verify_secondary_minus_fold.jl`:

```
   closure       <= 1e-7            frozen gate
   |G-|          <= 2e-8            frozen gate
   |dG-/dm_2|    <= 1e-6            stationarity  (D_v G- = 0)
   |dG-/dm_1|    >= 1.0             transversality (D_w G- != 0)
   exactly two independently corrected branch roots, bound to source cells 392 and 393,
   retaining both orientations, with opposite dm_2 signs and consistent secant curvature
```

**Honest gap.** The curvature condition `D_v^2 G- != 0` is evidenced in this repo by a
*finite-scale secant* across two catalog brackets, not by a computed second derivative
with an error bound. `verify_secondary_minus_fold.jl` says so in its own header. A secant
curvature over a `1e-3` mass cell and a nonzero second derivative at a point are different
statements, and the manuscript must not silently upgrade one to the other. The reason the
second derivative is not computed directly is cost, not principle: at `dps=60`, one
BigFloat `corrected_at` is about 5.5 minutes, a five-point mass stencil about 27 minutes,
and the `audit_fold` sweep over `hs = [8e-5, 4e-5, 2e-5, 1e-5]` needs a
`five_point_m2 + five_point_m1` pair at each `h` — roughly 154 minutes of pure finite
differencing. That cost is why three CI runs died. Cost is a legitimate reason to *not
have* evidence; it is never a reason to *claim* it.

### 3.2 The `+1` event: the vanishing pattern decides the name

At a transversal `G+ = 0` crossing, a single reciprocal pair reaches `lambda = +1`. Reduce
the parameterised return map to the one-dimensional critical eigendirection `x` with
parameter `mu`. Lyapunov--Schmidt gives (the coefficients `a`, `b`, `c` here are local to
this subsection and are **not** the trace invariants of Section 2):

```
   0 = c_0(mu) + a mu x + b x^2 + c x^3 + O(4)                                (20)
```

and the *pattern of vanishing coefficients*, not the shape of the numerically traced
curves, decides the name:

| pattern | name | why |
|---|---|---|
| `c_0 != 0` (trivial branch does not persist) | saddle-node / fold of periodic orbits | `0 = c_0 + b x^2` folds |
| `c_0 = 0` (trivial branch persists), `a != 0`, `b != 0` | transcritical | two branches cross and exchange stability |
| `c_0 = 0`, `a != 0`, `b = 0` forced by a `Z_2` symmetry `x -> -x`, `c != 0` | pitchfork (symmetry-breaking) | `0 = a mu x + c x^3` |
| `c_0 = 0`, `a != 0`, `b = 0` *not* forced by symmetry, `c != 0` | degenerate; needs a two-parameter unfolding | an accidental zero is a codimension-two statement |

In this project the parent Li--Li--Liao family persists across the event, so `c_0 = 0`, and
the live question is whether `b` vanishes. `b = 0` may only be asserted when a *symmetry*
forces it — an observed small `b` is a measurement, and a measurement of a coefficient
whose reference scale has not been stated is not evidence of vanishing. The distinction
between "`b` is zero because `Z_2` acts" and "`b` came out small" is the entire difference
between naming a pitchfork and naming nothing.

At a transversal `G- = 0` crossing the generic event is a period doubling: on the *second
iterate* the `Z_2` of the doubling forces the quadratic term to vanish identically, so
`0 = a mu x + c x^3` and the doubling is generically pitchfork-like, sub- or supercritical
by the sign of `c` against `a`. No routine in this repository computes any normal-form
coefficient — `grep -rn "normal.form\|cubic\|lyapunov" src scripts julia` returns nothing —
so `-1 wall` is the earned phrase and `period-doubling bifurcation` is not, until `c`
exists. The same applies to `+1 wall` versus `pitchfork` or `transcritical`.

### 3.3 The mixed organiser

`(G+, G-) = (0,0)` is a transversal intersection of two codimension-one walls
(Section 2.2), hence codimension two: a point in the `(m_1, m_2)` plane, not a curve. A
critical component can change its active mechanism from `+1` to `-1` *only* by passing
through such a point — or by not being one component at all. This is exactly why
`assemble_critical_graph.py` demands event-specific `+1` and `-1` continuation germs
through every retained mixed node rather than nearby-root heuristics, and why relaxing
that demand (Section 7.3) moved `missing_mixed_germs` from 0 to 12.

### 3.4 References for this section

- Yu. A. Kuznetsov, *Elements of Applied Bifurcation Theory* — normal forms, the
  centre-manifold and Lyapunov--Schmidt machinery, and the coefficient conditions that
  make each name legitimate.
- M. Golubitsky and D. G. Schaeffer, *Singularities and Groups in Bifurcation Theory* —
  the singularity-theoretic view: which vanishing patterns are genuinely distinct
  bifurcation problems, and how symmetry forces coefficients to vanish.
- J. Guckenheimer and P. Holmes, *Nonlinear Oscillations, Dynamical Systems, and
  Bifurcations of Vector Fields* — periodic orbits, Poincare maps, and the
  Hamiltonian-specific caveats.
- S.-N. Chow and J. K. Hale, *Methods of Bifurcation Theory* — the Lyapunov--Schmidt
  reduction as a rigorous functional-analytic tool with explicit remainder control.
- J. Moser, *Lectures on Hamiltonian Systems*; M. G. Krein's stability-signature work;
  I. M. Gelfand and V. B. Lidskii on the structure of the symplectic group's stability
  regions; V. A. Yakubovich and V. M. Starzhinskii, *Linear Differential Equations with
  Periodic Coefficients*.

---

## 4. Numerical epistemology

### 4.1 The evidence ladder

From `research/PROTOCOL.md`, and binding:

```
   candidate -> screened -> closure_verified -> variational_verified
             -> independently_reproduced -> release_claim
```

A record moves only forward. A machine-learning proposal is always a `candidate`, never an
orbit. Float64 DOP853 is the screening layer and is stated as such in the module docstring
of `dynamics.py` itself.

### 4.2 Backward error, forward error, and which one this project actually has

*Backward error* asks: is the computed answer the exact answer to a nearby problem?
*Forward error* asks: how far is the computed answer from the true answer to this problem?
They are connected by the condition number:

```
   forward error  <~  condition number  x  backward error                     (21)
```

For a geometric (symplectic) integrator, backward error analysis (Hairer, Lubich and
Wanner, *Geometric Numerical Integration*) gives something strong: the numerical flow is
exponentially close to the exact flow of a *modified* Hamiltonian
`H~ = H + h^p H_p + ...`, so energy error stays bounded over exponentially long times
instead of drifting secularly.

**This project does not have that guarantee.** `dynamics`, `reduced`, `variational`,
`canonical_jacobi` and `critical_manifold` all integrate with SciPy `DOP853`, an explicit
adaptive Runge--Kutta method that is not symplectic. What the project has instead is an
*a posteriori* structural check: the symplectic defect of the computed monodromy in
canonical coordinates,

```
   symplectic_defect = || M^T J M - J ||_inf                                   (22)
```

(`canonical_jacobi.compute_canonical_floquet`, `variational.compute_floquet`). That defect
is not a proof of anything, but it is an honest, computed, orbit-specific error bar on the
multipliers, and it is the right thing to carry beside them. Measured at `rtol=5e-13`
across three frozen census roots: `4.6e-9`, `9.7e-8`, `2.7e-7`.

Conservation defects over one period are the other a posteriori check. Measured on cell 497
at `rtol=1e-12`: `dE = -3.1e-11` on `E = -2.1868`, `dL = -2.4e-14` on `L = 5.07e-4`.

### 4.3 Conditioning must accompany every residual

This section is stated as a measured result rather than as advice, because writing it
required running the measurement and the measurement came back positive. All numbers below
were produced on 2026-08-16 against the frozen census
`research/evidence/V1_HYBRID_CRITICAL_ROOTS_2026-08-15.json`.

The event functions are polynomial invariants of the monodromy:

```
   beta = e_2(M) = ( (tr M)^2 - tr(M^2) ) / 2
```

`tr(M^2)` is assembled from products of entries of `M`, so it is formed at magnitude
`~ ||M||_F^2` and then cancelled down to a quantity of order 1. The unavoidable
double-precision rounding floor on `beta`, and therefore on `G+`, `G-` and `Delta`, is of
order

```
   floor(beta)  ~  u * ||M||_F^2,        u = 2^-53 = 1.11e-16                 (23)
```

**before any integration error at all.** Measured over **all 620** frozen census roots
(`scratchpad/conditioning_census.py`, reduced 8x8 monodromy at `rtol=5e-13, atol=5e-15`):

```
   ||M_8||_F                 min 6.57e+02   median 1.79e+03   max 2.36e+04
   u ||M_8||_F^2             min 4.79e-11   median 3.54e-10   max 6.18e-08

   roots whose ROUNDING FLOOR ALONE exceeds the frozen 2e-8 event gate:  116 / 620
      of those, recorded with estimator = float64:                        25
   roots with less than two decades of headroom (floor > 1e-9):          241 / 620
```

For comparison, the frozen census reports `max |event| = 1.9898e-8` against a gate of
`2e-8` — a margin of `0.51%` — with `165 / 620` roots above `1e-8`
(re-verified from `research/evidence/V1_HYBRID_CRITICAL_ROOTS_2026-08-15.json`;
`max_closure = 1.34e-9` against a `1e-7` closure gate, i.e. the closure gate has 74x of
headroom while the event gate has none).

The rounding floor (23) is only a lower bound on the achievable error. The *achieved*
error was measured directly, on all 620 roots, by evaluating the same event function at
the same stored point along four paths that agree in exact arithmetic
(`scratchpad/stability_census.py`): `rtol = 2e-11` (the `boundary.evaluate` default used
inside the localiser's own Illinois loop), `rtol = 5e-13` (`_precise_evaluate`),
`rtol = 1e-13`, and the `rtol = 5e-13` evaluation of the same orbit rigidly rotated by
`0.7` rad. Path-to-path spread:

```
   all four paths                       median 4.40e-06   spread > 2e-8 : 612 / 620
   the three tight paths (rtol <= 5e-13) median 1.16e-07   spread > 2e-8 : 413 / 620
   rtol=5e-13 vs its own rotated image   median 3.03e-08   spread > 2e-8 : 345 / 620
                                                           spread > 2e-7 : 144 / 620
```

The last line is the cleanest comparison available: identical code, identical tolerance,
identical orbit, differing only by a rigid rotation that cannot change any multiplier.
Resolved by mechanism:

```
   minus_one        n=168   median 1.46e-09   spread > gate:   2 / 168
   plus_one         n=198   median 4.72e-08   spread > gate: 126 / 198
   trace_collision  n=254   median 9.51e-08   spread > gate: 217 / 254
```

**`G-` is resolved at the `2e-8` gate; `G+` and `Delta` are not.** For the `Delta` events
this is structural, not incidental. Differentiating (11), `dDelta = 2a da - 4 db`, so the
error in `Delta` is at least four times the error in `b`; with `u ||M||_F^2` reaching
`6.2e-8` on the stiffest census orbits, `Delta` inherits an error floor of order `2.5e-7`,
more than ten times the gate, from the arithmetic alone. And because
`Delta = (t_1 - t_2)^2`, the inverse map is degenerate exactly where the event lives: an
error `dDelta` in the invariant becomes an error `dDelta / (2|t_1 - t_2|)` in the physically
meaningful separation of the two trace roots, which diverges as the collision is approached.
Resolving a trace collision is intrinsically harder than resolving a `+/-1` crossing, and
holding all three to a single scalar gate hides that.

The rule that follows:

> **A residual is not a measurement until it is quoted with the conditioning of the
> functional that produced it, the chart it was computed in, and the arithmetic used.**
> For this project that means: report `||M||_F` (or `u ||M||_F^2`) beside every event
> value. The root records currently store `alpha`, `beta`, `discriminant`, `closure` and
> `event`, and no norm and no condition estimate at all.

A gate whose value sits below the arithmetic floor of the quantity it gates is not a gate.
It is a filter on noise. The correct response is **not** to move the gate — see
Section 7.4 — but to move the arithmetic, or to narrow the claim.

### 4.4 Why N-version independence beats another RK of the same family

Rank verification lanes by *what they change*:

| lane | changes the step controller | changes the arithmetic | changes the logic |
|---|---|---|---|
| the same DOP853 at a tighter tolerance | yes | no | no |
| a different adaptive RK (RK45, Vern9) in float64 | yes | no | no |
| Julia BigFloat Vern9 at `dps=60` | yes | **yes** | no |
| CAPD interval flow + variational enclosure | yes | yes | **yes** |

Section 4.3 shows that on this problem the dominant error source is *cancellation in the
invariant*, not truncation in the integrator. A second float64 adaptive RK re-samples the
same error model: it changes the step sequence but leaves `u ||M||_F^2` exactly where it
was. That is why "we checked it with a different integrator" is not independence here, and
why the escalation policy in `hybrid_critical` / `merge_hybrid_critical_roots` — send the
shallow cells to Julia BigFloat — is the right shape. `158` of the `620` frozen roots were
escalated for exactly this reason, and Section 6.1's measurements show the escalation
threshold was, if anything, set too generously.

CAPD changes the logic: it stops approximating and starts enclosing. That is a different
kind of statement, and Section 5 is about what it would buy.

---

## 5. Validated numerics as the endgame

Everything above produces *approximations with error estimates*. A theorem needs
*enclosures with proofs*. Three tools, in the order this project would apply them.

### 5.1 Interval Newton / Krawczyk

For `F : R^n -> R^n`, an interval box `X`, a point `x` in `X`, and a nonsingular
preconditioner `A ~ F'(x)^{-1}`, the Krawczyk operator is

```
   K(X) = x - A F(x) + (I - A F'(X)) (X - x)                                  (24)
```

with `F'(X)` an interval enclosure of the Jacobian over the whole box. Then:

- `K(X) subset interior(X)`  =>  `F` has **exactly one** zero in `X`, and Newton from
  anywhere in `X` converges to it;
- `K(X) intersect X = empty` =>  `F` has **no** zero in `X`.

Both conclusions are theorems, not estimates. The cost is that `F` and `F'` must be
rigorously enclosed over `X` — for a periodic-orbit problem that means an interval ODE
solver carrying the variational equations, which is what CAPD provides.

References: R. E. Moore, R. B. Kearfott and M. J. Cloud, *Introduction to Interval
Analysis*; A. Neumaier, *Interval Methods for Systems of Equations*; W. Tucker,
*Validated Numerics: A Short Introduction to Rigorous Computations* (and Tucker's
computer-assisted resolution of Smale's 14th problem as the demonstration that this style
of argument settles genuine open questions).

### 5.2 Topological degree

If `F` is continuous on a bounded open `Omega`, `0 not in F(boundary Omega)`, and
`deg(F, Omega, 0) != 0`, then `F` has a zero in `Omega`. Degree is a homotopy invariant
and is computable from sign information on the boundary alone. It gives existence without
uniqueness and, critically, **without any statement about how many digits are correct**.
This is the formal content of house rule 7.2.

### 5.3 Radii polynomials

For a Newton-like fixed-point operator `T(x) = x - A F(x)` on a Banach space and an
approximate zero `xbar`, find bounds

```
   || T(xbar) - xbar ||                                  <= Y
   sup_{ ||u||,||w|| <= 1 }  || DT(xbar + r u) r w ||     <= Z(r)              (25)
```

If `Y + Z(r) < r` for some `r > 0`, then `T` is a contraction on the closed ball of radius
`r` about `xbar`, so `F` has a unique zero within `r` of `xbar`. `Y` is "how good is the
approximate solution", `Z(r)` is "how much does the derivative vary over the ball", and
`r` is the certified error radius. (Day, Lessard and Mischaikow; van den Berg and Lessard,
for the functional-analytic setting and the Fourier/Chebyshev variants relevant to
periodic orbits.)

### 5.4 What a theorem-grade statement would look like here

Not "the event residual is `1.4e-11`". Something with this shape:

> **Claim (fold, theorem-grade).** There exist an explicit box
> `X = [x1] x [v1] x [v2] x [T] x [m_1] x [m_2] subset R^6` and an interval-Newton
> certificate showing that the system
>
> ```
>    Pi(Phi_T(z(y)) - z(y))  = 0        (gauge-fixed reduced closure, 3 equations)
>    G-(y)                   = 0
>    D_{m_2} G-(y)           = 0
> ```
>
> has **exactly one** solution `y_* in X`, together with rigorous enclosures
> `D^2_{m_2} G-(X) subset [c_1, c_2]` with `0 not in [c_1, c_2]` and
> `D_{m_1} G-(X) subset [d_1, d_2]` with `0 not in [d_1, d_2]`. Consequently the `-1`
> critical set has exactly one nondegenerate `m_1`-projection fold in `X`, and its
> location is enclosed to within `X`.

Note what this shape forces and the current float64 workflow does not: the gauge
directions must be *removed*, not merely small; the nondegeneracy conditions must be
*enclosed over the whole box*, not evaluated at a point; and the uniqueness is proved,
not inferred from the fact that Newton converged to one thing.

**Current state, stated so it is not overread.** `validated/capd/validated_flow.cpp`
encloses the full-period flow, the first variational matrix, and interval `alpha`, `beta`,
`G+`, `G-` and `Delta` around each frozen mixed seed, against a pinned CAPD commit. Its
declared scope is validated-flow scaffolding: it does **not** claim periodic-orbit
existence or organiser existence, because there is no interval-Newton/Krawczyk root
certificate yet. That is the honest gap.

**Prohibition, restated from `research/STRONGEST_HAMMERS_2026-08-15.md` because it is the
easiest rule in the project to break by accident:** wrapping non-rigorous ODE output in
intervals is *not* validated numerics and may never be presented as proof evidence.
An interval around a float64 answer certifies nothing about the true answer.

---

## 6. Attack patterns

This is the operational core. Each pattern is stated as: the invariant, the test, and what
it caught or failed to catch here.

### 6.1 Metamorphic invariants

**Principle.** A transformation `S` that maps a solution of (1) to another solution, with
a known action on the period, gives a *metamorphic relation*: a quantity that must be
identical on the two runs. You do not need to know the true value to test it. The measured
spread across the family of images is the honest resolution of the quantity.

The relations, with their action on the nontrivial multipliers:

| transformation | action | period | nontrivial spectrum |
|---|---|---|---|
| body permutation `sigma in S_3` | `(r,v,m) -> (r_sigma, v_sigma, m_sigma)` | `T` | identical |
| translation | `r_i -> r_i + c` | `T` | identical |
| Galilean boost | `v_i -> v_i + u` | `T` | identical |
| rotation `R in SO(2)` | `(r_i, v_i) -> (R r_i, R v_i)` | `T` | identical |
| planar reflection | `(x,y) -> (x,-y)` | `T` | identical |
| time reversal | `(r, v, t) -> (r, -v, -t)` | `T` | `M -> ~ M^{-1}`; spectrum is reciprocal-closed, so identical |
| Newtonian similarity | `r -> s r`, `t -> s^{3/2} t`, `v -> s^{-1/2} v` | `s^{3/2} T` | identical |
| mass scaling | `m -> k m`, `r -> r`, `t -> k^{-1/2} t` | `k^{-1/2} T` | identical |
| chart change | `M_B = T M_A T^{-1}` | `T` | identical (trace invariants are similarity invariants) |

Under similarity, `E -> s^{-1} E` and `L -> s^{1/2} L`, so those are *covariant*, not
invariant; only the dimensionless spectrum is invariant. Test the right thing. (Mass
scaling is listed for completeness and was not exercised below: it leaves the declared
mass box, so it is a relation for a unit test on a synthetic orbit rather than on a census
root.)

**Measured (`scratchpad/metamorphic.py`, three frozen census roots, `rtol=5e-13`).**
Deviation of the event function from its value on the untransformed orbit:

| transformation | cell 497 `G-` | cell 22 `G+` | cell 17 `Delta` |
|---|---|---|---|
| translation | 0 (exact) | 0 (exact) | 0 (exact) |
| Galilean boost | 0 (exact) | 0 (exact) | 0 (exact) |
| time reversal | 0 (exact) | 0 (exact) | 0 (exact) |
| planar reflection | 0 (exact) | 0 (exact) | 0 (exact) |
| rotation by 0.7 rad | 9.5e-10 | **7.6e-07** | **7.5e-07** |
| similarity `s = 2` | 9.5e-10 | 2.1e-07 | **1.7e-06** |
| similarity `s = 0.5` | 2.0e-09 | 8.6e-08 | **3.1e-06** |
| body permutation `(1 2 3)` | 9.9e-10 | 7.0e-08 | **2.5e-06** |
| composite of all | 2.0e-09 | 2.9e-07 | 1.1e-07 |
| *frozen census recorded value* | `-1.44e-11` | `1.05e-12` | `-6.98e-13` |

Read that table carefully. The first four rows are **worthless as tests in this chart**:
the Li section (7) puts every body on the x-axis and every velocity on the y-axis, so
translation and boost cancel exactly in relative coordinates and reflection coincides with
time reversal. They return bit-identical inputs. A metamorphic test that cannot fail is
not a test — the same lesson as house rule 7.3, arriving from a different direction.

The rows that *do* re-derive the monodromy through different arithmetic — rotation,
similarity, permutation — disagree with the reference by up to `7.6e-07` (cell 22) and
`3.1e-06` (cell 17), that is by **38x** and **155x** the frozen `2e-8` event gate, on
orbits whose recorded event values are `1.05e-12` and `-6.98e-13`.

Scaled to the whole census (Section 4.3): the single rotation test, run at identical
tolerance on all 620 frozen roots, moves the event by more than the `2e-8` gate on
`345 / 620` roots and by more than ten times the gate on `144 / 620` — but only on
`2 / 168` of the `minus_one` roots, against `126 / 198` `plus_one` and `217 / 254`
`trace_collision`.

**What this does and does not falsify.** It does *not* say the roots are in the wrong
place; the localisation drives a noisy functional to its own noise floor, and the root
locations may well be fine. It *does* falsify the statement that `|event| <= 2e-8` is a
property of the root. For `G+` and `Delta` roots it is a property of one arithmetic path.
The frozen census's headline "620/620 localised roots, `max |event| = 1.9898e-8` under a
`2e-8` gate" is therefore chart- and path-dependent for two of the three mechanisms, and
must not be presented as a bound on the physical quantity without stating the chart, the
tolerance and the machine. The `minus_one` mechanism survives this attack cleanly and its
gate compliance may be stated without that caveat.

**Where to add this to the repo.** There is currently no metamorphic test of the Floquet
invariants anywhere in `tests/`. `test_dynamics.py` locks the Newtonian force invariants
(pair sum, zero net force) and `test_variational.py` cross-checks the Jacobian against
finite differences; neither transforms an orbit. The cheapest high-value addition is a
test that asserts a *declared* metamorphic spread on a fixed representative and fails when
it grows.

### 6.2 Sign-vector face consistency

**Principle.** The invariant plane `(G+, G-)` is cut by three walls into faces; each face
carries a sign vector `s = (sgn G+, sgn G-, sgn Delta)`. Crossing one wall transversally
and away from the three vertices must flip **exactly one** component, because the other
two functions are nonzero there and therefore locally sign-constant. Any recorded
neighbourhood whose sign vector is unreachable, or any recorded crossing that flips two
components at once, is either a mislabelled event or an aliased cell containing two roots.

**The trap, and it is a real one.** Enumerated exhaustively (`scratchpad/structure.py`):

| configuration | sign vector | stable? |
|---|---|---|
| both `t_i` in `(-2,2)` | `(+,+,+)` | **yes** |
| `t_1` in, `t_2 > 2` | `(-,+,+)` | no |
| `t_1` in, `t_2 < -2` | `(+,-,+)` | no |
| both `t_i > 2` | `(+,+,+)` | **no** |
| both `t_i < -2` | `(+,+,+)` | **no** |
| `t_1 < -2 < 2 < t_2` | `(-,-,+)` | no |
| complex conjugate pair | `(+,+,-)` | no |

`(+,+,+)` is shared by the stable face and by two unstable ones. **The sign vector is not
a complete stability invariant.** Geometrically (16): in the `(G+,G-)` chart the stable
region is exactly the connected component of `{G+ > 0, G- > 0, Delta > 0}` whose closure
contains the origin — the mixed organiser — and the other two components are separated
from it by the `Delta < 0` interior of the parabola, which is tangent to the two axes at
`(16, 0)` and `(0, 16)`. Witness: along the line `G+ = 2`,
`Delta = (G- - 2)^2/16 - 2 G- + 12` takes the values `+10.06` at `G- = 1`, `-4.00` at
`G- = 10` and `+1.00` at `G- = 30` — so `(2, 1)` (stable: `a = -0.25`, `b = -0.5`,
`t = -1.7111, 1.4611`) and `(2, 30)` (unstable: `a = 7`, `b = 14`, `t = 3, 4`) carry the
same sign vector `(+,+,+)` and lie in different components.

The repo passes this audit: `stability_invariants` and `stability_score` decide stability
from `|t_i| < 2` directly, not from the three signs. Keep it that way. Anything that
reduces the stability decision to "all three events positive" is wrong at exactly the
places where the boundary leaves the stable component.

### 6.3 Spectral flow and index conservation

**Principle.** Let `n_out(y)` be the number of multipliers strictly outside the unit
circle at continuation point `y`. Away from the critical network `n_out` is locally
constant. Crossing a wall transversally changes it by a known amount: `+2` or `-2` at a
transversal `G+ = 0` or `G- = 0` (one reciprocal pair leaves or returns), `+4` or `-4` at
an opposite-Krein `Delta = 0` (a quartet). Therefore:

> Around **any** closed loop in the declared mass box that avoids the critical network and
> avoids collision, the signed total of crossings must be **zero**.

This is a global, integer-valued, arithmetic-free constraint on the critical graph. It
does not care about `2e-8`.

**This project does not currently run it, and it is the natural attack on the graph's
single largest unverified assumption.** `polyline_edges()` in
`scripts/assemble_critical_graph.py` reconstructs "curves" by chaining consecutive `m_1`
slices under `MASS_JUMP = 0.025` and `M1_SLICE_GAP = 0.0015`. Measured directly from the
frozen root set (`scratchpad/check1.py`):

```
   (event_mode, orientation, m_1) slices in the census:   619
   slices containing exactly ONE root:                    618
   slices containing more than one root:                    1   -- (minus_one, U->S, m1 = 0.996)
   distinct m_1 values: 272;  the only successive difference is exactly 0.001
```

So the chaining rule performs branch discrimination at exactly **one** slice out of 619.
Everywhere else "join the nearest root in the next slice" has nothing to choose between.
That is why an adversarial sweep found `edge_count` pinned at 7 for `MASS_JUMP` anywhere in
the admissible window `[0.011097, 0.096141)` recorded at the constant's own definition — a
factor of `8.66` — and why `M1_SLICE_GAP` is documented in that same source as *currently
redundant*: the constants are not selecting a topology, the data has only one candidate.
"Nearby roots really are one continuous critical curve" is an assumption, and
the census as constructed cannot test it. A loop-based index count can, because it
constrains the *number* of walls a path crosses without knowing which polyline they belong
to.

### 6.4 Mutation testing of detectors

**Principle.** For every gate, construct an input that the gate *must* reject, and assert
that it does. A gate with only green tests is decoration. The stronger form: take the real
artifact, corrupt one field, and require the pipeline to notice.

Three detectors in this repo had never rejected anything, and all three were found wrong on
2026-08-16 (Section 7.3). That is a base rate, not an anecdote.

### 6.5 N-version and cross-implementation

Covered in Section 4.4. The operational rule: when adding a verification lane, write down
first *which* of {step controller, arithmetic, logic} it changes. If the answer is "step
controller only", it is a regression test, not independent evidence.

---

## 7. The house rules

Stated as rules, with the concrete evidence from this repository that produced them.

### 7.1 A finite search never proves the absence of a small component

The census samples the mass plane on a uniform grid of step `0.001` in both `m_1` and
`m_2` (verified: 272 distinct `m_1` values whose only successive difference is exactly
`0.001`; every `source_m2_bracket` has width exactly `0.001`). Two critical roots inside
one `0.001` cell are invisible to it *by construction*. A pair of roots separated by
`1e-4`, or a closed loop of diameter `5e-4`, would produce exactly the census we have.

Corollary about constants: `GERM_ATTACH_DISTANCE = 0.008` has a measured admissible window
of only `1.56x` (largest accepted attachment `0.006351`, nearest rejected candidate
`0.009933`) — and the rejected candidate is `minus_one_s_to_u_0`'s start, i.e. the
`secondary_left_birth` blocker. A constant that narrow is a statement about *this sample*,
not about the physics, and widening it to `0.0105` would silently resolve a blocked
endpoint. Any tolerance whose admissible window is less than an order of magnitude wide
must be treated as a fitted parameter and reported as one.

The honest form of a negative result here is: *"no additional component was found at
resolution `h` in region `R` under detector `D`"* — with `h`, `R` and `D` all stated. Never
*"there is no additional component"*.

### 7.2 If a million decimal digits disagree with a topological index, believe the index

Degree, winding number, spectral flow, parity of crossings, and Euler characteristics are
integers computed from *sign data*. They are stable under any perturbation that does not
cross a zero. A high-precision floating-point value is a real number computed through a
long chain of cancellations, and Section 4.3 measured how badly that chain can behave here.
When they disagree, the integer is almost always right and the decimals are telling you
that a cancellation, a chart, a branch cut, or a gauge direction has gone wrong.

Operationally: when a `dps=60` BigFloat event value and a parity/index argument conflict,
open an investigation into the decimals. Do not adjust the index.

### 7.3 A safety check that has never fired is evidence against the check

Three checks in this repository had never rejected anything. All three were examined on
2026-08-16 and all three were wrong.

1. **The self-hashing completeness certificate.** `valid_completeness_certificate()` checked
   only that a record's `sha256_content` matched a digest computed over *that same record*.
   A hand-written two-key dict `{"schema": "...", "passed": true}` plus a hash of itself
   satisfied the release contract's completeness gate. The check proved internal
   self-consistency and *nothing* about whether an active-learning pocket screen and a
   stability-neck raster existed. The certificate already carried a `sources` list with
   `{role, path, sha256}` entries that nothing ever read. Fixed by making those sources
   load-bearing: re-read each declared source, recompute its digest, require both roles,
   re-derive the substantive predicates from the referenced artifacts, and reject path
   escapes. Schema bumped so schema-1 self-sealed records can never be replayed.
   *Lesson: a hash of a document by that document is not tamper evidence. Tamper evidence
   binds a claim to an artifact the claimant did not write.*

2. **The by-name germ exemption.** `valid_germ()` exempted any germ whose `mixed_node` was
   one of the three headline organisers from the canonical-binding and frozen-gate checks.
   Those three contributed germs with no closure, no event value and no canonical binding
   at all — and two of them recorded `stopped_reason = "pseudo-arclength correction failed:
   augmented least-squares failed"` next to `status = "traced"`. The underlying trace had
   **zero** continuation points: the corrector failed on the first step, so those "germs"
   were the two localised seed cells relabelled `G+`/`G-`, and one of them was the sole
   attachment evidence for an edge endpoint at distance `1.8e-9`. Removing the exemption
   moved `missing_mixed_germs` from 0 to 12 and pushed `release_ready` further away.
   *Lesson: any check with a by-name allow-list is not a check. If the headline objects
   cannot pass the bar, the bar is telling you something about the headline objects.*

3. **Truncation recorded as a merge.** The stability-neck raster reported
   `any_vertical_merge = true`, blocking completeness. It was an artifact of the scan
   window: four `m_1` lines carried a single stable interval only because the `-1` `U->S`
   wall that opens the secondary lobe sits at `m_2 = 1.0080934`, *above* the scanned
   `m_2`-max of `1.006`. The merge criterion was being evaluated on an object the raster
   did not contain. Fixed by giving every line one of four explicit verdicts —
   `separated`, `interior_merge`, `truncation_undecidable`, `no_stable_sample` — counting
   only `interior_merge` as a merge, and making truncation *refuse* rather than pass.
   *Lesson: a detector must be able to answer "I cannot tell from this data". A binary
   detector forced to choose will choose, and the choice will look like a result.*

The rule: **treat "this check has never caught anything" as evidence against the check.**
Schedule a mutation test for it (Section 6.4) rather than trusting its silence.

### 7.4 Never loosen a numerical gate to make data pass

The frozen gates are `|event| <= 2e-8`, `closure <= 1e-7`, `residual <= 1e-7`. They may be
made **stricter**, never looser. The mechanism is in the code, not only in the culture:
`scripts/localize_full_critical_network.py` exits with
`"refusing to loosen the 2e-8 event gate"` if `--event-tolerance` exceeds `2e-8`, and
`assemble_critical_graph.py` repeats the gates at its own top with the comment that they
may only ever be made stricter.

Three corollaries, all earned by the measurements of 2026-08-16 reported in Section 4.3:

- When the data will not pass, the legitimate moves are: better arithmetic, a different
  formulation, or a narrower claim. Moving the gate is never one of them.
- When the data passes with a `0.51%` margin — `max |event| = 1.9898e-8` against `2e-8` —
  **distrust the pass**. A margin that thin usually means the localiser was optimising
  against the gate rather than converging to a root, and Section 4.3 shows that for `116`
  of `620` roots the gate is below the arithmetic floor of the quantity being gated.
- A single scalar gate applied to three functionals with different conditioning is itself
  a defect. Measured: a rigid rotation of the orbit at fixed tolerance moves the event by
  more than `2e-8` on `2/168` `minus_one` roots but on `126/198` `plus_one` and `217/254`
  `trace_collision` roots. The remedy consistent with rule 7.4 is a **stricter, per-mode**
  gate stated together with the conditioning, never a looser common one.

### 7.5 Only the assembler may set `release_ready`

`scripts/assemble_critical_graph.py` is the sole writer of `release_ready` in
`research/evidence/V1_CRITICAL_GRAPH.json`. Nothing else may set it, and no human may
hand-edit it. More generally: **nothing may be hand-written into `research/evidence/`.**
Only producing scripts write there. Test fixtures live in `tests/` or a scratch directory
and must be obviously synthetic. `release_ready` is the theorem-grade /
full-critical-set claim (`full_critical_set_release/v1`), not a synonym for
"the bounded AL/neck certificate verified". The assembler still reports that
narrower numerical result separately.

The current state of that file, re-verified for this document: `release_ready = false`,
schema `atlas.v1.critical-graph/3`, `unexplained_nodes = [secondary_left_birth]`,
`missing_mixed_germs = 12`, `unclassified_edge_endpoints = 8`,
`completeness_passed = false`.

### 7.6 Asserting in public that a problem is solved is a human act

Every check above the publish step is a machine check, and machine checks are the right way
to decide whether the numbers hold. Announcing to the world that a named open problem is
solved is a different kind of act. Until 2026-08-16 the release workflow ran on every push
to `main` and would have minted a tag and created a public GitHub Release the moment the
gate went green — with no approval and no second pair of eyes. The only thing preventing it
was that the gate was red, and *the absence of an event is not a safety property*.

Two human-initiated publish paths now remain: a person pushes a `solved-v*` tag, or a
person dispatches the workflow and retypes the exact tag. A push to `main` still validates,
builds and runs `--require-solved`; it simply cannot announce anything.

### 7.7 Never fabricate evidence

No hand-written numbers in `research/evidence/`. No inferred values presented as measured
ones. No "representative" figures assembled from different runs. Where a quantity is
estimated rather than measured, say `estimated` in the artifact. Where a run did not
happen, the artifact says so — `research/CLOSURE_STATUS_2026-08-15.md` records that
several verification lanes had zero executed steps because of a billing failure, and
states plainly that *those runs contain no scientific result*. That is the correct
handling.

### 7.8 Report the conditioning with the residual; name the mechanism only with the mechanism's evidence

The two rules this document adds, both from Section 4.3 and Section 2.5:

- A residual without a condition number, a chart and an arithmetic is not a measurement.
- `Delta = 0` is not "Hamiltonian--Hopf". `lambda = -1` is not "period doubling". A
  mechanism name requires that mechanism's own evidence — a Krein signature, or the
  transversality trio of Section 2.5, or a computed normal-form coefficient.

---

## 8. Scope discipline

### 8.1 What this project does not address

**It does not address the general three-body problem, and no result here should be read as
progress on it.**

There is no closed-form solution in the classical sense, and this is a theorem, not a
failure of effort:

- **Bruns (1887)** showed that the classical algebraic integrals — energy, momentum,
  centre of mass, angular momentum — are essentially the only algebraic first integrals of
  the three-body problem in the classical Cartesian variables. There are no further
  algebraic conservation laws to find.
- **Poincare (1890/1892, *Les Methodes Nouvelles de la Mecanique Celeste*)** showed the
  much stronger statement for the restricted problem: there is no additional uniform
  single-valued analytic first integral depending analytically on a small mass parameter.
  Homoclinic tangles obstruct integrability itself, not merely our ability to write the
  integral down.
- **Sundman (1912)** did produce a *convergent* series solution for the three-body problem
  with nonzero angular momentum (which excludes triple collision), in powers of `t^{1/3}`
  after a regularising change of time; **Wang Qiu-Dong (1991)** extended the construction
  to `n` bodies. These series converge and are numerically useless: the number of terms
  needed for any practical accuracy is astronomically large. "A convergent series exists"
  and "the problem is solved" are different statements, and conflating them is one of the
  standard ways popular accounts of this subject go wrong.

Consequently every result in this project is a **finite numerical statement about a frozen
catalog**, verified to declared tolerances by declared methods.

### 8.2 What v1 actually targets

> The mechanism-resolved **planar linear-Floquet critical graph** on **one
> continuation-connected sheet** of a **frozen catalog** within a **declared mass box**.

Each qualifier is load-bearing:

- **planar** — two-dimensional motion only. Nothing here says anything about
  three-dimensional stability, which requires the out-of-plane multipliers and a larger
  monodromy.
- **linear Floquet** — spectral stability of the linearised return map. This is **not**
  nonlinear stability, not Lyapunov stability, and not KAM stability. A Hamiltonian orbit
  can be spectrally stable and nonlinearly unstable through resonance; the KAM /
  Arnold-diffusion questions are entirely untouched. `research/OPEN_PROBLEM.md` lists
  "linear Floquet language sold as KAM or spatial stability" under *what does not count*.
- **critical graph** — the mechanism-labelled network of `G+ = 0`, `G- = 0` and
  `Delta = 0` arcs, their folds, and their codimension-two vertices, in the mass plane.
  Not a stability plot, not a scatter of S/U labels.
- **one continuation-connected sheet** — of the Li--Li--Liao unequal-mass non-hierarchical
  catalog, free-group word `bABabaBAba`, 135,445 orbits. Family identity is *continuation
  connectivity under a declared chart*, never topology: distinct branches may share a
  free-group conjugacy class (`research/PROTOCOL.md`, `topology.py`).
- **frozen catalog** — a fixed external data set with a fixed hash, not "the three-body
  problem".
- **declared mass box** — `m_1 in [0.8, 1.1]`, `m_2 in [0.7, 1.2]`, `m_3 = 1`
  (`declared_mass_domain` in `research/evidence/V1_CRITICAL_GRAPH.json`). Every negative
  result is a negative result *inside this box at the sampled resolution* (rule 7.1).

### 8.3 The success sentence, and what it costs

From `research/OPEN_PROBLEM.md`, unchanged and not to be softened:

> We show that the Li--Li--Liao catalog is one continuation-connected family, compute and
> independently verify its planar linear-stability critical graph over the declared mass
> domain, classify every qualitative Floquet mechanism and branch connection on that
> graph, and release the evidence required to reproduce every feature.

Until every item in that document's "still required" list closes, the status is **OPEN** —
and per rule 7.6, the transition from OPEN to SOLVED is announced by a person, not by a
green pipeline.

---

## Appendix A. Reproducing the measurements in this document

Environment used: the repository's pinned `uv` (`>=0.12.5,<0.13`) was **not** available
locally (local `uv` is `0.11.28`) and the pin was not modified. All Python numbers above
were produced in a throwaway virtual environment

```
   python3 -m venv venv && venv/bin/pip install numpy scipy pytest ruff
   PYTHONPATH=src venv/bin/python <script>
```

with `numpy 2.5.2`, `scipy 1.18.0`, on `darwin/arm64`. Note that this is a *different*
architecture from the CI runners that produced the frozen census; Section 4.3 and 6.1 are
partly about why that matters.

| script (scratch) | produces |
|---|---|
| `check1.py` | roots-per-`(mode, orientation, m1)` slice histogram (Section 6.3) |
| `verify_doctrine.py` | conservation defects, chart identities, Krein signs (Sections 1, 2) |
| `verify3.py` | three-chart cross-check of `(a,b)` and the events (Section 2.3) |
| `conditioning_census.py` | `||M_8||_F` and the rounding floor for all 620 roots (Section 4.3) |
| `metamorphic.py` | the metamorphic deviation table (Section 6.1) |
| `stability_census.py`, `spread2.py` | four-path event spread for all 620 roots (Sections 4.3, 6.1) |
| `structure.py` | the algebraic identities, vertices, tangency, face table (Sections 2, 6.2) |

These are analysis scripts, not producing scripts: none of them writes to
`research/evidence/`, and per rule 7.5 none of them ever may.

## Appendix B. References

Dynamics and celestial mechanics

- H. Bruns, on the absence of new algebraic integrals of the three-body problem (1887).
- H. Poincare, *Les Methodes Nouvelles de la Mecanique Celeste* (1892--1899).
- K. F. Sundman, on the regularisation and series solution of the three-body problem
  (1912); Q.-D. Wang, the global solution of the `n`-body problem (1991).
- C. L. Siegel and J. K. Moser, *Lectures on Celestial Mechanics*.
- V. I. Arnold, *Mathematical Methods of Classical Mechanics*.
- X. Li, Y. Li and S. Liao, on the periodic orbits of the unequal-mass non-hierarchical
  three-body problem (arXiv:2007.10184; *Sci. China Phys. Mech. Astron.* 64 (2021) 219511)
  — the frozen catalog this project maps.
- T. Kapela and C. Simo, on computer-assisted proofs of stability for choreographies —
  the trace-invariant formulation used by `reduced.py`.

Floquet theory, Krein signature, symplectic spectra

- M. G. Krein, on the stability of linear canonical systems with periodic coefficients.
- I. M. Gelfand and V. B. Lidskii, on the structure of stability regions of linear
  canonical systems.
- J. Moser, *Lectures on Hamiltonian Systems*; and on the stability of periodic orbits of
  Hamiltonian systems under collision of multipliers.
- V. A. Yakubovich and V. M. Starzhinskii, *Linear Differential Equations with Periodic
  Coefficients*.

Bifurcation and normal forms

- Yu. A. Kuznetsov, *Elements of Applied Bifurcation Theory*.
- M. Golubitsky and D. G. Schaeffer, *Singularities and Groups in Bifurcation Theory*.
- J. Guckenheimer and P. Holmes, *Nonlinear Oscillations, Dynamical Systems, and
  Bifurcations of Vector Fields*.
- S.-N. Chow and J. K. Hale, *Methods of Bifurcation Theory*.
- E. J. Doedel and colleagues, on numerical continuation and branch switching (AUTO).

Numerical analysis

- E. Hairer, C. Lubich and G. Wanner, *Geometric Numerical Integration*.
- E. Hairer, S. P. Norsett and G. Wanner, *Solving Ordinary Differential Equations I*
  (DOP853 and its embedded error control).
- N. J. Higham, *Accuracy and Stability of Numerical Algorithms* — backward error,
  conditioning, and why (23) is unavoidable.

Validated numerics

- R. E. Moore, R. B. Kearfott and M. J. Cloud, *Introduction to Interval Analysis*.
- A. Neumaier, *Interval Methods for Systems of Equations*.
- W. Tucker, *Validated Numerics: A Short Introduction to Rigorous Computations*.
- S. Day, J.-P. Lessard and K. Mischaikow, and J. B. van den Berg and J.-P. Lessard, on
  radii polynomials and rigorous computation of invariant objects.
- The CAPD library, for interval ODE and variational enclosures.

Software testing

- T. Y. Chen, S. C. Cheung and S. M. Yiu, and later work by Chen, Kuo, Liu, Poon, Towey,
  Tse and Zhou, on metamorphic testing — the discipline of testing programs that have no
  oracle by testing the relations between their outputs.
