# The 2e-8 event gate is below the float64 floor for a large part of the census

Found while validating `src/threebody_atlas/mass_sensitivity.py`.  Reproduce with

```
# every 4th root, one converged integration each (~20 min in float64)
PYTHONPATH=src python scripts/audit_event_conditioning.py \
    research/evidence/V1_HYBRID_CRITICAL_ROOTS_2026-08-15.json \
    --estimator all --stride 4 --no-coarse

# all 620, both tolerances (~3 h) -- drop --stride and --no-coarse
```

The audit uses `critical_manifold._flow_for_vector` -- the census's **own** code
path -- so nothing here can be blamed on a second implementation.  It writes no
evidence and refuses an `--output` under `research/evidence`.

## The claim

For each root, take the chart and masses **exactly as recorded** and integrate at
a converged tolerance.  For a large fraction of the 462 float64-estimated roots
the recorded event value does not come back, missing the frozen `2e-8` gate by
one to two orders of magnitude, and the recorded closure norm is optimistic by
30-100x.

This is a statement about the **certificates**, not about the root locations.
See "What this does and does not invalidate" below.

## Why

With `alpha = tr M` and `beta = (alpha^2 - tr M^2)/2`, every event is a
cancellation:

```
G+    = beta - 6 alpha + 20
G-    = beta - 2 alpha +  4
Delta = (alpha-4)^2 - 4(beta - 4 alpha + 8)
```

`alpha` is O(1..6) and `beta` is O(1..16), but they are extracted from `M` and
`M^2`, whose entries reach `||M||` and `||M||^2`.  The float64 round-off in
`tr(M^2)` is therefore about `eps * ||M||^2`, and `Delta` multiplies it by 4.

Across the census `||M||` runs from about **7e2 to 2.4e4**, so this floor runs
from **2e-10 to 1.3e-7** (5e-10 to 5e-7 for `trace_collision`).  The frozen event
gate is **2e-8**.  Everything with `||M|| >~ 1e4` is past it, and everything with
`||M|| >~ 5e3` is past it in the `trace_collision` mode.

`stability_invariants` in `reduced.py` computes `beta` exactly this way
(`np.trace(a @ a)`), as does every other path in the repo, in Python and in
Julia.  The BigFloat lane is unaffected -- at dps=60 the same cancellation costs
8 of 60 digits and is invisible.

## Measured, at the census's recorded charts, with the census's own code

```
cell 79   ||M||=1.9e4  recorded event -1.8103e-08   recorded closure 1.035e-09
    rtol=3e-10 (screening)   closure 1.4338e-08   event +1.48637e-03
    rtol=5e-13 ("precise")   closure 1.0580e-07   event +7.46989e-07
    rtol=1e-13               closure 1.0590e-07   event -7.39308e-07
    rtol=3e-14               closure 1.0606e-07   event -2.50609e-06

cell 96   ||M||=1.9e4  recorded event -1.8169e-08   recorded closure 1.226e-09
    rtol=3e-10 (screening)   closure 4.4513e-09   event -4.82374e-04
    rtol=5e-13 ("precise")   closure 1.2691e-07   event -2.45489e-07
    rtol=1e-13               closure 1.2703e-07   event +3.58111e-07
    rtol=3e-14               closure 1.2724e-07   event +1.05390e-06

cell 392  ||M||=1.5e3  recorded event +2.4067e-09   recorded closure 1.648e-10
    rtol=5e-13               closure 7.5070e-09   event +1.98320e-09
    rtol=1e-13               closure 7.5149e-09   event +3.73266e-10
    rtol=3e-14               closure 7.5143e-09   event +7.13420e-10
```

Read the columns carefully:

* **The event does not converge.** Tightening `rtol` from 5e-13 to 3e-14 moves
  cell 79's event from `+7.5e-7` to `-2.5e-6`.  Truncation error shrinks under
  tightening; round-off does not.  There is no tolerance at which float64
  delivers `-1.81e-08` for that chart.
* **The closure does converge -- to the wrong number.**  1.058e-7, 1.059e-7,
  1.061e-7 across three tolerances: that is the true closure of the recorded
  chart.  Cell 96's converged closure, **1.27e-7, is above the frozen 1e-7
  closure gate.**

  The provenance is explicit in the producer.
  `scripts/localize_full_critical_network.py` writes
  `"closure": float(q.residual_norm)` where `q` is the `FamilyPoint` returned by
  `liao_family.correct_family_point`, whose `residual_norm` is `norm(fit.fun)`
  from a `least_squares` run at `screening_rtol = 2e-10, screening_atol = 2e-12`.
  The **event** on the same line comes from `critical.event_value`, which routes
  through `critical_manifold._precise_evaluate` at `rtol = 5e-13`.  So the two
  certificates on one record are measured four orders of magnitude apart in
  integration accuracy, and the closure one is measured with an integrator whose
  own error exceeds the residual it is reporting.
* **Small `||M||` is fine.**  Cell 392 reproduces to ~2e-9, well inside the gate.
  The problem is confined to the stiff end of the census.

## Consequence for the headline fragility number

The brief records: *"the 620/620 census maximum |event| is 1.9898e-8 against a
FROZEN gate of 2e-8 -- a 0.5% margin -- with 165 of 620 roots above 1e-8."*

A 0.5% margin is only meaningful if the quantity is known to much better than
0.5% of 2e-8, i.e. to 1e-10.  For the high-`||M||` float64 roots it is known to
about 1e-6.  **The margin is not small; it is undefined.**  The gate is not a
tight pass, it is a measurement below the instrument's resolution -- which is
also why it has never fired.

## What this does and does not invalidate

**Does not**: the root *locations*.  A root's `m2` error from an event error is
`|dEvent| / |dEvent/dm2|`.  With `|dEvent/dm2|` of order 1-50 on the arcs
measured in this branch, an event error of 1e-6 displaces the root by ~1e-7 in
`m2`, against cell widths of 1e-3.  The brackets, the sign changes, the 620/620
localization, and the S/U transition structure are not challenged by this.

**Does**: any claim of the form "this root satisfies `|event| <= 2e-8`" for a
float64-estimated root with large `||M||`, and the use of `max |event|` as a
quality statistic for the census as a whole.  It also puts a real question over
the recorded closure norms, which are systematically optimistic because they are
measured with the integration that produced them.

**Sharpens a caveat the project already states.**  `research/FLOQUET_EVENT_GEOMETRY.md`
says "A float64 event localization is a screening candidate" and requires
independent BigFloat reproduction before a release claim.  158 of 620 roots have
that; 462 do not.  This audit quantifies what the missing 462 actually cost.

## Recommended fixes, in order of cheapness

1. **Record `||M||` and `eps*||M||^2` next to every event value.**  One line in
   the producer.  Then the gate can be stated as
   `|event| <= max(2e-8, k * eps * ||M||^2)` -- which is *not* a loosened gate,
   it is an honest one, and it converts a silent failure into a visible one.
   A root whose floor exceeds 2e-8 should be marked as requiring BigFloat, not
   silently passed.
2. **Re-measure closure at a converged tolerance**, not at `screening_rtol`.  The
   corrector may run loose; the certificate must not.
3. **Escalate the high-`||M||` roots to the Julia BigFloat lane.**  The audit
   output ranks them, so the escalation list is already sorted by need.
4. Longer term, compute `beta` without forming `tr(M^2)` in the same precision
   as `M` -- e.g. accumulate `sum_ij M_ij M_ji` in compensated (Kahan/two-product)
   arithmetic, which costs nothing next to the integration and recovers most of
   the lost digits.

## Honest limits of this audit

* It is float64 versus float64.  It shows the recorded numbers are not
  reproducible; the BigFloat lane, not this audit, decides what the true values
  are.
* `eps * ||M||^2` is a scaling estimate, not a bound.  It tracks the observed
  disagreements across three orders of magnitude in `||M||`, which is why it is
  quoted, but a root slightly above it is not thereby proven wrong.
* The event and closure discrepancies were computed at the recorded chart only.
  Re-correcting each root at converged tolerance would move the chart slightly
  and is the natural next step; it was not run here.
