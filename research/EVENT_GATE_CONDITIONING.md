# The 2e-8 event gate is below the float64 floor for a large part of the census

Found while validating `src/threebody_atlas/mass_sensitivity.py`.  Reproduce with

```
# seeded random 160 of 620, one converged integration each (~35 min in float64)
PYTHONPATH=src python scripts/audit_event_conditioning.py \
    research/evidence/V1_HYBRID_CRITICAL_ROOTS_2026-08-15.json \
    --estimator all --sample 160 --no-coarse

# the decisive round-off probe on a handful of roots (5x the cost per root)
PYTHONPATH=src python scripts/audit_event_conditioning.py \
    research/evidence/V1_HYBRID_CRITICAL_ROOTS_2026-08-15.json \
    --estimator all --sample 12 --no-coarse --jitter

# all 620, both tolerances (~3 h) -- drop --sample and --no-coarse
```

The audit uses `critical_manifold._flow_for_vector` -- the census's **own** code
path -- so nothing here can be blamed on a second implementation.  It writes no
evidence and refuses an `--output` under `research/evidence`.

## The claim

Take each root's chart and masses **exactly as recorded** and integrate at a
converged tolerance.  On a seeded random sample of 160 of the 620 roots:

* **61%** of recorded event values do not come back within the frozen `2e-8`
  gate, missing it by up to `2.19e-6` -- 110x.
* **100%** of recorded closure norms are optimistic by more than 10x (median
  48.6x), and **4 roots** have a converged closure above the frozen `1e-7`
  closure gate.
* The failure rate rises monotonically with `||M||`, from 37% below `2e3` to
  100% above `5e3`, exactly where `eps * ||M||^2` crosses the gate.
* `minus_one` is entirely clean (0 of 40).  The damage is in `trace_collision`
  (91%) and `plus_one` (71%).

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

## The decisive test: same everything, only the opening step changes

Tolerance ladders can always be argued about.  This one cannot.  Fix the chart,
the masses, the method (DOP853), the tolerance (`rtol = 1e-13`), the machine and
the build, and change **only** `first_step` -- a parameter that cannot affect a
quantity the tolerance has actually resolved, because the adaptive controller
settles into its own step sequence within a few steps either way.  Any spread it
produces is round-off by construction.  Run with `--jitter`:

```
                                   spread over first_step in {auto,1e-4,3e-4,1e-3,3e-3}
cell  79  ||M||=1.9e4  trace_collision   1.347e-06     67x the 2e-8 gate
cell  96  ||M||=1.9e4  plus_one          5.028e-07     25x
cell  36  ||M||=2.3e4  plus_one          3.128e-07     16x
cell 392  ||M||=1.5e3  minus_one         2.335e-09     inside the gate
cell 466  ||M||=1.0e3  minus_one         4.496e-09     inside the gate
```

Cell 79's event is recorded as `-1.8103e-08`.  Depending on nothing but the
integrator's opening step, the same code on the same machine at the same
tolerance returns anywhere from `-7.4e-07` to `-2.1e-06`.

The spread tracks `eps * ||M||^2` (times 4 for `trace_collision`) across an order
of magnitude in `||M||`, exactly as the cancellation analysis predicts, and it
disappears below `||M|| ~ 2e3`.  This also disposes of the "your build differs
from theirs" objection: the same build disagrees with itself by 67x the gate.

## Population statistics

Seeded random sample, **160 of the 620** localized roots
(`--sample 160 --seed 20260816`), one converged integration each at
`rtol = 1e-13`.  Mode mix 55 / 40 / 65 (`plus_one` / `minus_one` /
`trace_collision`), matching the census's 198 / 168 / 254.

| finding | count | rate | extrapolated to 620 |
| --- | --- | --- | --- |
| recorded event not reproducible within the 2e-8 gate | 98 | **61%** | ~380 |
| round-off floor `eps*\|\|M\|\|^2` alone exceeds the 2e-8 gate | 37 | 23% | ~143 |
| recomputed closure exceeds the frozen 1e-7 gate | 4 | 2.5% | ~15 |
| recorded closure optimistic by more than 10x | 160 | **100%** | 620 |

Worst event discrepancy `2.19e-6` (110x the gate); worst recomputed closure
`1.543e-7`; median closure optimism factor **48.6x**.

### Dose-response in `||M||` -- the reason this is a mechanism, not a fishing trip

| `\|\|M\|\|` | roots | event not reproducible | median discrepancy |
| --- | --- | --- | --- |
| < 2e3 | 89 | 33 (37%) | 1.1e-8 |
| 2e3 – 5e3 | 26 | 20 (77%) | 1.1e-7 |
| 5e3 – 1e4 | 11 | 11 (100%) | 1.3e-7 |
| >= 1e4 | 34 | 34 (100%) | 5.2e-7 |

Monotone, saturating exactly where `eps*||M||^2` crosses `2e-8`
(`||M|| ~ 9.5e3`, or `~4.7e3` in `trace_collision`).

### By event mode -- `minus_one` is clean

| mode | roots | event not reproducible |
| --- | --- | --- |
| `minus_one` | 40 | **0 (0%)** |
| `plus_one` | 55 | 39 (71%) |
| `trace_collision` | 65 | 59 (91%) |

This matters for what is currently being verified.  The secondary `G- = 0` fold
that three CI runs died on lives entirely in `minus_one`, the one mode where not
a single sampled root failed to reproduce.  The float64 corroboration of the fold
in `research/MASS_SENSITIVITY_PORT_PLAN.md` §3 is therefore on solid ground, and
so is `dG-/dm2` as a Newton slope.  The damage is concentrated in
`trace_collision` (whose `Delta` carries the extra factor of 4) and `plus_one`.

### By estimator

| estimator | roots | event not reproducible | floor > gate |
| --- | --- | --- | --- |
| `float64` | 125 | 64 (51%) | 15 (12%) |
| `julia_bigfloat` | 35 | 34 (97%) | 22 (63%) |

The `julia_bigfloat` row is the cleanest statement in the whole audit: those
recorded values are *correct*, computed at dps=60, and float64 cannot reproduce
34 of 35 of them to within the gate that float64-estimated roots are asserted to
meet.  If float64 cannot re-derive a known-good value to 2e-8, it cannot certify
an unknown one to 2e-8 either.

**Sampling caveat, stated because it bit this audit.**  Cell ids advance in
steps of about 4 per `m1` slice, so an earlier `--stride 4` run aliased onto one
track: 117 `plus_one`, 38 `minus_one`, and **zero** `trace_collision`, despite
`trace_collision` being 254 of the 620 -- and `trace_collision` is the worst
mode.  The strided run therefore understated the problem (45% vs the unbiased
61%).  `--sample` exists because of that and should be preferred.

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
`|dEvent| / |dEvent/dm2|`, and `dEvent/dm2` is exactly what
`mass_sensitivity.py` computes.  Measured at the worst-conditioned cells:

```
cell  79  ||M||=1.9e4  dEvent/dm2 = -441.10   1e-6 event error -> 2.3e-09 in m2
cell  96  ||M||=1.9e4  dEvent/dm2 = +117.80   1e-6 event error -> 8.5e-09 in m2
cell  36  ||M||=2.3e4  dEvent/dm2 = +250.30   1e-6 event error -> 4.0e-09 in m2
cell 112  ||M||=1.5e4  dEvent/dm2 = +101.59   1e-6 event error -> 9.8e-09 in m2
cell 158  ||M||=7.7e3  dEvent/dm2 = +111.19   1e-6 event error -> 9.0e-09 in m2
cell 192  ||M||=5.2e3  dEvent/dm2 = +133.42   1e-6 event error -> 7.5e-09 in m2
```

against cell widths of `1e-3`: five to six orders of margin.  The stiff cells
are, if anything, the *best* conditioned in the `m2 -> event` direction -- the
large `||M||` that wrecks the certificate also makes the event steep.  The
brackets, the sign changes, the 620/620 localization, and the S/U transition
structure are not challenged by this.

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

**None of these loosens the 2e-8 event gate or the 1e-7 closure gate.  The gates
are correct; what is wrong is asserting them from a measurement that cannot
resolve them.  The fix is always to strengthen the measurement or to refuse the
assertion -- never to widen the gate.**

1. **Record `||M||` and `eps*||M||^2` beside every event value.**  One line in
   the producer.  A root whose floor exceeds 2e-8 has not passed the gate, it has
   failed to be *measured* against the gate; mark it `precision_insufficient`
   and route it to BigFloat.  That is a stricter regime than today, where such a
   root is silently recorded as passing.  Explicitly: do **not** turn this into
   `|event| <= max(2e-8, k*eps*||M||^2)`.  That would be exactly the loosening
   this document exists to prevent.
2. **Re-measure closure at a converged tolerance**, not at `screening_rtol`.  The
   corrector may run loose; the certificate must not.  Expect some roots to move
   from "1e-9, comfortably inside 1e-7" to "1.3e-7, outside 1e-7" -- and those
   must then be treated as failures, not re-gated.
3. **Escalate the high-`||M||` roots to the Julia BigFloat lane.**  The audit
   output ranks them, so the escalation list is already sorted by need.
4. Longer term, compute `beta` without forming `tr(M^2)` in working precision --
   e.g. accumulate `sum_ij M_ij M_ji` with a two-product/compensated sum, which
   costs nothing next to the integration and recovers most of the lost digits.
   This raises the measurement to meet the gate, which is the right direction.

## The obvious objection, and the answer

*"The recorded event came from `_precise_evaluate` at `rtol = 5e-13`, at that
same chart, with that same code.  Your `rtol = 5e-13` row should therefore
reproduce it bit for bit.  It doesn't, so your setup differs from theirs."*

Correct that it doesn't reproduce, and I cannot fully account for the
provenance difference: this run uses numpy 2.5.2 / scipy 1.18.0, and some roots
went through the hybrid/augmented continuation whose event is evaluated at
`screening_rtol = 3e-10`, not `5e-13`.  Either could explain a *different*
answer.

But the objection does not touch the load-bearing observation, which needs no
provenance at all: **the value does not converge under tolerance refinement.**
Cell 79 goes `+7.5e-7 -> -7.4e-7 -> -2.5e-6` as `rtol` goes `5e-13 -> 1e-13 ->
3e-14`.  Truncation error shrinks monotonically under refinement; round-off does
not.  A quantity whose answer moves by 3e-6 when you *improve* the integration
is not determined to 2e-8 by that integration, whoever runs it.  Indeed, the
fact that a different scipy build gives a different answer at the same tolerance
is the finding restated, not a rebuttal of it.

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
