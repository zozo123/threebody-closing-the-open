# Why an S/U-label bracket criterion cannot see interior critical curves

Status: structural finding about the v1 sampling pipeline. It explains a Gate B
falsification; it is not itself a discovery claim about the family.

Date: 2026-08-16. Read this before proposing "just refine the grid".

## The one-sentence version

The census bracketed a critical curve only where the **published S/U label**
flipped between adjacent baseline rows. The label is a *thresholded* predicate on
the unstable dimension, not a continuous function, so a critical curve across
which the unstable dimension steps `2 -> 1` leaves both sides labelled `U`,
flips no label, and yields no bracket **at any grid resolution**. Seven such
curves were localized at the frozen gates on 2026-08-16, six of them recorded at
an interior `2 -> 1` crossing and the seventh the same crossing read from the
other side of an organizer. More raster does not help; changing the criterion
does.

## The algebra, in full

For the COM-reduced 8-dimensional monodromy, four multipliers are the
symmetry-forced unit multipliers. The other four form two reciprocal pairs whose
traces `t = lambda + 1/lambda` are the roots of

```
P(t) = t^2 - (alpha - 4) t + (beta - 4 alpha + 8),
alpha = tr M,   beta = ((tr M)^2 - tr(M^2)) / 2.
```

The three codimension-one Floquet events are the zero sets of

```
G_plus       = P(+2)       = beta - 6 alpha + 20            a nontrivial lambda = +1
G_minus      = P(-2)       = beta - 2 alpha +  4            a nontrivial lambda = -1
discriminant = (alpha-4)^2 - 4 (beta - 4 alpha + 8)         the two trace roots collide
```

Each is a **polynomial in `(alpha, beta)`**, hence continuous along any path on
which the periodic-orbit family continues smoothly. That is the whole reason the
sign-topology audit works: a sign change of a continuous function on a path
forces a zero on that path.

The published stability label is a different kind of object:

```
n_unstable(alpha, beta) = # reciprocal pairs off the unit circle, in {0, 1, 2}
label = S  <=>  n_unstable == 0
      = U  <=>  n_unstable >= 1
```

`n_unstable` is integer-valued and `label` collapses `{1, 2}` to a single symbol.
Two facts follow immediately.

**(1) The label criterion is blind to `2 -> 1` and `1 -> 2` crossings.**
Crossing a `G_plus` or `G_minus` zero moves one reciprocal pair on or off the
unit circle, changing `n_unstable` by one. If the *other* pair is off the circle
on both sides, then `n_unstable` goes `2 -> 1` and the label reads `U` on both
sides. `published_label_brackets` compares two symbols that are equal, and emits
nothing. Refining the m2 grid subdivides a `U`-`U` interval into more `U`-`U`
intervals; the criterion returns the empty set on every one of them. This is not
a resolution problem. It is a problem with the function being sampled.

**(2) The label criterion cannot count curves on a cell it does see.**
Even where the label does flip, the census pipeline turns one cell into exactly
one root: `critical_manifold.infer_event_mode` collects every event that changes
sign on the bracket, sorts by endpoint magnitude and **returns one mode**. On a
cell that two events cross, one critical curve is localized and the other is
discarded. This is a second, independent loss, and the comparison artifact keeps
it in its own bucket (`shadowed_by_single_mode_selection`) rather than folding it
into the first. Whether it actually bites on the published grid is a measurement,
not a deduction — the answer depends on whether the two crossings land in the
same `0.001`-wide cell — and on the ten slices measured below it does not: every
sampled cell carries exactly one crossing event, consistent with the ledger's
existing claim about all 620. Loss (2) is therefore a real property of the
pipeline that has not yet been observed to fire. It is repaired anyway, because
the cost of repairing it is one loop and the cost of discovering it the hard way
is another falsified gate.

## The evidence, on the real artifacts

`scripts/audit_sign_topology.py` reported seven critical curves absent from the
committed seven-polyline graph, each localized by
`critical_manifold.localize_critical_point` at the unchanged gates
`|event| <= 2e-8` and `closure <= 1e-7`
(`research/evidence/V1_SIGN_TOPOLOGY_AUDIT_2026-08-16.json`,
`V1_SIGN_TOPOLOGY_CROSSING_2026-08-16.json`). Their recorded endpoint unstable
dimensions are `2 -> 1` at six of the seven, and `1 -> 0` at
`(0.9295, 0.8860337144)`.

That seventh point is worth a paragraph, because it is the one place where a
summary of "all seven step `2 -> 1`" would mislead. Three of the seven sit at
`m1 = 0.9295` or `0.9305`, and the published `m1` grid has spacing `0.001`:
those probe lines are not baseline slices at all, so no criterion running on
published rows meets them at their own `m1`. They straddle the codimension-two
organizer near `(0.92925, 0.88538)` where the `plus_one` and `minus_one` curves
cross, and crossing it exchanges which of the two is the stability boundary and
which is interior. At `m1 = 0.929` the `minus_one` crossing is the interior one,
on the `U`/`U` cell `(0.884, 0.885)`, while `plus_one` owns the label-flipping
cell `(0.885, 0.886)` — census cell 258. Half a thousandth later, at
`m1 = 0.9295`, the roles have swapped and `plus_one` is the `2 -> 1` interior
crossing. Same `X`, read from two sides. Which curve the census kept and which
it dropped therefore flips from slice to slice across the organizer, and that —
not a mislocalization — is why cell 258 is filed as an unclassifiable edge
endpoint.

The cleanest instance is on the published grid itself, and it is what
`tests/test_bracket_criteria.py` runs against
`tests/fixtures/baseline_m1_0925_interior_minus_one.txt`, a verbatim excerpt of
the frozen Li--Li--Liao table at `m1 = 0.925`:

| m2    | published | n_unstable | sign G_minus | sign G_plus |
|-------|-----------|------------|--------------|-------------|
| 0.873 | U         | 2          | -            | -           |
| 0.874 | U         | 2          | -            | -           |
| 0.875 | U         | 1          | **+**        | -           |
| ...   | U         | 1          | +            | -           |
| 0.880 | U         | 1          | +            | -           |
| 0.881 | S         | 0          | +            | **+**       |

Two critical curves cross this window. The `G_plus` curve at `(0.880, 0.881)` is
also the stability boundary, so the label flips `U -> S` and the census has that
cell. The `G_minus` curve at `(0.874, 0.875)` takes the unstable dimension
`2 -> 1`; both rows are published `U`; the census has nothing.
`localize_critical_point`, fed that `U`/`U` pair with `event_mode="minus_one"`,
returns a root at `m2 = 0.87406129...` with an event and a closure both far
inside the frozen gates. It is a real critical point of exactly the kind the
census catalogues, and the criterion that built the census could never have
reached it.

(No event magnitude is quoted here on purpose. float64 events are not
reproducible across machines in this repository: the shipped sign-topology audit
records `+3.7e-10` for this very root, and re-localizing it while writing this
note returned `-8.9e-10` — opposite sign, different magnitude, both two orders
inside the `2e-8` gate, and the located `m2` agreeing to about `3e-12`. The gate
inequality and the discrete structure are the claims; the magnitude is not.)

## The repair

`src/threebody_atlas/bracket_criteria.py` names both criteria and keeps them
separate:

- `published_label_brackets` — the historical rule, preserved bit-for-bit. Run
  under `scripts/extract_mass_slice_brackets.py --criterion published-label`
  (the default) it still reproduces the 620-cell census byte-for-byte, which is
  required: the census and every artifact derived from it are frozen historical
  records and must remain reproducible.
- `event_sign_brackets` — brackets on a sign change of `G_plus`, `G_minus` or
  the discriminant between adjacent published rows, emitting **one bracket per
  crossing event per cell**. This repairs both losses at once: it sees crossings
  that move no label, and it does not collapse a two-event cell to one root.

The cost is honest: the label lives in a published column, while an event
function needs one variational Newton correction plus one monodromy per row.
The event criterion is a few seconds per baseline row against microseconds. That
is what not being blind costs.

## Measured difference on the real baseline

`research/evidence/V1_BRACKET_CRITERION_COMPARISON_2026-08-16.json`, produced by
`scripts/audit_bracket_criteria.py`. Ten published m1 slices, every published row
with `0.80 <= m2 <= 1.00` (201 rows per slice, 2010 rows, all of them closing
under `1e-7`), both criteria run over exactly the same rows:

| m1 | published-label | event-sign | label-invisible | certified at frozen gates |
|-------|---|---|---|---|
| 0.900 | 2 | 2 | 0 | — |
| 0.920 | 2 | 3 | 1 | `minus_one` at m2 = 0.85950367 |
| 0.925 | 2 | 3 | 1 | `minus_one` at m2 = 0.87406129 |
| 0.929 | 2 | 3 | 1 | `minus_one` at m2 = 0.88475175 |
| 0.930 | 2 | 3 | 1 | `plus_one` at m2 = 0.88627431 |
| 0.931 | 2 | 3 | 1 | `plus_one` at m2 = 0.88746072 |
| 0.940 | 2 | 3 | 1 | `plus_one` at m2 = 0.89781300 |
| 0.970 | 1 | 2 | 1 | `plus_one` at m2 = 0.92938655 |
| 1.000 | 1 | 2 | 1 | `minus_one` at m2 = 0.93813745 |
| 1.040 | 0 | 1 | 1 | `minus_one` at m2 = 0.86021152 |
| **total** | **16** | **25** | **9** | **9 of 9 passed, 0 missed** |

Every one of the nine sits on a `U`/`U` cell with the unstable dimension stepping
`2 -> 1`, so every one is unreachable for reason (1): the published label never
flipped and no bracket was ever emitted. All nine were localized by the
repository's own `localize_critical_point` under `|event| <= 2e-8` and
`closure <= 1e-7`, both unchanged.

Four supporting counts, all of which came out clean and would have been reported
had they not:

- **0** label brackets with no event sign change — the event criterion contains
  the label criterion here, as it must;
- **0** cells carrying more than one crossing event, so on this sample loss (2)
  never bites and the ledger's "all 620 cells exhibit exactly one
  endpoint-sign-changing event" survives;
- **0** rows failing the closure gate;
- **0** rows where the published S/U column disagrees with the label our own
  recomputed invariants imply.

Cross-check against the sign-topology audit: **7 of 7** already-certified curves
recovered. The four at published m1 values matched at the same m1 and the same
m2 to about `1e-9`. The three at `m1 = 0.9295 / 0.9305` are not on the published
grid at all and matched on adjacent slices, as they must.

Two of the nine are **not** among the seven — `(0.920, 0.85950367)` and
`(1.040, 0.86021152)`, both `minus_one`, both certified. The audit found what its
scan lines crossed; the criterion finds what the published rows contain. The gap
between "seven" and "nine on ten slices" is a reason to expect more, not a count.

One honest caveat on the certifications. The measured events span
`6e-10 .. 1.9e-8`, and the largest, `(0.930, 0.88627431)` at `1.94e-8`, sits just
inside the `2e-8` gate. float64 event magnitudes are not reproducible across
machines here — the same organizer chart has been measured spanning a factor of
3.4 on three boxes — so that particular point should be expected to *miss* the
gate on some machines. What is stable is the discrete structure: a `U`/`U` cell
with `n_unstable` stepping `2 -> 1` and an event sign change inside it. A root
whose certification is marginal in float64 is a candidate for the Julia BigFloat
path, not a reason to widen anything.

Finally, an observation offered as an observation. Reading the nine down the
table, the mechanism switches `minus_one -> plus_one` between `m1 = 0.929` and
`0.930` — across the organizer near `(0.92925, 0.88538)` where the two curves
cross — and switches back between `0.970` and `1.000`. That is what one would
see if a single interior branch threads the region and exchanges mechanism at
each organizer it passes. Nothing here establishes that: these are nine
independent localizations on nine slices, with no continuation between them.
Connecting them is exactly the pseudo-arclength work
`critical_manifold.trace_augmented_critical` exists for, and it has not been run.

## What the new criterion still cannot see

Do not read `event_sign_brackets` as complete. A sign change is *sufficient* for
a zero, never necessary. Three residual blind spots, all of them at the
sampling density rather than in the predicate:

- **Tangency.** A curve that touches an event's zero set without crossing it
  (even-order contact, e.g. a fold of the critical curve in the m2 direction)
  changes no sign anywhere. This criterion will not find it.
- **Two crossings in one cell.** Two zeros of the same event between adjacent
  published rows return the sign to where it started. A finer m2 grid *does*
  help here — unlike the label blindness, this one is a resolution problem.
- **Rows that fail closure.** A row whose periodic closure cannot be certified
  under `1e-7` bounds no bracket, which silently widens its neighbouring
  interval. The comparison artifact reports the count so the widening is visible
  rather than assumed to be zero.

The distinction that matters: the label criterion's blindness is invariant under
refinement, so it cannot be measured away. These three shrink with sampling and
can be bounded.

## What this does and does not settle

It settles the *cause*: the census's blind spot is in the sampling criterion, not
in the raster, not in the localizer, and not in the continuation. It supplies a
criterion without that blind spot.

It does not re-run the census, does not modify `V1_CRITICAL_GRAPH.json`, and does
not close Gate B. Gate B needs a critical graph whose completeness argument does
not rest on a criterion we now know to be incomplete; producing one is downstream
work. What has changed is that the limitation is explicit, tested, and no longer
has to be rediscovered.

One consequence deserves flagging before anyone reads the freeze as satisfied.
`OPEN_PROBLEM.md` item 1 is phrased over "all 620 published S/U cells". That
population is defined by the criterion this note is about, so discharging item 1
to the letter — every one of the 620 localized and assigned to a
mechanism-specific polyline — would still say nothing about a critical curve
interior to the unstable region, because such a curve is not one of the 620 and
never could have been. The frozen wording is not wrong; it is simply narrower
than the completeness it is being asked to certify, and the sign-topology audit
is the instrument that measures the gap between the two.

## Cross-references

- `research/FLOQUET_EVENT_GEOMETRY.md` — the trace-invariant algebra the event
  functions come from.
- `scripts/audit_sign_topology.py` — the audit that falsified Gate B, and the
  sign-vector completeness test in general.
- `tests/test_bracket_criteria.py` — the blind spot as a strict-xfail test: the
  published-label criterion is *required* to fail it.
