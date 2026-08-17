# Sign-topology detector calibration, 2026-08-17

Twelve audit runs: two scan lines x six `--max-gap` settings, everything else fixed
(`--m2-range 0.70,1.20 --max-probes 900 --probe-budget 1500 --refine-steps 7`, roots
`research/evidence/V1_COMBINED_CRITICAL_ROOTS_2026-08-17.json`).

`m1 = 0.889` is the test line: a `minus_one` critical curve sits near m2 0.741.
`m1 = 0.900` is the CONTROL: a committed edge provably passes through it, so a
missing-curve report there would be a false positive.

| max-gap | m1 0.889 | m1 0.900 (control) | converged |
|---------|----------|--------------------|-----------|
| 0.070   | clean    | clean              | 9/10, 8/13   |
| 0.040   | clean    | clean              | 14/16, 15/17 |
| 0.020   | **missing** m2 [0.736312, 0.754467] | clean | 26/27, 28/29 |
| 0.010   | **missing** m2 [0.739105, 0.748881] | clean | 52/52 |
| 0.005   | **missing** m2 [0.739105, 0.743993] | clean | 98/98 |

## What this establishes

**Detection threshold between 0.040 and 0.020.** Every committed audit ran at the
0.070 default, so every one was blind by a factor of two to four. The bracket tightens
monotonically with resolution, converging on m2 ~ 0.7415, which agrees with
extrapolating component 0's local secant (slope 4.187) from its tip at
(0.892, 0.7530796) to m1 0.889: m2 ~ 0.7405.

**No false-positive mode.** The control line stays clean at all six settings.

**Probe failure is a function of spacing, not physics.** Convergence goes 9/10 at
max-gap 0.070 to 98/98 with ZERO failures at 0.005, where the converged m2 hull is
0.7049..1.1950 -- essentially the whole declared span, including 20 converged probes
below m2 0.80. An earlier limitation in research/DISCOVERY_RELEASE.json inferred from
sparse-probe failures that the orbit family does not exist below m2 ~ 0.80. That was
wrong and has been retracted: each probe is seeded from its neighbours, so sparse
probing starves the solve.

## Why it is committed

`sign_topology_report` in scripts/assemble_critical_graph.py now refuses an audit that
cannot demonstrate coverage. That guard needs a measured detection threshold to be
more than an arbitrary constant, and these twelve runs are that measurement.
