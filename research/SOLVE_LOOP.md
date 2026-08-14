# ATLAS closed-loop open-problem solver

This repository is organized around a falsifiable scientific loop. The loop is complete only when every stage emits machine-readable evidence and later stages are allowed to reject earlier conclusions.

## Stage 0 — freeze the question

Primary v1 question:

> Determine the continuation-connected family decomposition of the Li--Li--Liao unequal-mass catalog and, for each connected family, compute the connected linear-stability critical manifold in mass space, classify its Floquet mechanisms and branch connections, and independently verify the qualitative critical events.

The question is frozen before large computation so the project cannot redefine success after seeing the data.

## Stage 1 — reproduce prior results

- hash the exact upstream data blob;
- reconstruct all initial conditions;
- recompute closure/conservation diagnostics;
- reproduce a statistically and dynamically representative set of published S/U labels;
- expand to the full catalog as compute permits.

Failure here invalidates all discovery work.

## Stage 2 — audit family identity

- recompute energy, angular momentum and scale-invariant diagnostics for every row;
- search for multimodality/discontinuities and duplicate mass triples;
- use those diagnostics only to propose possible branch partitions;
- test each proposed partition by bidirectional continuation connectivity;
- define families by continuation paths, never by clustering alone.

Outcome can confirm one family, multiple families, or unresolved connectivity.

## Stage 3 — discover the critical set

- extract every published S/U transition bracket across mass slices;
- Newton-correct both sides onto the periodic family;
- refine the zero of the Floquet stability score;
- connect neighboring refined zeros with secant/pseudo-arclength continuation;
- adapt step size to curvature and solver conditioning;
- detect potential folds, endpoints, junctions and topology changes.

Float64 is discovery/screening only.

## Stage 4 — canonical stability audit

For every representative and every candidate qualitative event:

- transform to canonical Jacobi coordinates;
- integrate the canonical monodromy;
- measure closure, reciprocal pairing and symplectic defect;
- track individual multiplier branches continuously;
- compute Krein information on the stable side where applicable;
- label +1, -1 and Hamiltonian-Hopf/Krein events only when the canonical spectrum supports the label.

An unexplained loss of symplecticity blocks the claim.

## Stage 5 — independent arbitrary-precision verification

- correct the critical orbit in Julia BigFloat;
- integrate state and tangent equations using adaptive high-order integration;
- repeat at increasing precision/tighter tolerances;
- require converged critical parameters and closure;
- reproduce qualitative event types with a numerically independent configuration;
- archive exact decimal inputs, solver versions and monodromy evidence.

Disagreement sends the point back to Stage 3 or 4; it is never averaged away.

## Stage 6 — adversarial search

Try to falsify the emerging atlas:

- seed continuation from both sides of each alleged family boundary;
- run smaller/larger arclength steps;
- perturb chart variables and restart Newton correction;
- search for missed stable pockets between published slices;
- use AI/active learning to target regions of high classifier disagreement, high curvature and near-critical multipliers;
- compare independent coordinate formulations;
- search the literature again for prior publication of the same structure.

AI proposes expensive tests; it does not decide truth.

## Stage 7 — freeze a release candidate

A release manifest contains:

- git commit and dependency lock/versions;
- upstream hashes;
- connected-family graph;
- critical-curve datasets and uncertainty estimates;
- high-precision orbit/monodromy records;
- rejected/ambiguous events;
- figure/table generation inputs;
- known limitations.

No result can enter the paper without a release-claim record.

## Stage 8 — generate and attack the paper

- generate tables/figures directly from the frozen manifest;
- run claim-lint checks against evidence status;
- rebuild the manuscript from a clean checkout;
- rerun representative numerical points from scratch;
- explicitly compare the contribution against Li--Li--Liao 2021 and later work;
- state exactly what is numerical evidence versus theorem/proof.

## Stage 9 — publishable success

The v1 open problem is considered solved only if the final evidence supports a new statement about continuation-connected family decomposition and/or the connected stability critical manifold and its mechanisms that was not provided in prior literature.

A refined isolated crossing, an ML prediction, or a denser stability plot does not satisfy this criterion.
