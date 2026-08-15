# ATLAS scientific protocol v1

## Research question

Systematically map connected families of linearly stable periodic solutions of the planar Newtonian general three-body problem, with emphasis on non-hierarchical unequal-mass systems. The atlas must resolve continuation geometry, linear stability, topology and bifurcation structure rather than merely enumerate isolated periodic initial conditions.

## Baseline to reproduce

The first external validation target is Li, Li & Liao, arXiv:2007.10184 / Science China Physics, Mechanics & Astronomy 64 (2021) 219511. They report one non-hierarchical unequal-mass family containing 135,445 periodic orbits, of which 13,315 are linearly stable. Their public repository contains supplementary data.

A v1 paper must clearly separate:

1. **External reported facts**: numbers/results quoted from prior literature.
2. **ATLAS reproductions**: quantities recomputed independently from published initial conditions.
3. **ATLAS discoveries**: continuation points, stability boundaries, bifurcations or families not contained in the frozen baseline dataset.

## Evidence ladder

A record progresses only forward through these states:

`candidate -> screened -> closure_verified -> variational_verified -> independently_reproduced -> release_claim`

A machine-learning prediction is always a `candidate`, never an orbit.

### Screening

Float64 DOP853 integration. Used for throughput and rejection only. Store closure residual, energy/angular-momentum defects and, when requested, a full-coordinate variational spectrum.

### Closure verification

Re-integrate from decimal-string initial data with independent arbitrary-precision arithmetic. Require explicit precision and step-refinement convergence metadata. Thresholds are experiment-specific and frozen in the release manifest.

### Variational verification

Integrate the tangent equations independently at higher precision. Store the monodromy matrix, Floquet multipliers, reciprocal/conjugate pairing diagnostics, trivial-mode diagnostics and symplectic defect in canonical coordinates. Stability classifications near unit-circle crossings are `numerically_ambiguous` until precision escalation resolves them.

### Independent reproduction

A publishable new orbit/family point must be reproduced by a numerically independent configuration (different precision/integration strategy and, for critical claims, preferably a second implementation).

## Family identity

Topology is not family identity. A family is a connected continuation object under a declared continuation chart. Store branch ID, continuation coordinates, pseudo-arclength, predictor/corrector history and topology signature. Distinct branches may share a free-group conjugacy class.

## Topology

Store raw symbolic observations separately from derived invariants. For F2 words: free-reduce, cyclically reduce, then canonicalize cyclic conjugacy. Never silently quotient time reversal, word inversion, spatial orientation or body-label permutation.

Syzygy extraction and braid classification must record tolerances and algorithm versions.

## Bifurcation evidence

A claimed bifurcation requires a bracket of verified continuation points plus an independently refined critical point. Record the crossing multiplier(s), critical parameter, branch relation and numerical uncertainty. Labels include saddle-node/fold, period-doubling, symmetry-breaking/pitchfork when symmetry justifies the term, and Hamiltonian-Hopf/Krein events where supported by the spectrum.

## v1 publishable target

The smallest defensible new paper is not "solve the three-body problem". It is:

> Reproduce a documented stable subset of the Li-Li-Liao family, build reproducible continuation and Floquet verification, then extend selected stable branches beyond the published sampling to resolve previously unmapped stability boundaries and bifurcation structure in unequal-mass parameter space.

A stronger v1 additionally reports one or more new verified connected branches or bifurcation links, with all numerical evidence released.

## Release gate

No manuscript sentence may use `new`, `discovered`, `first`, `stable`, or a bifurcation label based only on a screening result. Every table/figure of new scientific results must be generated from a frozen release manifest containing code commit, input hashes, solver configuration and verification status.
