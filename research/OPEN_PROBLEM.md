# ATLAS v1 open problem

## Claim we are trying to earn

For the Li--Li--Liao unequal-mass non-hierarchical periodic-orbit family with free-group word `bABabaBAba`, determine the connected linear-stability boundary in the two-dimensional mass-ratio plane and classify the critical Floquet mechanisms and branch/bifurcation structure along that boundary.

This is deliberately narrower than "solve the three-body problem". A successful result is still an open-problem contribution if the critical manifold and its dynamical organization were not previously resolved.

## What prior work already solved

Li, Li & Liao (2021; arXiv:2007.10184) continued one family on a mass grid with `m3=1`, `m1 in [0.8,1.1]`, `m2 in [0.7,1.2]`, spacing `0.001`, reported 135,445 periodic orbits, classified 13,315 as linearly stable, and plotted the resulting stable mass domain. Therefore ATLAS must not claim novelty for merely finding stable points, plotting the same domain, or interpolating that grid.

## Gap ATLAS targets

The v1 scientific contribution must go beyond that grid and resolve structure that the baseline paper does not provide:

1. Continue the *critical set itself* rather than sample stable/unstable points on a rectangular grid.
2. Resolve each boundary component with numerical uncertainty significantly below the published mass spacing.
3. Track Floquet multipliers continuously along each critical curve and classify the local loss/gain of stability (for example a +1/-1 crossing or Hamiltonian-Hopf/Krein event) only when the spectrum supports the label.
4. Detect folds, cusps, self-intersections, endpoints and branch junctions of the critical set.
5. Determine whether critical events spawn/connect distinct periodic-orbit branches; topology is attached but is not used as a substitute for continuation-based family identity.
6. Verify representative critical points with an independent arbitrary-precision implementation and frozen evidence manifests.

A stronger paper extends the same procedure to additional unequal-mass families and compares the geometry/topology of their stable islands.

## Decisive paper-level deliverables

The first manuscript may call the result new only after all of the following exist:

- a versioned curve dataset for every connected stability-boundary component in the chosen mass domain;
- high-precision corrected periodic orbit data on both sides and at representative critical points;
- continuous Floquet/monodromy diagnostics along the boundary;
- a reproducible event catalogue with numerical uncertainty and evidence for each bifurcation label;
- independent verification for every qualitative event type used in the paper;
- comparison against the original `0.001` grid showing which information is genuinely new;
- literature search frozen at release time documenting that the same critical-manifold result was not already published;
- figures/tables generated only from release-claim records.

## What does **not** count as solving this open problem

- reproducing the 13,315 stable points;
- refining one or two one-dimensional crossings without mapping the connected critical set;
- ML predictions without Newton correction and verification;
- a float64-only classification near a multiplier collision;
- calling spectral/linear stability a proof of nonlinear/KAM stability;
- calling a denser scatter plot an atlas.

## Numerical architecture

### Fast discovery/screening

Reference runtime: CPython 3.13, NumPy 2.x, SciPy. DOP853 plus analytic variational equations is the default CPU screening path. JAX is optional and benchmark-gated for batched candidate scoring, automatic differentiation experiments, and accelerator sweeps. It must not become a scientific dependency unless it demonstrates a clear throughput/reproducibility advantage.

### Publication verification

Julia 1.11 + BigFloat + adaptive high-order SciML integration is an independent implementation. Publication-critical points must be corrected and variationally re-integrated in arbitrary precision. A second independent numerical configuration is required for qualitative critical-event claims.

### Why JAX is not the verifier

JAX is excellent for accelerator-oriented array computation and automatic differentiation, but the publication gate needs arbitrary precision, controlled local error, and implementation independence from the Python screening stack. JAX can accelerate discovery; it cannot replace the independent evidence path.

## Success criterion

ATLAS v1 has solved its stated open problem when it can make and defend a statement of the following form:

> We compute and independently verify the connected stability-boundary manifold of the `bABabaBAba` unequal-mass periodic-orbit family over a declared mass domain, identify and classify its critical Floquet mechanisms and branch connections, and release the continuation/evidence data required to reproduce every qualitative feature.

Until those deliverables pass the release gate, the project is an open-problem solver under construction, not a solved problem.
