# V1 closure attack

This file is the execution contract for the stated open problem. It deliberately suppresses attractive side projects until the finite v1 claim is either proved by the evidence protocol or falsified.

## Frozen v1 question

Determine the continuation-connected family decomposition of the Li--Li--Liao unequal-mass non-hierarchical periodic-orbit catalog and, for each connected family, compute the connected planar linear-stability critical manifold in mass space, classify its Floquet mechanisms and branch connections, and independently verify the qualitative critical events.

The target is **not** a solution of the general three-body problem. It is a finite numerical/mathematical statement about one frozen catalog and its continuation/stability geometry.

## The four closure gates

### Gate A -- independent critical-point truth

For every headline mechanism used in the paper:

1. correct the periodic orbit independently in Julia BigFloat;
2. localize the same smooth Floquet event without importing Python dynamics;
3. demonstrate precision/tolerance convergence;
4. evaluate canonical Jacobi monodromy;
5. report closure, event, symplectic, reciprocal-pairing, and parameter uncertainty;
6. withhold the release claim if the independent root moves outside the declared tolerance.

A float64/JAX point is a proposal. It is never publication truth by itself.

### Gate B -- exact Floquet organizer/network geometry

The sampled U/S boundary already contains `+1`, `-1`, and trace-collision mechanisms. The decisive question is whether apparent mechanism switches are true intersections/vertices of critical arcs or unresolved multiple zeros in coarse cells.

Use the augmented continuation state

`y=(x1,v1,v2,T,m1,m2)`

and solve periodic closure plus smooth event equations directly. In particular, the mixed `+1/-1` organizer is the preimage of

`(alpha,beta)=(4,4)`,

or equivalently

`P(+2)=0`, `P(-2)=0`.

The final critical object must be a graph, not a cloud of boundary samples. Every edge stores mechanism, continuation-sheet identity, uncertainty, and physical canonical evidence. Every apparent endpoint must be explained by a physical/domain boundary, fold, branch point, cusp, collision/symmetry stratum, or connection to another critical edge. "Newton failed" is not an endpoint classification.

### Gate C -- family/sheet connectivity

Family identity is continuation connectivity, not clustering and not coincidence in an invariant plot.

Already passed evidence:

- the corrected `(T_si,L_si)` projection is folded/non-injective;
- five macroscopic sampled bottlenecks were crossed forward/reverse;
- a deliberately far-mass, invariant-near-duplicate pair was crossed in 80 steps in both directions with terminal chart mismatch of order `1e-13`.

Still required:

1. census the scaled shooting-Jacobian singular spectrum, including the full `m2=m3=1` zero-angular-momentum spine;
2. attack every serious rank-loss candidate with smaller steps and reverse continuation;
3. repeat the worst connection with a less symmetry-specialized formulation (generic gauge-fixed shooting, multiple shooting, or collocation BVP);
4. use path diversity/loop lifting where needed to distinguish an ordinary projection fold from nontrivial sheet monodromy;
5. do not promote topology words or projected invariant branches to family identifiers.

A chart singularity is not a moduli-space disconnection.

### Gate D -- release closure

The result is release-ready only when:

1. all declared critical components have been traced and connected into the mechanism graph;
2. representative points on every mechanism class pass independent high-precision/canonical checks;
3. family connectivity has survived the rank and chart-independence attacks;
4. adversarial searches find no hidden stability pocket/component in the declared domain;
5. a frozen manifest hashes every source dataset, solver environment, evidence artifact, figure, and table;
6. the manuscript is regenerated from that manifest;
7. a fresh literature search is run immediately before the novelty freeze.

## Mathematical contract

### Stability scope

Unless vertical/spatial variational modes are explicitly added, every use of "stable" in v1 means **linear Floquet stability within the planar problem**. It does not imply spatial stability and does not imply KAM/nonlinear stability.

### Strict periodicity

Quotienting rotation is a mathematical convenience, not permission to confuse a closed shape loop with an inertial periodic orbit. Any future fully rotation-reduced formulation must retain the reconstruction angle and distinguish strict periodicity from relative periodicity.

### Similarity and coordinates

The current Li normalization fixes the Newtonian scale operationally. Before intrinsic-coordinate claims are published, similarity-free quantities must be derived with the correct powers of `G` and mass rather than inferred from raw `(E,L,T)` plots.

### Moduli-space language

Do not publish a formal dimension for the quotient moduli space until the symmetry, first-integral, phase, scaling, and discrete-permutation reductions are written explicitly. For v1 the operational definition is sufficient: a dynamical family is a connected component under branch-preserving continuation after the declared gauge/normalization choices, with suspicious connections repeated in an independent chart.

## What is deliberately deferred

The following are valuable but are **not v1 blockers** unless a core gate forces them:

- Conley--Zehnder/Maslov/Floer bookkeeping;
- Broucke/GIT/B-signature refinements beyond what is needed to classify a critical event;
- KAM/nonlinear stability;
- 3D/vertical stability;
- complex-mass or complex-time continuation;
- large cross-family universality scans;
- AI discovery sweeps;
- number-theory/symbolic-regression searches;
- global collision regularization when no v1 endpoint approaches collision;
- exhaustive shape-physics explanations.

These projects reopen after the finite critical graph and connectivity statement are closed.

## Stop rule

No new research direction enters v1 merely because it is interesting. It enters only if it is necessary to decide one of Gates A--D or to answer a concrete falsification raised by those gates.
