# V1 closure attack

This file is the execution contract for the stated open problem. It deliberately suppresses attractive side projects until the finite v1 claim is either proved by the evidence protocol or falsified.

## Frozen v1 question

Determine the continuation-connected family decomposition of the Li--Li--Liao unequal-mass non-hierarchical periodic-orbit catalog and, for each connected family, compute the connected planar linear-stability critical manifold in mass space, classify its Floquet mechanisms and branch connections, and independently verify the qualitative critical events.

The target is **not** a solution of the general three-body problem. It is a finite numerical/mathematical statement about one frozen catalog and its continuation/stability geometry.

## The four closure gates

### Gate A -- independent critical-point truth

**Status: PASS for every mechanism label the manuscript is allowed to use.**

A mechanism word (`+1`, `-1`, mixed `(+1,-1)`, Hamiltonian--Hopf/Krein) may appear as an ATLAS result only if:

1. the periodic orbit is independently corrected in Julia BigFloat;
2. the same smooth event is localized without importing Python dynamics;
3. precision/tolerance convergence is recorded;
4. canonical Jacobi monodromy and the physical `E^omega/E` quotient are evaluated at the exact bracket;
5. closure, event, symplectic defect, reciprocal pairing, leakage, and parameter uncertainty are stored;
6. the independent root stays inside the declared tolerance.

Currently bound: principal lower `+1`; principal upper opposite-Krein Hamiltonian--Hopf; three mixed `(alpha,beta)=(4,4)` organizers.

Still Gate-A objects if the paper wants to use them: secondary-left fold, any fourth mixed organizer, daughter nondegeneracy. Until those exist, the paper must not use those labels as established.

A float64/JAX point is a proposal. It is never publication truth by itself.

### Gate B -- complete mechanism-resolved Floquet critical graph on the connected family sheet

**Status: PENDING. This is the remaining theorem.**

The final object is a graph, not a cloud, and it need not be connected just because the family sheet is connected. State `y=(x1,v1,v2,T,m1,m2)`. Mixed organizers are preimages of `(alpha,beta)=(4,4)`. The 620 catalog S/U cells are samples supporting the graph; they are not 620 edges.

Pass only when:

- all 620 catalog S/U cells are localized and each belongs to exactly one mechanism-specific polyline;
- mixed germs come from continuation artifacts, not nearby-root heuristics;
- secondary-left birth and secondary-right death are classified (fold, mixed, or domain boundary);
- the daughter is classified or the hypothesis is falsified (`no_branch_attachment` is allowed);
- completeness is frozen;
- no endpoint is `Newton failed`;
- `research/evidence/V1_CRITICAL_GRAPH.json` reports `release_ready: true`.

### Gate C -- family/sheet connectivity

**Status: PASS in the declared catalog domain.**

Family identity is continuation connectivity. The catalog is one continuation-connected component. Certificate: `research/V1_CONNECTIVITY_CERTIFICATE_2026-08-15.md`.

The corrected `(T_si,L_si)` projection is folded and non-injective. Projected two-set structure is not a dynamical split.

A chart singularity is not a moduli-space disconnection.

### Gate D -- release closure

**Status: PENDING.**

Pass only when Gate B is ready, the assembler has flipped `release_ready`, the novelty search is same-day, the manuscript is regenerated from `release_claim` records, and `--require-solved` passes on the tagged commit.

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
