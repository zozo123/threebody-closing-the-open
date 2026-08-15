# Strongest-hammers closure architecture — 2026-08-15

This file defines the end-to-end numerical attack for the frozen v1 problem. It does **not** change the scientific status from OPEN. Its purpose is to make every remaining claim survive multiple independent formulations rather than a single solver or tolerance choice.

## Rule zero: no single implementation certifies itself

Every headline object must survive at least two independent numerical formulations, and publication-critical existence/mechanism claims must additionally survive an arbitrary-precision or validated-numerics lane. Float64/JAX remains discovery and derivative-oracle infrastructure, never final truth.

## Lane A — frozen independent BigFloat truth

Keep the existing Julia 1.11.9 + Vern9 + GenericSchur environment unchanged as the formal independent reproduction lane. It must independently correct and localize:

1. the lower `+1` representative;
2. the upper trace-collision representative;
3. all three frozen mixed `(+1,-1)` organizer seeds;
4. representative points around the secondary `-1` fold.

Required outputs include closure, conserved quantities where applicable, event residual, mass uncertainty/bracket width, symplectic defect, reciprocal pairing, eigenspace conditioning, and canonical physical/Jordan/Krein diagnostics.

## Lane B — latest BifurcationKit cross-check

A separate Julia 1.12.6 environment targets BifurcationKit 0.8.2 and the current SciML ODE stack. It is deliberately **not** the release lock until an instantiated Manifest is frozen.

For each frozen mixed seed, the cross-check must:

- reconstruct the 8D translation-reduced ODE independently;
- seed a multiple-shooting representation from a fresh Vern9 integration;
- correct the strict periodic orbit with BifurcationKit shooting/Newton rather than the ATLAS Li-chart corrector;
- recompute the monodromy independently from analytic variational equations;
- report `G+`, `G-`, `(alpha,beta)`, closure and parameter drift;
- later continue the relevant fold/flip curves in two parameters and test whether BifurcationKit identifies the same codimension-two organizers.

The latest BifurcationKit lane is an independent formulation check. It does not supersede arbitrary precision or validated numerics.

## Lane C — validated numerics / computer-assisted proof

CAPD is the preferred rigorous-flow backend because it supports interval ODE integration, variational enclosures, Poincare maps and interval-Newton style existence/uniqueness arguments.

The rigorous lane is staged so we do not overclaim:

### C1: rigorous flow boxes

For each frozen seed, enclose one full reduced flow and its first variational derivative over a narrow interval box. Reject any box that approaches a binary collision or loses derivative control.

### C2: periodic-orbit certificate

Construct a phase-fixed local Poincare/shooting map with the continuous symmetry/gauge directions removed. Apply interval Newton/Krawczyk to prove existence and local uniqueness of the periodic representative inside a certified box.

### C3: organizer certificate

Augment the certified periodic equations with the two smooth event equations (`G+=0`, `G-=0`) and validate a nonsingular reduced Jacobian or equivalent Krawczyk contraction. This is the target certificate for a transverse mixed organizer.

### C4: fold certificate

For the secondary `-1` birth, validate `G-=0` together with the appropriate tangent degeneracy and nonzero quadratic/transverse coefficients. The current 0→2 root-count screen is only the seed geometry for this proof.

IntervalArithmetic/TaylorModels/Arb may be used for algebraic contraction, Taylor remainder control and ball arithmetic, but only when fed rigorously enclosed flow/variational data. Wrapping nonrigorous ODE output in intervals is explicitly forbidden as proof evidence.

## Lane D — Floquet spectral hardening

Use three independent views of the same physical return map:

1. invariant trace algebra `(alpha,beta,Delta)`;
2. the canonical Jacobi monodromy and physical quotient `E^omega/E`;
3. periodic-Schur based multiplier extraction when a compatible current stack is resolved.

No Hamiltonian-Hopf, mixed, flip, fold, Jordan or Krein mechanism label is promoted until the independently corrected critical point has consistent physical-spectrum evidence and reciprocal/symplectic defects below frozen gates.

## Lane E — family connectivity as a graph-certification problem

Do not infer connectedness from invariant plots or a finite mass adjacency graph. Attack the catalog with:

- adaptive `6→12→24→48` continuation on the globally worst MST edges;
- bidirectional terminal-match gates;
- generic translation-reduced strict-periodic fallback for every survivor;
- path diversity / loop lifting if multiple homotopy routes are available;
- rank-loss surveillance of the shooting Jacobian along every hard path.

The previously failing global-rank-3 edge is now resolved under unchanged gates. Gate C remains open until the rest of the adversarial matrix closes.

## Lane F — daughter genealogy

The lower `+1` daughter hypothesis must be treated as a branch-graph question, not a collection of nearby corrected points.

- generate two accepted generic branch-switch seeds;
- remove the amplitude constraint;
- continue by generic pseudo-arclength in `(z0,T,m2)`;
- compare each point against the independently corrected same-mass Li parent;
- search for reconnection, turning points and secondary bifurcations;
- reproduce a clean segment in an independent/high-precision formulation.

Only then classify the daughter as a distinct branch, reconnecting branch, or false branch-switch signal.

## Lane G — execution redundancy

GitHub-hosted Actions currently allocates zero steps because of the account payment/spending-limit condition. Therefore the repository carries two execution classes:

- hosted workflows, retained for ordinary reproducibility once runners resume;
- self-hosted workflows, capable of running the same frozen scripts on a user-controlled Linux x86_64 runner without changing scientific gates.

A compute outage is never converted into a scientific pass or fail.

## Release stop rule

`SOLVED` remains forbidden until Gates A–D in `V1_CLOSURE.md` pass. The strongest-hammers architecture changes **how aggressively we attack the gates**, not what counts as evidence.
