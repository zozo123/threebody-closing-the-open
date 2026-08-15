# V1 closure status — 2026-08-15

This note records the post-audit state of the frozen v1 open problem. It is a status checkpoint, not a release claim.

## What the evidence now supports

- The full 135,445-row mass-grid adjacency graph is connected.
- The corrected `(T_si,L_si)` projection is folded and non-injective, so projected branches cannot identify dynamical-family components by themselves.
- Five macroscopic MST cuts passed bidirectional Li-chart continuation.
- A deliberately far-mass / near-invariant-duplicate pair passed bidirectional continuation both in the Li chart and in the generic translation-reduced strict-periodic chart.
- The second-stage shooting-Jacobian audit re-corrected the 28 lowest-rank samples from the first census. The minimum corrected scaled rank ratio was `1.0803738448664921e-05`, the maximum corrected closure was `1.8017261738074185e-08`, and no candidate remained below the fixed `1e-6` suspicion threshold.
- All 620 published S/U transition cells have exactly one endpoint-sign-changing smooth reduced-Floquet event: 198 `+1`, 168 `-1`, and 254 trace-collision cells. The coarse topology consists of four macroscopic tracks and three cross-mechanism adjacency junctions.
- The physical four-dimensional symplectic quotient on `E^omega/E` has passed its float64 structural audit without loosening its acceptance gates.
- The lower `+1` generic branch probe produced ten distinct float64 periodic-orbit daughter candidates at the tested signed amplitudes. Their closures range from about `8.15e-9` to `1.55e-7`. This is a branch hypothesis only; continuation and independent reproduction are still required.
- A frozen earlier run produced a float64 candidate for the birth of the secondary stable lobe as a `-1` event-curve fold by solving `P(-2)=0` and `dP(-2)/dm2=0`. The candidate is near masses `(0.9957049987,0.9742436529,1)`, with shooting residual `1.294e-10`, `dF/dm1≈28.06`, `dF/dm2≈-2.89e-3`, and `d2F/dm2^2≈-787.3`. This is strong screening evidence for a generic quadratic fold but not a release claim.

## Important negative or incomplete results

### Mixed `+1/-1` organizers are unresolved

The direct six-variable search targeted `(alpha,beta)=(4,4)`. It scanned 180 U->S coarse rows and launched ten direct solves from the closest seeds, but every direct solve exhausted its function-evaluation budget and no mixed-vertex candidate was accepted.

The three coarse mechanism changes therefore cannot yet be called codimension-two mixed vertices. They may be separated critical components, projection/association effects, or true organizers that require event-specific pseudo-arclength geometry to reach. In particular, the secondary-lobe birth now has a concrete competing explanation: a `-1` critical-curve fold.

The secondary-fold candidate is itself not yet robust enough for release: a later run of the same high-level search missed its fixed float64 gate (`F≈8.61e-7`, `dF/dm2≈2.46e-2`). Independent high-precision/event-specific continuation must determine whether this is ordinary numerical conditioning or a weakness of the nested finite-difference formulation.

### Independent critical-curve reproduction timed out

The first monolithic Julia BigFloat verifier corrected the lower `+1` representative to closures down to roughly `1e-26` and localized its event near `m2 = 0.75401872...`. It also corrected the upper collision representative to roughly `1e-26` closure and found a sign-changing collision bracket. GitHub cancelled the job at the workflow time limit before the upper refinement completed and before an artifact was emitted.

This was a compute-architecture failure, not a detected numerical disagreement. The verifier is now split into independent lower and upper jobs so neither event can consume the other's execution budget.

### The strengthened worst-MST audit exposed a hard edge

The five balanced cuts and the first two globally largest MST chart jumps passed. The third globally largest edge, connecting masses `(0.839,0.721,1)` and `(0.838,0.721,1)`, failed the reverse six-substep Li-chart walk at `theta=1/3` with shooting residual about `7.609e-05` against a fixed `2e-7` gate.

This is not yet evidence of disconnection: it is a branch-basin/conditioning challenge until smaller substeps and an independent generic chart are attempted. A new rank-parallel adversarial workflow retries each of the twenty largest MST edges at 6, 12, 24, and 48 substeps without loosening residual or terminal-match thresholds.

### GitHub Actions compute is currently blocked externally

The new split Julia verifier, the twenty-edge adversarial connectivity matrix, and ordinary CI are presently unable to start because GitHub reports that recent account payments failed or the Actions spending limit must be increased. These runs contain no scientific result because their jobs never started. The workflows are already on `main`; compute can resume once the GitHub billing/spending block is cleared.

## Current verdict

The stated v1 open problem is **not solved yet**.

The shortest closure path is now finite:

1. finish independent BigFloat reproduction of the lower `+1` and upper collision representatives;
2. resolve all twenty worst MST links under adaptive continuation, then attack any survivors in the generic chart;
3. continue and independently reproduce at least one lower-`+1` daughter branch and determine whether it reconnects to the catalog sheet;
4. resolve the three coarse mechanism-junction neighborhoods by event-specific continuation, starting with the secondary `-1` fold hypothesis and then the two principal-lower `+1/-1` junctions;
5. complete physical/canonical mechanism classification at the independently corrected points;
6. freeze the connected critical graph, component decomposition, uncertainty records, current literature audit, evidence manifest, and manuscript.

No `SOLVED` label is permitted before all six items close.
