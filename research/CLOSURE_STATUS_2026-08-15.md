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

## Important negative or incomplete results

### Mixed `+1/-1` organizers are unresolved

The direct six-variable search targeted `(alpha,beta)=(4,4)`. It scanned 180 U->S coarse rows and launched ten direct solves from the closest seeds, but every direct solve exhausted its function-evaluation budget and no mixed-vertex candidate was accepted.

The three coarse mechanism changes therefore cannot yet be called codimension-two mixed vertices. They may be separated critical components, projection/association effects, or true organizers that require event-specific pseudo-arclength geometry to reach.

### Independent critical-curve reproduction timed out

The first monolithic Julia BigFloat verifier corrected the lower `+1` representative to closures down to roughly `1e-26` and localized its event near `m2 = 0.75401872...`. It also corrected the upper collision representative to roughly `1e-26` closure and found a sign-changing collision bracket. GitHub cancelled the job at the workflow time limit before the upper refinement completed and before an artifact was emitted.

This was a compute-architecture failure, not a detected numerical disagreement. The verifier is now split into independent lower and upper jobs so neither event can consume the other's execution budget.

### The strengthened worst-MST audit exposed a hard edge

The five balanced cuts and the first two globally largest MST chart jumps passed. The third globally largest edge, connecting masses `(0.839,0.721,1)` and `(0.838,0.721,1)`, failed the reverse six-substep Li-chart walk at `theta=1/3` with shooting residual about `7.609e-05` against a fixed `2e-7` gate.

This is not yet evidence of disconnection: it is a branch-basin/conditioning challenge until smaller substeps and an independent generic chart are attempted. A new rank-parallel adversarial workflow retries each of the twenty largest MST edges at 6, 12, 24, and 48 substeps without loosening residual or terminal-match thresholds.

## Current verdict

The stated v1 open problem is **not solved yet**.

The shortest closure path is now finite:

1. finish independent BigFloat reproduction of the lower `+1` and upper collision representatives;
2. resolve all twenty worst MST links under adaptive continuation, then attack any survivors in the generic chart;
3. continue and independently reproduce at least one lower-`+1` daughter branch and determine whether it reconnects to the catalog sheet;
4. trace the event-specific critical arcs through the three coarse mechanism-junction neighborhoods and determine whether the apparent switches are true mixed vertices or separate projected components;
5. complete physical/canonical mechanism classification at the independently corrected points;
6. freeze the connected critical graph, component decomposition, uncertainty records, current literature audit, evidence manifest, and manuscript.

No `SOLVED` label is permitted before all six items close.
