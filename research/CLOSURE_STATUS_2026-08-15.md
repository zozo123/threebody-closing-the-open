# V1 closure status — 2026-08-15

This note records the post-audit state of the frozen v1 open problem. It is a status checkpoint, not a release claim.

## What the evidence now supports

- The full 135,445-row mass-grid adjacency graph is connected.
- The corrected `(T_si,L_si)` projection is folded and non-injective, so projected branches cannot identify dynamical-family components by themselves.
- Five macroscopic MST cuts passed bidirectional Li-chart continuation.
- A deliberately far-mass / near-invariant-duplicate pair passed bidirectional continuation both in the Li chart and in the generic translation-reduced strict-periodic chart.
- The second-stage shooting-Jacobian audit re-corrected the 28 lowest-rank samples from the first census. The minimum corrected scaled rank ratio was `1.0803738448664921e-05`, the maximum corrected closure was `1.8017261738074185e-08`, and no candidate remained below the fixed `1e-6` suspicion threshold.
- The previously failing global-rank-3 MST edge `(0.839,0.721,1)<->(0.838,0.721,1)` has now passed a 12-substep bidirectional continuation in an independently reimplemented float64 relative-dynamics/shooting formulation without relaxing the frozen gates. Forward max residual was `2.224e-08` with terminal normalized mismatch `1.381e-09`; reverse max residual was `4.69844763e-08` with terminal mismatch `4.375e-10`. The exact old reverse failure location `theta=1/3` passed at residual `1.83048098e-09`.
- All 620 published S/U transition cells have exactly one endpoint-sign-changing smooth reduced-Floquet event: 198 `+1`, 168 `-1`, and 254 trace-collision cells. The coarse topology consists of four macroscopic tracks and three cross-mechanism adjacency junctions.
- The physical four-dimensional symplectic quotient on `E^omega/E` has passed its float64 structural audit without loosening its acceptance gates.
- The lower `+1` generic branch probe produced ten distinct float64 periodic-orbit daughter candidates at the tested signed amplitudes. Their closures range from about `8.15e-9` to `1.55e-7`. True generic pseudo-arclength daughter continuation is now implemented on `main`, but scientific branch-continuation evidence still requires an executed run and independent reproduction.

### All three coarse `+1/-1` switches now have mixed-vertex screening candidates

A fresh standalone float64 screening implementation independently reimplemented the eight-dimensional relative dynamics, analytic variational equations, fixed-mass shooting correction, reduced trace invariants, and a two-dimensional mass Newton solve for

`G+ = beta - 6 alpha + 20 = 0`,

`G- = beta - 2 alpha + 4 = 0`.

Starting only from the six retained junction localizations in the frozen global event-network artifact, it converged to three corrected candidates near the universal mixed spectral vertex `(alpha,beta)=(4,4)`:

1. principal lower left: masses `(0.9292392041495240, 0.8853664936499904, 1)`, closure `2.19e-11`, `alpha=4.0000000062292056`, `beta=4.000000013328596`, `G+=-2.40e-8`, `G-=8.70e-10`;
2. secondary lower switch: masses `(0.9967681989288031, 0.9560193632122531, 1)`, closure `1.25e-11`, `alpha=4.0000000005636283`, `beta=4.0000000017197053`, `G+=-1.66e-9`, `G-=5.92e-10`;
3. principal lower right: masses `(1.0495531760259145, 1.1294757891873943, 1)`, closure `1.37e-11`, `alpha=4.0000000055834377`, `beta=4.0000000068175865`, `G+=-2.67e-8`, `G-=-4.35e-9`.

The frozen screening record is `experiments/mixed_vertex_screening_2026-08-15.json`; the three chart/mass seeds are also frozen in `experiments/mixed_vertex_candidates.tsv` and split into one-row files for independent verification. This supersedes the earlier interpretation that the direct-search failure left the mixed organizers without concrete candidates. It does **not** promote the organizers to release claims: each candidate must still survive Julia BigFloat mass Newton, canonical physical/Jordan/nondegeneracy checks, and event-curve connection tracing.

### The secondary lobe has separate fold and mixed-switch evidence

The earlier fold candidate near `(0.995705,0.97424,1)` is not an alternative to the newly found mixed vertex at `(0.9967682,0.9560194,1)`; the two objects can coexist on the same secondary network.

A separate independent float64 root-count screen tested the local `G-=0` topology around the fold. At `m1=0.99560498`, sampled values at `m2=0.966261,0.974261,0.982261` were all negative. At `m1=0.99580498`, the corresponding signs were negative, positive, negative. Thus the local mass slice changes from no sampled `-1` roots to two sign-change brackets as `m1` crosses the fold neighborhood, which is the expected root-birth topology of a `-1` event-curve fold. The frozen screen is `experiments/secondary_minus_fold_topology_screen_2026-08-15.json`.

Finite-difference curvature estimates at the old fold coordinates remain numerically sensitive, so the release proof must use event-specific pseudo-arclength tangent/curvature geometry plus independent BigFloat reproduction rather than preserving the old nested finite-difference numbers.

## Important negative or incomplete results

### Mixed organizer verification is now the blocker, not organizer discovery

The earlier direct six-variable JAX-assisted search scanned 180 U->S coarse rows and launched ten direct solves but exhausted its function-evaluation budget without accepting `(alpha,beta)=(4,4)`. That negative result was a solver outcome, not a nonexistence result.

The new mass-eliminated screening solve supplies high-quality candidates for all three switches. The existing independent Julia verifier has therefore been converted to three one-candidate matrix jobs so a difficult organizer cannot consume the execution budget of the others. Until those jobs run and pass, the candidates remain screening-supported.

### Independent critical-curve reproduction timed out

The first monolithic Julia BigFloat verifier corrected the lower `+1` representative to closures down to roughly `1e-26` and localized its event near `m2 = 0.75401872...`. It also corrected the upper collision representative to roughly `1e-26` closure and found a sign-changing collision bracket. GitHub cancelled the job at the workflow time limit before the upper refinement completed and before an artifact was emitted.

This was a compute-architecture failure, not a detected numerical disagreement. The verifier is split into independent lower and upper jobs so neither event can consume the other's execution budget.

### The previously hard rank-3 MST edge is resolved; the full worst-edge matrix is not

The original strengthened audit found that the third globally largest MST edge, `(0.839,0.721,1)<->(0.838,0.721,1)`, failed the reverse six-substep Li-chart walk at `theta=1/3` with residual about `7.609e-05` against the frozen `2e-7` gate.

A new independent float64 reimplementation repeated that exact edge with 12 equal mass substeps and an analytic shooting corrector. Both directions passed. The old failure fraction `theta=1/3` passed at `1.83e-09`; terminal normalized chart mismatches were below `1.4e-09`. The frozen record is `experiments/hard_mst_rank3_12step_screen_2026-08-15.json`.

This removes the only currently observed continuation failure as evidence for disconnection and strongly supports a Newton-basin/step-size explanation. Gate C nevertheless remains pending because the remaining globally worst MST edges have not all completed the adaptive matrix, and any future survivor must still be repeated in the generic/path-diverse formulation.

### Daughter genealogy remains incomplete

Ten amplitude-constrained generic daughter candidates exist, but amplitude-constrained solves are branch switches, not branch continuations. `src/threebody_atlas/generic_branch.py` and `scripts/continue_lower_plus_one_daughter.py` now remove the artificial amplitude condition after two distinct seeds and continue `(z0,T,m2)` by pseudo-arclength while comparing every point against the independently corrected same-mass Li parent. The continuation geometry passed an analytic toy-branch execution test in the working environment, but no three-body daughter-continuation artifact exists yet because Actions runners are not starting.

### GitHub Actions compute is currently blocked externally

The split Julia critical-point verifier, split three-candidate Julia mixed-vertex verifier, twenty-edge adversarial connectivity matrix, three-junction organizer workflow, four-way daughter continuation matrix, latest BifurcationKit cross-check, CAPD validated-flow matrix, and ordinary CI are presently unable to start because GitHub reports that recent account payments failed or the Actions spending limit must be increased. Their latest hosted jobs have zero executed steps. These runs contain no scientific result.

### Strongest-hammers execution architecture is now on `main`

The compute-ready architecture is documented in `research/STRONGEST_HAMMERS_2026-08-15.md` and deliberately separates independent formulations:

- **Frozen BigFloat truth:** the existing Julia `1.11.9` + Vern9 + GenericSchur stack remains the formal independent reproduction lane.
- **Latest continuation cross-check:** a separate Julia `1.12.6` environment targets BifurcationKit `0.8.2`. `julia-latest/crosscheck_mixed_vertex.jl` independently reimplements the 8D relative dynamics and analytic variational equations, uses a rank-revealed normalized periodic section, and asks BifurcationKit Newton to solve four periodic-closure equations together with `G+=0` and `G-=0`. Its resolved `Manifest.toml` is retained with artifacts rather than treating a floating latest environment as reproducible evidence.
- **Validated numerics:** `validated/capd/validated_flow.cpp` uses the current frozen CAPD master commit `731079217a9254ea2948d742df2b170895effe7f` to rigorously enclose the full-period flow, first variational matrix, and interval `alpha`, `beta`, `G+`, `G-`, and discriminant values around each frozen mixed seed. Its declared scope is validated-flow scaffolding only; it does not claim periodic-orbit existence or organizer existence until an interval-Newton/Krawczyk root certificate is added.
- **Execution redundancy:** `.github/workflows/strong-hammers-self-hosted.yml` runs the same frozen BigFloat thresholds, the latest BifurcationKit cross-check, and the CAPD validation lane on a user-controlled Linux x86_64 runner, bypassing the GitHub-hosted billing condition without changing scientific gates.
- **Consensus firewall:** `scripts/organizer_consensus_gate.py` requires BigFloat, latest-BifurcationKit, and CAPD evidence to agree before the organizer evidence can be called cross-formulation supported. Passing that firewall still does not authorize a release claim.

The hosted latest-BifurcationKit run `31883185042` and CAPD run `31883494620` also received zero executed steps, confirming that their current failures are infrastructure-only rather than numerical outcomes.

## Current verdict

The stated v1 open problem is **not solved yet**.

The shortest closure path is now more specific:

1. independently reproduce the lower `+1` and upper collision representatives in Julia BigFloat and complete exact canonical mechanism checks;
2. independently reproduce all three mixed organizer candidates in the split Julia BigFloat verifier, cross-check them with the latest BifurcationKit lane, validate their flow/monodromy with CAPD, then add a rigorous interval-Newton/Krawczyk organizer root certificate and trace the `+1` and `-1` event arcs through each organizer;
3. independently verify the secondary `-1` root-birth fold with event-specific pseudo-arclength geometry and BigFloat/validated-numerics evidence;
4. finish the remaining globally worst MST links under adaptive continuation and attack every survivor in the generic chart/path-diverse formulation; the formerly failing rank-3 edge itself is now resolved;
5. execute generic daughter pseudo-arclength continuation, independently reproduce a clean daughter segment, and determine whether it reconnects to the catalog sheet;
6. assemble the exact critical graph, run hidden-pocket/component adversarial searches, freeze uncertainties/evidence hashes/component decomposition, rerun the release-date literature audit, and regenerate the manuscript from `release_claim` records only.

No `SOLVED` label is permitted before all six items close.
