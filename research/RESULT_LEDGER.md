# Research result ledger

This file is the claim firewall for the current open-problem attack. It records what the computations presently support and, equally importantly, what they do **not** support.

Status vocabulary:

- **VERIFIED-ARTIFACT**: reproduced by a frozen-data GitHub Actions workflow with explicit acceptance gates and retained artifact digest.
- **FLOAT64-STRUCTURAL**: a frozen, gated structural calculation in the screening implementation; stronger than a heuristic but not a substitute for independent arbitrary-precision reproduction of a publication-critical event.
- **INDEPENDENT-STRUCTURAL**: independently implemented numerical structure check, but not yet an exact publication-critical localization.
- **SCREENING-SUPPORTED**: reproducible float64 evidence that determines the next experiment but is not a release claim by itself.
- **CANDIDATE**: plausible interpretation awaiting a decisive numerical test.
- **INVALIDATED**: a previous interpretation or computation that failed an evidence gate.

## Family identity / invariant projection

### The corrected `(T_si,L_si)` projection is folded and non-injective

**Status: VERIFIED-ARTIFACT**

Frozen-data full-catalog audit:

- 135,445 catalog rows;
- 134,046 interior finite-difference Jacobians of `(m1,m2)->(T_si,L_si)`;
- normalized projection determinant has both signs: 113,514 positive and 20,532 negative;
- 590 adjacent determinant sign-change edges;
- those sign-change nodes form one connected fold locus;
- minimum normalized singular-value ratio about `1.39e-6`;
- 27,111 rows have a nearest invariant-space neighbor at least `0.05` away in mass space;
- 86 such far-mass matches have standardized invariant distance below `1e-4`.

Interpretation: the invariant plane is a many-to-one folded projection of the sampled orbit sheet. Therefore invariant functional branches alone are not a topological certificate of disconnected dynamical families.

This does **not** by itself prove that the full catalog is one continuation-connected family.

### Macroscopic shooting-chart bottlenecks continue bidirectionally

**Status: SCREENING-SUPPORTED**

Five minimum-spanning-tree cuts chosen to separate progressively larger fractions of the full sampled catalog were crossed by independent endpoint correction plus forward and reverse shooting continuation. All five passed. The hardest retained cut separates about 40% of the spanning tree and has terminal normalized chart mismatch below `1e-12` in both directions.

### A deliberately pathological invariant-near-duplicate pair continues bidirectionally

**Status: VERIFIED-ARTIFACT**

A frozen-data adversarial bridge selected two catalog orbits that are far apart in mass space while almost coincident in the corrected invariant projection. Direct continuation was then run independently forward and backward between them rather than inferring connectivity from the invariant plot.

Retained evidence:

- mass-space endpoint distance: `0.06171709649683785`;
- 80 continuation steps in each direction;
- maximum forward/reverse shooting residual about `2.25e-10`;
- forward/reverse terminal chart mismatch below `6e-13`;
- workflow run `31873131455`, artifact `9244246378`.

The same deliberately pathological bridge was subsequently repeated in the generic translation-reduced strict-periodic formulation and passed in both directions.

### Shooting-rank refinement found no surviving singularity below the fixed threshold

**Status: VERIFIED-ARTIFACT**

The 28 lowest-rank candidates from the first shooting-Jacobian census were tightly re-corrected. The minimum corrected scaled rank ratio was `1.0803738448664921e-05`, maximum corrected closure `1.8017261738074185e-08`, and zero candidates remained below the fixed `1e-6` suspicion threshold. Workflow run `31875617952`, artifact `9244717474`.

### Global worst-edge connectivity is not closed

**Status: CANDIDATE / ADVERSARIAL TEST ACTIVE**

Five balanced cuts and the first two globally largest MST chart jumps passed. The third globally largest edge, `(0.839,0.721,1)<->(0.838,0.721,1)`, failed the reverse six-substep Li-chart walk at residual about `7.609e-05` against the frozen `2e-7` gate.

This is not evidence of physical disconnection yet. The adaptive top-20 workflow retries each edge at `6 -> 12 -> 24 -> 48` substeps without relaxing thresholds; any survivor must be attacked in the generic chart and, if required, by path diversity/loop lifting.

**Family claim remains withheld until that adversarial closure completes.**

## Stability critical set

### The published S/U transition set contains four macroscopic coarse tracks

**Status: SCREENING-SUPPORTED**

The frozen `0.001` mass grid contains 620 adjacent S/U transition brackets. Continuity in mass space resolves four macroscopic tracks:

1. principal `U->S`, `m1=0.800..1.071`;
2. principal `S->U`, `m1=0.800..1.053`;
3. secondary `U->S`, `m1=0.996..1.042`;
4. secondary `S->U`, `m1=0.996..1.042`.

All 620 cells exhibit exactly one endpoint-sign-changing smooth reduced-Floquet event: 198 `+1`, 168 `-1`, and 254 trace collisions. The coarse network contains three cross-mechanism adjacency junctions.

### The critical set is a Floquet event network, not four single-mechanism curves

**Status: SCREENING-SUPPORTED**

The coarse mechanism assignment is:

- principal upper `S->U`: trace-root collision (`Delta=0`);
- secondary upper `S->U`: `-1`;
- secondary lower `U->S`: `-1 -> +1`;
- principal lower `U->S`: `+1 -> -1 -> +1`.

The universal trace polynomial implies that a genuine smooth `+1/-1` mechanism switch on one continuation sheet must pass through `(alpha,beta)=(4,4)`, unless the coarse association aliases multiple nearby roots.

### All three coarse `+1/-1` switches have mixed spectral-vertex candidates

**Status: SCREENING-SUPPORTED**

An earlier direct six-variable JAX-assisted solve exhausted its evaluation budget for the ten closest seeds and accepted no mixed vertex. That negative result is retained as solver evidence; it is **not** a nonexistence result.

A fresh independent float64 screening implementation then reimplemented the eight-dimensional relative dynamics, analytic variational equations, fixed-mass shooting correction, and reduced trace invariants. The four orbit variables were eliminated with the analytic shooting Jacobian and a two-dimensional mass Newton solve targeted

`G+ = beta - 6 alpha + 20 = 0`,

`G- = beta - 2 alpha + 4 = 0`.

Starting only from the six retained coarse-junction localizations, the solver converged to:

| organizer | `(m1,m2,m3)` | closure | `alpha` | `beta` | `G+` | `G-` |
|---|---|---:|---:|---:|---:|---:|
| principal lower left | `(0.9292392041495240, 0.8853664936499904, 1)` | `2.19e-11` | `4.0000000062292056` | `4.000000013328596` | `-2.40e-8` | `8.70e-10` |
| secondary lower switch | `(0.9967681989288031, 0.9560193632122531, 1)` | `1.25e-11` | `4.0000000005636283` | `4.0000000017197053` | `-1.66e-9` | `5.92e-10` |
| principal lower right | `(1.0495531760259145, 1.1294757891873943, 1)` | `1.37e-11` | `4.0000000055834377` | `4.0000000068175865` | `-2.67e-8` | `-4.35e-9` |

The frozen record is `experiments/mixed_vertex_screening_2026-08-15.json`. One-row seed files are frozen for each candidate, and the independent Julia BigFloat mixed-vertex verifier is split into three matrix jobs.

Interpretation: organizer **discovery** is no longer the main blocker. The remaining organizer gate is independent arbitrary-precision reproduction, event-arc connection, canonical physical/Jordan structure, and nondegeneracy classification.

These points remain screening results, not release claims.

### The secondary lobe is consistent with a `-1` fold followed by a distinct mixed switch

**Status: SCREENING-SUPPORTED**

The earlier float64 candidate placed a `-1` event-curve fold near `(m1,m2)=(0.995705,0.97424)`. Its nested finite-difference curvature values were not fully robust under reruns, so those exact derivative numbers are not treated as publication evidence.

A separate independent root-count screen gives a more robust topological signature. Around the same fold neighborhood:

- at `m1=0.99560498`, corrected `G-` values at `m2=0.966261,0.974261,0.982261` are all negative;
- at `m1=0.99580498`, the signs become negative, positive, negative.

Thus the local slice changes from no sampled `-1` zero to two sign-change brackets, the expected zero-to-two root birth of a fold. This is recorded in `experiments/secondary_minus_fold_topology_screen_2026-08-15.json`.

This object is geometrically distinct from the newly found secondary mixed `-1/+1` candidate near `(0.9967682,0.9560194)`. The current screening model is therefore: a `-1` fold creates the secondary lobe; farther along its lower boundary, a mixed organizer changes the active mechanism from `-1` to `+1`.

Event-specific pseudo-arclength geometry and independent BigFloat reproduction remain mandatory.

### A simple one-mechanism secondary critical loop

**Status: INVALIDATED**

A first pseudo-arclength attempt assumed that the visually adjacent secondary lower crossings at `m1=0.996` and `0.997` represented the same smooth event. It failed because the first localized crossing was `-1` while the second coarse bracket did not contain a `-1` zero. The later global event census, mixed organizer candidate, and separate fold screen explain why that one-mechanism assumption was too simple.

## Physical transverse Floquet reduction

### The regular physical return map is constructed on `E^omega/E`, not by deleting four eigenvalues

**Status: FLOAT64-STRUCTURAL / VERIFIED-ARTIFACT**

In canonical translation-reduced Jacobi coordinates let

`E = span{X_H, X_L}`,

where `X_H` is the time/energy generator and `X_L` is the planar rotation/angular-momentum generator. At regular points `E` is a two-dimensional isotropic subspace and the physical transverse return map acts on the four-dimensional symplectic quotient `E^omega/E`.

A direct numerical quotient construction was checked on published S/U anchors and the frozen lower-`+1` / upper-collision representatives. The acceptance gates were not loosened; tighter float64 orbit/tangent integration made the original invariant, symplectic, leakage, pairing and neutral-invariance gates pass.

Tightened frozen run `31875065636`, artifact `9244514392`:

- max physical/reduced invariant mismatch `3.98e-06`;
- max canonical monodromy symplectic defect `7.02e-07`;
- max physical quotient symplectic defect `1.59e-07`;
- max quotient leakage `6.88e-09`;
- max reciprocal-pairing error `1.76e-05` near the ill-conditioned upper collision;
- max neutral-subspace invariance defect `3.95e-08`.

The physical event equations agree exactly with the reduced trace algebra through

- `a=alpha-4`;
- `b=beta-4alpha+10`;
- `+1`: `b-2a+2=0`;
- `-1`: `b+2a+2=0`;
- collision: `a^2-4b+8=0`.

Publication-critical event locations and Krein/Jordan classifications remain gated on the independent BigFloat/canonical path.

## Upper-boundary mechanism

### Coarse upper transition is consistent with a Krein / Hamiltonian--Hopf loss of stability

**Status: INDEPENDENT-STRUCTURAL**

Independent Julia BigFloat canonical-Jacobi integration with Vern9 and GenericSchur passes hard numerical gates on published stable/unstable anchors: symplectic defect is of order `1e-21` and reciprocal pairing error of order `1e-25`.

Immediately before the coarse upper transition the two nontrivial elliptic modes have opposite Krein signs. Immediately after it the four nontrivial multipliers form a reciprocal complex quartet off the unit circle.

Release label is withheld until the independently corrected exact collision point is evaluated canonically at BigFloat precision.

## Lower `+1` daughter genealogy

### Ten distinct amplitude-constrained daughter candidates exist

**Status: SCREENING-SUPPORTED**

The physical lower-`+1` branch-switch probe produced ten distinct same-period generic periodic-orbit candidates across two soft physical directions and signed amplitudes. Closures range from about `8.15e-9` to `1.55e-7`; normalized distances from the independently corrected same-mass Li parent range from roughly `5.84e-4` upward.

These are branch-switch candidates, not a continued daughter family.

### True generic daughter pseudo-arclength continuation is implemented

**Status: ENGINEERING COMPLETE / SCIENTIFIC RUN PENDING**

`src/threebody_atlas/generic_branch.py` removes the artificial amplitude condition after two accepted daughter seeds and continues the strict-periodic generic state `(z0,T,m2)` by pseudo-arclength at fixed `(m1,m3)`. `scripts/continue_lower_plus_one_daughter.py` compares every accepted point with the independently corrected same-mass Li parent and emits independent-reproduction seeds.

The predictor/corrector geometry passed an analytic toy-branch execution test in the working environment, including a three-step monotone trace with zero synthetic closure/arclength residual. No claim about the three-body daughter topology follows from that toy test; a real daughter artifact is still required.

## High-precision verification

### Earlier failure from serialized float64 boundary seeds

**Status: INVALIDATED**

An earlier BigFloat run started from a TSV whose shooting parameters did not match the provenance header. The resulting order-unity initial closure was a bad-input artifact, not a scientific contradiction. The seed path was corrected.

### Lower `+1` and upper collision exact roots

**Status: CANDIDATE / COMPUTE-READY**

The first monolithic Julia verifier reached roughly `1e-26` periodic closure on both representative searches; it localized the lower event near `m2=0.75401872...` and found a sign-changing upper-collision bracket. The workflow timed out before final upper refinement/artifact emission. It is now split into independent lower and upper jobs.

### Three mixed organizers

**Status: CANDIDATE / COMPUTE-READY**

The three new float64 organizer seeds are frozen and the existing independent Julia BigFloat mass-Newton verifier has been split into one candidate per job. No BigFloat mixed-vertex artifact exists yet.

## Universal reduced Floquet geometry

**Status: STRUCTURAL ALGEBRA**

For the two nontrivial reciprocal multiplier pairs with trace roots `t=lambda+1/lambda`,

`P(t)=t^2-(alpha-4)t+beta-4alpha+8`.

The three smooth boundary equations are:

- `P(+2)=beta-6 alpha+20=0`;
- `P(-2)=beta-2 alpha+4=0`;
- `Delta=alpha^2+8 alpha-16-4 beta=0`.

The exact spectral vertices are:

- double `-1`: `(alpha,beta)=(0,-4)`;
- mixed `+1/-1`: `(4,4)`;
- double `+1`: `(8,28)`.

The mass-space stability diagram is the preimage of this universal spectral-stability domain under the continuation-sheet trace map.

## External execution status

GitHub Actions is presently refusing runner allocation because of the account payment/spending-limit condition. The affected jobs terminate with zero executed steps. This blocks the split Julia critical-point verifier, three-way Julia mixed-vertex verifier, adaptive top-20 connectivity matrix, junction continuation matrix, daughter continuation matrix and ordinary CI. A zero-step job is not scientific evidence either for or against a claim.

## Current publication threshold

The project is **not yet entitled to say the stated open problem is solved**. The shortest remaining route is:

1. independently reproduce the lower `+1`, upper collision, and all three mixed organizers in Julia BigFloat; evaluate corrected headline events canonically with convergence/uncertainty records;
2. trace `+1`/`-1` arcs through each mixed organizer and independently verify the secondary `-1` root-birth fold with event-specific pseudo-arclength geometry;
3. complete the adaptive top-20 MST attack and generic/path-diverse fallback for every survivor;
4. execute and independently reproduce a real lower-`+1` daughter continuation and classify reconnection versus distinct branch;
5. assemble and adversarially attack the complete critical graph/component decomposition, including hidden-pocket searches;
6. freeze evidence manifests and manuscript inputs, then rerun the current literature novelty audit immediately before release.
