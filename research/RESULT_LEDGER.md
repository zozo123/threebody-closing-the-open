# Research result ledger

This file is the claim firewall for the current open-problem attack.  It records what the computations presently support and, equally importantly, what they do **not** support.

Status vocabulary:

- **VERIFIED-ARTIFACT**: reproduced by a frozen-data GitHub Actions workflow with explicit acceptance gates and retained artifact digest.
- **INDEPENDENT-STRUCTURAL**: independently implemented numerical structure check (for example canonical BigFloat symplectic verification), but not yet an exact critical-point localization.
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

Interpretation: the invariant plane is a many-to-one folded projection of the sampled orbit sheet.  Therefore invariant functional branches alone are not a topological certificate of disconnected dynamical families.

This does **not** by itself prove that the full catalog is one continuation-connected family.

### Macroscopic shooting-chart bottlenecks continue bidirectionally

**Status: SCREENING-SUPPORTED**

Five minimum-spanning-tree cuts chosen to separate progressively larger fractions of the full sampled catalog were crossed by independent endpoint correction plus forward and reverse shooting continuation.  All five passed.  The hardest retained cut separates about 40% of the spanning tree and has terminal normalized chart mismatch below `1e-12` in both directions.

Interpretation: no branch hysteresis was detected across the strongest tested macroscopic sampled bottlenecks.

Pending gates:

- shooting-Jacobian rank census;
- direct bridge across an explicit far-mass/non-injective invariant pair;
- denser adversarial checks around worst-conditioned points.

## Stability critical set

### The published S/U transition set contains four macroscopic coarse tracks

**Status: SCREENING-SUPPORTED**

The frozen `0.001` mass grid contains 620 adjacent S/U transition brackets.  Continuity in mass space resolves four macroscopic tracks:

1. principal `U->S`, `m1=0.800..1.071`;
2. principal `S->U`, `m1=0.800..1.053`;
3. secondary `U->S`, `m1=0.996..1.042`;
4. secondary `S->U`, `m1=0.996..1.042`.

This falsifies a globally adequate description of the sampled stability boundary as only two monotone edges.

### The critical set is a Floquet event network, not four single-mechanism curves

**Status: SCREENING-SUPPORTED**

Representative smooth-event localization shows:

- principal upper `S->U`: trace-root collision (`Delta=0`) at all sampled representatives;
- secondary upper `S->U`: multiplier `-1` event at all sampled representatives;
- secondary lower `U->S`: changes from `-1` near its left end to `+1` in its interior/right end;
- principal lower `U->S`: changes `+1 -> -1 -> +1` across the sampled mass range.

The universal trace polynomial implies that a genuine mechanism switch between `+1` and `-1` arcs must pass through the exact mixed spectral vertex `(alpha,beta)=(4,4)`, unless a coarse S/U cell contains multiple distinct event zeros.  Both alternatives are now being tested directly.

### A simple one-mechanism secondary critical loop

**Status: INVALIDATED**

A first pseudo-arclength loop attempt assumed that the visually adjacent secondary lower crossings at `m1=0.996` and `0.997` represented the same smooth event.  It failed before continuation because the first localized crossing was a `-1` event while the second bracket did not contain a `-1` zero.

This failure is evidence against the simple-loop model and motivated the all-event network audit and exact Floquet-vertex search.

## Upper-boundary mechanism

### Coarse upper transition is consistent with a Krein / Hamiltonian--Hopf loss of stability

**Status: INDEPENDENT-STRUCTURAL**

Independent Julia BigFloat canonical-Jacobi integration with Vern9 and GenericSchur passes hard numerical gates:

- periodic closure of order `1e-12`;
- `||M^T J M-J||_inf` of order `1e-21`;
- Hamiltonian linearization defect exactly zero in the implemented BigFloat algebra;
- reciprocal multiplier pairing error of order `1e-25`.

Immediately before the coarse upper transition (published stable row 11), the two nontrivial upper-half-plane elliptic modes have opposite Krein signs.  Immediately after it (published unstable row 12), the four nontrivial multipliers form a reciprocal complex quartet off the unit circle.

Together with the independently derived reduced `Delta=0` crossing, this is strong qualitative evidence for a Krein/Hamiltonian--Hopf mechanism across the coarse cell.

Release label is still withheld until the independently corrected exact critical point is evaluated canonically at BigFloat precision.

## High-precision boundary verification

### Earlier failure from serialized float64 boundary seeds

**Status: INVALIDATED**

The earlier BigFloat run started from a TSV whose shooting parameters did not match the source workflow named in its provenance header.  The resulting initial closure of order unity was a bad-input artifact, not a scientific contradiction.

The seed file has been regenerated from the actual source run.  The independent verifier now anchors to frozen published rows first and treats sub-grid float64 seeds only as optional accelerators after they pass an independent BigFloat closure/event-sign gate.

### Current exact boundary roots

**Status: CANDIDATE / RUNNING**

The corrected independent BigFloat boundary workflow is still the publication gate for the exact `m1=0.8` lower and upper event locations.

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

The mass-space stability diagram is therefore the preimage of this universal spectral-stability domain under the continuation-sheet map `(m1,m2)->(alpha,beta)`.

## Current publication threshold

The project is **not yet entitled to say the stated open problem is solved**.  The shortest remaining route is:

1. locate and independently reproduce the mechanism-switch vertices/folds;
2. complete connected critical-arc continuation, including the secondary lobe/network;
3. finish BigFloat exact boundary roots and canonical evaluation at representative critical points;
4. finish family-sheet rank/bridge adversarial gates;
5. freeze evidence manifests and regenerate the manuscript from them;
6. rerun the current literature novelty audit immediately before release.
