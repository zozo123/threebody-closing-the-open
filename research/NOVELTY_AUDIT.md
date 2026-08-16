# Novelty and family-identity audit

Last targeted external refresh: **2026-08-16**.

This file is a working novelty firewall, not a substitute for the final release-date literature search.  Negative search results do not prove novelty; they only define what has and has not been located so far.

## Baseline literature

Li, Li & Liao (2021, Science China Physics, Mechanics & Astronomy 64, 219511; arXiv:2007.10184; doi:10.1007/s11433-020-1624-7) report 135,445 unequal-mass non-hierarchical periodic orbits, with 13,315 linearly stable samples, and show a coarse stable domain in the `(m1,m2)` plane at fixed `m3=1`.

Therefore these are **not** ATLAS novelty claims:

- existence of that orbit set;
- existence of a broad stable mass region;
- the published S/U labels themselves;
- a plot that merely interpolates or modestly refines the published grid.

Li & Liao's 2025 invited review, *A review on periodic orbits of the general planar three-body problem* (Science China Physics, Mechanics & Astronomy 68, 289501; doi:10.1007/s11433-024-2686-6), confirms that general planar periodic-orbit discovery, topology, linear stability, unequal masses, angular momentum, and high-precision computation remain active topics.  It is a bibliography map and a release-date citation-tree starting point; it does not by itself settle the continuation decomposition or critical-event network of the 135,445-orbit catalog.

## Family-identity challenge

Stančević, Vasiljević & Dmitrašinović (SSRN 2023, doi:10.2139/ssrn.4603360), *The Mass-, Period- and Angular Momentum Dependences of the “Moth-I” Family of 3-Body Orbits*, analyze the same 135,445 rows and argue that the reported single set actually consists of two independent sets/families.  Their argument uses mass-corrected scale-invariant period/angular-momentum relations and two zero-angular-momentum solutions.

This remains the sharpest located competing interpretation of the exact catalog.  ATLAS must reproduce their normalization/diagnostic before explaining it, but must **not** adopt projected invariant branches as the definition of a dynamical family.

Current internal evidence already makes this distinction essential: the corrected `(T_si,L_si)` projection is folded/non-injective, so two visible branches in that plane can be real projection structure without being disconnected continuation components.

## 2025--2026 nearby literature found in the refresh

The refresh deliberately searched for work that could collide with the stronger ATLAS thesis, not merely any paper containing "three-body" and "stability".

### Equal-mass stable-orbit searches

Hristov, Hristova & Tanikawa, *An extensive search for stable periodic orbits of the equal-mass zero angular momentum three-body problem* (New Astronomy 125, 102528, 2026; arXiv:2510.22802), report 971 verified linearly stable collisionless equal-mass periodic-orbit initial conditions and four stability regions in a specialized equal-mass zero-angular-momentum domain.

This is important contemporary high-precision stability work, but it is not the same problem as continuation connectivity and the mechanism-resolved critical manifold of the unequal-mass Li--Li--Liao catalog.

### Restricted-problem continuation/bifurcation work

Recent 2025--2026 restricted-three-body papers continue to develop family continuation, mass-ratio dependence, topology, and period-multiplication bifurcations.  These are methodological precedents and should be cited where relevant, but they do not establish the continuation-component decomposition of the general unequal-mass catalog studied here.

Examples located in the refresh include 2026 work on asymmetric periodic-orbit families in the circular restricted three-body problem and on mass-ratio dependence/bifurcations of retrograde CR3BP families.  They reinforce that branch continuation and mechanism-level bifurcation language must be precise; they do not currently appear to pre-empt the v1 claim.

### Same-day final search (2026-08-16)

The release-day search repeated the exact title, arXiv identifier, journal DOI,
catalog size, topology word, and combinations of `unequal mass`, `general planar
three body`, `continuation`, `Floquet`, `critical graph`, `mixed organizer`,
`fold`, and `bifurcation`.  It also followed the publicly exposed citation and
reference entry points of the 2021 catalog paper and the 2025 invited review.

The primary records checked on the release date were:

- Li, Li & Liao's catalog paper (arXiv:2007.10184; doi:10.1007/s11433-020-1624-7),
  which publishes the 135,445 samples and 13,315 stable labels but not the
  continuation/Floquet graph assembled here;
- Li & Liao's 2025 invited review (doi:10.1007/s11433-024-2686-6), whose scope
  includes topology, unequal masses, linear stability, and numerical periodic
  orbits, but which does not report this catalog's mechanism-resolved graph;
- Hristov, Hristova & Tanikawa (arXiv:2510.22802), an equal-mass,
  zero-angular-momentum stability search rather than this unequal-mass sheet;
- Portegies Zwart, Doelman & Sein (arXiv:2601.09843), which studies formation
  and survival of selected braids rather than continuation of the Li--Li--Liao
  sheet;
- Prieur & Robutel (arXiv:2604.00623), on Marchal's inclined co-orbital family
  in a planetary/co-orbital regime; and
- Park & Howell (arXiv:2606.08485), an atlas for averaged, Hill-restricted, and
  circular-restricted models near a smaller primary.

No inspected primary record reported the same finite object: the complete
mechanism-resolved planar Floquet critical graph of the continuation-connected
Li--Li--Liao `bABabaBAba` sheet, with every published S/U cell assigned once,
all ends classified, mixed-node continuation germs, and a frozen bounded
completeness certificate.  This remains a documented negative search result,
not a proof that no unindexed or unpublished result exists.  Accordingly the
manuscript may say "we did not locate a prior construction" but must not use an
unqualified universal-priority claim such as "the first".

## What the targeted 2026 search did **not** locate

As of the 2026-08-16 targeted search, no direct paper was located that simultaneously does all of the following for the Li--Li--Liao 135,445 unequal-mass non-hierarchical catalog:

1. defines family identity by branch-preserving continuation connectivity rather than projected invariants/topology alone;
2. resolves the one-family versus two-set ambiguity on that basis;
3. computes the connected stability critical set as separate `+1`, `-1`, and collision/Krein event branches rather than a coarse S/U mask;
4. locates and verifies the mechanism-switch organizers/branch connections of that critical network;
5. independently reproduces representative critical events at arbitrary precision with canonical symplectic diagnostics.

This is a **search status, not a novelty theorem**.  The search must be rerun immediately before release using citation chaining from the 2025 review, the 2021 baseline paper, the 2023 two-set preprint, and all relevant 2025--2026 papers discovered in the meantime.

## Strong v1 question

> What is the continuation-connected family decomposition of the Li--Li--Liao 135,445-orbit data set, and for each resulting family what is the connected planar linear-stability critical manifold in unequal-mass space, including its physical Floquet mechanisms, organizers, and branch connections?

This is stronger than simply refining the published stability domain because it combines:

1. family identity by dynamical continuation rather than topology/regression;
2. continuous critical-set computation rather than a rectangular grid;
3. event-level physical `Sp(4)` Floquet classification;
4. organizer/branch genealogy rather than independent boundary samples;
5. independent arbitrary-precision verification.

## Family identity decision rule

Scale-invariant period/angular-momentum plots are diagnostics, not the final definition of a family.  ATLAS defines family identity by continuation connectivity under a declared normalization/chart, with suspicious bridges repeated in a less specialized formulation.

The audit therefore proceeds in layers:

1. reproduce the published rows and the competing scale-invariant diagnostic exactly;
2. identify projection folds/multimodality and candidate separations;
3. attempt continuation across those candidate separations in both directions;
4. census local shooting rank and distinguish chart conditioning from physical sheet geometry;
5. repeat the strongest bridge in the generic 8D periodic formulation;
6. use loop lifting only where a surviving singularity/multiplicity makes sheet monodromy a live alternative;
7. declare separate families only when continuation evidence supports branch separation.

## Novelty claims currently allowed versus forbidden

### Allowed only after the evidence gates pass

- a continuation-resolved component decomposition of this catalog;
- a mechanism-specific connected critical graph on each verified component;
- verified physical `+1`, `-1`, and Krein/collision transitions and their organizers;
- a demonstrated explanation of why projected invariant branches do or do not correspond to true continuation families.

### Forbidden / already known / too weak

- "we found 135,445 unequal-mass periodic orbits";
- "13,315 of them are stable";
- "there is a broad stable region";
- "there appear to be two lines in a scale-invariant plot";
- "we refined a boundary point to more digits";
- "we used AI/Julia/JAX/high precision" as novelty by itself.

## Publishable outcomes

Either connectivity outcome remains valuable if independently verified:

- the 135,445 records decompose into multiple disconnected continuation components, giving a corrected family-specific critical atlas; or
- projected two-set structure is shown to belong to one continuation-connected sheet (possibly folded/multivalued in the diagnostic projection), resolving the 2023 ambiguity.

Likewise, either mechanism-switch outcome is valuable:

- the physical sheet passes through one or more exact mixed `(+1,-1)` organizers; or
- the coarse apparent switch resolves into multiple nearby critical zeros / projection branches, correcting the visual network interpretation.

The paper must report what survives these falsification tests, not protect a preferred story.

## Mandatory final freeze

Immediately before release:

1. rerun targeted searches for the exact catalog/title/DOI and all key phrases;
2. inspect citing/cited-by chains for the 2021 baseline, 2023 two-set work, and 2025 review;
3. search 2026+ papers for unequal-mass general-three-body continuation, Floquet critical manifolds, family decomposition, and bifurcation organizers;
4. rewrite every `new`, `first`, `resolve`, and `complete` sentence against that fresh evidence;
5. record the search date and sources in the frozen release manifest.
