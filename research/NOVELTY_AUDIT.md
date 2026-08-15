# Novelty and family-identity audit

## Baseline literature

Li, Li & Liao (2021, arXiv:2007.10184) report 135,445 unequal-mass non-hierarchical periodic orbits, with 13,315 linearly stable samples, and show a coarse stable domain in the `(m1,m2)` plane at fixed `m3=1`.

Therefore these are **not** ATLAS novelty claims:
- existence of that orbit set;
- existence of a broad stable mass region;
- the published S/U labels themselves;
- a plot that merely interpolates the published grid.

## Family-identity challenge

Stančević, Vasiljević & Dmitrašinović (SSRN 2023, doi:10.2139/ssrn.4603360) argue that the reported single 135,445-orbit set actually contains two independent sets/families, using distinct scale-invariant angular-momentum/period relations and the existence of two distinct zero-angular-momentum solutions.

ATLAS must not assume a single family until continuation connectivity resolves this.

## Strong v1 question

> What is the connected-family decomposition of the Li--Li--Liao 135,445-orbit data set, and for each resulting continuation family what is the connected linear-stability critical manifold in unequal-mass space, including the Floquet mechanisms and branch connections along that manifold?

This question is stronger than simply refining the published stability domain because it combines:
1. family identity by dynamical continuation rather than topology or regression;
2. continuous critical-set computation rather than a rectangular grid;
3. event-level Floquet/bifurcation classification;
4. independent arbitrary-precision verification.

## Family identity decision rule

Scale-invariant period/angular-momentum plots are diagnostics, not the final definition of a family. ATLAS defines family identity by continuation connectivity under a declared normalization/chart. Two samples are in the same dynamical family only when a verified continuation path connects them without switching solution branches.

The audit therefore proceeds in layers:
1. Recompute energy, angular momentum and scale-invariant diagnostics for every published row.
2. Identify discontinuities/multimodality and candidate branch partitions.
3. Attempt continuation across candidate partition boundaries in both directions with small pseudo-arclength steps.
4. Track orbit shape/topology and monodromy continuously along those attempts.
5. Declare separate families only when continuation evidence supports branch separation; do not use clustering alone as proof.

## Publishable outcomes

Either of the following is scientifically valuable if independently verified:
- confirmation that the 135,445 records decompose into multiple disconnected continuation families, with a corrected family/stability atlas; or
- demonstration that the apparent two-set invariant structure is connected through a continuation path, resolving the 2023 ambiguity.

The stability-boundary paper should report results per connected family, never per assumed catalog label.
