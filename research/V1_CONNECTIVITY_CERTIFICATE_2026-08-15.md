# V1 continuation-connectivity certificate — 2026-08-15

## Claim

Within the frozen Li--Li--Liao unequal-mass non-hierarchical catalog and the declared v1 normalization/gauge, the sampled catalog is supported as **one continuation-connected component**.

This is a numerical continuation statement for the declared v1 domain. It is not a theorem about the full planar three-body moduli space outside that domain.

## Evidence chain

1. The full 135,445-row sampled mass-grid adjacency graph is connected, with 270,087 adjacency edges and a 135,444-edge MST.
2. Five macroscopic sampled bottlenecks were crossed bidirectionally without branch hysteresis.
3. A deliberately far-mass / invariant-near-duplicate bridge was crossed in both directions in the Li chart, with terminal normalized chart mismatch of order 1e-13.
4. The same pathological bridge was independently repeated in a generic translation-reduced strict-periodic chart outside the Li collinearity/velocity ansatz, with closure of order 1e-9 and terminal matches of order 1e-9.
5. The shooting-Jacobian audit sampled 647 points, including the full m2=m3=1 zero-angular-momentum spine. After tight re-correction of the 28 lowest-rank samples, the minimum corrected scaled rank ratio was 1.0803738448664921e-05, the maximum corrected closure was 1.8017261738074185e-08, and zero points remained below the fixed 1e-6 suspicion threshold.
6. The globally worst MST chart jumps were attacked with a fixed-gate adaptive 6/12/24/48-substep bidirectional verifier.
7. GitHub Actions run 31877974928 (attempt 2, head ba646c7edb9c39d66a5bd00e82bfc63fe6d33616) completed successfully with all 20 global-edge jobs passing and 20 evidence artifacts retained.
8. The formerly problematic global-rank-3 edge (0.839,0.721,1) <-> (0.838,0.721,1) passed the matrix verifier and also has a separate frozen 12-substep bidirectional reproduction.
9. The last-running global-rank-5 edge (0.861,0.705,1) <-> (0.860,0.705,1) failed at 6 and 12 substeps but passed at 24 substeps in both directions without loosening gates. Its certificate reports endpoint corrected residuals 1.2022875155173181e-09 and 1.1332991175106424e-09, forward terminal match 7.143994597514666e-13, reverse terminal match 2.184621159574459e-12, and `passed: true`.

## Interpretation

No surviving rank-loss, directionality, chart-dependence, or adversarial-MST obstruction supports splitting the frozen catalog into multiple continuation components. The evidence therefore supports the v1 component decomposition:

**one continuation-connected component.**

Body permutations and topology labels remain metadata and do not create additional dynamical components under the declared operational definition.

## Reproducibility pointers

- workflow: `.github/workflows/adversarial-connectivity-edges.yml`
- Actions run: `31877974928`
- run head: `ba646c7edb9c39d66a5bd00e82bfc63fe6d33616`
- artifacts: 20 `connectivity-global-rank-*` bundles
- independent hard-edge screen: `experiments/hard_mst_rank3_12step_screen_2026-08-15.json`
- closure contract: `research/V1_CLOSURE.md`
- live closure dashboard: issue #82
