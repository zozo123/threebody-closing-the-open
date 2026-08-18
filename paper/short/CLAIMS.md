# Claim ledger for `paper/short/main_short.tex`

Every substantive claim in the short letter, the artifact that backs it, and the evidence-ladder
rung it is allowed to be stated at. Written so the letter can be checked against the repository
mechanically. Assembled at commit `5e59ead`; `release_ready` is `false` and the scientific status
is OPEN.

| # | Claim in the letter | Artifact | Rung |
|---|---|---|---|
| 1 | Catalog framing: 135,445 orbits, 13,315 stable, 620 adjacent S/U cells, word `bABabaBAba`, grid 0.001, m1∈[0.8,1.1], m2∈[0.7,1.2], m3=1 | upstream table pinned to Git blob `79b2963df43e62201c35690bfc22bec166132427`; `research/evidence/V1_CRITICAL_GRAPH.json` (`declared_mass_domain`, `topology`, `source_transition_cells`) | input, not a result |
| 2 | Sampled set **supported as** one continuation-connected component; 26 adversarial links crossed bidirectionally; chart independence on exactly 1 of 26; endpoints of the pathological bridge 0.0617 apart | `research/V1_CONNECTIVITY_CERTIFICATE_2026-08-15.json` / `.md` | certified continuation, bounded to 26 links |
| 3 | Corrected (T_si,L_si) projection on this sheet is folded and non-injective: 590 adjacent determinant sign-change edges forming one fold locus; 86 far-separated mass pairs within 1e-4 standardized invariant distance | connectivity certificate + fold-locus artifacts | demonstration on samples, **not** a disproof of the 2023 reading |
| 4 | Lower transition on m1=0.8023446113666945 is a generic physical +1 crossing, m2 bracket width 1.5776460402494633e-9 | `research/evidence/V1_CANONICAL_LOWER_PLUS_ONE_2026-08-15.json` | independently reproduced |
| 5 | Upper transition on m1=0.8022889780964406 is an opposite-Krein Hamiltonian–Hopf collision, m2 bracket width 6.2677772144967e-9; two opposite-Krein upper-half unit modes on the stable flank, none on the unstable flank | `research/evidence/V1_CANONICAL_UPPER_COLLISION_2026-08-15.json` | independently reproduced |
| 6 | Three mixed roots of G+=G-=0 at (0.9292391921,0.8853664625), (0.9967681995,0.9560193602), (1.0495531867,1.1294758073); 60 digits; relative event norm ~1e-14; localized by Newton residual, not a bracket | `research/evidence/V1_MIXED_CANONICAL_{PRINCIPAL_LEFT,SECONDARY_LEFT,PRINCIPAL_RIGHT}_2026-08-15.json` | **high-precision supported** — one rung below #4/#5. The graph's node labels are more generous; the artifacts govern. |
| 7 | Two independent BigFloat programs agree on all five objects to between 8.9e-27 and 5.7e-25 in mass | independent-roots artifacts | cross-program agreement |
| 8 | All 620 cells localized once, none a Newton failure, assigned to seven mechanism polylines: 198 G+, 168 G−, 254 Δ | `research/evidence/V1_CRITICAL_GRAPH.json` (`localized_roots`, `root_coverage`) | mixed (see #9) |
| 9 | 7 of the 13 polylines rest on float64 screening alone; 6 carry BigFloat corroboration; of 620 cells, 462 float64 / 158 BigFloat; the word "certified" appears 0 times in the graph | `V1_CRITICAL_GRAPH.json` per-edge `estimators` | screening for the float64-only majority |
| 10 | Six label-invisible event-sign polylines carry 153 cells in components of 19, 24, 29, 23, 34, 24 | `V1_CRITICAL_GRAPH.json` edges, source `full_domain_event_sign_sweep` | screening / mixed |
| 11 | 24 of 26 edge endpoints attached ⇒ exactly **two** unclassified ends, at (0.892, 0.7530796376143668) and (1.042, 0.8579752021443232); `edge_component_count` 2; three nodes with no incidence | `V1_CRITICAL_GRAPH.json` (`incidence`) | bookkeeping |
| 12 | Worst absolute event 1.989798081858396e-8 vs the frozen 2e-8 gate = 99.4899% occupancy, 0.5101% headroom, over 775 localized roots; 222 roots above 1e-8; 12 above 95% of gate; conditioning reported for 0 of 775 | `V1_CRITICAL_GRAPH.json` (`root_residual_margin`) | bookkeeping |
| 13 | A rotation re-derivation at identical tolerance moves the event past the gate on 345/620 and past 10× the gate on 144/620 — 126/198 G+, 217/254 Δ, 2/168 G− | `research/PHYSICS_DOCTRINE.md` §6.1 | doctrine-mandated disclosure |
| 14 | Closure certificates are written at screening rtol 2e-10 while the event is evaluated at 5e-13; on a seeded 160-root sample recorded closure is optimistic by median 48.6×, with 4/160 above the 1e-7 gate on re-measurement (worst 1.543e-7) | `research/EVENT_GATE_CONDITIONING.md` | limitation |
| 15 | BigFloat escalation of 16 float64-failed canary cells returned 4 as Δ collisions where float64 recorded G+ | `research/evidence/V1_JULIA_HARD_CANARY_2026-08-15.json`; `research/RESULT_LEDGER.md` | limitation — mechanism labels move under precision |
| 16 | Secondary-left birth is **not** a certified nondegenerate fold: transversality real (∂G/∂m1 = 28.045563274511, h-convergent) but the stationarity test converges to −30, not 0 | fold-geometry artifacts | **not claimed** |
| 17 | Secondary-right terminus carries `physical` evidence only, typed `endpoint`; **no fourth mixed organizer is claimed** | `V1_CRITICAL_GRAPH.json` node `secondary_right_death` | **not claimed** |
| 18 | Lower +1 daughter is **not** classified: `scripts/classify_lower_plus_one_daughter.py` has one `return` and hard-codes `class="distinct_branch"`, so it cannot discriminate the alternatives; reconnection excluded only over an m2 window of 5.60e-5 | that script; `V1_DAUGHTER_*` artifacts | **not claimed** |
| 19 | Sign-topology coverage withdrawn and still withdrawn; densest full-domain audit at HEAD: 121 scan lines at m1 step 0.0025, max_gap 0.0009, 273 planned / 229 converged / 44 failed, 25 unrefined curve endpoints | `research/evidence/V1_SIGN_TOPOLOGY_AUDIT_FULLDOMAIN_2026-08-18.json`; commits `6634943`, `84e721b` | limitation |
| 20 | The BigFloat verifier does not enforce the frozen event gate on arm extensions: two recorded SUCCESS rows carry \|event\| 1.15e-7 and 2.27e-7 | `julia/verify_critical_points.jl` | code defect, disclosed |
| 21 | Neck raster: m1∈[0.997,0.999]×m2∈[0.993,1.012], step 1e-4, 4011 samples, 21 lines separated, min resolved unstable gap 3.0e-4 = 0.0253% of the declared area | `research/evidence/V1_NECK_RASTER_2026-08-16.json`; `scripts/completeness_scope.py` | screening |
| 22 | AL pocket: 12 proposals, m1∈[0.80005,0.80276]×m2∈[0.75266,0.75407], area fraction 2.538980920314193e-05 = 0.002539%; self-described "AI proposals plus float64 screening only; not scientific discovery evidence" | `research/evidence/V1_AL_POCKET_SCREEN_2026-08-15.json` | **candidate**, explicitly not discovery evidence |
| 23 | Total searched as a 2-D region: 0.0279% of the declared domain; even-order root pairs and tangencies explicitly not excluded | `research/SEARCH_SCOPE_REGISTRY.json` (`excludes_even_root_pairs=false`, `excludes_tangencies=false`) | limitation |
| 24 | The 2026-08-17 sweep does not invalidate the 2026-08-16 completeness certificate (no source bytes changed), and the certificate lends no support to completeness of the critical *set*; interior cleanliness recorded as unresolved | `research/SEMANTIC_INVALIDATION.md`; `V1_COMPLETENESS_CERTIFICATE_2026-08-16.json` | reasoning from the repo's own rule |
| 25 | Exactly 2 of the assembler's 14 release conjuncts are false: no-unclassified-edge-endpoints and sign-topology-clean; novelty freeze separately pending; `release_ready: false` | `scripts/assemble_critical_graph.py`; `research/DISCOVERY_RELEASE.json` | bookkeeping |
| 26 | Re-running the assembler at HEAD reproduces the committed graph byte for byte and exits nonzero | `scripts/assemble_v1_critical_graph.sh` | verified locally |

## Claims deliberately NOT made

- No theorem of connectivity; no interval-arithmetic existence proof for any root.
- No transversality or nondegeneracy certificate.
- No complete critical set and no complete critical graph.
- No fourth mixed organizer; no fold classification; no daughter classification.
- No novelty claim (`novelty.status` is `pending`, recorded search date 2026-08-15, and the
  code enforces only `max_age_days=7`, so "same-day" is prose rather than a gate).
- No priority claim, and nothing about nonlinear, KAM, or spatial stability.

## Known prose drift elsewhere in the repository

`README.md`, `research/README.md:28` and `paper/main.tex` say **three** unclassified sweep ends;
`research/evidence/V1_CLAIM_ASSURANCE_MATRIX.json` and the graph's own incidence say **two**
(24 of 26 endpoints attached). This letter says two. The same files carry the sentence "the
plus_one component-12 high end is bound to `mixed_principal_right` by a certified variational
step", which the graph's `attachment` field does not record as certified; this letter does not
reproduce that claim.
