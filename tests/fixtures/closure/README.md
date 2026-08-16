# Synthetic closure-chain fixtures

**These files are not evidence.** Every one of them carries
`"SYNTHETIC_FIXTURE_NOT_EVIDENCE": true` and round numbers no solver would ever
produce. They exist so that `tests/test_close_v1_gates.py` and a human demo can
drive `scripts/close_v1_gates.py` end to end without touching
`research/evidence/`.

They live under `tests/` rather than `/tmp` for one mechanical reason: the
closure runner refuses inputs outside the repository, because
`threebody_atlas.completeness.verify_certificate` can only re-resolve and
re-hash a source path that lands inside the repo. A `/tmp` fixture would be
refused before it proved anything.

| file | plays the part of | shaped so that |
| --- | --- | --- |
| `fold_geometry.json` | `V1_SECONDARY_LEFT_GEOMETRY_*.json` | the float64 fold + reconnection screens pass |
| `fold_bigfloat.json` | `artifacts/secondary-minus-fold-bigfloat.json` | the independent-fold gates in `classify_secondary_left_birth.py` pass |
| `fold_bigfloat_forged.json` | the same artifact, hand-forged | `passed: true` but numbers that miss the frozen gates |
| `al_screen.json` | `V1_AL_POCKET_SCREEN_*.json` | the AL pocket predicate is clean |
| `neck_raster.json` | `artifacts/stability-neck-scan.json` | the neck predicate is clean |
| `neck_raster_merged.json` | the same raster, with a vertical merge | the neck predicate is *not* clean |
| `forged_left_birth_class.json` | a hand-written classification | `passed: true` with a screening evidence level |

The two "forged" files are the reviewer's attack surface: they are here to be
run and shown to fail, not to be believed.
