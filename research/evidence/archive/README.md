# Archived evidence

Files here are dead or duplicate: superseded by a newer committed artifact, byte- or
field-identical to one, or pre-merge shards fully subsumed by a merged census. Nothing
here is consumed by the canonical invocation, a gate, a test, or a workflow — verified
by reference-grep before each move (2026-08-17, issue #212 §7). They are archived
rather than deleted so a reader of research/evidence/ can still audit the history.

- V1_FLOAT64_MISSED_CELLS_2026-08-15.json — all 158 records are field-for-field
  duplicates of content in the merged hybrid census.
- V1_SECONDARY_LEFT_GEOMETRY_2026-08-15.json — superseded by the 2026-08-16 geometry
  (this one carries passed=false and no fold screen).
- V1_SUPPLEMENTAL_EVENT_SIGN_ROOTS_2026-08-17.json — derived from the redundant -17
  sweep; the canonical supplemental roots are the BIGFLOAT_TIPS revision.
- V1_FULL_DOMAIN_SIGN_SWEEP_2026-08-17.json — 9.12 MB, redundant with the 2026-08-16
  sweep (which has the real CI identity, run 31956195570) and the DENSE sweep.
- V1_MIXED_GERMS_PARTIAL.json — byte-identical duplicate of V1_MIXED_GERMS_2026-08-15.json.
- julia_escalate_2026-08-15/, julia_local_remainder_2026-08-15/ — pre-merge shards,
  every cell present in the merged 158-cell escalation record.
