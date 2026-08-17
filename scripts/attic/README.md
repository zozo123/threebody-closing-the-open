# Attic

Superseded or dead scripts, moved here 2026-08-17 (issue #212 §7) after verifying zero
references in scripts/, .github/, tests/, src/, paper/. Kept because deleting a
producing script erases the ability to re-derive whatever it once produced.

- trace_critical_components.py — superseded by trace_critical_components_from_published.py.
- refine_mass_slice_brackets.py — superseded by extract_mass_slice_brackets.py +
  refine_known_boundaries.py.
- export_missed_cells_for_julia.py — dead; its artifact is itself archived as duplicate.
- extract_zero_angular_momentum_locus.py — dead one-off (2026-08-14).

NOT moved despite looking dead: organizer_consensus_gate.py (sole consumer of the CAPD
validated-lane output — issue #211 Track E), validate_mass_sensitivities.py (validation
reference for the Julia mass-sensitivity port — Track D1),
build_audit_seeded_bigfloat_seeds.py (Track B pipeline), combine_graph_roots.py and
build_mixed_germs_from_junction.py (still referenced by live workflows and tests).
