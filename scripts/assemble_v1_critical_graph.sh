#!/usr/bin/env bash
# Canonical v1 critical-graph assembly invocation.
#
# This file is the single source of truth for WHICH evidence artifacts feed
# scripts/assemble_critical_graph.py.  CI (.github/workflows/critical-graph-assembly.yml)
# and any human regenerating research/evidence/V1_CRITICAL_GRAPH.json must both go
# through here, otherwise the committed graph and the CI-re-derived graph drift
# apart for reasons that have nothing to do with the science.
#
# Usage:
#   scripts/assemble_v1_critical_graph.sh <output-path>
#   PYTHON=python3 scripts/assemble_v1_critical_graph.sh /tmp/graph.json
#
# Exit status is the assembler's own: 0 == release_ready, 2 == assembled but not
# release_ready (a legitimate scientific state, NOT a tooling failure), anything
# else == the assembler genuinely failed.
#
# Artifact choices, and why:
#
#   --roots      V1_HYBRID_CRITICAL_ROOTS_2026-08-15.json
#                The 620/620 float64+Julia-BigFloat hybrid census.  This is the
#                canonical roots file: tests/test_critical_graph.py pins its
#                catalog shape (620 localized roots, 7 catalog polylines,
#                source-cell counts [1, 22, 46, 47, 120, 130, 254]).  Those
#                seven edges are not the complete critical set -- see
#                --supplemental-roots.
#                V1_HYBRID_CRITICAL_ROOTS_PARTIAL.json is byte-identical to it (a
#                leftover rolling name from the same run), and
#                V1_FLOAT64_CRITICAL_CENSUS_2026-08-15.json only carries the 462
#                float64-localized cells, so neither can be the graph's input.
#
#   --supplemental-roots V1_SUPPLEMENTAL_EVENT_SIGN_ROOTS_2026-08-16.json
#                Label-invisible / graph-absent event-sign roots harvested from
#                the 2026-08-16 full-domain sweep (CI run 31956195570).  Catalog
#                S/U cells never sample interior 2->1 / 1->2 curves.  Finite
#                lattice ends stay unclassified unless they bind to a domain
#                face, mixed germ, or classification artifact.  They are never
#                promoted to interior_lattice_terminus nodes.
#
#   --supplemental-roots V1_SUPPLEMENTAL_EVENT_SIGN_ROOTS_BIGFLOAT_TIPS_2026-08-17.json
#                The 2026-08-16 full-domain sweep roots (CI run 31956195570),
#                with the THREE unclassified sweep-component tips replaced by
#                their Julia BigFloat dps=60 re-localizations (cells 10000,
#                10042, 10131; run 32037241295).  Those three float64 records sat
#                5-8e-7 off the zero set in m2 -- 1.1e-6, 1.1e-6 and 4.3e-8 of
#                BigFloat event at the stored point, against a 2e-8 gate -- which
#                is why they failed re-correction and why continuation past them
#                returned event=inf.  It SUPERSEDES
#                V1_SUPPLEMENTAL_EVENT_SIGN_ROOTS_2026-08-16.json, kept as history.
#
#   --endpoint-resolution V1_ENDPOINT_RESOLUTION_BIGFLOAT_TIPS_2026-08-17.json
#                Outward continuations of components 0, 1 and 12 from the
#                BigFloat-corrected roots.  Records each walk's seed_masses and
#                final_masses, so the assembler can check BOTH ends: that the
#                walk began at the terminus it explains, and that it stopped
#                within the organizer reach.  Corrected cell 10131 re-certifies
#                under float64, so plus_one component 12 finally has two
#                certifying roots and its two sides no longer collapse onto one
#                seed; they resolve to DIFFERENT organizers
#                (secondary_right_death and mixed_principal_right).
#                Components 0 low and 1 high remain unresolved: their tips are
#                BigFloat-verified roots that float64 still cannot re-certify.
#
#   --supplemental-roots V1_PLUS_ONE_12_CONTINUATION_ROOT_2026-08-17.json
#                One certified variational step on plus_one sweep component 12
#                that lands inside GERM_ATTACH_DISTANCE of mixed_principal_right.
#                The other three unclassified lattice ends are NOT in this file.
#
#   --left-birth V1_LEFT_BIRTH_CLASS_2026-08-16.json
#                projection_fold, passed, evidence_level independently_reproduced,
#                with edge_endpoint_bindings for cells 392 (minus_one U->S) and
#                393 (minus_one S->U).  Emitted by
#                scripts/classify_secondary_left_birth.py from the float64/JAX
#                geometry screen (CI run 31932398513) plus the independent Julia
#                BigFloat fold verification (CI run 31942940282, 248 min, PASS):
#                  G-      =  3.9396e-18  (target 1e-12)
#                  dG/dm2  = -6.0956e-14  (gate 1e-6)
#                  dG/dm1  = 28.0456      (gate >= 1)
#                  secant curvatures [6.2899, 6.0506], disagreement 0.0380 (gate 0.15)
#                It SUPERSEDES V1_LEFT_BIRTH_CLASS_2026-08-15.json, which was
#                invalidated on audit (wrong_branch_pair) and is kept only as history.
#
#   --right-death V1_SECONDARY_RIGHT_CLASS_2026-08-16.json
#                Independently reproduced physical mixed (+1,-1) organizer, with
#                edge_endpoint_bindings for cells 576 (plus_one U->S end) and
#                577 (minus_one S->U end).
#
#   --daughter   V1_DAUGHTER_CLASS_2026-08-16.json
#                distinct_branch, evidence_level independently_reproduced.
#
#   --germs      V1_MIXED_GERMS_PRINCIPAL_LEFT_2026-08-16.json
#                V1_MIXED_GERMS_SECONDARY_LEFT_2026-08-16.json
#                V1_MIXED_GERMS_PRINCIPAL_RIGHT_2026-08-16.json
#                The twelve headline-organizer germs, four per organizer,
#                produced by scripts/trace_canonical_mixed_germs.py.
#                They REPLACE V1_MIXED_GERMS_2026-08-15.json, which is kept on
#                disk as history but is no longer an assembler input: those
#                twelve rows carried only (mixed_node, event_mode, direction,
#                status, ends_on, masses, stopped_reason).  They had no closure,
#                no event value and no canonical binding, two of them sat
#                outside GERM_ATTACH_DISTANCE entirely, and four of them recorded
#                a nonconvergent stopped_reason, so germ_rejections() rejected
#                all twelve and missing_mixed_germs listed all twelve.
#
#                V1_SECONDARY_RIGHT_GERMS_2026-08-16.json (the 4 germs the newly
#                retained secondary_right_death organizer owes; it is a retained
#                mixed node and may not borrow the headline twelve)
#
#   --completeness V1_COMPLETENESS_CERTIFICATE_2026-08-16.json
#                A sealed atlas.v1.completeness-certificate/2, frozen 2026-08-16
#                from the AL pocket screen plus the WIDENED neck raster
#                (V1_NECK_RASTER_2026-08-16.json).  This replaces the earlier
#                --al-screen input, which deliberately yielded
#                completeness_passed=false because an AL pocket screen is not a
#                sealed certificate.
#
#                The raster is the widened m2 in [0.993, 1.012] scan.  The previous
#                [0.993, 1.006] window reported any_vertical_merge=true, which was a
#                SCAN-WINDOW TRUNCATION ARTIFACT and not a physical lobe merge: the
#                upper stable lobe's lower edge climbs off the top of the old window
#                at m1 >= 0.9987, and a vanished lobe reads as a merge.  Widening
#                past the minus-one U->S wall (m2 = 1.0080934 at m1 = 0.999) recovers
#                it at 1.0065 .. 1.0081.  All 21 lines are now `separated`, with
#                interior_merge = 0 and truncation_undecidable = 0, while the true
#                minimum neck gap is UNCHANGED at 3.0e-4 = 3 grid steps.  Widening
#                the window changed the artifact, not the physics; no gate moved.
#
#                Note the certificate is verified, not merely sealed: the assembler
#                re-reads each declared source and recomputes its sha256, so editing
#                a source after sealing -- or re-sealing over an edited source --
#                fails.  A self-sealed record with no real sources also fails.
#                Note the schema is /2 as of the tamper-evidence work: a /2
#                record is only accepted if the assembler can re-read every
#                source it names, re-hash it to the recorded sha256, and
#                re-derive the AL and neck predicates itself.  Sealing alone
#                proves nothing.
#
# LEFT_BIRTH and COMPLETENESS may be overridden by environment variable.  That
# exists so scripts/close_v1_gates.py can point this pinned invocation at the
# classification and certificate it has just produced from live CI artifacts,
# without keeping a second copy of the evidence list that could drift from this
# one.  Defaults are the committed release configuration; overriding either does
# NOT relax anything, because every gate still runs inside the assembler.
set -euo pipefail

# LEFT_BIRTH and COMPLETENESS may be overridden by environment variable so
# scripts/close_v1_gates.py can point this pinned invocation at the artifacts it
# has just produced from live CI runs, rather than keeping a second evidence
# list that could drift from this one.  Overriding relaxes nothing: every gate
# still runs inside the assembler.
#
# PRINT_INPUTS=1 prints the evidence files this invocation would read, one per
# line, and exits without assembling.  The closure runner uses it to hash every
# assembler input for the provenance ledger without duplicating the list.
LEFT_BIRTH="${LEFT_BIRTH:-research/evidence/V1_LEFT_BIRTH_CLASS_2026-08-16.json}"
COMPLETENESS="${COMPLETENESS:-research/evidence/V1_COMPLETENESS_CERTIFICATE_2026-08-16.json}"

EVIDENCE_ARGS=(
  --roots research/evidence/V1_HYBRID_CRITICAL_ROOTS_2026-08-15.json
  --supplemental-roots research/evidence/V1_SUPPLEMENTAL_EVENT_SIGN_ROOTS_BIGFLOAT_TIPS_2026-08-17.json
  --supplemental-roots research/evidence/V1_PLUS_ONE_12_CONTINUATION_ROOT_2026-08-17.json
  --supplemental-roots research/evidence/V1_CONTINUATION_ARC_ROOTS_2026-08-17.json
  --left-birth "$LEFT_BIRTH"
  --right-death research/evidence/V1_SECONDARY_RIGHT_CLASS_2026-08-16.json
  --daughter research/evidence/V1_DAUGHTER_CLASS_2026-08-16.json
  --germs research/evidence/V1_MIXED_GERMS_PRINCIPAL_LEFT_2026-08-16.json
  --germs research/evidence/V1_MIXED_GERMS_SECONDARY_LEFT_2026-08-16.json
  --germs research/evidence/V1_MIXED_GERMS_PRINCIPAL_RIGHT_2026-08-16.json
  --germs research/evidence/V1_SECONDARY_RIGHT_GERMS_2026-08-16.json
  --completeness "$COMPLETENESS"
  --sign-topology research/evidence/V1_SIGN_TOPOLOGY_AUDIT_2026-08-16.json
  --sign-topology research/evidence/V1_SIGN_TOPOLOGY_CROSSING_2026-08-16.json
  --endpoint-resolution research/evidence/V1_ENDPOINT_RESOLUTION_BIGFLOAT_TIPS_2026-08-17.json
)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ "${PRINT_INPUTS:-0}" == "1" ]]; then
  for index in "${!EVIDENCE_ARGS[@]}"; do
    case "${EVIDENCE_ARGS[$index]}" in
      --*) continue ;;
      *) printf '%s\n' "${EVIDENCE_ARGS[$index]}" ;;
    esac
  done
  exit 0
fi

if [[ $# -ne 1 ]]; then
  echo "usage: $(basename "$0") <output-path>" >&2
  exit 64
fi

# Resolve OUTPUT against the CALLER's cwd -- this script cd's to the repo root
# below, so a bare relative path would otherwise silently land in the repo
# instead of where the caller asked.  Then, if the resolved path is inside the
# repo, express it back as repo-relative: the assembler echoes its --output into
# stdout, the closure runner captures that stdout into its provenance ledger,
# and an absolute path there would make the ledger machine-specific and defeat
# its byte-reproducibility.  Correct destination, reproducible record.
OUTPUT="$1"
case "$OUTPUT" in
  /*) ;;
  *) OUTPUT="$PWD/$OUTPUT" ;;
esac

# Relative evidence paths are load-bearing: the assembler stores str(path) in the
# graph's "evidence" and "source_artifact" fields, so the emitted JSON only
# reproduces byte-for-byte when these stay repo-relative and the cwd is the repo
# root.  That is why this script cd's above, and why OUTPUT is absolutised first.
read -r -a PYTHON_CMD <<<"${PYTHON:-uv run --no-sync python}"

case "$OUTPUT" in
  "$REPO_ROOT"/*) OUTPUT="${OUTPUT#"$REPO_ROOT"/}" ;;
esac

"${PYTHON_CMD[@]}" scripts/assemble_critical_graph.py \
  --output "$OUTPUT" \
  "${EVIDENCE_ARGS[@]}"
