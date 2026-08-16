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
#                canonical roots file: tests/test_critical_graph.py pins its shape
#                (620 localized roots, 7 polyline edges, source-cell counts
#                [1, 22, 46, 47, 120, 130, 254]).
#                V1_HYBRID_CRITICAL_ROOTS_PARTIAL.json is byte-identical to it (a
#                leftover rolling name from the same run), and
#                V1_FLOAT64_CRITICAL_CENSUS_2026-08-15.json only carries the 462
#                float64-localized cells, so neither can be the graph's input.
#
#   --left-birth V1_LEFT_BIRTH_CLASS_2026-08-15.json
#                DELIBERATELY passed even though it is INVALIDATED
#                (class=null, passed=false, evidence_level="invalid",
#                invalidated_reason="wrong_branch_pair").  Passing it cannot make
#                anything pass -- load_classification() requires passed=true, an
#                allowed class, AND an evidence_level in the release set -- so the
#                node stays "unresolved" either way.  What it buys is provenance:
#                the graph records the invalidation artifact and its reason instead
#                of a generic "no artifact supplied" placeholder.  When the live
#                BigFloat secondary-minus-fold verification lands, swap this path
#                for the new classification; do not delete the flag.
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
#   --al-screen  V1_AL_POCKET_SCREEN_2026-08-15.json
#                There is no completeness certificate on disk yet, so the AL pocket
#                screen is the only completeness input available.  It deliberately
#                yields completeness_passed=false: an AL pocket screen is not a
#                sealed atlas.v1.completeness-certificate/2.  When a real
#                certificate exists, replace --al-screen with --completeness.
#                Note the schema is /2 as of the tamper-evidence work: a /2
#                record is only accepted if the assembler can re-read every
#                source it names, re-hash it to the recorded sha256, and
#                re-derive the AL and neck predicates itself.  Sealing alone
#                proves nothing.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $(basename "$0") <output-path>" >&2
  exit 64
fi

OUTPUT="$1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Relative evidence paths are load-bearing: the assembler stores str(path) in the
# graph's "evidence" and "source_artifact" fields, so the emitted JSON only
# reproduces byte-for-byte when these stay repo-relative and the cwd is the repo
# root.  That is why this script cd's above.
read -r -a PYTHON_CMD <<<"${PYTHON:-uv run --no-sync python}"

"${PYTHON_CMD[@]}" scripts/assemble_critical_graph.py \
  --output "$OUTPUT" \
  --roots research/evidence/V1_HYBRID_CRITICAL_ROOTS_2026-08-15.json \
  --left-birth research/evidence/V1_LEFT_BIRTH_CLASS_2026-08-15.json \
  --right-death research/evidence/V1_SECONDARY_RIGHT_CLASS_2026-08-16.json \
  --daughter research/evidence/V1_DAUGHTER_CLASS_2026-08-16.json \
  --germs research/evidence/V1_MIXED_GERMS_PRINCIPAL_LEFT_2026-08-16.json \
  --germs research/evidence/V1_MIXED_GERMS_SECONDARY_LEFT_2026-08-16.json \
  --germs research/evidence/V1_MIXED_GERMS_PRINCIPAL_RIGHT_2026-08-16.json \
  --germs research/evidence/V1_SECONDARY_RIGHT_GERMS_2026-08-16.json \
  --al-screen research/evidence/V1_AL_POCKET_SCREEN_2026-08-15.json
