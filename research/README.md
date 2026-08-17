# Research protocol and evidence

- `PHYSICS_DOCTRINE.md` is the durable statement of the principles, equations and attack
  patterns this project reasons from. Read it before adding a mechanism label, a gate, or
  a verification lane.
- `OPEN_PROBLEM.md` freezes the v1 scientific question and success/rejection criteria.
- `PROTOCOL.md` defines the evidence ladder and claim discipline.
- `SOLVE_LOOP.md` defines the execution/falsification loop.
- `RESULT_LEDGER.md` is the permanent claim firewall.
- `CLOSURE_STATUS_2026-08-15.md` is the current post-audit checkpoint while the live closure workflows finish.
- `EXECUTION_DAG.md` is the live wave-ordered control graph for remaining closure work.
- `NOVELTY_AUDIT.md` records the literature/precedence audit.
- `BRACKET_CRITERION_BLINDNESS.md` explains why the S/U-label bracket criterion that produced the 620-cell census cannot see critical curves interior to the unstable region, and what replaces it. Read it before proposing to refine the raster.
- `evidence/NUMERIC_SERIALIZATION_SPEC.md` freezes the lossless typed representation,
  precision/rounding metadata, units, strict parsing, and canonical hash semantics for
  release-facing numerical evidence. Its Python/Julia round-trip matrix and hostile
  mutation audit live beside it.
- `graph/V1_GRAPH_SEMANTICS.md` freezes the sheet-aware labeled-multigraph
  invariants, canonical ordering, isomorphism levels, decimal coordinate matching,
  and structured graph-difference classes used to compare independent critical-graph
  reconstructions.
- `workflow/README.md` defines content-addressed scientific campaign identity,
  exactly-once task accounting, fail-closed cache reuse, structured incidents, and
  atomic evidence promotion under injected infrastructure faults.
- `CLAIM_ASSURANCE.md` documents the generated 14-dimensional per-claim assurance matrix, weakest-link report, and fail-closed numerical/theorem readiness policies.
- `provenance/` contains the deterministic environment lock manifest and scientific SBOM; CI rebuilds them and rejects moving third-party Action refs or dependency drift.

Gates: A pass, B pending (13-edge graph assembled; three finite-lattice sweep ends unclassified; plus_one_12 high bound to mixed_principal_right; no `interior_lattice_terminus` nodes; `release_ready` false), C pass, D pending (same-day novelty + `--require-solved`).
The repository must not label the frozen v1 problem `SOLVED` until `--require-solved` passes on a tagged commit.
