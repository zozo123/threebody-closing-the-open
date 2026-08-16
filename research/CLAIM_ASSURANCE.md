# Claim assurance matrix

`ASSURANCE_DIMENSIONS.json` is the executable policy for per-claim assurance.
`scripts/build_claim_assurance.py` joins that policy to `DISCOVERY_RELEASE.json`,
resolves every cited repository file or Actions artifact to a SHA-256 identity,
and generates:

- `evidence/V1_CLAIM_ASSURANCE_MATRIX.json`, containing all 14 dimensions for
  every registered claim plus headline, edge/node, completeness, numerical-paper,
  theorem-grade, and independence/common-mode views;
- `evidence/V1_WEAKEST_LINK_REPORT.json`, containing the strongest blocking
  state and exact blocking dimensions for each claim.

The six states are deliberately distinct: `pass`, `fail`, `not_applicable`,
`not_run`, `infrastructure_blocked`, and `scientifically_unresolved`. There is no
numeric confidence or aggregate score. A readiness profile is true only when
every dimension it requires is `pass`.

Green state is not editable in either generated artifact. Evaluators derive it
from discovery-manifest validation, immutable claim evidence, a boolean verdict
inside a content-hashed artifact, novelty state, or the absence of recorded
blockers. Editing a parent artifact, its registered digest, the policy, the
claim registry, novelty state, or blockers makes `--check` fail until the matrix
is regenerated.

The solved release path replays both generated artifacts and then consumes the
`numerical_paper` conjunction. `theorem_grade` remains separate and additionally
requires a rigorous certificate; numerical evidence cannot be averaged into a
theorem.

Regenerate and verify with:

```bash
uv run --no-sync python scripts/build_claim_assurance.py
uv run --no-sync python scripts/build_claim_assurance.py --check
```

Register new calibration, conditioning, mutation, platform-systematics, lineage,
or rigorous-certificate results as repository-file evidence with the role named
in the policy and a top-level boolean `passed`. Opaque or missing verdicts become
`infrastructure_blocked`; absence remains `not_run`; a false bound verdict is
`fail`.
