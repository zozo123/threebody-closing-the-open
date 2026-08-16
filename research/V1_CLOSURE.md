# V1 closure attack

This file is the execution contract for the stated open problem. It deliberately suppresses attractive side projects until the finite v1 claim is either proved by the evidence protocol or falsified.

## Frozen v1 question

Determine the continuation-connected family decomposition of the Li--Li--Liao unequal-mass non-hierarchical periodic-orbit catalog and, for each connected family, compute the connected planar linear-stability critical manifold in mass space, classify its Floquet mechanisms and branch connections, and independently verify the qualitative critical events.

The target is **not** a solution of the general three-body problem. It is a finite numerical/mathematical statement about one frozen catalog and its continuation/stability geometry.

## The four closure gates

### Gate A -- independent critical-point truth

**Status: PASS for every mechanism label the manuscript is allowed to use.**

A mechanism word (`+1`, `-1`, mixed `(+1,-1)`, Hamiltonian--Hopf/Krein) may appear as an ATLAS result only if:

1. the periodic orbit is independently corrected in Julia BigFloat;
2. the same smooth event is localized without importing Python dynamics;
3. precision/tolerance convergence is recorded;
4. canonical Jacobi monodromy and the physical `E^omega/E` quotient are evaluated at the exact bracket;
5. closure, event, symplectic defect, reciprocal pairing, leakage, and parameter uncertainty are stored;
6. the independent root stays inside the declared tolerance.

Currently bound: principal lower `+1`; principal upper opposite-Krein Hamiltonian--Hopf; three mixed `(alpha,beta)=(4,4)` organizers.

Still Gate-A objects if the paper wants to use them: secondary-left fold, any fourth mixed organizer. Daughter nondegeneracy is a follow-up, not a v1-graph blocker. Until those exist, the paper must not use those labels as established.

A float64/JAX point is a proposal. It is never publication truth by itself.

### Gate B -- complete mechanism-resolved Floquet critical graph on the connected family sheet

**Status: PENDING. This is the remaining theorem.**

The final object is a graph, not a cloud, and it need not be connected just because the family sheet is connected. State `y=(x1,v1,v2,T,m1,m2)`. Mixed organizers are preimages of `(alpha,beta)=(4,4)`. The 620 catalog S/U cells are samples supporting the graph; they are not 620 edges.

Pass only when:

- all 620 catalog S/U cells are localized and each belongs to exactly one mechanism-specific polyline;
- mixed germs come from continuation artifacts, not nearby-root heuristics. The assembler
  enforces this uniformly: every germ, including one attached to a base headline organizer,
  must carry `canonical_bound`/`canonical_bracketed`, a `canonical_distance` inside
  `GERM_ATTACH_DISTANCE`, and closure/event inside the frozen gates, and is rejected outright if
  its own `stopped_reason` records a nonconvergent trace. `research/evidence/V1_MIXED_GERMS_2026-08-15.json`
  failed all twelve of these checks; the `mixed_principal_right` `plus_one` pair additionally
  recorded a pseudo-arclength least-squares failure whose junction trace
  (`V1_JUNCTION_PRINCIPAL_RIGHT_2026-08-15.json`) has zero continuation points. **Remediated.**
  The twelve are now `V1_MIXED_GERMS_{PRINCIPAL_LEFT,SECONDARY_LEFT,PRINCIPAL_RIGHT}_2026-08-16.json`,
  produced by `scripts/trace_canonical_mixed_germs.py` -- the same script that produced the
  numerics-complete `V1_SECONDARY_RIGHT_GERMS_2026-08-16.json`. The 2026-08-15 file is retained as
  history and is no longer an assembler input. The float64 organizer centre those germs are
  launched from is the junction screens' own `direct_mixed_vertex_retry` candidate; two of the
  three had recorded only `RuntimeError: JAX + Diffrax are required` -- a missing dependency, not a
  missing vertex -- and `scripts/retry_direct_mixed_vertex.py` replays that one step with the
  accelerated extra installed. The centre is therefore still produced by a float64 pipeline that
  never reads the BigFloat chart, so the 1e-4 centre/organizer agreement stays an independent
  cross-pipeline test (measured: 1.4e-9, 9.5e-11, 1.8e-7);
- each declared-domain terminus is a distinct exit, not a shared face: two curves that leave the
  box through the same wall at different places are two nodes, not one;
- secondary-left birth is classified as a fold, two-arc alternative, or mixed organizer;
- secondary-right death is classified as a mixed organizer, fold, or declared domain boundary;
- completeness is frozen *and independently re-verifiable*: the assembler re-reads every source
  artifact named in the certificate (`active_learning` and `neck_scan` are both mandatory),
  re-hashes it, and re-derives the AL pocket-screen and neck-raster predicates from the artifact
  itself. A certificate sealed only with a digest over its own content is not evidence, and
  re-sealing a certificate after editing a source artifact does not launder it;
- the lower +1 daughter is classified, with `no_branch_attachment` accepted as a valid close;
- no endpoint is `Newton failed`;
- `research/evidence/V1_CRITICAL_GRAPH.json` reports `release_ready: true`.

#### Running Gate B: one command

Going from harvested CI artifacts to a Gate-B decision used to be a sequence of
undocumented manual steps. It is now one command:

```
scripts/close_v1_gates.py \
  --fold-geometry <path> --fold-geometry-run-id <run> \
  --fold-bigfloat <path> --fold-bigfloat-run-id <run> \
  --al-screen     <path> --al-screen-run-id     <run> \
  --neck-raster   <path> --neck-raster-run-id   <run>
```

It classifies the secondary-left birth, freezes the completeness certificate,
assembles the graph, and prints `release_ready` plus the exact list of false
conjuncts. Exit 0 means release_ready, 2 means an honest open state with the
blockers enumerated, 64 means it refused because an input was missing, and 3
means the chain itself is broken.

The runner orchestrates; it never decides. It writes nothing under
`research/evidence/` except by invoking `classify_secondary_left_birth.py`,
`freeze_completeness_certificate.py` and `assemble_v1_critical_graph.sh`, it
carries no numerical gate of its own, and it takes its answer from the
assembler's `release_ready` bit cross-checked against the assembler's exit
status and against a re-derived conjunct list. A partial input set is a
refusal, not a quieter invocation, and a producing script that refuses stops
the chain instead of falling back to a stale classification.

Every input is recorded in `V1_CLOSURE_PROVENANCE.json` with its sha256 and the
CI run id it came from — supplied as an argument, never guessed. One artifact
may not do two jobs: two roles resolving to the same path, two roles carrying
identical bytes, or one sha256 offered under two different CI run ids are all
refusals, because a closure resting on one measurement wearing two hats is not
a closure and a fabricated run id cannot be checked against CI. The real
closure runs in `.github/workflows/v1-gate-closure.yml`, which takes the four
run ids, fetches the artifacts with `gh run download`, and uploads inputs and
outputs together so the certificate stays re-verifiable.

##### The deciding code is pinned too

Pinning the input bytes pins only half of what produces a verdict; the other
half is the code that reads them. A reviewer demonstrated the gap: a ~20-line
bash shim on `$PYTHON` that forwarded most invocations to the real interpreter
and substituted passing output for `classify_secondary_left_birth.py` produced
`rc=0, release_ready=true` from a forged BigFloat fold, and left no trace in
`V1_CLOSURE_PROVENANCE.json` — the runner resolved its interpreter from the
environment and deliberately did not record which one it used, on the theory
that omitting a machine-varying field kept the ledger byte-reproducible. That
traded auditability for a cosmetic property.

The runner now:

- runs the producing scripts under `sys.executable` — the interpreter the
  operator invoked — and **refuses (exit 64)** if `$PYTHON` names anything else,
  including a multi-word wrapper. Choosing a throwaway venv is still supported
  and is now an explicit act: invoke `close_v1_gates.py` *with* that venv's
  python. The resolved absolute path is exported to
  `assemble_v1_critical_graph.sh`, so the shell cannot inherit a hostile
  `$PYTHON` or start a second unrecorded interpreter of its own;
- **refuses** unless every deciding script — `classify_secondary_left_birth.py`,
  `freeze_completeness_certificate.py`, `assemble_v1_critical_graph.sh` and
  `assemble_critical_graph.py` — hashes identically to its bytes at `git HEAD`.
  A shim that swaps the script instead of the interpreter fails here, and a
  closure run from an uncommitted working copy of a producing script is not a
  closure anybody can re-derive;
- records what it could not prevent. `V1_CLOSURE_PROVENANCE.json` gained an
  `environment` block: the interpreter's absolute path and its own sha256, its
  version banner, whether it sits inside the working tree, the `$PYTHON` that
  was seen, the git HEAD, and both digests of every producing script. The
  interpreter path and hash are printed in the report as well, because a ledger
  nobody opens is not an audit trail.

The ledger is split rather than stripped: `environment` is machine-varying by
construction and is excluded from `evidence_digest`, a sha256 over the rest.
Two honest machines still agree on `evidence_digest` while each discloses its
own environment.

A limit, stated rather than papered over: a program cannot bootstrap trust in
the machinery that runs it. If `sys.executable` is itself hostile, or `.git` has
been rewritten, every check above is executed by the attacker. The claim the
runner can defend is narrower and still worth having — **a substitution that
changes the decision cannot also stay out of the ledger.**

##### What the assembler cannot detect, and where that defence lives

`assemble_critical_graph.load_classification` opens a JSON file and believes its
`evidence_level`. A hand-written classification carrying `passed: true`, an
allowed class and `evidence_level: "independently_reproduced"` therefore yields
a *passed* node. That is not fixable in the assembler: at that layer the
artifact is only bytes, and nothing in them separates a forgery from a real
classification. `tests/test_close_v1_gates.py` asserts that behaviour honestly
instead of the flattering version — an earlier test appeared to show forgeries
being rejected, but it only showed one forgery that had left `evidence_level`
at `screening`.

The defence is in the runner, the only layer that knows where a classification
came from: it deletes the output path, invokes the digest-verified classifier
under the recorded interpreter, deletes the path again if the classifier
refuses, and binds the surviving record in the ledger to the input artifacts
(sha256 + CI run id), the producing script (both digests) and the interpreter.

### Gate C -- family/sheet connectivity

**Status: PASS in the declared catalog domain.**

Family identity is continuation connectivity. The catalog is one continuation-connected component. Certificate: `research/V1_CONNECTIVITY_CERTIFICATE_2026-08-15.md`.

The corrected `(T_si,L_si)` projection is folded and non-injective. Projected two-set structure is not a dynamical split.

A chart singularity is not a moduli-space disconnection.

### Gate D -- release closure

**Status: PENDING.**

Pass only when Gate B is ready, the assembler has flipped `release_ready`, the novelty search is same-day, the manuscript is regenerated from `release_claim` records, and `--require-solved` passes on the tagged commit.

## Mathematical contract

### Stability scope

Unless vertical/spatial variational modes are explicitly added, every use of "stable" in v1 means **linear Floquet stability within the planar problem**. It does not imply spatial stability and does not imply KAM/nonlinear stability.

### Strict periodicity

Quotienting rotation is a mathematical convenience, not permission to confuse a closed shape loop with an inertial periodic orbit. Any future fully rotation-reduced formulation must retain the reconstruction angle and distinguish strict periodicity from relative periodicity.

### Similarity and coordinates

The current Li normalization fixes the Newtonian scale operationally. Before intrinsic-coordinate claims are published, similarity-free quantities must be derived with the correct powers of `G` and mass rather than inferred from raw `(E,L,T)` plots.

### Moduli-space language

Do not publish a formal dimension for the quotient moduli space until the symmetry, first-integral, phase, scaling, and discrete-permutation reductions are written explicitly. For v1 the operational definition is sufficient: a dynamical family is a connected component under branch-preserving continuation after the declared gauge/normalization choices, with suspicious connections repeated in an independent chart.

## What is deliberately deferred

The following are valuable but are **not v1 blockers** unless a core gate forces them:

- Conley--Zehnder/Maslov/Floer bookkeeping;
- Broucke/GIT/B-signature refinements beyond what is needed to classify a critical event;
- KAM/nonlinear stability;
- 3D/vertical stability;
- complex-mass or complex-time continuation;
- large cross-family universality scans;
- AI discovery sweeps;
- number-theory/symbolic-regression searches;
- global collision regularization when no v1 endpoint approaches collision;
- exhaustive shape-physics explanations.

These projects reopen after the finite critical graph and connectivity statement are closed.

## Stop rule

No new research direction enters v1 merely because it is interesting. It enters only if it is necessary to decide one of Gates A--D or to answer a concrete falsification raised by those gates.
