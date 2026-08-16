# Three-Body Orbit Atlas

Reproducible research infrastructure for systematically mapping stable periodic-orbit **families** in the non-hierarchical, unequal-mass planar Newtonian three-body problem.

> **Scientific status: OPEN.** Gate A and Gate C passed. Gate B's defined conjuncts are now met: the assembler has flipped `release_ready` on a 13-edge graph (7 catalog S/U polylines + 6 event-sign polylines), 620/620 cells, 16 mixed germs, classified endpoints, frozen completeness, and both committed sign-topology suites report `missing_critical_curve = 0` and `forbidden_component_flip = 0`. Gate D is pending, so SOLVED is forbidden. Three G+ L-path face mismatches remain between m1=1.04 and 1.08 (the unjoined right-hand interior plus_one wall). "Completeness" is frozen but bounded: the neck raster (`research/evidence/V1_NECK_RASTER_2026-08-16.json`, m1 ∈ [0.997, 0.999] × m2 ∈ [0.993, 1.012] at step 1e-4, 4011 samples, all 21 lines separated) covers **0.0253%** of the declared mass domain, and the only other completeness input — the 12-proposal off-grid active-learning screen — is not a sample of the rest of that domain but one pocket at the principal lower transition, m1 ∈ [0.80005, 0.80276] × m2 ∈ [0.75266, 0.75407], **0.002539%** of the domain, self-described as "AI proposals plus float64 screening only; not scientific discovery evidence". Both figures come from `scripts/completeness_scope.py`. This is not a solution of the general three-body problem, and no priority claim is made. See `research/OPEN_PROBLEM.md`, `research/NOVELTY_AUDIT.md`, and `paper/main.tex`.

## Scientific target

The project asks a family-level question rather than an orbit-counting question:

- continue connected periodic-orbit branches through mass/energy/angular-momentum parameter space;
- compute monodromy matrices and Floquet stability with symmetry directions treated correctly;
- attach syzygy/free-group/braid topology without equating topology with family identity;
- localize stability boundaries and bifurcations;
- use ML/active learning to prioritize expensive continuation and verification;
- release every result with machine-readable provenance and a reproducible manuscript.

The first external baseline is Li, Li & Liao, arXiv:2007.10184. Their public supplementary table contains 135,445 unequal-mass non-hierarchical periodic-orbit samples and published S/U stability labels. ATLAS downloads the upstream bytes at run time and rejects them unless the Git blob is exactly `79b2963df43e62201c35690bfc22bec166132427`.

## Evidence ladder

```text
candidate
  -> float64 screening
  -> high-precision closure verification
  -> high-precision variational/Floquet verification
  -> numerically independent reproduction
  -> frozen release claim
```

ML outputs stop at `candidate`. Float64 output stops at `screening`. The manuscript has explicit generated-result gates so prose cannot silently outrun the numerical evidence.

## Current v0.1 evidence status

The first sharded baseline run independently reconstructed the published initial-condition convention and matched the published S/U labels on the first 15 inspected rows, including both sides of both stability transitions at `m1=0.8, m3=1`.

The first continuation/Floquet boundary experiment then narrowed the two published `m2` grid transitions to screening brackets:

```text
lower: [0.75571875000000, 0.75571923828125]
upper: [0.76072412109375, 0.76072460937500]
width: 4.8828125e-7 each
```

At the lower edge the screening trace root crosses `+2`; at the upper edge the reduced stability discriminant changes sign and the trace roots form a small complex pair. These are **candidate critical boundaries**, not release claims.

An intentionally independent mpmath fixed-step RK4 tangent implementation was also exercised. The cheap 64/128-step self-test was numerically unresolved (large state and monodromy step-convergence errors), so its output is rejected as evidence. This is an explicit verifier-design finding: publication-grade high-precision validation needs a higher-order/adaptive arbitrary-precision integrator or CNS-style Taylor method rather than merely increasing decimal precision around a coarse RK4 discretization.

## What is implemented in v0.1

- 12D planar Newtonian dynamics, energy/angular-momentum/COM diagnostics;
- 8D center-of-mass reduced dynamics and tangent equations;
- reduced monodromy and Floquet trace invariants;
- exact free-group reduction/cyclic-conjugacy canonicalization;
- family-chart shooting correction matching the Li-Li-Liao supplementary convention;
- variational shooting Jacobian for efficient mass continuation;
- mass continuation and stability-boundary bisection;
- independent arbitrary-precision state re-integration with step refinement;
- an independent mpmath reduced tangent implementation used as a verifier-development cross-check;
- Extra-Trees active-learning proposal model for continuation warm starts and boundary-focused acquisition;
- versioned candidate/verification schemas and artifact manifests;
- sharded baseline reproduction on GitHub-hosted runners;
- a first boundary-refinement experiment at `m1=0.8, m3=1`;
- pytest/Ruff CI, deterministic science smoke test, LaTeX paper build, and release automation.

## Install

The supported Python runtime is **CPython 3.13**. The Python environment, lockfile, development tools, and package build are managed with `uv`.

```bash
uv python install 3.13
uv sync --locked --group dev
uv run --no-sync pytest
uv run --no-sync ruff check src tests
```

Development-only tools (`pytest`, `ruff`) live in the standardized `dev` dependency group instead of a published package extra. Runtime features remain opt-in extras:

```bash
uv sync --locked --group dev --extra ml
uv sync --locked --group dev --extra accelerated
```

## Deterministic numerical smoke test

```bash
threebody-atlas smoke --floquet --output artifacts/figure-eight.json
```

This is a regression/screening calculation, not publication evidence.

## Reproduce the published baseline

Download the exact upstream file and run a chosen row range:

```bash
mkdir -p data/raw artifacts
curl -L --fail \
  https://raw.githubusercontent.com/sjtu-liao/three-body/main/non-hierarchical-3b-supplementary_data.txt \
  -o data/raw/non-hierarchical-3b-supplementary_data.txt

python scripts/reproduce_baseline.py \
  data/raw/non-hierarchical-3b-supplementary_data.txt \
  --start 1 --stop 20 --floquet \
  --output artifacts/baseline-1-20.jsonl
```

For production runs use the **Baseline reproduction** GitHub Actions workflow, which hash-checks the source and shards work across runners.

## First research experiment

The published `m1=0.8, m3=1` grid brackets two stability changes between `m2=0.755/0.756` and `m2=0.760/0.761`. The **Boundary experiment** workflow re-corrects the periodic orbit while bisecting each bracket and recomputing reduced Floquet invariants. Its output is explicitly `screening-only` until precision escalation and independent variational verification pass.

## AI / active-learning loop

`active_learning.py` learns a warm-start map from mass parameters to the family chart `(x1, v1, v2, T)` and a stability classifier. Ensemble disagreement plus proximity to the estimated stability boundary produces an acquisition score. The **Active learning round** workflow evaluates off-grid mass points, but every AI proposal is immediately handed to deterministic shooting and Floquet screening. A neural/ensemble prediction is never written as an orbit discovery by itself.

## Repository map

```text
src/threebody_atlas/
  dynamics.py                # 12D equations and invariants
  reduced.py                 # COM-reduced 8D dynamics + Floquet criterion
  variational.py             # full-coordinate tangent diagnostics
  liao_family.py             # family shooting chart + variational corrector
  boundary.py                # stability-boundary refinement
  topology.py                # F2 canonicalization
  high_precision.py          # arbitrary-precision state closure check
  high_precision_reduced.py  # independent mpmath reduced tangent verifier
  active_learning.py         # AI proposal/acquisition engine
  metamorphic.py             # oracle-free physics properties + their residuals
  root_audit.py              # re-derive published roots with the shipped dynamics
  schema.py                  # evidence-state records
  provenance.py              # content hashes/release manifests
scripts/
  reproduce_baseline.py
  refine_known_boundaries.py
  precision_crosscheck.py
  active_learning_round.py
  mutation_harness.py        # inject faults, record which detector fires
  audit_published_root_physics.py
  probe_graph_invariants.py
  probe_completeness_certificate.py
experiments/v1.yaml          # frozen experiment contract
research/PROTOCOL.md         # scientific claim rules
paper/                        # claim-gated LaTeX manuscript
.github/workflows/            # CI, compute, paper and release pipelines
```

## Checks that do not depend on the answer

Almost every numerical check here is anchored to an artifact somebody produced,
so it agrees with that artifact even when the artifact is wrong. Two suites are
deliberately not built that way.

**Metamorphic physics properties** (`src/threebody_atlas/metamorphic.py`,
`tests/test_metamorphic_physics.py`) relate two runs of the shipped dynamics to
each other: permutation covariance, translation and Galilean invariance,
rotation and reflection covariance, time reversal, Newtonian similarity
(including invariance of the reduced Floquet multipliers), and covariance of the
tangent map across the 12D Cartesian, 8D reduced and canonical Jacobi
formulations. No published number is used, so nothing they say can be wrong for
the same reason the catalog might be. `python -m threebody_atlas.metamorphic`
prints the residual table.

**Mutation testing of the truth machinery** (`scripts/mutation_harness.py`)
injects a deliberate fault into a temporary copy of the tree and requires some
independent detector to fire — a flipped force sign, a swapped mass
coefficient, a perturbed Hessian term, a shifted gravitational constant, a
dropped or duplicated transition cell, a reversed edge orientation, a
hand-written continuation germ, evidence altered after sealing, a truncated neck
raster, a loosened frozen gate. Each mutation declares what should catch it, and
the run fails if reality disagrees in either direction. Anything that survives
every detector is printed as a gap in the safety net.

Neither suite can make anything pass: they set no release bit, write nothing
into `research/evidence/`, and never touch a numerical gate.

## GitHub Actions as research compute

- **CI**: uv-locked Python 3.13 tests plus a deterministic Floquet smoke artifact.
- **Baseline reproduction**: four-way sharded comparisons against frozen upstream data; also schedulable/manual.
- **Boundary experiment**: refines known grid brackets to generate candidate critical points.
- **Precision cross-check**: exercises the independent arbitrary-precision tangent implementation; convergence diagnostics decide whether an output is admissible.
- **Active learning round**: ranks off-grid points and submits them to deterministic shooting/Floquet screening.
- **Paper**: compiles `paper/main.tex` and uploads the PDF.
- **Release**: on a `v*` tag, tests, builds the manuscript/package, generates hashes, and publishes a GitHub release artifact set.
- **Metamorphic and mutation suite**: prints the metamorphic residual table, then injects every declared fault into throwaway copies of the tree and publishes the kill/gap table to the job summary.

GitHub-hosted runners are useful for reproducible CPU experiments, but expensive sweeps should be explicitly dispatched/sharded rather than triggered by every commit.

## Publication rule

Do not write “new”, “discovered”, “stable”, or a named bifurcation into generated results unless the corresponding frozen record has passed the required evidence state. See `research/PROTOCOL.md`.

The intended first paper is narrower and defensible: **reproduce a documented stable subset, independently validate continuation/Floquet machinery, then extend selected unequal-mass branches to resolve previously unmapped stability boundaries and bifurcation structure with released numerical evidence.**
