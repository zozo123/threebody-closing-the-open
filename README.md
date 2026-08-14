# Three-Body Orbit Atlas

Reproducible research infrastructure for systematically mapping stable periodic-orbit **families** in the non-hierarchical, unequal-mass planar Newtonian three-body problem.

> **Scientific status:** research code. This repository does not claim to have solved the general three-body problem. A numerical statement becomes an ATLAS scientific claim only after the verification and release gates in `research/PROTOCOL.md` pass.

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

## What is implemented in v0.1

- 12D planar Newtonian dynamics, energy/angular-momentum/COM diagnostics;
- 8D center-of-mass reduced dynamics and tangent equations;
- reduced monodromy and Floquet trace invariants;
- exact free-group reduction/cyclic-conjugacy canonicalization;
- family-chart shooting correction matching the Li-Li-Liao supplementary convention;
- mass continuation and stability-boundary bisection;
- independent arbitrary-precision state re-integration with step refinement;
- versioned candidate/verification schemas and artifact manifests;
- sharded baseline reproduction on GitHub-hosted runners;
- a first boundary-refinement experiment at `m1=0.8, m3=1`;
- pytest/Ruff CI, deterministic science smoke test, LaTeX paper build, and release automation.

## Install

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[dev]'
pytest
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

## Repository map

```text
src/threebody_atlas/
  dynamics.py          # 12D equations and invariants
  reduced.py           # COM-reduced 8D dynamics + Floquet criterion
  variational.py       # full-coordinate tangent diagnostics
  liao_family.py       # published-family shooting chart + continuation
  boundary.py          # stability-boundary refinement
  topology.py          # F2 canonicalization
  high_precision.py    # independent arbitrary-precision closure check
  schema.py            # evidence-state records
  provenance.py        # content hashes/release manifests
scripts/
  reproduce_baseline.py
  refine_known_boundaries.py
experiments/v1.yaml    # frozen experiment contract
research/PROTOCOL.md   # scientific claim rules
paper/                  # claim-gated LaTeX manuscript
.github/workflows/      # CI, compute, paper and release pipelines
```

## GitHub Actions as research compute

- **CI**: Python 3.11–3.13 tests plus a deterministic Floquet smoke artifact.
- **Baseline reproduction**: four-way sharded comparisons against frozen upstream data; also schedulable/manual.
- **Boundary experiment**: refines known grid brackets to generate candidate critical points.
- **Paper**: compiles `paper/main.tex` and uploads the PDF.
- **Release**: on a `v*` tag, tests, builds the manuscript/package, generates hashes, and publishes a GitHub release artifact set.

GitHub-hosted runners are useful for reproducible CPU experiments, but expensive sweeps should be explicitly dispatched/sharded rather than triggered by every commit.

## Publication rule

Do not write “new”, “discovered”, “stable”, or a named bifurcation into generated results unless the corresponding frozen record has passed the required evidence state. See `research/PROTOCOL.md`.

The intended first paper is narrower and defensible: **reproduce a documented stable subset, independently validate continuation/Floquet machinery, then extend selected unequal-mass branches to resolve previously unmapped stability boundaries and bifurcation structure with released numerical evidence.**
