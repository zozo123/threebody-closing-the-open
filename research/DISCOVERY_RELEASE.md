# Discovery release contract

`research/DISCOVERY_RELEASE.json` is the machine-readable decision record for the frozen ATLAS v1 open problem. It makes every scientific release answer four questions without relying on memory or release prose:

1. **What problem did we claim to solve?** The frozen question, scope, baseline, and success criterion are explicit.
2. **What did we actually discover?** Only entries with `status: release_claim` are authorized scientific claims.
3. **How was each claim obtained?** Every release claim records a method and evidence IDs.
4. **Why is the problem considered solved?** Gates A--D, blockers, novelty freeze, paper, and hashes must close together.

## Status model

- `open`: the project is still attacking the frozen problem. The manifest is valid, but a solved release is forbidden.
- `solved`: all release gates pass, blockers are empty, the novelty audit is fresh, at least one claim is promoted to `release_claim`, and every required evidence role exists.
- `falsified`: the frozen target or a proposed result failed in a way that changes the scientific outcome. Record that outcome instead of redefining success after seeing the data.

Changing `status` to `solved` is intentionally insufficient. The validator also requires:

- Gate A: independent arbitrary-precision/canonical truth for every headline critical mechanism;
- Gate B: a closed, mechanism-resolved critical graph rather than disconnected screening points;
- Gate C: family/sheet connectivity that survives rank, reverse-continuation, and chart-independence attacks;
- Gate D: frozen evidence, adversarial search, paper inputs, hashes, and release-date novelty audit;
- no remaining blockers;
- at least one evidence-backed `release_claim`;
- explicit known limitations;
- a novelty search no older than the configured release window;
- required evidence roles, including the critical graph, independent verification, family connectivity, adversarial search, and built paper.

## Local / CI use

Validate the current state without claiming success:

```bash
python scripts/build_discovery_release.py --validate-only
```

Build an auditable preview dossier while the problem is still open:

```bash
python scripts/build_discovery_release.py \
  --output-dir artifacts/discovery-dossier \
  --emit-paper-status paper/generated/discovery-release.tex
```

Enforce the solved gate:

```bash
python scripts/build_discovery_release.py --validate-only --require-solved
```

That last command must fail until the open problem is genuinely release-ready.

## GitHub Actions

### Discovery gate

`.github/workflows/discovery-gate.yml` runs on changes to the research contract, paper, discovery validator, and release workflows. It runs lint/tests, validates the manifest, writes a human-readable GitHub job summary, and uploads a frozen status-preview dossier.

The gate may be green while `status` is `open`; green means the decision record is internally valid, not that the scientific problem is solved. The job summary states the scientific status explicitly.

### Paper

`.github/workflows/paper.yml` validates the discovery contract, generates `paper/generated/discovery-release.tex`, and then compiles the manuscript. The PDF therefore carries the same OPEN/SOLVED gate state as the machine-readable manifest used for that build.

### Solved discovery release

`.github/workflows/discovery-release.yml` is the hard publication path. A `solved-v*` tag triggers tests, manuscript generation, the final solved gate, Python package build, dossier freeze, checksums, and GitHub release publication.

A manual dispatch runs the same solved-release build without publishing a GitHub release. It is useful as a final release-candidate check.

## Release dossier

The generated dossier contains:

- the original `DISCOVERY_RELEASE.json`;
- a normalized `discovery.json` with commit/run metadata and SHA-256 for every included repository file;
- `DISCOVERY_SUMMARY.md`, answering what was solved, how, evidence, remaining blockers, limitations, and novelty status;
- a frozen source/evidence snapshot under `source/`;
- generated release artifacts such as the manuscript under `generated/`;
- `SHA256SUMS` for the dossier itself.

External GitHub Actions evidence is not copied blindly. The manifest records its immutable run ID, artifact ID, and artifact ZIP SHA-256 so the release claim points to a specific retained computation.

## Promoting the final result

When the numerical work closes the problem:

1. update `RESULT_LEDGER.md` with the final evidence state and remove superseded interpretations;
2. add the final critical graph, independent verification, family-connectivity, and adversarial-search evidence to `DISCOVERY_RELEASE.json`;
3. promote only defensible statements to `release_claim`, with method, evidence IDs, and limitations;
4. clear blockers only when their falsification tests are actually closed;
5. rerun and record the release-date novelty audit;
6. set Gates A--D to `pass` and `status` to `solved`;
7. build the paper from the manifest and run `--require-solved`;
8. create a `solved-v*` tag only on the exact commit that passed the gate.

The release is therefore not merely a version number. It is the frozen answer to: **what open problem was solved, what the result is, how it was obtained, what evidence supports it, what was rejected along the way, and what limitations remain.**
