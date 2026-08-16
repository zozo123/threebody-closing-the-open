# Scientific environment provenance

`ENVIRONMENT_LOCK_MANIFEST.json` is regenerated deterministically from the
Python and Julia lock inputs, every GitHub Action use, runner selectors, and
known external source pins. `SCIENTIFIC_SBOM.json` is a CycloneDX 1.6 inventory
of every package in the locked Python and release-Julia environments.

Regenerate both files with:

```bash
uv run --no-sync python scripts/build_supply_chain_manifest.py
```

CI uses `--check`; it fails if either committed artifact is stale or if any
third-party action uses a tag or branch instead of a full commit SHA.

Hosted runner images and apt packages are not immutable inputs today. Release
jobs therefore emit an uncommitted `environment-runtime.json` alongside the
release artifacts, recording the actual GitHub runner image identity, Python,
OpenSSL, uv, NumPy, and BLAS/LAPACK configuration. The lock manifest lists this
remaining limitation explicitly rather than treating `ubuntu-latest` as a
reproducible image digest.
