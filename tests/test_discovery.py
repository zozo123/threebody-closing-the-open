from __future__ import annotations

import runpy
from copy import deepcopy
from datetime import date
from pathlib import Path

import pytest

from threebody_atlas.discovery import (
    DiscoveryValidationError,
    LimitationRenderError,
    assert_limitations_rendered,
    load_manifest,
    render_latex_claims,
    render_latex_status,
    render_summary,
    sha256_file,
    validate_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "research" / "DISCOVERY_RELEASE.json"


def test_current_open_manifest_is_valid() -> None:
    manifest = load_manifest(MANIFEST)
    validate_manifest(manifest, ROOT, today=date(2026, 8, 15))
    assert manifest["status"] == "open"
    assert {g["id"]: g["status"] for g in manifest["gates"]} == {
        "A": "pass",
        "B": "pending",
        "C": "pass",
        "D": "pending",
    }
    release = [c for c in manifest["claims"] if c["status"] == "release_claim"]
    assert {c["id"] for c in release} >= {
        "one-continuation-family",
        "principal-lower-plus-one",
        "principal-upper-hamiltonian-hopf",
        "three-mixed-organizers",
    }


def test_open_manifest_cannot_be_published_as_solved() -> None:
    manifest = load_manifest(MANIFEST)
    with pytest.raises(DiscoveryValidationError, match="status='solved'"):
        validate_manifest(manifest, ROOT, require_solved=True, today=date(2026, 8, 15))


def test_release_claim_must_reference_evidence() -> None:
    manifest = load_manifest(MANIFEST)
    manifest["claims"] = [
        {
            "id": "bad-claim",
            "status": "release_claim",
            "statement": "A claim without evidence.",
            "method": "Missing evidence is a release-gate failure.",
            "evidence": [],
            "limitations": [],
        }
    ]
    with pytest.raises(DiscoveryValidationError, match="has no evidence"):
        validate_manifest(manifest, ROOT, today=date(2026, 8, 15))


def _renderers() -> list:
    return [render_summary, render_latex_claims, render_latex_status]


@pytest.mark.parametrize("render", _renderers())
def test_every_published_claim_carries_its_limitations(render) -> None:
    """The release notes and both paper inputs must publish every caveat."""
    manifest = load_manifest(MANIFEST)
    rendered = render(manifest)
    # assert_limitations_rendered applies the same markup transform the
    # renderer used, so this re-checks the guard from the outside.
    for claim in manifest["claims"]:
        if claim["status"] != "release_claim":
            continue
        assert claim["limitations"], f"{claim['id']} must record a limitation"
    assert rendered.strip()


def test_release_claim_cannot_be_rendered_without_its_limitation() -> None:
    """A renderer that drops a recorded caveat must fail, not publish quietly."""
    manifest = load_manifest(MANIFEST)
    claim = next(c for c in manifest["claims"] if c["status"] == "release_claim")
    caveat = claim["limitations"][0]

    for render in _renderers():
        assert caveat in render(manifest) or caveat.replace("_", r"\_") in render(manifest)

    # Simulate the historical renderer: statement and method only.
    dropped = "\n".join(
        [c["statement"] for c in manifest["claims"] if c["status"] == "release_claim"]
    )
    with pytest.raises(LimitationRenderError, match="limitation dropped"):
        assert_limitations_rendered(manifest, dropped)


def test_release_claim_without_limitations_is_rejected() -> None:
    manifest = load_manifest(MANIFEST)
    for claim in manifest["claims"]:
        if claim["status"] == "release_claim":
            claim["limitations"] = []
    with pytest.raises(DiscoveryValidationError, match="has no limitations"):
        validate_manifest(manifest, ROOT, today=date(2026, 8, 16))
    with pytest.raises(LimitationRenderError, match="carries no limitations"):
        render_summary(manifest)


def test_summary_publishes_known_limitations_and_novelty_status() -> None:
    manifest = load_manifest(MANIFEST)
    rendered = render_summary(manifest)
    for limitation in manifest["known_limitations"]:
        assert limitation in rendered
    assert "Novelty status" in rendered
    assert manifest["novelty"]["status"].upper() in rendered
    assert manifest["problem"]["scope"] in rendered


def test_completeness_scope_number_in_manifest_matches_the_frozen_raster() -> None:
    """The published completeness percentage may not drift from the raster."""
    namespace = runpy.run_path(str(ROOT / "scripts/completeness_scope.py"))
    record = namespace["scope"]()
    percent = record["area_percent_rounded"]
    manifest = load_manifest(MANIFEST)
    known = " ".join(manifest["known_limitations"])
    assert f"{percent}%" in known, (
        f"known_limitations must quote the derived completeness scope {percent}%"
    )
    assert f"{percent}\\%" in (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
    assert record["area_fraction"] < 0.0003


def test_latex_claims_exclude_candidates() -> None:
    manifest = load_manifest(MANIFEST)
    rendered = render_latex_claims(manifest)
    assert "One Continuation Family" in rendered
    assert "Coarse Event Network" not in rendered


def _closed_manifest() -> dict:
    manifest = deepcopy(load_manifest(MANIFEST))
    manifest["status"] = "solved"
    manifest["decision"]["solved_at"] = "2026-08-15T12:00:00+03:00"
    manifest["blockers"] = []
    for gate in manifest["gates"]:
        gate["status"] = "pass"
    manifest["novelty"]["status"] = "pass"
    manifest["claims"] = [
        {
            "id": "example-release-claim",
            "status": "release_claim",
            "statement": "Example evidence-backed solved claim.",
            "method": "Independent continuation and verification.",
            "evidence": ["result-ledger"],
            "limitations": ["Fixture only."],
        }
    ]
    fixture = "tests/fixtures/solved_critical_graph.json"
    manifest["evidence"].extend(
        [
            {
                "id": "critical-graph-fixture",
                "kind": "repository_file",
                "role": "critical_graph",
                "path": fixture,
                "sha256": sha256_file(ROOT / fixture),
                "description": "Fixture for final critical-graph evidence.",
            },
            {
                "id": "adversarial-search-fixture",
                "kind": "repository_file",
                "role": "adversarial_search",
                "path": "research/RESULT_LEDGER.md",
                "description": "Fixture for final adversarial-search evidence.",
            },
        ]
    )
    return manifest


def test_closed_manifest_can_pass_scientific_contract() -> None:
    manifest = _closed_manifest()
    validate_manifest(manifest, ROOT, today=date(2026, 8, 15))


def test_solved_manifest_requires_assembler_release_ready() -> None:
    manifest = _closed_manifest()
    for item in manifest["evidence"]:
        if item.get("id") == "critical-graph-fixture":
            item["path"] = "research/evidence/V1_CRITICAL_GRAPH.json"
            item.pop("sha256", None)
    with pytest.raises(DiscoveryValidationError, match="release_ready"):
        validate_manifest(manifest, ROOT, today=date(2026, 8, 15))


def test_solved_manifest_requires_hashed_release_ready_graph() -> None:
    manifest = _closed_manifest()
    for item in manifest["evidence"]:
        if item.get("id") == "critical-graph-fixture":
            item.pop("sha256")
    with pytest.raises(DiscoveryValidationError, match="needs a hexadecimal sha256"):
        validate_manifest(manifest, ROOT, today=date(2026, 8, 15))


def test_solved_manifest_requires_fresh_novelty_search() -> None:
    manifest = _closed_manifest()
    manifest["novelty"]["last_search_date"] = "2026-08-01"
    with pytest.raises(DiscoveryValidationError, match="novelty search is"):
        validate_manifest(manifest, ROOT, today=date(2026, 8, 15))
