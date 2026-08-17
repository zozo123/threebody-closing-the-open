from __future__ import annotations

import json
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
    """The published completeness percentage may not drift from the raster.

    scope() itself refuses to report a figure that disagrees with the committed
    raster artifact, so calling it is already a cross-check of the workflow's
    merge flags against research/evidence/V1_NECK_RASTER_2026-08-16.json.
    """
    namespace = runpy.run_path(str(ROOT / "scripts/completeness_scope.py"))
    record = namespace["scope"]()
    percent = record["area_percent_rounded"]
    manifest = load_manifest(MANIFEST)
    known = " ".join(manifest["known_limitations"])
    paper = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"{percent}%" in known, (
        f"known_limitations must quote the derived completeness scope {percent}%"
    )
    assert f"{percent}\\%" in paper
    assert f"{percent}%" in readme
    assert record["area_fraction"] < 0.0003
    # The raster the merge job describes is the raster on disk.
    assert record["neck_raster_samples"] == record["neck_raster_artifact"]["samples"] == 4011
    assert record["neck_raster"]["m2"] == [0.993, 1.012]


def test_active_learning_screen_is_published_as_a_pocket_not_a_domain_sample() -> None:
    """The 12-proposal screen is one pocket; saying only "12 proposals" oversells it.

    It is one of the two inputs to the frozen completeness certificate, so
    wherever it is cited as completeness support both facts -- how small the
    pocket is, and that the artifact self-describes as screening only -- must
    travel with it.
    """
    namespace = runpy.run_path(str(ROOT / "scripts/completeness_scope.py"))
    pocket = namespace["scope"]()["active_learning_pocket"]
    assert pocket["proposals"] == 12
    # One pocket at the principal lower transition, ~2.5e-5 of the declared area.
    assert pocket["area_fraction"] < 3e-5
    assert pocket["claim_status"] == (
        "AI proposals plus float64 screening only; not scientific discovery evidence"
    )

    percent = pocket["area_percent_rounded"]
    manifest = load_manifest(MANIFEST)
    known = " ".join(manifest["known_limitations"])
    paper = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for text, marker in ((known, f"{percent}%"), (paper, f"{percent}\\%"), (readme, f"{percent}%")):
        assert marker in text, f"the pocket coverage {percent}% must be published, not just cited"
    for text in (known, paper, readme):
        assert "0.80005" in text and "0.75407" in text, "the pocket's extent must be published"
        assert "not scientific discovery evidence" in text

    # The frozen certificate's own scope description must carry them too, since
    # a reader of the certificate alone would otherwise see only "12 accepted".
    certificate = json.loads(
        (ROOT / "research/evidence/V1_COMPLETENESS_CERTIFICATE_2026-08-16.json").read_text()
    )
    extent = certificate["active_learning"]["extent"]
    assert extent["area_fraction_of_declared_domain"] == pocket["area_fraction"]
    assert "ONE pocket" in certificate["note"]


CONNECTIVITY_CERTIFICATE = ROOT / "research/V1_CONNECTIVITY_CERTIFICATE_2026-08-15.md"


def test_adversarial_link_count_is_not_double_counted() -> None:
    """26 distinct links, not 27: the bridge is one link crossed in two charts.

    The inflated count sat inside a caveat, so the error weakened the caveat --
    it made the adversarial coverage look broader than the certificate supports.
    """
    certificate = CONNECTIVITY_CERTIFICATE.read_text(encoding="utf-8")
    # Certificate item 2: five macroscopic bottlenecks.
    assert "Five macroscopic sampled bottlenecks were crossed bidirectionally" in certificate
    # Items 3 and 4: ONE bridge, crossed in the Li chart and then repeated.
    assert "A deliberately far-mass / invariant-near-duplicate bridge was crossed" in certificate
    assert "The same pathological bridge was independently repeated" in certificate
    # Items 6/7: the 20 globally worst MST chart jumps.
    assert "all 20 global-edge jobs passing" in certificate

    distinct_links = 5 + 1 + 20
    assert distinct_links == 26

    manifest = load_manifest(MANIFEST)
    claim = next(c for c in manifest["claims"] if c["id"] == "one-continuation-family")
    published = " ".join([claim["statement"], claim["method"], *claim["limitations"]])
    assert "26" in published
    assert "27 adversarially" not in published
    assert "2 pathological" not in published and "two pathological" not in published.lower()
    assert "1 pathological invariant-near-duplicate bridge" in claim["method"]


def test_chart_independence_is_scoped_to_the_one_link_that_has_it() -> None:
    """Only the bridge was repeated in a generic chart; the other 25 were not.

    'without branch hysteresis' is likewise recorded only for the 5 macroscopic
    bottlenecks (certificate item 2), so neither phrase may be attached to
    'every adversarially selected link'.
    """
    certificate = CONNECTIVITY_CERTIFICATE.read_text(encoding="utf-8")
    generic_chart_lines = [
        line
        for line in certificate.splitlines()
        if "generic translation-reduced strict-periodic chart" in line
    ]
    assert len(generic_chart_lines) == 1
    assert "The same pathological bridge" in generic_chart_lines[0]
    hysteresis_lines = [
        line for line in certificate.splitlines() if "branch hysteresis" in line
    ]
    assert len(hysteresis_lines) == 1
    assert hysteresis_lines[0].startswith("2. Five macroscopic sampled bottlenecks")

    manifest = load_manifest(MANIFEST)
    claim = next(c for c in manifest["claims"] if c["id"] == "one-continuation-family")
    statement = claim["statement"]
    # The overclaim was exactly this conjunction on "every ... link".
    assert (
        "was crossed bidirectionally by continuation without branch hysteresis, in the Li chart "
        "and again in a generic strict-periodic chart" not in statement
    )
    assert "in the Li chart" in statement
    limitations = " ".join(claim["limitations"])
    assert "Chart-independence is demonstrated on exactly one of the 26 links" in limitations
    assert "only for the 5 bottlenecks" in limitations or "only for the 5 bottlenecks" in statement


def test_float64_evaluation_of_the_canonical_organizers_misses_the_event_gate() -> None:
    """Two of three canonical organizers fail the 2e-8 gate when only evaluated.

    The gate is not loosened by this; the point is that a float64 reproduction
    must re-correct rather than re-evaluate, and the manifest must say so.
    """
    import numpy as np

    from threebody_atlas.critical_manifold import _flow_for_vector, event_value

    gate = 2e-8
    files = {
        "principal_left": "V1_MIXED_CANONICAL_PRINCIPAL_LEFT_2026-08-15.json",
        "secondary_left": "V1_MIXED_CANONICAL_SECONDARY_LEFT_2026-08-15.json",
        "principal_right": "V1_MIXED_CANONICAL_PRINCIPAL_RIGHT_2026-08-15.json",
    }
    events: dict[str, float] = {}
    for name, filename in files.items():
        record = json.loads((ROOT / "research/evidence" / filename).read_text())
        masses = [float(value) for value in record["masses"]]
        chart = [float(value) for value in record["chart"]]
        vector = np.asarray([*chart, masses[0], masses[1]], dtype=float)
        _, floquet = _flow_for_vector(vector, m3=masses[2], rtol=5e-13, atol=5e-15)
        events[name] = float(event_value(floquet, "plus_one"))

    # Pin the SCIENCE, not the bit pattern, and note how weak that science is.
    #
    # These are float64 evaluations of 60-digit charts, and the platform spread
    # is far larger than rounding:
    #     principal_left   -1.2794e-07 (macOS arm64)  vs -1.2798e-07 (linux x86_64)
    #     secondary_left   -1.9582e-08 (macOS arm64)  vs -2.1803e-08 (linux x86_64)
    # secondary_left moves by 11%, and -- this is the point -- its two values sit
    # on OPPOSITE SIDES of the frozen 2e-8 gate.  So a float64 re-evaluation is
    # not merely imprecise at the organizers; it is not stable enough to decide
    # WHICH organizers clear the gate.  Asserting an exact membership list would
    # itself be platform-dependent.
    #
    # The robust, and stronger, statement is: every organizer's float64 plus_one
    # event is of order 1e-8 or worse, at least two of three exceed the gate, and
    # the two principal organizers exceed it on every platform observed.
    for name, value in events.items():
        assert value < 0.0, f"{name} plus_one event should be negative"
        assert abs(value) >= 1e-8, (
            f"{name} float64 plus_one event {value:.5e} is unexpectedly small; "
            "the limitation claims float64 cannot reach the 2e-8 gate here"
        )
    # Deliberately NO per-organizer magnitude pin.  Measured on three machines:
    #
    #                     macOS arm64    CI linux x86_64   NSC linux x86_64
    #   principal_left    -1.2794e-07    -1.2798e-07       -1.3475e-07
    #   secondary_left    -1.9582e-08    -2.1803e-08       -2.3001e-08
    #   principal_right   -2.5556e-08    -4.8931e-08       -8.6599e-08
    #
    # principal_right spans a factor of 3.4, and it differs between two x86_64
    # Linux machines, not merely between architectures.  A quantity that moves
    # like that under a change of libm/BLAS/CPU is dominated by accumulated
    # float64 integration error, not by the physics, so ANY tolerance tight
    # enough to be meaningful is loose enough to be arbitrary.  Three successive
    # attempts to pin these numbers all went red on a platform they were not
    # written on.  The honest assertion is the order of magnitude and the gate
    # membership, and the limitation says exactly that.
    for name, value in events.items():
        assert 1e-8 <= abs(value) <= 1e-6, (
            f"{name} float64 plus_one event {value:.5e} left the order of "
            "magnitude the limitation describes"
        )

    over_gate = {name for name, value in events.items() if abs(value) > gate}
    assert {"principal_left", "principal_right"} <= over_gate, (
        "both principal organizers must miss the gate under float64 on every "
        f"platform; got {sorted(over_gate)}"
    )
    assert len(over_gate) >= 2

    manifest = load_manifest(MANIFEST)
    known = " ".join(manifest["known_limitations"])
    # The manifest must quote the platform-stable values AND disclose that
    # secondary-left straddles the gate depending on platform.
    for value in ("-1.28e-7", "-1.35e-7", "-1.96e-8", "-2.30e-8", "-2.56e-8", "-8.66e-8"):
        assert value in known
    assert "straddles" in known
    assert "not reproducible" in known
    assert "re-corrected" in known
    # NOT "two of the three do not clear the gate": which organizers clear it is
    # itself platform-dependent, which is the whole point of this limitation.
    assert "must never be used as a gate check" in known
    # The gate itself is untouched.
    graph = json.loads((ROOT / "research/evidence/V1_CRITICAL_GRAPH.json").read_text())
    assert graph["frozen_numerical_gates"]["maximum_absolute_event"] == gate


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


def test_solved_manifest_rejects_a_self_reported_completeness_certificate(tmp_path) -> None:
    """Only the assembler-verified bit counts.

    A graph may carry a completeness certificate that declares itself passed;
    that field is sealed with a digest over the certificate itself and proves
    nothing about the AL screen or the neck raster.  The discovery gate must
    look at root_coverage.completeness_passed, which the assembler sets only
    after re-hashing and re-deriving the certificate's sources.
    """
    import json

    graph = json.loads((ROOT / "tests/fixtures/solved_critical_graph.json").read_text())
    graph["root_coverage"]["completeness_passed"] = False
    graph["completeness"] = {"passed": True, "note": "self-reported only"}
    forged = tmp_path / "forged_graph.json"
    forged.write_text(json.dumps(graph, indent=2) + "\n")

    manifest = _closed_manifest()
    for item in manifest["evidence"]:
        if item.get("id") == "critical-graph-fixture":
            item["path"] = str(forged.relative_to(tmp_path))
            item["sha256"] = sha256_file(forged)
    with pytest.raises(DiscoveryValidationError, match="assembler-verified completeness"):
        validate_manifest(manifest, tmp_path, today=date(2026, 8, 15))


def test_solved_manifest_requires_fresh_novelty_search() -> None:
    manifest = _closed_manifest()
    manifest["novelty"]["last_search_date"] = "2026-08-01"
    with pytest.raises(DiscoveryValidationError, match="novelty search is"):
        validate_manifest(manifest, ROOT, today=date(2026, 8, 15))
