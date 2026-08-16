"""The manuscript must not be able to assert more than the evidence supports.

Before this gate existed, a hardcoded sentence in ``paper/main.tex`` claiming the
v1 problem was solved built green while ``research/evidence/V1_CRITICAL_GRAPH.json``
carried ``release_ready`` false: nothing in CI ever read the manuscript sources.
These tests pin the three layers that now stop it.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
from copy import deepcopy
from pathlib import Path

import pytest

from threebody_atlas.discovery import (
    _completeness_sentence,
    evidence_state,
    load_manifest,
    render_latex_claims,
    render_latex_macros,
    render_latex_status,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_manuscript_claims.py"
SPEC = importlib.util.spec_from_file_location("check_manuscript_claims", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)

MANIFEST = ROOT / "research" / "DISCOVERY_RELEASE.json"

OPEN_STATE = {
    "status": "open",
    "gates": {"A": "pass", "B": "pending", "C": "pass", "D": "pending"},
    "release_claims": ["one-continuation-family"],
    "release_ready": False,
    "completeness": None,
    "solved": False,
}
SOLVED_STATE = {
    "status": "solved",
    "gates": {"A": "pass", "B": "pass", "C": "pass", "D": "pass"},
    "release_claims": ["one-continuation-family"],
    "release_ready": True,
    "completeness": {"passed": True},
    "solved": True,
}


def test_committed_manuscript_passes_the_gate() -> None:
    """The tree as committed must be clean, or the gate is decoration."""
    manifest = load_manifest(MANIFEST)
    state = evidence_state(manifest, ROOT)
    collected = GATE.collect_handwritten(ROOT)
    errors = (
        GATE.check_generated_channel(manifest, ROOT)
        + GATE.check_scope_tags(ROOT)
        + GATE.screen_vocabulary(collected, state)
        + GATE.screen_vocabulary(GATE.collect_release_claims(manifest), state)
        + GATE.check_lock(collected, state, ROOT)
    )
    assert errors == [], "\n".join(errors)


def test_evidence_state_tracks_the_assembler_bit() -> None:
    state = evidence_state(load_manifest(MANIFEST), ROOT)
    graph = json.loads(
        (ROOT / "research" / "evidence" / "V1_CRITICAL_GRAPH.json").read_text(encoding="utf-8")
    )
    assert state["release_ready"] is (graph["release_ready"] is True)
    assert state["solved"] is (state["status"] == "solved" and state["release_ready"])


def test_generated_files_are_byte_identical_to_the_generator() -> None:
    """The only way a claim reaches the manuscript is through the generator."""
    manifest = load_manifest(MANIFEST)
    generated = ROOT / "paper" / "generated"
    assert (generated / "discovery-release.tex").read_text(encoding="utf-8") == render_latex_status(
        manifest
    )
    assert (generated / "discoveries.tex").read_text(encoding="utf-8") == render_latex_claims(
        manifest
    )
    assert (generated / "claim-macros.tex").read_text(encoding="utf-8") == render_latex_macros(
        manifest, ROOT
    )


def test_generated_directory_holds_only_generated_files() -> None:
    owned = set(GATE.GENERATED_OWNERS)
    present = {path.name for path in (ROOT / "paper" / "generated").iterdir() if path.is_file()}
    assert present <= owned, f"hand-written prose in paper/generated/: {sorted(present - owned)}"


@pytest.mark.parametrize(
    "sentence",
    [
        "The v1 open problem is solved.",
        "We have solved the v1 question.",
        "The critical graph is complete.",
        "We computed the connected stability-boundary manifold.",
        "All 620 catalog cells are frozen as graph edges.",
        "This paper gives a solution of the general three-body problem.",
        "Completeness has been established for the declared mass box.",
    ],
)
def test_solvedness_vocabulary_is_rejected_while_the_graph_is_open(sentence: str) -> None:
    assert GATE.screen_vocabulary({"paper/main.tex": [sentence]}, OPEN_STATE)


@pytest.mark.parametrize(
    "sentence",
    [
        "The general Newtonian three-body problem is not being solved.",
        "Until those close, a sentence claiming the critical graph is complete would be false.",
        "Gate B requires that the critical graph is complete.",
        "None of this is a solution of the general Newtonian three-body problem.",
        "The critical graph is unfinished and the status remains open.",
    ],
)
def test_disclaimers_and_rules_are_not_flagged(sentence: str) -> None:
    """A grep alone fires on honest prose; negation and scoping analysis must not."""
    assert GATE.screen_vocabulary({"paper/main.tex": [sentence]}, OPEN_STATE) == []


def test_solvedness_becomes_admissible_only_when_the_evidence_says_so() -> None:
    sentence = "The critical graph is complete."
    assert GATE.screen_vocabulary({"paper/main.tex": [sentence]}, OPEN_STATE)
    assert GATE.screen_vocabulary({"paper/main.tex": [sentence]}, SOLVED_STATE) == []


def test_general_three_body_solution_is_never_admissible() -> None:
    """Even a fully closed v1 does not license a claim about the general problem."""
    sentence = "This is a solution of the general three-body problem."
    assert GATE.screen_vocabulary({"paper/main.tex": [sentence]}, SOLVED_STATE)


def test_release_claim_text_is_screened_too() -> None:
    """validate_manifest never reads claim wording, so flipping a record is cheap."""
    manifest = deepcopy(load_manifest(MANIFEST))
    for claim in manifest["claims"]:
        if claim["id"] == "coarse-event-network":
            claim["status"] = "release_claim"
            claim["statement"] = "The critical graph is complete: all 620 cells are frozen."
    errors = GATE.screen_vocabulary(GATE.collect_release_claims(manifest), OPEN_STATE)
    assert errors


def test_generator_gated_regions_are_exempt_from_the_screen() -> None:
    text = r"\atlasifsolved{The problem is solved.}{The problem is open.} Tail."
    assert "solved" not in GATE.drop_gated_regions(text)
    assert "Tail" in GATE.drop_gated_regions(text)
    claim = r"\atlasifclaim{some-claim}{Mixed L & $1$ & $2$ & mechanism \\}{} Tail."
    assert "Mixed L" not in GATE.drop_gated_regions(claim)


def test_bound_object_table_rows_are_claim_gated() -> None:
    """Mechanism words in the table must vanish with their release claim."""
    text = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
    ungated = GATE.drop_gated_regions(GATE.strip_comments(text))
    for mechanism in ("opposite-Krein", "0.8023446114", "0.9292391921"):
        assert mechanism not in ungated, f"{mechanism} survives outside a claim gate"


def test_lock_rejects_an_unreviewed_sentence() -> None:
    collected = GATE.collect_handwritten(ROOT)
    collected["paper/main.tex"] = list(collected["paper/main.tex"]) + [
        "Every catalog cell now sits on a frozen polyline, so the atlas is finished."
    ]
    errors = GATE.check_lock(collected, evidence_state(load_manifest(MANIFEST), ROOT), ROOT)
    assert any("unreviewed hand-written sentence" in error for error in errors)


def test_lock_catches_paraphrases_the_lexicon_has_never_seen() -> None:
    """Layer 2 is the backstop for wording Layer 3 cannot anticipate."""
    novel = "Our atlas now answers the posed question in full generality for every mass pair."
    state = evidence_state(load_manifest(MANIFEST), ROOT)
    assert GATE.screen_vocabulary({"paper/main.tex": [novel]}, state) == []
    collected = GATE.collect_handwritten(ROOT)
    collected["paper/main.tex"] = list(collected["paper/main.tex"]) + [novel]
    assert GATE.check_lock(collected, state, ROOT)


def test_freeze_is_refused_when_the_evidence_weakens() -> None:
    old = {
        "status": "solved",
        "release_ready": True,
        "gates": {"A": "pass", "B": "pass"},
        "release_claims": ["one-continuation-family", "three-mixed-organizers"],
        "completeness_frozen": True,
    }
    new = {
        "status": "open",
        "release_ready": False,
        "gates": {"A": "pass", "B": "pending"},
        "release_claims": ["one-continuation-family"],
        "completeness_frozen": False,
    }
    weaker = GATE.fingerprint_weakened(old, new)
    assert any("release_ready" in item for item in weaker)
    assert any("gate B" in item for item in weaker)
    assert any("three-mixed-organizers" in item for item in weaker)
    assert any("completeness" in item for item in weaker)
    assert GATE.fingerprint_weakened(new, old) == []


def test_missing_scope_statement_fails(tmp_path: Path) -> None:
    shutil.copytree(ROOT / "paper", tmp_path / "paper")
    target = tmp_path / "paper" / "main.tex"
    target.write_text(
        target.read_text(encoding="utf-8").replace("% ATLAS-SCOPE: linear-floquet-only", ""),
        encoding="utf-8",
    )
    errors = GATE.check_scope_tags(tmp_path)
    assert any("linear-floquet-only" in error for error in errors)


def test_hand_edited_generated_claims_are_rejected(tmp_path: Path) -> None:
    """This is the exact state the repository was committed in."""
    shutil.copytree(ROOT / "research", tmp_path / "research")
    shutil.copytree(ROOT / "paper", tmp_path / "paper")
    forged = tmp_path / "paper" / "generated" / "discoveries.tex"
    forged.write_text(
        "\\paragraph{Solved.} The v1 critical graph is complete.\n", encoding="utf-8"
    )
    errors = GATE.check_generated_channel(load_manifest(MANIFEST), tmp_path)
    assert any("does not match generator output" in error for error in errors)


def test_macro_file_forces_the_open_branch() -> None:
    """With release_ready false the generated conditionals select the weak branch."""
    macros = render_latex_macros(load_manifest(MANIFEST), ROOT)
    assert r"\newcommand{\atlasifsolved}[2]{#2}" in macros
    assert r"\newcommand{\atlasifgraphready}[2]{#2}" in macros
    assert r"\newcommand{\atlasreleaseready}{false}" in macros
    assert "makes no completeness claim" in macros


def test_macro_file_flips_only_when_both_facts_hold() -> None:
    manifest = deepcopy(load_manifest(MANIFEST))
    manifest["status"] = "solved"
    # Manifest alone must not flip the prose: the assembler bit is still false.
    assert r"\newcommand{\atlasifsolved}[2]{#2}" in render_latex_macros(manifest, ROOT)


def test_completeness_sentence_names_the_sub_box() -> None:
    """A passed certificate must still say how far it reaches, and no further."""
    certificate = {
        "schema": "atlas.v1.completeness-certificate/1",
        "passed": True,
        "domain": {"neck": {"m1": [0.997, 0.999], "m2": [0.993, 1.006], "step": 0.0001}},
        "resolution": 0.0001,
        "active_learning": {"attempted": 12},
    }
    text = _completeness_sentence(certificate)
    assert "bounded sense" in text
    assert "sub-box" in text
    assert "[0.997,0.999]" in text and "[0.993,1.006]" in text
    assert "claims no completeness" in text


def test_absent_completeness_certificate_claims_nothing() -> None:
    text = _completeness_sentence(None)
    assert "no completeness claim" in text
    assert "not excluded" in text
