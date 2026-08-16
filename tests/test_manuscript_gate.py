"""The manuscript must not be able to assert more than the evidence supports.

Before this gate existed, a hardcoded sentence in ``paper/main.tex`` claiming the
v1 problem was solved built green while ``research/evidence/V1_CRITICAL_GRAPH.json``
carried ``release_ready`` false: nothing in CI ever read the manuscript sources.
These tests pin the four layers that now stop it.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
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
    errors = GATE.gather_errors(manifest, state, ROOT)
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


def test_every_rendered_manifest_field_is_screened() -> None:
    """Not just statement/method: whatever the renderers typeset gets screened.

    The field list is discovered from the renderers themselves, so a field that
    starts reaching the manuscript cannot quietly escape the screen.
    """
    manifest = deepcopy(load_manifest(MANIFEST))
    collected, errors = GATE.collect_rendered_manifest_prose(manifest, ROOT)
    assert errors == []
    paths = {key.split(":: ")[1] for key in collected}
    generic = {GATE._generic_path(path) for path in paths}
    for field in (
        "decision.summary",
        "gates[].title",
        "gates[].criterion",
        "claims[].statement",
        "claims[].method",
        "claims[].limitations[]",
        "blockers[]",
    ):
        assert field in generic, f"{field} reaches the PDF but is not screened"


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (["decision", "summary"], "OPEN. The 620-cell census is complete; all twelve mixed germs are frozen."),
        (["gates", 1, "title"], "Complete critical graph, now established"),
        (["gates", 1, "criterion"], "The critical graph is complete and every endpoint is classified."),
        (["claims", 0, "limitations", 0], "None: the v1 question is hereby solved."),
        (["blockers", 0], "Nothing remains; the critical graph is complete."),
    ],
)
def test_overclaim_in_any_rendered_manifest_field_is_rejected(path: list, value: str) -> None:
    manifest = deepcopy(load_manifest(MANIFEST))
    node = manifest
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    collected, _ = GATE.collect_rendered_manifest_prose(manifest, ROOT)
    assert GATE.screen_vocabulary(collected, OPEN_STATE)


def test_release_claim_text_is_screened_too() -> None:
    """validate_manifest never reads claim wording, so flipping a record is cheap."""
    manifest = deepcopy(load_manifest(MANIFEST))
    for claim in manifest["claims"]:
        if claim["id"] == "coarse-event-network":
            claim["status"] = "release_claim"
            claim["statement"] = "The critical graph is complete: all 620 cells are frozen."
    collected, _ = GATE.collect_rendered_manifest_prose(manifest, ROOT)
    assert GATE.screen_vocabulary(collected, OPEN_STATE)


def test_gated_branches_are_screened_not_exempt() -> None:
    """A gate you exempt from screening is not a gate.

    The printed branch is screened under today's evidence, and the branch a
    solved state would print is screened under that state, so a claim parked in
    a dormant branch cannot ride the assembler bit into the PDF.
    """
    text = r"\atlasifgraphready{The graph is complete.}{The v1 problem is solved.} Tail."
    fragments = GATE.split_gated(text, OPEN_STATE)
    joined = " ".join(fragment.text for fragment in fragments)
    assert "The graph is complete." in joined and "The v1 problem is solved." in joined
    assert "Tail" in joined
    collected = {"paper/main.tex": GATE._fragments(text, OPEN_STATE)}
    errors = GATE.screen_vocabulary(collected, OPEN_STATE)
    assert any("branch 2" in error for error in errors), errors


def test_solved_branch_is_screened_against_the_solved_world() -> None:
    """Branch 1 prints only when solved, so it is screened as if solved."""
    ok = r"\atlasifsolved{The critical graph is complete.}{Still open.}"
    assert GATE.screen_vocabulary({"f": GATE._fragments(ok, OPEN_STATE)}, OPEN_STATE) == []
    never = r"\atlasifsolved{This paper gives a solution of the general three-body problem.}{x.}"
    assert GATE.screen_vocabulary({"f": GATE._fragments(never, OPEN_STATE)}, OPEN_STATE)


def test_standing_claim_branch_is_screened_under_todays_state() -> None:
    """An authorized claim record licenses its own statement, never solvedness."""
    text = r"\atlasifclaim{one-continuation-family}{The v1 problem is hereby solved.}{}"
    assert GATE.screen_vocabulary({"f": GATE._fragments(text, OPEN_STATE)}, OPEN_STATE)


def test_bound_object_table_rows_are_claim_gated() -> None:
    """Mechanism words in the table must vanish with their release claim."""
    text = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
    ungated = GATE.ungated_text(GATE.strip_comments(text))
    for mechanism in ("opposite-Krein", "0.8023446114", "0.9292391921"):
        assert mechanism not in ungated, f"{mechanism} survives outside a claim gate"


def test_preamble_is_screened_because_it_typesets() -> None:
    """\\title prints on page one and preamble macros carry prose into the body."""
    collected = GATE.collect_preamble_prose(ROOT, OPEN_STATE)
    assert collected, "the preamble must be screened"
    assert GATE.screen_vocabulary(collected, OPEN_STATE) == []
    forged = {"paper/main.tex (preamble)": ["The v1 problem is hereby solved."]}
    assert GATE.screen_vocabulary(forged, OPEN_STATE)


def test_generated_macros_may_not_be_rebound_by_hand(tmp_path: Path) -> None:
    """A preamble \\renewcommand printed the solved branch with release_ready false."""
    shutil.copytree(ROOT / "paper", tmp_path / "paper")
    target = tmp_path / "paper" / "main.tex"
    target.write_text(
        target.read_text(encoding="utf-8").replace(
            r"\begin{document}",
            "\\renewcommand{\\atlasifsolved}[2]{#1}\n\\begin{document}",
        ),
        encoding="utf-8",
    )
    errors = GATE.check_generated_namespace(tmp_path)
    assert any("atlasifsolved" in error and "outside" in error for error in errors)


def test_fallback_block_must_stay_the_weakest_branch(tmp_path: Path) -> None:
    shutil.copytree(ROOT / "paper", tmp_path / "paper")
    target = tmp_path / "paper" / "main.tex"
    target.write_text(
        target.read_text(encoding="utf-8").replace(
            r"\newcommand{\atlasifsolved}[2]{#2}", r"\newcommand{\atlasifsolved}[2]{#1}"
        ),
        encoding="utf-8",
    )
    errors = GATE.check_generated_namespace(tmp_path)
    assert any("weakest branch" in error for error in errors)


def test_committed_manuscript_has_a_clean_namespace() -> None:
    assert GATE.check_generated_namespace(ROOT) == []


def test_lock_rejects_an_unreviewed_sentence() -> None:
    collected = GATE.collect_handwritten(ROOT)
    collected["paper/main.tex"] = list(collected["paper/main.tex"]) + [
        "Every catalog cell now sits on a frozen polyline, so the atlas is finished."
    ]
    errors = GATE.check_lock(collected, evidence_state(load_manifest(MANIFEST), ROOT), ROOT)
    assert any("unreviewed hand-written sentence" in error for error in errors)


def test_lock_covers_preamble_control_lines() -> None:
    """Prose hashing cannot tell [2]{#1} from [2]{#2}; line hashing can."""
    state = evidence_state(load_manifest(MANIFEST), ROOT)
    lines = GATE.collect_control_lines(ROOT)
    assert lines["paper/main.tex"], "the preamble must be locked"
    assert GATE.check_lock({}, state, ROOT, lines) == []
    tampered = dict(lines)
    tampered["paper/main.tex"] = list(lines["paper/main.tex"]) + [
        r"\renewcommand{\atlasifsolved}[2]{#1}"
    ]
    errors = GATE.check_lock({}, state, ROOT, tampered)
    assert any("unreviewed manuscript control line" in error for error in errors)


def test_lock_records_the_reviewed_text_not_only_a_hash() -> None:
    """Blessing a paraphrase must show up as prose in the diff, not as a hash."""
    lock = json.loads((ROOT / "paper" / "CLAIM_LOCK.json").read_text(encoding="utf-8"))
    entries = lock["sentences"]["paper/main.tex"]
    assert isinstance(entries, dict) and entries
    for digest, text in entries.items():
        assert GATE.sentence_hash(text) == digest


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


# --------------------------------------------------------------------------- #
# The manifest summary must describe the graph that is actually committed.
# --------------------------------------------------------------------------- #

WORDS = {0: "no", 1: "one", 2: "two", 6: "six", 7: "seven", 8: "eight", 12: "twelve"}


def test_decision_summary_matches_the_committed_critical_graph() -> None:
    """The summary is typeset verbatim; it must track the artifact, not the story.

    It asserted "the 620-cell census is complete ... all twelve mixed germs are
    frozen" while the committed graph reported twelve *missing* germs, because
    uniform germ validation removed the base-node exemption.  These assertions
    fail the day the counts move, which forces the prose to move with them.
    """
    graph = json.loads(
        (ROOT / "research" / "evidence" / "V1_CRITICAL_GRAPH.json").read_text(encoding="utf-8")
    )
    coverage = graph["root_coverage"]
    summary = load_manifest(MANIFEST)["decision"]["summary"]

    assert str(graph["source_transition_cells"]) in summary
    assert WORDS[coverage["edge_count"]] in summary
    assert "all twelve mixed germs are frozen" not in summary

    missing = coverage["missing_mixed_germs"]
    if missing:
        assert f"{WORDS[len(missing)]} headline mixed germs" in summary
        assert "pending regeneration under uniform germ validation" in summary
    else:
        assert "pending regeneration" not in summary

    unclassified = coverage["unclassified_edge_endpoints"]
    if unclassified:
        assert f"{WORDS[len(unclassified)]} of their endpoints carry no classification" in summary

    components = graph["incidence"]["edge_component_count"]
    if components > 1:
        assert f"{WORDS[components]} disconnected components" in summary

    if coverage["completeness_passed"] is not True:
        assert "no completeness certificate verifies" in summary
    if graph["release_ready"] is not True:
        assert "release_ready false" in summary


def test_old_summary_wording_would_now_fail_the_screen() -> None:
    """The exact sentence that shipped in the PDF is an error today."""
    manifest = deepcopy(load_manifest(MANIFEST))
    manifest["decision"]["summary"] = (
        "OPEN. The 620-cell census is complete and assigned to seven slice-connected "
        "mechanism polylines; all twelve mixed germs are frozen."
    )
    collected, _ = GATE.collect_rendered_manifest_prose(manifest, ROOT)
    errors = GATE.screen_vocabulary(collected, evidence_state(load_manifest(MANIFEST), ROOT))
    assert any("germs-complete" in error for error in errors), errors
    assert any("census-complete" in error for error in errors), errors


# --------------------------------------------------------------------------- #
# The freeze is a human act, and it must look like one.
# --------------------------------------------------------------------------- #

def _freeze(root: Path, *flags: str, env: dict[str, str] | None = None):
    environment = dict(os.environ)
    environment.pop("CI", None)
    environment.pop("GITHUB_ACTIONS", None)
    environment.update(env or {})
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(root), "--freeze", *flags],
        capture_output=True, text=True, env=environment,
    )


def _scratch_tree(tmp_path: Path) -> Path:
    for item in ("paper", "research"):
        shutil.copytree(ROOT / item, tmp_path / item)
    return tmp_path


def test_freeze_refuses_to_bless_new_prose_without_an_explicit_review(tmp_path: Path) -> None:
    root = _scratch_tree(tmp_path)
    target = root / "paper" / "main.tex"
    target.write_text(
        target.read_text(encoding="utf-8").replace(
            r"\textbf{Scope.}",
            "Our atlas now answers the posed question for every mass pair.\n\n\\textbf{Scope.}",
            1,
        ),
        encoding="utf-8",
    )
    result = _freeze(root)
    assert result.returncode == 2
    assert "unreviewed item" in result.stderr
    assert "NEW " in result.stdout
    result = _freeze(root, "--review-new-prose")
    assert result.returncode == 0, result.stderr


def test_freeze_is_refused_inside_ci(tmp_path: Path) -> None:
    """A workflow that could re-bless prose would make the lock a rubber stamp."""
    root = _scratch_tree(tmp_path)
    result = _freeze(root, "--review-new-prose", env={"GITHUB_ACTIONS": "true"})
    assert result.returncode == 2
    assert "Refusing to freeze" in result.stderr


def test_freeze_cannot_bless_a_screened_overclaim(tmp_path: Path) -> None:
    root = _scratch_tree(tmp_path)
    target = root / "paper" / "main.tex"
    target.write_text(
        target.read_text(encoding="utf-8").replace(
            r"\textbf{Scope.}", "The v1 problem is hereby solved.\n\n\\textbf{Scope.}", 1
        ),
        encoding="utf-8",
    )
    result = _freeze(root, "--review-new-prose")
    assert result.returncode == 2
    assert "solvedness" in result.stderr
