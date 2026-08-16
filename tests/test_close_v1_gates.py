"""Tests for the end-to-end v1 closure chain.

The point of scripts/close_v1_gates.py is that it *cannot* manufacture a pass.
Most of what follows is therefore adversarial: forged artifacts, forged
classifications, missing inputs, and a deliberate desync between the runner's
blocker report and the assembler's own gate.

Every chain test writes into ``artifacts/`` (gitignored) and reads synthetic
fixtures from ``tests/fixtures/closure/``.  Nothing here touches
``research/evidence/``.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/closure"
COMMITTED_GRAPH = ROOT / "research/evidence/V1_CRITICAL_GRAPH.json"


def _runner():
    spec = importlib.util.spec_from_file_location(
        "close_v1_gates", ROOT / "scripts/close_v1_gates.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


runner = _runner()


@pytest.fixture(autouse=True)
def _pin_interpreter(monkeypatch):
    """No ``$PYTHON``: the producing scripts run under the test interpreter.

    This used to set ``PYTHON=sys.executable`` so the assembler shell script
    would not fall back to ``uv run --no-sync python`` (unavailable when the
    local uv is older than the pyproject pin).  The runner now exports that
    absolute path to the shell itself, so the variable is unnecessary here --
    and leaving it set would hide the fact that the runner no longer takes its
    interpreter from the environment.
    """
    monkeypatch.delenv("PYTHON", raising=False)
    monkeypatch.setenv("PYTHONPATH", str(ROOT / "src"))


@pytest.fixture
def evidence_dir(request):
    """A throwaway evidence directory *inside* the repository.

    Inside, not /tmp: the runner refuses out-of-repo paths, and a certificate
    that names an out-of-repo source can never be re-verified.
    """
    path = ROOT / "artifacts" / f"pytest-closure-{request.node.name}"
    shutil.rmtree(path, ignore_errors=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def tmp_copy(request):
    """Copy a fixture to a second in-repo path, so two roles carry equal bytes."""
    created: list[Path] = []

    def copy(source: str, name: str) -> str:
        destination = ROOT / "artifacts" / f"pytest-copy-{request.node.name}" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((FIXTURES / source).read_bytes())
        created.append(destination.parent)
        return str(destination.relative_to(ROOT))

    yield copy
    for path in created:
        shutil.rmtree(path, ignore_errors=True)


def _argv(evidence_dir: Path, **overrides: str) -> list[str]:
    inputs = {
        "fold-geometry": "tests/fixtures/closure/fold_geometry.json",
        "fold-bigfloat": "tests/fixtures/closure/fold_bigfloat.json",
        "al-screen": "tests/fixtures/closure/al_screen.json",
        "neck-raster": "tests/fixtures/closure/neck_raster.json",
    }
    inputs.update({key.replace("_", "-"): value for key, value in overrides.items()})
    argv: list[str] = []
    for index, (flag, value) in enumerate(inputs.items()):
        argv += [f"--{flag}", value, f"--{flag}-run-id", f"9999900000{index}"]
    return argv + ["--evidence-dir", str(evidence_dir.relative_to(ROOT))]


def _without(argv: list[str], flag: str) -> list[str]:
    """Drop a flag and the value that follows it."""
    index = argv.index(flag)
    return argv[:index] + argv[index + 2 :]


# --------------------------------------------------------------------------
# It refuses rather than degrading
# --------------------------------------------------------------------------


def test_refuses_a_missing_input(evidence_dir, capsys):
    argv = _without(_argv(evidence_dir), "--fold-bigfloat")
    assert runner.main(argv) == runner.REFUSED
    assert "--fold-bigfloat is required" in capsys.readouterr().err
    assert not evidence_dir.exists(), "a refusal must not create the evidence directory"


def test_refuses_an_input_that_does_not_exist(evidence_dir, capsys):
    argv = _argv(evidence_dir, fold_bigfloat="artifacts/not-downloaded-yet.json")
    assert runner.main(argv) == runner.REFUSED
    assert "is not a readable file" in capsys.readouterr().err
    assert not evidence_dir.exists()


def test_refuses_a_missing_run_id(evidence_dir, capsys):
    argv = _without(_argv(evidence_dir), "--neck-raster-run-id")
    assert runner.main(argv) == runner.REFUSED
    assert "run-id is required" in capsys.readouterr().err


def test_refuses_an_input_outside_the_repository(evidence_dir, tmp_path, capsys):
    stray = tmp_path / "stability-neck-scan.json"
    stray.write_text(
        (FIXTURES / "neck_raster.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    argv = _argv(evidence_dir, neck_raster=str(stray))
    assert runner.main(argv) == runner.REFUSED
    assert "outside the repository" in capsys.readouterr().err


def test_refuses_an_evidence_dir_outside_the_repository(tmp_path, capsys):
    argv = _argv(ROOT / "artifacts/unused")
    argv[-1] = str(tmp_path / "evidence")
    assert runner.main(argv) == runner.REFUSED
    assert "outside the repository" in capsys.readouterr().err


def test_all_input_problems_are_reported_at_once(evidence_dir, capsys):
    argv = _argv(
        evidence_dir,
        fold_bigfloat="artifacts/missing-a.json",
        neck_raster="artifacts/missing-b.json",
    )
    assert runner.main(argv) == runner.REFUSED
    err = capsys.readouterr().err
    assert "missing-a.json" in err and "missing-b.json" in err


# --------------------------------------------------------------------------
# The chain actually runs
# --------------------------------------------------------------------------


def test_chain_runs_end_to_end_and_is_not_release_ready(evidence_dir, capsys):
    assert runner.main(_argv(evidence_dir)) == runner.NOT_RELEASE_READY
    out = capsys.readouterr().out
    assert "release_ready: False" in out

    ledger = json.loads((evidence_dir / runner.CLOSURE_LEDGER).read_text())
    assert ledger["release_ready"] is False
    blockers = {entry["conjunct"] for entry in ledger["blockers"]}

    # The two synthetic inputs did their job: the left-birth node is resolved
    # and the completeness certificate verified, so neither is a blocker.
    assert "no_unexplained_nodes" not in blockers
    assert "completeness_certificate_verified" not in blockers
    # What remains is the genuine open work, reported exactly.  Until 2026-08-16
    # that was no_missing_mixed_germs; the twelve headline germs were regenerated
    # with real numbers that day, so the sole remaining blocker is the one the
    # sign-topology audit found: seven gate-passing critical curves outside the
    # committed edges.
    assert "sign_topology_clean" in blockers

    graph = json.loads((evidence_dir / runner.CRITICAL_GRAPH).read_text())
    assert graph["release_ready"] is False
    left_birth = next(
        item for item in graph["nodes"] if item["id"] == "secondary_left_birth"
    )
    assert left_birth["passed"] is True
    assert left_birth["evidence_level"] == "independently_reproduced"


def test_ledger_records_sha256_and_run_id_for_every_input(evidence_dir):
    runner.main(_argv(evidence_dir))
    ledger = json.loads((evidence_dir / runner.CLOSURE_LEDGER).read_text())
    roles = {entry["role"]: entry for entry in ledger["inputs"]}
    assert set(roles) == {"fold_geometry", "fold_bigfloat", "al_screen", "neck_raster"}
    for role, entry in roles.items():
        source = ROOT / entry["path"]
        assert source.is_file(), role
        assert entry["sha256"] == runner.sha256_file(source)
        assert entry["ci_run_id"], role
    # Every artifact the assembler read is hashed too, via the shell script's
    # own list rather than a second copy of it.
    assemble = next(
        step for step in ledger["steps"] if step["step"] == "assemble_v1_critical_graph"
    )
    assert assemble["evidence_inputs"]
    for entry in assemble["evidence_inputs"]:
        assert entry["present"] is True
        assert entry["sha256"] == runner.sha256_file(ROOT / entry["path"])


def test_chain_is_idempotent_and_byte_reproducible(evidence_dir):
    assert runner.main(_argv(evidence_dir)) == runner.NOT_RELEASE_READY
    first = {
        path.name: path.read_bytes() for path in sorted(evidence_dir.iterdir())
    }
    assert runner.main(_argv(evidence_dir)) == runner.NOT_RELEASE_READY
    second = {
        path.name: path.read_bytes() for path in sorted(evidence_dir.iterdir())
    }
    assert first == second
    assert set(first) == {
        runner.LEFT_BIRTH_CLASS,
        runner.COMPLETENESS_CERTIFICATE,
        runner.CRITICAL_GRAPH,
        runner.CLOSURE_LEDGER,
    }


def test_machine_state_is_confined_to_the_environment_block(evidence_dir):
    """Machine state is recorded, but only in one clearly separated place.

    The old rule was "no machine state anywhere", which is how the interpreter
    came to be omitted and how a shim on ``$PYTHON`` stayed invisible.  The rule
    now is that machine state lives in ``environment`` and nowhere else, so the
    evidence half of the ledger is still a pure function of the input bytes.
    """
    runner.main(_argv(evidence_dir))
    ledger = json.loads((evidence_dir / runner.CLOSURE_LEDGER).read_text())
    assert ledger["environment"]["interpreter"]["path"]

    stable = {
        key: value
        for key, value in ledger.items()
        if key not in ("environment", "evidence_digest")
    }
    text = json.dumps(stable)
    for forbidden in ("timestamp", "generated_at", "hostname", str(ROOT)):
        assert forbidden not in text, forbidden
    assert ledger["evidence_digest"] == runner.evidence_digest(stable)


def test_the_ledger_records_the_interpreter_that_ran_the_producing_scripts(evidence_dir):
    """The single fact whose absence let a shim forge release_ready=true."""
    runner.main(_argv(evidence_dir))
    ledger = json.loads((evidence_dir / runner.CLOSURE_LEDGER).read_text())
    interpreter = ledger["environment"]["interpreter"]

    resolved = Path(sys.executable).resolve()
    assert interpreter["path"] == str(resolved)
    assert Path(interpreter["path"]).is_absolute()
    assert interpreter["sha256"] == runner.sha256_file(resolved)
    assert interpreter["sys_executable"] == sys.executable
    assert interpreter["version"]
    assert interpreter["inside_working_tree"] in (True, False)


def test_the_ledger_records_both_digests_of_every_producing_script(evidence_dir):
    runner.main(_argv(evidence_dir))
    ledger = json.loads((evidence_dir / runner.CLOSURE_LEDGER).read_text())
    recorded = {entry["path"]: entry for entry in ledger["environment"]["producing_scripts"]}
    assert set(recorded) == set(runner.PRODUCING_SCRIPTS)
    for path, entry in recorded.items():
        assert entry["sha256"] == runner.sha256_file(ROOT / path)
        assert entry["matches_head"] is True
        assert entry["sha256_at_head"] == entry["sha256"]
    assert ledger["environment"]["git_head"]
    # The runner hashes itself too. It cannot verify itself -- nothing can --
    # but an auditor comparing two ledgers can at least see that the
    # orchestrator differed.
    assert ledger["environment"]["runner"]["sha256"] == runner.sha256_file(
        ROOT / "scripts/close_v1_gates.py"
    )


def test_the_printed_report_shows_the_interpreter(evidence_dir, capsys):
    """A ledger nobody opens is not an audit trail."""
    runner.main(_argv(evidence_dir))
    out = capsys.readouterr().out
    assert str(Path(sys.executable).resolve()) in out
    assert runner.sha256_file(Path(sys.executable).resolve()) in out


# --------------------------------------------------------------------------
# The interpreter is not selectable from the environment
# --------------------------------------------------------------------------


SHIM = """#!/usr/bin/env bash
# Reproduction of the reviewer's shim: forward most invocations to the real
# interpreter, but substitute a passing classification for the one script whose
# refusal is the only thing standing between a forged fold and release_ready.
set -uo pipefail
for a in "$@"; do
  case "$a" in
    */classify_secondary_left_birth.py)
      out="$3"
      mkdir -p "$(dirname "$out")"
      cat > "$out" <<'JSON'
{"id": "secondary_left_birth", "kind": "endpoint", "class": "projection_fold",
 "passed": true, "status": "independently_reproduced",
 "evidence_level": "independently_reproduced",
 "note": "forged", "edge_endpoint_bindings": []}
JSON
      exit 0
      ;;
  esac
done
exec @REAL@ "$@"
"""


@pytest.fixture
def shim(request):
    """A ``$PYTHON`` shim that intercepts the classifier, on disk and executable."""
    path = ROOT / "artifacts" / f"pytest-shim-{request.node.name}" / "fakepy"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(SHIM.replace("@REAL@", sys.executable), encoding="utf-8")
    path.chmod(0o755)
    yield path
    shutil.rmtree(path.parent, ignore_errors=True)


def test_a_python_shim_is_refused_outright(evidence_dir, shim, monkeypatch, capsys):
    """THE route that forged a pass: PYTHON pointing at an interception shim.

    It is refused before anything runs, not merely recorded afterwards.  An
    environment variable that chooses which code decides a gate is a
    substitution channel, and the only safe thing to do with it is to stop.
    """
    monkeypatch.setenv("PYTHON", str(shim))
    assert runner.main(_argv(evidence_dir)) == runner.REFUSED
    err = capsys.readouterr().err
    assert "names an interpreter other than the one running this script" in err
    assert str(shim) in err
    assert not evidence_dir.exists(), "a refused run must write nothing at all"


def test_a_python_shim_cannot_forge_a_pass_with_a_forged_fold(
    evidence_dir, shim, monkeypatch, capsys
):
    """The full attack, end to end: forged BigFloat + shim that hides it."""
    monkeypatch.setenv("PYTHON", str(shim))
    argv = _argv(
        evidence_dir,
        fold_bigfloat="tests/fixtures/closure/fold_bigfloat_forged.json",
        neck_raster="tests/fixtures/closure/neck_raster_merged.json",
    )
    assert runner.main(argv) != 0
    assert runner.main(argv) == runner.REFUSED
    assert not (evidence_dir / runner.CRITICAL_GRAPH).exists()


def test_a_multi_word_python_is_refused(evidence_dir, monkeypatch, capsys):
    """``uv run --no-sync python`` is a wrapper, and a wrapper cannot be hashed."""
    monkeypatch.setenv("PYTHON", "uv run --no-sync python")
    assert runner.main(_argv(evidence_dir)) == runner.REFUSED
    assert "names an interpreter other than" in capsys.readouterr().err


def test_a_python_that_agrees_with_sys_executable_is_accepted(
    evidence_dir, monkeypatch
):
    """Refusing is about substitution, not about the variable existing."""
    monkeypatch.setenv("PYTHON", sys.executable)
    assert runner.main(_argv(evidence_dir)) == runner.NOT_RELEASE_READY
    ledger = json.loads((evidence_dir / runner.CLOSURE_LEDGER).read_text())
    # It is still disclosed: an auditor sees what was set, not just that it was
    # tolerated.
    assert ledger["environment"]["interpreter"]["python_env_var"] == sys.executable


def test_the_assembler_shell_cannot_inherit_a_hostile_python(shim, monkeypatch):
    """The wrapper resolves its own interpreter, so the runner overwrites it.

    Otherwise closing the ``$PYTHON`` route in the runner would leave it open
    one process deeper, where ``assemble_critical_graph.py`` actually runs.
    """
    monkeypatch.setenv("PYTHON", str(shim))
    result = runner.run_step(
        [
            sys.executable,
            "-c",
            "import os; print(os.environ['PYTHON'])",
        ]
    )
    assert result.stdout.strip() == sys.executable


# --------------------------------------------------------------------------
# Swapping the script is the same attack one layer down
# --------------------------------------------------------------------------


@pytest.fixture
def swapped_producing_script(request):
    """Overwrite a producing script, then restore it whatever the test does."""
    target = ROOT / "scripts/classify_secondary_left_birth.py"
    original = target.read_bytes()

    def swap(text: str) -> None:
        target.write_text(text, encoding="utf-8")

    yield swap
    target.write_bytes(original)


def test_a_swapped_producing_script_is_refused(
    evidence_dir, swapped_producing_script, capsys
):
    swapped_producing_script(
        "import json, sys\n"
        "from pathlib import Path\n"
        "Path(sys.argv[2]).parent.mkdir(parents=True, exist_ok=True)\n"
        "Path(sys.argv[2]).write_text(json.dumps({\n"
        "    'id': 'secondary_left_birth', 'class': 'projection_fold',\n"
        "    'passed': True, 'evidence_level': 'independently_reproduced'}))\n"
    )
    assert runner.main(_argv(evidence_dir)) == runner.REFUSED
    err = capsys.readouterr().err
    assert "does not match HEAD" in err
    assert "classify_secondary_left_birth.py" in err
    assert not evidence_dir.exists()


def test_every_deciding_script_is_in_the_verified_set():
    """The set must cover the shell wrapper and what the wrapper runs.

    Pinning ``assemble_v1_critical_graph.sh`` while leaving
    ``assemble_critical_graph.py`` unpinned would pin the wrapper around the
    only module allowed to set release_ready.
    """
    assert set(runner.PRODUCING_SCRIPTS) == {
        "scripts/classify_secondary_left_birth.py",
        "scripts/freeze_completeness_certificate.py",
        "scripts/assemble_v1_critical_graph.sh",
        "scripts/assemble_critical_graph.py",
    }
    for relative in runner.PRODUCING_SCRIPTS:
        assert (ROOT / relative).is_file(), relative


def test_a_shadowed_completeness_module_is_refused(
    evidence_dir, request, monkeypatch, capsys
):
    """The freezer's gates live in a module, and PYTHONPATH outranks the repo.

    Pinning freeze_completeness_certificate.py while leaving
    threebody_atlas.completeness replaceable would pin the wrapper and not the
    gates.
    """
    shadow = ROOT / "artifacts" / f"pytest-shadow-{request.node.name}"
    package = shadow / "threebody_atlas"
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "completeness.py").write_text(
        "def build_record(*a, **k):\n"
        "    return {'passed': True, 'note': 'forged'}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PYTHONPATH", f"{shadow}{os.pathsep}{ROOT / 'src'}")
    try:
        assert runner.main(_argv(evidence_dir)) == runner.REFUSED
        err = capsys.readouterr().err
        assert "importable as threebody_atlas.completeness" in err
        assert not evidence_dir.exists()
    finally:
        shutil.rmtree(shadow, ignore_errors=True)


def test_the_committed_completeness_module_is_accepted_and_recorded(evidence_dir):
    runner.main(_argv(evidence_dir))
    ledger = json.loads((evidence_dir / runner.CLOSURE_LEDGER).read_text())
    recorded = ledger["environment"]["importable_completeness_module"]
    assert recorded, "the module that decides the completeness gates must be named"
    for entry in recorded:
        assert entry["matches_head"] is True
    assert ledger["environment"]["pythonpath"] == os.environ.get("PYTHONPATH")


# --------------------------------------------------------------------------
# One artifact may not do two jobs
# --------------------------------------------------------------------------


def test_refuses_the_geometry_bigfloat_pair_resolving_to_one_file(
    evidence_dir, capsys
):
    """The float64 screen may not stand in for its own independent check."""
    argv = _argv(
        evidence_dir, fold_bigfloat="tests/fixtures/closure/fold_geometry.json"
    )
    assert runner.main(argv) == runner.REFUSED
    err = capsys.readouterr().err
    assert "resolve to" in err
    assert "fold_bigfloat" in err and "fold_geometry" in err
    assert not evidence_dir.exists()


def test_refuses_the_geometry_bigfloat_pair_with_identical_bytes(
    evidence_dir, tmp_copy, capsys
):
    copy = tmp_copy("fold_geometry.json", "geometry_relabelled.json")
    argv = _argv(evidence_dir, fold_bigfloat=copy)
    assert runner.main(argv) == runner.REFUSED
    assert "identical bytes" in capsys.readouterr().err


def test_refuses_the_al_screen_neck_raster_pair_resolving_to_one_file(
    evidence_dir, capsys
):
    """The AL pocket screen may not also be the neck raster.

    Both feed the completeness certificate, and the certificate's whole claim
    is that two independent screens agree.
    """
    argv = _argv(
        evidence_dir, al_screen="tests/fixtures/closure/neck_raster.json"
    )
    assert runner.main(argv) == runner.REFUSED
    err = capsys.readouterr().err
    assert "al_screen" in err and "neck_raster" in err
    assert not evidence_dir.exists()


def test_refuses_the_al_screen_neck_raster_pair_with_identical_bytes(
    evidence_dir, tmp_copy, capsys
):
    copy = tmp_copy("neck_raster.json", "neck_relabelled.json")
    argv = _argv(evidence_dir, al_screen=copy)
    assert runner.main(argv) == runner.REFUSED
    assert "identical bytes" in capsys.readouterr().err


def test_refuses_one_sha256_presented_under_two_ci_run_ids():
    """A run id is the auditor's handle back to CI; a fake one breaks the chain."""
    records = [
        {
            "role": "fold_geometry",
            "path": "a.json",
            "sha256": "d" * 64,
            "ci_run_id": "111",
        },
        {
            "role": "fold_bigfloat",
            "path": "b.json",
            "sha256": "d" * 64,
            "ci_run_id": "222",
        },
    ]
    problems = runner.duplicate_input_problems(records)
    assert any("different CI run ids" in problem for problem in problems)
    assert any("111" in problem and "222" in problem for problem in problems)


def test_distinct_artifacts_from_one_ci_run_are_fine():
    """One workflow run legitimately produces several different artifacts."""
    records = [
        {"role": "al_screen", "path": "a.json", "sha256": "a" * 64, "ci_run_id": "7"},
        {"role": "neck_raster", "path": "b.json", "sha256": "b" * 64, "ci_run_id": "7"},
    ]
    assert runner.duplicate_input_problems(records) == []


# --------------------------------------------------------------------------
# It cannot be tricked into a pass
# --------------------------------------------------------------------------


def test_a_forged_bigfloat_halts_the_chain(evidence_dir, capsys):
    """`"passed": true` plus numbers that miss the frozen gates is refused.

    Critically, the chain then *stops*: it does not quietly reassemble with the
    stale invalidated classification or with no --left-birth at all.
    """
    argv = _argv(
        evidence_dir, fold_bigfloat="tests/fixtures/closure/fold_bigfloat_forged.json"
    )
    assert runner.main(argv) == runner.NOT_RELEASE_READY
    out = capsys.readouterr().out
    assert "independent fold fails frozen gates" in out
    assert not (evidence_dir / runner.LEFT_BIRTH_CLASS).exists()
    assert not (evidence_dir / runner.CRITICAL_GRAPH).exists()
    ledger = json.loads((evidence_dir / runner.CLOSURE_LEDGER).read_text())
    assert ledger["release_ready"] is False
    assert ledger["halted_because"]


def test_a_stale_classification_cannot_be_left_in_place(evidence_dir, capsys):
    """Pre-planting a passing classification at the output path does not help."""
    evidence_dir.mkdir(parents=True, exist_ok=True)
    planted = json.loads((FIXTURES / "forged_left_birth_class.json").read_text())
    planted["evidence_level"] = "independently_reproduced"
    (evidence_dir / runner.LEFT_BIRTH_CLASS).write_text(json.dumps(planted, indent=2))

    argv = _argv(
        evidence_dir, fold_bigfloat="tests/fixtures/closure/fold_bigfloat_forged.json"
    )
    assert runner.main(argv) == runner.NOT_RELEASE_READY
    assert not (evidence_dir / runner.LEFT_BIRTH_CLASS).exists(), (
        "the runner must delete a planted classification the classifier refused to produce"
    )


def _assembler():
    spec = importlib.util.spec_from_file_location(
        "assemble_critical_graph", ROOT / "scripts/assemble_critical_graph.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_a_careless_forgery_does_not_survive_the_assembler():
    """`"passed": true` with a *screening* evidence_level is still unresolved.

    load_classification() requires passed=true AND an allowed class AND an
    evidence_level in {independently_reproduced, physical, continuation,
    definition}.  This forgery satisfies the first two and fails the third.

    Read the next test before concluding anything from this one: the third
    condition is the forger's own typo, not a defence.
    """
    assembler = _assembler()
    forged = FIXTURES / "forged_left_birth_class.json"
    payload = json.loads(forged.read_text())
    assert payload["passed"] is True
    assert payload["class"] in assembler.LEFT_BIRTH_CLASSES
    assert payload["evidence_level"] not in assembler.RELEASE_EVIDENCE_LEVELS

    node = assembler.load_classification(
        forged,
        node_id="secondary_left_birth",
        default_kind="endpoint",
        allowed=assembler.LEFT_BIRTH_CLASSES,
        missing_note="unused",
    )
    assert node["passed"] is False
    assert node["status"] == "unresolved"
    # The forged claim is recorded, but only as the artifact's own unverified
    # assertion -- never as the node's verdict.
    assert node["screening_passed"] is True


def test_the_assembler_cannot_detect_a_competent_forgery():
    """A forgery that sets evidence_level correctly DOES produce a passed node.

    This is the honest statement of a limit, and it corrects a claim this file
    used to make.  The previous test was read as "hand-written classifications
    do not survive the assembler".  They do, as soon as the forger changes one
    string.  ``load_classification`` opens a JSON file and believes what it
    says; there is nothing in the bytes that distinguishes this artifact from
    one the classifier actually produced, so no amount of work inside
    ``assemble_critical_graph.py`` can tell them apart.

    Asserting the real behaviour here rather than the flattering one is the
    point: the assembler is not the layer that can defend this, and pretending
    it is would leave the actual defence untested.  The two tests that follow
    are the defence, in scripts/close_v1_gates.py, which is the only layer that
    knows where a classification came from.
    """
    assembler = _assembler()
    forged = FIXTURES / "forged_left_birth_class_release_level.json"
    payload = json.loads(forged.read_text())
    assert payload["passed"] is True
    assert payload["class"] in assembler.LEFT_BIRTH_CLASSES
    assert payload["evidence_level"] in assembler.RELEASE_EVIDENCE_LEVELS

    node = assembler.load_classification(
        forged,
        node_id="secondary_left_birth",
        default_kind="endpoint",
        allowed=assembler.LEFT_BIRTH_CLASSES,
        missing_note="unused",
    )
    assert node["passed"] is True
    assert node["status"] == "independently_reproduced"


def test_a_competent_forgery_planted_at_the_output_path_is_destroyed(
    evidence_dir, capsys
):
    """The defence, layer one: the runner owns the output path.

    The forgery from the previous test, planted exactly where the assembler
    would read it, with a BigFloat artifact the classifier refuses.  The runner
    deletes the path before invoking the classifier and deletes it again when
    the classifier refuses, so the assembler never sees the forgery and the
    chain halts instead of assembling around it.
    """
    evidence_dir.mkdir(parents=True, exist_ok=True)
    target = evidence_dir / runner.LEFT_BIRTH_CLASS
    target.write_bytes(
        (FIXTURES / "forged_left_birth_class_release_level.json").read_bytes()
    )

    argv = _argv(
        evidence_dir, fold_bigfloat="tests/fixtures/closure/fold_bigfloat_forged.json"
    )
    assert runner.main(argv) == runner.NOT_RELEASE_READY
    assert not target.exists(), "the planted forgery survived the classifier's refusal"
    assert not (evidence_dir / runner.CRITICAL_GRAPH).exists()

    ledger = json.loads((evidence_dir / runner.CLOSURE_LEDGER).read_text())
    assert ledger["release_ready"] is False
    assert ledger["halted_because"]


def test_a_competent_forgery_planted_before_an_honest_run_is_overwritten(
    evidence_dir,
):
    """The defence, layer two: what the assembler reads is what ran just now.

    Same forgery, but the fold artifacts are the ones the classifier accepts.
    The classification the graph is built from must be the classifier's output
    for *these* inputs, byte for byte -- not the planted file, and not a merge
    of the two.
    """
    evidence_dir.mkdir(parents=True, exist_ok=True)
    target = evidence_dir / runner.LEFT_BIRTH_CLASS
    target.write_bytes(
        (FIXTURES / "forged_left_birth_class_release_level.json").read_bytes()
    )

    assert runner.main(_argv(evidence_dir)) == runner.NOT_RELEASE_READY
    produced = json.loads(target.read_text())
    assert produced["estimator"] != "HAND-WRITTEN-CLAIM"
    assert "SYNTHETIC_FIXTURE_NOT_EVIDENCE" not in produced
    # ...and the ledger ties that classification to the code and the interpreter
    # that produced it, which is the record an auditor needs to go further.
    ledger = json.loads((evidence_dir / runner.CLOSURE_LEDGER).read_text())
    step = next(
        entry
        for entry in ledger["steps"]
        if entry["step"] == "classify_secondary_left_birth"
    )
    assert step["produced"]["sha256"] == runner.sha256_file(target)
    assert ledger["environment"]["interpreter"]["sha256"]
    assert all(
        entry["matches_head"] for entry in ledger["environment"]["producing_scripts"]
    )


def test_a_merged_neck_raster_blocks_completeness(evidence_dir):
    argv = _argv(
        evidence_dir, neck_raster="tests/fixtures/closure/neck_raster_merged.json"
    )
    assert runner.main(argv) == runner.NOT_RELEASE_READY
    ledger = json.loads((evidence_dir / runner.CLOSURE_LEDGER).read_text())
    blockers = {entry["conjunct"] for entry in ledger["blockers"]}
    assert "completeness_certificate_verified" in blockers
    certificate = json.loads((evidence_dir / runner.COMPLETENESS_CERTIFICATE).read_text())
    assert certificate["passed"] is False


def test_the_runner_never_encodes_a_numerical_gate():
    """The orchestrator must not carry gate constants of its own.

    If a threshold ever appears here, the runner has started deciding gates
    instead of orchestrating the scripts that own them.
    """
    source = (ROOT / "scripts/close_v1_gates.py").read_text()
    body = source.split('"""', 2)[2]  # skip the module docstring, which cites them
    for gate in ("2e-8", "1e-7", "2e-08", "1e-07"):
        assert gate not in body, f"{gate} must not be re-implemented in the runner"


# --------------------------------------------------------------------------
# The blocker report cannot drift away from the assembler
# --------------------------------------------------------------------------


def test_conjuncts_agree_with_the_committed_graph():
    graph = json.loads(COMMITTED_GRAPH.read_text())
    conjuncts = runner.release_conjuncts(graph)
    all_true = all(entry["satisfied"] for entry in conjuncts)
    assert all_true == (graph["release_ready"] is True)


def test_conjunct_names_are_unique_and_documented():
    graph = json.loads(COMMITTED_GRAPH.read_text())
    conjuncts = runner.release_conjuncts(graph)
    names = [entry["conjunct"] for entry in conjuncts]
    assert len(names) == len(set(names))
    assert all(entry["explanation"] for entry in conjuncts)


def test_decide_refuses_when_the_conjuncts_disagree_with_release_ready():
    """A drifted blocker list is a tooling failure, never a quiet mis-report."""
    graph = json.loads(COMMITTED_GRAPH.read_text())
    graph["release_ready"] = True
    with pytest.raises(runner.ToolingFailure) as excinfo:
        runner.decide(graph, assembler_exit=0)
    assert "re-derived conjunct list disagrees" in str(excinfo.value)


def test_decide_refuses_when_the_assembler_exit_disagrees():
    graph = json.loads(COMMITTED_GRAPH.read_text())
    assert graph["release_ready"] is False
    with pytest.raises(runner.ToolingFailure) as excinfo:
        runner.decide(graph, assembler_exit=0)
    assert "disagrees with release_ready" in str(excinfo.value)


def test_decide_accepts_the_honest_not_release_ready_state():
    graph = json.loads(COMMITTED_GRAPH.read_text())
    release_ready, blockers = runner.decide(graph, assembler_exit=2)
    assert release_ready is False
    assert blockers
    assert {entry["conjunct"] for entry in blockers} <= {
        entry["conjunct"] for entry in runner.release_conjuncts(graph)
    }


def test_release_ready_requires_every_conjunct():
    """Only an all-true conjunct list may be reported as release_ready."""
    graph = json.loads(COMMITTED_GRAPH.read_text())
    conjuncts = runner.release_conjuncts(graph)
    assert not all(entry["satisfied"] for entry in conjuncts)
    assert graph["release_ready"] is False


# --------------------------------------------------------------------------
# The pinned assembler invocation stays the single source of truth
# --------------------------------------------------------------------------


def test_print_inputs_lists_the_assembler_evidence_set():
    result = runner.run_step(
        [str(ROOT / "scripts/assemble_v1_critical_graph.sh")],
        env={"PRINT_INPUTS": "1"},
    )
    assert result.returncode == 0
    listed = [line for line in result.stdout.splitlines() if line.strip()]
    assert listed
    for path in listed:
        assert (ROOT / path).is_file(), path


def test_print_inputs_honours_the_two_closure_overrides():
    result = runner.run_step(
        [str(ROOT / "scripts/assemble_v1_critical_graph.sh")],
        env={
            "PRINT_INPUTS": "1",
            "LEFT_BIRTH": "some/left.json",
            "COMPLETENESS": "some/cert.json",
        },
    )
    listed = result.stdout.split()
    assert "some/left.json" in listed
    assert "some/cert.json" in listed
    # --completeness replaces --al-screen; both must never be passed together.
    assert not any("AL_POCKET_SCREEN" in path for path in listed)


# --------------------------------------------------------------------------
# The closure workflow drives the same one command
# --------------------------------------------------------------------------


def _closure_workflow() -> dict:
    import yaml

    return yaml.safe_load(
        (ROOT / ".github/workflows/v1-gate-closure.yml").read_text(encoding="utf-8")
    )


def test_closure_workflow_takes_a_run_id_per_artifact():
    workflow = _closure_workflow()
    dispatch = workflow[True]["workflow_dispatch"]["inputs"]
    expected = {
        f"{role}_run_id" for role, _flag, _description in runner.HARVESTED_INPUTS
    }
    assert set(dispatch) == expected
    for name, spec in dispatch.items():
        assert spec["required"] is True, name


def test_closure_workflow_runs_the_runner_and_never_writes_evidence():
    steps = _closure_workflow()["jobs"]["close"]["steps"]
    scripts = "\n".join(step.get("run") or "" for step in steps)
    assert "scripts/close_v1_gates.py" in scripts
    assert "gh run download" in scripts
    # It must not aim the producing scripts at the durable evidence directory,
    # and it must not hand-set anything there.
    assert "--evidence-dir artifacts/closure-evidence" in scripts
    assert "research/evidence" not in scripts
    # A missing artifact must fail the job rather than be retried with less.
    assert "exit 64" in scripts


def test_closure_workflow_surfaces_the_deciding_code():
    """The environment block belongs in front of the reviewer, not just in a zip."""
    steps = _closure_workflow()["jobs"]["close"]["steps"]
    scripts = "\n".join(step.get("run") or "" for step in steps)
    assert ".environment" in scripts
    # An inherited $PYTHON would only cause a refusal, but a refusal nobody
    # asked for still burns a run.
    assert "unset PYTHON" in scripts


def test_closure_workflow_uploads_inputs_beside_outputs():
    steps = _closure_workflow()["jobs"]["close"]["steps"]
    upload = next(
        step for step in steps if str(step.get("uses", "")).startswith("actions/upload-artifact")
    )
    paths = upload["with"]["path"]
    assert "artifacts/closure-inputs/**" in paths
    assert "artifacts/closure-evidence/**" in paths
