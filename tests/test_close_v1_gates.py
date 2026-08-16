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
    """Drive every producing script with the interpreter running the tests.

    The pinned assembler shell script defaults to ``uv run --no-sync python``,
    which is unavailable when the local uv is older than the pyproject pin.
    """
    monkeypatch.setenv("PYTHON", sys.executable)
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
    # What remains is the genuine open work, reported exactly.
    assert "no_missing_mixed_germs" in blockers

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


def test_ledger_carries_no_wall_clock_or_machine_state(evidence_dir):
    runner.main(_argv(evidence_dir))
    text = (evidence_dir / runner.CLOSURE_LEDGER).read_text()
    for forbidden in ("timestamp", "generated_at", "hostname", str(ROOT)):
        assert forbidden not in text, forbidden


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


def test_a_hand_written_passed_classification_does_not_survive_the_assembler():
    """`"passed": true` with a screening evidence_level is still unresolved.

    load_classification() requires passed=true AND an allowed class AND an
    evidence_level in {independently_reproduced, physical, continuation,
    definition}.  The forgery satisfies the first two and fails the third.
    """
    spec = importlib.util.spec_from_file_location(
        "assemble_critical_graph", ROOT / "scripts/assemble_critical_graph.py"
    )
    assembler = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(assembler)

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


def test_closure_workflow_uploads_inputs_beside_outputs():
    steps = _closure_workflow()["jobs"]["close"]["steps"]
    upload = next(
        step for step in steps if str(step.get("uses", "")).startswith("actions/upload-artifact")
    )
    paths = upload["with"]["path"]
    assert "artifacts/closure-inputs/**" in paths
    assert "artifacts/closure-evidence/**" in paths
