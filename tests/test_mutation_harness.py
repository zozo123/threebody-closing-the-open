"""Unit tests for the mutation harness itself.

The harness is a safety check about safety checks, so it needs its own.  These
tests are fast and deliberately do NOT run the mutations: the full suite is a
separate CI job (``.github/workflows/mutation-suite.yml``).  What they pin is
the harness's own contract -- that every mutation still finds its target in the
source tree, that expectations reference real detectors, that a surviving
mutation is reported as a failure, and that nothing it writes could be mistaken
for evidence.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _harness():
    spec = importlib.util.spec_from_file_location(
        "mutation_harness", ROOT / "scripts/mutation_harness.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    # ``@dataclass`` resolves annotations through ``sys.modules[cls.__module__]``,
    # so the module has to be registered before it is executed.
    sys.modules["mutation_harness"] = module
    spec.loader.exec_module(module)
    return module


HARNESS = _harness()


def test_mutation_ids_are_unique_and_categorised() -> None:
    mutations = HARNESS.mutations()
    ids = [item.id for item in mutations]
    assert len(ids) == len(set(ids))
    assert len(mutations) >= 11, "the task's mutation list is the floor, not the target"
    assert {item.category for item in mutations} <= {
        "physics",
        "artifact",
        "provenance",
        "gate",
        "regression",
    }


def test_every_expected_detector_exists() -> None:
    known = {detector.id for detector in HARNESS.detectors("python")}
    for mutation in HARNESS.mutations():
        unknown = set(mutation.expect) - known
        assert not unknown, f"{mutation.id} expects unknown detector(s) {sorted(unknown)}"


def test_every_declared_gap_explains_itself() -> None:
    for mutation in HARNESS.mutations():
        if mutation.known_gap:
            assert mutation.gap_note.strip(), f"{mutation.id} is a declared gap with no note"


@pytest.mark.parametrize("mutation", HARNESS.mutations(), ids=lambda item: item.id)
def test_every_mutation_still_finds_its_target(tmp_path, mutation) -> None:
    """A mutation whose target moved silently tests nothing.

    ``patch_text`` refuses to guess, so applying every mutation to a copy of the
    real tree is the cheapest possible guard against the suite rotting.  Only
    the files a mutation touches are copied, plus the synthetic fixtures the
    provenance mutations need.
    """
    tree = tmp_path / "tree"
    for relative in (
        "src/threebody_atlas/dynamics.py",
        "src/threebody_atlas/variational.py",
        "src/threebody_atlas/critical_manifold.py",
        "src/threebody_atlas/completeness.py",
        "scripts/assemble_critical_graph.py",
        "research/SEARCH_SCOPE_REGISTRY.json",
        HARNESS.ROOTS_FILE,
        HARNESS.RIGHT_GERMS_FILE,
    ):
        destination = tree / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative).read_bytes())

    fixtures = tree / HARNESS.FIXTURE_DIR
    fixtures.mkdir(parents=True, exist_ok=True)
    (tree / HARNESS.SYNTHETIC_AL).write_text(json.dumps(HARNESS._synthetic_al()))
    (tree / HARNESS.SYNTHETIC_NECK).write_text(json.dumps(HARNESS._synthetic_neck()))

    if mutation.id == "truncate_neck_raster":
        # This one re-runs the real freezer, which needs the whole tree; the
        # full harness covers it and the cheap guard here would only test the
        # copy list.
        pytest.skip("covered by the mutation-suite job, which stages a full tree")
    mutation.apply(tree)


def test_a_surviving_mutation_is_reported_as_a_failure() -> None:
    mutation = HARNESS.Mutation(
        "synthetic_never_caught",
        "physics",
        "a fault nothing notices",
        lambda tree: None,
        ("pytest_dynamics",),
    )
    detector = HARNESS.detectors("python")[0]
    outcome = HARNESS.MutationRun(mutation)
    outcome.runs = [HARNESS.DetectorRun(detector, False, 0, 0.1)]
    found = HARNESS.problems([], [outcome])
    assert any("survived every detector" in text for text in found)
    assert any("pytest_dynamics" in text for text in found)
    assert HARNESS._verdict(outcome) == "**GAP**"


def test_a_noisy_detector_is_refused_rather_than_counted() -> None:
    """A detector that fails on a healthy tree cannot be trusted to kill anything."""
    detector = HARNESS.detectors("python")[0]
    noisy = HARNESS.DetectorRun(detector, True, 1, 0.1)
    found = HARNESS.problems([noisy], [])
    assert any("not silent on an unmutated tree" in text for text in found)


def test_a_declared_gap_that_starts_being_caught_is_also_a_failure() -> None:
    """Stale declarations are as bad as stale checks, so both directions fail."""
    mutation = HARNESS.Mutation(
        "synthetic_declared_gap",
        "regression",
        "declared uncatchable",
        lambda tree: None,
        ("pytest_dynamics",),
        known_gap=True,
        gap_note="declared",
    )
    detector = next(item for item in HARNESS.detectors("python") if item.id == "pytest_dynamics")
    outcome = HARNESS.MutationRun(mutation)
    outcome.runs = [HARNESS.DetectorRun(detector, True, 1, 0.1)]
    assert HARNESS._verdict(outcome) == "partial kill (declared)"
    assert not HARNESS.problems([], [outcome])


def test_patch_text_refuses_a_target_it_cannot_find_exactly_once(tmp_path) -> None:
    path = tmp_path / "src" / "mod.py"
    path.parent.mkdir(parents=True)
    path.write_text("alpha\nalpha\n")
    with pytest.raises(AssertionError, match="appears 2 times"):
        HARNESS.patch_text(tmp_path, "src/mod.py", "alpha", "beta")
    with pytest.raises(AssertionError, match="appears 0 times"):
        HARNESS.patch_text(tmp_path, "src/mod.py", "gamma", "beta")
    HARNESS.patch_text(tmp_path, "src/mod.py", "alpha", "beta", count=2)
    assert path.read_text() == "beta\nbeta\n"


def test_harness_fixtures_are_obviously_synthetic() -> None:
    """Nothing the harness writes may be mistaken for evidence."""
    for name in (HARNESS.SYNTHETIC_AL, HARNESS.SYNTHETIC_NECK, HARNESS.SYNTHETIC_CERTIFICATE):
        assert name.startswith("tmp/"), name
        assert "SYNTHETIC" in Path(name).name, name
    assert "research/evidence" not in HARNESS.FIXTURE_DIR
    for payload in (HARNESS._synthetic_al(), HARNESS._synthetic_neck()):
        assert "SYNTHETIC" in payload["note"]


def test_harness_never_writes_into_the_real_repository() -> None:
    """The only paths the harness writes are inside the staged copy."""
    source = (ROOT / "scripts/mutation_harness.py").read_text(encoding="utf-8")
    assert "shutil.copytree(REPO" in source
    # REPO is read from, never written to: no write helper takes REPO as a base.
    for forbidden in ("REPO /", "REPO/"):
        for line in source.splitlines():
            if forbidden in line and "write_text" in line:
                raise AssertionError(f"harness writes into the real repository: {line}")
