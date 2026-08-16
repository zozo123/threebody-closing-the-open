#!/usr/bin/env python3
"""Mutation testing of the truth machinery.

The premise, learned the hard way on 2026-08-16: three safety checks that had
never been seen catching anything were all found to be wrong on the same day --
a completeness certificate that hashed itself and called that tamper-evidence,
three headline organizers exempted by name from every numeric check, and a
stable lobe leaving the scan window recorded as a vertical merge.  "This check
has never caught anything" is evidence against the check, not for the code.

So: inject a fault on purpose and require that some INDEPENDENT detector fires.
Anything that survives every detector is a hole in the safety net, and this
harness names it out loud.

Mechanics
---------
* Every mutation is applied to a fresh COPY of the repository in a temporary
  directory.  The working tree is never touched, and nothing here writes into
  ``research/evidence/`` of the real checkout.
* Fixtures the harness needs (a synthetic AL screen, a synthetic neck raster and
  a certificate frozen over them) are written under ``tmp/mutation_fixtures/``
  inside the COPY, with SYNTHETIC in every filename.
* A baseline pass runs every detector against an unmutated copy.  A detector
  that is not silent on a healthy tree is unusable and the harness says so
  rather than counting its noise as a kill.
* Each mutation declares which detectors are EXPECTED to fire.  The harness
  fails when reality disagrees in either direction: an expected detector that
  stayed silent is a regression, and a mutation declared as a known gap that
  suddenly gets caught means the declaration is stale and must be updated.

Exit status: 0 all mutations behaved as declared, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parents[1]
FIXTURE_DIR = "tmp/mutation_fixtures"
SYNTHETIC_AL = f"{FIXTURE_DIR}/SYNTHETIC_AL_SCREEN.json"
SYNTHETIC_NECK = f"{FIXTURE_DIR}/SYNTHETIC_NECK_SCAN.json"
SYNTHETIC_CERTIFICATE = f"{FIXTURE_DIR}/SYNTHETIC_COMPLETENESS_CERTIFICATE.json"
BASELINE_DIR = f"{FIXTURE_DIR}/baseline"
ROOT_AUDIT_BASELINE = f"{BASELINE_DIR}/SYNTHETIC_ROOT_AUDIT_BASELINE.json"
GRAPH_BASELINE = f"{BASELINE_DIR}/SYNTHETIC_GRAPH_FINGERPRINT.json"

ROOTS_FILE = "research/evidence/V1_HYBRID_CRITICAL_ROOTS_2026-08-15.json"
RIGHT_GERMS_FILE = "research/evidence/V1_SECONDARY_RIGHT_GERMS_2026-08-16.json"

IGNORED = shutil.ignore_patterns(
    ".git", "__pycache__", "*.pyc", ".pytest_cache", ".venv", "venv", "artifacts", "node_modules"
)


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Detector:
    """A command that is silent (exit 0) on a healthy tree and loud otherwise.

    ``tier`` records whether the detector already existed in the repository
    ("committed") or arrived with this suite ("new").  A mutation caught only by
    a "new" detector was, until today, a hole.
    """

    id: str
    tier: str
    description: str
    argv: list[str]
    timeout: int = 900


def detectors(python: str) -> list[Detector]:
    return [
        Detector(
            "metamorphic_properties",
            "new",
            "permutation/translation/Galilean/rotation/time-reversal/similarity/"
            "coordinate covariance of the shipped dynamics",
            [python, "-m", "threebody_atlas.metamorphic"],
        ),
        Detector(
            "root_physics_audit",
            "new",
            "re-derive published critical roots with the shipped dynamics and "
            "compare against an unmutated baseline",
            [
                python,
                "scripts/audit_published_root_physics.py",
                "--count",
                "3",
                "--compare",
                ROOT_AUDIT_BASELINE,
                "--tolerance",
                "1e-9",
            ],
        ),
        Detector(
            "graph_structural_invariants",
            "new",
            "re-assemble the critical graph and compare its structural fingerprint",
            [python, "scripts/probe_graph_invariants.py", "--baseline", GRAPH_BASELINE],
        ),
        Detector(
            "completeness_certificate_verifier",
            "committed",
            "threebody_atlas.completeness.verify_certificate over a sealed certificate",
            [python, "scripts/probe_completeness_certificate.py", SYNTHETIC_CERTIFICATE],
        ),
        Detector(
            "pytest_dynamics",
            "committed",
            "tests/test_dynamics.py -- pairwise force sum, zero net force, conservation",
            [python, "-m", "pytest", "tests/test_dynamics.py", "-q", "-p", "no:cacheprovider"],
        ),
        Detector(
            "pytest_variational",
            "committed",
            "tests/test_variational.py -- analytic tangent vs centered finite difference",
            [python, "-m", "pytest", "tests/test_variational.py", "-q", "-p", "no:cacheprovider"],
        ),
        Detector(
            "pytest_reduced_charts",
            "committed",
            "tests/test_reduced.py + tests/test_canonical_jacobi.py",
            [
                python,
                "-m",
                "pytest",
                "tests/test_reduced.py",
                "tests/test_canonical_jacobi.py",
                "-q",
                "-p",
                "no:cacheprovider",
            ],
        ),
        Detector(
            "pytest_physical_quotient",
            "committed",
            "tests/test_physical_floquet.py -- physical quotient / Krein diagnostics",
            [
                python,
                "-m",
                "pytest",
                "tests/test_physical_floquet.py",
                "-q",
                "-p",
                "no:cacheprovider",
            ],
        ),
        Detector(
            "pytest_completeness",
            "committed",
            "tests/test_completeness.py -- certificate contract and frozen gates",
            [python, "-m", "pytest", "tests/test_completeness.py", "-q", "-p", "no:cacheprovider"],
        ),
        Detector(
            "pytest_critical_graph",
            "committed",
            "tests/test_critical_graph.py -- assembler, germ validator, frozen gates, staleness",
            [
                python,
                "-m",
                "pytest",
                "tests/test_critical_graph.py",
                "-q",
                "-p",
                "no:cacheprovider",
            ],
        ),
        Detector(
            "pytest_float64_census",
            "committed",
            "tests/test_float64_census.py -- recorded census numbers against the frozen gates",
            [
                python,
                "-m",
                "pytest",
                "tests/test_float64_census.py",
                "-q",
                "-p",
                "no:cacheprovider",
            ],
        ),
        Detector(
            "pytest_neck_topology",
            "committed",
            "tests/test_neck_topology.py -- merge verdicts and scan-window truncation",
            [
                python,
                "-m",
                "pytest",
                "tests/test_neck_topology.py",
                "-q",
                "-p",
                "no:cacheprovider",
            ],
        ),
    ]


# ---------------------------------------------------------------------------
# Mutation primitives
# ---------------------------------------------------------------------------
def patch_text(tree: Path, relative: str, old: str, new: str, *, count: int = 1) -> None:
    """Replace ``old`` with ``new``, insisting it appears exactly ``count`` times.

    The insistence is the point: if a refactor moves the line this mutation
    targets, the harness must fail loudly rather than quietly test nothing.
    """
    path = tree / relative
    text = path.read_text(encoding="utf-8")
    found = text.count(old)
    if found != count:
        raise AssertionError(
            f"mutation target appears {found} times in {relative} (expected {count}): {old!r}"
        )
    path.write_text(text.replace(old, new), encoding="utf-8")


def patch_json(tree: Path, relative: str, mutate: Callable[[Any], Any]) -> None:
    path = tree / relative
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload = mutate(payload)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _find_root(payload: dict[str, Any], cell_id: int) -> dict[str, Any]:
    for row in payload["roots"]:
        if int(row["cell_id"]) == cell_id:
            return row
    raise AssertionError(f"cell {cell_id} is not in the roots file")


# ---------------------------------------------------------------------------
# Mutations
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Mutation:
    id: str
    category: str
    description: str
    apply: Callable[[Path], None]
    expect: tuple[str, ...]
    rationale: str = ""
    known_gap: bool = False
    gap_note: str = ""


def _flip_force_sign(tree: Path) -> None:
    patch_text(
        tree,
        "src/threebody_atlas/dynamics.py",
        "out[i] += g * masses[j] * delta / (r2 ** 1.5)",
        "out[i] += (-1.0 if (i, j) == (0, 1) else 1.0) * g * masses[j] * delta / (r2 ** 1.5)",
    )


def _swap_m1_m2_coefficient(tree: Path) -> None:
    """Attach m2's coefficient to body 1 and m1's to body 2 in the force law.

    Permutation covariance is exactly the property that pins a mass to its body,
    so this is the mutation it must kill.  Formally: the mutated law is
    ``a_i = sum_j G (swap01 m)_j d_ij / r^3``, and covariance under a relabelling
    Q requires ``swap01 . Q == Q . swap01``.  Any Q that does not commute with
    the transposition (0 1) -- for instance (0 2) -- breaks it.
    """
    patch_text(
        tree,
        "src/threebody_atlas/dynamics.py",
        "    masses = np.asarray(masses, dtype=float)\n    positions = np.asarray(positions, dtype=float)",
        "    masses = np.asarray(masses, dtype=float)[[1, 0, 2]]\n    positions = np.asarray(positions, dtype=float)",
    )


def _own_mass_coefficient(tree: Path) -> None:
    """Use the ATTRACTED body's mass instead of the attracting one's.

    Kept deliberately, even though permutation covariance is blind to it: the
    coefficient still travels with its own body, so the mutated law is perfectly
    permutation covariant.  Recording that blindness is more useful than
    pretending the property is stronger than it is.
    """
    patch_text(
        tree,
        "src/threebody_atlas/dynamics.py",
        "out[i] += g * masses[j] * delta / (r2 ** 1.5)",
        "out[i] += g * masses[i] * delta / (r2 ** 1.5)",
    )


def _perturb_tangent_term(tree: Path) -> None:
    patch_text(
        tree,
        "src/threebody_atlas/variational.py",
        "block = g * masses[j] * (np.eye(2) / r**3 - 3.0 * np.outer(d, d) / r**5)",
        "block = g * masses[j] * (np.eye(2) / r**3 - 3.000003 * np.outer(d, d) / r**5)",
    )


def _shift_gravitational_constant(tree: Path) -> None:
    """Shift G by 1e-6 relative *inside the pair force*, where it is really used."""
    patch_text(
        tree,
        "src/threebody_atlas/dynamics.py",
        "out[i] += g * masses[j] * delta / (r2 ** 1.5)",
        "out[i] += 1.000001 * g * masses[j] * delta / (r2 ** 1.5)",
    )


def _shift_gravitational_constant_default(tree: Path) -> None:
    """Shift the ``g`` DEFAULT on ``acceleration`` by 1e-6 relative.

    Included because it turned out to be almost inert: ``rhs`` passes ``g=g``
    from its own default of 1.0, so every integration path in the library
    overrides this default and never sees the shift.  Only code that calls
    ``acceleration`` directly does.  That is a real (if small) finding about the
    shape of the API, and it is recorded here rather than quietly dropped.
    """
    patch_text(
        tree,
        "src/threebody_atlas/dynamics.py",
        "def acceleration(positions: Array, masses: Array, *, g: float = 1.0) -> Array:",
        "def acceleration(positions: Array, masses: Array, *, g: float = 1.000001) -> Array:",
    )


def _drop_transition_cell(tree: Path) -> None:
    def mutate(payload: dict[str, Any]) -> dict[str, Any]:
        before = len(payload["roots"])
        payload["roots"] = [row for row in payload["roots"] if int(row["cell_id"]) != 300]
        assert len(payload["roots"]) == before - 1
        return payload

    patch_json(tree, ROOTS_FILE, mutate)


def _duplicate_transition_cell(tree: Path) -> None:
    def mutate(payload: dict[str, Any]) -> dict[str, Any]:
        payload["roots"].append(json.loads(json.dumps(_find_root(payload, 300))))
        return payload

    patch_json(tree, ROOTS_FILE, mutate)


def _reverse_edge_orientation(tree: Path) -> None:
    def mutate(payload: dict[str, Any]) -> dict[str, Any]:
        row = _find_root(payload, 300)
        flipped = {"U->S": "S->U", "S->U": "U->S"}[str(row["orientation"])]
        row["orientation"] = flipped
        return payload

    patch_json(tree, ROOTS_FILE, mutate)


def _hand_written_germ(tree: Path) -> None:
    """Replace a traced germ with something a human could have typed.

    Keeps the labels that look authoritative -- status "traced",
    canonical_bound/canonical_bracketed true -- and drops exactly the numbers a
    real continuation would have produced.
    """

    def mutate(payload: dict[str, Any]) -> dict[str, Any]:
        germ = payload["germs"][0]
        payload["germs"][0] = {
            "mixed_node": germ["mixed_node"],
            "event_mode": germ["event_mode"],
            "direction": germ["direction"],
            "status": "traced",
            "ends_on": germ.get("ends_on"),
            "masses": germ["masses"],
            "canonical_bound": True,
            "canonical_bracketed": True,
            "note": "SYNTHETIC hand-written germ injected by scripts/mutation_harness.py",
        }
        return payload

    patch_json(tree, RIGHT_GERMS_FILE, mutate)


def _alter_evidence_after_sealing(tree: Path) -> None:
    """Change a source file the certificate already sealed, and leave the seal."""
    path = tree / SYNTHETIC_AL
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["harmless_looking_annotation"] = "added after the certificate was sealed"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _truncate_neck_raster(tree: Path) -> None:
    """Truncate the raster, then re-freeze AND forge ``passed`` back to true.

    Re-sealing is what an adversary (or a well-meaning script) would do, and it
    defeats any check that only compares digests.  Only re-deriving the neck
    predicate from the raster can catch this.
    """
    neck_path = tree / SYNTHETIC_NECK
    neck = json.loads(neck_path.read_text(encoding="utf-8"))
    neck["any_boundary_truncated_merge_test"] = True
    neck["boundary_truncated_lines"] = [{"m1": 0.997, "reason": "stable lobe left the window"}]
    neck["merge_verdict_counts"] = {
        "separated": 0,
        "interior_merge": 0,
        "truncation_undecidable": 1,
        "no_stable_sample": 0,
    }
    neck["all_lines_separated"] = False
    neck_path.write_text(json.dumps(neck, indent=2) + "\n", encoding="utf-8")

    _freeze_certificate(tree)
    certificate_path = tree / SYNTHETIC_CERTIFICATE
    record = json.loads(certificate_path.read_text(encoding="utf-8"))
    record["passed"] = True
    record.pop("self_verification_errors", None)
    record["neck"]["any_boundary_truncated_merge_test"] = False
    record["neck"]["all_lines_separated"] = True
    record["neck"]["topology_clean"] = True
    _reseal(record)
    certificate_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def _change_sampling_semantics_without_version_bump(tree: Path) -> None:
    """Strengthen a criterion's meaning while keeping its /v1 identifier.

    Source bytes and their sha256 bindings remain untouched.  Only the semantic
    contract changes, so a digest-only evidence DAG would miss this mutation.
    """
    patch_text(
        tree,
        "research/SEARCH_SCOPE_REGISTRY.json",
        '"description": "Corrected proposals in one bounded acquisition pocket; screening evidence only.",',
        '"description": "MUTATED to imply a broader search without a criterion version bump.",',
    )


def _loosen_assembler_event_gate(tree: Path) -> None:
    patch_text(tree, "scripts/assemble_critical_graph.py", "EVENT_GATE = 2e-8", "EVENT_GATE = 2e-7")


def _loosen_classifier_event_gate(tree: Path) -> None:
    """Loosen the localizer's event tolerance wherever it is declared."""
    patch_text(
        tree,
        "src/threebody_atlas/critical_manifold.py",
        "event_tolerance: float = 2e-8",
        "event_tolerance: float = 2e-7",
        # 3 since the conditioning work added a third entry point carrying the
        # same default.  The count is asserted, not inferred, so that a NEW
        # declaration of the gate cannot appear without this mutation being
        # re-checked against it.
        count=3,
    )


def _loosen_shooting_residual_gate(tree: Path) -> None:
    patch_text(
        tree,
        "src/threebody_atlas/completeness.py",
        "SHOOTING_RESIDUAL_GATE = 1e-7",
        "SHOOTING_RESIDUAL_GATE = 1e-6",
    )


def _exempt_headline_organizers(tree: Path) -> None:
    """Re-introduce the by-name exemption that was found wrong today."""
    patch_text(
        tree,
        "scripts/assemble_critical_graph.py",
        "    reasons: list[str] = []\n    if record.get(\"status\") not in",
        "    reasons: list[str] = []\n"
        "    if record.get(\"mixed_node\") in BASE_MIXED_NODE_IDS:\n"
        "        return reasons\n"
        "    if record.get(\"status\") not in",
    )


def _lump_domain_exits(tree: Path) -> None:
    """Collapse distinct domain exits back onto one node per face."""
    patch_text(
        tree,
        "scripts/assemble_critical_graph.py",
        "    return f\"{hit['face']}_{hit['along_axis']}_{grid_label(hit['along'])}\"",
        "    return str(hit['face'])",
    )


def _widen_mass_jump(tree: Path) -> None:
    patch_text(tree, "scripts/assemble_critical_graph.py", "MASS_JUMP = 0.025", "MASS_JUMP = 0.05")


def mutations() -> list[Mutation]:
    return [
        Mutation(
            "flip_force_sign",
            "physics",
            "flip the sign of the force body 0 feels from body 1",
            _flip_force_sign,
            (
                "metamorphic_properties",
                "root_physics_audit",
                "pytest_dynamics",
                "pytest_variational",
            ),
            rationale="breaks Newton's third law and permutation covariance at once",
        ),
        Mutation(
            "swap_m1_m2_coefficient",
            "physics",
            "swap the m1 and m2 coefficients in the pair force (wrong m1/m2 coefficient)",
            _swap_m1_m2_coefficient,
            (
                "metamorphic_properties",
                "root_physics_audit",
            ),
            rationale="permutation covariance is the property that pins a mass to its body",
        ),
        Mutation(
            "own_mass_coefficient",
            "physics",
            "use the attracted body's mass instead of the attracting body's",
            _own_mass_coefficient,
            ("root_physics_audit", "pytest_dynamics"),
            known_gap=True,
            gap_note=(
                "PERMUTATION COVARIANCE IS BLIND TO THIS. The wrong coefficient still "
                "travels with its own body, so the mutated force law is exactly permutation "
                "covariant and the metamorphic suite stays silent. Not every 'wrong mass "
                "coefficient' is a relabelling violation; this one is caught by Newton's "
                "third law (tests/test_dynamics.py) and by re-deriving the published roots, "
                "not by the symmetry."
            ),
            rationale="documents the exact boundary of what permutation covariance can see",
        ),
        Mutation(
            "perturb_tangent_term",
            "physics",
            "perturb the 3.0 in the tidal Hessian block by 1e-6 relative",
            _perturb_tangent_term,
            ("pytest_variational", "metamorphic_properties"),
            rationale=(
                "the flow is untouched, so only a tangent-map check can see it: the analytic "
                "vs finite-difference audit, and the three-chart covariance (the Jacobi chart "
                "carries its own Hessian and therefore disagrees)"
            ),
        ),
        Mutation(
            "shift_gravitational_constant",
            "physics",
            "shift the gravitational constant by 1e-6 relative inside the pair force",
            _shift_gravitational_constant,
            ("root_physics_audit", "pytest_variational", "pytest_dynamics"),
            rationale=(
                "the published roots are ~1e6x sensitive to G, so re-deriving them is the "
                "sharpest available probe of a constant nobody re-checks"
            ),
        ),
        Mutation(
            "shift_gravitational_constant_default",
            "physics",
            "shift the g DEFAULT ARGUMENT on acceleration() by 1e-6 relative",
            _shift_gravitational_constant_default,
            ("pytest_dynamics",),
            known_gap=True,
            gap_note=(
                "NEARLY INERT, AND THAT IS THE FINDING. rhs() passes g=g from its own "
                "default of 1.0, so every integration path in the library overrides "
                "acceleration()'s default and never sees the shift. Only direct callers of "
                "acceleration() do -- which today means tests/test_dynamics.py and nothing "
                "else. The gravitational constant is effectively declared twice."
            ),
            rationale="documents a physically wrong constant that most of the library cannot see",
        ),
        Mutation(
            "drop_transition_cell",
            "artifact",
            "delete transition cell 300 from the published roots file",
            _drop_transition_cell,
            ("graph_structural_invariants", "pytest_critical_graph"),
            rationale="exact 620/620 coverage must be exact",
        ),
        Mutation(
            "duplicate_transition_cell",
            "artifact",
            "append a second copy of transition cell 300",
            _duplicate_transition_cell,
            ("graph_structural_invariants", "pytest_critical_graph"),
            rationale="duplicate_cell_ids exists for exactly this",
        ),
        Mutation(
            "reverse_edge_orientation",
            "artifact",
            "reverse the S/U orientation recorded for transition cell 300",
            _reverse_edge_orientation,
            ("graph_structural_invariants", "pytest_critical_graph"),
            rationale=(
                "orientation decides which mechanism polyline a cell joins; reversing one "
                "must change the edge decomposition, not be absorbed silently"
            ),
        ),
        Mutation(
            "hand_written_germ",
            "artifact",
            "replace a traced continuation germ with a hand-written record carrying no numbers",
            _hand_written_germ,
            ("graph_structural_invariants",),
            rationale=(
                "the germ validator must demand closure/event/canonical numbers from every "
                "germ, however authoritative its labels look"
            ),
        ),
        Mutation(
            "alter_evidence_after_sealing",
            "provenance",
            "edit a source file after the completeness certificate sealed it",
            _alter_evidence_after_sealing,
            ("completeness_certificate_verifier",),
            rationale="the recorded per-source sha256 is one half of the schema /3 contract",
        ),
        Mutation(
            "truncate_neck_raster",
            "provenance",
            "truncate the neck raster, re-freeze the certificate and forge passed=true",
            _truncate_neck_raster,
            ("completeness_certificate_verifier",),
            rationale=(
                "re-sealing defeats digest-only tamper evidence; only re-deriving the neck "
                "predicate catches a raster whose merge question left the scan window"
            ),
        ),
        Mutation(
            "change_sampling_semantics_without_version_bump",
            "provenance",
            "change a parent criterion's semantics without changing its /v1 identifier",
            _change_sampling_semantics_without_version_bump,
            ("completeness_certificate_verifier",),
            rationale=(
                "file hashes still match; the certificate's semantic-contract digest must "
                "invalidate when the meaning of a parent criterion changes"
            ),
        ),
        Mutation(
            "loosen_assembler_event_gate",
            "gate",
            "loosen the assembler's frozen |event| gate from 2e-8 to 2e-7",
            _loosen_assembler_event_gate,
            ("pytest_critical_graph",),
            rationale="the gate is frozen; only a literal pin can notice it moving",
        ),
        Mutation(
            "loosen_classifier_event_gate",
            "gate",
            "loosen classify_localized_cell's event tolerance from 2e-8 to 2e-7",
            _loosen_classifier_event_gate,
            ("pytest_critical_graph",),
            rationale="same gate, different owner: the cell classifier",
        ),
        Mutation(
            "loosen_shooting_residual_gate",
            "gate",
            "loosen the completeness shooting-residual gate from 1e-7 to 1e-6",
            _loosen_shooting_residual_gate,
            ("pytest_completeness",),
            rationale="the third frozen gate",
        ),
        Mutation(
            "exempt_headline_organizers",
            "regression",
            "re-introduce the by-name germ exemption for the three headline organizers",
            _exempt_headline_organizers,
            ("pytest_critical_graph",),
            rationale=(
                "this exact exemption was found wrong on 2026-08-16; the test that killed it "
                "must stay able to kill it"
            ),
        ),
        Mutation(
            "lump_domain_exits",
            "regression",
            "collapse distinct declared-domain exits back onto one node per face",
            _lump_domain_exits,
            ("pytest_critical_graph",),
            rationale=(
                "lumping manufactured incidence between curves that never meet; that was "
                "fixed, so the fix needs a demonstrated kill"
            ),
        ),
        Mutation(
            "widen_mass_jump",
            "regression",
            "widen MASS_JUMP from 0.025 to 0.05",
            _widen_mass_jump,
            ("pytest_critical_graph",),
            known_gap=True,
            gap_note=(
                "PARTIAL KILL ONLY. The census has essentially one root per "
                "(event_mode, orientation, m1) slice, so edge_count stays 7 for MASS_JUMP "
                "anywhere in [0.0112, 0.096]: the polyline reconstruction performs no branch "
                "discrimination and this mutation changes NOTHING about the assembled graph. "
                "It is caught only by test_mass_jump_window_is_pinned, which pins the literal "
                "0.025. No recomputation anywhere notices. The single largest unverified "
                "assumption in the graph -- that nearby roots really are one continuous "
                "critical curve -- has no detector at all."
            ),
            rationale="documents a semantic blind spot rather than a kill",
        ),
    ]


# ---------------------------------------------------------------------------
# Fixtures and staging
# ---------------------------------------------------------------------------
def _synthetic_al() -> dict[str, Any]:
    rows = [
        {
            "shooting_success": True,
            "shooting_residual": 1e-12,
            "corrected": {"screening_stable": False},
        }
        for _ in range(12)
    ]
    return {
        "note": "SYNTHETIC active-learning pocket screen, generated by scripts/mutation_harness.py. "
        "Not evidence. Never write this into research/evidence/.",
        "attempted": rows,
        "accepted_candidates": rows,
    }


def _synthetic_neck() -> dict[str, Any]:
    return {
        "note": "SYNTHETIC neck raster, generated by scripts/mutation_harness.py. Not evidence.",
        "completed": True,
        "grid": {"m1": [0.997, 0.999], "m2": [0.993, 1.006], "step": 0.0001, "samples": 12},
        "minimum_resolved_unstable_gap": 0.0002,
        "any_vertical_merge": False,
        "any_boundary_truncated_merge_test": False,
        "any_line_without_stable_sample": False,
        "any_stable_interval_touches_boundary": False,
        "all_lines_separated": True,
        "merge_verdict_counts": {
            "separated": 1,
            "interior_merge": 0,
            "truncation_undecidable": 0,
            "no_stable_sample": 0,
        },
        "boundary_truncated_lines": [],
        "max_shooting_residual": 1e-9,
        "line_summaries": [{"m1": 0.997, "stable_intervals": [[0.994, 0.996]]}],
    }


def _reseal(record: dict[str, Any]) -> dict[str, Any]:
    """Seal a forged record with the REAL repository's sealer, not the mutant's.

    The harness's own tooling must stay unmutated; otherwise a mutation could
    quietly disable the very forgery it is supposed to perform, and the detector
    would look clean for the wrong reason.
    """
    sys.path.insert(0, str(REPO / "src"))
    from threebody_atlas.completeness import seal

    return seal(record)


def _freeze_certificate(tree: Path) -> None:
    """Run the real freezer over the synthetic sources inside ``tree``."""
    result = subprocess.run(
        [
            sys.executable,
            "scripts/freeze_completeness_certificate.py",
            SYNTHETIC_CERTIFICATE,
            "--al-screen",
            SYNTHETIC_AL,
            "--neck-scan",
            SYNTHETIC_NECK,
        ],
        cwd=str(tree),
        env=_env(tree),
        capture_output=True,
        text=True,
        check=False,
    )
    # Exit 2 means "frozen but not passed", which is exactly what the truncated
    # variant produces; both are useful states for this harness.
    if result.returncode not in (0, 2):
        raise RuntimeError(f"freezer failed: {result.stdout}\n{result.stderr}")


def _env(tree: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tree / "src")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # Per-mutant temp root, deliberately a SIBLING of the tree rather than a
    # directory inside it.  Mutants run in parallel and several committed tests
    # write into ``tmp_path.parent``, so sharing the system temp root let one
    # mutant's pytest collide with another's and be scored as a spurious kill.
    # It must stay outside the tree because
    # test_verification_refuses_an_absolute_source_outside_the_allowed_roots
    # relies on ``tmp_path.parent`` genuinely being outside the repository.
    scratch = tree.parent / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    env["TMPDIR"] = str(scratch)
    env["PYTEST_DEBUG_TEMPROOT"] = str(scratch)
    return env


def stage(destination: Path) -> Path:
    tree = destination / "tree"
    shutil.copytree(REPO, tree, ignore=IGNORED, symlinks=True)
    (tree / FIXTURE_DIR).mkdir(parents=True, exist_ok=True)
    (tree / BASELINE_DIR).mkdir(parents=True, exist_ok=True)
    (tree / SYNTHETIC_AL).write_text(json.dumps(_synthetic_al(), indent=2) + "\n", encoding="utf-8")
    (tree / SYNTHETIC_NECK).write_text(
        json.dumps(_synthetic_neck(), indent=2) + "\n", encoding="utf-8"
    )
    _freeze_certificate(tree)
    return tree


def emit_baselines(tree: Path) -> None:
    for argv in (
        [
            sys.executable,
            "scripts/audit_published_root_physics.py",
            "--count",
            "3",
            "--emit",
            ROOT_AUDIT_BASELINE,
        ],
        [sys.executable, "scripts/probe_graph_invariants.py", "--emit", GRAPH_BASELINE],
    ):
        result = subprocess.run(
            argv, cwd=str(tree), env=_env(tree), capture_output=True, text=True, check=False
        )
        if result.returncode != 0:
            raise RuntimeError(f"baseline emission failed: {argv}\n{result.stdout}{result.stderr}")


def copy_baselines(source: Path, tree: Path) -> None:
    (tree / BASELINE_DIR).mkdir(parents=True, exist_ok=True)
    for name in (ROOT_AUDIT_BASELINE, GRAPH_BASELINE):
        shutil.copyfile(source / name, tree / name)


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------
@dataclass
class DetectorRun:
    detector: Detector
    fired: bool
    status: int
    seconds: float
    tail: str = ""


@dataclass
class MutationRun:
    mutation: Mutation
    runs: list[DetectorRun] = field(default_factory=list)
    error: str | None = None

    @property
    def fired(self) -> list[str]:
        return [run.detector.id for run in self.runs if run.fired]

    @property
    def fired_tiers(self) -> set[str]:
        return {run.detector.tier for run in self.runs if run.fired}


def run_detector(tree: Path, detector: Detector) -> DetectorRun:
    started = time.monotonic()
    try:
        result = subprocess.run(
            detector.argv,
            cwd=str(tree),
            env=_env(tree),
            capture_output=True,
            text=True,
            timeout=detector.timeout,
            check=False,
        )
        status, output = result.returncode, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        status, output = 124, "detector timed out"
    tail = "\n".join(output.strip().splitlines()[-6:])
    return DetectorRun(detector, status != 0, status, time.monotonic() - started, tail)


def run_one(
    mutation: Mutation, workspace: Path, baseline_tree: Path, detector_list: list[Detector]
) -> MutationRun:
    outcome = MutationRun(mutation)
    tree = stage(workspace)
    copy_baselines(baseline_tree, tree)
    try:
        mutation.apply(tree)
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        outcome.error = f"{type(exc).__name__}: {exc}"
        return outcome
    for detector in detector_list:
        outcome.runs.append(run_detector(tree, detector))
    return outcome


def format_report(baseline: list[DetectorRun], outcomes: list[MutationRun]) -> str:
    lines: list[str] = ["# Mutation testing of the truth machinery", ""]
    lines.append("## Baseline (unmutated tree)")
    lines.append("")
    unusable = [run for run in baseline if run.fired]
    for run in baseline:
        state = "NOISY (unusable)" if run.fired else "silent"
        lines.append(f"- `{run.detector.id}` [{run.detector.tier}] {state} ({run.seconds:.1f}s)")
    if unusable:
        lines.append("")
        lines.append(
            "**A detector that is not silent on a healthy tree cannot be counted as a kill.**"
        )
    lines.append("")
    lines.append("## Mutations")
    lines.append("")
    lines.append("| mutation | category | detectors that fired | verdict |")
    lines.append("| --- | --- | --- | --- |")
    for outcome in outcomes:
        fired = outcome.fired
        verdict = _verdict(outcome)
        shown = ", ".join(f"`{name}`" for name in fired) if fired else "**NO DETECTOR FIRED**"
        lines.append(
            f"| `{outcome.mutation.id}` | {outcome.mutation.category} | {shown} | {verdict} |"
        )
    lines.append("")
    gaps = [outcome for outcome in outcomes if not outcome.fired]
    lines.append("## Gaps in the safety net")
    lines.append("")
    if gaps:
        for outcome in gaps:
            lines.append(f"- **{outcome.mutation.id}**: {outcome.mutation.description}")
            if outcome.mutation.gap_note:
                lines.append(f"  - {outcome.mutation.gap_note}")
    else:
        lines.append("Every mutation was caught by at least one detector.")
    partial = [outcome for outcome in outcomes if outcome.mutation.known_gap and outcome.fired]
    if partial:
        lines.append("")
        lines.append("## Declared partial kills")
        lines.append("")
        for outcome in partial:
            lines.append(f"- **{outcome.mutation.id}**: {outcome.mutation.gap_note}")
    lines.append("")
    lines.append("## Detail")
    lines.append("")
    for outcome in outcomes:
        lines.append(f"### {outcome.mutation.id}")
        lines.append("")
        lines.append(f"{outcome.mutation.description}")
        if outcome.mutation.rationale:
            lines.append("")
            lines.append(f"_Why this fault matters:_ {outcome.mutation.rationale}")
        lines.append("")
        if outcome.error:
            lines.append(f"- COULD NOT APPLY: {outcome.error}")
            lines.append("")
            continue
        for run in outcome.runs:
            mark = "FIRED" if run.fired else "silent"
            lines.append(
                f"- `{run.detector.id}` [{run.detector.tier}] {mark} "
                f"(exit {run.status}, {run.seconds:.1f}s)"
            )
        lines.append("")
    return "\n".join(lines)


def _verdict(outcome: MutationRun) -> str:
    if outcome.error:
        return "HARNESS ERROR"
    fired = outcome.fired
    if not fired:
        return "GAP (declared)" if outcome.mutation.known_gap else "**GAP**"
    missing = [name for name in outcome.mutation.expect if name not in fired]
    if missing:
        return f"caught, but {', '.join(missing)} stayed silent"
    if outcome.mutation.known_gap:
        return "partial kill (declared)"
    return "killed"


def problems(baseline: list[DetectorRun], outcomes: list[MutationRun]) -> list[str]:
    found: list[str] = []
    for run in baseline:
        if run.fired:
            found.append(
                f"detector {run.detector.id} is not silent on an unmutated tree "
                f"(exit {run.status}); its kills cannot be trusted"
            )
    for outcome in outcomes:
        if outcome.error:
            found.append(f"mutation {outcome.mutation.id} could not be applied: {outcome.error}")
            continue
        fired = set(outcome.fired)
        if not fired and not outcome.mutation.known_gap:
            found.append(
                f"mutation {outcome.mutation.id} survived every detector; "
                "this is a hole in the safety net"
            )
        missing = [name for name in outcome.mutation.expect if name not in fired]
        if missing:
            found.append(
                f"mutation {outcome.mutation.id}: expected detector(s) "
                f"{', '.join(missing)} stayed silent"
            )
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", action="append", default=[], help="run only these mutation ids")
    parser.add_argument("--report", help="write the markdown report here")
    parser.add_argument("--json", dest="json_report", help="write a machine-readable report here")
    parser.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--list", action="store_true", help="list mutations and exit")
    args = parser.parse_args()

    selected = mutations()
    if args.only:
        wanted = set(args.only)
        unknown = wanted - {item.id for item in selected}
        if unknown:
            parser.error(f"unknown mutation id(s): {sorted(unknown)}")
        selected = [item for item in selected if item.id in wanted]
    if args.list:
        for item in selected:
            print(f"{item.id:<34} {item.category:<11} {item.description}")
        return 0

    detector_list = detectors(sys.executable)
    with tempfile.TemporaryDirectory(prefix="mutation-harness-") as scratch:
        root = Path(scratch)
        baseline_workspace = root / "baseline"
        baseline_workspace.mkdir()
        baseline_tree = stage(baseline_workspace)
        emit_baselines(baseline_tree)
        baseline_runs = [run_detector(baseline_tree, detector) for detector in detector_list]

        outcomes: list[MutationRun] = []
        with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
            futures = []
            for index, mutation in enumerate(selected):
                workspace = root / f"mutant-{index:02d}"
                workspace.mkdir()
                futures.append(
                    pool.submit(run_one, mutation, workspace, baseline_tree, detector_list)
                )
            for future in futures:
                outcomes.append(future.result())

        report = format_report(baseline_runs, outcomes)
        print(report)
        if args.report:
            Path(args.report).parent.mkdir(parents=True, exist_ok=True)
            Path(args.report).write_text(report + "\n", encoding="utf-8")
        if args.json_report:
            payload = {
                "schema": "atlas.v1.mutation-report/1",
                "baseline": [
                    {
                        "detector": run.detector.id,
                        "tier": run.detector.tier,
                        "silent": not run.fired,
                        "status": run.status,
                    }
                    for run in baseline_runs
                ],
                "mutations": [
                    {
                        "id": outcome.mutation.id,
                        "category": outcome.mutation.category,
                        "description": outcome.mutation.description,
                        "known_gap": outcome.mutation.known_gap,
                        "gap_note": outcome.mutation.gap_note,
                        "error": outcome.error,
                        "fired": outcome.fired,
                        "fired_tiers": sorted(outcome.fired_tiers),
                        "expected": list(outcome.mutation.expect),
                        "verdict": _verdict(outcome),
                    }
                    for outcome in outcomes
                ],
            }
            Path(args.json_report).parent.mkdir(parents=True, exist_ok=True)
            Path(args.json_report).write_text(
                json.dumps(payload, indent=2) + "\n", encoding="utf-8"
            )

    found = problems(baseline_runs, outcomes)
    if found:
        print("\nMUTATION HARNESS FAILURES:")
        for item in found:
            print(f"  - {item}")
        return 1
    print("\nevery mutation behaved exactly as declared")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
