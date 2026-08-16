#!/usr/bin/env python3
"""Run the whole v1 closure chain from harvested CI artifacts to a decision.

WHY PYTHON AND NOT BASH
-----------------------
The task this replaces is "one command", and a shell version would have been a
thin wrapper around three inline ``python -`` heredocs anyway: it has to hash
every input, normalise paths into the repository, parse the assembled graph, and
enumerate *which* conjuncts of the assembler's ``release_ready`` expression are
false.  Bash can shell out for each of those, but then the interesting logic
lives in unreviewable heredocs and the exit-code plumbing is duplicated three
times.  Python keeps it in one reviewable file with real tests.

WHAT THIS SCRIPT IS ALLOWED TO DO
---------------------------------
Orchestrate.  Nothing else.  It runs, in order:

  1. ``scripts/classify_secondary_left_birth.py``  (geometry JSON + BigFloat JSON)
  2. ``scripts/freeze_completeness_certificate.py`` (AL pocket screen + neck raster)
  3. ``scripts/assemble_v1_critical_graph.sh``      (every evidence artifact)

and then *reads back* the graph those scripts produced.

WHAT THIS SCRIPT IS INCAPABLE OF DOING
--------------------------------------
Deciding that a gate is satisfied.

* It never writes a classification, a certificate, or a graph.  Every file under
  the evidence directory is written by the producing script named above, invoked
  as a subprocess.  This module contains no JSON templating for those records.
* It never compares a number against a numerical gate.  The gates
  (|event| <= 2e-8, closure <= 1e-7, residual <= 1e-7) live in the classifier,
  the freezer and ``threebody_atlas.completeness``; grep this file for ``2e-8``
  and you will not find it.
* The pass/fail decision is ``graph["release_ready"] is True``, cross-checked
  against the assembler's own exit status *and* against the independently
  re-derived conjunct list.  If those three ever disagree the run aborts as a
  tooling failure (exit 3) rather than picking the friendlier answer.
* It refuses to degrade.  Missing input => refuse (exit 64).  A producing script
  that refuses => stop (exit 2), never fall back to a weaker assembler
  invocation or to a stale committed classification.

EXIT STATUS
-----------
  0   release_ready
  2   the chain ran (or a producing script legitimately refused) and the result
      is not release_ready -- an honest open scientific state, with the exact
      blocker list printed
  3   tooling failure: a producing script crashed, or the three independent
      readings of the decision disagreed
  64  refused: a required input is missing, unreadable, or outside the repository

REPRODUCIBILITY
---------------
Every artifact this script writes is a pure function of the input bytes: no
timestamps, no hostnames, no absolute paths, no run ordering.  Input paths are
normalised to repository-relative POSIX strings before they are handed to the
producing scripts, because those strings are embedded verbatim in the
certificate's ``sources`` block and in the graph's ``evidence`` fields -- and
because ``threebody_atlas.completeness.verify_certificate`` can only re-resolve
and re-hash a source that lives inside the repository.  Running this twice on
the same inputs yields byte-identical files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

CLASSIFY_LEFT_BIRTH = "scripts/classify_secondary_left_birth.py"
FREEZE_COMPLETENESS = "scripts/freeze_completeness_certificate.py"
ASSEMBLE_GRAPH = "scripts/assemble_v1_critical_graph.sh"

#: Deterministic output names.  These are constants, not derived from the clock:
#: a date-stamped default would break byte-reproducibility across days for the
#: same inputs, and the provenance of each record lives in the ledger, not in
#: its filename.
LEFT_BIRTH_CLASS = "V1_LEFT_BIRTH_CLASS_FOLD.json"
COMPLETENESS_CERTIFICATE = "V1_COMPLETENESS_CERTIFICATE.json"
CRITICAL_GRAPH = "V1_CRITICAL_GRAPH.json"
CLOSURE_LEDGER = "V1_CLOSURE_PROVENANCE.json"

#: Every harvested CI artifact this chain consumes, with the flag pair that
#: carries it and the run id it came from.  ``required`` is not negotiable: the
#: whole point of the runner is that a partial closure is a refusal, not a
#: quieter invocation.
HARVESTED_INPUTS = (
    (
        "fold_geometry",
        "--fold-geometry",
        "float64/JAX secondary-left fold geometry screen",
    ),
    (
        "fold_bigfloat",
        "--fold-bigfloat",
        "independent Julia BigFloat secondary-minus-one fold verification",
    ),
    ("al_screen", "--al-screen", "active-learning hidden-pocket screen"),
    ("neck_raster", "--neck-raster", "merged stability-neck raster"),
)

REFUSED = 64
NOT_RELEASE_READY = 2
TOOLING_FAILURE = 3


class Refusal(Exception):
    """A required input is missing or unusable; nothing has been written."""


class ToolingFailure(Exception):
    """A producing script crashed, or the decision readings disagreed."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repo_relative(raw: str, *, flag: str) -> str:
    """Normalise an input path to a repository-relative POSIX string.

    Refuses anything outside the repository.  This is not tidiness: a
    certificate that names a path the assembler cannot re-resolve can never be
    re-verified, so accepting such a path would produce a certificate that is
    guaranteed to be rejected later, for a reason that has nothing to do with
    the science.  ``gh run download -D artifacts/...`` already lands inside the
    repository, which is the intended flow.
    """
    candidate = Path(raw).expanduser()
    resolved = (candidate if candidate.is_absolute() else Path.cwd() / candidate).resolve()
    root = REPO_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise Refusal(
            f"{flag} {raw!r} resolves to {resolved}, which is outside the repository "
            f"{root}. Download CI artifacts into the working tree (for example "
            "'gh run download -D artifacts/closure-inputs') so that the frozen "
            "certificate names a path the assembler can re-resolve and re-hash."
        )
    if not resolved.is_file():
        raise Refusal(f"{flag} {raw!r} is not a readable file ({resolved})")
    return resolved.relative_to(root).as_posix()


def readable_json(relative: str, *, flag: str) -> None:
    path = REPO_ROOT / relative
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise Refusal(f"{flag} {relative!r} is not readable JSON: {exc}") from exc


def collect_inputs(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Validate every harvested input up front and describe it for the ledger.

    All problems are reported together.  Nothing is written before this returns:
    a refusal must leave the evidence directory exactly as it found it.
    """
    problems: list[str] = []
    records: list[dict[str, Any]] = []
    for role, flag, description in HARVESTED_INPUTS:
        raw = getattr(args, role)
        run_id = getattr(args, f"{role}_run_id")
        if not raw or not str(raw).strip():
            problems.append(f"{flag} is required and was not supplied")
            continue
        if not run_id or not str(run_id).strip():
            problems.append(
                f"{flag}-run-id is required: the CI run id is recorded, never guessed"
            )
            continue
        try:
            relative = repo_relative(str(raw), flag=flag)
            readable_json(relative, flag=flag)
        except Refusal as exc:
            problems.append(str(exc))
            continue
        records.append(
            {
                "role": role,
                "description": description,
                "path": relative,
                "sha256": sha256_file(REPO_ROOT / relative),
                "bytes": (REPO_ROOT / relative).stat().st_size,
                "ci_run_id": str(run_id).strip(),
            }
        )
    if problems:
        raise Refusal("\n".join(f"  - {problem}" for problem in problems))
    return records


def python_command() -> list[str]:
    """The interpreter used for the producing scripts.

    Deliberately overridable: local uv is too old to honour the pyproject pin,
    so a throwaway venv with PYTHONPATH=src has to be usable without editing the
    pin.  The chosen interpreter is *not* recorded in the ledger -- it is an
    environment detail, and recording it would make the ledger non-reproducible
    across machines for identical inputs.
    """
    override = os.environ.get("PYTHON")
    if override:
        return override.split()
    return [sys.executable]


def run_step(argv: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    merged = dict(os.environ)
    if env:
        merged.update(env)
    return subprocess.run(  # noqa: S603 - fixed argv built from repo constants
        argv,
        cwd=REPO_ROOT,
        env=merged,
        capture_output=True,
        text=True,
        check=False,
    )


def describe(path: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(REPO_ROOT.resolve()).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def classify_left_birth(
    inputs: dict[str, dict[str, Any]], output: Path
) -> tuple[dict[str, Any], str | None]:
    """Step 1: the classifier decides the secondary-left birth, or refuses."""
    output.unlink(missing_ok=True)
    argv = [
        *python_command(),
        CLASSIFY_LEFT_BIRTH,
        inputs["fold_geometry"]["path"],
        output.resolve().relative_to(REPO_ROOT.resolve()).as_posix(),
        "--bigfloat",
        inputs["fold_bigfloat"]["path"],
    ]
    result = run_step(argv)
    step = {
        "step": "classify_secondary_left_birth",
        "command": argv[len(python_command()) :],
        "exit_status": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }
    if result.returncode != 0:
        # The classifier's own gates refused this fold.  That is a legitimate
        # scientific outcome and it is reported as a blocker -- but the chain
        # stops here rather than assembling with the stale invalidated
        # classification or with no classification at all.  Either fallback
        # would be exactly the silent degradation this runner exists to prevent.
        output.unlink(missing_ok=True)
        step["produced"] = None
        return step, (
            "secondary_left_birth: classify_secondary_left_birth.py refused the "
            "supplied fold artifacts (exit "
            f"{result.returncode}): {result.stderr.strip() or result.stdout.strip()}"
        )
    if not output.is_file():
        raise ToolingFailure(
            f"{CLASSIFY_LEFT_BIRTH} exited 0 without writing {output}"
        )
    record = json.loads(output.read_text(encoding="utf-8"))
    if record.get("id") != "secondary_left_birth":
        raise ToolingFailure(
            f"{CLASSIFY_LEFT_BIRTH} wrote a record for {record.get('id')!r}, "
            "not secondary_left_birth"
        )
    step["produced"] = describe(output)
    return step, None


def freeze_completeness(
    inputs: dict[str, dict[str, Any]], output: Path
) -> dict[str, Any]:
    """Step 2: the freezer seals a certificate, passing or not.

    Exit status 2 means "sealed, but the sources do not support a completeness
    claim".  That is a certificate the assembler will read and reject on its own
    terms, which is the honest outcome; the chain continues so that the rest of
    the blocker list is still computed.  Any other non-zero status is a crash.
    """
    output.unlink(missing_ok=True)
    argv = [
        *python_command(),
        FREEZE_COMPLETENESS,
        output.resolve().relative_to(REPO_ROOT.resolve()).as_posix(),
        "--al-screen",
        inputs["al_screen"]["path"],
        "--neck-scan",
        inputs["neck_raster"]["path"],
    ]
    result = run_step(argv)
    if result.returncode not in (0, 2):
        raise ToolingFailure(
            f"{FREEZE_COMPLETENESS} failed with exit {result.returncode}:\n"
            f"{result.stderr.strip()}"
        )
    if not output.is_file():
        raise ToolingFailure(
            f"{FREEZE_COMPLETENESS} exited {result.returncode} without writing {output}"
        )
    return {
        "step": "freeze_completeness_certificate",
        "command": argv[len(python_command()) :],
        "exit_status": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
        "produced": describe(output),
    }


def assemble_graph(
    left_birth: Path, certificate: Path, output: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Step 3: the assembler -- the only thing allowed to set release_ready."""
    output.unlink(missing_ok=True)
    relative = {
        path: path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
        for path in (left_birth, certificate, output)
    }
    env = {
        "LEFT_BIRTH": relative[left_birth],
        "COMPLETENESS": relative[certificate],
    }
    argv = [str(REPO_ROOT / ASSEMBLE_GRAPH), relative[output]]
    result = run_step(argv, env=env)
    if result.returncode not in (0, 2):
        raise ToolingFailure(
            f"{ASSEMBLE_GRAPH} failed with exit {result.returncode}:\n"
            f"{result.stdout.strip()}\n{result.stderr.strip()}"
        )
    if not output.is_file():
        raise ToolingFailure(
            f"{ASSEMBLE_GRAPH} exited {result.returncode} without writing {output}"
        )
    # The evidence set the assembler reads is owned by the shell script, not by
    # this module: ask it, so a second copy of that list cannot drift.
    listing = run_step([str(REPO_ROOT / ASSEMBLE_GRAPH)], env={**env, "PRINT_INPUTS": "1"})
    if listing.returncode != 0:
        raise ToolingFailure(
            f"{ASSEMBLE_GRAPH} could not enumerate its evidence inputs "
            f"(exit {listing.returncode})"
        )
    consumed = []
    for line in listing.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        source = REPO_ROOT / line
        consumed.append(
            {
                "path": line,
                "sha256": sha256_file(source) if source.is_file() else None,
                "present": source.is_file(),
            }
        )
    step = {
        "step": "assemble_v1_critical_graph",
        "command": [ASSEMBLE_GRAPH, relative[output]],
        "environment_overrides": env,
        "evidence_inputs": consumed,
        "exit_status": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
        "produced": describe(output),
    }
    return step, json.loads(output.read_text(encoding="utf-8"))


def release_conjuncts(graph: dict[str, Any]) -> list[dict[str, Any]]:
    """Re-derive, one by one, the conjuncts of the assembler's release_ready.

    This is a *report*, never a decision: :func:`decide` still takes its answer
    from ``graph["release_ready"]``.  The list exists so that "not release_ready"
    comes with the exact set of false conjuncts instead of a single bit, and so
    that a drift between this list and the assembler is caught (see
    :func:`decide`) rather than silently mis-reported.

    Mirrors ``scripts/assemble_critical_graph.py::main`` -- keep in step; the
    consistency check and tests/test_close_v1_gates.py enforce that.
    """
    coverage = graph.get("root_coverage") or {}
    nodes = graph.get("nodes") or []
    headline = {
        "mixed_principal_left",
        "mixed_secondary_left",
        "mixed_principal_right",
        "headline_lower_plus_one",
        "headline_upper_collision",
    }
    illegal = sorted(
        str(item.get("id")) for item in nodes if item.get("status") == "illegal"
    )
    unpassed_headline = sorted(
        str(item.get("id"))
        for item in nodes
        if str(item.get("id")) in headline and not item.get("passed")
    )
    unclassified = coverage.get("unclassified_edge_endpoints") or []
    return [
        {
            "conjunct": "no_missing_required_nodes",
            "satisfied": not (graph.get("missing_required_nodes") or []),
            "detail": graph.get("missing_required_nodes") or [],
            "explanation": "every required headline node must exist and have passed",
        },
        {
            "conjunct": "no_unexplained_nodes",
            "satisfied": not (graph.get("unexplained_nodes") or []),
            "detail": graph.get("unexplained_nodes") or [],
            "explanation": "no node may be left unresolved or illegal",
        },
        {
            "conjunct": "no_illegal_nodes",
            "satisfied": not illegal,
            "detail": illegal,
            "explanation": "Newton-failed is not an allowed endpoint class",
        },
        {
            "conjunct": "root_coverage_complete",
            "satisfied": bool(coverage.get("complete")),
            "detail": {
                "localized_roots": coverage.get("localized_roots"),
                "required_cells": coverage.get("required_cells"),
            },
            "explanation": "all 620 source transition cells must be localized",
        },
        {
            "conjunct": "at_least_one_edge",
            "satisfied": int(coverage.get("edge_count") or 0) >= 1,
            "detail": coverage.get("edge_count"),
            "explanation": "the graph must carry at least one mechanism polyline",
        },
        {
            "conjunct": "all_cells_on_edges",
            "satisfied": int(coverage.get("cells_on_edges") or -1) == 620,
            "detail": coverage.get("cells_on_edges"),
            "explanation": "every localized cell must sit on exactly one polyline",
        },
        {
            "conjunct": "no_duplicate_cells",
            "satisfied": not (coverage.get("duplicate_cell_ids") or []),
            "detail": coverage.get("duplicate_cell_ids") or [],
            "explanation": "a cell may not appear on two polylines",
        },
        {
            "conjunct": "no_missing_mixed_germs",
            "satisfied": not (coverage.get("missing_mixed_germs") or []),
            "detail": coverage.get("missing_mixed_germs") or [],
            "explanation": "every retained organizer owes G+/G- germs in both directions",
        },
        {
            "conjunct": "no_newton_failed_roots",
            "satisfied": int(coverage.get("newton_failed") or 0) == 0,
            "detail": coverage.get("newton_failed"),
            "explanation": "a Newton failure is a diagnostic, never a classification",
        },
        {
            "conjunct": "no_unclassified_edge_endpoints",
            "satisfied": not unclassified,
            "detail": [
                {
                    "edge": entry.get("edge"),
                    "side": entry.get("side"),
                    "reserved_for": entry.get("reserved_for"),
                }
                for entry in unclassified
            ],
            "explanation": "every polyline end must attach to a node",
        },
        {
            "conjunct": "no_classification_binding_errors",
            "satisfied": not (coverage.get("classification_binding_errors") or []),
            "detail": coverage.get("classification_binding_errors") or [],
            "explanation": "classification endpoint bindings must resolve uniquely",
        },
        {
            "conjunct": "completeness_certificate_verified",
            "satisfied": bool(coverage.get("completeness_passed")),
            "detail": coverage.get("completeness_verification_errors") or [],
            "explanation": (
                "the assembler must re-read, re-hash and re-derive every source "
                "the completeness certificate names"
            ),
        },
        {
            "conjunct": "all_headline_nodes_passed",
            "satisfied": not unpassed_headline,
            "detail": unpassed_headline,
            "explanation": "the five headline nodes must each carry passed=true",
        },
    ]


def decide(
    graph: dict[str, Any], assembler_exit: int
) -> tuple[bool, list[dict[str, Any]]]:
    """Three independent readings must agree, or this is a tooling failure.

    (a) the assembler's exit status, (b) the ``release_ready`` bit it wrote, and
    (c) the conjuncts re-derived from the artifact.  Disagreement is never
    resolved in favour of the friendlier answer.
    """
    conjuncts = release_conjuncts(graph)
    blockers = [entry for entry in conjuncts if not entry["satisfied"]]
    declared = graph.get("release_ready") is True
    if declared != (assembler_exit == 0):
        raise ToolingFailure(
            f"assembler exit {assembler_exit} disagrees with release_ready="
            f"{graph.get('release_ready')!r}"
        )
    if declared != (not blockers):
        raise ToolingFailure(
            "the re-derived conjunct list disagrees with the assembler's "
            f"release_ready={graph.get('release_ready')!r}; false conjuncts: "
            + ", ".join(entry["conjunct"] for entry in blockers)
            + ". scripts/close_v1_gates.py::release_conjuncts has drifted from "
            "scripts/assemble_critical_graph.py::main and must be re-synced "
            "before any closure claim is made."
        )
    return declared, blockers


def render(summary: dict[str, Any]) -> str:
    lines = [
        "=" * 72,
        "v1 closure chain",
        "=" * 72,
        "",
        "Inputs (sha256 / CI run):",
    ]
    for entry in summary["inputs"]:
        lines.append(f"  {entry['role']:<14} {entry['path']}")
        lines.append(f"  {'':<14} sha256 {entry['sha256']}  run {entry['ci_run_id']}")
    lines.append("")
    lines.append("Produced by the producing scripts:")
    for step in summary["steps"]:
        produced = step.get("produced")
        status = f"exit {step['exit_status']}"
        if produced:
            lines.append(f"  {step['step']:<36} {status}  {produced['path']}")
            lines.append(f"  {'':<36} sha256 {produced['sha256']}")
        else:
            lines.append(f"  {step['step']:<36} {status}  (no artifact)")
    lines.append("")
    lines.append(f"release_ready: {summary['release_ready']}")
    if summary["release_ready"]:
        lines.append("")
        lines.append("Every conjunct of the assembler's release gate is satisfied.")
        return "\n".join(lines) + "\n"
    lines.append("")
    lines.append(f"BLOCKERS ({len(summary['blockers'])}):")
    for entry in summary["blockers"]:
        lines.append(f"  x {entry['conjunct']}")
        lines.append(f"      {entry['explanation']}")
        detail = json.dumps(entry["detail"])
        if len(detail) > 400:
            detail = detail[:397] + "..."
        lines.append(f"      false because: {detail}")
    if summary.get("halted_because"):
        lines.append("")
        lines.append(
            "The chain halted before assembly, so the remaining conjuncts were "
            "not evaluated. They are not thereby satisfied."
        )
    lines.append("")
    lines.append("Not release_ready. This is an honest open scientific state.")
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the v1 closure chain from harvested CI artifacts to a decision. "
            "Every input is required; there is no partial mode."
        )
    )
    for role, flag, description in HARVESTED_INPUTS:
        parser.add_argument(flag, dest=role, help=description)
        parser.add_argument(
            f"{flag}-run-id",
            dest=f"{role}_run_id",
            help=f"GitHub Actions run id that produced {flag}",
        )
    parser.add_argument(
        "--evidence-dir",
        default="research/evidence",
        help=(
            "where the producing scripts write their records "
            "(default research/evidence; point this at a tmp dir for fixtures)"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        inputs = collect_inputs(args)
    except Refusal as exc:
        print("REFUSED: the closure chain will not run on an incomplete input set.", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        print(
            "\nNothing was written. Supply every artifact and its CI run id; a "
            "partial closure is a refusal, not a weaker invocation.",
            file=sys.stderr,
        )
        return REFUSED

    by_role = {entry["role"]: entry for entry in inputs}
    evidence_dir = Path(args.evidence_dir)
    if not evidence_dir.is_absolute():
        evidence_dir = REPO_ROOT / evidence_dir
    try:
        evidence_dir.resolve().relative_to(REPO_ROOT.resolve())
    except ValueError:
        print(
            f"REFUSED: --evidence-dir {args.evidence_dir!r} is outside the repository.",
            file=sys.stderr,
        )
        return REFUSED
    evidence_dir.mkdir(parents=True, exist_ok=True)

    steps: list[dict[str, Any]] = []
    halted: list[str] = []
    summary: dict[str, Any]
    try:
        class_step, refusal = classify_left_birth(
            by_role, evidence_dir / LEFT_BIRTH_CLASS
        )
        steps.append(class_step)
        if refusal is not None:
            halted.append(refusal)
            summary = {
                "schema": "atlas.v1.closure-provenance/1",
                "inputs": inputs,
                "steps": steps,
                "release_ready": False,
                "blockers": [
                    {
                        "conjunct": "secondary_left_birth_classified",
                        "satisfied": False,
                        "detail": refusal,
                        "explanation": (
                            "the fold classifier refused these artifacts; the chain "
                            "stops rather than assembling with a stale or absent "
                            "classification"
                        ),
                    }
                ],
                "halted_because": halted,
                "graph": None,
            }
        else:
            steps.append(
                freeze_completeness(by_role, evidence_dir / COMPLETENESS_CERTIFICATE)
            )
            assemble_step, graph = assemble_graph(
                evidence_dir / LEFT_BIRTH_CLASS,
                evidence_dir / COMPLETENESS_CERTIFICATE,
                evidence_dir / CRITICAL_GRAPH,
            )
            steps.append(assemble_step)
            release_ready, blockers = decide(graph, assemble_step["exit_status"])
            summary = {
                "schema": "atlas.v1.closure-provenance/1",
                "inputs": inputs,
                "steps": steps,
                "release_ready": release_ready,
                "blockers": blockers,
                "halted_because": [],
                "graph": assemble_step["produced"],
            }
    except ToolingFailure as exc:
        print(f"TOOLING FAILURE: {exc}", file=sys.stderr)
        print(
            "\nThis is not an open scientific state; it is a broken chain. "
            "No closure claim may be made from this run.",
            file=sys.stderr,
        )
        return TOOLING_FAILURE

    ledger = evidence_dir / CLOSURE_LEDGER
    ledger.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(render(summary))
    print(f"provenance ledger: {ledger.relative_to(REPO_ROOT)}")
    if evidence_dir.resolve() == (REPO_ROOT / "research/evidence").resolve():
        # Writing here changes which artifacts the graph was assembled from, so
        # the pinned invocation has to be told about it in the same commit or
        # critical-graph-assembly.yml will correctly report the committed graph
        # as stale.
        print(
            "\nThese records went into the durable evidence directory. Before "
            "committing, update the pinned --left-birth and completeness inputs "
            f"in {ASSEMBLE_GRAPH} to {LEFT_BIRTH_CLASS} and "
            f"{COMPLETENESS_CERTIFICATE}, so the pinned invocation and this "
            "closure agree about the evidence set."
        )
    return 0 if summary["release_ready"] else NOT_RELEASE_READY


if __name__ == "__main__":
    raise SystemExit(main())
