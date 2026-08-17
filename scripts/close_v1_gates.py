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

WHERE THE TRUST BOUNDARY ACTUALLY IS
------------------------------------
An earlier version of this file got the boundary wrong, and a reviewer walked
straight through the hole.  The mistake is worth recording because the fix only
makes sense against it.

Every gate in this project is enforced by *code reading bytes*.  Pinning the
bytes -- which this runner already did, with a sha256 and a CI run id per input
-- pins only half of that.  The other half is the code, and "the code" is two
things: the producing scripts, and the interpreter that executes them.  The old
``python_command()`` took the interpreter from ``$PYTHON`` and then deliberately
did *not* record it, arguing that an environment detail in the ledger would cost
byte-reproducibility across machines.  That argument traded auditability for a
cosmetic property and it was the wrong trade: a ~20-line bash shim on ``$PYTHON``
that forwarded most invocations to the real interpreter and substituted passing
output for ``classify_secondary_left_birth.py`` was completely invisible in
V1_CLOSURE_PROVENANCE.json.  rc=0, release_ready=true, forged fold, and a ledger
that named only honest sha256s.

So the boundary this runner defends is: **the deciding code is the interpreter
the operator personally invoked, running the producing scripts exactly as they
are committed at HEAD.**  Three mechanisms, in decreasing order of strength:

1. ``$PYTHON`` no longer selects anything.  The producing scripts run under
   ``sys.executable`` -- the interpreter already running this file -- and the
   runner exports that absolute path to ``assemble_v1_critical_graph.sh`` so the
   shell cannot fall back to ``uv run`` or to an inherited hostile ``$PYTHON``
   either.  If ``$PYTHON`` is set to anything that is not this same interpreter,
   the run is REFUSED (exit 64) rather than silently honoured: an operator who
   wants a throwaway venv invokes *this script* with that venv's python, which
   makes the choice an explicit act at the command line instead of an ambient
   variable a shim can own.  ``PYTHON=<venv>/bin/python close_v1_gates.py`` is
   not equivalent to ``<venv>/bin/python close_v1_gates.py``, and only the
   second is accepted.
2. The producing scripts are digest-verified before they are invoked.  Each of
   the four (classifier, freezer, assembler shell, assembler) is hashed on disk
   and compared against the bytes of the same path at ``git HEAD``.  A shim that
   swaps the *script* instead of the interpreter is caught here.  A dirty
   working copy of a producing script is a refusal: closing a gate with
   uncommitted deciding code is not a closure anybody can re-derive.  The same
   rule covers ``src/threebody_atlas/completeness.py``, which is where the
   freezer's gates actually live -- ``$PYTHONPATH`` precedes site-packages, so
   one directory on it would otherwise replace those gates for every subprocess
   in the chain.
3. Everything that could not be prevented is recorded, loudly, in the ledger's
   ``environment`` block: the interpreter's absolute path, its own sha256, its
   version banner, whether it lives inside the working tree, the raw ``$PYTHON``
   that was seen and rejected, the git HEAD, and both digests of every producing
   script.  An auditor reading the ledger can see a substituted interpreter.

What this cannot do, stated plainly rather than papered over: a program cannot
bootstrap trust in the machinery that runs it (Thompson, *Reflections on
Trusting Trust*).  If ``sys.executable`` is itself a hostile build, or ``.git``
has been rewritten, every check above is executed by the attacker.  Mechanism 3
is the answer to that -- not prevention, but an unavoidable record.  The
strongest honest claim is: **a substitution that changes the decision cannot
also stay out of the ledger.**

WHAT THE ASSEMBLER CANNOT DO, AND WHERE THAT DEFENCE MOVED
----------------------------------------------------------
``assemble_critical_graph.load_classification`` reads a JSON file and believes
its ``evidence_level``.  A hand-written classification carrying ``passed: true``,
an allowed class and ``evidence_level: "independently_reproduced"`` therefore
produces a *passed* node.  That is not a bug in the assembler and it is not
fixable there: at that layer the artifact is just bytes, and there is nothing in
it to distinguish a forgery from a real classification.  ``tests/`` asserts this
behaviour honestly rather than pretending otherwise.

The defence lives here instead, at the only layer that knows where a
classification came from.  The runner deletes the output path, invokes the
digest-verified classifier under the recorded interpreter, and binds what came
back to its origin in the ledger: which script produced it (both digests), from
which input artifacts (sha256 + CI run id), under which interpreter.  A planted
file does not survive the unlink; a forged file the classifier refused to write
is deleted again; and a classification whose recorded producer does not match
HEAD is a refusal before any of it runs.

EXIT STATUS
-----------
  0   release_ready
  2   the chain ran (or a producing script legitimately refused) and the result
      is not release_ready -- an honest open scientific state, with the exact
      blocker list printed
  3   tooling failure: a producing script crashed, or the three independent
      readings of the decision disagreed
  64  refused: a required input is missing, unreadable, duplicated across roles,
      or outside the repository; ``$PYTHON`` names an interpreter other than the
      one running this file; or a producing script does not match HEAD

REPRODUCIBILITY
---------------
The ledger is split in two, and the split is the point.

``environment`` is machine-varying by construction: interpreter path,
interpreter sha256, version banner, git HEAD.  It is *supposed* to differ
between machines -- that is what makes it evidence.

Everything else -- ``inputs``, ``steps``, ``blockers``, ``graph`` -- stays a
pure function of the input bytes: no timestamps, no hostnames, no absolute
paths, no run ordering.  Input paths are normalised to repository-relative
POSIX strings before they are handed to the producing scripts, because those
strings are embedded verbatim in the certificate's ``sources`` block and in the
graph's ``evidence`` fields -- and because
``threebody_atlas.completeness.verify_certificate`` can only re-resolve and
re-hash a source that lives inside the repository.  ``evidence_digest`` is a
sha256 over exactly that byte-stable half, so two machines that agree on the
science produce the same digest while still each disclosing their own
environment.  Running this twice on one machine yields byte-identical files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
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

#: The code that actually decides.  These four are hashed on disk and compared
#: against their bytes at ``git HEAD`` before any of them is invoked, because
#: swapping a *script* is the same attack as swapping the interpreter and must
#: fail the same way.  ``scripts/assemble_critical_graph.py`` is in the set even
#: though this module never spawns it directly: the shell wrapper does, and the
#: wrapper is only worth pinning if what it runs is pinned too.
PRODUCING_SCRIPTS = (
    CLASSIFY_LEFT_BIRTH,
    FREEZE_COMPLETENESS,
    ASSEMBLE_GRAPH,
    "scripts/assemble_critical_graph.py",
    # graph_root_sources.py is the sole producer of the combined roots file that
    # BOTH pinned --sign-topology audits name as their inputs.roots, and the only
    # correct resolver of supplemental cell ids >= 10000 (they are per-run
    # sequential, so a glob-newest resolution silently repoints them).  It was
    # absent from this pin while already deciding what the audits audited --
    # found by the 2026-08-17 consolidation audit, issue #212 section 6.
    "scripts/graph_root_sources.py",
    # build_continuation_arc_roots.py is the producer of the arc-roots evidence
    # the canonical invocation consumes.  The artifact existed with no producer
    # in the repository at all (it had been generated inline); the script now
    # regenerates it byte-for-byte and carries a --check mode.
    "scripts/build_continuation_arc_roots.py",
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
    problems.extend(duplicate_input_problems(records))
    if problems:
        raise Refusal("\n".join(f"  - {problem}" for problem in problems))
    return records


def duplicate_input_problems(records: list[dict[str, Any]]) -> list[str]:
    """Refuse an input set in which one artifact is doing two jobs.

    The four roles are four *independent* computations.  If two of them resolve
    to the same file, or to different files with the same bytes, then whatever
    the chain concludes rests on one measurement wearing two hats -- the fold
    geometry screen confirming itself as its own independent BigFloat check, or
    the AL pocket screen standing in for the neck raster.  The chain would run
    happily and report a coherent-looking result, which is exactly why this has
    to be a refusal and not a warning.

    The third case is subtler and is the one a forger actually reaches for:
    identical bytes offered under two different ``ci_run_id`` values.  One
    artifact cannot have been produced by two runs, so at least one of the run
    ids is a fiction, and a fictitious run id defeats the only handle an auditor
    has for going back to CI and re-downloading the artifact to compare.
    """
    problems: list[str] = []

    by_path: dict[str, list[str]] = {}
    for entry in records:
        by_path.setdefault(entry["path"], []).append(entry["role"])
    for path, roles in sorted(by_path.items()):
        if len(roles) > 1:
            problems.append(
                f"roles {', '.join(sorted(roles))} resolve to {path!r}. Each role "
                "is a separate computation; one file cannot be its own independent "
                "cross-check."
            )

    by_digest: dict[str, list[dict[str, Any]]] = {}
    for entry in records:
        by_digest.setdefault(entry["sha256"], []).append(entry)
    for digest, group in sorted(by_digest.items()):
        if len(group) < 2:
            continue
        roles = sorted(entry["role"] for entry in group)
        run_ids = sorted({entry["ci_run_id"] for entry in group})
        if len({entry["path"] for entry in group}) > 1:
            problems.append(
                f"roles {', '.join(roles)} are different paths with identical bytes "
                f"(sha256 {digest}). Copying one artifact into two roles does not "
                "make it two independent results."
            )
        if len(run_ids) > 1:
            problems.append(
                f"sha256 {digest} is presented under {len(run_ids)} different CI run "
                f"ids ({', '.join(run_ids)}) for roles {', '.join(roles)}. One "
                "artifact has one producing run; at least one of these run ids is "
                "fabricated, and a fabricated run id cannot be checked against CI."
            )

    return problems


def python_command() -> list[str]:
    """The interpreter used for the producing scripts: this one, always.

    ``$PYTHON`` used to select it.  It no longer selects anything, because an
    ambient variable naming the interpreter is an ambient variable naming the
    code that decides -- and a bash shim on that variable forged a pass while
    leaving the ledger honest-looking.  Choosing a throwaway venv is still
    supported, and is now an explicit act: invoke *this script* with that venv's
    python.  :func:`resolve_interpreter` refuses a ``$PYTHON`` that disagrees.
    """
    return [sys.executable]


def resolve_interpreter() -> dict[str, Any]:
    """Pin the interpreter to ``sys.executable`` and describe it for the ledger.

    Refuses when ``$PYTHON`` names anything else.  The refusal is deliberate
    rather than a silent override: an operator who set ``$PYTHON`` on purpose
    believes the producing scripts run under it, and quietly running them under
    a different interpreter would be its own kind of dishonesty.  The fix is one
    word of command line, and it moves the choice somewhere a shim cannot own.

    Everything recorded here is machine-varying on purpose; it lands in the
    ledger's ``environment`` block, which is excluded from ``evidence_digest``.
    """
    interpreter = Path(sys.executable).resolve()
    raw = os.environ.get("PYTHON")
    if raw and raw.strip():
        declared = raw.split()
        program = shutil.which(declared[0]) if declared else None
        resolved = Path(program).resolve() if program else None
        if len(declared) != 1 or resolved != interpreter:
            raise Refusal(
                f"PYTHON={raw!r} names an interpreter other than the one running "
                f"this script ({interpreter}). The producing scripts are run under "
                "sys.executable and nothing else: an environment variable that "
                "selects the deciding code is a substitution channel, and a shim on "
                "it once forged release_ready=true invisibly. To use a different "
                f"interpreter, invoke this script with it directly:\n"
                f"      {declared[0]} {Path(__file__).name} ...\n"
                "    and unset PYTHON."
            )

    inside_tree = interpreter.is_relative_to(REPO_ROOT.resolve()) or Path(
        sys.executable
    ).is_relative_to(REPO_ROOT.resolve())
    return {
        "path": str(interpreter),
        # sys.executable is usually a venv symlink into a system framework
        # build; both ends are recorded because a reviewer recognises one and an
        # auditor re-hashing the binary needs the other.
        "sys_executable": sys.executable,
        "sha256": sha256_file(interpreter) if interpreter.is_file() else None,
        "version": sys.version.replace("\n", " "),
        "implementation": platform.python_implementation(),
        "inside_working_tree": inside_tree,
        "python_env_var": raw,
        "note": (
            "The producing scripts ran under this interpreter. It is recorded "
            "because a substituted interpreter is otherwise undetectable in the "
            "artifacts it produces."
            + (
                " It lives inside the working tree. That is normal for a venv "
                "built by 'uv sync --locked', and it is also where a substituted "
                "interpreter would sit, so an auditor quoting this closure should "
                "confirm the venv was built from the lockfile rather than shipped "
                "with the checkout."
                if inside_tree
                else ""
            )
        ),
    }


def git_show(relative: str) -> bytes | None:
    """The bytes of ``relative`` as committed at HEAD, or None if unavailable."""
    result = subprocess.run(  # noqa: S603 - fixed argv
        ["git", "cat-file", "blob", f"HEAD:{relative}"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    return result.stdout if result.returncode == 0 else None


def verify_producing_scripts() -> list[dict[str, Any]]:
    """Refuse unless every deciding script on disk is its committed self.

    This is the second half of the trust boundary.  Pinning the interpreter
    stops a shim that swaps the *runner* of the code; this stops a shim that
    swaps the *code*, which is the same attack one layer down and must fail the
    same way.  HEAD is the reference rather than a hand-maintained table of
    digests, because a table in this file is one more thing an attacker can edit
    in the same commit, and because it would go stale every time a producing
    script legitimately changes.

    A modified-but-uncommitted producing script is refused too.  A closure run
    is a claim that a specific revision of specific code reached a specific
    conclusion; nobody can re-derive that from a working copy.
    """
    head = subprocess.run(  # noqa: S603 - fixed argv
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if head.returncode != 0:
        raise Refusal(
            "the producing scripts cannot be verified against a committed "
            "revision: this is not a git checkout (git rev-parse HEAD failed: "
            f"{head.stderr.strip()}). Run the closure from a checkout; the whole "
            "claim is that a named revision of the deciding code reached this "
            "conclusion."
        )

    problems: list[str] = []
    records: list[dict[str, Any]] = []
    for relative in PRODUCING_SCRIPTS:
        path = REPO_ROOT / relative
        if not path.is_file():
            problems.append(f"{relative} is missing from the working tree")
            continue
        on_disk = sha256_file(path)
        blob = git_show(relative)
        committed = hashlib.sha256(blob).hexdigest() if blob is not None else None
        if committed is None:
            problems.append(
                f"{relative} is not tracked at HEAD, so there is nothing to verify "
                "it against"
            )
            continue
        if on_disk != committed:
            problems.append(
                f"{relative} does not match HEAD: on disk {on_disk}, committed "
                f"{committed}. A producing script is the code that decides a gate; "
                "if the copy that ran is not the copy under review, the result is "
                "unreviewable. Commit the change (and get it reviewed) or restore "
                "the file."
            )
        records.append(
            {
                "path": relative,
                "sha256": on_disk,
                "sha256_at_head": committed,
                "matches_head": on_disk == committed,
            }
        )
    if problems:
        raise Refusal("\n".join(f"  - {problem}" for problem in problems))
    return records


#: The freezer does not implement the completeness gates; it imports them.  So
#: this module is deciding code too, reached through a different substitution
#: channel: ``$PYTHONPATH``, which takes precedence over everything installed.
COMPLETENESS_MODULE = "src/threebody_atlas/completeness.py"
COMPLETENESS_IMPORT = "threebody_atlas/completeness.py"


def verify_completeness_module() -> list[dict[str, Any]]:
    """Refuse a ``threebody_atlas.completeness`` that is not the committed one.

    ``freeze_completeness_certificate.py`` is a thin shell around
    ``build_record``/``seal``/``verify_certificate``; pinning the freezer while
    leaving the module it imports unpinned would pin the wrapper and not the
    gates.  ``$PYTHONPATH`` is the channel that matters -- it precedes
    site-packages, so one directory on it replaces the completeness predicates
    for every subprocess in the chain.

    Candidates are found by scanning the search path rather than importing,
    because importing to find out whether the import is safe runs the code
    first.  Every reachable copy must hash to the committed bytes; a stale
    installed copy that disagrees with HEAD is a genuine ambiguity about which
    gates ran, and a closure cannot rest on an ambiguity.
    """
    blob = git_show(COMPLETENESS_MODULE)
    if blob is None:
        raise Refusal(
            f"{COMPLETENESS_MODULE} is not tracked at HEAD, so the completeness "
            "gates cannot be pinned to a reviewed revision"
        )
    committed = hashlib.sha256(blob).hexdigest()

    search: list[str] = []
    for entry in (os.environ.get("PYTHONPATH") or "").split(os.pathsep):
        if entry:
            search.append(entry)
    search.append(str(REPO_ROOT / "src"))
    search.extend(path for path in sys.path if path)

    problems: list[str] = []
    records: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for entry in search:
        candidate = Path(entry) / COMPLETENESS_IMPORT
        try:
            resolved = candidate.resolve()
        except OSError:  # pragma: no cover - unreadable path entry
            continue
        if not candidate.is_file() or resolved in seen:
            continue
        seen.add(resolved)
        on_disk = sha256_file(candidate)
        if on_disk != committed:
            problems.append(
                f"{candidate} is importable as threebody_atlas.completeness but "
                f"does not match {COMPLETENESS_MODULE} at HEAD (on disk {on_disk}, "
                f"committed {committed}). The completeness gates are decided "
                "there; an unreviewed copy on the import path decides them "
                "instead. Remove it from PYTHONPATH, or commit it."
            )
        records.append(
            {
                "path": str(candidate),
                "sha256": on_disk,
                "sha256_at_head": committed,
                "matches_head": on_disk == committed,
            }
        )
    if problems:
        raise Refusal("\n".join(f"  - {problem}" for problem in problems))
    return records


def run_step(argv: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    merged = dict(os.environ)
    if env:
        merged.update(env)
    # The shell wrapper resolves its own interpreter from $PYTHON, defaulting to
    # `uv run --no-sync python`.  Overwrite it with the pinned absolute path so
    # the wrapper cannot inherit a hostile value or silently start a second,
    # unrecorded interpreter of its own.
    merged["PYTHON"] = sys.executable
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
            "detail": {
                "cells_on_edges": coverage.get("cells_on_edges"),
                "supplemental_roots": coverage.get("supplemental_roots"),
                "all_vertices_on_edges": coverage.get("all_vertices_on_edges"),
            },
            "explanation": "every catalog S/U cell 0..619 must sit on exactly one polyline",
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
            "conjunct": "sign_topology_clean",
            "satisfied": bool(coverage.get("sign_topology_clean")),
            "detail": coverage.get("sign_topology_errors") or [],
            "explanation": (
                "a sign-vector face-consistency audit must show no critical curve "
                "outside the committed edges; fail-closed, because not having "
                "looked is not the same as having looked and found nothing"
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
    ]
    environment = summary.get("environment") or {}
    interpreter = environment.get("interpreter") or {}
    if interpreter:
        # Printed, not just filed: the reviewer reading a step summary is the
        # person most likely to notice that this is not the interpreter they
        # expected, and they will only notice if it is in front of them.
        lines.append("Deciding code (machine-varying; see ledger.environment):")
        lines.append(f"  interpreter    {interpreter.get('path')}")
        lines.append(f"  {'':<14} sha256 {interpreter.get('sha256')}")
        if interpreter.get("python_env_var"):
            lines.append(
                f"  {'':<14} $PYTHON was {interpreter['python_env_var']!r} (accepted: "
                "it names this same interpreter)"
            )
        if interpreter.get("inside_working_tree"):
            lines.append(
                f"  {'':<14} inside the working tree (normal for a lockfile-built "
                "venv; confirm it was built, not shipped)"
            )
        for entry in environment.get("producing_scripts") or []:
            lines.append(
                f"  {entry['path']:<44} {'matches HEAD' if entry['matches_head'] else 'DIFFERS FROM HEAD'}"
            )
        lines.append("")
    lines.append("Inputs (sha256 / CI run):")
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


def evidence_digest(summary: dict[str, Any]) -> str:
    """sha256 over the byte-stable half of the ledger.

    ``environment`` is excluded because it is *meant* to differ per machine.
    Everything else is a pure function of the input bytes, so two honest runs on
    two machines agree on this digest while each still discloses its own
    interpreter -- which is the whole point of splitting the ledger instead of
    dropping the environment to keep a clean hash.
    """
    stable = {
        key: value
        for key, value in summary.items()
        if key not in ("environment", "evidence_digest")
    }
    payload = json.dumps(stable, indent=2, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        # Order matters only for the error the operator sees first, and the
        # interpreter comes first on purpose: if the deciding code is being run
        # by something unexpected, no amount of correct input matters.
        interpreter = resolve_interpreter()
        producing_scripts = verify_producing_scripts()
        importable_completeness = verify_completeness_module()
        inputs = collect_inputs(args)
    except Refusal as exc:
        print(
            "REFUSED: the closure chain will not run unless the deciding code and "
            "the input set are both pinned.",
            file=sys.stderr,
        )
        print(str(exc), file=sys.stderr)
        print(
            "\nNothing was written. Supply every artifact and its CI run id, run "
            "the committed producing scripts, and let the interpreter you invoked "
            "be the one that runs them; a partial or unpinned closure is a "
            "refusal, not a weaker invocation.",
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
                "schema": "atlas.v1.closure-provenance/2",
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
                "schema": "atlas.v1.closure-provenance/2",
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

    summary["evidence_digest"] = evidence_digest(summary)
    # Appended, not interleaved: the machine-varying facts live in exactly one
    # place so that "which parts of this ledger are supposed to differ between
    # machines" needs no judgement call from the reader.
    summary["environment"] = {
        "note": (
            "Machine-varying by construction and excluded from evidence_digest. "
            "It is here because a substituted interpreter or a swapped producing "
            "script is otherwise invisible in the records this chain produces."
        ),
        "interpreter": interpreter,
        "producing_scripts": producing_scripts,
        "importable_completeness_module": importable_completeness,
        "pythonpath": os.environ.get("PYTHONPATH"),
        "git_head": subprocess.run(  # noqa: S603 - fixed argv
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip(),
        "runner": {
            "path": Path(__file__).resolve().relative_to(REPO_ROOT.resolve()).as_posix(),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }

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
