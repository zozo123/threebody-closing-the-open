#!/usr/bin/env python3
"""Refuse to let manuscript prose outrun the committed numerical evidence.

The repository's stated design is that ``paper/main.tex`` has "explicit
generated-result gates so prose cannot silently outrun the numerical evidence".
That was only true of the two ``\\input``-ed generated sections.  Nothing read
``main.tex`` itself, so a hardcoded sentence asserting the v1 problem was solved
built green while ``research/evidence/V1_CRITICAL_GRAPH.json`` had
``release_ready`` false.  This script closes that hole.

Design: a lexicon grep alone is brittle in both directions.  It fires on negated
prose ("the general three-body problem is not being solved") and it misses any
paraphrase an author invents.  So the check is four independent layers, and the
load-bearing ones do not depend on wording at all.

  Layer 1 -- generated channel integrity (wording independent).
      Every file under ``paper/generated/`` must byte-match what
      ``scripts/build_discovery_release.py`` emits right now, and no other file
      may live there.  A claim can therefore only enter the manuscript through
      the generator, from a manifest record.

  Layer 2 -- hand-written source lock (wording independent).
      Every sentence of hand-written manuscript prose -- preamble included, and
      including *both* branches of every generator-controlled conditional -- is
      normalized and hashed into ``paper/CLAIM_LOCK.json`` together with the
      evidence fingerprint it was reviewed under.  Preamble control lines, where
      there is no prose to hash, are locked verbatim instead, so a
      ``\\renewcommand`` that overrides a generated gate is an unreviewed line.
      Freezing is refused outright when the evidence has weakened since the last
      freeze, so prose can never be re-blessed on the way down.

  Layer 3 -- claim-vocabulary screen (best effort, evidence conditional).
      Fast, precise feedback at authoring time.  It scans hand-written prose and
      every manifest string the LaTeX renderers actually typeset, applies
      negation and scoping analysis so honest disclaimers pass, and evaluates
      each fragment against *the evidence state under which that fragment is
      printed*.  A gated branch is not exempt: it is screened under the state
      that selects it, and the branch that a future solved/release-ready state
      would print is screened under that hypothetical state as well, so a claim
      parked in a dormant branch cannot become true prose the day the assembler
      flips a bit.

  Layer 4 -- generated-namespace integrity (wording independent).
      Hand-written sources may define ``\\atlas...`` macros only inside the
      documented fallback block, only with ``\\newcommand``, and the fallback
      must collapse every conditional to its weakest branch.  Without this,
      ``\\renewcommand{\\atlasifsolved}[2]{#1}`` in the preamble would print the
      solved branch while the assembler still says the graph is not ready.

Layer 3 catches the obvious mistake immediately; Layers 1, 2 and 4 catch
everything else by construction, including paraphrases Layer 3 has never heard
of.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, NamedTuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from threebody_atlas.discovery import (  # noqa: E402
    evidence_state,
    load_manifest,
    render_latex_claims,
    render_latex_macros,
    render_latex_status,
)

MANIFEST = Path("research/DISCOVERY_RELEASE.json")
GENERATED_DIR = Path("paper/generated")
LOCK_PATH = Path("paper/CLAIM_LOCK.json")

# Files under paper/generated/ and the renderer that owns each one.  A file in
# that directory with no owner is itself an error: it is hand-written prose
# wearing a machine-authored costume.  This mapping is also what drives the
# manifest screen in Layer 3 -- these renderers are, by definition, everything
# the manuscript typesets out of the manifest -- so a new generated section
# cannot appear without its manifest fields being screened too.
GENERATED_OWNERS: dict[str, Callable[[dict[str, Any], Path], str]] = {
    "discovery-release.tex": lambda manifest, root: render_latex_status(manifest),
    "discoveries.tex": lambda manifest, root: render_latex_claims(manifest),
    "claim-macros.tex": render_latex_macros,
}

# Hand-written manuscript sources.  Everything here is locked and screened.
HANDWRITTEN = (
    Path("paper/main.tex"),
    Path("paper/handwritten/reproduction.tex"),
)

# Scope statements the manuscript must carry.  Keyed by marker rather than by
# literal sentence so the wording can be improved without defeating the check,
# while deletion of the disclaimer itself still fails.
REQUIRED_SCOPE_TAGS = {
    "frozen-catalog-sheet": "one continuation-connected sheet of the frozen catalog",
    "declared-mass-box": "the declared mass box the results are confined to",
    "linear-floquet-only": "planar linear Floquet stability only",
    "bounded-completeness": "completeness is bounded, not global",
    "not-general-three-body": "this is not a solution of the general three-body problem",
}
SCOPE_TAG_RE = re.compile(r"%\s*ATLAS-SCOPE:\s*([a-z0-9-]+)")

# Vocabulary that asserts closure.  Each entry is (name, pattern, requirement),
# where requirement names the evidence fact that would have to hold for the
# assertion to be legitimate.  This lexicon is deliberately not the load-bearing
# layer: it is fast feedback, and Layers 1, 2 and 4 catch what it misses.
CLAIM_PATTERNS: tuple[tuple[str, str, str], ...] = (
    ("solvedness", r"\b(?:is|are|was|has been|have been|now)\s+(?:hereby\s+)?solved\b", "solved"),
    ("solvedness", r"\bwe\s+(?:have\s+)?solved\b", "solved"),
    ("solvedness", r"\b(?:a|the)\s+solution\s+of\s+the\s+(?:general\s+)?(?:newtonian\s+)?three-body\b", "never"),
    ("closure", r"\b(?:problem|question|graph|atlas)\s+is\s+(?:thereby\s+)?(?:closed|complete|completed|finished)\b", "solved"),
    ("closure", r"\bclosure\s+is\s+(?:achieved|established|proved|proven)\b", "solved"),
    ("graph-complete", r"\b(?:complete|completed|full|finished|closed)\b[^.]{0,60}?\bcritical\s+graph\b", "release_ready"),
    ("graph-complete", r"\bcritical\s+graph\b[^.]{0,40}?\bis\s+(?:complete|completed|closed|finished|done)\b", "release_ready"),
    ("graph-complete", r"\b(?:computed|constructed|established)\b[^.]{0,60}?\bconnected\s+stability-boundary\s+manifold\b", "release_ready"),
    ("census-complete", r"\ball\s+(?:620|six\s+hundred\s+and\s+twenty)\b[^.]{0,80}?\b(?:frozen|classified|assigned|resolved|localized)\b", "release_ready"),
    ("census-complete", r"\b620\s*/\s*620\b", "release_ready"),
    ("census-complete", r"\bcensus\b[^.]{0,40}?\b(?:is|was|has been)\s+(?:complete|completed|finished|closed)\b", "release_ready"),
    ("germs-complete", r"\ball\s+twelve\b", "release_ready"),
    ("germs-complete", r"\b(?:twelve|12)\b[^.]{0,40}?\bgerms?\b[^.]{0,40}?\b(?:is|are|were|have been)\s+(?:frozen|complete|completed|regenerated|closed|in\s+place)\b", "release_ready"),
    ("germs-complete", r"\b(?:every|each)\s+(?:mixed\s+)?germ\b[^.]{0,40}?\b(?:is|are)\s+(?:frozen|complete|completed|present)\b", "release_ready"),
    ("endpoints-classified", r"\b(?:every|all|each)\b[^.]{0,40}?\bendpoints?\b[^.]{0,40}?\b(?:is|are|were|have been)\s+(?:classified|assigned|resolved|explained|bound)\b", "release_ready"),
    ("completeness", r"\b(?:completeness|exhaustive(?:ness)?)\b[^.]{0,40}?\b(?:is|has been)\s+(?:established|proved|proven|certified|frozen)\b", "completeness"),
    ("no-more-pockets", r"\bno\s+(?:further|other|additional|hidden)\s+(?:stable\s+)?(?:pockets?|islands?|families|components?)\s+(?:exist|remain)\b", "completeness"),
    ("release-ready", r"\brelease[-_\s]?ready\b[^.]{0,20}?\b(?:is\s+)?true\b", "release_ready"),
)

# Operators that turn an occurrence into a disclaimer, a definition, a rule, or a
# quotation rather than an assertion by this manuscript.
NEGATORS = (
    r"\bnot\b", r"\bnever\b", r"\bno\b", r"\bnone\b", r"\bneither\b", r"\bnor\b",
    r"\bcannot\b", r"\bcan not\b", r"\bmay not\b", r"\boutside\b",
    r"\bwould be false\b", r"\bis false\b", r"\bis forbidden\b", r"\bforbidden\b",
    r"\buntil\b", r"\bunless\b", r"\bwithout\b", r"\bremains? open\b", r"\bstill\b",
    r"\bpending\b", r"\bunfinished\b", r"\brequires?\b", r"\bmust\b", r"\bwhether\b",
    r"\bdoes not\b", r"\bdo not\b", r"\bnothing\b", r"\bfails?\b", r"\bwithheld\b",
)
NEGATOR_RE = re.compile("|".join(NEGATORS))

# The generator-controlled conditionals, and how many brace groups each takes.
# ``\atlasifclaim`` spends its first group on the claim id, which is a lookup
# key and is never typeset; the two after it are the branches.
GATED_MACROS: dict[str, int] = {
    "atlasifsolved": 2,
    "atlasifgraphready": 2,
    "atlasifclaim": 3,
}

# Layer 4.  Definition operators that could rebind a generated macro.  A bare
# ``\newcommand`` cannot silently override anything -- LaTeX errors out if the
# name already exists -- but every other operator can, so the fallback block is
# allowed to use ``\newcommand`` only.
DEFINITION_OPS = (
    "newcommand", "renewcommand", "providecommand", "DeclareRobustCommand",
    "def", "gdef", "edef", "xdef", "let",
)
ATLAS_DEFINITION_RE = re.compile(
    r"\\(?P<op>" + "|".join(DEFINITION_OPS) + r")\b\*?\s*"
    r"(?:\\csname\s*|\{\s*\\csname\s*|\{\s*\\|\\)?(?P<name>atlas[A-Za-z@]*)"
)
FALLBACK_BEGIN = "% ATLAS-FALLBACK-BEGIN"
FALLBACK_END = "% ATLAS-FALLBACK-END"
# The fallback must be the weakest reading of every conditional: no macro file,
# no claim.  Whitespace is normalized before this comparison.
FALLBACK_WEAKEST_BRANCHES = (
    r"\newcommand{\atlasifsolved}[2]{#2}",
    r"\newcommand{\atlasifgraphready}[2]{#2}",
    r"\newcommand{\atlasifclaim}[3]{#3}",
)

# Manifest fields whose values are enum-constrained by validate_manifest and so
# carry no prose.  Everything else the renderers typeset is screened.  The
# allowlist is verified against the enum, so it cannot be used as a hiding spot.
MANIFEST_ENUM_FIELDS: dict[str, set[str]] = {
    "status": {"open", "solved", "falsified"},
    "gates[].id": {"A", "B", "C", "D"},
    "gates[].status": {"pending", "pass", "fail"},
}
CANARY = "ATLASCANARYQZX"


class ManuscriptGateError(Exception):
    """Raised when the manuscript may assert more than the evidence supports."""


class Fragment(NamedTuple):
    """A piece of typeset text plus the evidence state that governs printing it."""

    text: str
    state: dict[str, Any]
    where: str


# --------------------------------------------------------------------------- #
# LaTeX -> screenable prose
# --------------------------------------------------------------------------- #

def strip_comments(text: str) -> str:
    out = []
    for line in text.splitlines():
        stripped, escaped = [], False
        for char in line:
            if char == "\\" and not escaped:
                escaped = True
                stripped.append(char)
                continue
            if char == "%" and not escaped:
                break
            escaped = False
            stripped.append(char)
        out.append("".join(stripped))
    return "\n".join(out)


def _matching_brace(text: str, open_index: int) -> int:
    """Index just past the group that opens at ``open_index``."""
    depth, i, escaped = 0, open_index, False
    while i < len(text):
        char = text[i]
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return len(text)


def _conditioned(state: dict[str, Any], **facts: Any) -> dict[str, Any]:
    conditioned = dict(state)
    conditioned.update(facts)
    return conditioned


def branch_states(macro: str, state: dict[str, Any]) -> list[dict[str, Any] | None]:
    """Evidence state under which each brace group of ``macro`` is typeset.

    ``None`` marks a group that is not typeset at all (the claim id).  The rule
    is uniform: a branch is screened under the state that would print it, so no
    branch is ever unscreened, and the branch reserved for the closed world is
    screened against the closed world rather than against today's evidence.
    """
    if macro == "atlasifsolved":
        return [
            _conditioned(state, solved=True, release_ready=True, status="solved"),
            _conditioned(state, solved=False),
        ]
    if macro == "atlasifgraphready":
        return [
            _conditioned(
                state, release_ready=True, solved=str(state.get("status")) == "solved"
            ),
            _conditioned(state, release_ready=False, solved=False),
        ]
    if macro == "atlasifclaim":
        # A claim record being authorized says nothing about solvedness or about
        # the assembler bit, so both branches are screened under today's state.
        return [None, dict(state), dict(state)]
    raise ManuscriptGateError(f"unknown gated macro {macro}")


BRANCH_LABELS = {
    "atlasifsolved": ("printed when solved", "printed when not solved"),
    "atlasifgraphready": ("printed when release_ready", "printed when not release_ready"),
    "atlasifclaim": ("claim id", "printed while the claim stands", "printed once withdrawn"),
}


def split_gated(text: str, state: dict[str, Any], where: str = "") -> list[Fragment]:
    """Cut ``text`` into fragments, each tagged with the state that prints it.

    Nothing is dropped.  The previous version of this script deleted the
    conditionals' arguments outright, which meant the branch actually typeset
    today -- the ``not solved`` branch, and every standing claim's branch -- was
    never screened at all.  A gate you exempt from screening is not a gate.
    """
    pattern = re.compile(r"\\(" + "|".join(GATED_MACROS) + r")(?![a-zA-Z@])")
    fragments: list[Fragment] = []
    cursor = 0
    while True:
        match = pattern.search(text, cursor)
        if not match:
            fragments.append(Fragment(text[cursor:], state, where))
            return fragments
        fragments.append(Fragment(text[cursor : match.start()], state, where))
        macro = match.group(1)
        groups: list[str] = []
        scan = match.end()
        for _ in range(GATED_MACROS[macro]):
            while scan < len(text) and text[scan] in " \n\t":
                scan += 1
            if scan >= len(text) or text[scan] != "{":
                break
            end = _matching_brace(text, scan)
            groups.append(text[scan + 1 : end - 1])
            scan = end
        states = branch_states(macro, state)
        labels = BRANCH_LABELS[macro]
        for index, group in enumerate(groups):
            branch_state = states[index] if index < len(states) else None
            if branch_state is None:
                continue  # lookup key, never typeset
            label = f"\\{macro} branch {index + 1} ({labels[index]})"
            nested = f"{where} / {label}" if where else label
            fragments.extend(split_gated(group, branch_state, nested))
        cursor = scan


def ungated_text(text: str) -> str:
    """The part of ``text`` that is typeset unconditionally.

    Used to prove that mechanism words and released numbers live inside a claim
    gate, so withdrawing the claim deletes them.
    """
    state: dict[str, Any] = {}
    return " ".join(fragment.text for fragment in split_gated(text, state) if not fragment.where)


def to_prose(text: str) -> str:
    """Normalize LaTeX into flat lowercase prose suitable for screening."""
    text = re.sub(r"\$[^$]*\$", " NUM ", text)
    text = re.sub(r"\\\[.*?\\\]", " NUM ", text, flags=re.S)
    text = re.sub(r"\\begin\{(align|equation|tabular|itemize|enumerate|description)\*?\}", " ", text)
    text = re.sub(r"\\end\{(align|equation|tabular|itemize|enumerate|description)\*?\}", " ", text)
    text = re.sub(r"\\texttt\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\emph\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\textbf\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\[a-zA-Z@]+\*?", " ", text)
    text = re.sub(r"\\[^a-zA-Z@]", " ", text)  # control symbols such as an explicit "\ "
    text = text.replace("{", " ").replace("}", " ").replace("~", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def sentences(prose: str) -> list[str]:
    parts = re.split(r"(?<=[.!?:;])\s+", prose)
    return [part.strip() for part in parts if part.strip()]


def normalize_sentence(sentence: str) -> str:
    return re.sub(r"\s+", " ", sentence.strip().lower())


def sentence_hash(sentence: str) -> str:
    return hashlib.sha256(normalize_sentence(sentence).encode("utf-8")).hexdigest()[:32]


# --------------------------------------------------------------------------- #
# Layer 1 -- generated channel
# --------------------------------------------------------------------------- #

def check_generated_channel(manifest: dict[str, Any], root: Path) -> list[str]:
    errors: list[str] = []
    directory = root / GENERATED_DIR
    if not directory.is_dir():
        return [f"{GENERATED_DIR} is missing; the generated claim channel does not exist"]
    present = {path.name for path in directory.iterdir() if path.is_file()}
    for orphan in sorted(present - set(GENERATED_OWNERS)):
        errors.append(
            f"{GENERATED_DIR}/{orphan} has no generator: hand-written prose must not live in "
            f"{GENERATED_DIR}, move it to paper/handwritten/"
        )
    for name, render in GENERATED_OWNERS.items():
        path = directory / name
        expected = render(manifest, root)
        if not path.is_file():
            errors.append(f"{GENERATED_DIR}/{name} is missing; regenerate it from the manifest")
            continue
        actual = path.read_text(encoding="utf-8")
        if actual != expected:
            diff = "\n".join(
                list(
                    difflib.unified_diff(
                        actual.splitlines(), expected.splitlines(),
                        fromfile=f"committed {name}", tofile=f"generated {name}", lineterm="",
                    )
                )[:40]
            )
            errors.append(
                f"{GENERATED_DIR}/{name} does not match generator output; it was hand-edited or is "
                f"stale. Regenerate with scripts/build_discovery_release.py.\n{diff}"
            )
    return errors


def check_scope_tags(root: Path) -> list[str]:
    found: set[str] = set()
    for rel in HANDWRITTEN:
        path = root / rel
        if path.is_file():
            found |= set(SCOPE_TAG_RE.findall(path.read_text(encoding="utf-8")))
    missing = set(REQUIRED_SCOPE_TAGS) - found
    return [
        f"manuscript is missing required scope statement '{tag}' ({REQUIRED_SCOPE_TAGS[tag]}); "
        f"mark it with a '% ATLAS-SCOPE: {tag}' comment above the prose"
        for tag in sorted(missing)
    ]


# --------------------------------------------------------------------------- #
# Layer 4 -- generated-namespace integrity
# --------------------------------------------------------------------------- #

def _fallback_region(text: str) -> tuple[int, int] | None:
    start = text.find(FALLBACK_BEGIN)
    end = text.find(FALLBACK_END)
    if start < 0 or end < 0 or end < start:
        return None
    return start, end


def check_generated_namespace(root: Path) -> list[str]:
    """No hand-written source may rebind a macro the generator owns.

    ``\\renewcommand{\\atlasifsolved}[2]{#1}`` sitting in the preamble printed
    the solved branch while the assembler bit was false, and no prose check
    could see it: the overriding text is macro plumbing, not a sentence.
    """
    errors: list[str] = []
    for rel in HANDWRITTEN:
        path = root / rel
        if not path.is_file():
            continue
        raw = path.read_text(encoding="utf-8")
        region = _fallback_region(raw)
        for match in ATLAS_DEFINITION_RE.finditer(raw):
            inside = region is not None and region[0] <= match.start() < region[1]
            name, op = match.group("name"), match.group("op")
            if not inside:
                errors.append(
                    f"{rel.as_posix()}: \\{op} rebinds the generated macro \\{name} outside the "
                    f"documented fallback block. Generated macros are the evidence channel; "
                    f"redefining one lets hand-written source choose the branch."
                )
            elif op != "newcommand":
                errors.append(
                    f"{rel.as_posix()}: the fallback block uses \\{op} on \\{name}; only "
                    f"\\newcommand is allowed there, because \\newcommand cannot silently "
                    f"override the definition loaded from {GENERATED_DIR}/claim-macros.tex"
                )
        if rel.name != "main.tex":
            continue
        if region is None:
            errors.append(
                f"{rel.as_posix()}: the no-evidence fallback block must be delimited by "
                f"'{FALLBACK_BEGIN}' and '{FALLBACK_END}' so the gate can prove no other part "
                f"of the manuscript defines a generated macro"
            )
            continue
        body = re.sub(r"\s+", "", raw[region[0] : region[1]])
        for required in FALLBACK_WEAKEST_BRANCHES:
            if re.sub(r"\s+", "", required) not in body:
                errors.append(
                    f"{rel.as_posix()}: the fallback block must collapse the conditionals to "
                    f"their weakest branch; '{required}' is missing. A build with no evidence "
                    f"channel attached must assert strictly less, never more."
                )
        if raw.find(FALLBACK_END) > raw.find(r"\begin{document}") >= 0:
            errors.append(f"{rel.as_posix()}: the fallback block must sit in the preamble")
    return errors


# --------------------------------------------------------------------------- #
# Collection
# --------------------------------------------------------------------------- #

def _split_preamble(root: Path, rel: Path) -> tuple[str, str]:
    """Return (preamble, body) of a hand-written source, comments removed."""
    text = strip_comments((root / rel).read_text(encoding="utf-8"))
    preamble, marker, body = text.partition(r"\begin{document}")
    if not marker:
        return "", text
    return preamble, body


def _fragments(text: str, state: dict[str, Any]) -> list[Fragment]:
    items: list[Fragment] = []
    for fragment in split_gated(text, state):
        for sentence in sentences(to_prose(fragment.text)):
            items.append(Fragment(sentence, fragment.state, fragment.where))
    return items


def collect_handwritten(root: Path, state: dict[str, Any] | None = None) -> dict[str, list[Fragment]]:
    """Hand-written body sentences per file, each tagged with its governing state.

    Both branches of every conditional are included.  The preamble is handled by
    ``collect_preamble_prose`` (screened) and ``collect_control_lines`` (locked
    verbatim), because prose hashing cannot represent macro plumbing.
    """
    base: dict[str, Any] = dict(state or {})
    collected: dict[str, list[Fragment]] = {}
    for rel in HANDWRITTEN:
        if not (root / rel).is_file():
            continue
        collected[rel.as_posix()] = _fragments(_split_preamble(root, rel)[1], base)
    return collected


def collect_preamble_prose(root: Path, state: dict[str, Any] | None = None) -> dict[str, list[Fragment]]:
    """Preamble prose, which typesets (``\\title``) and feeds body macros.

    ``\\title{The three-body problem is solved}`` is printed on page one, and
    ``\\newcommand{\\intro}{...}`` carries its argument into the body, so the
    preamble is screened exactly like the body even though it is locked by line
    rather than by sentence.
    """
    base: dict[str, Any] = dict(state or {})
    collected: dict[str, list[Fragment]] = {}
    for rel in HANDWRITTEN:
        if not (root / rel).is_file():
            continue
        preamble = _split_preamble(root, rel)[0]
        if preamble.strip():
            collected[f"{rel.as_posix()} (preamble)"] = _fragments(preamble, base)
    return collected


def collect_control_lines(root: Path) -> dict[str, list[str]]:
    """Verbatim preamble lines, which carry no prose to hash but do typeset.

    Sentence hashing cannot see the difference between ``[2]{#1}`` and
    ``[2]{#2}``; line hashing can, so the preamble is locked literally.
    """
    collected: dict[str, list[str]] = {}
    for rel in HANDWRITTEN:
        if not (root / rel).is_file():
            continue
        preamble = _split_preamble(root, rel)[0]
        if not preamble.strip():
            continue
        lines = [re.sub(r"\s+", " ", line).strip() for line in preamble.splitlines()]
        collected[rel.as_posix()] = [line for line in lines if line]
    return collected


def _iter_string_leaves(node: Any, path: str = "") -> list[tuple[str, str]]:
    if isinstance(node, dict):
        found: list[tuple[str, str]] = []
        for key, value in node.items():
            found += _iter_string_leaves(value, f"{path}.{key}" if path else str(key))
        return found
    if isinstance(node, list):
        found = []
        for index, value in enumerate(node):
            found += _iter_string_leaves(value, f"{path}[{index}]")
        return found
    if isinstance(node, str):
        return [(path, node)]
    return []


def _set_at(root: Any, path: str, value: Any) -> None:
    keys: list[Any] = []
    for chunk in path.split("."):
        head, *indices = chunk.replace("]", "").split("[")
        keys.append(head)
        keys.extend(int(index) for index in indices)
    node = root
    for key in keys[:-1]:
        node = node[key]
    node[keys[-1]] = value


def _generic_path(path: str) -> str:
    return re.sub(r"\[\d+\]", "[]", path)


def collect_rendered_manifest_prose(
    manifest: dict[str, Any], root: Path
) -> tuple[dict[str, list[str]], list[str]]:
    """Every manifest string the manuscript renderers actually typeset.

    Screening only ``statement`` and ``method`` was a hand-maintained list, and
    it was already wrong: ``decision.summary``, each gate's title and criterion,
    each claim's limitations and each blocker all reach the PDF.  Rather than
    grow that list -- and forget the next field -- the fields are *discovered*:
    one string leaf at a time is replaced with a canary, every renderer in
    GENERATED_OWNERS is run, and any canary that reaches rendered output marks a
    field that must be screened.  Adding a rendered field therefore screens it
    automatically, and a field that renders but is neither screened nor a
    declared enum is a hard failure rather than a silent gap.
    """
    collected: dict[str, list[str]] = {}
    errors: list[str] = []
    for path, value in _iter_string_leaves(manifest):
        probe = deepcopy(manifest)
        _set_at(probe, path, CANARY)
        rendered = False
        for name, render in GENERATED_OWNERS.items():
            try:
                output = render(probe, root)
            except Exception as exc:  # noqa: BLE001 - a renderer must not decide policy
                errors.append(
                    f"renderer for {GENERATED_DIR}/{name} raised {type(exc).__name__} while "
                    f"probing manifest field {path}; the screen cannot certify what it cannot "
                    f"render: {exc}"
                )
                continue
            if CANARY in output:
                rendered = True
        if not rendered:
            continue
        generic = _generic_path(path)
        if generic in MANIFEST_ENUM_FIELDS:
            allowed = MANIFEST_ENUM_FIELDS[generic]
            if value not in allowed:
                errors.append(
                    f"{MANIFEST}: {path} is declared enum-valued and exempt from prose "
                    f"screening, but its value {value!r} is outside {sorted(allowed)}"
                )
            continue
        readable = value.replace("-", " ").replace("_", " ") if " " not in value else value
        collected[f"{MANIFEST} :: {path}"] = sentences(readable)
    return collected, errors


# --------------------------------------------------------------------------- #
# Layer 3 -- vocabulary screen
# --------------------------------------------------------------------------- #

def requirement_met(requirement: str, state: dict[str, Any]) -> bool:
    if requirement == "never":
        return False
    if requirement == "solved":
        return bool(state.get("solved"))
    if requirement == "release_ready":
        return bool(state.get("release_ready"))
    if requirement == "completeness":
        return state.get("completeness") is not None
    return False


def _fragment(item: Fragment | str, state: dict[str, Any]) -> Fragment:
    if isinstance(item, Fragment):
        return item
    return Fragment(item, state, "")


def screen_vocabulary(
    collected: dict[str, list[Fragment | str]] | dict[str, list[str]],
    state: dict[str, Any],
) -> list[str]:
    """Flag assertions no evidence licenses, each under its own printing state."""
    errors: list[str] = []
    for filename, items in collected.items():
        for item in items:
            fragment = _fragment(item, state)
            lowered = normalize_sentence(fragment.text)
            if NEGATOR_RE.search(lowered):
                continue  # disclaimer, rule, condition or question, not an assertion
            for name, pattern, requirement in CLAIM_PATTERNS:
                if re.search(pattern, lowered) and not requirement_met(requirement, fragment.state):
                    where = f" [{fragment.where}]" if fragment.where else ""
                    errors.append(
                        f"{filename}{where}: ungated {name} assertion while {requirement} is not "
                        f"satisfied (status={fragment.state.get('status')}, "
                        f"release_ready={fragment.state.get('release_ready')}, "
                        f"solved={fragment.state.get('solved')}):\n    {fragment.text[:220]}"
                    )
                    break
    return errors


# --------------------------------------------------------------------------- #
# Layer 2 -- lock
# --------------------------------------------------------------------------- #

def evidence_fingerprint(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": state["status"],
        "release_ready": state["release_ready"],
        "gates": state["gates"],
        "release_claims": state["release_claims"],
        "completeness_frozen": state.get("completeness") is not None,
    }


def fingerprint_weakened(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    """Facts that got worse since the last freeze."""
    weaker: list[str] = []
    if old.get("release_ready") and not new.get("release_ready"):
        weaker.append("release_ready went true -> false")
    if old.get("completeness_frozen") and not new.get("completeness_frozen"):
        weaker.append("completeness certificate was withdrawn")
    if old.get("status") == "solved" and new.get("status") != "solved":
        weaker.append("manifest status left 'solved'")
    for gate, status in (old.get("gates") or {}).items():
        if status == "pass" and (new.get("gates") or {}).get(gate) != "pass":
            weaker.append(f"gate {gate} left 'pass'")
    for claim in set(old.get("release_claims") or ()) - set(new.get("release_claims") or ()):
        weaker.append(f"release claim '{claim}' was withdrawn")
    return weaker


def check_lock(
    collected: dict[str, list[Fragment | str]],
    state: dict[str, Any],
    root: Path,
    control_lines: dict[str, list[str]] | None = None,
) -> list[str]:
    lock_file = root / LOCK_PATH
    if not lock_file.is_file():
        return [f"{LOCK_PATH} is missing; run scripts/check_manuscript_claims.py --freeze"]
    lock = json.loads(lock_file.read_text(encoding="utf-8"))
    errors: list[str] = []
    weaker = fingerprint_weakened(lock.get("evidence", {}), evidence_fingerprint(state))
    if weaker:
        errors.append(
            "evidence weakened since the manuscript prose was last reviewed ("
            + "; ".join(weaker)
            + "); re-review the hand-written prose and re-freeze the lock"
        )
    reviewed = lock.get("sentences", {})
    for filename, items in collected.items():
        known = set(reviewed.get(filename, []))
        for item in items:
            fragment = _fragment(item, state)
            if sentence_hash(fragment.text) not in known:
                where = f" [{fragment.where}]" if fragment.where else ""
                errors.append(
                    f"{filename}{where}: unreviewed hand-written sentence. Hand-written manuscript "
                    f"prose is locked; re-freeze only after confirming it does not outrun the "
                    f"evidence:\n    {fragment.text[:220]}"
                )
    reviewed_lines = lock.get("control_lines", {})
    for filename, lines in (control_lines or {}).items():
        known = set(reviewed_lines.get(filename, []))
        for line in lines:
            if sentence_hash(line) not in known:
                errors.append(
                    f"{filename}: unreviewed manuscript control line. The preamble typesets and "
                    f"can rebind the generated evidence macros, so it is locked verbatim:"
                    f"\n    {line[:220]}"
                )
    return errors


def build_lock(
    collected: dict[str, list[Fragment | str]],
    state: dict[str, Any],
    control_lines: dict[str, list[str]],
) -> dict[str, Any]:
    """Hash -> text, so blessing a paraphrase shows up as prose in the diff.

    A lock that stores hashes alone hides the one thing a reviewer has to read:
    the sentence a previous freeze blessed.  The hash stays authoritative -- it
    is what the check compares -- and the text makes the review possible.
    """
    return {
        "note": (
            "Every hand-written manuscript sentence -- both branches of every "
            "generator-controlled conditional included -- and every preamble control line, "
            "reviewed by a human against the recorded evidence state. Keyed by normalized "
            "hash, with the reviewed text kept verbatim so a re-freeze shows the prose it "
            "blesses. Regenerate with scripts/check_manuscript_claims.py --freeze "
            "--review-new-prose; the freeze is refused under CI and whenever the evidence "
            "has weakened."
        ),
        "evidence": evidence_fingerprint(state),
        "sentences": {
            filename: {
                sentence_hash(_fragment(item, state).text): _fragment(item, state).text
                for item in items
            }
            for filename, items in sorted(collected.items())
        },
        "control_lines": {
            filename: {sentence_hash(line): line for line in lines}
            for filename, lines in sorted(control_lines.items())
        },
    }


# --------------------------------------------------------------------------- #

def gather_errors(manifest: dict[str, Any], state: dict[str, Any], root: Path) -> list[str]:
    collected = collect_handwritten(root, state)
    control_lines = collect_control_lines(root)
    manifest_prose, discovery_errors = collect_rendered_manifest_prose(manifest, root)
    errors: list[str] = []
    errors += check_generated_channel(manifest, root)
    errors += check_scope_tags(root)
    errors += check_generated_namespace(root)
    errors += discovery_errors
    errors += screen_vocabulary(collected, state)
    errors += screen_vocabulary(collect_preamble_prose(root, state), state)
    errors += screen_vocabulary(manifest_prose, state)
    errors += check_lock(collected, state, root, control_lines)
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument(
        "--freeze",
        action="store_true",
        help="re-review: print new prose and rewrite paper/CLAIM_LOCK.json",
    )
    parser.add_argument(
        "--review-new-prose",
        action="store_true",
        help=(
            "acknowledge, as a human, that every sentence listed as NEW has been read and "
            "does not outrun the evidence; required before a freeze may bless new prose"
        ),
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()

    manifest = load_manifest(root / MANIFEST)
    state = evidence_state(manifest, root)
    collected = collect_handwritten(root, state)
    control_lines = collect_control_lines(root)

    if args.freeze:
        # A freeze is a human act of review.  An automated pipeline that could
        # run it would be able to bless any paraphrase the lexicon cannot see,
        # which is the one thing Layer 2 exists to prevent.
        ci = next((name for name in ("GITHUB_ACTIONS", "CI") if os.getenv(name)), None)
        if ci:
            print(
                f"Refusing to freeze: {ci} is set. The claim lock records a human review of "
                "manuscript prose; a workflow re-blessing it would make the lock a rubber "
                "stamp. Run this on a workstation and commit the result.",
                file=sys.stderr,
            )
            return 2
        lock_file = root / LOCK_PATH
        pending: list[str] = []
        if lock_file.is_file():
            old = json.loads(lock_file.read_text(encoding="utf-8"))
            weaker = fingerprint_weakened(old.get("evidence", {}), evidence_fingerprint(state))
            if weaker:
                print(
                    "Refusing to freeze: the evidence has weakened since the last review "
                    "(" + "; ".join(weaker) + "). Fix the prose, do not re-bless it.",
                    file=sys.stderr,
                )
                return 2
            for filename, items in collected.items():
                known = set(old.get("sentences", {}).get(filename, []))
                for item in items:
                    fragment = _fragment(item, state)
                    if sentence_hash(fragment.text) not in known:
                        pending.append(f"NEW  {filename}: {fragment.text[:200]}")
            for filename, lines in control_lines.items():
                known = set(old.get("control_lines", {}).get(filename, []))
                for line in lines:
                    if sentence_hash(line) not in known:
                        pending.append(f"NEWLINE  {filename}: {line[:200]}")
        for item in pending:
            print(item)
        if pending and not args.review_new_prose:
            print(
                f"\nRefusing to freeze: {len(pending)} unreviewed item(s) above would be "
                "blessed silently. Read them against the evidence "
                f"(status={state['status']}, release_ready={state['release_ready']}), then "
                "re-run with --review-new-prose. The blessed text is written into "
                f"{LOCK_PATH} verbatim so the diff shows what was approved.",
                file=sys.stderr,
            )
            return 2
        manifest_prose, discovery_errors = collect_rendered_manifest_prose(manifest, root)
        blocking = (
            screen_vocabulary(collected, state)
            + screen_vocabulary(collect_preamble_prose(root, state), state)
            + screen_vocabulary(manifest_prose, state)
            + discovery_errors
            + check_scope_tags(root)
            + check_generated_namespace(root)
        )
        if blocking:
            print("Refusing to freeze:", file=sys.stderr)
            for error in blocking:
                print(f"- {error}", file=sys.stderr)
            return 2
        lock_file.write_text(
            json.dumps(build_lock(collected, state, control_lines), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"Froze {LOCK_PATH} at status={state['status']} release_ready={state['release_ready']}")
        return 0

    errors = gather_errors(manifest, state, root)
    if errors:
        print("Manuscript claim gate FAILED:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 2
    print(
        f"Manuscript claim gate OK: status={state['status']} "
        f"release_ready={state['release_ready']} "
        f"release_claims={len(state['release_claims'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
