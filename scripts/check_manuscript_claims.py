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
paraphrase an author invents.  So the check is three independent layers, and the
two load-bearing ones do not depend on wording at all.

  Layer 1 -- generated channel integrity (wording independent).
      Every file under ``paper/generated/`` must byte-match what
      ``scripts/build_discovery_release.py`` emits right now, and no other file
      may live there.  A claim can therefore only enter the manuscript through
      the generator, from a ``release_claim`` record.  Hand-forging
      ``paper/generated/discoveries.tex`` -- which is exactly what the committed
      tree contained -- is now a hard failure.

  Layer 2 -- hand-written prose lock (wording independent).
      Every sentence of hand-written manuscript prose is normalized and hashed
      into ``paper/CLAIM_LOCK.json``, together with the evidence fingerprint it
      was reviewed under.  A new or altered sentence fails until a human reruns
      this script with ``--freeze``, which prints the diff first.  Unrelated
      sentences never churn, so the lock stays meaningful instead of becoming a
      rubber stamp.  Freezing is refused outright when the evidence has weakened
      since the last freeze, so prose can never be re-blessed on the way down.

  Layer 3 -- claim-vocabulary screen (best effort, evidence conditional).
      Fast, precise feedback at authoring time.  It scans only hand-written
      prose, applies negation and scoping analysis so honest disclaimers pass,
      and is evaluated against the current evidence state -- a sentence that is
      legitimate while a claim holds becomes an error the moment that claim is
      withdrawn or ``release_ready`` goes false.

Layer 3 catches the obvious mistake immediately; Layer 2 catches everything else
by construction, including paraphrases Layer 3 has never heard of.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

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
# wearing a machine-authored costume.
GENERATED_OWNERS = {
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
# assertion to be legitimate in hand-written prose.
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
    ("germs-complete", r"\ball\s+twelve\b", "release_ready"),
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

# Regions of hand-written source that are exempt from Layer 3 because the
# generator, not the author, decides whether they are typeset at all.
GATED_MACROS = ("atlasifsolved", "atlasifgraphready", "atlasifclaim")


class ManuscriptGateError(Exception):
    """Raised when the manuscript may assert more than the evidence supports."""


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


def drop_gated_regions(text: str) -> str:
    """Remove arguments of the generator-controlled conditionals.

    Whatever sits inside ``\\atlasifsolved``/``\\atlasifgraphready``/
    ``\\atlasifclaim`` is printed only if the generated macro file says so, so it
    is gated by construction and must not be screened as a free assertion.
    """
    pattern = re.compile(r"\\(" + "|".join(GATED_MACROS) + r")\s*")
    while True:
        match = pattern.search(text)
        if not match:
            return text
        cursor = match.end()
        # \atlasifclaim takes three groups, the others two.
        groups = 3 if match.group(1) == "atlasifclaim" else 2
        for _ in range(groups):
            while cursor < len(text) and text[cursor] in " \n\t":
                cursor += 1
            if cursor >= len(text) or text[cursor] != "{":
                break
            cursor = _matching_brace(text, cursor)
        text = text[: match.start()] + " " + text[cursor:]


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
# Layers
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


def collect_handwritten(root: Path) -> dict[str, list[str]]:
    """Hand-written sentences per file, with generator-gated regions removed."""
    collected: dict[str, list[str]] = {}
    for rel in HANDWRITTEN:
        path = root / rel
        if not path.is_file():
            continue
        text = strip_comments(path.read_text(encoding="utf-8"))
        # The preamble typesets nothing a reader can mistake for a claim, and
        # locking it would fill the lock with package options.
        _, _, body = text.partition(r"\begin{document}")
        collected[rel.as_posix()] = sentences(to_prose(drop_gated_regions(body or text)))
    return collected


def collect_release_claims(manifest: dict[str, Any]) -> dict[str, list[str]]:
    """Sentences the generator would print from ``release_claim`` records.

    The generated channel is authorized, but authorization is per record, not per
    sentence: flipping a claim to ``release_claim`` is a one-line manifest edit
    and ``validate_manifest`` never reads the wording.  Screening the rendered
    statements closes that path, so a claim cannot assert graph completeness
    while the assembler still says the graph is not release-ready.
    """
    collected: list[str] = []
    for claim in manifest.get("claims", []):
        if claim.get("status") != "release_claim":
            continue
        for field in ("statement", "method"):
            value = claim.get(field)
            if isinstance(value, str):
                collected.extend(sentences(value))
    return {"research/DISCOVERY_RELEASE.json (release_claim text)": collected}


def requirement_met(requirement: str, state: dict[str, Any]) -> bool:
    if requirement == "never":
        return False
    if requirement == "solved":
        return bool(state["solved"])
    if requirement == "release_ready":
        return bool(state["release_ready"])
    if requirement == "completeness":
        return state.get("completeness") is not None
    return False


def screen_vocabulary(collected: dict[str, list[str]], state: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for filename, items in collected.items():
        for sentence in items:
            lowered = normalize_sentence(sentence)
            if NEGATOR_RE.search(lowered):
                continue  # disclaimer, rule, condition or question, not an assertion
            for name, pattern, requirement in CLAIM_PATTERNS:
                if re.search(pattern, lowered) and not requirement_met(requirement, state):
                    errors.append(
                        f"{filename}: ungated {name} assertion while {requirement} is not "
                        f"satisfied (status={state['status']}, "
                        f"release_ready={state['release_ready']}):\n    {sentence[:220]}"
                    )
                    break
    return errors


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


def check_lock(collected: dict[str, list[str]], state: dict[str, Any], root: Path) -> list[str]:
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
        for sentence in items:
            if sentence_hash(sentence) not in known:
                errors.append(
                    f"{filename}: unreviewed hand-written sentence. Hand-written manuscript prose "
                    f"is locked; re-freeze only after confirming it does not outrun the evidence:"
                    f"\n    {sentence[:220]}"
                )
    return errors


def build_lock(collected: dict[str, list[str]], state: dict[str, Any]) -> dict[str, Any]:
    return {
        "note": (
            "Hashes of every hand-written manuscript sentence that a human has reviewed against "
            "the recorded evidence state. Regenerate with scripts/check_manuscript_claims.py "
            "--freeze; the freeze is refused when the evidence has weakened."
        ),
        "evidence": evidence_fingerprint(state),
        "sentences": {
            filename: sorted({sentence_hash(item) for item in items})
            for filename, items in sorted(collected.items())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument(
        "--freeze",
        action="store_true",
        help="re-review: print new prose and rewrite paper/CLAIM_LOCK.json",
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()

    manifest = load_manifest(root / MANIFEST)
    state = evidence_state(manifest, root)
    collected = collect_handwritten(root)

    if args.freeze:
        lock_file = root / LOCK_PATH
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
                for sentence in items:
                    if sentence_hash(sentence) not in known:
                        print(f"NEW  {filename}: {sentence[:200]}")
        blocking = screen_vocabulary(collected, state) + check_scope_tags(root)
        if blocking:
            print("Refusing to freeze:", file=sys.stderr)
            for error in blocking:
                print(f"- {error}", file=sys.stderr)
            return 2
        lock_file.write_text(
            json.dumps(build_lock(collected, state), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"Froze {LOCK_PATH} at status={state['status']} release_ready={state['release_ready']}")
        return 0

    errors: list[str] = []
    errors += check_generated_channel(manifest, root)
    errors += check_scope_tags(root)
    errors += screen_vocabulary(collected, state)
    errors += screen_vocabulary(collect_release_claims(manifest), state)
    errors += check_lock(collected, state, root)
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
