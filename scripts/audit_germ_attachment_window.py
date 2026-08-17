#!/usr/bin/env python3
"""Measure how tightly GERM_ATTACH_DISTANCE is pinned by the assembly it produces.

The assembler binds a mechanism-polyline edge end to a mixed organizer when the
mass distance between them is at most ``GERM_ATTACH_DISTANCE`` (0.008).  That
number is a modelling choice, not a measured quantity, so the honest question is
how much of it the current graph depends on: below some threshold an attachment
that the committed graph relies on is lost, and above some other threshold an
extra, unintended attachment appears.

This script re-runs the real assembler with only that constant varied and
bisects both edges of the window.  It is an audit tool: it writes no evidence
file, touches no numerical gate, and leaves the assembler's default unchanged.
The window it reports is quoted in research/DISCOVERY_RELEASE.json and in the
paper, so the prose cannot drift away from the code.

Two things this file is careful about, because an earlier version of it was
wrong about both:

* **It uses the release configuration.**  The inputs are parsed out of
  ``scripts/assemble_v1_critical_graph.sh``, the single source of truth for
  which artifacts feed the committed graph.  The earlier hard-coded pair
  (roots + ``V1_MIXED_GERMS_2026-08-15.json``) is the superseded germ file that
  the assembler now rejects wholesale, so it produced *zero* attachments and a
  window derived from a test fixture rather than from the release graph.
* **Two different questions get two different answers.**  The nearest
  mode-matching endpoint/germ pair that the threshold *excludes* is closer than
  the threshold at which the assembled graph actually *changes*, because some
  excluded pairs sit on endpoints that a classification artifact already binds.
  Both are reported; the tighter pair-based window is the one the caveats
  quote.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import re
import shlex
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ASSEMBLER = ROOT / "scripts/assemble_critical_graph.py"
RELEASE_INVOCATION = ROOT / "scripts/assemble_v1_critical_graph.sh"


def release_inputs(script: Path = RELEASE_INVOCATION) -> list[str]:
    """Parse the canonical assembler invocation for its evidence flags.

    Reading the shell script keeps this audit pinned to whatever the committed
    graph is actually assembled from; hard-coding the list is how it drifted
    onto a superseded germ artifact in the first place.
    """
    text = script.read_text(encoding="utf-8")
    # The canonical invocation builds its flags in an EVIDENCE_ARGS array so the
    # closure runner can enumerate them via PRINT_INPUTS without a second copy of
    # the list.  Parse that array, not the "${PYTHON_CMD[@]}" line, and resolve
    # the two documented ${VAR:-default} overrides to their committed defaults.
    match = re.search(r"^EVIDENCE_ARGS=\((.*?)^\)", text, re.M | re.S)
    if match is None:
        raise RuntimeError(f"{script} has no EVIDENCE_ARGS array")
    defaults = dict(re.findall(r'^(\w+)="\$\{\1:-([^}]*)\}"', text, re.M))
    body = match.group(1)
    for name, value in defaults.items():
        body = body.replace(f'"${name}"', value).replace(f"${name}", value)
    tokens = shlex.split(body)
    inputs: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.startswith("--") and token != "--output" and index + 1 < len(tokens):
            inputs += [token, tokens[index + 1]]
            index += 2
            continue
        index += 1
    if "--roots" not in inputs or "--germs" not in inputs:
        raise RuntimeError(f"{script} did not yield --roots and --germs")
    return inputs


def _load_assembler() -> Any:
    spec = importlib.util.spec_from_file_location("_atlas_assembler_audit", ASSEMBLER)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"cannot import {ASSEMBLER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assemble_at(threshold: float, inputs: list[str]) -> dict[str, Any]:
    """Run the assembler with GERM_ATTACH_DISTANCE replaced by ``threshold``."""
    module = _load_assembler()
    module.GERM_ATTACH_DISTANCE = threshold
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "graph.json"
        argv = sys.argv
        sys.argv = ["assemble_critical_graph.py", "--output", str(out), *inputs]
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                module.main()
        except SystemExit:
            pass
        finally:
            sys.argv = argv
        return json.loads(out.read_text(encoding="utf-8"))


def germ_attachments(graph: dict[str, Any]) -> list[float]:
    return sorted(
        endpoint["distance_to_germ"]
        for edge in graph["edges"]
        for endpoint in (edge["endpoints"]["start"], edge["endpoints"]["end"])
        if endpoint.get("attachment") == "continuation_germ"
    )


def candidate_pairs(graph: dict[str, Any], module: Any) -> list[dict[str, Any]]:
    """Every mode-matching (edge endpoint, valid germ) pair, by mass distance.

    This is the set the threshold arbitrates over.  Direction is deliberately
    ignored: what the constant decides is which of these pairs are close enough
    to be considered at all.
    """
    germs = [row for row in graph["mixed_germs"] if row.get("valid")]
    pairs = [
        {
            "distance": module.mass_distance(
                graph_edge["endpoints"][side]["masses"], germ["masses"]
            ),
            "edge": graph_edge["id"],
            "side": side,
            "organizer": germ["mixed_node"],
            "germ_direction": germ["direction"],
        }
        for graph_edge in graph["edges"]
        for side in ("start", "end")
        for germ in germs
        if germ["event_mode"] == graph_edge["mechanism"]
    ]
    pairs.sort(key=lambda row: row["distance"])
    return pairs


def _bisect(inputs: list[str], keep: float, change: float, target: int) -> float:
    """Bisect between a threshold that yields ``target`` attachments and one that does not."""
    for _ in range(60):
        mid = (keep + change) / 2.0
        if mid in (keep, change):
            break
        if len(germ_attachments(assemble_at(mid, inputs))) == target:
            keep = mid
        else:
            change = mid
    return change


def window(inputs: list[str] | None = None) -> dict[str, Any]:
    inputs = list(inputs if inputs is not None else release_inputs())
    module = _load_assembler()
    default = float(module.GERM_ATTACH_DISTANCE)
    graph = assemble_at(default, inputs)
    baseline = germ_attachments(graph)
    target = len(baseline)
    if not baseline:
        raise RuntimeError(
            "the supplied inputs produce no germ attachments at all, so this audit would "
            "measure nothing; check that they are the release configuration"
        )
    pairs = candidate_pairs(graph, module)
    accepted = [row for row in pairs if row["distance"] <= default]
    rejected = [row for row in pairs if row["distance"] > default]
    if not rejected:
        raise RuntimeError("no rejected candidate: the upper edge of the window is unbounded")
    # Lower edge: below the largest distance the assembly actually uses, an
    # attachment is lost.
    lower = max(baseline)
    # Upper edge, question 1: the nearest mode-matching pair the threshold turns
    # away.  This is the tighter, honest margin, and it is what the caveats quote.
    nearest_rejected = rejected[0]
    # Upper edge, question 2: the first threshold at which the assembled graph
    # actually changes.  It is larger than question 1 whenever the nearest
    # rejected pair sits on an endpoint a classification artifact already binds.
    assembly_change = _bisect(inputs, keep=default, change=default * 4.0, target=target)
    return {
        "inputs": inputs,
        "default_germ_attach_distance": default,
        "attachments_at_default": target,
        "attachment_distances": baseline,
        "largest_accepted": accepted[-1],
        "nearest_rejected": nearest_rejected,
        "admissible_window": [lower, nearest_rejected["distance"]],
        "window_ratio": nearest_rejected["distance"] / lower,
        "assembly_change_threshold": assembly_change,
        "assembly_change_ratio": assembly_change / lower,
        "default_inside_window": lower <= default < nearest_rejected["distance"],
        # GERM_ATTACH_DISTANCE does double duty: a germ may sit this far from its
        # organizer AND an endpoint may sit this far from that germ, so the
        # effective organizer-to-endpoint reach is twice the constant.
        "effective_organizer_reach": 2.0 * default,
        "interpretation": (
            f"The {target}-attachment assembly produced by the release configuration keeps every "
            f"attachment it has for GERM_ATTACH_DISTANCE >= {lower!r}, and admits no further "
            f"mode-matching candidate below {nearest_rejected['distance']!r} "
            f"({nearest_rejected['edge']}:{nearest_rejected['side']} <-> "
            f"{nearest_rejected['organizer']}), a window "
            f"{nearest_rejected['distance'] / lower:.4f}x wide. The default {default} sits inside "
            f"it. "
            + (
                f"The assembled graph does not actually change until {assembly_change!r} "
                f"({assembly_change / lower:.4f}x), because the nearest rejected candidate's "
                f"endpoint is already bound by its own classification artifact. "
                if assembly_change > nearest_rejected["distance"]
                else (
                    f"Raising the threshold to {nearest_rejected['distance']!r} would attach "
                    f"{nearest_rejected['edge']}:{nearest_rejected['side']}; that end is not "
                    f"already bound by a classification artifact. "
                )
            )
            + (
                f"Because the constant also caps how far a germ may sit from its organizer, "
                f"the effective organizer-to-endpoint reach is up to {2.0 * default}."
            )
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit GERM_ATTACH_DISTANCE against the release assembler invocation."
    )
    parser.add_argument(
        "--invocation",
        default=str(RELEASE_INVOCATION),
        help="shell script whose assembler flags define the release configuration",
    )
    args = parser.parse_args()
    print(json.dumps(window(release_inputs(Path(args.invocation))), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
