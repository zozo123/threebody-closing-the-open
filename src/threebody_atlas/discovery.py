"""Discovery-release gate and evidence dossier helpers."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "atlas.discovery-release.v1"
REQUIRED_GATES = {"A", "B", "C", "D"}
REQUIRED_SOLVED_ROLES = {
    "open_problem",
    "protocol",
    "result_ledger",
    "novelty_audit",
    "critical_graph",
    "independent_verification",
    "family_connectivity",
    "adversarial_search",
    "paper",
}


class DiscoveryValidationError(ValueError):
    """Raised when a discovery manifest violates the release contract."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise DiscoveryValidationError("manifest root must be an object")
    return data


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_manifest(
    manifest: dict[str, Any],
    root: str | Path,
    *,
    require_solved: bool = False,
    today: date | None = None,
) -> None:
    """Validate the scientific state and, optionally, the final solved-release gate."""
    errors: list[str] = []
    root = Path(root)

    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    if not _nonempty(manifest.get("release_id")):
        errors.append("release_id must be non-empty")
    if manifest.get("status") not in {"open", "solved", "falsified"}:
        errors.append("status must be open, solved, or falsified")

    problem = manifest.get("problem", {})
    for key in ("id", "title", "question", "scope", "success_criterion", "baseline"):
        if not _nonempty(problem.get(key)):
            errors.append(f"problem.{key} must be non-empty")

    decision = manifest.get("decision", {})
    if not _nonempty(decision.get("summary")):
        errors.append("decision.summary must be non-empty")

    evidence = manifest.get("evidence", [])
    evidence_ids: set[str] = set()
    roles: set[str] = set()
    for item in evidence:
        if not isinstance(item, dict) or not _nonempty(item.get("id")):
            errors.append("every evidence record needs an id")
            continue
        evidence_id = item["id"]
        if evidence_id in evidence_ids:
            errors.append(f"duplicate evidence id: {evidence_id}")
        evidence_ids.add(evidence_id)
        roles.add(str(item.get("role", "")))
        if not _nonempty(item.get("role")) or not _nonempty(item.get("description")):
            errors.append(f"evidence {evidence_id} needs role and description")
        if item.get("kind") == "repository_file":
            path = root / str(item.get("path", ""))
            if not path.is_file():
                errors.append(f"missing repository evidence: {path}")
        if item.get("kind") == "actions_artifact":
            digest = str(item.get("sha256", "")).lower()
            valid_digest = len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)
            if not valid_digest:
                errors.append(f"artifact {evidence_id} needs a hexadecimal sha256 digest")

    gates = manifest.get("gates", [])
    gate_ids = {str(gate.get("id")) for gate in gates if isinstance(gate, dict)}
    if gate_ids != REQUIRED_GATES:
        errors.append("closure gates must be exactly A, B, C, and D")
    for gate in gates:
        if gate.get("status") not in {"pending", "pass", "fail"}:
            errors.append(f"gate {gate.get('id')} has invalid status")
        if not _nonempty(gate.get("title")) or not _nonempty(gate.get("criterion")):
            errors.append(f"gate {gate.get('id')} needs a title and criterion")
        for ref in gate.get("evidence", []):
            if ref not in evidence_ids:
                errors.append(f"gate {gate.get('id')} references unknown evidence {ref}")

    release_claims = []
    for claim in manifest.get("claims", []):
        claim_id = claim.get("id")
        if not _nonempty(claim_id):
            errors.append("every claim needs an id")
        if claim.get("status") not in {"candidate", "withheld", "release_claim", "rejected"}:
            errors.append(f"claim {claim_id} has invalid status")
        if not _nonempty(claim.get("statement")) or not _nonempty(claim.get("method")):
            errors.append(f"claim {claim_id} needs a statement and method")
        if claim.get("status") == "release_claim":
            release_claims.append(claim)
            if not claim.get("evidence"):
                errors.append(f"release claim {claim_id} has no evidence")
        for ref in claim.get("evidence", []):
            if ref not in evidence_ids:
                errors.append(f"claim {claim_id} references unknown evidence {ref}")

    novelty = manifest.get("novelty", {})
    try:
        novelty_date = date.fromisoformat(str(novelty.get("last_search_date")))
    except ValueError:
        novelty_date = None
        errors.append("novelty.last_search_date must be an ISO date")

    include_paths = manifest.get("release", {}).get("include_paths", [])
    if not include_paths:
        errors.append("release.include_paths must not be empty")
    for path in include_paths:
        if not (root / path).exists():
            errors.append(f"release include path does not exist: {path}")

    solved = manifest.get("status") == "solved" or require_solved
    if solved:
        if manifest.get("status") != "solved":
            errors.append("scientific release requires status='solved'")
        if not _nonempty(decision.get("solved_at")):
            errors.append("scientific release requires decision.solved_at")
        if manifest.get("blockers"):
            errors.append("scientific release requires blockers to be empty")
        if not manifest.get("known_limitations"):
            errors.append("scientific release requires known_limitations")
        if any(gate.get("status") != "pass" for gate in gates):
            errors.append("all closure gates must pass")
        if not release_claims:
            errors.append("scientific release requires at least one release_claim")
        if novelty.get("status") != "pass":
            errors.append("novelty freeze must pass")
        if novelty_date is not None:
            age = ((today or datetime.now(timezone.utc).date()) - novelty_date).days
            if age < 0 or age > int(novelty.get("max_age_days", 0)):
                errors.append(f"novelty search is {age} days old")
        missing_roles = REQUIRED_SOLVED_ROLES - roles
        if missing_roles:
            errors.append("missing solved evidence roles: " + ", ".join(sorted(missing_roles)))
        if require_solved:
            for item in evidence:
                if item.get("kind") == "generated_artifact":
                    generated = root / item["path"]
                    if not generated.is_file():
                        errors.append(f"missing generated artifact: {item['path']}")
        graph_records = [
            item
            for item in evidence
            if item.get("role") == "critical_graph"
            and item.get("kind") == "repository_file"
            and str(item.get("path", "")).endswith(".json")
        ]
        if not graph_records:
            errors.append("scientific release requires a JSON critical_graph repository file")
        else:
            ready = False
            for item in graph_records:
                graph_path = root / str(item.get("path", ""))
                if not graph_path.is_file():
                    errors.append(f"missing critical graph: {graph_path}")
                    continue
                try:
                    graph = json.loads(graph_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    errors.append(f"critical graph is not JSON: {graph_path}")
                    continue
                if graph.get("release_ready") is True:
                    ready = True
                    digest = str(item.get("sha256") or "").lower()
                    valid_digest = len(digest) == 64 and all(
                        character in "0123456789abcdef" for character in digest
                    )
                    if not valid_digest:
                        errors.append(
                            f"release-ready critical graph {item.get('path')} needs a hexadecimal sha256 digest"
                        )
                    else:
                        actual = sha256_file(graph_path)
                        if digest != actual:
                            errors.append(
                                f"critical graph sha256 mismatch for {item.get('path')}: "
                                f"manifest={digest} file={actual}"
                            )
                    coverage = graph.get("root_coverage") or {}
                    if int(graph.get("source_transition_cells") or 0) != 620:
                        errors.append(f"{item.get('path')} must declare 620 source transition cells")
                    if int(graph.get("localized_roots") or coverage.get("localized_roots") or 0) != 620:
                        errors.append(f"{item.get('path')} must have 620 localized roots")
                    if coverage.get("complete") is False:
                        errors.append(f"{item.get('path')} source-cell assignment is incomplete")
                    if graph.get("unexplained_nodes"):
                        errors.append(
                            f"{item.get('path')} still has unexplained endpoints: "
                            + ", ".join(str(x) for x in graph["unexplained_nodes"])
                        )
                    # Only the assembler-verified bit counts.  A certificate's
                    # own "passed" field is self-reported: the record is sealed
                    # with a digest over itself, so accepting it would let a
                    # hand-written certificate satisfy this gate.  The verified
                    # form in root_coverage is set only after the assembler
                    # re-hashed every source artifact and re-derived the AL and
                    # neck predicates from them.
                    if coverage.get("completeness_passed") is not True:
                        errors.append(
                            f"{item.get('path')} is missing an assembler-verified completeness "
                            "certificate (root_coverage.completeness_passed)"
                        )
                    if graph.get("organizer_count") is None and "organizer_count" not in coverage:
                        errors.append(f"{item.get('path')} must report organizer_count")
            if not ready:
                errors.append(
                    "scientific release requires a critical_graph JSON with release_ready true "
                    "(only the assembler may set that bit)"
                )

    if errors:
        raise DiscoveryValidationError("\n".join(f"- {error}" for error in errors))


def render_summary(manifest: dict[str, Any]) -> str:
    lines = [
        f"# Discovery release: {manifest['release_id']}",
        "",
        f"**Scientific status:** {manifest['status'].upper()}",
        "",
        "## Open problem",
        "",
        manifest["problem"]["question"],
        "",
        "## Decision",
        "",
        manifest["decision"]["summary"],
        "",
        "## Closure gates",
        "",
    ]
    for gate in manifest["gates"]:
        status = gate["status"].upper()
        lines.append(f"- **{gate['id']} {gate['title']} [{status}]** — {gate['criterion']}")
    lines += ["", "## Release claims", ""]
    claims = [claim for claim in manifest["claims"] if claim["status"] == "release_claim"]
    if not claims:
        lines.append("No scientific release claim is authorized yet.")
    for claim in claims:
        lines += [
            f"### {claim['id']}",
            "",
            claim["statement"],
            "",
            f"**How:** {claim['method']}",
            "",
        ]
    if manifest.get("blockers"):
        lines += ["## Remaining blockers", ""]
        lines += [f"- {blocker}" for blocker in manifest["blockers"]]
        lines.append("")
    lines += ["## Evidence", ""]
    for item in manifest["evidence"]:
        where = item.get("path") or f"run {item.get('run_id')} / artifact {item.get('artifact_id')}"
        lines.append(f"- **{item['id']}** ({item['role']}): {item['description']} — `{where}`")
    return "\n".join(lines) + "\n"


def _latex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("#", r"\#")
        .replace("^", r"\^{}")
        .replace("~", r"\textasciitilde{}")
    )


def render_latex_claims(manifest: dict[str, Any]) -> str:
    """Render only release-authorized claims for direct manuscript inclusion."""
    claims = [claim for claim in manifest.get("claims", []) if claim.get("status") == "release_claim"]
    lines = ["% Generated from release_claim records in research/DISCOVERY_RELEASE.json.", r"\sloppy"]
    if not claims:
        lines.append(r"No scientific release claim is authorized.")
    for claim in claims:
        limitations = " ".join(str(item) for item in claim.get("limitations", []))
        lines.extend(
            [
                rf"\paragraph{{{_latex_escape(str(claim['id']).replace('-', ' ').title())}.}}",
                _latex_escape(str(claim["statement"])),
                "",
                rf"\emph{{Method.}} {_latex_escape(str(claim['method']))}",
                "",
                *(
                    [rf"\emph{{Limitations.}} {_latex_escape(limitations)}", ""]
                    if limitations
                    else []
                ),
            ]
        )
    return "\n".join(lines) + "\n"


def load_critical_graphs(manifest: dict[str, Any], root: str | Path) -> list[tuple[str, dict[str, Any]]]:
    """Return every assembler critical-graph JSON the manifest points at.

    Only records that actually carry a ``release_ready`` bit count: that bit is
    written exclusively by ``scripts/assemble_critical_graph.py``, so it is the
    single machine-authored fact that says whether the v1 graph is closed.
    """
    root = Path(root)
    graphs: list[tuple[str, dict[str, Any]]] = []
    for item in manifest.get("evidence", []):
        if not isinstance(item, dict):
            continue
        if item.get("role") != "critical_graph" or item.get("kind") != "repository_file":
            continue
        rel = str(item.get("path", ""))
        if not rel.endswith(".json"):
            continue
        path = root / rel
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "release_ready" in payload:
            graphs.append((rel, payload))
    return graphs


def evidence_state(manifest: dict[str, Any], root: str | Path) -> dict[str, Any]:
    """Collapse manifest plus assembler output into the facts prose may depend on."""
    graphs = load_critical_graphs(manifest, root)
    release_ready = bool(graphs) and all(graph.get("release_ready") is True for _, graph in graphs)
    completeness: dict[str, Any] | None = None
    for _, graph in graphs:
        candidate = graph.get("completeness")
        if isinstance(candidate, dict) and candidate.get("passed") is True:
            completeness = candidate
            break
    return {
        "status": str(manifest.get("status", "")).lower(),
        "gates": {str(g.get("id")): str(g.get("status", "")).lower() for g in manifest.get("gates", [])},
        "release_claims": sorted(
            str(c.get("id"))
            for c in manifest.get("claims", [])
            if c.get("status") == "release_claim" and _nonempty(c.get("id"))
        ),
        "release_ready": release_ready,
        "graph_paths": sorted(rel for rel, _ in graphs),
        "completeness": completeness,
        "solved": manifest.get("status") == "solved" and release_ready,
    }


def _format_interval(pair: Any) -> str | None:
    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
        return None
    try:
        low, high = float(pair[0]), float(pair[1])
    except (TypeError, ValueError):
        return None
    return rf"[{low:g},{high:g}]"


def _completeness_sentence(certificate: dict[str, Any] | None) -> str:
    """Describe exactly how far a frozen completeness certificate reaches."""
    if not isinstance(certificate, dict) or certificate.get("passed") is not True:
        return (
            "No bounded completeness certificate is frozen, so this manuscript makes "
            "no completeness claim of any kind: additional stability pockets inside "
            "the declared mass box are not excluded."
        )
    pieces: list[str] = []
    sample = certificate.get("active_learning", {}) or {}
    attempted = sample.get("attempted")
    if isinstance(attempted, int):
        pieces.append(
            rf"an active-learning pocket screen of {attempted} off-grid proposals, all of "
            "which corrected onto the known sheet and none of which was screening-stable"
        )
    grid = (certificate.get("domain", {}) or {}).get("neck") or {}
    m1_span = _format_interval(grid.get("m1"))
    m2_span = _format_interval(grid.get("m2"))
    step = grid.get("step", certificate.get("resolution"))
    if m1_span and m2_span:
        resolution = rf" at step ${float(step):g}$" if isinstance(step, (int, float)) else ""
        pieces.append(
            rf"a completed neck raster over the sub-box $m_1\in{m1_span}$, $m_2\in{m2_span}$"
            rf"{resolution} with no vertical merge"
        )
    if not pieces:
        pieces.append("a frozen certificate whose declared domain is recorded in the artifact")
    body = pieces[0] if len(pieces) == 1 else " and ".join(pieces)
    return (
        "Completeness is certified only in a bounded sense, from "
        + body
        + ". Outside that sub-box the manuscript claims no completeness: the declared "
        "mass box as a whole is not certified free of further stability pockets."
    )


def render_latex_macros(manifest: dict[str, Any], root: str | Path) -> str:
    """Emit the switches that let manuscript prose depend on the evidence state.

    ``paper/main.tex`` may not hardcode a sentence whose truth depends on how far
    the computation got.  Instead it writes ``\\atlasifsolved``, ``\\atlasifgraphready``
    or ``\\atlasifclaim`` and this file decides which branch survives.  The branch
    is chosen here, from the manifest and from the assembler's ``release_ready``
    bit, so no manuscript edit can widen a claim on its own.
    """
    state = evidence_state(manifest, root)
    esc = _latex_escape
    lines = [
        "% Generated from research/DISCOVERY_RELEASE.json plus the assembler critical graph.",
        "% Do not hand-edit: scripts/check_manuscript_claims.py requires byte equality with",
        "% the generator output, and the manuscript prose branches on these macros.",
        rf"\newcommand{{\atlasstatus}}{{{esc(state['status'].upper() or 'UNKNOWN')}}}",
        rf"\newcommand{{\atlasreleaseready}}{{{'true' if state['release_ready'] else 'false'}}}",
    ]
    for gate_id in sorted(state["gates"]):
        lines.append(
            rf"\expandafter\newcommand\csname atlasgate{esc(gate_id)}\endcsname"
            rf"{{{esc(state['gates'][gate_id].upper())}}}"
        )
    lines.append(
        r"\newcommand{\atlasifsolved}[2]{" + ("#1" if state["solved"] else "#2") + "}"
    )
    lines.append(
        r"\newcommand{\atlasifgraphready}[2]{" + ("#1" if state["release_ready"] else "#2") + "}"
    )
    lines.append(rf"\newcommand{{\atlascompleteness}}{{{_completeness_sentence(state['completeness'])}}}")
    lines.append(r"\makeatletter")
    lines.append(r"\newcommand{\atlasifclaim}[3]{\@ifundefined{atlas@claim@#1}{#3}{#2}}")
    for claim_id in state["release_claims"]:
        lines.append(rf"\expandafter\gdef\csname atlas@claim@{claim_id}\endcsname{{}}")
    lines.append(r"\makeatother")
    return "\n".join(lines) + "\n"


def render_latex_status(manifest: dict[str, Any]) -> str:
    esc = _latex_escape

    status = esc(manifest["status"].upper())
    summary = esc(manifest["decision"]["summary"])
    lines = [
        "% Generated from research/DISCOVERY_RELEASE.json.",
        r"\subsection*{Machine-readable discovery status}",
        rf"Scientific status: \textbf{{{status}}}. {summary}",
        r"\paragraph{Gates.}",
        r"\begin{itemize}",
    ]
    for gate in manifest["gates"]:
        lines.append(
            rf"\item Gate {esc(gate['id'])} --- {esc(gate['title'])}: "
            rf"\textbf{{{esc(gate['status'].upper())}}}. {esc(gate['criterion'])}"
        )
    lines.append(r"\end{itemize}")
    lines += [r"\paragraph{Authorized release claims.}", r"\begin{itemize}"]
    claims = [c for c in manifest.get("claims", []) if c.get("status") == "release_claim"]
    if not claims:
        lines.append(r"\item No scientific release claim is authorized.")
    for claim in claims:
        cid = esc(claim["id"])
        lines.append(rf"\item \textbf{{{cid}}}. {esc(claim['statement'])}")
    lines.append(r"\end{itemize}")
    if manifest.get("blockers"):
        lines += [r"\paragraph{Remaining blockers.}", r"\begin{itemize}"]
        for blocker in manifest["blockers"]:
            lines.append(rf"\item {esc(blocker)}")
        lines.append(r"\end{itemize}")
    return "\n".join(lines) + "\n"


def build_dossier(
    manifest: dict[str, Any],
    manifest_path: str | Path,
    root: str | Path,
    output: str | Path,
) -> Path:
    """Write a frozen source/evidence snapshot with a self-verifying checksum index."""
    root, output = Path(root), Path(output)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    files: dict[str, Path] = {}
    for rel in manifest["release"]["include_paths"]:
        path = root / rel
        if path.is_file():
            files[rel] = path
        elif path.is_dir():
            for child in path.rglob("*"):
                ignored = ".git" in child.parts or "__pycache__" in child.parts
                if child.is_file() and not ignored:
                    files[child.relative_to(root).as_posix()] = child

    snapshot = json.loads(json.dumps(manifest))
    snapshot["release_snapshot"] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_revision": os.getenv("GITHUB_SHA"),
        "run_id": os.getenv("GITHUB_RUN_ID"),
        "files": [
            {"path": rel, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for rel, path in sorted(files.items())
        ],
    }
    for rel, path in files.items():
        dest = output / "source" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    for item in manifest["evidence"]:
        if item.get("kind") == "generated_artifact" and (root / item["path"]).is_file():
            dest = output / "generated" / item["path"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / item["path"], dest)

    shutil.copy2(manifest_path, output / "DISCOVERY_RELEASE.json")
    (output / "discovery.json").write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "DISCOVERY_SUMMARY.md").write_text(render_summary(manifest), encoding="utf-8")
    checks = [
        f"{sha256_file(path)}  {path.relative_to(output).as_posix()}"
        for path in sorted(output.rglob("*"))
        if path.is_file()
    ]
    (output / "SHA256SUMS").write_text("\n".join(checks) + "\n", encoding="utf-8")
    return output
