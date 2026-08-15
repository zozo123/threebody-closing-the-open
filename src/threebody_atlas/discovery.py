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
                    digest = item.get("sha256")
                    if digest:
                        actual = sha256_file(graph_path)
                        if str(digest).lower() != actual:
                            errors.append(
                                f"critical graph sha256 mismatch for {item.get('path')}: "
                                f"manifest={digest} file={actual}"
                            )
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


def render_latex_status(manifest: dict[str, Any]) -> str:
    def esc(text: str) -> str:
        return (
            text.replace("\\", r"\textbackslash{}")
            .replace("_", r"\_")
            .replace("&", r"\&")
            .replace("%", r"\%")
            .replace("#", r"\#")
            .replace("^", r"\^{}")
            .replace("~", r"\textasciitilde{}")
        )

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
