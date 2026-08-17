#!/usr/bin/env python3
"""Freeze hosted continuation evidence into the canonical 139/139 certificate.

The full-domain sweep uses campaign-local root ids, so identity is established
in the corrected six-dimensional shooting chart.  Roots that reproduce an
already-ingested canonical root are then assigned through that root's sampled
component to one of the six continuously certified physical branches.  The six
genuinely new localizations must intersect a hosted continuous segment.

Input paths and GitHub run metadata come from one provenance manifest.  Every
input is hashed into the output; the graph assembler re-hashes the same files
before it consumes a single edge.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import runpy
from pathlib import Path
from typing import Any

import numpy as np

from threebody_atlas.critical_geometry import continuation_scales


ROOT = Path(__file__).resolve().parents[1]
EVENT_GATE = 2e-8
CLOSURE_GATE = 1e-7
CANONICAL_MATCH_GATE = 1e-5
CONTINUOUS_SEGMENT_GATE = 1.2e-2
PHYSICAL_SHEET = "one continuation-connected Li-Li-Liao catalog sheet"

EXPECTED_ROLES = frozenset(
    {
        "sweep_census",
        "baseline_graph",
        "continuous_campaign",
        "endpoint_0_low",
        "endpoint_1_high",
        "endpoint_12_low",
        "endpoint_12_high",
        "corridor_bridge",
        "bigfloat_marginal",
        "bigfloat_corridor",
    }
)

COMPONENT_TO_BRANCH = {
    0: "principal_left_minus_to_domain",
    1: "secondary_left_minus_to_domain",
    3: "secondary_right_minus_to_domain",
    4: "principal_right_minus_to_domain",
    5: "secondary_left_minus_to_domain",
    10: "principal_left_plus_to_secondary_left",
    11: "secondary_right_plus_to_principal_right",
    12: "secondary_right_plus_to_principal_right",
}

EXPECTED_BRANCHES = frozenset(COMPONENT_TO_BRANCH.values())


def load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path}: JSON root must be an object")
    return payload


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def vector(record: dict[str, Any]) -> np.ndarray:
    masses = record["masses"]
    return np.asarray(
        [
            float(record[key])
            for key in ("x1", "v1", "v2", "period")
        ]
        + [float(masses[0]), float(masses[1])],
        dtype=float,
    )


def segment_distance(a: np.ndarray, b: np.ndarray, target: np.ndarray) -> float:
    scale = continuation_scales(target)
    aa = (a - target) / scale
    bb = (b - target) / scale
    delta = bb - aa
    denominator = float(np.dot(delta, delta))
    if denominator == 0.0:
        return float(np.linalg.norm(aa))
    fraction = float(np.clip(-np.dot(aa, delta) / denominator, 0.0, 1.0))
    return float(np.linalg.norm(aa + fraction * delta))


def point_gate(record: dict[str, Any], *, label: str) -> None:
    event = float(record["event"])
    closure = float(record["closure"])
    if not math.isfinite(event) or abs(event) > EVENT_GATE:
        raise RuntimeError(f"{label}: event {event:.3e} misses the frozen gate")
    if not math.isfinite(closure) or closure > CLOSURE_GATE:
        raise RuntimeError(f"{label}: closure {closure:.3e} misses the frozen gate")


def provenance(path: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    manifest = load(path)
    records = manifest.get("records") or []
    by_role: dict[str, dict[str, Any]] = {}
    parents: list[dict[str, Any]] = []
    for record in records:
        role = str(record.get("role") or "")
        if role in by_role:
            raise RuntimeError(f"duplicate provenance role: {role}")
        relative = str(record.get("path") or "")
        candidate = (ROOT / relative).resolve()
        try:
            candidate.relative_to(ROOT.resolve())
        except ValueError as exc:
            raise RuntimeError(f"provenance path escapes repository: {relative}") from exc
        if not candidate.is_file():
            raise RuntimeError(f"provenance parent absent: {relative}")
        commit = str(record.get("source_commit") or "").lower()
        if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
            raise RuntimeError(f"{role}: invalid source commit")
        item = {
            **record,
            "role": role,
            "path": relative,
            "source_commit": commit,
            "sha256": sha256(candidate),
            "_absolute_path": candidate,
        }
        by_role[role] = item
        parents.append({key: value for key, value in item.items() if key != "_absolute_path"})
    missing = sorted(EXPECTED_ROLES - set(by_role))
    extra = sorted(set(by_role) - EXPECTED_ROLES)
    if missing or extra:
        raise RuntimeError(f"provenance roles differ: missing={missing}, extra={extra}")
    parents.append(
        {
            "role": "provenance_manifest",
            "path": str(path.resolve().relative_to(ROOT.resolve())),
            "source_commit": str(manifest.get("source_commit") or "").lower(),
            "sha256": sha256(path),
        }
    )
    return by_role, parents


def validate_bigfloat(marginal: dict[str, Any], corridor: dict[str, Any]) -> dict[str, Any]:
    results = [*(marginal.get("results") or []), *(corridor.get("results") or [])]
    if marginal.get("dps") != 60 or corridor.get("dps") != 60 or len(results) != 18:
        raise RuntimeError("BigFloat evidence must contain 1 + 17 roots at 60 digits")
    max_event = 0.0
    max_closure = 0.0
    for result in results:
        representative = result.get("representative") or {}
        event = abs(float(representative["event_value"]))
        closure = float(representative["closure"])
        if result.get("representative_passed_event_gate") is not True or event > EVENT_GATE:
            raise RuntimeError("BigFloat representative misses the frozen event gate")
        if closure > CLOSURE_GATE:
            raise RuntimeError("BigFloat representative misses the frozen closure gate")
        max_event = max(max_event, event)
        max_closure = max(max_closure, closure)
    return {
        "independently_reproduced_roots": len(results),
        "decimal_digits": 60,
        "max_abs_event": max_event,
        "max_closure": max_closure,
        "passed": True,
    }


def validate_endpoints(records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    expected = {
        "endpoint_0_low": "0:low",
        "endpoint_1_high": "1:high",
        "endpoint_12_low": "12:low",
        "endpoint_12_high": "12:high",
    }
    result = {}
    for role, job in expected.items():
        payload = load(records[role]["_absolute_path"])
        rows = payload.get("results") or []
        if (
            payload.get("selected_jobs") != [job]
            or payload.get("all_sampled_endpoints_resolved") is not True
            or len(rows) != 1
            or rows[0].get("scientific_endpoint_resolved") is not True
        ):
            raise RuntimeError(f"{role}: endpoint semantics are not green")
        terminal = rows[0].get("terminal") or {}
        if terminal.get("kind") not in {
            "declared_domain_boundary",
            "mixed_organizer",
            "closed_loop",
        }:
            raise RuntimeError(f"{role}: endpoint has no legitimate terminal")
        points = rows[0].get("accepted_points") or []
        if not points:
            raise RuntimeError(f"{role}: endpoint has no accepted continuous points")
        for index, point in enumerate(points):
            point_gate(point, label=f"{role}:{index}")
        result[job] = {
            "terminal": terminal,
            "accepted_points": len(points),
            "max_abs_event": max(abs(float(point["event"])) for point in points),
            "max_closure": max(float(point["closure"]) for point in points),
        }
    return result


def validate_campaign(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if payload.get("schema") != "atlas.v1.label-invisible-continuation/2":
        raise RuntimeError("continuous campaign did not finish as a /2 certificate")
    if payload.get("continuous_witness_passed") is not True:
        raise RuntimeError("continuous campaign is not passed")
    if payload.get("all_nine_seed_continuations_passed") is not True:
        raise RuntimeError("the original nine-seed regression corpus is not passed")
    branches = payload.get("branches") or []
    branch_ids = {str(branch.get("branch_id") or "") for branch in branches}
    if branch_ids != EXPECTED_BRANCHES or len(branches) != len(EXPECTED_BRANCHES):
        raise RuntimeError("continuous campaign lacks one of the six physical branches")

    continuous_edges = []
    summary: dict[str, Any] = {}
    for branch in branches:
        branch_id = str(branch["branch_id"])
        if branch.get("continuous_branch_passed") is not True:
            raise RuntimeError(f"{branch_id}: branch semantics are not green")
        overlap = branch.get("sampled_component_overlap") or []
        if not overlap or not all(item.get("covered") for item in overlap):
            raise RuntimeError(f"{branch_id}: sampled component overlap is incomplete")
        witnesses = branch.get("issue_seed_witnesses") or []
        if not all(
            witness.get("continuous_incidence_passed") is True
            and witness.get("jax_diagnostics_passed") is True
            for witness in witnesses
        ):
            raise RuntimeError(f"{branch_id}: a seed or JAX tangent witness failed")
        current = ((branch.get("start_germs") or {}).get("current") or {}).get("point")
        accepted = branch.get("accepted_points") or []
        if not isinstance(current, dict) or not accepted:
            raise RuntimeError(f"{branch_id}: branch has no continuous geometry")
        vertices = [current, *accepted]
        for index, point in enumerate(vertices):
            point_gate(point, label=f"{branch_id}:{index}")
        terminal = branch.get("terminal") or {}
        if terminal.get("kind") == "canonically_bound_continuation_germ":
            end_terminal = {
                "kind": terminal["kind"],
                "node_id": str(terminal["target"]).split(":", 1)[0],
                "masses": vertices[-1]["masses"],
            }
        elif terminal.get("kind") == "declared_domain_boundary":
            end_terminal = {
                "kind": terminal["kind"],
                "masses": terminal["masses"],
            }
        else:
            raise RuntimeError(f"{branch_id}: illegitimate terminal {terminal}")
        source_id = str(((branch.get("start_germs") or {}).get("current") or {}).get("source_id") or "")
        start_node = source_id.split(":", 1)[0]
        if not start_node:
            raise RuntimeError(f"{branch_id}: start germ does not name an organizer")
        continuous_edges.append(
            {
                "branch_id": branch_id,
                "event_mode": str(branch["event_mode"]),
                "start_terminal": {
                    "kind": "canonically_bound_continuation_germ",
                    "node_id": start_node,
                    "masses": vertices[0]["masses"],
                },
                "end_terminal": end_terminal,
                "vertices": vertices,
                "sampled_components": sorted(
                    {int(item["component"]) for item in overlap}
                ),
                "issue_seed_witnesses": len(witnesses),
                "retry_history": branch.get("retry_history") or [],
                "stopped_reason": branch.get("stopped_reason"),
            }
        )
        summary[branch_id] = {
            "accepted_points": len(accepted),
            "sampled_components": continuous_edges[-1]["sampled_components"],
            "issue_seed_witnesses": len(witnesses),
            "max_abs_event": max(abs(float(point["event"])) for point in vertices),
            "max_closure": max(float(point["closure"]) for point in vertices),
        }
    return continuous_edges, summary


def sampled_cluster_map(census: dict[str, Any]) -> dict[tuple[str, float, float], int]:
    out: dict[tuple[str, float, float], int] = {}
    for index, component in enumerate(census.get("curve_components") or []):
        mechanism = str(component["mechanism"])
        for vertex in component.get("vertices") or []:
            key = (mechanism, round(float(vertex["m1"]), 12), round(float(vertex["m2"]), 12))
            if key in out:
                raise RuntimeError(f"one sweep root belongs to two sampled clusters: {key}")
            out[key] = index
    return out


def stable_root_id(record: dict[str, Any]) -> str:
    identity = {
        "mechanism": record["mechanism"],
        "masses": [float(value) for value in record["masses"]],
        "event_value": float(record["event_value"]),
        "closure": float(record["closure"]),
        "shard_label": record.get("shard_label"),
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return f"full-domain-root-{digest}"


def nearest_canonical(
    target: dict[str, Any], canonical: list[dict[str, Any]]
) -> tuple[float, dict[str, Any]]:
    target_vector = vector(target)
    candidates = [
        root
        for root in canonical
        if str(root.get("event_mode")) == str(target.get("mechanism"))
    ]
    return min(
        (
            float(
                np.linalg.norm(
                    (target_vector - vector(root)) / continuation_scales(vector(root))
                )
            ),
            root,
        )
        for root in candidates
    )


def segment_witnesses(
    endpoint0: dict[str, Any], bridge: dict[str, Any]
) -> dict[str, list[np.ndarray]]:
    endpoint_result = (endpoint0.get("results") or [None])[0]
    if not isinstance(endpoint_result, dict):
        raise RuntimeError("endpoint 0 evidence has no result")
    low_points = [endpoint_result["seed_current"], *(endpoint_result.get("accepted_points") or [])]
    bridge_points = [bridge["seed_current"]["point"], *(bridge.get("accepted_points") or [])]
    return {
        "principal_left_minus_to_domain": [vector(point) for point in low_points],
        "secondary_left_minus_to_domain": [vector(point) for point in bridge_points],
    }


def reconcile_ledger(
    census: dict[str, Any], segment_paths: dict[str, list[np.ndarray]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    canonical = runpy.run_path(str(ROOT / "scripts/graph_root_sources.py"))[
        "load_roots"
    ]()
    clusters = sampled_cluster_map(census)
    off_graph = [
        row
        for row in census.get("localizations") or []
        if row.get("status") == "passed"
        and not (row.get("committed_edge") or {}).get("matched")
    ]
    if len(off_graph) != 139:
        raise RuntimeError(f"expected 139 off-graph roots, got {len(off_graph)}")
    ledger = []
    equivalent = 0
    new_segment = 0
    for row in off_graph:
        mechanism = str(row["mechanism"])
        masses = row["masses"]
        cluster_key = (mechanism, round(float(masses[0]), 12), round(float(masses[1]), 12))
        if cluster_key not in clusters:
            raise RuntimeError(f"off-graph root has no sampled cluster: {cluster_key}")
        distance, matched = nearest_canonical(row, canonical)
        entry: dict[str, Any] = {
            "root_id": stable_root_id(row),
            "sampled_cluster_id": f"full-domain-component-{clusters[cluster_key]}",
            "mechanism": mechanism,
            "masses": [float(value) for value in masses],
            "event": float(row["event_value"]),
            "closure": float(row["closure"]),
            "source_shard": row.get("shard_label"),
        }
        if distance <= CANONICAL_MATCH_GATE:
            component = int(matched.get("sweep_component", -1))
            branch = COMPONENT_TO_BRANCH.get(component)
            if branch is None:
                raise RuntimeError(
                    f"canonical-equivalent root belongs to unmapped component {component}"
                )
            entry.update(
                {
                    "resolution": "campaign_equivalent_canonical_root",
                    "physical_object": branch,
                    "canonical_cell_id": int(matched["cell_id"]),
                    "canonical_sweep_component": component,
                    "six_dimensional_miss_scaled": distance,
                }
            )
            equivalent += 1
        else:
            target = vector(row)
            best_branch = None
            best_miss = math.inf
            for branch, points in segment_paths.items():
                for left, right in zip(points, points[1:], strict=False):
                    miss = segment_distance(left, right, target)
                    if miss < best_miss:
                        best_branch, best_miss = branch, miss
            if best_branch is None or best_miss > CONTINUOUS_SEGMENT_GATE:
                raise RuntimeError(
                    f"new root {masses[:2]} misses every continuous segment: {best_miss:.3e}"
                )
            entry.update(
                {
                    "resolution": "new_root_on_continuous_segment",
                    "physical_object": best_branch,
                    "six_dimensional_segment_miss_scaled": best_miss,
                }
            )
            new_segment += 1
        ledger.append(entry)
    if (equivalent, new_segment) != (133, 6):
        raise RuntimeError(
            f"physical reconciliation changed: canonical={equivalent}, new={new_segment}"
        )
    ledger.sort(key=lambda row: row["root_id"])
    return ledger, {
        "campaign_equivalent_canonical_roots": equivalent,
        "genuinely_new_continuous_segment_roots": new_segment,
        "maximum_canonical_match_gate": CANONICAL_MATCH_GATE,
        "maximum_continuous_segment_miss_gate": CONTINUOUS_SEGMENT_GATE,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("provenance_manifest")
    parser.add_argument("output")
    args = parser.parse_args()

    manifest_path = Path(args.provenance_manifest).resolve()
    records, parents = provenance(manifest_path)
    census = load(records["sweep_census"]["_absolute_path"])
    headline = census.get("headline") or {}
    if headline.get("roots_absent_from_the_committed_graph") != 139:
        raise RuntimeError("sweep headline no longer reports 139 off-graph roots")

    endpoints = validate_endpoints(records)
    bridge = load(records["corridor_bridge"]["_absolute_path"])
    if (
        bridge.get("continuous_bridge_passed") is not True
        or bridge.get("accepted_points_after_target", 0) < 3
        or (bridge.get("isolated_target") or {}).get("jax_diagnostics_passed") is not True
    ):
        raise RuntimeError("low-minus corridor bridge semantics are not green")
    for index, point in enumerate(bridge.get("accepted_points") or []):
        point_gate(point, label=f"corridor_bridge:{index}")

    campaign = load(records["continuous_campaign"]["_absolute_path"])
    continuous_edges, branch_summary = validate_campaign(campaign)
    bigfloat = validate_bigfloat(
        load(records["bigfloat_marginal"]["_absolute_path"]),
        load(records["bigfloat_corridor"]["_absolute_path"]),
    )
    endpoint0 = load(records["endpoint_0_low"]["_absolute_path"])
    ledger, accounting_summary = reconcile_ledger(
        census, segment_witnesses(endpoint0, bridge)
    )

    coverage = census.get("coverage") or {}
    result = {
        "schema": "atlas.v1.continuous-critical-reconciliation/1",
        "passed": True,
        "claim_status": (
            "all 139 currently observed off-graph roots reconciled to six "
            "continuation-certified physical branches"
        ),
        "physical_sheet": PHYSICAL_SHEET,
        "frozen_gates": {
            "maximum_absolute_event": EVENT_GATE,
            "maximum_periodic_closure": CLOSURE_GATE,
        },
        "parents": parents,
        "independent_bigfloat": bigfloat,
        "endpoint_resolutions": endpoints,
        "corridor_bridge": {
            "passed": True,
            "target_hit_step": bridge.get("target_hit_step"),
            "accepted_points_after_target": bridge.get("accepted_points_after_target"),
        },
        "branch_summary": branch_summary,
        "continuous_edges": continuous_edges,
        "off_graph_accounting": {
            "observed_roots": 139,
            "resolved_roots": len(ledger),
            "unresolved_roots": [],
            **accounting_summary,
            "ledger": ledger,
        },
        "partial_sweep_blind_spots": [
            coverage.get("honest_scope"),
            {
                "m1_scan_line_fraction_of_published_slices": coverage.get(
                    "m1_scan_line_fraction_of_published_slices"
                ),
                "probe_fraction_of_published_raster_rows": coverage.get(
                    "probe_fraction_of_published_raster_rows"
                ),
            },
            "Even-order roots, same-cell double crossings, and curves confined between adjacent scan lines remain unexcluded by this finite sign-change sweep.",
            "This closes the observed 139-root contradiction; it is not a proof that no unsampled critical curve exists.",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "passed": True,
                "observed_roots": 139,
                "resolved_roots": len(ledger),
                "continuous_edges": len(continuous_edges),
                **accounting_summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
