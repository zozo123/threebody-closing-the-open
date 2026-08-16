#!/usr/bin/env python3
"""Assemble the v1 mechanism-resolved Floquet critical graph.

620 catalog S/U cells are samples supporting the graph. They are not 620 edges.
An edge is a mechanism-specific polyline carrying a list of source-cell ids.
Endpoints, mixed germs, and completeness must come from artifacts.
The assembler never invents a classification and never flips
release_ready without the mandatory artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


try:  # pragma: no cover - exercised implicitly by both install layouts
    from threebody_atlas.completeness import verification_report
    from threebody_atlas.conditioning import summarize_conditioning
except ModuleNotFoundError:  # running from a source checkout without an install
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from threebody_atlas.completeness import verification_report
    from threebody_atlas.conditioning import summarize_conditioning

REQUIRED_HEADLINE_IDS = (
    "mixed_principal_left",
    "mixed_secondary_left",
    "mixed_principal_right",
    "headline_lower_plus_one",
    "headline_upper_collision",
)
BASE_MIXED_NODE_IDS = (
    "mixed_principal_left",
    "mixed_secondary_left",
    "mixed_principal_right",
)
REQUIRED_GERM_KEYS = (
    ("plus_one", "+"),
    ("plus_one", "-"),
    ("minus_one", "+"),
    ("minus_one", "-"),
)
LEFT_BIRTH_CLASSES = frozenset(
    {"projection_fold", "two_separate_arcs", "mixed_organizer"}
)
RIGHT_DEATH_CLASSES = frozenset({"mixed_organizer", "projection_fold", "domain_boundary"})
DAUGHTER_CLASSES = frozenset(
    {
        "reconnecting",
        "closed_loop",
        "distinct_branch",
        "obstruction",
        "no_branch_attachment",
        "falsified",
    }
)

# ---------------------------------------------------------------------------
# Frozen numerical gates.  These are the project-wide gates; they are repeated
# here (not re-derived) so germ records are held to exactly the same bar as
# localized roots.  They may only ever be made STRICTER.
# ---------------------------------------------------------------------------
EVENT_GATE = 2e-8
CLOSURE_GATE = 1e-7

# ---------------------------------------------------------------------------
# Mass-plane grid.
#
# The 620 catalog transition cells are sampled on a uniform mass grid whose
# step is 0.001 in BOTH m1 and m2.  Verified against the release root set
# research/evidence/V1_HYBRID_CRITICAL_ROOTS_2026-08-15.json: its 620 localized
# roots occupy 272 distinct m1 values whose ONLY successive difference is
# exactly 0.001, and every ``source_m2_bracket`` has width exactly 0.001.
#
# Every tolerance below is a multiple of this one step, so a future regrid can
# be reasoned about rather than guessed at.  ``tests/test_critical_graph.py``
# pins the grid and the empirical margin of each constant.
# ---------------------------------------------------------------------------
MASS_GRID_STEP = 0.001
GRID_DECIMALS = max(0, -int(round(math.log10(MASS_GRID_STEP))))

# MASS_JUMP -- 25 grid steps.
# Maximum mass-plane distance at which two roots living in linkable m1 slices
# may be joined into the same mechanism polyline.
# Empirical margin on the release root set: the largest link actually ACCEPTED
# is 0.011097 (minus_one S->U, cells 393 -> 397); the smallest cross-slice link
# it REJECTS is 0.096141 (plus_one U->S, cells 576 -> 594).  Admissible window
# [0.011097, 0.096141), a factor of 8.66; 0.025 sits 2.25x above the largest
# accepted and 3.85x below the smallest rejected.
# LOAD-BEARING ON THE LOW SIDE: lowering it below 0.011097 splits minus_one
# S->U into two edges (edge_count 8 instead of 7).  On the high side it and
# M1_SLICE_GAP guard the same two joins redundantly -- relaxing either one
# alone leaves edge_count at 7, while relaxing BOTH (jump >= 0.096141 and
# slice_gap >= 0.008) merges plus_one_u_to_s_1 into plus_one_u_to_s_2.
# Pinned by test_mass_jump_window_is_pinned.
MASS_JUMP = 0.025

# M1_SLICE_GAP -- 1.5 grid steps.
# Two consecutive m1 slices may only be linked when they are grid-adjacent.
# Empirical margin on the release root set: the successive-slice gaps that
# actually occur are exactly {0.001, 0.008, 0.068}.  0.0015 sits at 1.5x the
# adjacent step and 5.33x below the smallest genuine gap, an admissible window
# 8x wide.
# CURRENTLY REDUNDANT WITH MASS_JUMP: both non-adjacent slice pairs it excludes
# (the 0.008 gap and the 0.068 gap in plus_one U->S) are already excluded by
# MASS_JUMP, because their closest cross-slice pairs are 0.096141 and 0.098435
# apart.  Relaxing it all the way to 0.068 leaves edge_count at 7.  It is
# retained as an independent, purely topological guard -- only grid-adjacent
# slices may link, whatever the mass-plane distance happens to be -- and its
# redundancy is pinned by test_m1_slice_gap_is_currently_redundant so that a
# future data set in which it starts biting is noticed rather than assumed.
M1_SLICE_GAP = 0.0015

# GERM_ATTACH_DISTANCE -- 8 grid steps.  Two jobs:
#   (a) the maximum mass-plane distance at which an edge endpoint may be
#       attached to a continuation germ, and
#   (b) the maximum canonical_distance a germ may sit from its organizer.
# DOUBLE DUTY IS NOT FREE: because (a) and (b) compose, an edge endpoint may be
# bound to an organizer it sits up to 2 x 0.008 = 0.016 away from.  The
# effective organizer-to-endpoint reach of this constant is 0.016, and that is
# the number to compare against when asking whether an attachment is close.
# Empirical margin on the RELEASE configuration (scripts/assemble_v1_critical_graph.sh,
# i.e. the four 2026-08-16 germ artifacts, not the superseded
# V1_MIXED_GERMS_2026-08-15.json and not a test fixture): the largest ACCEPTED
# attachment is 0.006837449100337747 (minus_one_u_to_s_1 end ->
# mixed_secondary_left, minus_one/-) and the nearest REJECTED mode-matching
# candidate is 0.00943057726978081 (plus_one_u_to_s_1 end <->
# secondary_right_death, plus_one/-).  The admissible window is
# [0.006837449100337747, 0.00943057726978081) -- half-open, since a threshold
# equal to the rejected distance would admit it -- only 1.3793x wide; 0.008
# sits 1.17x above the largest accepted and 1.18x below the nearest rejected.
# The assembled graph does not visibly change until 0.010349059396785794
# (1.5136x), where minus_one_u_to_s_1's start gains a seventh attachment,
# because the 0.00943 candidate's endpoint is already bound by
# V1_SECONDARY_RIGHT_CLASS_2026-08-16.json.
# WHAT THIS CONSTANT NO LONGER GUARDS: minus_one_s_to_u_0's start endpoint is
# the secondary_left_birth blocker, and on the release germs it is 2.81e-2 from
# mixed_secondary_left (2.6379e-2 from that organizer's nearest mode-matching
# germ) -- beyond even the 0.016 composed reach.  This constant would have to
# grow 3.30x before it could glue that blocker, so it is no longer what keeps
# the blocker honest; the classification artifact is.  (The behavioural test
# test_widening_germ_attach_distance_would_resolve_a_blocked_endpoint still
# demonstrates the gluing at 0.0105, but on synthetic germ fixtures whose
# masses sit closer to that endpoint than the released germs do.)
# Re-derive all of the above with scripts/audit_germ_attachment_window.py.
# Pinned by test_germ_attach_distance_window_is_pinned.
GERM_ATTACH_DISTANCE = 0.008

# DOMAIN_TOLERANCE -- 1.5 grid steps.
# How close an edge endpoint must sit to a declared face of the mass box for
# the terminus to count as a declared domain exit.
# Empirical margin on the release configuration: the four real boundary exits
# sit 0.0, 0.0, 3.97e-5 and 4.17e-4 from their face, while the nearest
# non-boundary endpoint is 0.05 away -- a margin better than 100x.  This
# constant is not delicate; it is the LUMPING of distinct exits onto one node
# that used to be wrong, and that is fixed by domain_exit_node_id below.
DOMAIN_TOLERANCE = 0.0015

DECLARED_DOMAIN = {"m1": (0.8, 1.1), "m2": (0.7, 1.2)}

# Substrings that mark a stopped_reason as a recorded *failure* of the
# continuation that produced the germ.  A germ whose own artifact records that
# its corrector never converged is not evidence of a germ, whatever its
# ``status`` field says.  Absence of a stopped_reason is not treated as a
# failure -- germ tracers that emit full numerics (closure/event/canonical
# binding) do not always emit one -- but those numerics are required of every
# germ regardless, so omission cannot buy a pass.
GERM_FAILURE_TOKENS = (
    "fail",
    "error",
    "exceed",
    "diverg",
    "nonconverg",
    "non-converg",
    "not converg",
    "unconverged",
    "abort",
    "singular",
    "timeout",
    "stall",
    "max_iter",
    "maximum number",
    "could not",
    "unable",
)
RELEASE_EVIDENCE_LEVELS = frozenset(
    {"independently_reproduced", "physical", "continuation", "definition"}
)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def node(node_id: str, kind: str, *, status: str, masses: list[Any] | None = None, **extra: Any) -> dict[str, Any]:
    record = {"id": node_id, "kind": kind, "status": status, "masses": masses}
    record.update(extra)
    return record


def headline_nodes(root: Path) -> list[dict[str, Any]]:
    mixed_left = load(root / "research/evidence/V1_MIXED_CANONICAL_PRINCIPAL_LEFT_2026-08-15.json")
    mixed_sec = load(root / "research/evidence/V1_MIXED_CANONICAL_SECONDARY_LEFT_2026-08-15.json")
    mixed_right = load(root / "research/evidence/V1_MIXED_CANONICAL_PRINCIPAL_RIGHT_2026-08-15.json")
    plus_one = load(root / "research/evidence/V1_CANONICAL_LOWER_PLUS_ONE_2026-08-15.json")
    collision = load(root / "research/evidence/V1_CANONICAL_UPPER_COLLISION_2026-08-15.json")
    return [
        node(
            "mixed_principal_left",
            "mixed_organizer",
            status="independently_reproduced",
            masses=mixed_left.get("masses"),
            mechanism="mixed_plus_one_minus_one",
            evidence="research/evidence/V1_MIXED_CANONICAL_PRINCIPAL_LEFT_2026-08-15.json",
            passed=mixed_left.get("passed"),
        ),
        node(
            "mixed_secondary_left",
            "mixed_organizer",
            status="independently_reproduced",
            masses=mixed_sec.get("masses"),
            mechanism="mixed_plus_one_minus_one",
            evidence="research/evidence/V1_MIXED_CANONICAL_SECONDARY_LEFT_2026-08-15.json",
            passed=mixed_sec.get("passed"),
        ),
        node(
            "mixed_principal_right",
            "mixed_organizer",
            status="independently_reproduced",
            masses=mixed_right.get("masses"),
            mechanism="mixed_plus_one_minus_one",
            evidence="research/evidence/V1_MIXED_CANONICAL_PRINCIPAL_RIGHT_2026-08-15.json",
            passed=mixed_right.get("passed"),
        ),
        node(
            "headline_lower_plus_one",
            "event_representative",
            status="independently_reproduced",
            masses=None,
            mechanism=plus_one.get("mechanism"),
            evidence="research/evidence/V1_CANONICAL_LOWER_PLUS_ONE_2026-08-15.json",
            passed=plus_one.get("passed"),
            bracket_m2=plus_one.get("critical_bracket_m2"),
        ),
        node(
            "headline_upper_collision",
            "event_representative",
            status="independently_reproduced",
            masses=None,
            mechanism=collision.get("mechanism"),
            evidence="research/evidence/V1_CANONICAL_UPPER_COLLISION_2026-08-15.json",
            passed=collision.get("passed"),
            bracket_m2=collision.get("critical_bracket_m2"),
        ),
    ]


def forbidden_class(value: Any) -> bool:
    text = str(value or "").lower().replace(" ", "_")
    return "newton" in text and "fail" in text


def load_classification(
    path: Path | None,
    *,
    node_id: str,
    default_kind: str,
    allowed: frozenset[str],
    missing_note: str,
) -> dict[str, Any]:
    if path is None:
        return node(
            node_id,
            default_kind,
            status="unresolved",
            masses=None,
            mechanism="unknown",
            note=missing_note,
            passed=False,
        )
    payload = load(path)
    raw_class = payload.get("class") or payload.get("mechanism") or payload.get("classification")
    if forbidden_class(raw_class) or forbidden_class(payload.get("status")) or forbidden_class(payload.get("error")):
        return node(
            node_id,
            default_kind,
            status="illegal",
            masses=payload.get("masses"),
            mechanism=str(raw_class),
            note="Newton-failed is not an allowed endpoint class",
            passed=False,
            evidence=str(path),
        )
    evidence_level = str(payload.get("evidence_level") or "screening")
    passed = (
        bool(payload.get("passed"))
        and str(raw_class) in allowed
        and evidence_level in RELEASE_EVIDENCE_LEVELS
    )
    return node(
        node_id,
        str(payload.get("kind") or default_kind),
        status="independently_reproduced" if passed else "unresolved",
        masses=payload.get("masses"),
        mechanism=str(raw_class) if raw_class is not None else "unknown",
        note=payload.get("note"),
        passed=passed,
        evidence=str(path),
        estimator=payload.get("estimator"),
        evidence_level=evidence_level,
        screening_passed=bool(payload.get("passed")),
        edge_endpoint_bindings=payload.get("edge_endpoint_bindings") or [],
    )


def root_residual_margin(roots: list[dict[str, Any]]) -> dict[str, Any]:
    """How much of the frozen 2e-8 event budget the census actually consumes.

    ``max |event| <= 2e-8`` is not a finding on its own.  A reader needs to know
    whether the worst cell sits at 1% or at 99% of the gate, how many cells are
    in the top decade, and -- when the producing run recorded it -- the
    conditioning that turns those residuals into m2 uncertainties.  Reporting
    only; this function never gates anything.
    """
    events = sorted(abs(float(root.get("event") or 0.0)) for root in roots)
    if not events:
        return {"localized_roots": 0}

    def quantile(q: float) -> float:
        pos = q * (len(events) - 1)
        lo = int(math.floor(pos))
        hi = min(lo + 1, len(events) - 1)
        return events[lo] * (1.0 - (pos - lo)) + events[hi] * (pos - lo)

    return {
        "localized_roots": len(events),
        "event_gate": EVENT_GATE,
        "max_abs_event": events[-1],
        "gate_occupancy_max": events[-1] / EVENT_GATE,
        "headroom_fraction": 1.0 - events[-1] / EVENT_GATE,
        "median_abs_event": quantile(0.5),
        "p90_abs_event": quantile(0.90),
        "p99_abs_event": quantile(0.99),
        "roots_above_1e_8": sum(1 for value in events if value > 1e-8),
        "roots_above_half_gate": sum(1 for value in events if value > 0.5 * EVENT_GATE),
        "roots_above_95_percent_of_gate": sum(
            1 for value in events if value > 0.95 * EVENT_GATE
        ),
        "closure_conditioning": summarize_conditioning(
            [root.get("closure_conditioning") for root in roots]
        ),
        "event_conditioning": summarize_conditioning(
            [root.get("event_conditioning") for root in roots]
        ),
        "max_reported_m2_uncertainty": max(
            (
                float(root["m2_uncertainty"])
                for root in roots
                if root.get("m2_uncertainty") is not None
            ),
            default=None,
        ),
    }


def polyline_edges(
    roots: list[dict[str, Any]],
    *,
    jump: float = MASS_JUMP,
    slice_gap: float = M1_SLICE_GAP,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for root in roots:
        grouped[
            (
                str(root.get("event_mode") or "unknown"),
                str(root.get("orientation") or "unknown"),
            )
        ].append(root)
    edges: list[dict[str, Any]] = []
    for (mode, orientation), items in grouped.items():
        by_slice: dict[float, list[dict[str, Any]]] = defaultdict(list)
        for item in items:
            masses = item.get("masses") or [0.0, 0.0]
            by_slice[round(float(masses[0]), 10)].append(item)
        parent = {int(item["cell_id"]): int(item["cell_id"]) for item in items}

        def find(cell: int, parent_map: dict[int, int] = parent) -> int:
            while parent_map[cell] != cell:
                parent_map[cell] = parent_map[parent_map[cell]]
                cell = parent_map[cell]
            return cell

        def union(
            left: int,
            right: int,
            parent_map: dict[int, int] = parent,
        ) -> None:
            a, b = find(left), find(right)
            if a != b:
                parent_map[max(a, b)] = min(a, b)

        slice_keys = sorted(by_slice)
        for left_key, right_key in zip(slice_keys, slice_keys[1:], strict=False):
            if right_key - left_key > slice_gap:
                continue
            candidates: list[tuple[float, int, int]] = []
            for left in by_slice[left_key]:
                for right in by_slice[right_key]:
                    lm, rm = left.get("masses") or [0.0, 0.0], right.get("masses") or [0.0, 0.0]
                    distance = (
                        (float(lm[0]) - float(rm[0])) ** 2
                        + (float(lm[1]) - float(rm[1])) ** 2
                    ) ** 0.5
                    if distance <= jump:
                        candidates.append((distance, int(left["cell_id"]), int(right["cell_id"])))
            used_left: set[int] = set()
            used_right: set[int] = set()
            for _distance, left_cell, right_cell in sorted(candidates):
                if left_cell in used_left or right_cell in used_right:
                    continue
                union(left_cell, right_cell)
                used_left.add(left_cell)
                used_right.add(right_cell)

        components: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for item in items:
            components[find(int(item["cell_id"]))].append(item)
        runs = sorted(
            components.values(),
            key=lambda run: min(int(item["cell_id"]) for item in run),
        )
        orientation_id = orientation.lower().replace("->", "_to_").replace("-", "_")
        for index, run in enumerate(runs):
            run.sort(
                key=lambda item: (
                    float((item.get("masses") or [0.0, 0.0])[0]),
                    float((item.get("masses") or [0.0, 0.0])[1]),
                    int(item["cell_id"]),
                )
            )
            cell_ids = [int(row["cell_id"]) for row in run]
            start_masses = [float(value) for value in (run[0].get("masses") or [])]
            end_masses = [float(value) for value in (run[-1].get("masses") or [])]
            widths = []
            for row in run:
                bracket = row.get("source_m2_bracket") or []
                if len(bracket) == 2:
                    widths.append(abs(float(bracket[1]) - float(bracket[0])))
            edges.append(
                {
                    "id": f"{mode}_{orientation_id}_{index}",
                    "kind": "mechanism_polyline",
                    "mechanism": mode,
                    "orientation": orientation,
                    "cell_ids": cell_ids,
                    "source_cell_count": len(cell_ids),
                    "estimators": sorted({str(row.get("estimator") or "float64") for row in run}),
                    "uncertainty": {
                        "max_abs_event": max(abs(float(row.get("event") or 0.0)) for row in run),
                        "max_closure": max(float(row.get("closure") or 0.0) for row in run),
                        "max_source_m2_bracket_width": max(widths) if widths else None,
                    },
                    "endpoints": {
                        "start": {"cell_id": cell_ids[0], "masses": start_masses, "node": None},
                        "end": {"cell_id": cell_ids[-1], "masses": end_masses, "node": None},
                        "classified": False,
                    },
                }
            )
    edges.sort(key=lambda edge: (str(edge["mechanism"]), edge["cell_ids"][0]))
    return edges


def mass_distance(left: list[Any] | None, right: list[Any] | None) -> float:
    if not left or not right or len(left) < 2 or len(right) < 2:
        return float("inf")
    return ((float(left[0]) - float(right[0])) ** 2 + (float(left[1]) - float(right[1])) ** 2) ** 0.5


def snap_to_grid(value: float) -> float:
    """Round a mass coordinate onto the sampling grid."""
    return round(round(float(value) / MASS_GRID_STEP) * MASS_GRID_STEP, GRID_DECIMALS + 3)


def grid_label(value: float) -> str:
    """Filename/id-safe label for a grid-snapped mass coordinate."""
    return f"{snap_to_grid(value):.{GRID_DECIMALS}f}".replace("-", "neg").replace(".", "p")


def domain_face_hit(masses: list[Any] | None) -> dict[str, Any] | None:
    """Locate the declared face a terminus ran into, if any.

    Returns the face, the distance to it, and the coordinate that runs ALONG
    that face -- the coordinate which distinguishes one exit from another.
    """
    if not masses or len(masses) < 2:
        return None
    m1, m2 = float(masses[0]), float(masses[1])
    candidates = (
        (abs(m1 - DECLARED_DOMAIN["m1"][0]), "domain_m1_min", "m2", m2),
        (abs(m1 - DECLARED_DOMAIN["m1"][1]), "domain_m1_max", "m2", m2),
        (abs(m2 - DECLARED_DOMAIN["m2"][0]), "domain_m2_min", "m1", m1),
        (abs(m2 - DECLARED_DOMAIN["m2"][1]), "domain_m2_max", "m1", m1),
    )
    distance, face, along_axis, along = min(candidates)
    if distance > DOMAIN_TOLERANCE:
        return None
    return {
        "face": face,
        "distance_to_face": distance,
        "along_axis": along_axis,
        "along": along,
        "along_grid": snap_to_grid(along),
    }


def domain_exit_node_id(hit: dict[str, Any]) -> str:
    """Name a domain exit by its face AND the grid cell where it left the box.

    Two curves that both run into the same wall at different places are two
    different termini, not one graph node.  Collapsing them onto a bare face id
    manufactures incidence between curves that never meet: on the release
    configuration it glued plus_one_u_to_s_0 (exit m2=0.75572) to
    trace_collision_s_to_u_0 (exit m2=0.76073), five grid steps away, and
    plus_one_u_to_s_2 (exit m1=1.071) to trace_collision_s_to_u_0 (exit
    m1=1.053), eighteen grid steps away.  The exit coordinate is snapped to the
    sampling grid, which is the finest distinction the data supports.
    """
    return f"{hit['face']}_{hit['along_axis']}_{grid_label(hit['along'])}"


def domain_node(masses: list[Any] | None) -> str | None:
    """Per-exit declared-domain-boundary node id, or None if not a domain exit."""
    hit = domain_face_hit(masses)
    return None if hit is None else domain_exit_node_id(hit)


def apply_classification_bindings(
    edges: list[dict[str, Any]], nodes: list[dict[str, Any]]
) -> list[str]:
    errors: list[str] = []
    existing_nodes = {str(item["id"]): item for item in nodes}
    for item in list(nodes):
        for binding in item.get("edge_endpoint_bindings") or []:
            side = str(binding.get("side") or "")
            if side not in {"start", "end"}:
                errors.append(f"{item['id']}: binding side must be start or end")
                continue
            try:
                cell_id = int(binding["cell_id"])
            except (KeyError, TypeError, ValueError):
                errors.append(f"{item['id']}: binding needs an integer cell_id")
                continue
            matches = [
                edge
                for edge in edges
                if int(edge["endpoints"][side]["cell_id"]) == cell_id
                and (
                    binding.get("mechanism") is None
                    or edge.get("mechanism") == binding.get("mechanism")
                )
                and (
                    binding.get("orientation") is None
                    or edge.get("orientation") == binding.get("orientation")
                )
            ]
            if len(matches) != 1:
                errors.append(
                    f"{item['id']}: binding cell={cell_id} side={side} matched {len(matches)} edges"
                )
                continue
            endpoint = matches[0]["endpoints"][side]
            if endpoint.get("node") or endpoint.get("reserved_for"):
                errors.append(f"{item['id']}: duplicate binding for cell={cell_id} side={side}")
                continue
            endpoint["attachment"] = (
                "verified_classification_artifact"
                if item.get("passed")
                else "unverified_classification_artifact"
            )
            endpoint["binding_class"] = item.get("mechanism")
            endpoint["binding_evidence"] = item.get("evidence")
            if item.get("passed"):
                attachment_node = str(binding.get("node_id") or item["id"])
                if not attachment_node:
                    errors.append(f"{item['id']}: binding node_id must be non-empty")
                    continue
                if attachment_node not in existing_nodes:
                    derived = node(
                        attachment_node,
                        str(binding.get("node_kind") or item.get("mechanism") or item.get("kind")),
                        status=str(item.get("status")),
                        masses=binding.get("masses"),
                        mechanism=item.get("mechanism"),
                        passed=True,
                        evidence=item.get("evidence"),
                        evidence_level=item.get("evidence_level"),
                        parent_classification=item["id"],
                    )
                    nodes.append(derived)
                    existing_nodes[attachment_node] = derived
                elif attachment_node != item["id"]:
                    existing = existing_nodes[attachment_node]
                    if existing.get("parent_classification") != item["id"]:
                        errors.append(
                            f"{item['id']}: binding node_id {attachment_node} conflicts with an existing node"
                        )
                        continue
                endpoint["node"] = attachment_node
            else:
                endpoint["reserved_for"] = item["id"]
    return errors


def attach_edge_endpoints(
    edges: list[dict[str, Any]],
    nodes: list[dict[str, Any]],
    germs: list[dict[str, Any]],
    mixed_node_ids: frozenset[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    binding_errors = apply_classification_bindings(edges, nodes)
    used_domains: dict[str, dict[str, Any]] = {}
    for edge in edges:
        for side in ("start", "end"):
            endpoint = edge["endpoints"][side]
            if endpoint.get("node") or endpoint.get("reserved_for"):
                continue
            masses = endpoint.get("masses")
            hit = domain_face_hit(masses)
            if hit:
                boundary = domain_exit_node_id(hit)
                endpoint["node"] = boundary
                endpoint["attachment"] = "declared_domain_boundary"
                endpoint["domain_face"] = hit["face"]
                endpoint["distance_to_domain_face"] = hit["distance_to_face"]
                record = used_domains.setdefault(
                    boundary,
                    {
                        "face": hit["face"],
                        "along_axis": hit["along_axis"],
                        "along_grid": hit["along_grid"],
                        "exits": [],
                    },
                )
                record["exits"].append(
                    {
                        "edge": edge["id"],
                        "side": side,
                        "masses": masses,
                        "distance_to_face": hit["distance_to_face"],
                    }
                )

    passed_nodes = {str(item["id"]) for item in nodes if item.get("passed")}
    candidates: list[tuple[float, str, str, int, str]] = []
    for edge in edges:
        for side in ("start", "end"):
            endpoint = edge["endpoints"][side]
            if endpoint.get("node") or endpoint.get("reserved_for"):
                continue
            for germ_index, germ in enumerate(germs):
                if not valid_germ(germ, mixed_node_ids):
                    continue
                node_id = str(germ.get("mixed_node"))
                if node_id not in passed_nodes or germ.get("event_mode") != edge.get("mechanism"):
                    continue
                distance = mass_distance(endpoint.get("masses"), germ.get("masses"))
                if distance <= GERM_ATTACH_DISTANCE:
                    candidates.append((distance, edge["id"], side, germ_index, node_id))
    used_endpoints: set[tuple[str, str]] = set()
    used_germs: set[int] = set()
    edge_by_id = {str(edge["id"]): edge for edge in edges}
    for distance, edge_id, side, germ_index, node_id in sorted(candidates):
        endpoint_key = (edge_id, side)
        if endpoint_key in used_endpoints or germ_index in used_germs:
            continue
        endpoint = edge_by_id[edge_id]["endpoints"][side]
        endpoint["node"] = node_id
        endpoint["attachment"] = "continuation_germ"
        endpoint["distance_to_germ"] = distance
        endpoint["germ_direction"] = germs[germ_index].get("direction")
        endpoint["binding_evidence"] = germs[germ_index].get("source_artifact")
        used_endpoints.add(endpoint_key)
        used_germs.add(germ_index)

    existing = {str(item["id"]) for item in nodes}
    for node_id in sorted(used_domains):
        if node_id not in existing:
            info = used_domains[node_id]
            nodes.append(
                node(
                    node_id,
                    "declared_domain_boundary",
                    status="frozen_domain",
                    masses=None,
                    passed=True,
                    evidence_level="definition",
                    domain_face=info["face"],
                    exit_coordinate={
                        "axis": info["along_axis"],
                        "grid_value": info["along_grid"],
                    },
                    observed_exits=info["exits"],
                )
            )
    unclassified: list[dict[str, Any]] = []
    for edge in edges:
        for side in ("start", "end"):
            endpoint = edge["endpoints"][side]
            if not endpoint.get("node"):
                unclassified.append(
                    {
                        "edge": edge["id"],
                        "side": side,
                        "masses": endpoint.get("masses"),
                        "reserved_for": endpoint.get("reserved_for"),
                    }
                )
        edge["endpoints"]["classified"] = all(
            edge["endpoints"][side].get("node") for side in ("start", "end")
        )
    return unclassified, binding_errors


def germ_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(record.get("mixed_node") or record.get("node") or ""),
        str(record.get("event_mode") or record.get("mode") or ""),
        str(record.get("direction") or ""),
    )


def collect_germs(paths: list[Path]) -> list[dict[str, Any]]:
    germs: list[dict[str, Any]] = []
    for path in paths:
        payload = load(path)
        rows = payload.get("germs") or payload.get("results") or ([payload] if payload.get("event_mode") or payload.get("mode") else [])
        for row in rows:
            if isinstance(row, dict):
                germ = dict(row)
                germ["source_artifact"] = str(path)
                germ["source_schema"] = payload.get("schema")
                germs.append(germ)
    return germs


def germ_trace_failed(record: dict[str, Any]) -> bool:
    """True when the germ's own artifact records a failed/nonconvergent trace.

    ``status`` is a label the producer writes; ``stopped_reason`` is what the
    continuation actually reported.  When the two disagree the reported failure
    wins.  research/evidence/V1_MIXED_GERMS_2026-08-15.json is exactly this
    case: mixed_principal_right / plus_one / +- both carry
    status="traced" alongside "pseudo-arclength correction failed: augmented
    least-squares failed: The maximum number of function evaluations is
    exceeded."  The underlying trace in
    research/evidence/V1_JUNCTION_PRINCIPAL_RIGHT_2026-08-15.json has zero
    continuation points -- the corrector failed on the first step -- so those
    two "germs" are just the two localized seed cells relabelled G+/G-.
    """
    text = str(record.get("stopped_reason") or "").lower()
    return any(token in text for token in GERM_FAILURE_TOKENS)


def germ_numbers(record: dict[str, Any]) -> tuple[float, float, float]:
    """(canonical_distance, |closure|, |event|); infinite when absent or unparsable."""
    try:
        canonical_distance = float(record.get("canonical_distance", float("inf")))
        closure = abs(float(record.get("closure", float("inf"))))
        event = abs(float(record.get("event", float("inf"))))
    except (TypeError, ValueError):
        return (float("inf"), float("inf"), float("inf"))
    return (canonical_distance, closure, event)


def germ_rejections(
    record: dict[str, Any], mixed_node_ids: frozenset[str]
) -> list[str]:
    """Every reason this germ record fails validation, in a stable order.

    The numeric and status checks apply UNIFORMLY.  Membership in
    BASE_MIXED_NODE_IDS used to short-circuit the canonical-binding and frozen
    gate checks; that exemption meant the three headline organizers could
    contribute germs carrying no closure, no event value, no canonical binding
    and an explicitly nonconvergent stopped_reason, and still be counted.  A
    base mixed node is a node like any other: it does not exempt a germ from
    having to show its numbers.
    """
    reasons: list[str] = []
    if record.get("status") not in {"traced", "verified", "passed"}:
        reasons.append("status")
    if germ_trace_failed(record):
        reasons.append("stopped_reason_records_nonconvergence")
    if record.get("mixed_node") not in mixed_node_ids:
        reasons.append("mixed_node")
    if record.get("event_mode") not in {"plus_one", "minus_one"}:
        reasons.append("event_mode")
    if record.get("direction") not in {"+", "-"}:
        reasons.append("direction")
    masses = record.get("masses")
    if not isinstance(masses, list) or len(masses) < 2:
        reasons.append("masses")
    if not record.get("source_artifact"):
        reasons.append("source_artifact")
    if record.get("canonical_bound") is not True:
        reasons.append("canonical_bound")
    if record.get("canonical_bracketed") is not True:
        reasons.append("canonical_bracketed")
    canonical_distance, closure, event = germ_numbers(record)
    if not canonical_distance <= GERM_ATTACH_DISTANCE:
        reasons.append("canonical_distance")
    if not closure <= CLOSURE_GATE:
        reasons.append("closure")
    if not event <= EVENT_GATE:
        reasons.append("event")
    return reasons


def valid_germ(record: dict[str, Any], mixed_node_ids: frozenset[str]) -> bool:
    return not germ_rejections(record, mixed_node_ids)


def retained_mixed_nodes(nodes: list[dict[str, Any]]) -> frozenset[str]:
    """Return every passed organizer, including endpoint classifications.

    A newly verified mixed endpoint is a retained mixed node just as much as
    the three headline organizers are.  It therefore cannot borrow their
    twelve germs: it must contribute its own G+/G- germs in both directions.
    """
    retained = set(BASE_MIXED_NODE_IDS)
    for item in nodes:
        if not item.get("passed"):
            continue
        if item.get("kind") == "mixed_organizer" or item.get("mechanism") in {
            "mixed_organizer",
            "mixed_plus_one_minus_one",
        }:
            retained.add(str(item["id"]))
    return frozenset(retained)


def missing_mixed_germs(
    germs: list[dict[str, Any]], mixed_node_ids: frozenset[str]
) -> list[str]:
    have = {germ_key(row) for row in germs if valid_germ(row, mixed_node_ids)}
    missing: list[str] = []
    for node_id in sorted(mixed_node_ids):
        for mode, direction in REQUIRED_GERM_KEYS:
            key = (node_id, mode, direction)
            if key in have:
                continue
            # A germ that immediately ends on another classified node may omit
            # the opposite unused direction only if that exact key is present
            # with ends_on set; absence is still missing.
            missing.append(f"{node_id}:{mode}:{direction}")
    return missing


def incidence_summary(
    edges: list[dict[str, Any]], nodes: list[dict[str, Any]]
) -> dict[str, Any]:
    """Report which nodes are actually shared, and how many pieces the graph has.

    This is diagnostic output only -- it feeds no release gate.  It exists so
    that a change in how termini are named (for example splitting one lumped
    declared-domain node into the distinct exits it was hiding) shows up as a
    visible change in incidence and component count instead of silently
    reshaping the topology.
    """
    node_to_edges: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        for side in ("start", "end"):
            attached = edge["endpoints"][side].get("node")
            if attached:
                node_to_edges[str(attached)].append(f"{edge['id']}:{side}")

    parent = {str(edge["id"]): str(edge["id"]) for edge in edges}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    for members in node_to_edges.values():
        edge_ids = sorted({name.split(":")[0] for name in members})
        for other in edge_ids[1:]:
            a, b = find(edge_ids[0]), find(other)
            if a != b:
                parent[max(a, b)] = min(a, b)

    components: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        components[find(str(edge["id"]))].append(str(edge["id"]))
    shared = {
        node_id: sorted(members)
        for node_id, members in sorted(node_to_edges.items())
        if len({name.split(":")[0] for name in members}) > 1
    }
    return {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "attached_endpoints": sum(len(members) for members in node_to_edges.values()),
        "nodes_touching_more_than_one_edge": shared,
        "edge_component_count": len(components),
        "edge_components": sorted(sorted(members) for members in components.values()),
    }


def completeness_verification(
    record: dict[str, Any] | None,
    *,
    root: Path,
    certificate_path: Path | None,
) -> dict[str, Any]:
    """Re-verify a completeness certificate against the artifacts it names.

    A self-consistent digest proves only that the record was not edited after
    sealing; it says nothing about whether an AL screen and a neck raster
    actually support the claim.  The verifier therefore re-reads every declared
    source, re-hashes it, and re-derives the AL and neck predicates.  A record
    that was re-sealed after a source file changed still fails, because the
    recomputed source digest no longer matches the one inside the record.
    """
    return verification_report(record, repo_root=root, certificate_path=certificate_path)


def valid_completeness_certificate(
    record: dict[str, Any] | None,
    *,
    root: Path,
    certificate_path: Path | None = None,
) -> bool:
    return bool(
        completeness_verification(record, root=root, certificate_path=certificate_path)["passed"]
    )


def sign_topology_report(paths: list[Path]) -> dict[str, Any]:
    """Require positive evidence that no critical curve is missing from the graph.

    Every other conjunct in release_ready asks whether the objects we ALREADY
    catalogued are sound.  None of them asks whether the catalogue is the whole
    catalogue, and on 2026-08-16 that gap became concrete: scripts/audit_sign_topology.py
    localized seven critical curves absent from the seven committed polylines,
    each passing the frozen gates, five with |event| below the census's own worst
    accepted root.

    The cause is structural.  The 620 cells come from transition_brackets, which
    brackets only where the PUBLISHED S/U label flips between adjacent baseline
    rows.  A critical curve interior to the unstable region flips no label and so
    produces no bracket at ANY grid resolution -- at all seven points n_unstable
    steps 2 -> 1, unstable on both sides.  A denser raster would have returned the
    same seven-edge graph, faster.  So "no missing curve" cannot be inferred from
    anything the census contains; it needs its own, independent check.

    This is FAIL-CLOSED on purpose.  With no audit supplied the answer is false,
    because the absence of a completeness audit is not evidence of completeness.
    That is the same reasoning that makes an unrun test worthless, and it is the
    error this project already made once with a certificate that hashed itself.
    """
    report: dict[str, Any] = {
        "passed": False,
        "audits": [],
        "errors": [],
        "missing_critical_curve": None,
        "forbidden_component_flip": None,
    }
    if not paths:
        report["errors"].append(
            "no sign-topology audit supplied: completeness of the critical set is "
            "unestablished, which is not the same as established"
        )
        return report

    missing_total = 0
    forbidden_total = 0
    for path in paths:
        if not path.is_file():
            report["errors"].append(f"sign-topology audit not readable: {path}")
            continue
        record = load(path)
        counts = record.get("violation_counts")
        if not isinstance(counts, dict):
            report["errors"].append(f"{path}: no violation_counts block")
            continue
        missing = int(counts.get("missing_critical_curve", 0))
        forbidden = int(counts.get("forbidden_component_flip", 0))
        missing_total += missing
        forbidden_total += forbidden
        report["audits"].append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "probes": record.get("probe_count") or record.get("probes"),
                "missing_critical_curve": missing,
                "forbidden_component_flip": forbidden,
            }
        )
        if missing:
            report["errors"].append(
                f"{path}: {missing} critical curve(s) localized outside the committed "
                "edges; the graph is not the complete critical set"
            )
        if forbidden:
            report["errors"].append(
                f"{path}: {forbidden} edge crossing(s) changed a state component the "
                "edge's mechanism label does not permit"
            )

    report["missing_critical_curve"] = missing_total
    report["forbidden_component_flip"] = forbidden_total
    report["passed"] = not report["errors"]
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="research/evidence/V1_CRITICAL_GRAPH.json")
    parser.add_argument("--roots")
    parser.add_argument("--left-birth")
    parser.add_argument("--right-death")
    parser.add_argument("--daughter")
    parser.add_argument("--germs", action="append", default=[])
    parser.add_argument("--completeness")
    parser.add_argument(
        "--sign-topology",
        action="append",
        default=[],
        help="sign-vector face-consistency audit(s); required for release_ready",
    )
    parser.add_argument("--al-screen")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]

    nodes = headline_nodes(root)
    nodes.append(
        load_classification(
            Path(args.left_birth) if args.left_birth else None,
            node_id="secondary_left_birth",
            default_kind="endpoint",
            allowed=LEFT_BIRTH_CLASSES,
            missing_note="Need event-specific G- geometry: projection fold with nonzero quadratic, or a recorded falsification.",
        )
    )
    nodes.append(
        load_classification(
            Path(args.right_death) if args.right_death else None,
            node_id="secondary_right_death",
            default_kind="endpoint",
            allowed=RIGHT_DEATH_CLASSES,
            missing_note="Allowed classes: mixed_organizer, projection_fold, domain_boundary. Newton-failed is forbidden.",
        )
    )
    daughter_node = load_classification(
        Path(args.daughter) if args.daughter else None,
        node_id="lower_plus_one_daughter",
        default_kind="branch",
        allowed=DAUGHTER_CLASSES,
        missing_note=(
            "Classify the lower +1 daughter as reconnecting, closed_loop, distinct_branch, "
            "obstruction, no_branch_attachment, or falsified."
        ),
    )
    nodes.append(daughter_node)
    daughter_status = {
        "required_for_v1_graph": True,
        "status": daughter_node["status"],
        "class": daughter_node.get("mechanism"),
    }

    roots: list[dict[str, Any]] = []
    if args.roots:
        payload = load(Path(args.roots))
        roots = [row for row in payload.get("roots", []) if row.get("status") == "ok" or row.get("passed") is True]

    germs = collect_germs([Path(path) for path in args.germs])
    mixed_node_ids = retained_mixed_nodes(nodes)
    missing_germs = missing_mixed_germs(germs, mixed_node_ids)

    completeness = None
    completeness_path = None
    if args.completeness:
        completeness_path = Path(args.completeness)
        completeness = load(completeness_path)
    elif args.al_screen:
        al = load(Path(args.al_screen))
        accepted = al.get("accepted_candidates", [])
        stable = [row for row in accepted if row.get("corrected", {}).get("screening_stable")]
        completeness = {
            "passed": False,
            "note": "AL pocket screen only; neck raster and vertex harvest still required",
            "attempted": len(al.get("attempted", [])),
            "accepted": len(accepted),
            "screening_stable": len(stable),
        }

    newton_failed = [
        item
        for item in roots
        if forbidden_class(item.get("status")) or forbidden_class(item.get("error"))
    ]
    edges = polyline_edges(roots) if roots else []
    unclassified_edge_endpoints, classification_binding_errors = attach_edge_endpoints(
        edges, nodes, germs, mixed_node_ids
    )
    completeness_report = completeness_verification(
        completeness, root=root, certificate_path=completeness_path
    )
    sign_report = sign_topology_report([Path(p) for p in args.sign_topology])
    cell_ids = [int(root["cell_id"]) for root in roots]
    assigned = [cell for edge in edges for cell in edge["cell_ids"]]
    duplicates = sorted({cell for cell in assigned if assigned.count(cell) > 1})
    coverage = {
        "source_transition_cells": 620,
        "localized_roots": len(roots),
        "required_cells": 620,
        "complete": len(roots) == 620 and set(cell_ids) == set(range(620)),
        "edge_count": len(edges),
        "cells_on_edges": len(assigned),
        "duplicate_cell_ids": duplicates,
        "missing_mixed_germs": missing_germs,
        "newton_failed": len(newton_failed),
        "sign_topology_clean": bool(sign_report["passed"]),
        "sign_topology_errors": sign_report["errors"],
        "sign_topology_audits": sign_report["audits"],
        "completeness_passed": bool(completeness_report["passed"]),
        "completeness_verification_errors": completeness_report["errors"],
        "full_critical_set_scope_passed": bool(
            completeness_report["release_scope_passed"]
        ),
        "completeness_scope_errors": completeness_report["release_scope_errors"],
        "completeness_sources_in_repository": completeness_report["sources_in_repository"],
        "unclassified_edge_endpoints": unclassified_edge_endpoints,
        "classification_binding_errors": classification_binding_errors,
        "edge_topology_complete": not unclassified_edge_endpoints
        and not classification_binding_errors,
    }

    missing_required = [
        node_id
        for node_id in REQUIRED_HEADLINE_IDS
        if not any(item["id"] == node_id and item.get("passed") for item in nodes)
    ]
    mandatory_unresolved = {
        item["id"] for item in nodes if item["status"] in {"unresolved", "illegal"}
    }
    unexplained = sorted(mandatory_unresolved)
    illegal = [item["id"] for item in nodes if item["status"] == "illegal"]
    organizer_count = len(mixed_node_ids)

    release_ready = (
        not missing_required
        and not unexplained
        and not illegal
        and coverage["complete"]
        and coverage["edge_count"] >= 1
        and coverage["cells_on_edges"] == 620
        and not duplicates
        and not missing_germs
        and not newton_failed
        and not unclassified_edge_endpoints
        and not classification_binding_errors
        and coverage["completeness_passed"]
        and coverage["full_critical_set_scope_passed"]
        and coverage["sign_topology_clean"]
        and all(item.get("passed") for item in nodes if item["id"] in REQUIRED_HEADLINE_IDS)
    )
    graph = {
        # /3: declared-domain termini are named per exit rather than per face,
        # and germ validation is uniform (no base-organizer exemption).  Both
        # change node identity, so the schema version moves with them.
        "schema": "atlas.v1.critical-graph/3",
        "claim_status": (
            "release_ready complete mechanism-resolved Floquet critical graph on the connected family sheet"
            if release_ready
            else "partial graph: 620 cells are samples, not edges; unresolved gates are enumerated by the artifact"
        ),
        "release_ready": release_ready,
        "family_component": "one continuation-connected Li-Li-Liao catalog sheet",
        "topology": {
            "free_group_word": "bABabaBAba",
            "role": "catalog topology metadata; topology is not used as family identity",
            "source": "research/OPEN_PROBLEM.md",
        },
        "declared_mass_domain": {
            "m1": list(DECLARED_DOMAIN["m1"]),
            "m2": list(DECLARED_DOMAIN["m2"]),
            "m3": 1.0,
        },
        "frozen_numerical_gates": {
            "maximum_absolute_event": 2e-8,
            "maximum_periodic_closure": 1e-7,
        },
        "source_transition_cells": 620,
        "localized_roots": len(roots),
        "nodes": nodes,
        "edges": edges,
        "mixed_germs": [
            {
                "mixed_node": row.get("mixed_node") or row.get("node"),
                "event_mode": row.get("event_mode") or row.get("mode"),
                "direction": row.get("direction"),
                "ends_on": row.get("ends_on"),
                "masses": row.get("masses"),
                "status": row.get("status"),
                "canonical_bound": row.get("canonical_bound"),
                "canonical_bracketed": row.get("canonical_bracketed"),
                "canonical_distance": row.get("canonical_distance"),
                "closure": row.get("closure"),
                "event": row.get("event"),
                "stopped_reason": row.get("stopped_reason"),
                "source_artifact": row.get("source_artifact"),
                "valid": valid_germ(row, mixed_node_ids),
                "invalid_reasons": germ_rejections(row, mixed_node_ids),
            }
            for row in germs
        ],
        "root_coverage": coverage,
        "root_residual_margin": root_residual_margin(roots),
        "incidence": incidence_summary(edges, nodes),
        "unexplained_nodes": unexplained,
        "missing_required_nodes": missing_required,
        "organizer_count": organizer_count,
        "daughter_classification": daughter_status,
        "completeness": completeness,
        "completeness_verification": completeness_report,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(graph, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(out),
                "release_ready": release_ready,
                "unexplained_nodes": unexplained,
                "localized_roots": coverage["localized_roots"],
                "edge_count": coverage["edge_count"],
                "missing_mixed_germs": coverage["missing_mixed_germs"],
                "completeness_passed": coverage["completeness_passed"],
                "completeness_verification_errors": coverage["completeness_verification_errors"],
                "full_critical_set_scope_passed": coverage[
                    "full_critical_set_scope_passed"
                ],
                "completeness_scope_errors": coverage["completeness_scope_errors"],
            },
            indent=2,
        )
    )
    if not release_ready:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
