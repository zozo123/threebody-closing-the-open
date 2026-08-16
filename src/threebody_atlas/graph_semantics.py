"""Canonical, sheet-aware mechanism multigraph comparison.

The comparator treats discovery order and local node/edge IDs as presentation
metadata.  Physical labels constrain matching before coordinates are considered,
and every ambiguous comparison fails closed instead of accepting a proximity-only
mapping.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "atlas.mechanism-multigraph.v1"
CANONICAL_SCHEMA_VERSION = "atlas.mechanism-multigraph-canonical.v1"
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/+|><=,() -]*\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class GraphSemanticsError(ValueError):
    """Raised when a graph is invalid or cannot be compared safely."""


class ComparisonLevel(StrEnum):
    TOPOLOGY_ONLY = "topology_only"
    MECHANISM_LABELED = "mechanism_labeled"
    ORIENTED = "oriented"
    SHEET_AWARE = "sheet_aware"
    EVIDENCE_EQUIVALENT = "evidence_equivalent"


_LEVEL_RANK = {
    ComparisonLevel.TOPOLOGY_ONLY: 0,
    ComparisonLevel.MECHANISM_LABELED: 1,
    ComparisonLevel.ORIENTED: 2,
    ComparisonLevel.SHEET_AWARE: 3,
    ComparisonLevel.EVIDENCE_EQUIVALENT: 4,
}


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class EvidenceIdentity(_StrictModel):
    status: str | None = None
    level: str | None = None
    passed: bool | None = None
    artifact_sha256s: tuple[str, ...] = ()
    payload_sha256: str | None = None

    @model_validator(mode="after")
    def validate_digests(self) -> EvidenceIdentity:
        for digest in (*self.artifact_sha256s, self.payload_sha256):
            if digest is not None and not _SHA256.fullmatch(digest):
                raise ValueError("evidence digests must be 64 lowercase hexadecimal digits")
        if tuple(sorted(set(self.artifact_sha256s))) != self.artifact_sha256s:
            raise ValueError("artifact_sha256s must be unique and sorted")
        return self


class BoundaryCoordinate(_StrictModel):
    axis: str
    value: str

    @model_validator(mode="after")
    def validate_coordinate(self) -> BoundaryCoordinate:
        _validate_identifier(self.axis, "boundary coordinate axis")
        _parse_coordinate(self.value)
        return self


class MechanismNode(_StrictModel):
    id: str
    physical_class: str
    mechanism: str | None = None
    sheet_id: str
    domain_face: str | None = None
    coordinates: tuple[str, ...] | None = None
    boundary_coordinate: BoundaryCoordinate | None = None
    local_sectors: tuple[str, ...] = ()
    evidence: EvidenceIdentity | None = None

    @model_validator(mode="after")
    def validate_node(self) -> MechanismNode:
        _validate_identifier(self.id, "node id")
        _validate_identifier(self.physical_class, "node physical_class")
        _validate_identifier(self.sheet_id, "node sheet_id")
        if self.mechanism is not None:
            _validate_identifier(self.mechanism, "node mechanism")
        if self.domain_face is not None:
            _validate_identifier(self.domain_face, "node domain_face")
        if self.coordinates is not None:
            for value in self.coordinates:
                _parse_coordinate(value)
        if tuple(sorted(set(self.local_sectors))) != self.local_sectors:
            raise ValueError("node local_sectors must be unique and sorted")
        return self


class MechanismEdge(_StrictModel):
    id: str
    source: str
    target: str
    kind: str
    mechanism: str
    orientation: str | None = None
    sheet_id: str
    source_sector: str | None = None
    target_sector: str | None = None
    multiplicity: int = Field(default=1, ge=1)
    evidence: EvidenceIdentity | None = None

    @model_validator(mode="after")
    def validate_edge(self) -> MechanismEdge:
        for label, value in (
            ("edge id", self.id),
            ("edge source", self.source),
            ("edge target", self.target),
            ("edge kind", self.kind),
            ("edge mechanism", self.mechanism),
            ("edge sheet_id", self.sheet_id),
        ):
            _validate_identifier(value, label)
        for label, value in (
            ("edge orientation", self.orientation),
            ("edge source_sector", self.source_sector),
            ("edge target_sector", self.target_sector),
        ):
            if value is not None:
                _validate_identifier(value, label)
        return self


class MechanismGraph(_StrictModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    coordinate_system: str
    coordinate_axes: tuple[str, ...]
    declared_domain: dict[str, tuple[str, str]]
    nodes: tuple[MechanismNode, ...]
    edges: tuple[MechanismEdge, ...]

    @model_validator(mode="after")
    def validate_graph(self) -> MechanismGraph:
        _validate_identifier(self.coordinate_system, "coordinate_system")
        if not self.coordinate_axes:
            raise ValueError("coordinate_axes must not be empty")
        if len(set(self.coordinate_axes)) != len(self.coordinate_axes):
            raise ValueError("coordinate_axes must be unique")
        for axis in self.coordinate_axes:
            _validate_identifier(axis, "coordinate axis")
        if set(self.declared_domain) != set(self.coordinate_axes):
            raise ValueError("declared_domain must define every coordinate axis exactly once")
        for axis, bounds in self.declared_domain.items():
            lower, upper = map(_parse_coordinate, bounds)
            if lower > upper:
                raise ValueError(f"declared domain for {axis!r} has reversed bounds")

        node_ids = [node.id for node in self.nodes]
        edge_ids = [edge.id for edge in self.edges]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("node ids must be unique")
        if len(edge_ids) != len(set(edge_ids)):
            raise ValueError("edge ids must be unique")
        if not self.nodes:
            raise ValueError("graph must contain at least one node")

        nodes = {node.id: node for node in self.nodes}
        for node in self.nodes:
            if node.coordinates is not None and len(node.coordinates) != len(self.coordinate_axes):
                raise ValueError(
                    f"node {node.id!r} has {len(node.coordinates)} coordinates; "
                    f"expected {len(self.coordinate_axes)}"
                )
            if node.domain_face is not None and node.boundary_coordinate is None:
                raise ValueError(f"domain-boundary node {node.id!r} needs a boundary_coordinate")
        for edge in self.edges:
            if edge.source not in nodes or edge.target not in nodes:
                raise ValueError(f"edge {edge.id!r} references an unknown endpoint")
        return self


class GraphDifference(_StrictModel):
    kind: Literal[
        "added_edge",
        "removed_edge",
        "changed_endpoint",
        "changed_mechanism",
        "orientation_flip",
        "sheet_reassignment",
        "node_split",
        "node_merge",
        "changed_node_class",
        "coordinate_shift",
        "evidence_changed",
    ]
    left: str | None = None
    right: str | None = None
    detail: str


class GraphComparison(_StrictModel):
    level: ComparisonLevel
    equivalent: bool
    coordinate_tolerance: str
    mapping: dict[str, str]
    differences: tuple[GraphDifference, ...]
    left_canonical_sha256: str
    right_canonical_sha256: str


def _validate_identifier(value: str, label: str) -> None:
    if not value or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{label} has invalid syntax: {value!r}")


def _parse_coordinate(value: str) -> Decimal:
    if not isinstance(value, str):
        raise ValueError("coordinates must be decimal strings")
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"invalid coordinate {value!r}") from exc
    if not parsed.is_finite():
        raise ValueError("coordinates must be finite")
    if _canonical_decimal(parsed) != value:
        raise ValueError(f"coordinate is not canonical decimal syntax: {value!r}")
    return parsed


def _canonical_decimal(value: Decimal | str | int | float) -> str:
    parsed = value if isinstance(value, Decimal) else Decimal(str(value))
    if not parsed.is_finite():
        raise GraphSemanticsError("graph coordinates must be finite")
    if parsed == 0:
        return "0"
    rendered = format(parsed, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _evidence_payload(evidence: EvidenceIdentity | None) -> tuple[Any, ...]:
    if evidence is None:
        return ()
    return (
        evidence.status or "",
        evidence.level or "",
        evidence.passed,
        evidence.artifact_sha256s,
        evidence.payload_sha256 or "",
    )


def _node_payload(
    node: MechanismNode,
    level: ComparisonLevel,
    *,
    include_coordinates: bool,
) -> tuple[Any, ...]:
    rank = _LEVEL_RANK[level]
    payload: list[Any] = []
    if rank >= 1:
        payload.extend((node.physical_class, node.mechanism or ""))
    if rank >= 3:
        payload.extend(
            (
                node.sheet_id,
                node.domain_face or "",
                (
                    (node.boundary_coordinate.axis, node.boundary_coordinate.value)
                    if node.boundary_coordinate is not None
                    else ()
                ),
                node.local_sectors,
            )
        )
        if include_coordinates:
            payload.append(node.coordinates or ())
    if rank >= 4:
        payload.append(_evidence_payload(node.evidence))
    return tuple(payload)


def _edge_payload(edge: MechanismEdge, level: ComparisonLevel) -> tuple[Any, ...]:
    rank = _LEVEL_RANK[level]
    payload: list[Any] = []
    if rank >= 1:
        payload.extend((edge.kind, edge.mechanism))
    if rank >= 2:
        payload.append(edge.orientation or "")
    if rank >= 3:
        payload.extend(
            (
                edge.sheet_id,
                edge.source_sector or "",
                edge.target_sector or "",
            )
        )
    if rank >= 4:
        payload.append(_evidence_payload(edge.evidence))
    return tuple(payload)


def _directed(level: ComparisonLevel) -> bool:
    return _LEVEL_RANK[level] >= _LEVEL_RANK[ComparisonLevel.ORIENTED]


def _node_index(graph: MechanismGraph) -> dict[str, MechanismNode]:
    return {node.id: node for node in graph.nodes}


def _incident_signature(
    graph: MechanismGraph,
    node_id: str,
    level: ComparisonLevel,
) -> tuple[Any, ...]:
    items: list[tuple[Any, ...]] = []
    directed = _directed(level)
    for edge in graph.edges:
        label = _edge_payload(edge, level)
        if edge.source == node_id:
            tag = "loop" if edge.target == node_id else ("out" if directed else "incident")
            items.append((tag, label, edge.multiplicity))
        if edge.target == node_id and edge.source != node_id:
            tag = "in" if directed else "incident"
            items.append((tag, label, edge.multiplicity))
    return tuple(sorted(items, key=repr))


def _coordinates_match(
    left: MechanismNode,
    right: MechanismNode,
    tolerance: Decimal,
) -> bool:
    if (left.coordinates is None) != (right.coordinates is None):
        return False
    if left.coordinates is not None and right.coordinates is not None:
        for left_value, right_value in zip(left.coordinates, right.coordinates, strict=True):
            if abs(_parse_coordinate(left_value) - _parse_coordinate(right_value)) > tolerance:
                return False
    if (left.boundary_coordinate is None) != (right.boundary_coordinate is None):
        return False
    if left.boundary_coordinate is not None and right.boundary_coordinate is not None:
        if left.boundary_coordinate.axis != right.boundary_coordinate.axis:
            return False
        if (
            abs(
                _parse_coordinate(left.boundary_coordinate.value)
                - _parse_coordinate(right.boundary_coordinate.value)
            )
            > tolerance
        ):
            return False
    return True


def _candidate_nodes(
    left: MechanismGraph,
    right: MechanismGraph,
    level: ComparisonLevel,
    tolerance: Decimal,
) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    for left_node in left.nodes:
        candidates = []
        for right_node in right.nodes:
            if _node_payload(left_node, level, include_coordinates=False) != _node_payload(
                right_node,
                level,
                include_coordinates=False,
            ):
                continue
            if _LEVEL_RANK[level] >= 3 and not _coordinates_match(
                left_node,
                right_node,
                tolerance,
            ):
                continue
            if _incident_signature(left, left_node.id, level) != _incident_signature(
                right,
                right_node.id,
                level,
            ):
                continue
            candidates.append(right_node.id)
        result[left_node.id] = tuple(sorted(candidates))
    return result


def _edge_bundle(
    graph: MechanismGraph,
    first: str,
    second: str,
    level: ComparisonLevel,
) -> Counter[tuple[Any, ...]]:
    bundle: Counter[tuple[Any, ...]] = Counter()
    directed = _directed(level)
    for edge in graph.edges:
        if directed:
            if edge.source == first and edge.target == second:
                bundle[_edge_payload(edge, level)] += edge.multiplicity
        elif {edge.source, edge.target} == {first, second}:
            if first == second and edge.source != edge.target:
                continue
            bundle[_edge_payload(edge, level)] += edge.multiplicity
    return bundle


def find_isomorphism(
    left: MechanismGraph,
    right: MechanismGraph,
    level: ComparisonLevel | str = ComparisonLevel.SHEET_AWARE,
    coordinate_tolerance: Decimal | str = Decimal("0"),
) -> dict[str, str] | None:
    """Return a deterministic valid mapping, or ``None`` when graphs differ."""

    level = ComparisonLevel(level)
    tolerance = Decimal(str(coordinate_tolerance))
    if not tolerance.is_finite() or tolerance < 0:
        raise GraphSemanticsError("coordinate tolerance must be finite and nonnegative")
    if len(left.nodes) != len(right.nodes):
        return None
    if sum(edge.multiplicity for edge in left.edges) != sum(
        edge.multiplicity for edge in right.edges
    ):
        return None
    if _LEVEL_RANK[level] >= _LEVEL_RANK[ComparisonLevel.SHEET_AWARE] and (
        left.coordinate_system != right.coordinate_system
        or left.coordinate_axes != right.coordinate_axes
        or left.declared_domain != right.declared_domain
    ):
        return None

    candidates = _candidate_nodes(left, right, level, tolerance)
    if any(not options for options in candidates.values()):
        return None

    mapping: dict[str, str] = {}
    used: set[str] = set()

    def compatible(left_id: str, right_id: str) -> bool:
        if _edge_bundle(left, left_id, left_id, level) != _edge_bundle(
            right,
            right_id,
            right_id,
            level,
        ):
            return False
        for other_left, other_right in mapping.items():
            if _edge_bundle(left, left_id, other_left, level) != _edge_bundle(
                right,
                right_id,
                other_right,
                level,
            ):
                return False
            if _directed(level) and _edge_bundle(
                left,
                other_left,
                left_id,
                level,
            ) != _edge_bundle(right, other_right, right_id, level):
                return False
        return True

    def search() -> bool:
        if len(mapping) == len(left.nodes):
            return True
        remaining = [node.id for node in left.nodes if node.id not in mapping]
        left_id = min(
            remaining,
            key=lambda item: (
                sum(candidate not in used for candidate in candidates[item]),
                repr(_node_payload(_node_index(left)[item], level, include_coordinates=True)),
                item,
            ),
        )
        for right_id in candidates[left_id]:
            if right_id in used or not compatible(left_id, right_id):
                continue
            mapping[left_id] = right_id
            used.add(right_id)
            if search():
                return True
            used.remove(right_id)
            del mapping[left_id]
        return False

    return dict(sorted(mapping.items())) if search() else None


def _refined_colors(
    graph: MechanismGraph,
    level: ComparisonLevel,
) -> dict[str, int]:
    node_lookup = _node_index(graph)
    labels = {
        node.id: _node_payload(node, level, include_coordinates=True) for node in graph.nodes
    }
    unique = {label: index for index, label in enumerate(sorted(set(labels.values()), key=repr))}
    colors = {node_id: unique[label] for node_id, label in labels.items()}

    def partition(coloring: dict[str, int]) -> tuple[tuple[str, ...], ...]:
        return tuple(
            sorted(
                (
                    tuple(sorted(node_id for node_id, value in coloring.items() if value == color))
                    for color in set(coloring.values())
                )
            )
        )

    while True:
        signatures: dict[str, tuple[Any, ...]] = {}
        for node_id in node_lookup:
            incident: list[tuple[Any, ...]] = []
            for edge in graph.edges:
                label = _edge_payload(edge, level)
                if edge.source == node_id:
                    direction = "out" if _directed(level) else "incident"
                    incident.append((direction, label, colors[edge.target], edge.multiplicity))
                if edge.target == node_id and edge.source != node_id:
                    direction = "in" if _directed(level) else "incident"
                    incident.append((direction, label, colors[edge.source], edge.multiplicity))
            signatures[node_id] = (labels[node_id], tuple(sorted(incident, key=repr)))
        palette = {
            signature: index
            for index, signature in enumerate(sorted(set(signatures.values()), key=repr))
        }
        updated = {node_id: palette[signature] for node_id, signature in signatures.items()}
        if partition(updated) == partition(colors):
            return updated
        colors = updated


def _semantic_node_json(node: MechanismNode, level: ComparisonLevel) -> dict[str, Any]:
    rank = _LEVEL_RANK[level]
    payload: dict[str, Any] = {}
    if rank >= 1:
        payload.update(physical_class=node.physical_class, mechanism=node.mechanism)
    if rank >= 3:
        payload.update(
            sheet_id=node.sheet_id,
            domain_face=node.domain_face,
            coordinates=list(node.coordinates) if node.coordinates is not None else None,
            boundary_coordinate=(
                node.boundary_coordinate.model_dump(mode="json")
                if node.boundary_coordinate is not None
                else None
            ),
            local_sectors=list(node.local_sectors),
        )
    if rank >= 4:
        payload["evidence"] = (
            node.evidence.model_dump(mode="json") if node.evidence is not None else None
        )
    return payload


def _semantic_edge_json(
    edge: MechanismEdge,
    level: ComparisonLevel,
    node_numbers: dict[str, int],
) -> dict[str, Any]:
    source = node_numbers[edge.source]
    target = node_numbers[edge.target]
    if not _directed(level) and source > target:
        source, target = target, source
    rank = _LEVEL_RANK[level]
    payload: dict[str, Any] = {
        "source": source,
        "target": target,
        "multiplicity": edge.multiplicity,
    }
    if rank >= 1:
        payload.update(kind=edge.kind, mechanism=edge.mechanism)
    if rank >= 2:
        payload["orientation"] = edge.orientation
    if rank >= 3:
        payload.update(
            sheet_id=edge.sheet_id,
            source_sector=edge.source_sector,
            target_sector=edge.target_sector,
        )
    if rank >= 4:
        payload["evidence"] = (
            edge.evidence.model_dump(mode="json") if edge.evidence is not None else None
        )
    return payload


def canonical_form(
    graph: MechanismGraph,
    level: ComparisonLevel | str = ComparisonLevel.SHEET_AWARE,
    *,
    max_permutations: int = 2_000_000,
) -> dict[str, Any]:
    """Return an ID/order-independent exact canonical form.

    Equal final color classes are searched exactly.  The bounded search is a
    deliberate fail-closed guard against pathological symmetric inputs.
    """

    level = ComparisonLevel(level)
    colors = _refined_colors(graph, level)
    groups: list[tuple[str, ...]] = []
    for color in sorted(set(colors.values())):
        group = tuple(node.id for node in graph.nodes if colors[node.id] == color)
        groups.append(group)
    search_size = math.prod(math.factorial(len(group)) for group in groups)
    if search_size > max_permutations:
        raise GraphSemanticsError(
            f"canonical labeling needs {search_size} permutations, above the "
            f"fail-closed limit {max_permutations}"
        )

    node_lookup = _node_index(graph)
    best_serialized: str | None = None
    best_payload: dict[str, Any] | None = None
    permutation_groups: Iterable[tuple[tuple[str, ...], ...]] = itertools.product(
        *(itertools.permutations(group) for group in groups)
    )
    for selected_groups in permutation_groups:
        ordered_ids = tuple(itertools.chain.from_iterable(selected_groups))
        node_numbers = {node_id: index for index, node_id in enumerate(ordered_ids)}
        payload: dict[str, Any] = {
            "schema_version": CANONICAL_SCHEMA_VERSION,
            "comparison_level": level.value,
            "nodes": [_semantic_node_json(node_lookup[node_id], level) for node_id in ordered_ids],
            "edges": sorted(
                (_semantic_edge_json(edge, level, node_numbers) for edge in graph.edges),
                key=_canonical_json,
            ),
        }
        if _LEVEL_RANK[level] >= 3:
            payload.update(
                coordinate_system=graph.coordinate_system,
                coordinate_axes=list(graph.coordinate_axes),
                declared_domain={
                    axis: list(graph.declared_domain[axis]) for axis in sorted(graph.declared_domain)
                },
            )
        serialized = _canonical_json(payload)
        if best_serialized is None or serialized < best_serialized:
            best_serialized = serialized
            best_payload = payload
    if best_payload is None:
        raise GraphSemanticsError("canonical labeling produced no candidate")
    return best_payload


def canonical_sha256(
    graph: MechanismGraph,
    level: ComparisonLevel | str = ComparisonLevel.SHEET_AWARE,
) -> str:
    return _sha256_json(canonical_form(graph, level))


def _edge_total(graph: MechanismGraph) -> int:
    return sum(edge.multiplicity for edge in graph.edges)


def _differences_with_mapping(
    left: MechanismGraph,
    right: MechanismGraph,
    mapping: dict[str, str],
    tolerance: Decimal,
) -> list[GraphDifference]:
    differences: list[GraphDifference] = []
    left_nodes = _node_index(left)
    right_nodes = _node_index(right)
    for left_id, right_id in mapping.items():
        left_node = left_nodes[left_id]
        right_node = right_nodes[right_id]
        if (left_node.physical_class, left_node.mechanism) != (
            right_node.physical_class,
            right_node.mechanism,
        ):
            kind = (
                "changed_mechanism"
                if left_node.physical_class == right_node.physical_class
                else "changed_node_class"
            )
            differences.append(
                GraphDifference(
                    kind=kind,
                    left=left_id,
                    right=right_id,
                    detail="node physical class or local mechanism changed",
                )
            )
        if (left_node.sheet_id, left_node.domain_face) != (
            right_node.sheet_id,
            right_node.domain_face,
        ):
            differences.append(
                GraphDifference(
                    kind="sheet_reassignment",
                    left=left_id,
                    right=right_id,
                    detail="node sheet identity or boundary face changed",
                )
            )
        if not _coordinates_match(left_node, right_node, tolerance):
            differences.append(
                GraphDifference(
                    kind="coordinate_shift",
                    left=left_id,
                    right=right_id,
                    detail=f"node coordinates differ by more than tolerance {tolerance}",
                )
            )
        if _evidence_payload(left_node.evidence) != _evidence_payload(right_node.evidence):
            differences.append(
                GraphDifference(
                    kind="evidence_changed",
                    left=left_id,
                    right=right_id,
                    detail="node evidence identity changed",
                )
            )

    right_to_left = {right_id: left_id for left_id, right_id in mapping.items()}
    left_groups: dict[tuple[str, str], list[MechanismEdge]] = defaultdict(list)
    right_groups: dict[tuple[str, str], list[MechanismEdge]] = defaultdict(list)
    for edge in left.edges:
        endpoints = tuple(sorted((edge.source, edge.target)))
        left_groups[endpoints].append(edge)
    for edge in right.edges:
        if edge.source not in right_to_left or edge.target not in right_to_left:
            continue
        endpoints = tuple(sorted((right_to_left[edge.source], right_to_left[edge.target])))
        right_groups[endpoints].append(edge)

    for endpoints in sorted(set(left_groups) | set(right_groups)):
        left_edges = sorted(left_groups[endpoints], key=lambda edge: edge.id)
        right_edges = sorted(right_groups[endpoints], key=lambda edge: edge.id)
        if sum(edge.multiplicity for edge in left_edges) < sum(
            edge.multiplicity for edge in right_edges
        ):
            differences.append(
                GraphDifference(
                    kind="added_edge",
                    detail=f"edge multiplicity increased between {endpoints}",
                )
            )
        elif sum(edge.multiplicity for edge in left_edges) > sum(
            edge.multiplicity for edge in right_edges
        ):
            differences.append(
                GraphDifference(
                    kind="removed_edge",
                    detail=f"edge multiplicity decreased between {endpoints}",
                )
            )
        for left_edge, right_edge in zip(left_edges, right_edges, strict=False):
            if (left_edge.kind, left_edge.mechanism) != (
                right_edge.kind,
                right_edge.mechanism,
            ):
                differences.append(
                    GraphDifference(
                        kind="changed_mechanism",
                        left=left_edge.id,
                        right=right_edge.id,
                        detail="edge kind or active mechanism changed",
                    )
                )
            if (
                left_edge.orientation,
                left_edge.source == endpoints[0],
            ) != (
                right_edge.orientation,
                right_to_left[right_edge.source] == endpoints[0],
            ):
                differences.append(
                    GraphDifference(
                        kind="orientation_flip",
                        left=left_edge.id,
                        right=right_edge.id,
                        detail="edge orientation label or endpoint direction flipped",
                    )
                )
            if left_edge.sheet_id != right_edge.sheet_id:
                differences.append(
                    GraphDifference(
                        kind="sheet_reassignment",
                        left=left_edge.id,
                        right=right_edge.id,
                        detail="edge sheet identity changed",
                    )
                )
            if _evidence_payload(left_edge.evidence) != _evidence_payload(right_edge.evidence):
                differences.append(
                    GraphDifference(
                        kind="evidence_changed",
                        left=left_edge.id,
                        right=right_edge.id,
                        detail="edge evidence identity changed",
                    )
                )
    return differences


def compare_graphs(
    left: MechanismGraph,
    right: MechanismGraph,
    level: ComparisonLevel | str = ComparisonLevel.SHEET_AWARE,
    coordinate_tolerance: Decimal | str = Decimal("0"),
) -> GraphComparison:
    level = ComparisonLevel(level)
    tolerance = Decimal(str(coordinate_tolerance))
    mapping = find_isomorphism(left, right, level, tolerance)
    left_hash = canonical_sha256(left, level)
    right_hash = canonical_sha256(right, level)
    if mapping is not None:
        return GraphComparison(
            level=level,
            equivalent=True,
            coordinate_tolerance=_canonical_decimal(tolerance),
            mapping=mapping,
            differences=(),
            left_canonical_sha256=left_hash,
            right_canonical_sha256=right_hash,
        )

    topology_mapping = find_isomorphism(
        left,
        right,
        ComparisonLevel.TOPOLOGY_ONLY,
        Decimal("0"),
    )
    differences: list[GraphDifference] = []
    if topology_mapping is None:
        if len(left.nodes) < len(right.nodes):
            differences.append(
                GraphDifference(
                    kind="node_split",
                    detail=f"node count increased from {len(left.nodes)} to {len(right.nodes)}",
                )
            )
        elif len(left.nodes) > len(right.nodes):
            differences.append(
                GraphDifference(
                    kind="node_merge",
                    detail=f"node count decreased from {len(left.nodes)} to {len(right.nodes)}",
                )
            )
        if _edge_total(left) < _edge_total(right):
            differences.append(
                GraphDifference(
                    kind="added_edge",
                    detail=f"edge multiplicity increased from {_edge_total(left)} to {_edge_total(right)}",
                )
            )
        elif _edge_total(left) > _edge_total(right):
            differences.append(
                GraphDifference(
                    kind="removed_edge",
                    detail=f"edge multiplicity decreased from {_edge_total(left)} to {_edge_total(right)}",
                )
            )
        if len(left.nodes) == len(right.nodes) and _edge_total(left) == _edge_total(right):
            differences.append(
                GraphDifference(
                    kind="changed_endpoint",
                    detail="edge incidence changed while node and edge counts stayed fixed",
                )
            )
    else:
        differences.extend(_differences_with_mapping(left, right, topology_mapping, tolerance))
    if not differences:
        differences.append(
            GraphDifference(
                kind="changed_endpoint",
                detail="graphs are not equivalent at the requested semantic level",
            )
        )
    differences = sorted(
        differences,
        key=lambda item: (item.kind, item.left or "", item.right or "", item.detail),
    )
    return GraphComparison(
        level=level,
        equivalent=False,
        coordinate_tolerance=_canonical_decimal(tolerance),
        mapping=topology_mapping or {},
        differences=tuple(differences),
        left_canonical_sha256=left_hash,
        right_canonical_sha256=right_hash,
    )


def _artifact_digest(path: Any, root: Path | None, *, label: str) -> str | None:
    """Hash a referenced evidence file.

    A missing path is allowed and means "no supporting artifact". A non-null
    path must resolve to a readable file under ``root``; otherwise evidence
    equivalence would silently compare two unverifiable reconstructions.
    """
    if path is None:
        return None
    if not isinstance(path, str) or not path:
        raise GraphSemanticsError(f"{label} must be a non-empty path")
    if root is None:
        raise GraphSemanticsError(
            f"{label} {path!r} cannot be hashed without a repository root"
        )
    root = root.resolve()
    candidate = (root / path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise GraphSemanticsError(f"{label} artifact escapes the repository: {path}") from exc
    if not candidate.is_file():
        raise GraphSemanticsError(f"{label} artifact is missing or unreadable: {path}")
    return hashlib.sha256(candidate.read_bytes()).hexdigest()


def _endpoint_sector(endpoint: dict[str, Any]) -> str | None:
    parts = []
    for key in ("attachment", "germ_direction", "binding_class", "domain_face"):
        if endpoint.get(key) is not None:
            parts.append(f"{key}:{endpoint[key]}")
    return "|".join(parts) or None


def _node_sectors(node: dict[str, Any]) -> tuple[str, ...]:
    sectors = []
    for binding in node.get("edge_endpoint_bindings", []):
        sectors.append(
            "|".join(
                f"{key}:{binding[key]}"
                for key in ("side", "mechanism", "orientation")
                if binding.get(key) is not None
            )
        )
    return tuple(sorted(set(sectors)))


def adapt_v1_critical_graph(
    document: dict[str, Any],
    *,
    repository_root: Path | None = None,
) -> MechanismGraph:
    """Adapt the shipped ``atlas.v1.critical-graph/3`` into frozen semantics."""

    if document.get("schema") != "atlas.v1.critical-graph/3":
        raise GraphSemanticsError("expected atlas.v1.critical-graph/3 input")
    sheet_id = str(document.get("family_component") or "")
    if not sheet_id:
        raise GraphSemanticsError("critical graph has no family_component/sheet identity")

    nodes = []
    for raw in document.get("nodes", []):
        masses = raw.get("masses")
        coordinates = (
            tuple(_canonical_decimal(value) for value in masses) if masses is not None else None
        )
        exit_coordinate = raw.get("exit_coordinate")
        boundary_coordinate = None
        if exit_coordinate is not None:
            boundary_coordinate = BoundaryCoordinate(
                axis=str(exit_coordinate["axis"]),
                value=_canonical_decimal(exit_coordinate["grid_value"]),
            )
        node_id = str(raw.get("id") or "<missing-id>")
        artifact = _artifact_digest(
            raw.get("evidence"),
            repository_root,
            label=f"node {node_id} evidence",
        )
        evidence = EvidenceIdentity(
            status=str(raw["status"]) if raw.get("status") is not None else None,
            level=str(raw["evidence_level"]) if raw.get("evidence_level") is not None else None,
            passed=bool(raw["passed"]) if raw.get("passed") is not None else None,
            artifact_sha256s=tuple(sorted({artifact} if artifact else set())),
            payload_sha256=_sha256_json(
                {
                    key: raw[key]
                    for key in ("screening_passed", "estimator")
                    if raw.get(key) is not None
                }
            ),
        )
        nodes.append(
            MechanismNode(
                id=str(raw["id"]),
                physical_class=str(raw["kind"]),
                mechanism=str(raw["mechanism"]) if raw.get("mechanism") is not None else None,
                sheet_id=sheet_id,
                domain_face=(
                    str(raw["domain_face"]) if raw.get("domain_face") is not None else None
                ),
                coordinates=coordinates,
                boundary_coordinate=boundary_coordinate,
                local_sectors=_node_sectors(raw),
                evidence=evidence,
            )
        )

    edges = []
    for raw in document.get("edges", []):
        endpoint_document = raw.get("endpoints") or {}
        start = endpoint_document.get("start") or {}
        end = endpoint_document.get("end") or {}
        edge_id = str(raw.get("id") or "<missing-id>")
        artifacts = {
            digest
            for digest in (
                _artifact_digest(
                    start.get("binding_evidence"),
                    repository_root,
                    label=f"edge {edge_id} start binding_evidence",
                ),
                _artifact_digest(
                    end.get("binding_evidence"),
                    repository_root,
                    label=f"edge {edge_id} end binding_evidence",
                ),
            )
            if digest is not None
        }
        evidence_payload = {
            "cell_ids": raw.get("cell_ids", []),
            "estimators": raw.get("estimators", []),
            "source_cell_count": raw.get("source_cell_count"),
            "uncertainty": raw.get("uncertainty", {}),
        }
        edges.append(
            MechanismEdge(
                id=str(raw["id"]),
                source=str(start["node"]),
                target=str(end["node"]),
                kind=str(raw["kind"]),
                mechanism=str(raw["mechanism"]),
                orientation=(
                    str(raw["orientation"]) if raw.get("orientation") is not None else None
                ),
                sheet_id=sheet_id,
                source_sector=_endpoint_sector(start),
                target_sector=_endpoint_sector(end),
                multiplicity=1,
                evidence=EvidenceIdentity(
                    artifact_sha256s=tuple(sorted(artifacts)),
                    payload_sha256=_sha256_json(evidence_payload),
                ),
            )
        )

    domain = document.get("declared_mass_domain") or {}
    declared_domain = {
        "m1": tuple(_canonical_decimal(value) for value in domain["m1"]),
        "m2": tuple(_canonical_decimal(value) for value in domain["m2"]),
        "m3": (_canonical_decimal(domain["m3"]), _canonical_decimal(domain["m3"])),
    }
    return MechanismGraph(
        coordinate_system="atlas_mass_chart_m3_fixed",
        coordinate_axes=("m1", "m2", "m3"),
        declared_domain=declared_domain,
        nodes=tuple(nodes),
        edges=tuple(edges),
    )


def load_graph(path: str | Path, *, repository_root: Path | None = None) -> MechanismGraph:
    path = Path(path)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GraphSemanticsError(f"cannot read graph {path}: {exc}") from exc
    if document.get("schema") == "atlas.v1.critical-graph/3":
        return adapt_v1_critical_graph(document, repository_root=repository_root)
    try:
        # JSON arrays naturally arrive as lists; the in-memory model freezes them
        # as tuples after the field and semantic validators have run.
        return MechanismGraph.model_validate(document, strict=False)
    except ValueError as exc:
        raise GraphSemanticsError(str(exc)) from exc
