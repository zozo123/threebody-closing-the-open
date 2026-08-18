from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GRAPH = ROOT / "research/evidence/V1_CRITICAL_GRAPH.json"


def test_assembler_never_emits_interior_lattice_terminus_nodes() -> None:
    """#204 forbids the synthetic class even in a non-release graph.

    A finite event-sign lattice end may remain unclassified.  It may not be
    manufactured into a passed node of kind interior_lattice_terminus.
    """
    graph = json.loads(GRAPH.read_text(encoding="utf-8"))
    synthetic = [
        node
        for node in graph.get("nodes", [])
        if node.get("kind") == "interior_lattice_terminus"
        or node.get("status") == "certified_on_event_sign_lattice"
    ]
    assert synthetic == []


def test_release_graph_cannot_promote_finite_sweep_ends_to_scientific_termini() -> None:
    """A finite event-sign lattice endpoint is candidate geometry, not a node.

    #192 requires continuation through projection turns and allows only a
    classified organizer/fold/domain boundary/closed loop/explicit
    contradiction as a scientific terminus.  A raster/component ending inside
    the declared domain can be scan clipping, pass-filter fragmentation, a
    missed turn, or an incompletely traced loop.  Therefore a release-ready
    graph may never contain the synthetic node class introduced by #201.

    This regression intentionally fails the current false-green graph.  It is
    satisfied only when the assembler keeps unmatched sampled ends unresolved
    or replaces them with continuous-witness bindings to legitimate termini.
    """
    graph = json.loads(GRAPH.read_text(encoding="utf-8"))
    synthetic = [
        node
        for node in graph.get("nodes", [])
        if node.get("kind") == "interior_lattice_terminus"
        or node.get("status") == "certified_on_event_sign_lattice"
    ]
    if graph.get("release_ready"):
        assert not synthetic, (
            "release_ready=true while finite event-sign sweep endpoints are "
            "manufactured into passed scientific termini: "
            + ", ".join(str(node.get("id")) for node in synthetic)
        )


def test_sampled_terminus_attachment_name_is_never_release_evidence() -> None:
    graph = json.loads(GRAPH.read_text(encoding="utf-8"))
    bad = []
    for edge in graph.get("edges", []):
        endpoints = edge.get("endpoints") or {}
        for side in ("start", "end"):
            endpoint = endpoints.get(side) or {}
            if endpoint.get("attachment") == "certified_interior_lattice_terminus":
                bad.append(f"{edge.get('id')}:{side}")
    if graph.get("release_ready"):
        assert not bad, (
            "release_ready=true with sampled-lattice endpoint attachments: "
            + ", ".join(bad)
        )


def test_an_existence_boundary_node_must_carry_its_measured_frontier() -> None:
    """The fourth stop class may not become the third way to fake a terminus.

    ``existence_boundary_terminus`` is the one node kind the assembler CREATES
    while ingesting a resolution, so it is the one most easily faked by editing
    the committed graph: a node with the right name and no numbers behind it would
    read exactly like an earned one.  Anything of that kind must therefore name
    the artifact it came from, carry the grid cell of the frontier that was
    measured, and be reachable from an edge endpoint that says how it bound.

    The committed graph has none, and that is the current truth: the two
    unclassified minus_one termini sit where the full-domain audit still closes
    periodic orbits, so the class refuses them.
    """
    graph = json.loads(GRAPH.read_text(encoding="utf-8"))
    frontier_nodes = [
        node
        for node in graph.get("nodes", [])
        if node.get("kind") == "existence_boundary_terminus"
    ]
    bound = {
        endpoint.get("node")
        for edge in graph.get("edges", [])
        for endpoint in (edge.get("endpoints") or {}).values()
        if isinstance(endpoint, dict)
        and endpoint.get("terminal_kind") == "existence_boundary_terminus"
    }
    for node in frontier_nodes:
        assert node.get("evidence"), f"{node.get('id')} names no resolution artifact"
        assert node.get("frontier_coordinate"), f"{node.get('id')} records no frontier"
        assert node.get("evidence_level") == "continuation", node.get("id")
        assert node.get("observed_frontiers"), f"{node.get('id')} lists no edge ends"
        assert node.get("id") in bound, (
            f"{node.get('id')} is not bound by any edge endpoint that reports "
            "terminal_kind=existence_boundary_terminus"
        )
