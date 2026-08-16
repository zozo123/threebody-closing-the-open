from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GRAPH = ROOT / "research/evidence/V1_CRITICAL_GRAPH.json"


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
