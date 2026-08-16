# V1 canonical mechanism-multigraph semantics

Status: frozen v1

Schema: `atlas.mechanism-multigraph.v1`

Normative implementation: `src/threebody_atlas/graph_semantics.py`

## 1. Purpose

This contract defines when two reconstructions of the v1 mechanism network are the
same scientific graph. It replaces comparisons based on screenshots, discovery
order, local IDs, sampling density, or coordinate proximity alone.

The object is a finite labeled multigraph. Parallel edges and explicit multiplicity
are semantic. A node may be isolated. An edge is undirected at the two lowest
comparison levels and directed from `source` to `target` at the oriented and stricter
levels.

## 2. Invariants and presentation metadata

The following fields are presentation metadata and never affect equivalence:

- node `id`;
- edge `id`;
- input array order;
- discovery/run order;
- JSON indentation and object-member order.

The following fields are semantic, subject to the selected comparison level:

- node physical class and local mechanism;
- node sheet identity;
- node mass coordinates, declared coordinate system, and mass domain;
- domain-boundary face and coordinate along that face;
- organizer/fold local-sector labels;
- edge kind, active mechanism, endpoint incidence, direction, and orientation;
- edge sheet and endpoint-sector identities;
- edge multiplicity;
- content identities and states of supporting evidence.

Sampling lists, artifact paths, estimator prose, and uncertainty presentation are
not copied directly into topology. The adapter hashes them into an evidence payload
identity. Thus they are excluded below `evidence_equivalent` but cannot change
silently at the strictest level.

## 3. Canonical schema

`V1_GRAPH_CANONICAL_SCHEMA.json` is the machine-readable schema. A graph contains:

```json
{
  "schema_version": "atlas.mechanism-multigraph.v1",
  "coordinate_system": "atlas_mass_chart_m3_fixed",
  "coordinate_axes": ["m1", "m2", "m3"],
  "declared_domain": {
    "m1": ["0.8", "1.1"],
    "m2": ["0.7", "1.2"],
    "m3": ["1", "1"]
  },
  "nodes": [],
  "edges": []
}
```

Coordinates are finite canonical decimal strings. Scientific matching never parses
them through binary64. Every axis has an explicit closed domain. Unknown fields,
duplicate node IDs, duplicate edge IDs, missing endpoints, noncanonical coordinates,
reversed domains, and unsorted duplicate sector/evidence identities are errors.

### 3.1 Nodes

A node has:

- `id`: local reference only;
- `physical_class`: organizer, boundary, event representative, branch, or another
  frozen physical class;
- `mechanism`: local physical mechanism, or null when none is asserted;
- `sheet_id`: orbit/sheet identity, never inferred from nearby masses;
- `domain_face`: named boundary face, or null;
- `coordinates`: one value per coordinate axis, or null;
- `boundary_coordinate`: axis and value along a declared face, or null;
- `local_sectors`: sorted unique organizer/fold sectors;
- `evidence`: optional content identity and evidence state.

A domain-face node must include a boundary coordinate. Multiple nodes may share the
same coordinates if their physical class, sheet, or incidence differs.

### 3.2 Edges

An edge has:

- `id`: local reference only;
- `source` and `target` node IDs;
- `kind` and active `mechanism`;
- `orientation`, such as `U->S`;
- `sheet_id`;
- optional `source_sector` and `target_sector`;
- positive integer `multiplicity`;
- optional evidence identity.

Parallel edges remain distinct, and multiplicity is never reduced to Boolean
adjacency. Cross-sheet incidence is legal only when the two endpoint sheet IDs and
the edge sheet are written explicitly; it cannot be manufactured by coordinate
matching.

## 4. Comparison levels

Levels are nested from weakest to strongest.

### 4.1 `topology_only`

Compares unlabeled undirected multigraph incidence and multiplicity. Node/edge IDs,
classes, mechanisms, direction, sheets, coordinates, and evidence are ignored.

This level answers only whether the abstract multigraphs are isomorphic. It is not
sufficient for a scientific release comparison.

### 4.2 `mechanism_labeled`

Adds node physical class/local mechanism and edge kind/active mechanism. Edges remain
undirected. Sheet identity and coordinates remain excluded.

### 4.3 `oriented`

Adds ordered source/target incidence and the orientation label. Swapping endpoints
or changing `U->S` to `S->U` is a semantic change.

### 4.4 `sheet_aware`

Adds:

- node and edge sheet identities;
- declared coordinate system, axes, and domain;
- domain-face identity and along-face coordinate;
- node coordinates;
- node local sectors and edge endpoint sectors.

This is the default scientific comparison level.

### 4.5 `evidence_equivalent`

Adds evidence state, level, pass bit, content digests, and hashed supporting payloads.
Two graphs may be sheet-aware equivalent while differing at this level because one
uses stale or different evidence.

## 5. Coordinate tolerance

Tolerance applies only at `sheet_aware` and `evidence_equivalent`. Candidate nodes
must first agree on all applicable physical labels, sheet identity, boundary face,
local sectors, and incidence signatures. Only then may each coordinate differ by at
most the declared absolute decimal tolerance.

Consequences:

1. close nodes on different sheets never match;
2. close `+1` and `-1` mechanisms never match;
3. a boundary node on `domain_m1_min` never matches one on `domain_m2_max`;
4. tolerance cannot repair a changed endpoint or orientation;
5. zero tolerance means exact canonical decimal equality.

The domain itself is exact frozen metadata and is not tolerance-matched.

## 6. Isomorphism algorithm

The implementation:

1. validates both graphs;
2. forms node candidates from semantic labels and incidence signatures;
3. applies decimal coordinate tolerance only to candidates that survived step 2;
4. backtracks over the smallest candidate set first;
5. checks every already-mapped edge bundle, direction, label, and multiplicity;
6. accepts only a complete bijection.

No hand mapping is required. Returned mappings use input IDs only to explain the
result; those IDs never decide equivalence.

## 7. Canonical ordering and identity

Canonicalization first performs iterative labeled-neighborhood color refinement.
Any unresolved color classes are searched exactly, and the lexicographically least
semantic graph is selected after IDs have been replaced by canonical node numbers.
Edges are sorted by their canonical semantic JSON.

The search has a fail-closed permutation bound. A graph exceeding the bound raises
an error instead of falling back to input order. This is deliberate: an
order-dependent digest is forbidden. The v1 shipped graph and every mutation audit
case are below the bound.

Canonical SHA-256 is the digest of compact, key-sorted UTF-8 JSON for the selected
comparison level. Exact canonical hashes need not be equal when a nonzero coordinate
tolerance makes two graphs semantically equivalent; the comparison result and its
explicit tolerance are authoritative in that case.

## 8. Minimal difference classes

Failed comparisons return structured differences:

- `added_edge`;
- `removed_edge`;
- `changed_endpoint`;
- `changed_mechanism`;
- `orientation_flip`;
- `sheet_reassignment`;
- `node_split`;
- `node_merge`;
- `changed_node_class`;
- `coordinate_shift`;
- `evidence_changed`.

The tool first seeks a topology-only mapping. If one exists, it reports semantic
changes against that mapping. If topology itself differs, node/edge counts identify
add/remove/split/merge changes and equal-count rewiring is reported as a changed
endpoint. Differences are deterministic and sorted.

## 9. Shipped graph adapter

`adapt_v1_critical_graph` converts `atlas.v1.critical-graph/3` without changing the
source artifact:

- `family_component` becomes the explicit sheet identity;
- node `kind`, `mechanism`, masses, boundary face, exit coordinate, and endpoint
  bindings become semantic node fields;
- each polyline becomes one mechanism edge with its frozen endpoint binding;
- evidence files are addressed by SHA-256 when present;
- cell IDs, estimators, counts, and uncertainties are hashed into an evidence payload
  identity;
- edge endpoint attachment/germ/boundary class becomes local-sector metadata.

Paths and prose do not enter topology. Content and state enter only the strictest
comparison level.

## 10. Required use and stop rules

N-version reconstruction, portability, systematics, and distributed reducers should
emit or adapt this schema and compare at least `sheet_aware`. Screenshots and nearest
coordinate joins are not evidence of graph equivalence.

Stop the release if:

1. comparison changes under input reordering or ID renaming;
2. canonical labeling exceeds its fail-closed bound;
3. different sheets or mechanisms match through proximity;
4. an orientation, endpoint, multiplicity, domain face, or sector change is hidden;
5. evidence-equivalent comparison encounters different content identities;
6. a reducer needs a hand-authored node mapping.

`V1_GRAPH_ISOMORPHISM_MUTATION_AUDIT_2026-08-16.json` records executable kills for
every required graph-difference class plus ID/order invariance, coordinate tolerance,
and evidence-level separation.
