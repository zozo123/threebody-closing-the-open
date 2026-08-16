from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_extract_and_assemble_keeps_catalog_620(tmp_path) -> None:
    sweep = {
        "run_id": "test",
        "localizations": [
            {
                "status": "passed",
                "mechanism": "plus_one",
                "masses": [0.94, 0.90, 1.0],
                "m1": 0.94,
                "m2": 0.90,
                "event_value": 1e-10,
                "closure": 1e-12,
                "x1": 0.1,
                "v1": 0.2,
                "v2": 0.3,
                "period": 5.5,
            }
        ],
        "curve_components": [
            {
                "mechanism": "plus_one",
                "in_committed_graph": False,
                "partly_in_committed_graph": False,
                "vertices": [
                    {
                        "m1": 0.94,
                        "m2": 0.90,
                        "event_value": 1e-10,
                        "closure": 1e-12,
                        "committed_edge_matched": False,
                    },
                    {
                        "m1": 0.97,
                        "m2": 0.93,
                        "event_value": -2e-10,
                        "closure": 2e-12,
                        "committed_edge_matched": False,
                    },
                ],
            }
        ],
    }
    sweep["localizations"].append(
        {
            "status": "passed",
            "mechanism": "plus_one",
            "masses": [0.97, 0.93, 1.0],
            "m1": 0.97,
            "m2": 0.93,
            "event_value": -2e-10,
            "closure": 2e-12,
            "x1": 0.11,
            "v1": 0.21,
            "v2": 0.31,
            "period": 5.6,
        }
    )
    sweep_path = tmp_path / "sweep.json"
    sweep_path.write_text(json.dumps(sweep))
    out = tmp_path / "supp.json"
    argv = sys.argv
    sys.argv = ["extract_supplemental_sweep_roots.py", str(sweep_path), str(out)]
    try:
        try:
            runpy.run_path(str(ROOT / "scripts/extract_supplemental_sweep_roots.py"), run_name="__main__")
        except SystemExit as exc:
            assert exc.code in (0, None)
    finally:
        sys.argv = argv
    payload = json.loads(out.read_text())
    assert payload["localized_roots"] == 2
    assert all(int(row["cell_id"]) >= 10000 for row in payload["roots"])
    assert float(payload["max_abs_event"]) <= 2e-8
    assert float(payload["max_closure"]) <= 1e-7
    assert all(row.get("x1") is not None for row in payload["roots"])
    assert all(row.get("period") is not None for row in payload["roots"])


def test_assembler_counts_catalog_cells_not_supplemental(tmp_path) -> None:
    import runpy as rp

    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "roots": [
                    {
                        "cell_id": i,
                        "status": "ok",
                        "event_mode": "plus_one",
                        "orientation": "U->S",
                        "event": 1e-12,
                        "closure": 1e-12,
                        "masses": [0.8 + 0.001 * i, 0.75, 1.0],
                    }
                    for i in range(620)
                ]
            }
        )
    )
    supp = tmp_path / "supp.json"
    supp.write_text(
        json.dumps(
            {
                "roots": [
                    {
                        "cell_id": 10000,
                        "status": "ok",
                        "passed": True,
                        "event_mode": "minus_one",
                        "orientation": "sweep_component_0",
                        "sweep_component": 0,
                        "event": 1e-12,
                        "closure": 1e-12,
                        "masses": [0.94, 0.90, 1.0],
                    },
                    {
                        "cell_id": 10001,
                        "status": "ok",
                        "passed": True,
                        "event_mode": "minus_one",
                        "orientation": "sweep_component_0",
                        "sweep_component": 0,
                        "event": 1e-12,
                        "closure": 1e-12,
                        "masses": [0.97, 0.93, 1.0],
                    },
                ]
            }
        )
    )
    output = tmp_path / "graph.json"
    argv = sys.argv
    sys.argv = [
        "assemble_critical_graph.py",
        "--output",
        str(output),
        "--roots",
        str(catalog),
        "--supplemental-roots",
        str(supp),
    ]
    try:
        try:
            rp.run_path(str(ROOT / "scripts/assemble_critical_graph.py"), run_name="__main__")
        except SystemExit as exc:
            assert exc.code == 2
    finally:
        sys.argv = argv
    graph = json.loads(output.read_text())
    assert graph["localized_roots"] == 620
    assert graph["root_coverage"]["complete"] is True
    assert graph["root_coverage"]["cells_on_edges"] == 620
    assert graph["root_coverage"]["supplemental_roots"] == 2
    assert graph["root_coverage"]["all_vertices_on_edges"] == 622
    sweep_edges = [edge for edge in graph["edges"] if edge.get("source") == "full_domain_event_sign_sweep"]
    assert len(sweep_edges) == 1
    assert sweep_edges[0]["source_cell_count"] == 2


def test_combine_graph_roots_concatenates_without_inventing(tmp_path) -> None:
    import runpy as rp

    left = tmp_path / "a.json"
    right = tmp_path / "b.json"
    left.write_text(json.dumps({"roots": [{"cell_id": 0, "x1": 1.0}]}))
    right.write_text(json.dumps({"roots": [{"cell_id": 10000, "x1": 2.0}]}))
    out = tmp_path / "c.json"
    argv = sys.argv
    sys.argv = ["combine_graph_roots.py", str(out), str(left), str(right)]
    try:
        try:
            rp.run_path(str(ROOT / "scripts/combine_graph_roots.py"), run_name="__main__")
        except SystemExit as exc:
            assert exc.code in (0, None)
    finally:
        sys.argv = argv
    payload = json.loads(out.read_text())
    assert payload["n_roots"] == 2
    assert [row["cell_id"] for row in payload["roots"]] == [0, 10000]


def test_assembler_does_not_promote_a_single_lattice_hit_to_an_edge(tmp_path) -> None:
    import runpy as rp

    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"roots": []}))
    supp = tmp_path / "supp.json"
    supp.write_text(
        json.dumps(
            {
                "roots": [
                    {
                        "cell_id": 10000,
                        "status": "ok",
                        "passed": True,
                        "event_mode": "minus_one",
                        "orientation": "sweep_component_5",
                        "sweep_component": 5,
                        "event": 1e-12,
                        "closure": 1e-12,
                        "masses": [1.066, 0.842, 1.0],
                    }
                ]
            }
        )
    )
    output = tmp_path / "graph.json"
    argv = sys.argv
    sys.argv = [
        "assemble_critical_graph.py",
        "--output",
        str(output),
        "--roots",
        str(catalog),
        "--supplemental-roots",
        str(supp),
    ]
    try:
        try:
            rp.run_path(str(ROOT / "scripts/assemble_critical_graph.py"), run_name="__main__")
        except SystemExit as exc:
            assert exc.code == 2
    finally:
        sys.argv = argv
    graph = json.loads(output.read_text())
    assert graph["root_coverage"]["supplemental_roots"] == 1
    assert not any(edge.get("source") == "full_domain_event_sign_sweep" for edge in graph["edges"])
