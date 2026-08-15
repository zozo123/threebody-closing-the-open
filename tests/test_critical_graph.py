from __future__ import annotations

import json
from pathlib import Path

from threebody_atlas.critical_manifold import classify_localized_cell


ROOT = Path(__file__).resolve().parents[1]


def test_headline_canonical_records_passed() -> None:
    for name in (
        "V1_CANONICAL_LOWER_PLUS_ONE_2026-08-15.json",
        "V1_CANONICAL_UPPER_COLLISION_2026-08-15.json",
        "V1_MIXED_CANONICAL_PRINCIPAL_LEFT_2026-08-15.json",
        "V1_MIXED_CANONICAL_SECONDARY_LEFT_2026-08-15.json",
        "V1_MIXED_CANONICAL_PRINCIPAL_RIGHT_2026-08-15.json",
    ):
        payload = json.loads((ROOT / "research/evidence" / name).read_text())
        assert payload["passed"] is True


def test_assemble_critical_graph_stays_unready_without_endpoints(tmp_path, capsys) -> None:
    import runpy
    import sys

    output = tmp_path / "graph.json"
    argv = sys.argv
    sys.argv = ["assemble_critical_graph.py", "--output", str(output)]
    try:
        try:
            runpy.run_path(str(ROOT / "scripts/assemble_critical_graph.py"), run_name="__main__")
        except SystemExit as exc:
            assert exc.code == 2
    finally:
        sys.argv = argv
    graph = json.loads(output.read_text())
    assert graph["release_ready"] is False
    assert graph["edges"] == []
    assert "secondary_right_death" in graph["unexplained_nodes"]
    assert "secondary_left_fold" in graph["unexplained_nodes"]
    assert "lower_plus_one_daughter" in graph["unexplained_nodes"]


def test_event_gate_is_unchanged() -> None:
    assert classify_localized_cell(closure=1e-10, event=1.9e-8, m2=0.75, lo=0.75, hi=0.751) == "ok"
    assert classify_localized_cell(closure=1e-10, event=2.1e-8, m2=0.75, lo=0.75, hi=0.751) == "missed_event"


def test_julia_hard_canary_harvest_holds_frozen_gates() -> None:
    payload = json.loads((ROOT / "research/evidence/V1_JULIA_HARD_CANARY_2026-08-15.json").read_text())
    assert payload["localized_cells"] == 12
    assert payload["max_abs_event"] <= 2e-8
    assert payload["max_closure"] <= 1e-7
    assert payload["pending_cells"] == [0, 30, 50, 619]
    ids = []
    for row in payload["cells"]:
        assert row["passed"] is True
        assert abs(float(row["event_value"])) <= 2e-8
        assert float(row["closure_norm"]) <= 1e-7
        ids.append(int(row["cell_id"]))
    assert ids == sorted(ids)
    assert len(set(ids)) == 12


def test_hard_canary_seed_file_covers_failed_cells() -> None:
    import csv

    path = ROOT / "experiments/hard_canary_cells.tsv"
    rows = list(csv.DictReader(path.open(encoding="utf-8"), delimiter="\t"))
    ids = [int(row["cell_id"]) for row in rows]
    assert ids == [0, 1, 2, 3, 4, 5, 10, 15, 20, 30, 50, 100, 148, 200, 610, 619]


def test_localize_cli_refuses_to_loosen_gates() -> None:
    import subprocess
    import sys

    script = ROOT / "scripts/localize_full_critical_network.py"
    loosened = subprocess.run(
        [
            sys.executable,
            str(script),
            "missing.tsv",
            "out.json",
            "--event-tolerance",
            "1e-6",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert loosened.returncode != 0
    blob = loosened.stdout + loosened.stderr
    assert "2e-8" in blob

    closure = subprocess.run(
        [
            sys.executable,
            str(script),
            "missing.tsv",
            "out.json",
            "--max-closure",
            "1e-5",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert closure.returncode != 0
    assert "1e-7" in (closure.stdout + closure.stderr)


def test_hybrid_merger_refuses_to_loosen_event_gate(tmp_path) -> None:
    import runpy
    import sys

    python_roots = tmp_path / "python.json"
    julia_cell = tmp_path / "julia.json"
    output = tmp_path / "hybrid.json"
    python_roots.write_text(
        json.dumps(
            {
                "roots": [
                    {
                        "cell_id": 0,
                        "status": "ok",
                        "event_mode": "plus_one",
                        "event": 1e-10,
                        "closure": 1e-10,
                        "masses": [0.8, 0.75, 1.0],
                    }
                ]
            }
        )
    )
    julia_cell.write_text(
        json.dumps(
            {
                "cell_id": 1,
                "event_mode": "plus_one",
                "event_value": "3e-8",
                "closure_norm": "1e-12",
                "passed": True,
                "m1": "0.8",
                "m2": "0.76",
                "m3": "1.0",
            }
        )
    )
    argv = sys.argv
    sys.argv = [
        "merge_hybrid_critical_roots.py",
        str(python_roots),
        str(output),
        "--julia",
        str(julia_cell),
    ]
    try:
        try:
            runpy.run_path(str(ROOT / "scripts/merge_hybrid_critical_roots.py"), run_name="__main__")
            raise AssertionError("merger must reject a 3e-8 Julia event")
        except SystemExit as exc:
            assert exc.code not in (0, None)
    finally:
        sys.argv = argv
    if output.exists():
        hybrid = json.loads(output.read_text())
        assert all(int(root["cell_id"]) != 1 for root in hybrid["roots"])


def test_assembler_emits_edges_but_stays_unready_with_partial_roots(tmp_path) -> None:
    import runpy
    import sys

    roots = tmp_path / "roots.json"
    roots.write_text(
        json.dumps(
            {
                "roots": [
                    {
                        "cell_id": 0,
                        "status": "ok",
                        "event_mode": "plus_one",
                        "orientation": "U->S",
                        "estimator": "float64",
                        "event": 1e-10,
                        "closure": 1e-10,
                        "masses": [0.8, 0.755, 1.0],
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
        str(roots),
    ]
    try:
        try:
            runpy.run_path(str(ROOT / "scripts/assemble_critical_graph.py"), run_name="__main__")
        except SystemExit as exc:
            assert exc.code == 2
    finally:
        sys.argv = argv
    graph = json.loads(output.read_text())
    assert graph["release_ready"] is False
    assert len(graph["edges"]) == 1
    assert graph["edges"][0]["mechanism"] == "plus_one"
    assert graph["unexplained_nodes"]
