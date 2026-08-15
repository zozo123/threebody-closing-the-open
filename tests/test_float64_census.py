from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_float64_census_has_every_cell_and_ok_roots_hold_gates() -> None:
    payload = json.loads(
        (ROOT / "research/evidence/V1_FLOAT64_CRITICAL_CENSUS_2026-08-15.json").read_text()
    )
    attempts = payload["attempts"]
    ids = [int(row["cell_id"]) for row in attempts]
    assert len(attempts) == 620
    assert set(ids) == set(range(620))
    assert payload["source_transition_cells"] == 620
    ok = [row for row in attempts if row.get("status") == "ok"]
    assert len(ok) == payload["localized_roots"]
    assert payload["localized_roots"] >= 400
    for row in ok:
        assert abs(float(row["event"])) <= 2e-8
        assert float(row["closure"]) <= 1e-7
    assert float(payload["max_abs_event"]) <= 2e-8
    assert float(payload["max_closure"]) <= 1e-7


def test_hybrid_merge_of_census_and_julia_hard_cells_holds_gates() -> None:
    import runpy
    import sys

    output = Path("/var/folders/4l/24cf6m8566bdsdbgvjm26r_r0000gn/T/grok-goal-e40eac9f893d/implementer/hybrid-test.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    census = ROOT / "research/evidence/V1_FLOAT64_CRITICAL_CENSUS_2026-08-15.json"
    miss = {
        int(row["cell_id"])
        for row in json.loads(census.read_text())["attempts"]
        if row.get("status") != "ok"
    }
    julia = []
    for path in sorted((ROOT / "research/evidence/julia_hard_cells_2026-08-15").glob("cell-*.json")):
        cell = int(json.loads(path.read_text())["cell_id"])
        if cell in miss:
            julia.extend(["--julia", str(path)])
    argv = sys.argv
    sys.argv = [
        "merge_hybrid_critical_roots.py",
        str(census),
        str(output),
        *julia,
    ]
    try:
        try:
            runpy.run_path(str(ROOT / "scripts/merge_hybrid_critical_roots.py"), run_name="__main__")
        except SystemExit as exc:
            assert exc.code == 2
    finally:
        sys.argv = argv
    hybrid = json.loads(output.read_text())
    assert hybrid["localized_roots"] >= 474
    assert float(hybrid["max_abs_event"]) <= 2e-8
    assert float(hybrid["max_closure"]) <= 1e-7
    assert hybrid["estimator_counts"]["julia_bigfloat"] >= 12
