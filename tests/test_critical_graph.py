from __future__ import annotations

import importlib.util
import json
import hashlib
import os
from pathlib import Path

from threebody_atlas.critical_manifold import classify_localized_cell


ROOT = Path(__file__).resolve().parents[1]
REAL_ROOTS = ROOT / "research/evidence/V1_HYBRID_CRITICAL_ROOTS_2026-08-15.json"
REAL_GERMS = ROOT / "research/evidence/V1_MIXED_GERMS_2026-08-15.json"


def _assembler():
    """Import the assembler as a real module so its constants can be poked.

    Poking a constant is how the sensitivity tests below demonstrate what a
    future widening would silently do.  Nothing here mutates the shipped file.
    """
    spec = importlib.util.spec_from_file_location(
        "assemble_critical_graph", ROOT / "scripts/assemble_critical_graph.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _real_roots() -> list[dict]:
    payload = json.loads(REAL_ROOTS.read_text())
    return [
        row
        for row in payload["roots"]
        if row.get("status") == "ok" or row.get("passed") is True
    ]


def _numeric_germ_fixture(target: Path) -> Path:
    """Re-emit the real germ masses in the numerics-complete germ schema.

    THIS IS A TEST FIXTURE, NEVER EVIDENCE.  It exists so the *geometric*
    sensitivity of GERM_ATTACH_DISTANCE -- a property of the mass coordinates
    alone -- can be pinned independently of whether the underlying pseudo-
    arclength traces converged.  It is written to tmp_path only.  The real
    artifact is and stays rejected; see
    test_real_mixed_germ_artifact_is_rejected_by_uniform_validation.
    """
    real = json.loads(REAL_GERMS.read_text())
    target.write_text(
        json.dumps(
            {
                "schema": "atlas.v1.mixed-germs/1",
                "claim_status": "TEST FIXTURE, not evidence",
                "germs": [
                    {
                        **{
                            key: value
                            for key, value in row.items()
                            if key != "stopped_reason"
                        },
                        "canonical_bound": True,
                        "canonical_bracketed": True,
                        "canonical_distance": 0.0,
                        "closure": 1e-12,
                        "event": 1e-12,
                    }
                    for row in real["germs"]
                ],
            }
        )
    )
    return target


def _secondary_right_germ_fixture(target: Path) -> Path:
    target.write_text(
        json.dumps(
            {
                "schema": "atlas.v1.mixed-germs/1",
                "claim_status": "TEST FIXTURE, not evidence",
                "germs": [
                    {
                        "mixed_node": "secondary_right_death",
                        "event_mode": mode,
                        "direction": direction,
                        "status": "traced",
                        "masses": [1.0426, 1.0460, 1.0],
                        "canonical_bound": True,
                        "canonical_bracketed": True,
                        "canonical_distance": 0.0,
                        "closure": 1e-10,
                        "event": 1e-10,
                    }
                    for mode in ("plus_one", "minus_one")
                    for direction in ("+", "-")
                ],
            }
        )
    )
    return target


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
    assert graph["source_transition_cells"] == 620
    assert graph["localized_roots"] == 0
    assert "secondary_right_death" in graph["unexplained_nodes"]
    assert "secondary_left_birth" in graph["unexplained_nodes"]
    assert "lower_plus_one_daughter" in graph["unexplained_nodes"]
    assert graph["daughter_classification"]["status"] == "unresolved"
    assert graph["daughter_classification"]["required_for_v1_graph"] is True


def test_event_gate_is_unchanged() -> None:
    assert classify_localized_cell(closure=1e-10, event=1.9e-8, m2=0.75, lo=0.75, hi=0.751) == "ok"
    assert classify_localized_cell(closure=1e-10, event=2.1e-8, m2=0.75, lo=0.75, hi=0.751) == "missed_event"


def test_julia_hard_canary_harvest_holds_frozen_gates() -> None:
    payload = json.loads((ROOT / "research/evidence/V1_JULIA_HARD_CANARY_2026-08-15.json").read_text())
    assert payload["localized_cells"] >= 12
    assert payload["max_abs_event"] <= 2e-8
    assert payload["max_closure"] <= 1e-7
    assert set(payload["pending_cells"]) <= {0, 30, 50, 619}
    ids = []
    for row in payload["cells"]:
        assert row["passed"] is True
        assert abs(float(row["event_value"])) <= 2e-8
        assert float(row["closure_norm"]) <= 1e-7
        ids.append(int(row["cell_id"]))
    assert ids == sorted(ids)
    assert len(set(ids)) == payload["localized_cells"]
    assert payload["localized_cells"] + len(payload["pending_cells"]) == 16


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


def test_julia_merge_keeps_raw_strings_and_run_provenance() -> None:
    import runpy

    namespace = runpy.run_path(str(ROOT / "scripts/merge_hybrid_critical_roots.py"))
    merged = namespace["from_julia"](
        {
            "cell_id": 7,
            "event_mode": "minus_one",
            "event_value": "1.234567890123456789e-12",
            "closure_norm": "9.876543210987654321e-20",
            "passed": True,
            "m1": "0.8",
            "m2": "0.7555",
            "m3": "1.0",
        },
        {
            "status": "missed_event",
            "event_mode": "minus_one",
            "source_m2_bracket": [0.755, 0.756],
        },
        provenance={"source_run": 12345, "source_sha": "abc123"},
    )
    assert merged["raw_bigfloat"]["event_value"] == "1.234567890123456789e-12"
    assert merged["raw_bigfloat"]["closure_norm"] == "9.876543210987654321e-20"
    assert merged["source_run"] == 12345
    assert merged["source_sha"] == "abc123"


def _run_assembler(
    tmp_path, extra_args: list[str], *, expected_ready: bool = False
) -> dict:
    import runpy
    import sys

    output = tmp_path / "graph.json"
    argv = sys.argv
    sys.argv = ["assemble_critical_graph.py", "--output", str(output), *extra_args]
    exit_code = 0
    try:
        try:
            runpy.run_path(str(ROOT / "scripts/assemble_critical_graph.py"), run_name="__main__")
        except SystemExit as exc:
            exit_code = int(exc.code or 0)
    finally:
        sys.argv = argv
    assert exit_code == (0 if expected_ready else 2)
    return json.loads(output.read_text())


AL_SCREEN = ROOT / "research/evidence/V1_AL_POCKET_SCREEN_2026-08-15.json"


def _write_clean_neck_scan(path: Path) -> dict:
    """A neck raster that legitimately supports a bounded completeness claim."""
    neck = {
        "completed": True,
        "grid": {"m1": [0.997, 0.999], "m2": [0.993, 1.006], "step": 0.0001, "samples": 12},
        "minimum_resolved_unstable_gap": 0.0002,
        "any_vertical_merge": False,
        # Post-integration a legitimate raster must also have decided the merge
        # question inside its own window, not merely failed to observe a merge.
        "any_boundary_truncated_merge_test": False,
        "any_line_without_stable_sample": False,
        "any_stable_interval_touches_boundary": False,
        "all_lines_separated": True,
        "merge_verdict_counts": {
            "separated": 1,
            "interior_merge": 0,
            "truncation_undecidable": 0,
            "no_stable_sample": 0,
        },
        "boundary_truncated_lines": [],
        "max_shooting_residual": 1e-9,
        "line_summaries": [
            {
                "m1": 0.997,
                "stable_intervals": [[0.994, 0.996], [0.998, 1.0]],
                "interior_unstable_gaps": [0.0019],
            }
        ],
    }
    path.write_text(json.dumps(neck, indent=2) + "\n")
    return neck


def _freeze_certificate(tmp_path, *, neck_path: Path | None = None) -> Path:
    """Produce a genuine certificate with scripts/freeze_completeness_certificate.py."""
    import runpy
    import sys

    if neck_path is None:
        neck_path = tmp_path / "neck.json"
        _write_clean_neck_scan(neck_path)
    certificate = tmp_path / "completeness.json"
    argv = sys.argv
    sys.argv = [
        "freeze_completeness_certificate.py",
        str(certificate),
        "--al-screen",
        str(AL_SCREEN),
        "--neck-scan",
        str(neck_path),
    ]
    exit_code = 0
    try:
        try:
            runpy.run_path(
                str(ROOT / "scripts/freeze_completeness_certificate.py"), run_name="__main__"
            )
        except SystemExit as exc:
            exit_code = int(exc.code or 0)
    finally:
        sys.argv = argv
    assert exit_code == 0, "the freezer must accept a clean AL screen plus a clean neck raster"
    return certificate


def _reseal(path: Path, record: dict) -> None:
    """Re-seal a certificate after editing it, exactly as a forger would."""
    body = {key: value for key, value in record.items() if key != "sha256_content"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    body["sha256_content"] = hashlib.sha256(canonical.encode()).hexdigest()
    path.write_text(json.dumps(body, indent=2) + "\n")


def _completeness_state(tmp_path, certificate: Path) -> dict:
    graph = _run_assembler(tmp_path, ["--completeness", str(certificate)])
    return graph["root_coverage"]


def test_hand_written_self_sealed_certificate_is_rejected(tmp_path) -> None:
    """The two-key forgery that used to satisfy the completeness gate."""
    certificate = tmp_path / "forged.json"
    for schema in (
        "atlas.v1.completeness-certificate/1",
        "atlas.v1.completeness-certificate/2",
    ):
        _reseal(certificate, {"schema": schema, "passed": True})
        coverage = _completeness_state(tmp_path, certificate)
        assert coverage["completeness_passed"] is False
        assert coverage["completeness_verification_errors"]


def test_certificate_without_neck_scan_source_is_rejected(tmp_path) -> None:
    certificate = _freeze_certificate(tmp_path)
    record = json.loads(certificate.read_text())
    record["sources"] = [row for row in record["sources"] if row["role"] != "neck_scan"]
    _reseal(certificate, record)
    coverage = _completeness_state(tmp_path, certificate)
    assert coverage["completeness_passed"] is False
    assert any("neck_scan" in error for error in coverage["completeness_verification_errors"])


def test_certificate_fails_when_a_source_changes_after_sealing(tmp_path) -> None:
    neck_path = tmp_path / "neck.json"
    neck = _write_clean_neck_scan(neck_path)
    certificate = _freeze_certificate(tmp_path, neck_path=neck_path)
    assert _completeness_state(tmp_path, certificate)["completeness_passed"] is True

    neck["minimum_resolved_unstable_gap"] = 0.0005
    neck_path.write_text(json.dumps(neck, indent=2) + "\n")
    coverage = _completeness_state(tmp_path, certificate)
    assert coverage["completeness_passed"] is False
    assert any("sha256 mismatch" in error for error in coverage["completeness_verification_errors"])


def test_resealed_certificate_over_a_modified_source_still_fails(tmp_path) -> None:
    """Re-hashing the source and re-sealing the record does not launder it."""
    neck_path = tmp_path / "neck.json"
    neck = _write_clean_neck_scan(neck_path)
    certificate = _freeze_certificate(tmp_path, neck_path=neck_path)
    record = json.loads(certificate.read_text())

    # (a) source edited, its digest refreshed, record re-sealed, but the
    # certificate's own recorded numbers now disagree with the artifact.
    neck["minimum_resolved_unstable_gap"] = 0.0005
    neck_path.write_text(json.dumps(neck, indent=2) + "\n")
    for row in record["sources"]:
        if row["role"] == "neck_scan":
            row["sha256"] = hashlib.sha256(neck_path.read_bytes()).hexdigest()
    _reseal(certificate, record)
    coverage = _completeness_state(tmp_path, certificate)
    assert coverage["completeness_passed"] is False
    assert any(
        "minimum_resolved_unstable_gap" in error
        for error in coverage["completeness_verification_errors"]
    )

    # (b) a fully self-consistent re-seal over a neck raster that merges
    # vertically: the re-derived predicate, not the self-report, decides.
    neck["minimum_resolved_unstable_gap"] = 0.0002
    neck["any_vertical_merge"] = True
    neck_path.write_text(json.dumps(neck, indent=2) + "\n")
    record = json.loads(certificate.read_text())
    record["neck"]["any_vertical_merge"] = True
    record["neck"]["minimum_resolved_unstable_gap"] = 0.0002
    for row in record["sources"]:
        if row["role"] == "neck_scan":
            row["sha256"] = hashlib.sha256(neck_path.read_bytes()).hexdigest()
    _reseal(certificate, record)
    coverage = _completeness_state(tmp_path, certificate)
    assert coverage["completeness_passed"] is False
    assert any(
        "neck raster does not support" in error
        for error in coverage["completeness_verification_errors"]
    )


def test_certificate_source_paths_may_not_escape_the_allowed_roots(tmp_path) -> None:
    outside = tmp_path.parent / "outside_neck.json"
    _write_clean_neck_scan(outside)
    certificate = _freeze_certificate(tmp_path)
    record = json.loads(certificate.read_text())
    for row in record["sources"]:
        if row["role"] == "neck_scan":
            row["path"] = str(outside)
            row["sha256"] = hashlib.sha256(outside.read_bytes()).hexdigest()
    _reseal(certificate, record)
    coverage = _completeness_state(tmp_path, certificate)
    assert coverage["completeness_passed"] is False

    record["sources"] = [
        {**row, "path": "../" + row["path"]} if row["role"] == "neck_scan" else row
        for row in record["sources"]
    ]
    _reseal(certificate, record)
    coverage = _completeness_state(tmp_path, certificate)
    assert coverage["completeness_passed"] is False
    assert any("'..'" in error for error in coverage["completeness_verification_errors"])


def test_frozen_certificate_from_the_freezer_is_accepted(tmp_path) -> None:
    certificate = _freeze_certificate(tmp_path)
    record = json.loads(certificate.read_text())
    assert record["schema"] == "atlas.v1.completeness-certificate/2"
    assert record["passed"] is True
    assert {row["role"] for row in record["sources"]} == {"active_learning", "neck_scan"}
    coverage = _completeness_state(tmp_path, certificate)
    assert coverage["completeness_passed"] is True
    assert coverage["completeness_verification_errors"] == []


def test_assembler_emits_edges_but_stays_unready_with_partial_roots(tmp_path) -> None:
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
    graph = _run_assembler(tmp_path, ["--roots", str(roots)])
    assert graph["release_ready"] is False
    assert graph["localized_roots"] == 1
    assert graph["source_transition_cells"] == 620
    assert len(graph["edges"]) == 1
    assert graph["edges"][0]["kind"] == "mechanism_polyline"
    assert graph["edges"][0]["mechanism"] == "plus_one"
    assert graph["edges"][0]["cell_ids"] == [0]
    assert graph["unexplained_nodes"]


def test_assembler_groups_cells_into_polylines_not_one_edge_per_cell(tmp_path) -> None:
    roots = tmp_path / "roots.json"
    roots.write_text(
        json.dumps(
            {
                "roots": [
                    {
                        "cell_id": 0,
                        "status": "ok",
                        "event_mode": "plus_one",
                        "masses": [0.80, 0.750, 1.0],
                    },
                        {
                            "cell_id": 1,
                            "status": "ok",
                            "event_mode": "plus_one",
                            "masses": [0.801, 0.751, 1.0],
                    },
                    {
                        "cell_id": 2,
                        "status": "ok",
                        "event_mode": "plus_one",
                        "masses": [1.05, 1.13, 1.0],
                    },
                    {
                        "cell_id": 3,
                        "status": "ok",
                        "event_mode": "minus_one",
                        "masses": [0.90, 0.90, 1.0],
                    },
                ]
            }
        )
    )
    graph = _run_assembler(tmp_path, ["--roots", str(roots)])
    assert graph["localized_roots"] == 4
    assert len(graph["edges"]) == 3
    plus = [edge for edge in graph["edges"] if edge["mechanism"] == "plus_one"]
    assert sorted(plus, key=lambda edge: edge["cell_ids"][0])[0]["cell_ids"] == [0, 1]
    assert graph["release_ready"] is False


def test_stale_left_birth_screen_is_explicitly_invalidated() -> None:
    payload = json.loads((ROOT / "research/evidence/V1_LEFT_BIRTH_CLASS_2026-08-15.json").read_text())
    assert payload["class"] is None
    assert payload["passed"] is False
    assert payload["evidence_level"] == "invalid"
    assert payload["invalidated_reason"] == "wrong_branch_pair"


def test_real_hybrid_roots_form_seven_slice_connected_edges(tmp_path) -> None:
    roots = ROOT / "research/evidence/V1_HYBRID_CRITICAL_ROOTS_2026-08-15.json"
    graph = _run_assembler(tmp_path, ["--roots", str(roots)])
    assert graph["localized_roots"] == 620
    assert graph["root_coverage"]["complete"] is True
    assert graph["root_coverage"]["edge_count"] == 7
    assert len({edge["id"] for edge in graph["edges"]}) == 7
    counts = sorted(edge["source_cell_count"] for edge in graph["edges"])
    assert counts == [1, 22, 46, 47, 120, 130, 254]
    assert graph["root_coverage"]["edge_topology_complete"] is False
    assert graph["release_ready"] is False


def test_assembler_refuses_screening_endpoint_even_if_payload_says_passed(tmp_path) -> None:
    endpoint = tmp_path / "left.json"
    endpoint.write_text(
        json.dumps(
            {
                "class": "projection_fold",
                "passed": True,
                "evidence_level": "screening",
                "masses": [0.9957, 0.9742, 1.0],
            }
        )
    )
    graph = _run_assembler(tmp_path, ["--left-birth", str(endpoint)])
    node = next(item for item in graph["nodes"] if item["id"] == "secondary_left_birth")
    assert node["screening_passed"] is True
    assert node["passed"] is False
    assert node["status"] == "unresolved"


def test_assembler_consumes_verified_endpoint_bindings(tmp_path) -> None:
    roots = tmp_path / "roots.json"
    roots.write_text(
        json.dumps(
            {
                "roots": [
                    {
                        "cell_id": 10,
                        "status": "ok",
                        "event_mode": "minus_one",
                        "orientation": "S->U",
                        "masses": [0.996, 0.984, 1.0],
                    }
                ]
            }
        )
    )
    endpoint = tmp_path / "left.json"
    endpoint.write_text(
        json.dumps(
            {
                "class": "projection_fold",
                "passed": True,
                "evidence_level": "continuation",
                "edge_endpoint_bindings": [
                    {
                        "cell_id": 10,
                        "side": "start",
                        "mechanism": "minus_one",
                        "orientation": "S->U",
                    }
                ],
            }
        )
    )
    graph = _run_assembler(
        tmp_path,
        ["--roots", str(roots), "--left-birth", str(endpoint)],
    )
    edge = graph["edges"][0]
    assert edge["endpoints"]["start"]["node"] == "secondary_left_birth"
    assert edge["endpoints"]["start"]["attachment"] == "verified_classification_artifact"
    assert graph["root_coverage"]["classification_binding_errors"] == []


def test_separate_fold_bindings_do_not_conflate_two_endpoints(tmp_path) -> None:
    roots = tmp_path / "roots.json"
    roots.write_text(
        json.dumps(
            {
                "roots": [
                    {
                        "cell_id": 10,
                        "status": "ok",
                        "event_mode": "plus_one",
                        "orientation": "U->S",
                        "masses": [1.042, 1.036, 1.0],
                    },
                    {
                        "cell_id": 11,
                        "status": "ok",
                        "event_mode": "minus_one",
                        "orientation": "S->U",
                        "masses": [1.042, 1.045, 1.0],
                    },
                ]
            }
        )
    )
    endpoint = tmp_path / "right.json"
    endpoint.write_text(
        json.dumps(
            {
                "class": "projection_fold",
                "passed": True,
                "evidence_level": "continuation",
                "edge_endpoint_bindings": [
                    {"cell_id": 10, "side": "end", "node_id": "secondary_right_plus_fold"},
                    {"cell_id": 11, "side": "end", "node_id": "secondary_right_minus_fold"},
                ],
            }
        )
    )
    graph = _run_assembler(
        tmp_path,
        ["--roots", str(roots), "--right-death", str(endpoint)],
    )
    attached = {
        edge["endpoints"]["end"]["node"]
        for edge in graph["edges"]
        if edge["endpoints"]["end"].get("node")
    }
    assert attached == {"secondary_right_plus_fold", "secondary_right_minus_fold"}
    assert {item["id"] for item in graph["nodes"]} >= attached


def test_assembler_rejects_failed_rows_as_mixed_germs(tmp_path) -> None:
    germs = tmp_path / "germs.json"
    germs.write_text(
        json.dumps(
            {
                "schema": "atlas.v1.mixed-germs/1",
                "germs": [
                    {
                        "mixed_node": node,
                        "event_mode": mode,
                        "direction": direction,
                        "status": "failed",
                        "masses": [1.0, 1.0, 1.0],
                    }
                    for node in (
                        "mixed_principal_left",
                        "mixed_secondary_left",
                        "mixed_principal_right",
                    )
                    for mode in ("plus_one", "minus_one")
                    for direction in ("+", "-")
                ],
            }
        )
    )
    graph = _run_assembler(tmp_path, ["--germs", str(germs)])
    assert len(graph["root_coverage"]["missing_mixed_germs"]) == 12
    assert all(germ["valid"] is False for germ in graph["mixed_germs"])
    assert graph["release_ready"] is False


def test_daughter_is_a_v1_blocker_and_no_branch_attachment_is_valid(tmp_path) -> None:
    daughter = tmp_path / "daughter.json"
    daughter.write_text(
        json.dumps(
            {
                "class": "no_branch_attachment",
                "passed": True,
                "evidence_level": "continuation",
            }
        )
    )
    graph = _run_assembler(tmp_path, ["--daughter", str(daughter)])
    assert "lower_plus_one_daughter" not in graph["unexplained_nodes"]
    assert graph["daughter_classification"]["required_for_v1_graph"] is True
    assert "secondary_left_birth" in graph["unexplained_nodes"]
    assert graph["release_ready"] is False


def test_assembler_is_the_only_path_to_a_fully_ready_graph(tmp_path) -> None:
    roots = REAL_ROOTS
    germs = REAL_GERMS

    left = tmp_path / "left.json"
    left.write_text(
        json.dumps(
            {
                "class": "projection_fold",
                "passed": True,
                "evidence_level": "independently_reproduced",
                "edge_endpoint_bindings": [
                    {"cell_id": 392, "side": "start", "mechanism": "minus_one", "orientation": "U->S"},
                    {"cell_id": 393, "side": "start", "mechanism": "minus_one", "orientation": "S->U"},
                ],
            }
        )
    )
    right = tmp_path / "right.json"
    right.write_text(
        json.dumps(
            {
                "class": "mixed_organizer",
                "passed": True,
                "evidence_level": "physical",
                "edge_endpoint_bindings": [
                    {"cell_id": 576, "side": "end", "mechanism": "plus_one", "orientation": "U->S"},
                    {"cell_id": 577, "side": "end", "mechanism": "minus_one", "orientation": "S->U"},
                ],
            }
        )
    )
    daughter = tmp_path / "daughter.json"
    daughter.write_text(
        json.dumps(
            {
                "class": "distinct_branch",
                "passed": True,
                "evidence_level": "independently_reproduced",
            }
        )
    )
    # The completeness certificate has to be a real one: the assembler re-hashes
    # every source it names and re-derives the AL and neck predicates, so a
    # hand-written self-sealed record cannot stand in for it here.
    neck_path = tmp_path / "neck.json"
    _write_clean_neck_scan(neck_path)
    completeness = _freeze_certificate(tmp_path, neck_path=neck_path)

    right_germs = _secondary_right_germ_fixture(tmp_path / "right-germs.json")

    # The real 2026-08-15 germ artifact carries no closure, no event value and
    # no canonical binding for ANY of its twelve germs, and two of them record
    # an explicit pseudo-arclength nonconvergence.  Uniform validation rejects
    # all twelve, so every base organizer is left without germs.
    graph = _run_assembler(
        tmp_path,
        [
            "--roots", str(roots),
            "--left-birth", str(left),
            "--right-death", str(right),
            "--daughter", str(daughter),
            "--germs", str(germs),
            "--germs", str(right_germs),
            "--completeness", str(completeness),
            "--sign-topology", str(_clean_sign_topology(tmp_path)),
        ],
    )
    assert graph["release_ready"] is False
    assert sorted(graph["root_coverage"]["missing_mixed_germs"]) == sorted(
        f"{node_id}:{mode}:{direction}"
        for node_id in (
            "mixed_principal_left",
            "mixed_secondary_left",
            "mixed_principal_right",
        )
        for mode in ("plus_one", "minus_one")
        for direction in ("+", "-")
    )

    # A newly retained mixed endpoint needs its own four continuation germs;
    # the headline germs cannot silently satisfy that contract.
    base_germs = _numeric_germ_fixture(tmp_path / "base-germs.json")
    graph = _run_assembler(
        tmp_path,
        [
            "--roots", str(roots),
            "--left-birth", str(left),
            "--right-death", str(right),
            "--daughter", str(daughter),
            "--germs", str(base_germs),
            "--completeness", str(completeness),
            "--sign-topology", str(_clean_sign_topology(tmp_path)),
        ],
    )
    assert graph["release_ready"] is False
    assert graph["root_coverage"]["missing_mixed_germs"] == [
        "secondary_right_death:plus_one:+",
        "secondary_right_death:plus_one:-",
        "secondary_right_death:minus_one:+",
        "secondary_right_death:minus_one:-",
    ]

    graph = _run_assembler(
        tmp_path,
        [
            "--roots", str(roots),
            "--left-birth", str(left),
            "--right-death", str(right),
            "--daughter", str(daughter),
            "--germs", str(base_germs),
            "--germs", str(right_germs),
            "--completeness", str(completeness),
            "--sign-topology", str(_clean_sign_topology(tmp_path)),
        ],
        expected_ready=True,
    )
    assert graph["release_ready"] is True
    assert graph["root_coverage"]["cells_on_edges"] == 620
    assert graph["root_coverage"]["unclassified_edge_endpoints"] == []
    assert len(graph["edges"]) == 7
    assert graph["topology"]["free_group_word"] == "bABabaBAba"
    assert graph["frozen_numerical_gates"] == {
        "maximum_absolute_event": 2e-8,
        "maximum_periodic_closure": 1e-7,
    }

    # Tampering with a source digest breaks the release, and so does re-sealing
    # the tampered record: the assembler recomputes the digest from the file.
    record = json.loads(completeness.read_text())
    for row in record["sources"]:
        if row["role"] == "neck_scan":
            row["sha256"] = "b" * 64
    completeness.write_text(json.dumps(record))
    graph = _run_assembler(
        tmp_path,
        [
            "--roots", str(roots),
            "--left-birth", str(left),
            "--right-death", str(right),
            "--daughter", str(daughter),
            "--germs", str(base_germs),
            "--germs", str(right_germs),
            "--completeness", str(completeness),
            "--sign-topology", str(_clean_sign_topology(tmp_path)),
        ],
    )
    assert graph["release_ready"] is False
    assert graph["root_coverage"]["completeness_passed"] is False


# ---------------------------------------------------------------------------
# (1) germ integrity: a base mixed node must not exempt a germ from validation
# ---------------------------------------------------------------------------


def test_real_mixed_germ_artifact_records_a_nonconvergent_trace() -> None:
    """Pin the defect in the shipped germ artifact so it cannot be forgotten.

    Two of the twelve germs -- mixed_principal_right / plus_one / + and - --
    carry status "traced" next to a stopped_reason that records an explicit
    pseudo-arclength least-squares nonconvergence.  The underlying junction
    trace has ZERO continuation points: the corrector failed on the first step,
    so those two "germs" are the two localized seed cells relabelled G+/G-.
    """
    payload = json.loads(REAL_GERMS.read_text())
    germs = payload["germs"]
    assert len(germs) == 12

    failed = [
        row
        for row in germs
        if "failed" in str(row.get("stopped_reason") or "").lower()
    ]
    assert [(row["mixed_node"], row["event_mode"], row["direction"]) for row in failed] == [
        ("mixed_principal_right", "plus_one", "+"),
        ("mixed_principal_right", "plus_one", "-"),
    ]
    for row in failed:
        assert row["status"] == "traced"
        assert "pseudo-arclength correction failed" in row["stopped_reason"]

    junction = json.loads(
        (
            ROOT / "research/evidence/V1_JUNCTION_PRINCIPAL_RIGHT_2026-08-15.json"
        ).read_text()
    )
    plus_one = next(t for t in junction["traces"] if t["event_mode"] == "plus_one")
    assert plus_one["points"] == []
    assert "failed" in plus_one["stopped_reason"]

    # None of the twelve carry any numeric evidence at all.
    for row in germs:
        assert "closure" not in row
        assert "event" not in row
        assert "canonical_bound" not in row


def test_real_mixed_germ_artifact_is_rejected_by_uniform_validation(tmp_path) -> None:
    """The twelve headline germs are rejected, and the graph says why."""
    graph = _run_assembler(tmp_path, ["--germs", str(REAL_GERMS)])
    assert len(graph["mixed_germs"]) == 12
    assert all(row["valid"] is False for row in graph["mixed_germs"])
    assert len(graph["root_coverage"]["missing_mixed_germs"]) == 12

    by_key = {
        (row["mixed_node"], row["event_mode"], row["direction"]): row
        for row in graph["mixed_germs"]
    }
    nonconvergent = by_key[("mixed_principal_right", "plus_one", "+")]
    assert "stopped_reason_records_nonconvergence" in nonconvergent["invalid_reasons"]
    converged = by_key[("mixed_principal_left", "plus_one", "+")]
    assert "stopped_reason_records_nonconvergence" not in converged["invalid_reasons"]
    # ... but it is still rejected: no numbers, no germ.
    assert set(converged["invalid_reasons"]) >= {
        "canonical_bound",
        "canonical_bracketed",
        "canonical_distance",
        "closure",
        "event",
    }


def test_base_mixed_node_does_not_exempt_a_germ_from_validation() -> None:
    """Identical numerics-free records fail for a base node and a derived one."""
    module = _assembler()
    mixed_node_ids = frozenset(
        set(module.BASE_MIXED_NODE_IDS) | {"secondary_right_death"}
    )
    for node_id in ("mixed_principal_left", "secondary_right_death"):
        record = {
            "mixed_node": node_id,
            "event_mode": "plus_one",
            "direction": "+",
            "status": "traced",
            "masses": [1.0, 1.0, 1.0],
            "source_artifact": "fixture.json",
        }
        assert module.valid_germ(record, mixed_node_ids) is False
        assert set(module.germ_rejections(record, mixed_node_ids)) == {
            "canonical_bound",
            "canonical_bracketed",
            "canonical_distance",
            "closure",
            "event",
        }


def test_recorded_nonconvergence_beats_a_traced_status() -> None:
    """A germ with perfect numbers is still rejected if its trace failed."""
    module = _assembler()
    mixed_node_ids = frozenset(module.BASE_MIXED_NODE_IDS)
    good = {
        "mixed_node": "mixed_principal_right",
        "event_mode": "plus_one",
        "direction": "+",
        "status": "traced",
        "masses": [1.05, 1.1318, 1.0],
        "source_artifact": "fixture.json",
        "canonical_bound": True,
        "canonical_bracketed": True,
        "canonical_distance": 1e-4,
        "closure": 1e-12,
        "event": 1e-12,
    }
    assert module.valid_germ(good, mixed_node_ids) is True
    assert module.valid_germ({**good, "stopped_reason": "requested_steps_completed"}, mixed_node_ids) is True

    failed = {
        **good,
        "stopped_reason": (
            "pseudo-arclength correction failed: augmented least-squares failed: "
            "The maximum number of function evaluations is exceeded."
        ),
    }
    assert module.germ_trace_failed(failed) is True
    assert module.valid_germ(failed, mixed_node_ids) is False
    assert module.germ_rejections(failed, mixed_node_ids) == [
        "stopped_reason_records_nonconvergence"
    ]


def test_germ_validation_still_honours_the_frozen_gates() -> None:
    """The uniform path holds germs to |event| <= 2e-8 and closure <= 1e-7."""
    module = _assembler()
    assert module.EVENT_GATE == 2e-8
    assert module.CLOSURE_GATE == 1e-7
    mixed_node_ids = frozenset(module.BASE_MIXED_NODE_IDS)
    base = {
        "mixed_node": "mixed_principal_left",
        "event_mode": "minus_one",
        "direction": "-",
        "status": "traced",
        "masses": [0.93, 0.885, 1.0],
        "source_artifact": "fixture.json",
        "canonical_bound": True,
        "canonical_bracketed": True,
        "canonical_distance": 1e-4,
        "closure": 1e-8,
        "event": 1.9e-8,
    }
    assert module.valid_germ(base, mixed_node_ids) is True
    assert module.valid_germ({**base, "event": 2.1e-8}, mixed_node_ids) is False
    assert module.valid_germ({**base, "event": -2.1e-8}, mixed_node_ids) is False
    assert module.valid_germ({**base, "closure": 1.1e-7}, mixed_node_ids) is False


def test_nonconvergent_germ_leaves_its_edge_endpoint_unclassified(tmp_path) -> None:
    """plus_one_u_to_s_2's start was attached solely by a failed germ.

    Its only attachment evidence was the mixed_principal_right / plus_one germ
    whose trace never converged, at a distance of 1.8e-9.  Under uniform
    validation the endpoint is unclassified instead of silently organizer-bound.
    """
    graph = _run_assembler(
        tmp_path, ["--roots", str(REAL_ROOTS), "--germs", str(REAL_GERMS)]
    )
    edge = next(e for e in graph["edges"] if e["id"] == "plus_one_u_to_s_2")
    assert edge["endpoints"]["start"]["node"] is None
    unclassified = {
        (row["edge"], row["side"])
        for row in graph["root_coverage"]["unclassified_edge_endpoints"]
    }
    assert ("plus_one_u_to_s_2", "start") in unclassified
    assert graph["root_coverage"]["edge_topology_complete"] is False
    assert graph["release_ready"] is False


# ---------------------------------------------------------------------------
# (2) distinct boundary exits are distinct nodes
# ---------------------------------------------------------------------------



def _clean_sign_topology(tmp_path) -> Path:
    """A sign-topology audit reporting no missing curve.

    The assembler's sign_topology_clean conjunct is FAIL-CLOSED: with no audit
    supplied the answer is false, because not having looked is not the same as
    having looked and found nothing.  Fixtures that exercise the rest of the
    release configuration therefore have to supply one explicitly, which is the
    intended friction -- it is no longer possible to reach release_ready without
    someone having asked whether the catalogue is the whole catalogue.
    """
    path = tmp_path / "sign_topology_clean.json"
    path.write_text(
        json.dumps(
            {
                "schema": "atlas.v1.sign-topology-audit/1",
                "probe_count": 0,
                "violation_counts": {
                    "missing_critical_curve": 0,
                    "forbidden_component_flip": 0,
                    "no_flip_across_edge": 0,
                    "face_state_mismatch": 0,
                },
                "note": "SYNTHETIC FIXTURE, NOT EVIDENCE",
            }
        )
    )
    return path

def _release_graph(tmp_path) -> dict:
    """The full release configuration, with numerics-complete germ fixtures."""
    left = tmp_path / "left.json"
    left.write_text(
        json.dumps(
            {
                "class": "projection_fold",
                "passed": True,
                "evidence_level": "independently_reproduced",
                "edge_endpoint_bindings": [
                    {"cell_id": 392, "side": "start", "mechanism": "minus_one", "orientation": "U->S"},
                    {"cell_id": 393, "side": "start", "mechanism": "minus_one", "orientation": "S->U"},
                ],
            }
        )
    )
    right = tmp_path / "right.json"
    right.write_text(
        json.dumps(
            {
                "class": "mixed_organizer",
                "passed": True,
                "evidence_level": "physical",
                "edge_endpoint_bindings": [
                    {"cell_id": 576, "side": "end", "mechanism": "plus_one", "orientation": "U->S"},
                    {"cell_id": 577, "side": "end", "mechanism": "minus_one", "orientation": "S->U"},
                ],
            }
        )
    )
    daughter = tmp_path / "daughter.json"
    daughter.write_text(
        json.dumps(
            {
                "class": "distinct_branch",
                "passed": True,
                "evidence_level": "independently_reproduced",
            }
        )
    )
    return _run_assembler(
        tmp_path,
        [
            "--roots", str(REAL_ROOTS),
            "--left-birth", str(left),
            "--right-death", str(right),
            "--daughter", str(daughter),
            "--germs", str(_numeric_germ_fixture(tmp_path / "base-germs.json")),
            "--germs", str(_secondary_right_germ_fixture(tmp_path / "right-germs.json")),
        ],
    )


def test_distinct_domain_exits_are_distinct_nodes(tmp_path) -> None:
    """Two curves hitting the same wall in different places are two termini.

    plus_one_u_to_s_0 leaves at (0.8, 0.75572) and trace_collision_s_to_u_0 at
    (0.8, 0.76073) -- five m2 grid steps apart.  plus_one_u_to_s_2 leaves at
    (1.071, 1.19996) and trace_collision_s_to_u_0 at (1.053, 1.19958) --
    eighteen m1 grid steps apart.  Lumping either pair onto one face node
    manufactures an incidence the continuation never produced.
    """
    graph = _release_graph(tmp_path)
    by_id = {edge["id"]: edge for edge in graph["edges"]}

    plus0 = by_id["plus_one_u_to_s_0"]["endpoints"]["start"]
    collision_start = by_id["trace_collision_s_to_u_0"]["endpoints"]["start"]
    plus2 = by_id["plus_one_u_to_s_2"]["endpoints"]["end"]
    collision_end = by_id["trace_collision_s_to_u_0"]["endpoints"]["end"]

    # The coordinates the lumping was hiding.
    assert plus0["masses"][:2] == [0.8, 0.7557199219114411]
    assert collision_start["masses"][:2] == [0.8, 0.7607259420854386]
    assert plus2["masses"][:2] == [1.071, 1.1999602553532644]
    assert collision_end["masses"][:2] == [1.053, 1.1995833507161973]
    assert round(collision_start["masses"][1] - plus0["masses"][1], 6) == 0.005006
    assert round(plus2["masses"][0] - collision_end["masses"][0], 9) == 0.018

    assert plus0["node"] == "domain_m1_min_m2_0p756"
    assert collision_start["node"] == "domain_m1_min_m2_0p761"
    assert plus2["node"] == "domain_m2_max_m1_1p071"
    assert collision_end["node"] == "domain_m2_max_m1_1p053"

    # A domain exit is still a legitimate, passing, declared terminus.
    domain_nodes = [
        item for item in graph["nodes"] if item["kind"] == "declared_domain_boundary"
    ]
    assert len(domain_nodes) == 4
    assert {item["id"] for item in domain_nodes} == {
        plus0["node"], collision_start["node"], plus2["node"], collision_end["node"]
    }
    for item in domain_nodes:
        assert item["passed"] is True
        assert item["status"] == "frozen_domain"
        assert item["evidence_level"] == "definition"
        assert len(item["observed_exits"]) == 1
    for endpoint in (plus0, collision_start, plus2, collision_end):
        assert endpoint["attachment"] == "declared_domain_boundary"
        assert endpoint["distance_to_domain_face"] <= 4.2e-4

    assert graph["root_coverage"]["unclassified_edge_endpoints"] == []


def test_domain_exits_no_longer_manufacture_edge_incidence(tmp_path) -> None:
    """The collision curve is its own component; it never met the plus_one arcs.

    Before the split, domain_m1_min and domain_m2_max each joined two unrelated
    edges, dragging trace_collision_s_to_u_0 into the same component as
    plus_one_u_to_s_0/2 and minus_one_u_to_s_0.  Component count 2 -> 3.
    """
    graph = _release_graph(tmp_path)
    incidence = graph["incidence"]
    assert incidence["edge_count"] == 7
    assert incidence["attached_endpoints"] == 14
    assert incidence["edge_component_count"] == 3
    assert incidence["edge_components"] == [
        ["minus_one_s_to_u_0", "minus_one_u_to_s_1", "plus_one_u_to_s_1"],
        ["minus_one_u_to_s_0", "plus_one_u_to_s_0", "plus_one_u_to_s_2"],
        ["trace_collision_s_to_u_0"],
    ]
    # Every remaining shared node is a real organizer or a classified endpoint.
    assert set(incidence["nodes_touching_more_than_one_edge"]) == {
        "mixed_principal_left",
        "mixed_principal_right",
        "mixed_secondary_left",
        "secondary_left_birth",
        "secondary_right_death",
    }
    assert not any(
        node_id.startswith("domain_")
        for node_id in incidence["nodes_touching_more_than_one_edge"]
    )


def test_domain_exit_naming_is_grid_snapped_and_face_aware() -> None:
    module = _assembler()
    assert module.MASS_GRID_STEP == 0.001
    # m1 faces are distinguished by m2, m2 faces by m1.
    assert module.domain_node([0.8, 0.7557199219114411, 1.0]) == "domain_m1_min_m2_0p756"
    assert module.domain_node([0.8, 0.7607259420854386, 1.0]) == "domain_m1_min_m2_0p761"
    assert module.domain_node([1.071, 1.1999602553532644, 1.0]) == "domain_m2_max_m1_1p071"
    assert module.domain_node([1.053, 1.1995833507161973, 1.0]) == "domain_m2_max_m1_1p053"
    assert module.domain_node([1.1, 0.9, 1.0]) == "domain_m1_max_m2_0p900"
    assert module.domain_node([0.95, 0.7, 1.0]) == "domain_m2_min_m1_0p950"
    # Interior points are not domain exits.
    assert module.domain_node([0.95, 0.95, 1.0]) is None
    assert module.domain_node(None) is None
    # Two exits inside one grid cell are the same terminus; that is the finest
    # distinction the sampling supports.
    assert module.domain_node([0.8, 0.75570, 1.0]) == module.domain_node([0.8, 0.75574, 1.0])


# ---------------------------------------------------------------------------
# (3) the four magic constants: documented justification + pinned sensitivity
# ---------------------------------------------------------------------------


def test_mass_grid_step_matches_the_release_root_set() -> None:
    from collections import defaultdict

    module = _assembler()
    roots = _real_roots()
    m1_values = sorted({round(float(row["masses"][0]), 10) for row in roots})
    assert len(m1_values) == 272
    # Globally the m1 samples are a contiguous 0.001 lattice from 0.8 to 1.071.
    assert m1_values[0] == 0.8 and m1_values[-1] == 1.071
    assert {
        round(right - left, 10)
        for left, right in zip(m1_values, m1_values[1:], strict=False)
    } == {0.001}
    # Per mechanism/orientation the slices show the three gaps the assembler
    # constants are calibrated against.
    per_branch = defaultdict(set)
    for row in roots:
        key = (str(row.get("event_mode")), str(row.get("orientation")))
        per_branch[key].add(round(float(row["masses"][0]), 10))
    gaps = set()
    for values in per_branch.values():
        ordered = sorted(values)
        gaps |= {
            round(right - left, 10)
            for left, right in zip(ordered, ordered[1:], strict=False)
        }
    assert gaps == {0.001, 0.008, 0.068}
    widths = {
        round(abs(float(row["source_m2_bracket"][1]) - float(row["source_m2_bracket"][0])), 12)
        for row in roots
        if row.get("source_m2_bracket")
    }
    assert widths == {0.001}
    assert module.MASS_GRID_STEP == 0.001


def test_mass_jump_window_is_pinned() -> None:
    """MASS_JUMP is load-bearing on the low side; pin both edges of its window."""
    module = _assembler()
    roots = _real_roots()
    assert module.MASS_JUMP == 0.025

    baseline = module.polyline_edges(roots)
    assert len(baseline) == 7

    # Largest link actually accepted: 0.011097 (minus_one S->U, 393 -> 397).
    assert len(module.polyline_edges(roots, jump=0.0111)) == 7
    assert len(module.polyline_edges(roots, jump=0.011)) == 8
    assert module.MASS_JUMP > 0.011097121

    # Smallest cross-slice link it rejects: 0.096141 (plus_one U->S, 576 -> 594),
    # reachable only if M1_SLICE_GAP is relaxed at the same time.
    assert len(module.polyline_edges(roots, jump=0.0961405, slice_gap=0.009)) == 7
    assert len(module.polyline_edges(roots, jump=0.0961407, slice_gap=0.009)) == 6
    assert module.MASS_JUMP < 0.096140639

    # Documented margins: 2.25x above the largest accepted, 3.85x below the
    # smallest rejected, an admissible window 8.66x wide.
    assert round(module.MASS_JUMP / 0.011097121, 2) == 2.25
    assert round(0.096140639 / module.MASS_JUMP, 2) == 3.85


def test_m1_slice_gap_is_currently_redundant() -> None:
    """Pin the redundancy so a data set where it starts biting gets noticed."""
    module = _assembler()
    roots = _real_roots()
    assert module.M1_SLICE_GAP == 0.0015
    assert module.M1_SLICE_GAP > module.MASS_GRID_STEP
    assert module.M1_SLICE_GAP < 0.008

    # Relaxing it alone changes nothing: MASS_JUMP already blocks both gaps.
    assert len(module.polyline_edges(roots, slice_gap=0.0015)) == 7
    assert len(module.polyline_edges(roots, slice_gap=0.008)) == 7
    assert len(module.polyline_edges(roots, slice_gap=0.068)) == 7
    # Relaxing both merges arcs -- neither guard may be widened casually.
    assert len(module.polyline_edges(roots, jump=0.11, slice_gap=0.009)) == 6
    assert len(module.polyline_edges(roots, jump=0.11, slice_gap=0.07)) == 5


def test_germ_attach_distance_window_is_pinned() -> None:
    """The admissible window for GERM_ATTACH_DISTANCE is only 1.216x wide.

    Measured on the COMMITTED graph, i.e. the release configuration in
    scripts/assemble_v1_critical_graph.sh -- not on the synthetic germ fixtures
    this file uses elsewhere.  That distinction is the point: the numbers this
    test used to pin (0.006351 accepted, 0.009933 rejected, 1.56x) came from
    _numeric_germ_fixture, so the published caveat quoted a fixture rather than
    the release.

    Largest ACCEPTED attachment 0.006837449100337747 (minus_one_u_to_s_1 end ->
    mixed_secondary_left, minus_one/-).  Nearest REJECTED mode-matching
    candidate 0.008316624803743627 (plus_one_sweep_component_12 end <->
    mixed_principal_right, plus_one/+).  The window is HALF-OPEN: a threshold
    equal to the rejected distance would admit the leftover plus_one sweep
    stub.  The previous catalog-only nearest-rejected (plus_one_u_to_s_1 end
    <-> secondary_right_death at 0.009430577) is still rejected; it is no
    longer the nearest one.

    The rejected candidate is not the secondary_left_birth endpoint.  That
    endpoint, minus_one_s_to_u_0's start, sits 2.6379e-2 from its nearest
    mode-matching germ -- past even the composed 2 x 0.008 = 0.016 reach that
    this constant's double duty creates -- and is bound by the classification
    artifact, not this threshold.
    """
    module = _assembler()
    assert module.GERM_ATTACH_DISTANCE == 0.008
    # Double duty: germ-to-organizer cap AND endpoint-to-germ cap, composed.
    effective_reach = 2 * module.GERM_ATTACH_DISTANCE
    assert effective_reach == 0.016

    graph = json.loads((ROOT / "research/evidence/V1_CRITICAL_GRAPH.json").read_text())
    germs = [row for row in graph["mixed_germs"] if row["valid"]]
    assert len(germs) == 16

    pairs = []
    for edge in graph["edges"]:
        for side in ("start", "end"):
            endpoint = edge["endpoints"][side]
            for germ in germs:
                if germ["event_mode"] != edge["mechanism"]:
                    continue
                pairs.append(
                    (
                        module.mass_distance(endpoint["masses"], germ["masses"]),
                        edge["id"],
                        side,
                        germ["mixed_node"],
                    )
                )
    pairs.sort()
    accepted = [row for row in pairs if row[0] <= module.GERM_ATTACH_DISTANCE]
    rejected = [row for row in pairs if row[0] > module.GERM_ATTACH_DISTANCE]

    largest_accepted = accepted[-1]
    nearest_rejected = rejected[0]
    assert round(largest_accepted[0], 6) == 0.006837
    # minus_one_u_to_s_1 carries a single cell, so both of its termini sit at
    # this same distance from the mixed_secondary_left germ.
    assert largest_accepted[1] == "minus_one_u_to_s_1"
    assert largest_accepted[3] == "mixed_secondary_left"
    assert {row[1:] for row in accepted if round(row[0], 6) == 0.006837} >= {
        ("minus_one_u_to_s_1", "start", "mixed_secondary_left"),
        ("minus_one_u_to_s_1", "end", "mixed_secondary_left"),
    }
    assert round(nearest_rejected[0], 6) == 0.008317
    assert nearest_rejected[1:] == ("plus_one_sweep_component_12", "end", "mixed_principal_right")

    assert largest_accepted[0] < module.GERM_ATTACH_DISTANCE < nearest_rejected[0]
    assert round(nearest_rejected[0] / largest_accepted[0], 3) == 1.216

    # The secondary_left_birth blocker is far outside even the composed reach,
    # so this constant is not what leaves it unresolved.
    blocker = min(
        row[0]
        for row in pairs
        if row[1] == "minus_one_s_to_u_0" and row[2] == "start"
    )
    assert round(blocker, 6) == 0.026379
    assert blocker > effective_reach


def test_widening_germ_attach_distance_would_resolve_a_blocked_endpoint(tmp_path) -> None:
    """The behavioural half of the pin: 0.0105 silently binds the left birth.

    With secondary_left_birth unclassified -- the project's current state --
    minus_one_s_to_u_0's start endpoint must stay unclassified.  At a germ
    attach distance of 0.0105 it is instead glued to mixed_secondary_left,
    which would make the blocker disappear without any new evidence.

    SCOPE: this runs on the SYNTHETIC germ fixtures, whose masses sit closer to
    that endpoint than the released germs do.  On the committed release germs
    the same endpoint is 2.6379e-2 from its nearest mode-matching germ, so 0.0105
    would not bind it there -- see
    test_germ_attach_distance_window_is_pinned.  What this test shows is that
    the assembler WILL glue a blocked endpoint once a germ comes into range,
    which is why the constant may not be widened casually; it is not evidence
    about how much slack the release configuration currently has.
    """
    module = _assembler()
    germs = module.collect_germs(
        [
            _numeric_germ_fixture(tmp_path / "base-germs.json"),
            _secondary_right_germ_fixture(tmp_path / "right-germs.json"),
        ]
    )
    roots = _real_roots()

    def start_node_of_minus_one_s_to_u_0() -> str | None:
        edges = module.polyline_edges(roots)
        nodes = module.headline_nodes(ROOT)
        mixed_node_ids = module.retained_mixed_nodes(nodes)
        module.attach_edge_endpoints(edges, nodes, germs, mixed_node_ids)
        edge = next(e for e in edges if e["id"] == "minus_one_s_to_u_0")
        return edge["endpoints"]["start"].get("node")

    assert start_node_of_minus_one_s_to_u_0() is None

    original = module.GERM_ATTACH_DISTANCE
    try:
        module.GERM_ATTACH_DISTANCE = 0.0105
        assert start_node_of_minus_one_s_to_u_0() == "mixed_secondary_left"
    finally:
        module.GERM_ATTACH_DISTANCE = original
    assert module.GERM_ATTACH_DISTANCE == 0.008


def test_domain_tolerance_margin_is_pinned(tmp_path) -> None:
    """Four real exits sit within 4.17e-4 of a face; nothing else is nearer than 0.05."""
    module = _assembler()
    assert module.DOMAIN_TOLERANCE == 0.0015

    graph = _release_graph(tmp_path)
    distances = []
    for edge in graph["edges"]:
        for side in ("start", "end"):
            hit_masses = edge["endpoints"][side]["masses"]
            m1, m2 = float(hit_masses[0]), float(hit_masses[1])
            distances.append(
                min(
                    abs(m1 - module.DECLARED_DOMAIN["m1"][0]),
                    abs(m1 - module.DECLARED_DOMAIN["m1"][1]),
                    abs(m2 - module.DECLARED_DOMAIN["m2"][0]),
                    abs(m2 - module.DECLARED_DOMAIN["m2"][1]),
                )
            )
    distances.sort()
    on_face = [value for value in distances if value <= module.DOMAIN_TOLERANCE]
    off_face = [value for value in distances if value > module.DOMAIN_TOLERANCE]
    assert len(on_face) == 4
    assert round(max(on_face), 6) == 0.000417
    assert round(min(off_face), 6) == 0.05
    # Margin better than 100x in both directions.
    assert module.DOMAIN_TOLERANCE / max(on_face) > 3.5
    assert min(off_face) / max(on_face) > 100

def test_committed_critical_graph_is_not_stale(tmp_path) -> None:
    """The committed graph must be exactly what the assembler emits today.

    research/evidence/V1_CRITICAL_GRAPH.json carries release_ready, and the
    assembler is the only thing allowed to set it.  If the committed file drifts
    from a fresh assembly -- because evidence landed, because the assembler
    gained a field, or because somebody hand-edited it -- that is a provenance
    break and must fail here as well as in
    .github/workflows/critical-graph-assembly.yml.
    """
    import subprocess
    import sys

    output = tmp_path / "graph.json"
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/assemble_v1_critical_graph.sh"), str(output)],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHON": sys.executable},
    )
    # 0 == release_ready, 2 == assembled but legitimately not release_ready.
    assert result.returncode in (0, 2), result.stdout + result.stderr

    committed = (ROOT / "research/evidence/V1_CRITICAL_GRAPH.json").read_text()
    assert output.read_text() == committed, (
        "Committed critical graph is stale. Regenerate with "
        "scripts/assemble_v1_critical_graph.sh research/evidence/V1_CRITICAL_GRAPH.json"
    )
    assert (json.loads(committed)["release_ready"] is True) == (result.returncode == 0)


def test_merger_refuses_to_overwrite_accepted_float64_without_audit(tmp_path) -> None:
    import runpy
    import sys

    attempts = []
    for cell in range(620):
        attempts.append(
            {
                "cell_id": cell,
                "status": "ok" if cell == 0 else "missed_event",
                "event_mode": "plus_one",
                "event": 1e-10 if cell == 0 else 1e-6,
                "closure": 1e-10,
                "masses": [0.8, 0.755, 1.0],
                "source_m2_bracket": [0.754, 0.756],
            }
        )
    python_roots = tmp_path / "python.json"
    python_roots.write_text(json.dumps({"attempts": attempts, "roots": [attempts[0]]}))
    julia = tmp_path / "julia.json"
    julia.write_text(
        json.dumps(
            {
                "cell_id": 0,
                "event_mode": "plus_one",
                "event_value": "1e-12",
                "closure_norm": "1e-20",
                "passed": True,
                "m1": "0.8",
                "m2": "0.755",
                "m3": "1.0",
            }
        )
    )
    argv = sys.argv
    sys.argv = [
        "merge_hybrid_critical_roots.py",
        str(python_roots),
        str(tmp_path / "out.json"),
        "--julia",
        str(julia),
    ]
    try:
        try:
            runpy.run_path(str(ROOT / "scripts/merge_hybrid_critical_roots.py"), run_name="__main__")
            raise AssertionError("must refuse overwrite")
        except SystemExit as exc:
            assert exc.code not in (0, None)
    finally:
        sys.argv = argv


def test_merger_refuses_julia_m2_outside_source_bracket(tmp_path) -> None:
    import runpy
    import sys

    attempts = []
    for cell in range(620):
        attempts.append(
            {
                "cell_id": cell,
                "status": "missed_event",
                "event_mode": "plus_one",
                "event": 1e-6,
                "closure": 1e-10,
                "masses": [0.8, 0.755, 1.0],
                "source_m2_bracket": [0.754, 0.756],
            }
        )
    python_roots = tmp_path / "python.json"
    python_roots.write_text(json.dumps({"attempts": attempts, "roots": []}))
    julia = tmp_path / "julia.json"
    julia.write_text(
        json.dumps(
            {
                "cell_id": 1,
                "event_mode": "plus_one",
                "event_value": "1e-12",
                "closure_norm": "1e-20",
                "passed": True,
                "m1": "0.8",
                "m2": "0.90",
                "m3": "1.0",
            }
        )
    )
    argv = sys.argv
    sys.argv = [
        "merge_hybrid_critical_roots.py",
        str(python_roots),
        str(tmp_path / "out.json"),
        "--julia",
        str(julia),
    ]
    try:
        try:
            runpy.run_path(str(ROOT / "scripts/merge_hybrid_critical_roots.py"), run_name="__main__")
            raise AssertionError("must refuse out-of-bracket m2")
        except SystemExit as exc:
            assert exc.code not in (0, None)
    finally:
        sys.argv = argv


def test_published_caveats_match_the_committed_evidence() -> None:
    """The manifest's honesty caveats must stay true of the artifacts on disk.

    Each assertion re-derives a number that the public release notes quote.
    If an artifact changes and the prose does not, this fails.
    """
    manifest = json.loads((ROOT / "research/DISCOVERY_RELEASE.json").read_text())
    known = " ".join(manifest["known_limitations"])

    census = json.loads(
        (ROOT / "research/evidence/V1_HYBRID_CRITICAL_ROOTS_2026-08-15.json").read_text()
    )
    events = [abs(float(row["event"])) for row in census["roots"]]
    gate = float(census["frozen_gates"]["event"])
    assert len(events) == 620
    assert f"{max(events):.4e}".replace("e-08", "e-8") == "1.9898e-8"
    assert "1.9898e-8" in known
    assert sum(1 for value in events if value > 1e-8) == 165
    assert "165 of the 620" in known
    assert max(events) <= gate

    graph = json.loads((ROOT / "research/evidence/V1_CRITICAL_GRAPH.json").read_text())
    parent: dict[str, str] = {}

    def find(item: str) -> str:
        parent.setdefault(item, item)
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    incident: set[str] = set()
    for edge in graph["edges"]:
        start = edge["endpoints"]["start"].get("node")
        end = edge["endpoints"]["end"].get("node")
        for name in (start, end):
            if name:
                incident.add(name)
                find(name)
        if start and end:
            parent[find(start)] = find(end)
    components = {find(name) for name in incident}
    isolated = [n["id"] for n in graph["nodes"] if n["id"] not in incident]
    # These assertions are tripwires, not permanent truths. When the two
    # unclassified edge ends are genuinely closed the graph becomes connected
    # and this test SHOULD fail -- the correct response is to update the
    # known_limitations prose to match the new graph, never to relax the test.
    assert len(components) == 3, (
        f"the committed graph's edge incidence now has {len(components)} components, not 3; "
        "update the 'assembled graph is not yet one connected object' limitation"
    )
    assert len(isolated) == 3, (
        f"{len(isolated)} nodes now carry no edge incidence, not 3; "
        "update the known_limitations wording to match"
    )
    assert "three components" in known
    assert "three further declared nodes" in known
    # Both of these closed on 2026-08-16 when the independent BigFloat fold
    # verification landed and secondary_left_birth became a projection_fold.
    assert graph["root_coverage"]["unclassified_edge_endpoints"] == []
    assert graph["unexplained_nodes"] == []
    # The defined sign-topology conjuncts are clean and the assembler has
    # flipped release_ready.  The previous G+ L-paths were the unfinished
    # climb to secondary_right_death, now recorded as closed.
    assert graph["release_ready"] is True
    assert graph["root_coverage"]["sign_topology_clean"] is True
    assert "secondary_right_death" in known


def test_germ_attach_distance_window_matches_the_published_number() -> None:
    """0.008 is a modelling choice; the release notes must quote its real slack.

    The audit runs the RELEASE configuration (scripts/assemble_v1_critical_graph.sh),
    and the window it reports is half-open: a threshold equal to the nearest
    rejected distance would admit that candidate.  The prose must say so, and
    must also disclose the constant's double duty, which composes into a
    2 x 0.008 = 0.016 organizer-to-endpoint reach.
    """
    import runpy

    namespace = runpy.run_path(str(ROOT / "scripts/audit_germ_attachment_window.py"))
    record = namespace["window"]()
    low, high = record["admissible_window"]
    assert record["default_germ_attach_distance"] == 0.008
    assert record["default_inside_window"] is True
    assert record["effective_organizer_reach"] == 0.016
    # The audit must be reading the release germs, not the superseded file.
    assert "research/evidence/V1_MIXED_GERMS_2026-08-15.json" not in record["inputs"]
    assert record["attachments_at_default"] == 12
    manifest = json.loads((ROOT / "research/DISCOVERY_RELEASE.json").read_text())
    prose = " ".join(manifest["known_limitations"]) + " ".join(
        limitation
        for claim in manifest["claims"]
        for limitation in claim.get("limitations", [])
    )
    assert f"[{low:.6f}, {high:.6f})" in prose
    assert f"{record['window_ratio']:.3f}x" in prose
    assert f"{record['effective_organizer_reach']}" in prose
