from __future__ import annotations

import json
import hashlib
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
    roots = ROOT / "research/evidence/V1_HYBRID_CRITICAL_ROOTS_2026-08-15.json"
    germs = ROOT / "research/evidence/V1_MIXED_GERMS_2026-08-15.json"

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

    # A newly retained mixed endpoint needs its own four continuation germs;
    # the twelve headline germs cannot silently satisfy that contract.
    graph = _run_assembler(
        tmp_path,
        [
            "--roots", str(roots),
            "--left-birth", str(left),
            "--right-death", str(right),
            "--daughter", str(daughter),
            "--germs", str(germs),
            "--completeness", str(completeness),
        ],
    )
    assert graph["release_ready"] is False
    assert graph["root_coverage"]["missing_mixed_germs"] == [
        "secondary_right_death:plus_one:+",
        "secondary_right_death:plus_one:-",
        "secondary_right_death:minus_one:+",
        "secondary_right_death:minus_one:-",
    ]

    right_germs = tmp_path / "right-germs.json"
    right_germs.write_text(
        json.dumps(
            {
                "schema": "atlas.v1.mixed-germs/1",
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
            "--germs", str(germs),
            "--germs", str(right_germs),
            "--completeness", str(completeness),
        ],
    )
    assert graph["release_ready"] is False
    assert graph["root_coverage"]["completeness_passed"] is False


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
