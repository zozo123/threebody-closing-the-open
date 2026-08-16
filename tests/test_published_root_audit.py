"""Re-derive published critical roots with the shipped dynamics.

Nothing in this repository did that before.  ``tests/test_float64_census.py``
checks the numbers a census *recorded*; ``tests/test_critical_graph.py`` hands
the roots file to the assembler.  Neither integrates anything, so the entire
620-root release would have kept passing CI with a broken force law.

These tests close that loop.  They are also the reason the mutation harness can
demonstrate a kill for a shifted gravitational constant.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from threebody_atlas import root_audit

ROOT = Path(__file__).resolve().parents[1]
REAL_ROOTS = ROOT / "research/evidence/V1_HYBRID_CRITICAL_ROOTS_2026-08-15.json"

# Eight roots spread deterministically across the census: two per estimator
# regime in practice, all three event modes represented.  Each costs ~3s.
SAMPLE_COUNT = 8


@pytest.fixture(scope="module")
def audits() -> list[root_audit.RootAudit]:
    return root_audit.audit(REAL_ROOTS, count=SAMPLE_COUNT)


def test_float64_reproduces_the_julia_bigfloat_trace_invariants(
    audits: list[root_audit.RootAudit],
) -> None:
    """A genuine cross-implementation check.

    ``alpha``/``beta``/``discriminant`` in the roots file were produced by the
    Julia BigFloat estimator.  Recomputing them from the recorded chart
    coordinates with the Python SciPy path is therefore an independent
    reproduction, not a tautology, and it is sensitive enough to notice a
    relative change of order 1e-11 in the force law.
    """
    assert len(audits) == SAMPLE_COUNT
    problems = [problem for item in audits for problem in item.failures()]
    assert not problems, "\n".join(problems)


def test_frozen_closure_gate_survives_float64_rederivation(
    audits: list[root_audit.RootAudit],
) -> None:
    """The 1e-7 closure gate is reproducible in float64, and is asserted at 1e-7."""
    worst = max(item.closure for item in audits)
    assert worst <= root_audit.CLOSURE_GATE, f"worst float64 closure {worst:.3e}"


def test_float64_cannot_reverify_the_event_gate_and_says_so(
    audits: list[root_audit.RootAudit],
) -> None:
    """Document the precision cliff instead of pretending it is not there.

    The frozen |event| <= 2e-8 gate belongs to the BigFloat estimator.  Re-run
    in float64, the same roots show |event| up to a few times 1e-6 -- roughly
    two orders of magnitude above the gate -- because
    ``event = beta - 6 alpha + 20`` amplifies an alpha that float64 only carries
    to ~1e-7.  If this test ever starts finding float64 events at or below 2e-8
    across the sample, either the integrator got dramatically better or the
    audit stopped auditing; both deserve a human look.
    """
    worst = max(abs(item.event) for item in audits)
    assert worst <= root_audit.FLOAT64_EVENT_BAND, (
        f"float64 |event| {worst:.3e} exceeds even the documented float64 band"
    )
    assert worst > 2e-8, (
        f"float64 |event| max {worst:.3e} is at or below the frozen BigFloat gate; "
        "this contradicts the measured precision cliff and must be investigated, "
        "not celebrated"
    )


def test_audit_covers_all_three_event_modes(audits: list[root_audit.RootAudit]) -> None:
    modes = {item.event_mode for item in audits}
    assert modes == {"plus_one", "minus_one", "trace_collision"}, modes


def test_differential_comparison_detects_a_perturbed_invariant(tmp_path) -> None:
    """The harness's detector must actually reject drift.

    A detector nobody has watched reject anything is worth very little, so the
    differential comparison is exercised against a deliberately corrupted
    baseline here rather than only inside the mutation harness.
    """
    audits = root_audit.audit(REAL_ROOTS, count=2)
    baseline = root_audit.to_json(audits)
    assert not root_audit.compare(audits, baseline, tolerance=1e-12)

    tampered = json.loads(json.dumps(baseline))
    tampered["audits"][0]["alpha"] += 1e-6
    problems = root_audit.compare(audits, tampered, tolerance=1e-9)
    assert problems and "alpha drifted" in problems[0]

    dropped = json.loads(json.dumps(baseline))
    dropped["audits"] = dropped["audits"][:1]
    assert any("absent from the baseline" in text for text in
               root_audit.compare(audits, dropped, tolerance=1e-9))


def test_roots_file_shape_is_what_the_audit_assumes() -> None:
    payload = json.loads(REAL_ROOTS.read_text(encoding="utf-8"))
    roots = root_audit.load_roots(REAL_ROOTS)
    assert len(roots) == 620
    assert payload["localized_roots"] == 620
    for row in roots[:5]:
        for key in ("x1", "v1", "v2", "period", "alpha", "beta", "discriminant", "masses"):
            assert key in row, key
