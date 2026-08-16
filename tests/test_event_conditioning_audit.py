"""Tests for scripts/audit_event_conditioning.py.

The audit's job is to say when a recorded event certificate is below the float64
floor.  These tests pin the two things that could quietly make it useless: the
floor formula, and the refusal to write into evidence.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "audit_event_conditioning.py"
_spec = importlib.util.spec_from_file_location("audit_event_conditioning", _PATH)
assert _spec is not None and _spec.loader is not None
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)

GATE = 2e-8


def test_roundoff_floor_scales_with_monodromy_norm_squared():
    assert audit.roundoff_floor(2000.0, "minus_one") == pytest.approx(
        4.0 * audit.roundoff_floor(1000.0, "minus_one")
    )


def test_trace_collision_carries_the_extra_factor_of_four():
    """Delta = (alpha-4)^2 - 4(beta - 4 alpha + 8) amplifies beta's error fourfold."""
    plain = audit.roundoff_floor(5000.0, "plus_one")
    assert audit.roundoff_floor(5000.0, "trace_collision") == pytest.approx(4.0 * plain)


def test_floor_crosses_the_frozen_event_gate_in_the_observed_range():
    """The crossover must land inside the census's actual ||M|| range, 7e2 .. 2.4e4.

    If it did not, the audit would either flag everything or nothing, and would
    be incapable of discriminating -- which is the failure mode of a check that
    never fires.
    """
    crossover = np.sqrt(GATE / audit.EPS)
    assert 7e2 < crossover < 2.4e4
    assert audit.roundoff_floor(2.4e4, "plus_one") > GATE
    assert audit.roundoff_floor(7e2, "trace_collision") < GATE


def test_audit_refuses_to_write_into_research_evidence(monkeypatch, tmp_path):
    census = tmp_path / "census.json"
    census.write_text('{"frozen_gates": {"event": 2e-8, "closure": 1e-7}, "roots": []}')
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_event_conditioning.py",
            str(census),
            "--output",
            "research/evidence/SHOULD_NOT_BE_WRITTEN.json",
        ],
    )
    with pytest.raises(SystemExit) as excinfo:
        audit.main()
    assert "research/evidence" in str(excinfo.value)


def test_audit_reports_zero_findings_on_an_empty_census(monkeypatch, tmp_path, capsys):
    census = tmp_path / "census.json"
    census.write_text('{"frozen_gates": {"event": 2e-8, "closure": 1e-7}, "roots": []}')
    monkeypatch.setattr(sys, "argv", ["audit_event_conditioning.py", str(census), "--estimator", "all"])
    assert audit.main() == 0
    out = capsys.readouterr().out
    assert '"roots_audited": 0' in out
