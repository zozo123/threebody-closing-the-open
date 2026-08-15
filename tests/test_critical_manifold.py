from __future__ import annotations

from types import SimpleNamespace

import pytest

from threebody_atlas.critical_manifold import event_value, infer_event_mode


def invariants_from_trace_roots(t1: float, t2: float):
    alpha = 4.0 + t1 + t2
    beta = t1 * t2 + 4.0 * alpha - 8.0
    disc = (t1 - t2) ** 2
    roots = (complex(t1), complex(t2))
    return SimpleNamespace(
        alpha=alpha,
        beta=beta,
        discriminant=disc,
        trace_roots=roots,
    )


def sample(t1: float, t2: float):
    return SimpleNamespace(floquet=invariants_from_trace_roots(t1, t2))


def test_plus_one_event_is_trace_polynomial_at_plus_two() -> None:
    inv = invariants_from_trace_roots(2.0, -0.25)
    assert event_value(inv, "plus_one") == pytest.approx(0.0, abs=1e-13)
    assert event_value(inv, "minus_one") != pytest.approx(0.0, abs=1e-6)


def test_minus_one_event_is_trace_polynomial_at_minus_two() -> None:
    inv = invariants_from_trace_roots(-2.0, 0.4)
    assert event_value(inv, "minus_one") == pytest.approx(0.0, abs=1e-13)
    assert event_value(inv, "plus_one") != pytest.approx(0.0, abs=1e-6)


def test_trace_collision_event_is_discriminant() -> None:
    inv = invariants_from_trace_roots(0.8, 0.8)
    assert event_value(inv, "trace_collision") == pytest.approx(0.0, abs=1e-13)


def test_infer_plus_one_from_sign_change() -> None:
    a = sample(1.98, -0.2)
    b = sample(2.02, -0.2)
    assert infer_event_mode(a, b) == "plus_one"


def test_infer_trace_collision_from_discriminant_sign_change() -> None:
    # Synthetic Floquet objects are used here because two real trace roots have
    # nonnegative discriminant.  The sign change is the defining algebraic gate.
    a = SimpleNamespace(
        floquet=SimpleNamespace(alpha=5.7, beta=15.65, discriminant=1e-3, trace_roots=(0.9 + 0j, 0.8 + 0j))
    )
    b = SimpleNamespace(
        floquet=SimpleNamespace(alpha=5.7, beta=15.65, discriminant=-1e-3, trace_roots=(0.85 + 0.01j, 0.85 - 0.01j))
    )
    assert infer_event_mode(a, b) == "trace_collision"
