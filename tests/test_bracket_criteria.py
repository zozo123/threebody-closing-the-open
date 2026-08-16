"""The published-S/U-label bracket criterion is blind, and here is the proof.

The point of this file is one demonstrated blind spot, built from the real
frozen baseline rather than argued.  ``tests/fixtures/baseline_m1_0925_interior_minus_one.txt``
is a verbatim excerpt of the Li--Li--Liao supplementary table at m1 = 0.925,
m2 in [0.868, 0.882].  Inside that window:

  * the published label flips U -> S once, between m2 = 0.880 and m2 = 0.881,
    where G_plus crosses zero -- a stability boundary, which the historical
    criterion sees;
  * G_minus *also* crosses zero, between m2 = 0.874 and m2 = 0.875, where the
    unstable dimension steps 2 -> 1.  Both rows are published U, so the label
    criterion emits nothing, and refining the m2 grid only produces more U rows.

``critical_manifold.localize_critical_point`` certifies a root in that second
cell at the frozen gates -- (0.925, 0.87406129...) is one of the seven curves
reported in ``research/evidence/V1_SIGN_TOPOLOGY_AUDIT_2026-08-16.json`` and
``V1_SIGN_TOPOLOGY_CROSSING_2026-08-16.json``.  So the interval contains a
genuine critical point of exactly the kind the census catalogues, and the
historical criterion cannot produce a bracket for it at any resolution.

The headline test is parametrized over both criteria with a *strict* xfail on
``published-label``: the current criterion must fail it, and if anyone ever makes
it pass, the strict xfail turns into a failure -- which is the correct alarm,
because the 620-cell census is defined by that criterion's exact behaviour and
must stay reproducible.

Numerical policy: float64 event magnitudes here are not reproducible across
machines (the same organizer chart has been measured spanning a factor of 3.4 on
three boxes), so nothing below pins an event magnitude.  The assertions are on
discrete structure -- signs, labels, unstable dimensions, which cell a root falls
in -- and on the frozen gates as inequalities.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

from threebody_atlas.baseline import BaselineRow, iter_baseline
from threebody_atlas.bracket_criteria import (
    COMPONENT_EVENT_MODE,
    EVENT_COMPONENTS,
    MAX_CLOSURE,
    evaluate_row,
    event_sign_brackets,
    label_invisible_brackets,
    published_label_brackets,
    row_event_from_invariants,
    unstable_count,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/baseline_m1_0925_interior_minus_one.txt"

#: The published cell that contains the interior minus_one root.  Cell, not
#: coordinate: the baseline m2 grid is exact decimal, the root inside it is not.
INTERIOR_CELL = (0.874, 0.875)
#: The published cell that contains the plus_one stability boundary.
LABEL_FLIP_CELL = (0.880, 0.881)

EVENT_TOLERANCE = 2e-8  # frozen gate; never widened here


def _audit_module():
    """Load the falsifying audit script to pin our predicates against its own."""
    spec = importlib.util.spec_from_file_location(
        "audit_sign_topology_for_criteria", ROOT / "scripts/audit_sign_topology.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fixture_rows() -> list[BaselineRow]:
    rows = sorted(iter_baseline(FIXTURE), key=lambda r: r.m2)
    assert rows, "fixture excerpt did not parse"
    return rows


_STATE_CACHE: dict[float, object] = {}


def _states(rows):
    """Evaluate the event functions once per m2, cached across tests in this file.

    Each row costs one variational Newton correction plus one monodromy, so the
    cache is what keeps the real-data tests affordable.
    """
    for row in rows:
        if row.m2 not in _STATE_CACHE:
            _STATE_CACHE[row.m2] = evaluate_row(row)
    return [_STATE_CACHE[row.m2] for row in rows]


def _window(rows, lo: float, hi: float):
    return [r for r in rows if lo - 1e-9 <= r.m2 <= hi + 1e-9]


def _contains(interval: tuple[float, float], cell: tuple[float, float]) -> bool:
    return abs(interval[0] - cell[0]) < 1e-9 and abs(interval[1] - cell[1]) < 1e-9


# --------------------------------------------------------------------------
# synthetic first: the mechanism in closed form, no integration
# --------------------------------------------------------------------------
def _synthetic_row(m2: float, label: str) -> BaselineRow:
    return BaselineRow(0, 0.9, m2, 1.0, 0.0, 0.0, 0.0, 1.0, label)


def _synthetic_slice():
    """alpha = 8, beta = 44 + s: G_minus = 32 + s crosses zero at s = -32.

    With alpha = 8 the trace polynomial is P(t) = t^2 - 4t + (beta - 24), so at
    beta = 44 + s the trace roots are 2 +- sqrt(4 - (20 + s)) -- a real pair
    straddling t = -2 as s passes -32, i.e. exactly one pair leaves the unit
    circle while the other stays off it.  n_unstable steps 2 -> 1 and both rows
    are therefore published U.
    """
    out = []
    for k, s in enumerate((-36.0, -34.0, -32.5, -31.5, -30.0)):
        row = _synthetic_row(0.80 + 0.01 * k, "U")
        out.append(row_event_from_invariants(row, 8.0, 44.0 + s))
    return out


def test_synthetic_interior_crossing_keeps_every_row_unstable():
    states = _synthetic_slice()
    assert [s.n_unstable for s in states] == [2, 2, 2, 1, 1]
    assert all(s.row.published_stability == "U" for s in states)
    signs = [s.values["g_minus"] > 0 for s in states]
    assert signs == [False, False, False, True, True]


@pytest.mark.parametrize(
    "criterion",
    [
        pytest.param(
            "published-label",
            marks=pytest.mark.xfail(
                strict=True,
                reason="the S/U-label criterion cannot bracket a crossing that leaves "
                "both sides unstable -- that is the defect this module repairs",
            ),
        ),
        "event-sign",
    ],
)
def test_criterion_sees_synthetic_interior_crossing(criterion: str):
    states = _synthetic_slice()
    rows = [s.row for s in states]
    if criterion == "published-label":
        found = [(a.m2, b.m2) for a, b in published_label_brackets(rows)]
    else:
        found = [b.m2_bracket for b in event_sign_brackets(states)]
    assert found, f"{criterion} produced no bracket for a genuine G_minus crossing"


def test_unstable_count_agrees_with_the_falsifying_audit():
    """Our predicate and the audit's must be the same predicate.

    The comparison in ``scripts/audit_bracket_criteria.py`` claims "the published
    label would have read U on both sides".  That claim is only meaningful if the
    unstable-dimension predicate used here is the one the audit used to falsify
    Gate B.
    """
    audit = _audit_module()
    checked = 0
    for i in range(-40, 41):
        for j in range(-40, 41):
            alpha = 4.0 + 0.25 * i
            beta = 8.0 + 0.5 * j
            assert unstable_count(alpha, beta) == audit.unstable_count(alpha, beta)
            checked += 1
    assert checked > 6000


# --------------------------------------------------------------------------
# the real artifact: m1 = 0.925 out of the frozen baseline
# --------------------------------------------------------------------------
def test_fixture_is_a_verbatim_baseline_excerpt():
    rows = _fixture_rows()
    assert [r.m1 for r in rows] == [0.925] * len(rows)
    assert [r.m3 for r in rows] == [1.0] * len(rows)
    labels = {r.m2: r.published_stability for r in rows}
    assert labels[INTERIOR_CELL[0]] == labels[INTERIOR_CELL[1]] == "U"
    assert labels[LABEL_FLIP_CELL[0]] == "U"
    assert labels[LABEL_FLIP_CELL[1]] == "S"


def test_published_label_criterion_finds_only_the_stability_boundary():
    rows = _fixture_rows()
    found = [(a.m2, b.m2) for a, b in published_label_brackets(rows)]
    assert len(found) == 1
    assert _contains(found[0], LABEL_FLIP_CELL)


@pytest.mark.parametrize(
    "criterion",
    [
        pytest.param(
            "published-label",
            marks=pytest.mark.xfail(
                strict=True,
                reason="both endpoints of the real cell (0.925, 0.874)-(0.925, 0.875) "
                "are published U, so no label flips and no bracket is emitted -- at "
                "this or any finer m2 grid",
            ),
        ),
        "event-sign",
    ],
)
def test_criterion_brackets_the_real_interior_minus_one_curve(criterion: str):
    """The headline: a certified critical point the label criterion cannot reach.

    A root of G_minus sits inside (0.874, 0.875) at m1 = 0.925 --
    ``localize_critical_point`` certifies it at |event| <= 2e-8 and
    closure <= 1e-7 (see ``test_localizer_certifies_the_interior_root``, and the
    shipped sign-topology audit evidence).  Any criterion worth using must hand
    that cell to the localizer.
    """
    rows = _window(_fixture_rows(), 0.872, 0.877)
    if criterion == "published-label":
        found = [(a.m2, b.m2) for a, b in published_label_brackets(rows)]
    else:
        found = [
            b.m2_bracket
            for b in event_sign_brackets(_states(rows), components=("g_minus",))
        ]
    assert any(_contains(interval, INTERIOR_CELL) for interval in found), (
        f"{criterion} produced {found}; the certified interior minus_one root in "
        f"{INTERIOR_CELL} was not bracketed"
    )


#: Both interesting cells, and nothing else to integrate: each extra row costs a
#: Newton correction plus a monodromy.
BOTH_CELLS = (0.872, 0.881)


def test_event_criterion_reports_the_interior_cell_as_label_invisible():
    rows = _window(_fixture_rows(), *BOTH_CELLS)
    states = _states(rows)
    assert all(s.ok for s in states), [s.note for s in states if not s.ok]
    assert all(s.closure <= MAX_CLOSURE for s in states)

    brackets = event_sign_brackets(states)
    by_cell = {(b.component, b.m2_bracket) for b in brackets}
    assert ("g_minus", INTERIOR_CELL) in by_cell
    assert ("g_plus", LABEL_FLIP_CELL) in by_cell

    invisible = label_invisible_brackets(brackets)
    assert [(b.component, b.m2_bracket) for b in invisible] == [("g_minus", INTERIOR_CELL)]
    only = invisible[0]
    # The discrete signature of an interior crossing: the unstable dimension
    # steps down but does not reach zero, so no label can flip.
    assert (only.left.n_unstable, only.right.n_unstable) == (2, 1)
    assert only.left.row.published_stability == only.right.row.published_stability == "U"
    assert only.interior_to_unstable_region
    assert only.event_mode == "minus_one"
    # Signs only.  Magnitudes are not reproducible across machines.
    left, right = only.values
    assert left < 0.0 < right


def test_event_criterion_is_a_superset_of_the_label_criterion_here():
    rows = _window(_fixture_rows(), *BOTH_CELLS)
    label_cells = {(a.m2, b.m2) for a, b in published_label_brackets(rows)}
    event_cells = {b.m2_bracket for b in event_sign_brackets(_states(rows))}
    assert label_cells <= event_cells
    assert len(event_cells) > len(label_cells)


@pytest.mark.parametrize("component", EVENT_COMPONENTS)
def test_every_component_maps_to_a_localizer_event_mode(component: str):
    from threebody_atlas.critical_manifold import _EVENT_MODES

    assert COMPONENT_EVENT_MODE[component] in _EVENT_MODES


# --------------------------------------------------------------------------
# the extractor CLI: both criteria, and the old one unchanged
# --------------------------------------------------------------------------
def _run_extractor(tmp_path: Path, *args: str) -> list[list[str]]:
    import csv
    import subprocess

    out = tmp_path / "brackets.tsv"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/extract_mass_slice_brackets.py"),
         str(FIXTURE), str(out), *args],
        check=True,
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
    )
    with out.open(encoding="utf-8") as handle:
        return list(csv.reader(handle, delimiter="\t"))


def test_extractor_default_criterion_is_the_historical_one(tmp_path: Path):
    """Default output must stay exactly what every existing workflow consumes.

    Same columns, same order, same population: only the S/U flip.  The 620-cell
    census and every artifact derived from it depend on this staying true.
    """
    rows = _run_extractor(tmp_path, "--m1-min", "0.9", "--m1-max", "0.95")
    header, *body = rows
    assert header == [
        "m1", "m3",
        "left_m2", "left_label", "left_x1", "left_v1", "left_v2", "left_period",
        "right_m2", "right_label", "right_x1", "right_v1", "right_v2", "right_period",
    ]
    assert len(body) == 1
    assert (float(body[0][2]), float(body[0][8])) == LABEL_FLIP_CELL
    assert (body[0][3], body[0][9]) == ("U", "S")


def test_extractor_event_criterion_reports_the_interior_cell(tmp_path: Path):
    rows = _run_extractor(
        tmp_path,
        "--criterion", "event-sign",
        "--m1-min", "0.9", "--m1-max", "0.95",
        "--m2-min", "0.873", "--m2-max", "0.876",
        "--component", "g_minus",
    )
    header, *body = rows
    assert header[:14][0] == "m1"
    assert {"criterion", "component", "event_mode", "label_flip"} <= set(header)
    assert len(body) == 1
    record = dict(zip(header, body[0], strict=True))
    assert (float(record["left_m2"]), float(record["right_m2"])) == INTERIOR_CELL
    assert (record["left_label"], record["right_label"]) == ("U", "U")
    assert record["label_flip"] == "0"
    assert record["interior_to_unstable_region"] == "1"
    assert record["event_mode"] == "minus_one"
    assert (record["left_n_unstable"], record["right_n_unstable"]) == ("2", "1")
    assert float(record["left_event"]) < 0.0 < float(record["right_event"])
    assert float(record["left_closure"]) <= MAX_CLOSURE
    assert float(record["right_closure"]) <= MAX_CLOSURE


# --------------------------------------------------------------------------
# the committed census, read back: the diagnosis on the shipped artifact
# --------------------------------------------------------------------------
COMMITTED_ROOTS = ROOT / "research/evidence/V1_HYBRID_CRITICAL_ROOTS_2026-08-15.json"


def _committed_roots():
    import json

    return json.loads(COMMITTED_ROOTS.read_text(encoding="utf-8"))["roots"]


@pytest.mark.skipif(not COMMITTED_ROOTS.exists(), reason="committed roots not present")
def test_every_committed_census_root_came_from_a_published_label_flip():
    """The census population *is* the label-flip population, on the artifact.

    Not an inference about the code: every one of the 620 committed roots records
    the published labels of the cell it came from, and not one of them has equal
    labels.  So the census contains, by construction, zero roots from cells
    interior to the unstable region.
    """
    roots = _committed_roots()
    assert len(roots) == 620
    labels = [tuple(r["published_labels"]) for r in roots]
    assert set(labels) == {("U", "S"), ("S", "U")}


@pytest.mark.skipif(not COMMITTED_ROOTS.exists(), reason="committed roots not present")
def test_committed_census_carries_exactly_one_root_per_cell():
    """One cell, one root -- the second loss, visible in the shipped artifact.

    ``infer_event_mode`` returns a single event mode per bracket, so a cell that
    two event functions cross contributes one root and one mechanism label.  The
    census has exactly as many roots as cells, which is what that collapse looks
    like from the outside.
    """
    roots = _committed_roots()
    cell_ids = [r["cell_id"] for r in roots]
    assert len(set(cell_ids)) == len(cell_ids) == 620


# --------------------------------------------------------------------------
# certification: opt-in, because it is a ~30 s continuation
# --------------------------------------------------------------------------
@pytest.mark.skipif(
    not os.environ.get("ATLAS_LOCALIZE_TESTS"),
    reason="set ATLAS_LOCALIZE_TESTS=1 to run the ~30 s localizer certification",
)
def test_localizer_certifies_the_interior_root():
    """The interval really does contain a gate-passing critical point.

    Same localizer and same frozen gates that produced the 620 committed roots.
    Asserted as inequalities against the gates and as containment in the
    published cell; no float64 magnitude is pinned.
    """
    from threebody_atlas.critical_manifold import localize_critical_point
    from threebody_atlas.liao_family import correct_family_point

    rows = _window(_fixture_rows(), *INTERIOR_CELL)
    assert [r.published_stability for r in rows] == ["U", "U"]
    points = [
        correct_family_point((r.m1, r.m2, r.m3), (r.x1, r.v1, r.v2, r.period), max_nfev=60)
        for r in rows
    ]
    localized = localize_critical_point(points[0], points[1], event_mode="minus_one")
    point = localized.sample.point
    assert localized.event_mode == "minus_one"
    assert abs(localized.event_value) <= EVENT_TOLERANCE
    assert point.residual_norm <= MAX_CLOSURE
    assert INTERIOR_CELL[0] < point.masses[1] < INTERIOR_CELL[1]
    assert point.masses[0] == pytest.approx(0.925, abs=1e-12)
