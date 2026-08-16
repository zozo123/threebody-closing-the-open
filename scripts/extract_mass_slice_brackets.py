#!/usr/bin/env python3
"""Extract mass-slice critical-curve brackets from the frozen baseline.

TWO CRITERIA, BOTH NAMED
------------------------
``--criterion published-label`` (the default) is the historical rule: bracket an
adjacent published-row pair exactly when the published S/U label flips.  Every
byte of this mode is unchanged, because the 620-cell census and everything
derived from it were produced under it and must stay reproducible.  Its output
columns and their order are unchanged too.

``--criterion event-sign`` brackets on a sign change of a Floquet *event
function* (G_plus, G_minus, discriminant) between adjacent rows.  The label rule
is structurally blind to a critical curve interior to the unstable region -- one
across which the unstable dimension steps 2 -> 1, so both sides read ``U`` and no
label flips at any grid resolution.  Seven such curves were localized at the
frozen gates in ``research/evidence/V1_SIGN_TOPOLOGY_AUDIT_2026-08-16.json`` and
``V1_SIGN_TOPOLOGY_CROSSING_2026-08-16.json``.  See
``research/BRACKET_CRITERION_BLINDNESS.md``.

CONSUMERS, READ THIS
--------------------
In ``event-sign`` mode a bracket may have ``left_label == right_label == "U"``.
Downstream scripts that pick ``stable = left if left_label == "S" else right``
are only meaningful on label-flipping brackets, so the emitted TSV carries a
``label_flip`` column: filter on ``label_flip == 1`` to recover the old
population, or pass the ``event_mode`` column to
``critical_manifold.localize_critical_point(..., event_mode=...)``, which needs a
sign-changing bracket, not a stable endpoint.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from threebody_atlas.baseline import BaselineRow, iter_baseline
from threebody_atlas.bracket_criteria import (
    CRITERIA,
    EVENT_COMPONENTS,
    evaluate_row,
    event_sign_brackets,
    published_label_brackets,
)

BASE_COLUMNS = [
    "m1",
    "m3",
    "left_m2",
    "left_label",
    "left_x1",
    "left_v1",
    "left_v2",
    "left_period",
    "right_m2",
    "right_label",
    "right_x1",
    "right_v1",
    "right_v2",
    "right_period",
]

#: Appended only in ``event-sign`` mode, so ``published-label`` output stays
#: byte-identical to every artifact produced before this criterion existed.
EVENT_COLUMNS = [
    "criterion",
    "component",
    "event_mode",
    "label_flip",
    "interior_to_unstable_region",
    "left_event",
    "right_event",
    "left_n_unstable",
    "right_n_unstable",
    "left_closure",
    "right_closure",
]


def transition_brackets(rows: list[BaselineRow]) -> list[tuple[BaselineRow, BaselineRow]]:
    """Deprecated alias for the published-label criterion.

    Kept so nothing that imports this name changes behaviour.  New code should
    call ``threebody_atlas.bracket_criteria.published_label_brackets`` and choose
    a criterion explicitly.
    """
    return published_label_brackets(rows)


def _chart_cells(m1: float, left: BaselineRow, right: BaselineRow) -> list[str]:
    return [
        f"{m1:.15g}",
        f"{left.m3:.15g}",
        f"{left.m2:.15g}",
        left.published_stability,
        f"{left.x1:.17g}",
        f"{left.v1:.17g}",
        f"{left.v2:.17g}",
        f"{left.period:.17g}",
        f"{right.m2:.15g}",
        right.published_stability,
        f"{right.x1:.17g}",
        f"{right.v1:.17g}",
        f"{right.v2:.17g}",
        f"{right.period:.17g}",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset")
    parser.add_argument("output")
    parser.add_argument("--m1-min", type=float, default=0.8)
    parser.add_argument("--m1-max", type=float, default=1.1)
    parser.add_argument("--stride", type=int, default=1, help="keep every Nth distinct m1 slice")
    parser.add_argument(
        "--criterion",
        choices=CRITERIA,
        default="published-label",
        help="published-label reproduces the 620-cell census gate; event-sign also "
        "sees critical curves interior to the unstable region (default: %(default)s)",
    )
    parser.add_argument(
        "--m2-min",
        type=float,
        default=None,
        help="event-sign only: restrict the evaluated m2 window (one Newton solve "
        "plus one monodromy per row is paid inside it)",
    )
    parser.add_argument("--m2-max", type=float, default=None)
    parser.add_argument(
        "--component",
        action="append",
        choices=EVENT_COMPONENTS,
        help="event-sign only: restrict to these event functions (default: all three)",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="event-sign only: worker processes for the per-row Floquet evaluation",
    )
    parser.add_argument(
        "--no-correct",
        action="store_true",
        help="event-sign only: read the published chart as printed instead of "
        "re-certifying closure with the variational Newton corrector.  Faster, "
        "but the emitted closure column is then NaN and no closure gate was applied.",
    )
    args = parser.parse_args()

    grouped: dict[float, list[BaselineRow]] = defaultdict(list)
    for row in iter_baseline(args.dataset):
        if args.m1_min - 1e-12 <= row.m1 <= args.m1_max + 1e-12:
            grouped[row.m1].append(row)

    slices = [m1 for i, m1 in enumerate(sorted(grouped)) if i % args.stride == 0]
    columns = BASE_COLUMNS if args.criterion == "published-label" else BASE_COLUMNS + EVENT_COLUMNS

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    total = 0
    interior = 0
    failed_rows = 0
    with Path(args.output).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(columns)
        for m1 in slices:
            if args.criterion == "published-label":
                for left, right in published_label_brackets(grouped[m1]):
                    writer.writerow(_chart_cells(m1, left, right))
                    total += 1
                continue

            rows = [
                row
                for row in sorted(grouped[m1], key=lambda r: r.m2)
                if (args.m2_min is None or row.m2 >= args.m2_min - 1e-12)
                and (args.m2_max is None or row.m2 <= args.m2_max + 1e-12)
            ]
            states = _evaluate_rows(rows, jobs=args.jobs, correct=not args.no_correct)
            failed_rows += sum(1 for s in states if not s.ok)
            brackets = event_sign_brackets(
                states, components=tuple(args.component or EVENT_COMPONENTS)
            )
            for bracket in brackets:
                left_event, right_event = bracket.values
                writer.writerow(
                    _chart_cells(m1, bracket.left.row, bracket.right.row)
                    + [
                        args.criterion,
                        bracket.component,
                        bracket.event_mode,
                        int(bracket.label_flip),
                        int(bracket.interior_to_unstable_region),
                        f"{left_event:.17g}",
                        f"{right_event:.17g}",
                        "" if bracket.left.n_unstable is None else bracket.left.n_unstable,
                        "" if bracket.right.n_unstable is None else bracket.right.n_unstable,
                        f"{bracket.left.closure:.6g}",
                        f"{bracket.right.closure:.6g}",
                    ]
                )
                total += 1
                if not bracket.label_flip:
                    interior += 1
            print(
                f"  m1={m1:.4f} rows={len(rows)} brackets={len(brackets)} "
                f"label_invisible={sum(1 for b in brackets if not b.label_flip)}",
                flush=True,
            )

    summary = f"m1_slices={len(grouped)} criterion={args.criterion} brackets={total}"
    if args.criterion == "event-sign":
        summary += f" label_invisible={interior} rows_failing_closure={failed_rows}"
    else:
        # Historical wording, kept because logs and workflow summaries grep for it.
        summary = f"m1_slices={len(grouped)} transition_brackets={total}"
    print(f"{summary} output={args.output}")


def _evaluate_rows(rows, *, jobs: int, correct: bool):
    from functools import partial

    worker = partial(evaluate_row, correct=correct)
    if jobs <= 1:
        return [worker(row) for row in rows]
    from multiprocessing import get_context

    with get_context("spawn").Pool(jobs) as pool:
        return pool.map(worker, rows)


if __name__ == "__main__":
    main()
