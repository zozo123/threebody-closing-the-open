#!/usr/bin/env python3
"""Scan-window-aware topology bookkeeping for the stability-neck raster.

The neck raster exists to decide one question on every vertical (fixed-m1)
line: are the main stable region and the secondary stable lobe separated by a
resolved unstable gap, or have they merged?

The original detector answered that question with ``len(stable_intervals) <= 1``.
That test conflates two physically different situations:

* a genuine merge -- the line carries a single stable interval that begins and
  ends strictly inside the scan window, so the raster really did look at both
  sides of the putative neck and found no unstable gap; and
* a scan-window truncation -- the line carries a single stable interval that
  runs off the edge of the window, so the second lobe (and the wall that opens
  it) simply was never sampled.

The second case is not evidence of a merge.  It is evidence that the raster
cannot answer the question at all, and a raster that cannot answer the question
must not be allowed to certify completeness.  This module therefore assigns one
of four explicit verdicts per line and keeps the truncation visible instead of
folding it into ``any_vertical_merge``.

Everything here is pure and stdlib-only: the merge step of the CI workflow runs
on a bare ``setup-python`` interpreter with no project install.
"""
from __future__ import annotations

from typing import Any

# m2 grid values are produced by round(start + i * step, 10); the coarsest step
# in use is 1e-4, so a 1e-9 window-edge tolerance is at least five orders of
# magnitude tighter than one grid cell and cannot absorb a real interior gap.
BOUNDARY_ATOL = 1e-9

SEPARATED = "separated"
INTERIOR_MERGE = "interior_merge"
TRUNCATION_UNDECIDABLE = "truncation_undecidable"
NO_STABLE_SAMPLE = "no_stable_sample"

VERDICTS = (SEPARATED, INTERIOR_MERGE, TRUNCATION_UNDECIDABLE, NO_STABLE_SAMPLE)


def interval_truncation(
    interval: Any,
    m2_min: float,
    m2_max: float,
    atol: float = BOUNDARY_ATOL,
) -> dict[str, bool]:
    """Report which scan-window edges a single stable interval runs into."""
    low = float(interval[0])
    high = float(interval[1])
    return {
        "touches_m2_min": low <= float(m2_min) + atol,
        "touches_m2_max": high >= float(m2_max) - atol,
    }


def annotate_line_summary(
    summary: dict[str, Any],
    *,
    m2_min: float,
    m2_max: float,
    step: float,
    atol: float = BOUNDARY_ATOL,
) -> dict[str, Any]:
    """Return ``summary`` enriched with explicit truncation and merge verdicts.

    The input keys ``stable_intervals`` and ``interior_unstable_gaps`` are left
    untouched; only new keys are added, so the annotation is idempotent.
    """
    intervals = [[float(edge[0]), float(edge[1])] for edge in summary.get("stable_intervals") or []]
    gaps = [float(gap) for gap in summary.get("interior_unstable_gaps") or []]
    flags = [interval_truncation(interval, m2_min, m2_max, atol) for interval in intervals]
    truncated_low = any(flag["touches_m2_min"] for flag in flags)
    truncated_high = any(flag["touches_m2_max"] for flag in flags)
    boundary_truncated = truncated_low or truncated_high

    # An interior unstable gap is bracketed by stable samples on both sides
    # *inside* the window, so it witnesses separation no matter how far the
    # lobes themselves extend beyond the window.
    separation_witnessed = len(intervals) >= 2 and len(gaps) >= 1
    resolved_gaps = [gap for gap in gaps if gap + 1e-12 >= float(step)]

    if separation_witnessed:
        verdict = SEPARATED
    elif not intervals:
        verdict = NO_STABLE_SAMPLE
    elif boundary_truncated:
        verdict = TRUNCATION_UNDECIDABLE
    else:
        verdict = INTERIOR_MERGE

    annotated = dict(summary)
    annotated.update(
        {
            "interval_truncation": flags,
            "truncated_at_m2_min": truncated_low,
            "truncated_at_m2_max": truncated_high,
            "boundary_truncated": boundary_truncated,
            "separation_witnessed": separation_witnessed,
            "resolved_interior_gap_count": len(resolved_gaps),
            "merge_verdict": verdict,
        }
    )
    return annotated


def annotate_line_summaries(
    summaries: list[dict[str, Any]],
    *,
    m2_min: float,
    m2_max: float,
    step: float,
    atol: float = BOUNDARY_ATOL,
) -> list[dict[str, Any]]:
    return [
        annotate_line_summary(summary, m2_min=m2_min, m2_max=m2_max, step=step, atol=atol)
        for summary in summaries
    ]


def aggregate_line_verdicts(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Roll annotated line summaries up into explicit raster-level flags.

    ``any_vertical_merge`` now means *interior* merge only.  Truncated lines get
    their own flag, ``any_boundary_truncated_merge_test``; they are never
    silently reported as separated, and the certificate freezer refuses them.
    """
    verdicts = [summary.get("merge_verdict") for summary in summaries]
    counts = {name: verdicts.count(name) for name in VERDICTS}
    unknown = [verdict for verdict in verdicts if verdict not in VERDICTS]
    return {
        "any_vertical_merge": any(verdict == INTERIOR_MERGE for verdict in verdicts),
        "any_boundary_truncated_merge_test": any(
            verdict == TRUNCATION_UNDECIDABLE for verdict in verdicts
        ),
        "any_line_without_stable_sample": any(verdict == NO_STABLE_SAMPLE for verdict in verdicts),
        "any_stable_interval_touches_boundary": any(
            bool(summary.get("boundary_truncated")) for summary in summaries
        ),
        "all_lines_separated": bool(summaries)
        and not unknown
        and all(verdict == SEPARATED for verdict in verdicts),
        "merge_verdict_counts": counts,
        "boundary_truncated_lines": sorted(
            float(summary["m1"])
            for summary in summaries
            if summary.get("merge_verdict") == TRUNCATION_UNDECIDABLE
        ),
    }


def summarize(
    summaries: list[dict[str, Any]],
    *,
    m2_min: float,
    m2_max: float,
    step: float,
    atol: float = BOUNDARY_ATOL,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Annotate every line and return ``(annotated_summaries, raster_flags)``."""
    annotated = annotate_line_summaries(
        summaries, m2_min=m2_min, m2_max=m2_max, step=step, atol=atol
    )
    return annotated, aggregate_line_verdicts(annotated)
