"""Parser for the Li--Li--Liao non-hierarchical unequal-mass baseline dataset."""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from .schema import OrbitCandidate


@dataclass(frozen=True)
class BaselineRow:
    index: int
    m1: float
    m2: float
    m3: float
    x1: float
    v1: float
    v2: float
    period: float
    published_stability: str

    def candidate(self) -> OrbitCandidate:
        v3 = -(self.m1 * self.v1 + self.m2 * self.v2) / self.m3
        return OrbitCandidate(
            source=f"Li-Li-Liao arXiv:2007.10184 supplementary row {self.index}",
            masses=(self.m1, self.m2, self.m3),
            period=self.period,
            initial_state=(
                self.x1,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                self.v1,
                0.0,
                self.v2,
                0.0,
                v3,
            ),
        )


def iter_baseline(path: str | Path) -> Iterator[BaselineRow]:
    """Yield numerical rows; header/rule lines are ignored robustly."""
    row_index = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.split()
            if len(fields) != 8 or fields[-1] not in {"S", "U"}:
                continue
            try:
                values = [float(x) for x in fields[:7]]
            except ValueError:
                continue
            row_index += 1
            yield BaselineRow(row_index, *values, fields[-1])


def load_range(path: str | Path, start: int, stop: int) -> list[BaselineRow]:
    if start < 1 or stop < start:
        raise ValueError("range must use 1-based indices with stop >= start")
    return [row for row in iter_baseline(path) if start <= row.index <= stop]
