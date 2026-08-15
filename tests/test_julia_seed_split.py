from pathlib import Path


def _data_rows(path: Path) -> list[str]:
    return [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ][1:]


def test_split_critical_seed_files_match_original_rows():
    root = Path(__file__).resolve().parents[1]
    original = _data_rows(root / "experiments" / "critical_curve_representatives.tsv")
    split = _data_rows(root / "experiments" / "critical_curve_lower_plus_one.tsv") + _data_rows(
        root / "experiments" / "critical_curve_upper_collision.tsv"
    )
    assert split == original
