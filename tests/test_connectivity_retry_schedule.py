import importlib.util
from pathlib import Path
import sys


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SCRIPT = SCRIPTS / "verify_single_connectivity_edge.py"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("verify_single_connectivity_edge", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_retry_schedule_doubles_without_exceeding_maximum():
    assert MODULE.retry_schedule(6, 48) == [6, 12, 24, 48]


def test_retry_schedule_includes_non_power_of_two_maximum_once():
    assert MODULE.retry_schedule(5, 36) == [5, 10, 20, 36]
