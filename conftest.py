import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
ROOT = Path(__file__).parent.parent


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def dm02_input() -> dict:
    return _load(ROOT / "examples/input/manometer-dm02.json")


@pytest.fixture
def dm02_response() -> dict:
    return _load(ROOT / "tests/fixtures/manometer-dm02.response.json")


@pytest.fixture
def demo_input() -> dict:
    return _load(ROOT / "examples/input/demo-pressure-transmitter.json")


@pytest.fixture
def demo_response() -> dict:
    return _load(ROOT / "tests/fixtures/demo-pressure-transmitter.response.json")
