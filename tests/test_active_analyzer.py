"""Tests for the active analyzer manager utilities."""

from pathlib import Path
import importlib.util
import sys
import types

if "cv2" not in sys.modules:
    sys.modules["cv2"] = types.SimpleNamespace()
if "numpy" not in sys.modules:
    numpy_stub = types.SimpleNamespace(
        ndarray=type("ndarray", (), {}),
        expand_dims=lambda array, axis: array,
    )
    sys.modules["numpy"] = numpy_stub

ACTIVE_ANALYZER_PATH = Path(__file__).resolve().parents[1] / "src/managers/active_analyzer.py"
SPEC = importlib.util.spec_from_file_location("legacy_active_analyzer", ACTIVE_ANALYZER_PATH)
active_analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(active_analyzer)
EngineSpec = active_analyzer.EngineSpec


def test_engine_spec_creates_unique_instances():
    spec = EngineSpec(
        module="tests.stubs",
        class_name="DummyEngine",
        init_args=(),
        init_kwargs={},
    )

    first_engine = spec.create_engine()
    second_engine = spec.create_engine()

    assert first_engine is not second_engine
    assert first_engine.predictions == []
    assert second_engine.predictions == []
