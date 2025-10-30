"""Behavioural tests for the legacy main pipeline helpers."""

from pathlib import Path
import importlib.util
import datetime
import sys
import types

if "cv2" not in sys.modules:
    sys.modules["cv2"] = types.SimpleNamespace()
if "requests" not in sys.modules:
    requests_stub = types.ModuleType("requests")
    urllib3_exceptions_module = types.ModuleType("requests.packages.urllib3.exceptions")
    urllib3_exceptions_module.InsecureRequestWarning = RuntimeError

    urllib3_module = types.ModuleType("requests.packages.urllib3")
    urllib3_module.exceptions = urllib3_exceptions_module
    urllib3_module.disable_warnings = lambda *args, **kwargs: None

    packages_module = types.ModuleType("requests.packages")
    packages_module.urllib3 = urllib3_module

    requests_stub.packages = packages_module
    requests_stub.get = lambda *args, **kwargs: None

    sys.modules["requests"] = requests_stub
    sys.modules["requests.packages"] = packages_module
    sys.modules["requests.packages.urllib3"] = urllib3_module
    sys.modules["requests.packages.urllib3.exceptions"] = urllib3_exceptions_module

MODULE_PATH = Path(__file__).resolve().parents[1] / "main-file_and_active_analysis_pipeline.py"
SPEC = importlib.util.spec_from_file_location("legacy_main_pipeline", MODULE_PATH)
legacy_main = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(legacy_main)


def test_is_within_schedule_handles_midnight(monkeypatch):
    fake_now = datetime.datetime(2024, 1, 1, 0, 30)

    class _FixedDateTime(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return fake_now

    monkeypatch.setattr(legacy_main.datetime, "datetime", _FixedDateTime)

    assert legacy_main.is_within_schedule("23:00", "01:00") is True


def test_is_within_schedule_invalid_format():
    assert legacy_main.is_within_schedule("invalid", "time") is False
