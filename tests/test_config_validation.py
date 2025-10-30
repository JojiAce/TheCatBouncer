"""Configuration validation behaviour tests."""

from pathlib import Path
import importlib.util

CONFIG_DATA_PATH = Path(__file__).resolve().parents[1] / "src/config/config_data.py"
CONFIG_SPEC = importlib.util.spec_from_file_location("legacy_config_data", CONFIG_DATA_PATH)
config_data = importlib.util.module_from_spec(CONFIG_SPEC)
CONFIG_SPEC.loader.exec_module(config_data)
GeneralConfig = config_data.GeneralConfig

VALIDATION_PATH = Path(__file__).resolve().parents[1] / "src/config/validation.py"
VALIDATION_SPEC = importlib.util.spec_from_file_location("legacy_validation", VALIDATION_PATH)
validation = importlib.util.module_from_spec(VALIDATION_SPEC)
VALIDATION_SPEC.loader.exec_module(validation)
validate_percentage_range = validation.validate_percentage_range
validate_camera_index = validation.validate_camera_index


def test_general_config_defaults():
    config = GeneralConfig()
    assert config.camera.camera_index == 0
    assert 0.0 <= config.image_recognition.cat_confidence_threshold <= 1.0


def test_validation_helpers():
    is_valid, _ = validate_camera_index(0)
    assert is_valid is True

    is_valid, _ = validate_camera_index(-1)
    assert is_valid is False

    valid_percentage, _ = validate_percentage_range(0.5)
    invalid_percentage, _ = validate_percentage_range(2.0)

    assert valid_percentage is True
    assert invalid_percentage is False
