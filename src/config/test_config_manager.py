"""
Unit tests for configuration management.
"""
import pytest
import tempfile
import os
from pathlib import Path
from src.config.config_manager import ConfigManager
from src.config.config_data import GeneralConfig


def test_config_creation():
    """Test that ConfigManager creates default configuration when file doesn't exist."""
    # Create a temporary directory for the test
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "test_config.ini"
        
        # Create ConfigManager with non-existent config file
        config_manager = ConfigManager(config_path)
        
        # Check that the file was created
        assert config_path.exists()
        
        # Check that default values are loaded
        assert config_manager.config_data.camera.camera_index == 0
        assert config_manager.config_data.image_recognition.cat_confidence_threshold == 0.8


def test_config_loading():
    """Test that ConfigManager can load existing configuration."""
    # Create a temporary directory for the test
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "test_config.ini"
        
        # Create a basic config file
        with open(config_path, 'w') as f:
            f.write("""[Camera]
camera_index = 1
low_resolution = 320,240
high_resolution = 640,480
fps_low = 10
fps_high = 20

[ImageRecognition]
yolo_model_path = custom_model_path
inference_device = cpu
cat_class_id = 16
cat_confidence_threshold = 0.9

[Actions]
intruder_light_minutes = 5.0
show_live_window = false
""")
        
        # Load the config
        config_manager = ConfigManager(config_path)
        
        # Verify values are loaded correctly
        assert config_manager.config_data.camera.camera_index == 1
        assert config_manager.config_data.camera.low_resolution == (320, 240)
        assert config_manager.config_data.camera.high_resolution == (640, 480)
        assert config_manager.config_data.camera.fps_low == 10
        assert config_manager.config_data.camera.fps_high == 20
        
        assert config_manager.config_data.image_recognition.yolo_model_path == "custom_model_path"
        assert config_manager.config_data.image_recognition.inference_device == "cpu"
        assert config_manager.config_data.image_recognition.cat_class_id == 16
        assert config_manager.config_data.image_recognition.cat_confidence_threshold == 0.9
        
        assert config_manager.config_data.actions.intruder_light_minutes == 5.0
        assert config_manager.config_data.actions.show_live_window is False


def test_config_validation():
    """Test configuration validation."""
    # Create a temporary directory for the test
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "test_config.ini"
        
        # Create a config with valid values
        with open(config_path, 'w') as f:
            f.write("""[Camera]
camera_index = 0
low_resolution = 640,480
high_resolution = 1920,1080
fps_low = 5
fps_high = 30

[ImageRecognition]
yolo_model_path = test_model_path
inference_device = cpu
cat_class_id = 15
cat_confidence_threshold = 0.8

[Actions]
intruder_light_minutes = 4.0
show_live_window = true
""")
        
        config_manager = ConfigManager(config_path)
        assert config_manager.validate() == True


def test_config_validation_with_invalid_values():
    """Test configuration validation with invalid values."""
    # Create a temporary directory for the test
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "test_config.ini"
        
        # Create a config with invalid values
        with open(config_path, 'w') as f:
            f.write("""[Camera]
camera_index = -1
low_resolution = 0,0
high_resolution = 1920,1080
fps_low = -5
fps_high = 30

[ImageRecognition]
yolo_model_path = 
inference_device = invalid_device
cat_class_id = -1
cat_confidence_threshold = 1.5

[Actions]
intruder_light_minutes = -4.0
show_live_window = true

[TimeManagement]
start_time = 25:00
end_time = invalid_time

[StorageManagement]
min_free_space_gb = -10
max_file_age_days = 0
""")
        
        config_manager = ConfigManager(config_path)
        assert config_manager.validate() == False


def test_config_get_set():
    """Test get and set methods."""
    # Create a temporary directory for the test
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "test_config.ini"
        
        config_manager = ConfigManager(config_path)
        
        # Test get method
        assert config_manager.get("camera.camera_index") == 0
        assert config_manager.get("image_recognition.cat_confidence_threshold") == 0.8
        
        # Test get with default for non-existent key
        assert config_manager.get("nonexistent.key", "default") == "default"
        
        # Test set method
        config_manager.set("camera.camera_index", 2)
        assert config_manager.get("camera.camera_index") == 2
        
        # Test nested set
        config_manager.set("image_recognition.cat_confidence_threshold", 0.95)
        assert config_manager.get("image_recognition.cat_confidence_threshold") == 0.95


def test_config_save():
    """Test saving configuration."""
    # Create a temporary directory for the test
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "test_config.ini"
        
        config_manager = ConfigManager(config_path)
        
        # Modify some values
        config_manager.set("camera.camera_index", 3)
        config_manager.set("image_recognition.cat_confidence_threshold", 0.75)
        
        # Save the config
        config_manager.save()
        
        # Create a new ConfigManager to load the saved config
        config_manager2 = ConfigManager(config_path)
        
        # Verify the values were saved
        assert config_manager2.get("camera.camera_index") == 3
        assert config_manager2.get("image_recognition.cat_confidence_threshold") == 0.75


if __name__ == "__main__":
    test_config_creation()
    test_config_loading()
    test_config_validation()
    test_config_validation_with_invalid_values()
    test_config_get_set()
    test_config_save()
    print("All configuration tests passed!")