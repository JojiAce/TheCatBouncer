"""
Unit tests for passive monitoring system.
"""
import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch, MagicMock

from src.managers.passive_monitor import (
    PassiveMonitor, 
    PassiveMonitorState,
    brightness_check_strategy,
    motion_detection_strategy,
    combined_strategy
)


def test_brightness_check_strategy():
    """Test the brightness check strategy."""
    # Create a bright frame
    bright_frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    config = {
        'brightness_threshold': 220,
        'brightness_pixel_percentage': 0.5
    }
    
    result = brightness_check_strategy(bright_frame, config)
    assert result == True
    
    # Create a dark frame
    dark_frame = np.full((480, 640, 3), 10, dtype=np.uint8)
    result = brightness_check_strategy(dark_frame, config)
    assert result == False


def test_motion_detection_strategy():
    """Test the motion detection strategy."""
    # Create two different frames (motion present)
    frame1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    frame2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    config = {'motion_threshold': 10.0}
    result = motion_detection_strategy(frame2, frame1, config)
    # Motion detection might or might not detect motion depending on random values
    # Just ensure it returns a boolean
    assert isinstance(result, bool)
    
    # Test with same frames (no motion)
    result = motion_detection_strategy(frame1, frame1, config)
    assert result == False
    
    # Test with None previous frame
    result = motion_detection_strategy(frame1, None, config)
    assert result == False


def test_combined_strategy():
    """Test the combined strategy."""
    # Create a bright frame
    bright_frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    dark_frame = np.full((480, 640, 3), 10, dtype=np.uint8)
    
    config = {
        'brightness_threshold': 220,
        'brightness_pixel_percentage': 0.5,
        'motion_threshold': 10.0
    }
    
    # Test with bright frame and motion
    result = combined_strategy(bright_frame, dark_frame, config)
    assert result == True
    
    # Test with dark frame (should return False even with motion)
    result = combined_strategy(dark_frame, bright_frame, config)
    assert result == False


def test_passive_monitor_state_enum():
    """Test the PassiveMonitorState enum."""
    states = [state.value for state in PassiveMonitorState]
    expected_states = [
        "waiting_for_darkness", 
        "waiting_for_brightness", 
        "triggered",
        "stopped"
    ]
    assert set(states) == set(expected_states)


def test_passive_monitor_initialization():
    """Test PassiveMonitor initialization."""
    config = {
        'camera_index': 0,
        'low_resolution': [640, 480],
        'fps_low': 5,
        'darkness_threshold': 20,
        'brightness_threshold': 220,
        'brightness_pixel_percentage': 0.01,
        'trigger_frame_count': 5
    }
    
    # Mock the CameraManager to avoid actual camera access
    with patch('src.managers.passive_monitor.CameraManager') as mock_camera_manager:
        mock_camera = Mock()
        mock_camera.read_frame.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_camera.is_opened.return_value = True
        mock_camera.release = Mock()
        mock_camera_manager.return_value = mock_camera
        
        monitor = PassiveMonitor(config)
        
        assert monitor.camera_index == 0
        assert monitor.resolution == (640, 480)
        assert monitor.fps == 5
        assert monitor.darkness_threshold == 20
        assert monitor.brightness_threshold == 220
        assert monitor.brightness_pixel_percentage == 0.01
        assert monitor.trigger_frame_count == 5
        assert monitor.state == PassiveMonitorState.WAITING_FOR_DARKNESS
        assert monitor.triggered == False


def test_calculate_brightness():
    """Test brightness calculation."""
    # Create test instances to access the method
    config = {
        'camera_index': 0,
        'low_resolution': [640, 480],
        'fps_low': 5,
        'darkness_threshold': 20,
        'brightness_threshold': 220,
        'brightness_pixel_percentage': 0.01,
        'trigger_frame_count': 5
    }
    
    with patch('src.managers.passive_monitor.CameraManager') as mock_camera_manager:
        mock_camera = Mock()
        mock_camera.read_frame.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_camera.is_opened.return_value = True
        mock_camera.release = Mock()
        mock_camera_manager.return_value = mock_camera
        
        monitor = PassiveMonitor(config)
        
        # Test with white frame
        white_frame = np.full((480, 640, 3), 255, dtype=np.uint8)
        brightness = monitor._calculate_brightness(white_frame)
        assert brightness == 255.0
        
        # Test with black frame
        black_frame = np.full((480, 640, 3), 0, dtype=np.uint8)
        brightness = monitor._calculate_brightness(black_frame)
        assert brightness == 0.0


def test_is_bright_frame():
    """Test bright frame detection."""
    config = {
        'camera_index': 0,
        'low_resolution': [640, 480],
        'fps_low': 5,
        'darkness_threshold': 20,
        'brightness_threshold': 220,
        'brightness_pixel_percentage': 0.01,  # 1% of pixels
        'trigger_frame_count': 5
    }
    
    with patch('src.managers.passive_monitor.CameraManager') as mock_camera_manager:
        mock_camera = Mock()
        mock_camera.read_frame.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_camera.is_opened.return_value = True
        mock_camera.release = Mock()
        mock_camera_manager.return_value = mock_camera
        
        monitor = PassiveMonitor(config)
        
        # Test with very bright frame (all pixels above threshold)
        bright_frame = np.full((480, 640, 3), 255, dtype=np.uint8)
        result = monitor._is_bright_frame(bright_frame)
        assert result == True
        
        # Test with very dark frame (no pixels above threshold)
        dark_frame = np.full((480, 640, 3), 10, dtype=np.uint8)
        result = monitor._is_bright_frame(dark_frame)
        assert result == False


if __name__ == "__main__":
    test_brightness_check_strategy()
    test_motion_detection_strategy()
    test_combined_strategy()
    test_passive_monitor_state_enum()
    test_passive_monitor_initialization()
    test_calculate_brightness()
    test_is_bright_frame()
    print("All passive monitoring system tests passed!")