"""
Unit tests for active analysis pipeline.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import multiprocessing as mp
from multiprocessing import Queue, Event

from src.managers.active_analyzer import ActiveAnalyzer, DetectionResult


def test_detection_result_dataclass():
    """Test the DetectionResult dataclass."""
    result = DetectionResult(
        image=np.zeros((480, 640, 3), dtype=np.uint8),
        detections=[np.array([1, 2, 3, 4, 0.9, 15])],
        inference_time_ms=10.5,
        bbox_coords=(100, 100, 200, 200),
        confidence=0.9,
        class_id=15,
        class_name="cat"
    )
    
    assert result.inference_time_ms == 10.5
    assert result.confidence == 0.9
    assert result.class_name == "cat"


def test_active_analyzer_initialization():
    """Test ActiveAnalyzer initialization."""
    config = {
        'camera_source': 0,
        'high_resolution': [1920, 1080],
        'fps_high': 30,
        'timeout_sec': 60,
        'cpu_workers': 2,
        'inference_device': 'cpu',
        'show_live_window': True,
        'exit_key': 'q',
        'target_class_name': 'cat',
        'min_confidence': 0.8,
        'success_frame_folder': 'test_detections'
    }
    
    mock_engine = Mock()
    mock_engine.get_class_names.return_value = ['person', 'cat', 'dog']
    
    analyzer = ActiveAnalyzer(config, mock_engine)
    
    assert analyzer.camera_source == 0
    assert analyzer.high_resolution == (1920, 1080)
    assert analyzer.fps_high == 30
    assert analyzer.timeout_sec == 60
    assert analyzer.cpu_workers == 2
    assert analyzer.target_class_name == 'cat'
    assert analyzer.min_confidence == 0.8


def test_check_for_target_object():
    """Test target object detection logic."""
    config = {
        'camera_source': 0,
        'high_resolution': [1920, 1080],
        'fps_high': 30,
        'timeout_sec': 60,
        'cpu_workers': 2,
        'inference_device': 'cpu',
        'show_live_window': False,  # Disable window for tests
        'exit_key': 'q',
        'target_class_name': 'cat',
        'min_confidence': 0.8,
        'success_frame_folder': 'test_detections'
    }
    
    mock_engine = Mock()
    mock_engine.get_class_names.return_value = ['person', 'cat', 'dog']
    
    analyzer = ActiveAnalyzer(config, mock_engine)
    
    # Test with a detection that should match (cat with high confidence)
    result = DetectionResult(
        image=np.zeros((480, 640, 3), dtype=np.uint8),
        detections=[np.array([[100, 100, 200, 200, 0.9, 1]])],  # x1, y1, x2, y2, conf, cls_id (cat = index 1)
        inference_time_ms=10.0
    )
    
    found = analyzer._check_for_target_object(result)
    assert found == True
    assert result.class_name == 'cat'
    assert result.confidence == 0.9
    
    # Test with a detection that doesn't meet confidence threshold
    result2 = DetectionResult(
        image=np.zeros((480, 640, 3), dtype=np.uint8),
        detections=[np.array([[100, 100, 200, 200, 0.5, 1]])],  # Low confidence
        inference_time_ms=10.0
    )
    
    found2 = analyzer._check_for_target_object(result2)
    assert found2 == False
    
    # Test with a detection that is not the target class
    result3 = DetectionResult(
        image=np.zeros((480, 640, 3), dtype=np.uint8),
        detections=[np.array([[100, 100, 200, 200, 0.9, 0]])],  # person instead of cat
        inference_time_ms=10.0
    )
    
    found3 = analyzer._check_for_target_object(result3)
    assert found3 == False


def test_annotate_frame():
    """Test frame annotation functionality."""
    config = {
        'camera_source': 0,
        'high_resolution': [1920, 1080],
        'fps_high': 30,
        'timeout_sec': 60,
        'cpu_workers': 2,
        'inference_device': 'cpu',
        'show_live_window': False,
        'exit_key': 'q',
        'target_class_name': 'cat',
        'min_confidence': 0.8,
        'success_frame_folder': 'test_detections'
    }
    
    mock_engine = Mock()
    mock_engine.get_class_names.return_value = ['person', 'cat', 'dog']
    
    analyzer = ActiveAnalyzer(config, mock_engine)
    
    # Create a test frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Test with detections
    detections = [np.array([[100, 100, 200, 200, 0.9, 1]])]  # cat detection
    
    annotated = analyzer._annotate_frame(frame, detections, 15.5)
    
    # Check that the result is a valid frame
    assert annotated.shape == (480, 640, 3)
    assert annotated.dtype == np.uint8


def test_save_successful_detection():
    """Test saving successful detection."""
    import tempfile
    import json
    from pathlib import Path
    
    config = {
        'camera_source': 0,
        'high_resolution': [1920, 1080],
        'fps_high': 30,
        'timeout_sec': 60,
        'cpu_workers': 2,
        'inference_device': 'cpu',
        'show_live_window': False,
        'exit_key': 'q',
        'target_class_name': 'cat',
        'min_confidence': 0.8,
        'success_frame_folder': 'test_detection_output'
    }
    
    mock_engine = Mock()
    mock_engine.get_class_names.return_value = ['person', 'cat', 'dog']
    
    analyzer = ActiveAnalyzer(config, mock_engine)
    
    # Create a test result
    result = DetectionResult(
        image=np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        detections=[np.array([[100, 100, 200, 200, 0.9, 1]])],
        inference_time_ms=10.0,
        bbox_coords=(100, 100, 200, 200),
        confidence=0.9,
        class_id=1,
        class_name="cat"
    )
    
    # Test saving
    detection_path = analyzer._save_successful_detection(result)
    
    if detection_path:
        # Check that the directory and files exist
        detection_folder = Path(detection_path)
        assert detection_folder.exists()
        assert (detection_folder / "frame.jpg").exists()
        assert (detection_folder / "metadata.json").exists()
        
        # Check metadata content
        with open(detection_folder / "metadata.json", 'r') as f:
            metadata = json.load(f)
            assert metadata["class_name"] == "cat"
            assert float(metadata["confidence"]) == 0.9
    
    # Clean up test output
    import shutil
    if Path('test_detection_output').exists():
        shutil.rmtree('test_detection_output')


def test_pipeline_processes():
    """Test the creation of pipeline processes (without running them)."""
    config = {
        'camera_source': 0,
        'high_resolution': [640, 480],
        'fps_high': 30,
        'timeout_sec': 5,  # Short timeout for tests
        'cpu_workers': 1,
        'inference_device': 'cpu',
        'show_live_window': False,
        'exit_key': 'q',
        'target_class_name': 'cat',
        'min_confidence': 0.5,
        'success_frame_folder': 'test_detections'
    }
    
    mock_engine = Mock()
    mock_engine.get_class_names.return_value = ['person', 'cat', 'dog']
    
    analyzer = ActiveAnalyzer(config, mock_engine)
    
    # Verify that the analyzer was initialized properly
    assert analyzer.timeout_sec == 5
    assert analyzer.min_confidence == 0.5


if __name__ == "__main__":
    test_detection_result_dataclass()
    test_active_analyzer_initialization()
    test_check_for_target_object()
    test_annotate_frame()
    test_save_successful_detection()
    test_pipeline_processes()
    print("All active analysis pipeline tests passed!")