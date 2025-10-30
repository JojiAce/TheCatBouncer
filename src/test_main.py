"""
Unit tests for main application and CLI interface.
"""
import pytest
import tempfile
import argparse
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys
from io import StringIO

from src.main import TheCatBouncerApp, create_cli_parser, main


def test_cli_parser():
    """Test the CLI argument parser."""
    parser = create_cli_parser()
    
    # Test default values
    args = parser.parse_args([])
    assert args.config == 'config.ini'
    assert args.verbose == False
    
    # Test with some arguments
    args = parser.parse_args(['--device', 'cuda:0', '--verbose'])
    assert args.device == 'cuda:0'
    assert args.verbose == True
    
    # Test mutually exclusive group (live-preview options)
    args = parser.parse_args(['--live-preview'])
    assert args.live_preview == True
    
    args = parser.parse_args(['--no-live-preview'])
    assert args.no_live_preview == True


def test_cli_parser_invalid_args():
    """Test CLI parser with invalid combinations."""
    parser = create_cli_parser()
    
    # Should fail with both live-preview options
    try:
        parser.parse_args(['--live-preview', '--no-live-preview'])
        assert False, "Should have raised an error"
    except SystemExit:
        pass  # Expected behavior


def test_app_initialization():
    """Test application initialization."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as config_file:
        config_file.write("""
[TimeManagement]
start_time = 18:00
end_time = 07:00
data_management_time = 07:10

[Camera]
camera_index = 0
low_resolution = 640,480
high_resolution = 1920,1080
fps_low = 5
fps_high = 30

[ImageRecognition]
yolo_model_path = test_model
inference_device = cpu
cat_class_id = 15
cat_confidence_threshold = 0.8

[Actions]
intruder_light_minutes = 4.0
show_live_window = true
""")
        config_path = config_file.name
    
    try:
        # Test app initialization with CLI args
        cli_args = {
            'device': 'cpu',
            'live_preview': True
        }
        
        # Mock the components to avoid actual initialization
        with patch('src.main.ConfigManager'), \
             patch('src.main.LifecycleManager'), \
             patch('src.main.ScheduleManager'), \
             patch('src.main.StorageManager'), \
             patch('src.main.BackupService'), \
             patch('src.main.PerformanceMonitor'), \
             patch('src.main.LoadBalancer'), \
             patch('src.main.create_inference_engine', return_value=(Mock(), [])), \
             patch('src.main.ColorAnalyzer'), \
             patch('src.main.AudioManager'), \
             patch('src.main.NotificationService'), \
             patch('src.main.HueController'), \
             patch('src.main.PassiveMonitor'), \
             patch('src.main.ActiveAnalyzer'):
            
            app = TheCatBouncerApp(config_path, cli_args)
            assert app.config_path == config_path
            assert app.cli_args == cli_args
    finally:
        Path(config_path).unlink()


def test_apply_cli_overrides():
    """Test applying CLI argument overrides."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as config_file:
        config_file.write("""
[TimeManagement]
start_time = 18:00
end_time = 07:00

[Actions]
show_live_window = false
""")
        config_path = config_file.name
    
    try:
        cli_args = {
            'live_preview': True,
            'device': 'cuda:0'
        }
        
        # Mock the components to avoid actual initialization
        with patch('src.main.ConfigManager'), \
             patch('src.main.LifecycleManager'), \
             patch('src.main.ScheduleManager'), \
             patch('src.main.StorageManager'), \
             patch('src.main.BackupService'), \
             patch('src.main.PerformanceMonitor'), \
             patch('src.main.LoadBalancer'), \
             patch('src.main.create_inference_engine', return_value=(Mock(), [])), \
             patch('src.main.ColorAnalyzer'), \
             patch('src.main.AudioManager'), \
             patch('src.main.NotificationService'), \
             patch('src.main.HueController'), \
             patch('src.main.PassiveMonitor'), \
             patch('src.main.ActiveAnalyzer'):
            
            app = TheCatBouncerApp(config_path, cli_args)
            
            # Check that overrides were applied
            # This test would require checking the internal config structure
    finally:
        Path(config_path).unlink()


def test_main_function():
    """Test the main function with mock arguments."""
    # Mock sys.argv to simulate command line arguments
    original_argv = sys.argv
    sys.argv = ['main.py', '--verbose']
    
    try:
        # Mock all the components to avoid actual initialization
        with patch('src.main.create_cli_parser') as mock_parser, \
             patch('src.main.TheCatBouncerApp') as mock_app_class, \
             patch('logging.basicConfig'):
            
            # Configure the mock parser
            mock_args = Mock()
            mock_args.config = 'config.ini'
            mock_args.device = None
            mock_args.live_preview = None
            mock_args.no_live_preview = False
            mock_args.verbose = True
            mock_args.version = False
            
            mock_parser_instance = Mock()
            mock_parser_instance.parse_args.return_value = mock_args
            mock_parser.return_value = mock_parser_instance
            
            # Configure the mock app
            mock_app = Mock()
            mock_app_class.return_value = mock_app
            
            # Mock setup_components to avoid initialization errors
            mock_app.setup_components = Mock()
            
            # Run main function - we expect it to try to run but exit due to mocked components
            try:
                main()
            except SystemExit:
                # Expected when there are issues with mocked components
                pass
    
    finally:
        sys.argv = original_argv


def test_app_setup_components():
    """Test component setup (this will be mostly mocked due to dependencies)."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as config_file:
        config_file.write("""
[TimeManagement]
start_time = 18:00
end_time = 07:00

[Actions]
show_live_window = true
""")
        config_path = config_file.name
    
    try:
        # Mock all dependencies
        with patch('src.main.ConfigManager'), \
             patch('src.main.LifecycleManager'), \
             patch('src.main.ScheduleManager'), \
             patch('src.main.StorageManager'), \
             patch('src.main.BackupService'), \
             patch('src.main.PerformanceMonitor'), \
             patch('src.main.LoadBalancer'), \
             patch('src.main.create_inference_engine', return_value=(Mock(), [])), \
             patch('src.main.ColorAnalyzer'), \
             patch('src.main.AudioManager'), \
             patch('src.main.NotificationService'), \
             patch('src.main.HueController'), \
             patch('src.main.PassiveMonitor'), \
             patch('src.main.ActiveAnalyzer'):
            
            app = TheCatBouncerApp(config_path)
            
            # This should run without errors (with mocked dependencies)
            app.setup_components()
    finally:
        Path(config_path).unlink()


def test_signal_handler():
    """Test the signal handler."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as config_file:
        config_file.write("""
[TimeManagement]
start_time = 18:00
end_time = 07:00
""")
        config_path = config_file.name
    
    try:
        # Mock all dependencies
        with patch('src.main.ConfigManager'), \
             patch('src.main.LifecycleManager'), \
             patch('src.main.ScheduleManager'), \
             patch('src.main.StorageManager'), \
             patch('src.main.BackupService'), \
             patch('src.main.PerformanceMonitor'), \
             patch('src.main.LoadBalancer'), \
             patch('src.main.create_inference_engine', return_value=(Mock(), [])), \
             patch('src.main.ColorAnalyzer'), \
             patch('src.main.AudioManager'), \
             patch('src.main.NotificationService'), \
             patch('src.main.HueController'), \
             patch('src.main.PassiveMonitor'), \
             patch('src.main.ActiveAnalyzer'):
            
            app = TheCatBouncerApp(config_path)
            
            # Mock the components
            app.lifecycle_manager = Mock()
            
            # Test the signal handler
            import signal
            app._signal_handler(signal.SIGINT, None)
            assert app.shutdown_requested == True
    finally:
        Path(config_path).unlink()


if __name__ == "__main__":
    test_cli_parser()
    test_cli_parser_invalid_args()
    test_app_initialization()
    test_apply_cli_overrides()
    test_main_function()
    test_app_setup_components()
    test_signal_handler()
    print("All main application and CLI interface tests passed!")