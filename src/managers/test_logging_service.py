"""
Unit tests for comprehensive logging system.
"""
import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import logging
import json
import time

from src.managers.logging_service import (
    LoggingService,
    StructuredFormatter,
    AsyncLogHandler,
    setup_logging,
    log_function_call,
    PerformanceTracker,
    get_global_logger
)


def test_logging_service_initialization():
    """Test LoggingService initialization."""
    config = {
        'log_level': 'DEBUG',
        'log_directory': 'test_logs',
        'rotation_when': 'midnight',
        'rotation_interval': 1,
        'rotation_backup_count': 7,
        'async_logging': True
    }
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config['log_directory'] = temp_dir
        service = LoggingService(config)
        
        assert service.log_level == 'DEBUG'
        assert service.rotation_when == 'midnight'
        assert service.async_logging == True


def test_structured_formatter():
    """Test StructuredFormatter output."""
    formatter = StructuredFormatter()
    
    # Create a mock log record
    record = logging.LogRecord(
        name='test_logger',
        level=logging.INFO,
        pathname='test.py',
        lineno=10,
        msg='Test message',
        args=(),
        exc_info=None
    )
    
    formatted = formatter.format(record)
    
    # Parse the JSON output
    log_entry = json.loads(formatted)
    
    assert 'timestamp' in log_entry
    assert log_entry['level'] == 'INFO'
    assert log_entry['message'] == 'Test message'
    assert log_entry['logger'] == 'test_logger'


def test_log_detection():
    """Test logging detection events."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = {'log_directory': temp_dir}
        service = LoggingService(config)
        
        detection_data = {
            'class_name': 'cat',
            'confidence': 0.95,
            'bbox_coords': [100, 100, 200, 200]
        }
        
        service.log_detection(detection_data, 'test_image.jpg')
        
        # Verify the log was created
        log_files = list(Path(temp_dir).glob('thecatbouncer.log*'))
        assert len(log_files) >= 1


def test_performance_metric_logging():
    """Test logging performance metrics."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = {'log_directory': temp_dir}
        service = LoggingService(config)
        
        service.log_performance_metric('fps', 30.0, {'component': 'active_analysis'})
        service.log_performance_metric('latency_ms', 25.5, {'component': 'inference'})
        
        summary = service.get_performance_summary()
        
        assert 'fps' in summary
        assert 'latency_ms' in summary
        assert summary['fps']['avg'] == 30.0
        assert summary['latency_ms']['latest'] == 25.5


def test_system_event_logging():
    """Test logging system events."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = {'log_directory': temp_dir}
        service = LoggingService(config)
        
        service.log_system_event('startup', 'Application started', 'INFO', 
                               {'version': '1.0.0', 'platform': 'linux'})
        
        # The log should be written without errors


def test_error_logging_with_context():
    """Test logging errors with context."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = {'log_directory': temp_dir}
        service = LoggingService(config)
        
        try:
            # Force an exception
            raise ValueError("Test error")
        except ValueError as e:
            service.log_error_with_context(
                e, 
                {'input_data': 'test', 'function': 'test_func'}, 
                'test_component'
            )


def test_get_performance_summary():
    """Test getting performance summary."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = {'log_directory': temp_dir}
        service = LoggingService(config)
        
        # Add some metrics
        for i in range(5):
            service.log_performance_metric('response_time_ms', 100 + i * 10)
        
        summary = service.get_performance_summary()
        
        assert 'response_time_ms' in summary
        assert summary['response_time_ms']['count'] == 5
        assert summary['response_time_ms']['min'] == 100
        assert summary['response_time_ms']['max'] == 140


def test_enable_disable_debug_logging():
    """Test enabling and disabling debug logging."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = {'log_directory': temp_dir}
        service = LoggingService(config)
        
        # Initially should be at configured level (INFO)
        assert service.logger.level == logging.INFO
        
        service.enable_debug_logging()
        assert service.logger.level == logging.DEBUG
        
        service.disable_debug_logging()
        assert service.logger.level == logging.INFO


def test_cleanup_old_logs():
    """Test cleaning up old logs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = {'log_directory': temp_dir}
        service = LoggingService(config)
        
        # Create some old log files manually
        old_log = Path(temp_dir) / 'thecatbouncer.log.old'
        old_log.write_text('old log content')
        
        # Set the file's modification time to the past
        old_time = time.time() - (2 * 24 * 60 * 60)  # 2 days ago
        os.utime(old_log, (old_time, old_time))
        
        # Clean up logs older than 1 day
        service.cleanup_old_logs(days_to_keep=1)
        
        # The old file should be removed
        assert not old_log.exists()


def test_log_function_call_decorator():
    """Test the log_function_call decorator."""
    @log_function_call
    def test_function(x, y):
        return x + y
    
    # Capture logs (this is difficult to test directly without complex mocking)
    # Instead, just ensure the decorator doesn't break the function
    result = test_function(2, 3)
    assert result == 5


def test_performance_tracker():
    """Test PerformanceTracker utility."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = {'log_directory': temp_dir}
        service = LoggingService(config)
        
        tracker = PerformanceTracker(service)
        
        tracker.start_timer('test_operation')
        time.sleep(0.01)  # Sleep briefly
        elapsed = tracker.stop_timer('test_operation', log=True)
        
        assert elapsed >= 0.01  # Should be at least the sleep time


def test_async_log_handler():
    """Test AsyncLogHandler functionality."""
    # Mock a delegate handler
    delegate_handler = Mock()
    async_handler = AsyncLogHandler(delegate_handler, max_queue_size=100)
    
    # Create a log record
    record = logging.LogRecord(
        name='test',
        level=logging.INFO,
        pathname='test.py',
        lineno=10,
        msg='Test message',
        args=(),
        exc_info=None
    )
    
    # Emit the record
    async_handler.emit(record)
    
    # Process the queue (in a real scenario, this would happen in the background thread)
    time.sleep(0.1)  # Allow time for async processing
    
    # Handler should have received the record
    # Note: This test is limited because the async handler processes in a background thread
    async_handler._cleanup()


def test_setup_logging():
    """Test the setup_logging function."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = {'log_directory': temp_dir}
        service = setup_logging(config)
        
        assert isinstance(service, LoggingService)


def test_get_global_logger():
    """Test getting the global logger."""
    logger1 = get_global_logger()
    logger2 = get_global_logger()
    
    # Should return the same instance
    assert logger1 is logger2


def test_flush_logs():
    """Test flushing logs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = {'log_directory': temp_dir, 'async_logging': False}  # Disable async for this test
        service = LoggingService(config)
        
        # Log a message
        service.logger.info("Test message for flush")
        
        # Flush logs
        service.flush_logs()
        
        # The method should execute without errors


if __name__ == "__main__":
    test_logging_service_initialization()
    test_structured_formatter()
    test_log_detection()
    test_performance_metric_logging()
    test_system_event_logging()
    test_error_logging_with_context()
    test_get_performance_summary()
    test_enable_disable_debug_logging()
    test_cleanup_old_logs()
    test_log_function_call_decorator()
    test_performance_tracker()
    test_async_log_handler()
    test_setup_logging()
    test_get_global_logger()
    test_flush_logs()
    print("All comprehensive logging system tests passed!")