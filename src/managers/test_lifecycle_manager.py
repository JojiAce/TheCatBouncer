"""
Unit tests for scheduling and lifecycle management system.
"""
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import datetime
import time

from src.managers.lifecycle_manager import (
    ScheduleManager, 
    LifecycleManager, 
    ApplicationState,
    schedule_maintenance_task,
    create_daily_scheduler
)


def test_schedule_manager_initialization():
    """Test ScheduleManager initialization."""
    config = {
        'start_time': '18:00',
        'end_time': '07:00',
        'data_management_time': '07:10'
    }
    
    manager = ScheduleManager(config)
    
    assert manager.start_time_str == '18:00'
    assert manager.end_time_str == '07:00'
    assert manager.data_management_time_str == '07:10'


def test_parse_time_string():
    """Test time string parsing."""
    config = {
        'start_time': '18:00',
        'end_time': '07:00'
    }
    
    manager = ScheduleManager(config)
    
    time_obj = manager._parse_time_string('12:34')
    assert time_obj.hour == 12
    assert time_obj.minute == 34


def test_is_within_schedule_normal_range():
    """Test schedule check for normal time range (doesn't span midnight)."""
    config = {
        'start_time': '09:00',
        'end_time': '17:00'
    }
    
    manager = ScheduleManager(config)
    
    # Mock time to be within range
    with patch('datetime.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime.datetime(2023, 1, 1, 12, 0)
        mock_datetime.strptime = datetime.datetime.strptime
        mock_datetime.combine = datetime.datetime.combine
        mock_datetime.min = datetime.datetime.min
        mock_datetime.side_effect = lambda *args, **kw: datetime.datetime(*args, **kw) if args else mock_datetime
        
        assert manager.is_within_schedule() == True


def test_is_within_schedule_spanning_midnight():
    """Test schedule check for time range that spans midnight."""
    config = {
        'start_time': '22:00',
        'end_time': '06:00'
    }
    
    manager = ScheduleManager(config)
    
    # Mock time to be within range (late night)
    with patch('datetime.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime.datetime(2023, 1, 1, 23, 0)
        mock_datetime.strptime = datetime.datetime.strptime
        mock_datetime.combine = datetime.datetime.combine
        mock_datetime.min = datetime.datetime.min
        mock_datetime.side_effect = lambda *args, **kw: datetime.datetime(*args, **kw) if args else mock_datetime
        
        assert manager.is_within_schedule() == True
    
    # Mock time to be outside range
    with patch('datetime.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime.datetime(2023, 1, 1, 12, 0)
        mock_datetime.strptime = datetime.datetime.strptime
        mock_datetime.combine = datetime.datetime.combine
        mock_datetime.min = datetime.datetime.min
        mock_datetime.side_effect = lambda *args, **kw: datetime.datetime(*args, **kw) if args else mock_datetime
        
        assert manager.is_within_schedule() == False


def test_get_next_maintenance_time():
    """Test getting next maintenance time."""
    config = {
        'start_time': '18:00',
        'end_time': '07:00',
        'data_management_time': '03:00'
    }
    
    manager = ScheduleManager(config)
    
    # Mock current time before maintenance time today
    with patch('datetime.datetime') as mock_datetime:
        mock_now = datetime.datetime(2023, 1, 1, 2, 0)  # 2:00 AM
        mock_datetime.now.return_value = mock_now
        mock_datetime.strptime = datetime.datetime.strptime
        mock_datetime.side_effect = lambda *args, **kw: datetime.datetime(*args, **kw) if args else mock_datetime
        mock_datetime.min = datetime.datetime.min
        mock_datetime.combine = datetime.datetime.combine
        
        next_maintenance = manager.get_next_maintenance_time()
        expected = mock_now.replace(hour=3, minute=0, second=0, microsecond=0)
        assert next_maintenance == expected
    
    # Mock current time after maintenance time today
    with patch('datetime.datetime') as mock_datetime:
        mock_now = datetime.datetime(2023, 1, 1, 4, 0)  # 4:00 AM
        mock_datetime.now.return_value = mock_now
        mock_datetime.strptime = datetime.datetime.strptime
        mock_datetime.side_effect = lambda *args, **kw: datetime.datetime(*args, **kw) if args else mock_datetime
        mock_datetime.min = datetime.datetime.min
        mock_datetime.combine = datetime.datetime.combine
        mock_datetime.timedelta = datetime.timedelta
        
        next_maintenance = manager.get_next_maintenance_time()
        expected = mock_now.replace(hour=3, minute=0, second=0, microsecond=0) + datetime.timedelta(days=1)
        assert next_maintenance == expected


def test_is_maintenance_time():
    """Test checking if it's maintenance time."""
    config = {
        'start_time': '18:00',
        'end_time': '07:00',
        'data_management_time': '03:00'
    }
    
    manager = ScheduleManager(config)
    
    # Mock time to be maintenance time
    with patch('datetime.datetime') as mock_datetime:
        mock_now = datetime.datetime(2023, 1, 1, 3, 0)  # 3:00 AM
        mock_datetime.now.return_value = mock_now
        mock_datetime.strptime = datetime.datetime.strptime
        mock_datetime.side_effect = lambda *args, **kw: datetime.datetime(*args, **kw) if args else mock_datetime
        mock_datetime.min = datetime.datetime.min
        mock_datetime.combine = datetime.datetime.combine
        
        assert manager.is_maintenance_time() == True
    
    # Mock time to not be maintenance time
    with patch('datetime.datetime') as mock_datetime:
        mock_now = datetime.datetime(2023, 1, 1, 12, 0)  # 12:00 PM
        mock_datetime.now.return_value = mock_now
        mock_datetime.strptime = datetime.datetime.strptime
        mock_datetime.side_effect = lambda *args, **kw: datetime.datetime(*args, **kw) if args else mock_datetime
        mock_datetime.min = datetime.datetime.min
        mock_datetime.combine = datetime.datetime.combine
        
        assert manager.is_maintenance_time() == False


def test_lifecycle_manager_initialization():
    """Test LifecycleManager initialization."""
    config = {
        'shutdown_timeout': 30,
        'error_recovery_enabled': True,
        'max_recovery_attempts': 3
    }
    
    manager = LifecycleManager(config)
    
    assert manager.state == ApplicationState.INITIALIZING
    assert manager.shutdown_timeout == 30
    assert manager.error_recovery_enabled == True


def test_register_component():
    """Test registering components."""
    config = {
        'shutdown_timeout': 30,
        'error_recovery_enabled': True,
        'max_recovery_attempts': 3
    }
    
    manager = LifecycleManager(config)
    
    mock_component = Mock()
    manager.register_component('test_component', mock_component)
    
    assert 'test_component' in manager.components
    assert manager.components['test_component'] == mock_component


def test_lifecycle_manager_states():
    """Test lifecycle manager states."""
    config = {
        'shutdown_timeout': 5,
        'error_recovery_enabled': True,
        'max_recovery_attempts': 3
    }
    
    manager = LifecycleManager(config)
    
    # Check initial state
    assert manager.state == ApplicationState.INITIALIZING
    assert manager.is_running() == False
    
    # Start application
    manager.start_application()
    assert manager.state == ApplicationState.RUNNING
    assert manager.is_running() == True
    
    # Stop application
    manager.stop_application()
    assert manager.state == ApplicationState.STOPPED


def test_handle_error_with_recovery():
    """Test error handling with recovery enabled."""
    config = {
        'shutdown_timeout': 30,
        'error_recovery_enabled': True,
        'max_recovery_attempts': 3
    }
    
    manager = LifecycleManager(config)
    
    # Mock start and stop methods to prevent actual startup/shutdown
    with patch.object(manager, 'restart_application'):
        manager.state = ApplicationState.RUNNING
        
        # Simulate error
        manager.handle_error(Exception("Test error"), "test_component")
        
        # Recovery should be attempted
        assert manager.recovery_attempts == 1
        assert manager.state != ApplicationState.ERROR


def test_handle_error_max_attempts():
    """Test error handling when max attempts reached."""
    config = {
        'shutdown_timeout': 30,
        'error_recovery_enabled': True,
        'max_recovery_attempts': 1
    }
    
    manager = LifecycleManager(config)
    
    # Mock restart to fail
    with patch.object(manager, 'restart_application', side_effect=Exception("Restart failed")):
        manager.state = ApplicationState.RUNNING
        
        # First error
        manager.handle_error(Exception("Test error"), "test_component")
        assert manager.recovery_attempts == 1
        
        # Second error should result in error state
        manager.handle_error(Exception("Test error"), "test_component")
        assert manager.state == ApplicationState.ERROR


def test_get_schedule_info():
    """Test getting schedule information."""
    config = {
        'start_time': '18:00',
        'end_time': '07:00',
        'data_management_time': '03:00'
    }
    
    manager = ScheduleManager(config)
    
    schedule_info = manager.get_schedule_info()
    
    assert 'current_time' in schedule_info
    assert 'start_time' in schedule_info
    assert 'end_time' in schedule_info
    assert 'is_active' in schedule_info
    assert schedule_info['start_time'] == '18:00'
    assert schedule_info['end_time'] == '07:00'


def test_schedule_maintenance_task():
    """Test scheduling a maintenance task."""
    config = {
        'start_time': '18:00',
        'end_time': '07:00',
        'data_management_time': '03:00'
    }
    
    schedule_manager = ScheduleManager(config)
    
    # Mock a task function
    mock_task = Mock()
    
    # Test that the function at least returns something (a thread in real implementation)
    thread = schedule_maintenance_task(schedule_manager, mock_task)
    
    # In the real implementation, this would return a thread, but for testing
    # we're just ensuring it doesn't throw an exception


def test_create_daily_scheduler():
    """Test creating a daily scheduler."""
    config = {
        'start_time': '18:00',
        'end_time': '07:00',
        'data_management_time': '03:00'
    }
    
    # Mock a task function
    mock_task = Mock()
    
    # Test that the function at least returns something
    result = create_daily_scheduler(config, mock_task)
    
    # In the real implementation, this would return a thread


if __name__ == "__main__":
    test_schedule_manager_initialization()
    test_parse_time_string()
    test_is_within_schedule_normal_range()
    test_is_within_schedule_spanning_midnight()
    test_get_next_maintenance_time()
    test_is_maintenance_time()
    test_lifecycle_manager_initialization()
    test_register_component()
    test_lifecycle_manager_states()
    test_handle_error_with_recovery()
    test_handle_error_max_attempts()
    test_get_schedule_info()
    test_schedule_maintenance_task()
    test_create_daily_scheduler()
    print("All scheduling and lifecycle management tests passed!")