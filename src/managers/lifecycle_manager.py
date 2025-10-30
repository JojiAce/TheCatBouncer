"""
Scheduling and lifecycle management system.
"""
import logging
import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import time
import threading
from enum import Enum

from src.interfaces.scheduling import ScheduleManager, LifecycleManager


class ApplicationState(Enum):
    """Enum for application states."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class ScheduleManager(ScheduleManager):
    """
    Schedule manager for time-based operation with support for time ranges spanning midnight.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the schedule manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Extract configuration values
        self.start_time_str = config.get('start_time', '18:00')
        self.end_time_str = config.get('end_time', '07:00')
        self.data_management_time_str = config.get('data_management_time', '07:10')
        
        # Parse and validate time strings
        self.start_time = self._parse_time_string(self.start_time_str)
        self.end_time = self._parse_time_string(self.end_time_str)
        self.data_management_time = self._parse_time_string(self.data_management_time_str) if self.data_management_time_str else None
        
        self.logger.info(f"Schedule manager initialized. Active hours: {self.start_time_str} - {self.end_time_str}")
    
    def _parse_time_string(self, time_str: str) -> datetime.time:
        """
        Parse a time string in HH:MM format to a datetime.time object.
        
        Args:
            time_str: Time string in HH:MM format
            
        Returns:
            datetime.time object
        """
        try:
            return datetime.datetime.strptime(time_str, '%H:%M').time()
        except ValueError:
            raise ValueError(f"Invalid time format: {time_str}. Expected HH:MM")
    
    def is_within_schedule(self, start_str: Optional[str] = None, end_str: Optional[str] = None) -> bool:
        """
        Check if the current time is within the specified schedule.
        Handles time ranges that span midnight (e.g., 22:00 to 06:00).
        
        Args:
            start_str: Start time in HH:MM format (defaults to configured start_time)
            end_str: End time in HH:MM format (defaults to configured end_time)
            
        Returns:
            True if within schedule, False otherwise
        """
        if start_str is None:
            start_time = self.start_time
            start_str = self.start_time_str
        else:
            start_time = self._parse_time_string(start_str)
        
        if end_str is None:
            end_time = self.end_time
            end_str = self.end_time_str
        else:
            end_time = self._parse_time_string(end_str)
        
        now = datetime.datetime.now().time()
        
        # Handle time ranges spanning midnight
        # E.g., 22:00 to 06:00 means from 22:00 to midnight and from midnight to 06:00
        if start_time <= end_time:
            # Normal range (does not span midnight)
            is_active = start_time <= now <= end_time
        else:
            # Range spans midnight (e.g., 22:00 to 06:00)
            is_active = now >= start_time or now <= end_time
        
        self.logger.debug(f"Time check: {now.strftime('%H:%M')} within {start_str}-{end_str}: {is_active}")
        return is_active
    
    def get_next_maintenance_time(self) -> datetime.datetime:
        """
        Get the next scheduled maintenance time.
        
        Returns:
            Datetime of next maintenance
        """
        if not self.data_management_time:
            raise ValueError("Data management time not configured")
        
        now = datetime.datetime.now()
        next_maintenance = now.replace(
            hour=self.data_management_time.hour,
            minute=self.data_management_time.minute,
            second=0,
            microsecond=0
        )
        
        # If the maintenance time has already passed today, schedule for tomorrow
        if next_maintenance <= now:
            next_maintenance += datetime.timedelta(days=1)
        
        return next_maintenance
    
    def is_maintenance_time(self) -> bool:
        """
        Check if it's currently maintenance time.
        
        Returns:
            True if it's maintenance time, False otherwise
        """
        if not self.data_management_time:
            return False
        
        now = datetime.datetime.now()
        current_time = now.time()
        
        # Check if it's the maintenance time (within a minute window)
        maintenance_time = self.data_management_time
        time_diff = abs(
            (current_time.hour * 60 + current_time.minute) - 
            (maintenance_time.hour * 60 + maintenance_time.minute)
        )
        
        is_maintenance = time_diff <= 1  # Within 1 minute of scheduled time
        
        self.logger.debug(f"Maintenance time check: {current_time.strftime('%H:%M')} - Maintenance: {maintenance_time.strftime('%H:%M')} - Is Maintenance Time: {is_maintenance}")
        return is_maintenance
    
    def get_schedule_info(self) -> Dict[str, Any]:
        """
        Get information about the current schedule.
        
        Returns:
            Dictionary with schedule information
        """
        now = datetime.datetime.now()
        is_active = self.is_within_schedule()
        next_maintenance = self.get_next_maintenance_time() if self.data_management_time else None
        
        info = {
            'current_time': now.strftime('%H:%M:%S'),
            'start_time': self.start_time_str,
            'end_time': self.end_time_str,
            'is_active': is_active,
            'active_duration_seconds': self._get_active_duration().total_seconds(),
            'next_maintenance': next_maintenance.isoformat() if next_maintenance else None,
            'time_until_maintenance_seconds': (next_maintenance - now).total_seconds() if next_maintenance else None
        }
        
        return info
    
    def _get_active_duration(self) -> datetime.timedelta:
        """
        Get the duration of the active period.
        
        Returns:
            Duration of the active period
        """
        if self.start_time <= self.end_time:
            # Normal range (does not span midnight)
            duration = datetime.datetime.combine(datetime.date.min, self.end_time) - datetime.datetime.combine(datetime.date.min, self.start_time)
        else:
            # Range spans midnight (e.g., 22:00 to 06:00)
            duration = (datetime.timedelta(days=1) - 
                       (datetime.datetime.combine(datetime.date.min, self.start_time) - datetime.datetime.min).time() + 
                       datetime.datetime.combine(datetime.date.min, self.end_time).time())
        
        return duration


class LifecycleManager(LifecycleManager):
    """
    Lifecycle manager for application lifecycle management.
    Handles initialization, graceful shutdown, and error recovery.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the lifecycle manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Extract configuration values
        self.shutdown_timeout = config.get('shutdown_timeout', 30)  # seconds
        self.error_recovery_enabled = config.get('error_recovery_enabled', True)
        self.max_recovery_attempts = config.get('max_recovery_attempts', 3)
        
        # Initialize state
        self.state = ApplicationState.INITIALIZING
        self.components = {}
        self.recovery_attempts = 0
        
        self.logger.info("Lifecycle manager initialized")
    
    def start_application(self):
        """
        Start the application.
        """
        self.logger.info("Starting application...")
        self.state = ApplicationState.RUNNING
        
        # Initialize all components
        self._initialize_components()
        
        self.logger.info("Application started successfully")
    
    def stop_application(self):
        """
        Stop the application gracefully.
        """
        self.logger.info("Stopping application gracefully...")
        self.state = ApplicationState.STOPPING
        
        # Stop all components
        self._stop_components()
        
        # Perform cleanup
        self._cleanup()
        
        self.state = ApplicationState.STOPPED
        self.logger.info("Application stopped successfully")
    
    def restart_application(self):
        """
        Restart the application.
        """
        self.logger.info("Restarting application...")
        self.stop_application()
        time.sleep(1)  # Small delay before restart
        self.start_application()
        self.recovery_attempts = 0  # Reset recovery attempts
    
    def is_running(self) -> bool:
        """
        Check if the application is running.
        
        Returns:
            True if running, False otherwise
        """
        return self.state == ApplicationState.RUNNING
    
    def register_component(self, name: str, component: Any):
        """
        Register a component for lifecycle management.
        
        Args:
            name: Name of the component
            component: Component instance
        """
        self.components[name] = component
        self.logger.debug(f"Registered component: {name}")
    
    def handle_error(self, error: Exception, component_name: str = "unknown"):
        """
        Handle an error in a component.
        
        Args:
            error: The error that occurred
            component_name: Name of the component where error occurred
        """
        self.logger.error(f"Error in component {component_name}: {error}")
        
        if self.error_recovery_enabled and self.recovery_attempts < self.max_recovery_attempts:
            self.recovery_attempts += 1
            self.logger.info(f"Attempting recovery (attempt {self.recovery_attempts}/{self.max_recovery_attempts})...")
            
            try:
                # Try to restart the application
                self.restart_application()
                self.logger.info("Recovery successful")
            except Exception as e:
                self.logger.error(f"Recovery failed: {e}")
                self.state = ApplicationState.ERROR
        else:
            self.logger.error("Max recovery attempts reached. Application in error state.")
            self.state = ApplicationState.ERROR
    
    def _initialize_components(self):
        """
        Initialize all registered components.
        """
        for name, component in self.components.items():
            try:
                if hasattr(component, 'start'):
                    component.start()
                    self.logger.debug(f"Started component: {name}")
                elif hasattr(component, 'initialize'):
                    component.initialize()
                    self.logger.debug(f"Initialized component: {name}")
            except Exception as e:
                self.logger.error(f"Failed to initialize component {name}: {e}")
                raise
    
    def _stop_components(self):
        """
        Stop all registered components gracefully.
        """
        # Stop components in reverse order of initialization
        for name, component in reversed(list(self.components.items())):
            try:
                if hasattr(component, 'stop'):
                    component.stop()
                    self.logger.debug(f"Stopped component: {name}")
                elif hasattr(component, 'cleanup'):
                    component.cleanup()
                    self.logger.debug(f"Cleaned up component: {name}")
            except Exception as e:
                self.logger.error(f"Error stopping component {name}: {e}")
    
    def _cleanup(self):
        """
        Perform general cleanup operations.
        """
        self.logger.info("Performing application cleanup...")
        
        # Clear component references
        self.components.clear()
        
        # Perform any other cleanup tasks as needed
        # For example, close file handles, database connections, etc.
    
    def add_shutdown_hook(self, hook_function):
        """
        Add a function to be called during shutdown.
        
        Args:
            hook_function: Function to call during shutdown
        """
        if not hasattr(self, '_shutdown_hooks'):
            self._shutdown_hooks = []
        
        self._shutdown_hooks.append(hook_function)
        self.logger.debug(f"Added shutdown hook: {hook_function.__name__}")
    
    def run_with_lifecycle(self, main_loop_function):
        """
        Run a main loop function with proper lifecycle management.
        
        Args:
            main_loop_function: Function to run as the main application loop
        """
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, shutting down gracefully...")
            self.stop_application()
        
        # Set up signal handlers for graceful shutdown (this would be done in the main app)
        import signal
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            self.start_application()
            main_loop_function()
        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt, shutting down...")
        except Exception as e:
            self.handle_error(e)
        finally:
            self.stop_application()


def schedule_maintenance_task(schedule_manager: ScheduleManager, task_function):
    """
    Schedule a maintenance task to run at the configured maintenance time.
    
    Args:
        schedule_manager: Schedule manager instance
        task_function: Function to run during maintenance
    """
    def maintenance_loop():
        while True:
            if schedule_manager.is_maintenance_time():
                try:
                    task_function()
                except Exception as e:
                    logging.error(f"Error during maintenance task: {e}")
            
            # Sleep for 1 minute before checking again
            time.sleep(60)
    
    # Run in a separate thread
    thread = threading.Thread(target=maintenance_loop, daemon=True)
    thread.start()
    return thread


def create_daily_scheduler(schedule_config: Dict[str, Any], task_function):
    """
    Create a scheduler that runs a task daily at a specific time.
    
    Args:
        schedule_config: Schedule configuration
        task_function: Function to run on schedule
    """
    schedule_manager = ScheduleManager(schedule_config)
    return schedule_maintenance_task(schedule_manager, task_function)