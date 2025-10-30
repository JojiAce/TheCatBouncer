"""
Abstract base classes for scheduling and lifecycle management.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import datetime


class ScheduleManager(ABC):
    """
    Abstract base class for schedule management.
    """
    
    @abstractmethod
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the schedule manager.
        
        Args:
            config: Configuration dictionary
        """
        pass

    @abstractmethod
    def is_within_schedule(self, start_time: str, end_time: str) -> bool:
        """
        Check if the current time is within the specified schedule.
        
        Args:
            start_time: Start time in HH:MM format
            end_time: End time in HH:MM format
            
        Returns:
            True if within schedule, False otherwise
        """
        pass

    @abstractmethod
    def get_next_maintenance_time(self) -> datetime.datetime:
        """
        Get the next scheduled maintenance time.
        
        Returns:
            Datetime of next maintenance
        """
        pass

    @abstractmethod
    def is_maintenance_time(self) -> bool:
        """
        Check if it's currently maintenance time.
        
        Returns:
            True if it's maintenance time, False otherwise
        """
        pass


class LifecycleManager(ABC):
    """
    Abstract base class for application lifecycle management.
    """
    
    @abstractmethod
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the lifecycle manager.
        
        Args:
            config: Configuration dictionary
        """
        pass

    @abstractmethod
    def start_application(self):
        """
        Start the application.
        """
        pass

    @abstractmethod
    def stop_application(self):
        """
        Stop the application gracefully.
        """
        pass

    @abstractmethod
    def restart_application(self):
        """
        Restart the application.
        """
        pass

    @abstractmethod
    def is_running(self) -> bool:
        """
        Check if the application is running.
        
        Returns:
            True if running, False otherwise
        """
        pass