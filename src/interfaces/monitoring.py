"""
Abstract base classes for monitoring and analysis systems.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import numpy as np


class PassiveMonitor(ABC):
    """
    Abstract base class for passive monitoring system.
    """
    
    @abstractmethod
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the passive monitor.
        
        Args:
            config: Configuration dictionary
        """
        pass

    @abstractmethod
    def start_monitoring(self):
        """
        Start the passive monitoring process.
        """
        pass

    @abstractmethod
    def stop_monitoring(self):
        """
        Stop the passive monitoring process.
        """
        pass

    @abstractmethod
    def is_triggered(self) -> bool:
        """
        Check if the monitoring has been triggered.
        
        Returns:
            True if triggered, False otherwise
        """
        pass


class ActiveAnalyzer(ABC):
    """
    Abstract base class for active analysis pipeline.
    """
    
    @abstractmethod
    def __init__(self, config: Dict[str, Any], inference_engine):
        """
        Initialize the active analyzer.
        
        Args:
            config: Configuration dictionary
            inference_engine: Inference engine instance to use
        """
        pass

    @abstractmethod
    def start_analysis(self) -> Optional[str]:
        """
        Start the active analysis process.
        
        Returns:
            Path to detection result if successful, None otherwise
        """
        pass

    @abstractmethod
    def stop_analysis(self):
        """
        Stop the active analysis process.
        """
        pass


class ColorAnalyzer(ABC):
    """
    Abstract base class for color analysis system.
    """
    
    @abstractmethod
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the color analyzer.
        
        Args:
            config: Configuration dictionary
        """
        pass

    @abstractmethod
    def analyze_color(self, image_path: str, bbox: Dict[str, int]) -> bool:
        """
        Analyze the color of an object in an image.
        
        Args:
            image_path: Path to the image file
            bbox: Bounding box coordinates (x1, y1, x2, y2)
            
        Returns:
            True if the color matches the target, False otherwise
        """
        pass


class SmartHomeController(ABC):
    """
    Abstract base class for smart home integration.
    """
    
    @abstractmethod
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the smart home controller.
        
        Args:
            config: Configuration dictionary
        """
        pass

    @abstractmethod
    def turn_on_lights(self, light_ids: list):
        """
        Turn on the specified lights.
        
        Args:
            light_ids: List of light IDs to turn on
        """
        pass

    @abstractmethod
    def turn_off_lights(self, light_ids: list):
        """
        Turn off the specified lights.
        
        Args:
            light_ids: List of light IDs to turn off
        """
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """
        Check if the smart home system is connected.
        
        Returns:
            True if connected, False otherwise
        """
        pass


class AudioManager(ABC):
    """
    Abstract base class for audio management.
    """
    
    @abstractmethod
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the audio manager.
        
        Args:
            config: Configuration dictionary
        """
        pass

    @abstractmethod
    def play_sound(self, sound_file: Optional[str] = None):
        """
        Play a sound file.
        
        Args:
            sound_file: Path to sound file to play. If None, picks random file from directory.
        """
        pass

    @abstractmethod
    def stop_sound(self):
        """
        Stop any currently playing sound.
        """
        pass


class NotificationService(ABC):
    """
    Abstract base class for notification service.
    """
    
    @abstractmethod
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the notification service.
        
        Args:
            config: Configuration dictionary
        """
        pass

    @abstractmethod
    def send_notification(self, message: str, image_path: Optional[str] = None):
        """
        Send a notification.
        
        Args:
            message: Notification message
            image_path: Optional path to image attachment
        """
        pass


class DataManagerInterface(ABC):
    """
    Abstract base class for data management.
    """
    
    @abstractmethod
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data manager.
        
        Args:
            config: Configuration dictionary
        """
        pass

    @abstractmethod
    def store_detection(self, detection_data: Dict[str, Any]) -> str:
        """
        Store detection data.
        
        Args:
            detection_data: Detection information to store
            
        Returns:
            Path where data was stored
        """
        pass

    @abstractmethod
    def cleanup_old_data(self):
        """
        Clean up old data based on retention policies.
        """
        pass

    @abstractmethod
    def backup_data(self):
        """
        Back up data to external storage.
        """
        pass