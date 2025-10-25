"""
Passive monitoring system for brightness detection.
"""
import cv2
import numpy as np
import logging
from typing import Dict, Any, Callable, Optional, Tuple
import time
from enum import Enum

from src.interfaces.monitoring import PassiveMonitor
from src.managers.camera_manager import CameraManager


class PassiveMonitorState(Enum):
    """States for the passive monitoring system."""
    WAITING_FOR_DARKNESS = "waiting_for_darkness"
    WAITING_FOR_BRIGHTNESS = "waiting_for_brightness"
    TRIGGERED = "triggered"
    STOPPED = "stopped"


class PassiveMonitor(PassiveMonitor):
    """
    Passive monitoring system that detects brightness changes to trigger active analysis.
    Implements state machine for darkness → brightness detection.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the passive monitor.
        
        Args:
            config: Configuration dictionary containing passive monitoring settings
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Extract configuration values
        self.camera_index = config.get('camera_index', 0)
        self.resolution = tuple(config.get('low_resolution', (640, 480)))
        self.fps = config.get('fps_low', 5)
        self.darkness_threshold = config.get('darkness_threshold', 20)
        self.brightness_threshold = config.get('brightness_threshold', 220)
        self.brightness_pixel_percentage = config.get('brightness_pixel_percentage', 0.01)
        self.trigger_frame_count = config.get('trigger_frame_count', 5)
        self.max_idle_time = config.get('max_idle_time', 3600)  # 1 hour default
        
        # Initialize state
        self.state = PassiveMonitorState.WAITING_FOR_DARKNESS
        self.bright_frame_counter = 0
        self.triggered = False
        self.camera_manager = None
        self.start_time = time.time()
        
        # Initialize camera
        self._initialize_camera()
        
        self.logger.info(f"Passive monitor initialized with config: "
                        f"resolution={self.resolution}, fps={self.fps}, "
                        f"darkness_threshold={self.darkness_threshold}, "
                        f"brightness_threshold={self.brightness_threshold}")
    
    def _initialize_camera(self):
        """Initialize the camera manager."""
        try:
            self.camera_manager = CameraManager(
                source=self.camera_index,
                resolution=self.resolution,
                fps=self.fps
            )
            self.logger.info("Camera initialized for passive monitoring")
        except Exception as e:
            self.logger.error(f"Failed to initialize camera: {e}")
            raise
    
    def _calculate_brightness(self, frame: np.ndarray) -> float:
        """
        Calculate the average brightness of a frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Average brightness value (0-255)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        return float(np.mean(gray))
    
    def _is_bright_frame(self, frame: np.ndarray) -> bool:
        """
        Check if a frame meets the brightness criteria.
        
        Args:
            frame: Input frame
            
        Returns:
            True if the frame is considered bright, False otherwise
        """
        # Convert to grayscale for brightness analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        # Count pixels above brightness threshold
        bright_pixels = np.sum(gray > self.brightness_threshold)
        total_pixels = gray.size
        current_percentage = bright_pixels / total_pixels
        
        return current_percentage >= self.brightness_pixel_percentage
    
    def _check_idle_timeout(self) -> bool:
        """
        Check if the monitoring has exceeded the maximum idle time.
        
        Returns:
            True if timeout exceeded, False otherwise
        """
        return (time.time() - self.start_time) > self.max_idle_time
    
    def start_monitoring(self):
        """
        Start the passive monitoring process.
        """
        self.logger.info(f"Starting passive monitoring in state: {self.state.value}")
        
        try:
            while not self.triggered and not self._check_idle_timeout():
                # Read a frame from the camera
                success, frame = self.camera_manager.read_frame()
                
                if not success or frame is None:
                    self.logger.warning("Failed to read frame from camera, retrying...")
                    time.sleep(0.1)
                    continue
                
                # Calculate average brightness for state transition
                avg_brightness = self._calculate_brightness(frame)
                
                if self.state == PassiveMonitorState.WAITING_FOR_DARKNESS:
                    if avg_brightness < self.darkness_threshold:
                        self.logger.info(f"Darkness detected (avg brightness: {avg_brightness:.1f}). "
                                       f"Transitioning to {PassiveMonitorState.WAITING_FOR_BRIGHTNESS.value}")
                        self.state = PassiveMonitorState.WAITING_FOR_BRIGHTNESS
                        self.bright_frame_counter = 0
                
                elif self.state == PassiveMonitorState.WAITING_FOR_BRIGHTNESS:
                    if self._is_bright_frame(frame):
                        self.bright_frame_counter += 1
                        self.logger.debug(f"Bright frame detected. Counter: {self.bright_frame_counter}/{self.trigger_frame_count}")
                        
                        if self.bright_frame_counter >= self.trigger_frame_count:
                            self.logger.info("TRIGGER! Stable brightness detected. Transitioning to triggered state.")
                            self.state = PassiveMonitorState.TRIGGERED
                            self.triggered = True
                            break
                    else:
                        # If we get a dark frame after detecting some bright frames, reset the counter
                        self.logger.debug("Brightness not stable, resetting counter.")
                        self.bright_frame_counter = 0
                
                # Small delay to control frame rate
                time.sleep(1.0 / self.fps)
        
        except KeyboardInterrupt:
            self.logger.info("Passive monitoring interrupted by user")
        except Exception as e:
            self.logger.error(f"Error in passive monitoring loop: {e}")
        finally:
            self.state = PassiveMonitorState.STOPPED
            self.stop_monitoring()
    
    def stop_monitoring(self):
        """
        Stop the passive monitoring process.
        """
        if self.camera_manager:
            self.camera_manager.release()
            self.logger.info("Camera resources released")
    
    def is_triggered(self) -> bool:
        """
        Check if the monitoring has been triggered.
        
        Returns:
            True if triggered, False otherwise
        """
        return self.triggered
    
    def get_state(self) -> PassiveMonitorState:
        """
        Get the current monitoring state.
        
        Returns:
            Current state of the passive monitor
        """
        return self.state


def brightness_check_strategy(frame: np.ndarray, config: Dict[str, Any]) -> bool:
    """
    Default brightness check strategy: checks if sufficient percentage of pixels are bright enough.
    
    Args:
        frame: Input frame to analyze
        config: Configuration dictionary containing brightness parameters
        
    Returns:
        True if brightness criteria are met, False otherwise
    """
    brightness_threshold = config.get('brightness_threshold', 220)
    pixel_percentage = config.get('brightness_pixel_percentage', 0.01)
    
    # Convert to grayscale for brightness analysis
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    
    # Count pixels above the brightness threshold
    bright_pixels = np.sum(gray_frame > brightness_threshold)
    total_pixels = gray_frame.size
    current_percentage = bright_pixels / total_pixels
    
    return current_percentage >= pixel_percentage


def motion_detection_strategy(frame: np.ndarray, prev_frame: np.ndarray, config: Dict[str, Any]) -> bool:
    """
    Motion detection strategy: checks for significant movement between frames.
    
    Args:
        frame: Current frame
        prev_frame: Previous frame for comparison
        config: Configuration dictionary containing motion parameters
        
    Returns:
        True if motion criteria are met, False otherwise
    """
    if prev_frame is None:
        return False
    
    motion_threshold = config.get('motion_threshold', 30.0)
    
    # Calculate absolute difference
    diff = cv2.absdiff(prev_frame, frame)
    
    # Convert to grayscale if needed
    if len(diff.shape) == 3:
        diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    # Calculate mean difference as motion value
    motion_value = float(np.mean(diff))
    
    return motion_value > motion_threshold


def combined_strategy(frame: np.ndarray, prev_frame: np.ndarray, config: Dict[str, Any]) -> bool:
    """
    Combined strategy: checks both brightness and motion.
    
    Args:
        frame: Current frame
        prev_frame: Previous frame for motion comparison
        config: Configuration dictionary containing all parameters
        
    Returns:
        True if both brightness and motion criteria are met, False otherwise
    """
    brightness_result = brightness_check_strategy(frame, config)
    motion_result = motion_detection_strategy(frame, prev_frame, config)
    
    return brightness_result and motion_result