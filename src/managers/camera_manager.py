"""
Camera manager for cross-platform camera access.
"""
import cv2
import logging
import numpy as np
from typing import Tuple, Optional, Any
from pathlib import Path
import time

from src.interfaces.inference import CameraInterface


class CameraManager(CameraInterface):
    """
    Camera manager for cross-platform camera access.
    Implements proper error handling and supports different camera sources.
    """
    
    def __init__(self, source: Any, resolution: Tuple[int, int] = (640, 480), fps: int = 30, **kwargs):
        """
        Initialize the camera manager.
        
        Args:
            source: Camera source (index, file path, URL, etc.)
            resolution: Desired resolution as (width, height)
            fps: Desired frames per second
            **kwargs: Additional camera-specific parameters
        """
        self.source = source
        self.resolution = resolution
        self.fps = fps
        self.cap = None
        self.is_capturing = False
        
        # Initialize logging
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize camera
        self._initialize_camera()
    
    def _initialize_camera(self):
        """Initialize the camera with proper error handling."""
        try:
            # Handle different source types
            if isinstance(self.source, (int, str)) and str(self.source).isdigit():
                # Camera index
                self.cap = cv2.VideoCapture(int(self.source))
            elif isinstance(self.source, str) and Path(self.source).exists():
                # Video file path
                self.cap = cv2.VideoCapture(self.source)
            elif isinstance(self.source, str) and self.source.startswith(('rtsp://', 'http://', 'https://')):
                # RTSP or HTTP stream
                self.cap = cv2.VideoCapture(self.source)
            else:
                # Treat as camera index by default
                self.cap = cv2.VideoCapture(int(self.source))
            
            if not self.cap.isOpened():
                raise IOError(f"Cannot open camera source: {self.source}")
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Additional optimization properties
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to minimize latency
            
            self.is_capturing = True
            self.logger.info(f"Camera initialized successfully with source {self.source} at {self.resolution[0]}x{self.resolution[1]} @ {self.fps} FPS")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize camera: {e}")
            if self.cap:
                self.cap.release()
            raise
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the camera.
        
        Returns:
            Tuple of (success, frame) where success is a boolean and frame is the image array
        """
        if not self.is_capturing or not self.cap or not self.cap.isOpened():
            return False, None
        
        try:
            ret, frame = self.cap.read()
            if not ret:
                # Try to reinitialize camera if read fails
                self.logger.warning("Failed to read frame, attempting to reinitialize camera")
                self._initialize_camera()
                ret, frame = self.cap.read()
                if not ret:
                    return False, None
            
            return True, frame
        except Exception as e:
            self.logger.error(f"Error reading frame: {e}")
            return False, None
    
    def release(self):
        """
        Release the camera resources.
        """
        if self.cap:
            self.cap.release()
            self.is_capturing = False
            self.logger.info("Camera resources released")
    
    def set_resolution(self, width: int, height: int):
        """
        Set the camera resolution.
        
        Args:
            width: Width in pixels
            height: Height in pixels
        """
        if self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.resolution = (width, height)
            self.logger.info(f"Resolution set to {width}x{height}")
        else:
            self.logger.warning("Cannot set resolution: camera not initialized or already released")
    
    def is_opened(self) -> bool:
        """
        Check if the camera is opened.
        
        Returns:
            True if camera is opened, False otherwise
        """
        return self.cap and self.cap.isOpened() and self.is_capturing

    def get_fps(self) -> int:
        """
        Get the current frames per second.
        
        Returns:
            Current FPS
        """
        if self.cap and self.cap.isOpened():
            return int(self.cap.get(cv2.CAP_PROP_FPS))
        return 0

    def get_frame_size(self) -> Tuple[int, int]:
        """
        Get the current frame size.
        
        Returns:
            Tuple of (width, height)
        """
        if self.cap and self.cap.isOpened():
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (width, height)
        return (0, 0)


def preprocess_frame(frame: np.ndarray, target_resolution: Tuple[int, int], 
                    target_format: str = "rgb") -> np.ndarray:
    """
    Preprocess a frame (resize, color conversion).
    
    Args:
        frame: Input frame
        target_resolution: Target resolution (width, height)
        target_format: Target format ('rgb', 'bgr', 'gray')
        
    Returns:
        Preprocessed frame
    """
    # Resize frame if needed
    if frame.shape[:2][::-1] != target_resolution:
        frame = cv2.resize(frame, target_resolution, interpolation=cv2.INTER_AREA)
    
    # Convert color format if needed
    if target_format.lower() == "rgb":
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    elif target_format.lower() == "gray":
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Convert back to 3-channel if needed to maintain shape consistency
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    
    return frame


def validate_frame(frame: np.ndarray) -> bool:
    """
    Validate a frame.
    
    Args:
        frame: Frame to validate
        
    Returns:
        True if frame is valid, False otherwise
    """
    if frame is None:
        return False
    
    if not hasattr(frame, 'shape') or len(frame.shape) < 2:
        return False
    
    if frame.size == 0:
        return False
    
    return True


def get_camera_info(source: Any) -> Optional[dict]:
    """
    Get camera information.
    
    Args:
        source: Camera source
        
    Returns:
        Dictionary with camera information or None if not available
    """
    try:
        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            info = {
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': int(cap.get(cv2.CAP_PROP_FPS)),
                'backend': cap.getBackendName() if hasattr(cv2, 'getBackendName') else 'unknown'
            }
            
            cap.release()
            return info
        
        cap.release()
        return None
    except Exception as e:
        logging.error(f"Error getting camera info: {e}")
        return None


def test_camera_connection(source: Any, timeout: int = 5) -> bool:
    """
    Test camera connection.
    
    Args:
        source: Camera source
        timeout: Timeout in seconds
        
    Returns:
        True if camera is accessible, False otherwise
    """
    try:
        cap = cv2.VideoCapture(source)
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            ret, frame = cap.read()
            if ret and frame is not None:
                cap.release()
                return True
            time.sleep(0.1)
        
        cap.release()
        return False
    except Exception as e:
        logging.error(f"Error testing camera connection: {e}")
        return False