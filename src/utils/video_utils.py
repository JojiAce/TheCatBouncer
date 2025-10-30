"""
Video frame processing utilities.
"""
import cv2
import numpy as np
from typing import Tuple, Optional, List
import logging
import time
from pathlib import Path


def resize_frame(frame: np.ndarray, target_size: Tuple[int, int], 
                 interpolation: int = cv2.INTER_AREA) -> np.ndarray:
    """
    Resize a frame to the target size.
    
    Args:
        frame: Input frame
        target_size: Target size as (width, height)
        interpolation: Interpolation method
        
    Returns:
        Resized frame
    """
    return cv2.resize(frame, target_size, interpolation=interpolation)


def convert_color_space(frame: np.ndarray, target_format: str) -> np.ndarray:
    """
    Convert frame to target color space.
    
    Args:
        frame: Input frame
        target_format: Target format ('RGB', 'BGR', 'GRAY', 'HSV')
        
    Returns:
        Converted frame
    """
    if target_format.upper() == 'RGB':
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    elif target_format.upper() == 'GRAY':
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif target_format.upper() == 'HSV':
        return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    else:
        return frame  # Return as is if format not recognized


def normalize_frame(frame: np.ndarray, target_range: Tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
    """
    Normalize frame pixel values to target range.
    
    Args:
        frame: Input frame
        target_range: Target range as (min, max)
        
    Returns:
        Normalized frame
    """
    min_val, max_val = target_range
    frame_min = frame.min()
    frame_max = frame.max()
    
    if frame_max != frame_min:
        normalized = (frame - frame_min) / (frame_max - frame_min)
        return normalized * (max_val - min_val) + min_val
    else:
        # Return frame filled with minimum value if it's constant
        return np.full(frame.shape, min_val, dtype=frame.dtype)


def frame_brightness(frame: np.ndarray) -> float:
    """
    Calculate the average brightness of a frame.
    
    Args:
        frame: Input frame
        
    Returns:
        Average brightness value
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    return float(np.mean(gray))


def frame_contrast(frame: np.ndarray) -> float:
    """
    Calculate the contrast of a frame (std deviation of pixel values).
    
    Args:
        frame: Input frame
        
    Returns:
        Contrast value (std deviation)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    return float(np.std(gray))


def detect_motion(prev_frame: np.ndarray, curr_frame: np.ndarray, 
                  threshold: float = 30.0) -> Tuple[bool, float]:
    """
    Detect motion between two frames.
    
    Args:
        prev_frame: Previous frame
        curr_frame: Current frame
        threshold: Motion threshold
        
    Returns:
        Tuple of (motion_detected, motion_value)
    """
    # Convert to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY) if len(prev_frame.shape) == 3 else prev_frame
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY) if len(curr_frame.shape) == 3 else curr_frame
    
    # Calculate absolute difference
    diff = cv2.absdiff(prev_gray, curr_gray)
    
    # Calculate mean difference as motion value
    motion_value = float(np.mean(diff))
    
    # Check if motion exceeds threshold
    motion_detected = motion_value > threshold
    
    return motion_detected, motion_value


def validate_frame_size(frame: np.ndarray, min_size: Tuple[int, int] = (1, 1)) -> bool:
    """
    Validate frame size.
    
    Args:
        frame: Input frame
        min_size: Minimum required size (width, height)
        
    Returns:
        True if frame size is valid, False otherwise
    """
    if frame is None:
        return False
    
    if len(frame.shape) < 2:
        return False
    
    height, width = frame.shape[:2]
    min_width, min_height = min_size
    
    return width >= min_width and height >= min_height


def crop_bounding_box(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Crop a bounding box from the frame.
    
    Args:
        frame: Input frame
        bbox: Bounding box as (x1, y1, x2, y2)
        
    Returns:
        Cropped frame
    """
    x1, y1, x2, y2 = bbox
    return frame[y1:y2, x1:x2]


def add_frame_timestamp(frame: np.ndarray, timestamp: Optional[str] = None) -> np.ndarray:
    """
    Add timestamp to frame.
    
    Args:
        frame: Input frame
        timestamp: Timestamp string. If None, uses current time.
        
    Returns:
        Frame with timestamp overlay
    """
    import datetime
    
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    frame_with_timestamp = frame.copy()
    cv2.putText(frame_with_timestamp, timestamp, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    return frame_with_timestamp


class FrameBuffer:
    """
    Circular buffer for frames with performance optimization.
    """
    
    def __init__(self, capacity: int = 30):
        """
        Initialize the frame buffer.
        
        Args:
            capacity: Maximum number of frames to store
        """
        self.capacity = capacity
        self.buffer = []
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def add_frame(self, frame: np.ndarray):
        """
        Add a frame to the buffer.
        
        Args:
            frame: Frame to add
        """
        if not validate_frame_size(frame):
            self.logger.warning("Invalid frame size, skipping")
            return
        
        # If buffer is full, remove oldest frame
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        
        # Add new frame (make a copy to avoid reference issues)
        self.buffer.append(frame.copy())
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """
        Get the latest frame from the buffer.
        
        Returns:
            Latest frame or None if buffer is empty
        """
        if self.buffer:
            return self.buffer[-1]
        return None
    
    def get_frames(self) -> List[np.ndarray]:
        """
        Get all frames in the buffer.
        
        Returns:
            List of frames
        """
        return self.buffer.copy()
    
    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()
    
    def size(self) -> int:
        """Get the current buffer size."""
        return len(self.buffer)
    
    def is_full(self) -> bool:
        """Check if the buffer is full."""
        return len(self.buffer) >= self.capacity


class VideoPerformanceMonitor:
    """
    Monitor video processing performance metrics.
    """
    
    def __init__(self):
        """Initialize performance monitor."""
        self.frame_times = []
        self.fps_history = []
        self.start_time = time.time()
        self.frame_count = 0
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def start_frame_timing(self) -> float:
        """
        Start timing for a frame.
        
        Returns:
            Start timestamp
        """
        return time.time()
    
    def end_frame_timing(self, start_time: float):
        """
        End timing for a frame and update metrics.
        
        Args:
            start_time: Start timestamp from start_frame_timing
        """
        end_time = time.time()
        frame_time = end_time - start_time
        self.frame_times.append(frame_time)
        self.frame_count += 1
        
        # Calculate FPS
        if frame_time > 0:
            fps = 1.0 / frame_time
            self.fps_history.append(fps)
            
            # Keep only the last 100 FPS values for memory efficiency
            if len(self.fps_history) > 100:
                self.fps_history.pop(0)
    
    def get_current_fps(self) -> float:
        """
        Get the current FPS based on recent measurements.
        
        Returns:
            Current FPS
        """
        if self.fps_history:
            return sum(self.fps_history) / len(self.fps_history)
        return 0.0
    
    def get_average_frame_time(self) -> float:
        """
        Get average frame processing time.
        
        Returns:
            Average frame time in seconds
        """
        if self.frame_times:
            return sum(self.frame_times) / len(self.frame_times)
        return 0.0
    
    def get_total_frames_processed(self) -> int:
        """
        Get total number of frames processed.
        
        Returns:
            Total frame count
        """
        return self.frame_count
    
    def get_uptime(self) -> float:
        """
        Get total uptime in seconds.
        
        Returns:
            Uptime in seconds
        """
        return time.time() - self.start_time
    
    def reset(self):
        """Reset the performance monitor."""
        self.frame_times.clear()
        self.fps_history.clear()
        self.start_time = time.time()
        self.frame_count = 0


def save_frame(frame: np.ndarray, path: str, quality: int = 95):
    """
    Save a frame to file.
    
    Args:
        frame: Frame to save
        path: Path to save the frame
        quality: JPEG quality (1-100)
    """
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    # Set JPEG quality
    params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    
    success = cv2.imwrite(str(path_obj), frame, params)
    if not success:
        raise IOError(f"Failed to save frame to {path}")


def batch_preprocess_frames(frames: List[np.ndarray], target_size: Optional[Tuple[int, int]] = None,
                           target_format: Optional[str] = None) -> List[np.ndarray]:
    """
    Batch preprocess a list of frames.
    
    Args:
        frames: List of frames to preprocess
        target_size: Target size as (width, height)
        target_format: Target format ('RGB', 'BGR', 'GRAY', 'HSV')
        
    Returns:
        List of preprocessed frames
    """
    processed_frames = []
    
    for frame in frames:
        processed_frame = frame
        
        if target_size is not None:
            processed_frame = resize_frame(processed_frame, target_size)
        
        if target_format is not None:
            processed_frame = convert_color_space(processed_frame, target_format)
        
        processed_frames.append(processed_frame)
    
    return processed_frames