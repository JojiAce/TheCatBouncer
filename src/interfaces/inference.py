"""
Abstract base classes for inference engines.
"""
from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional, Tuple
import numpy as np


class InferenceEngine(ABC):
    """
    Abstract base class for all AI inference engines.
    Defines the common interface for different AI backend implementations.
    """
    
    @abstractmethod
    def __init__(self, model_path: str, device: str = "cpu", **kwargs):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to the model file
            device: Device to run inference on (cpu, gpu, cuda:0, etc.)
            **kwargs: Additional engine-specific parameters
        """
        pass

    @abstractmethod
    def predict(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Run inference on an image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of inference results
        """
        pass

    @abstractmethod
    def get_class_names(self) -> List[str]:
        """
        Get the class names for the model.
        
        Returns:
            List of class names
        """
        pass

    @abstractmethod
    def warm_up(self):
        """
        Warm up the model by running a dummy inference.
        """
        pass


class CameraInterface(ABC):
    """
    Abstract base class for camera access and video processing.
    """
    
    @abstractmethod
    def __init__(self, source: Any, **kwargs):
        """
        Initialize the camera interface.
        
        Args:
            source: Camera source (index, file path, URL, etc.)
            **kwargs: Additional camera-specific parameters
        """
        pass

    @abstractmethod
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the camera.
        
        Returns:
            Tuple of (success, frame) where success is a boolean and frame is the image array
        """
        pass

    @abstractmethod
    def release(self):
        """
        Release the camera resources.
        """
        pass
    
    @abstractmethod
    def set_resolution(self, width: int, height: int):
        """
        Set the camera resolution.
        
        Args:
            width: Width in pixels
            height: Height in pixels
        """
        pass

    @abstractmethod
    def is_opened(self) -> bool:
        """
        Check if the camera is opened.
        
        Returns:
            True if camera is opened, False otherwise
        """
        pass