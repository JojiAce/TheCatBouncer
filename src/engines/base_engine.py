"""
Inference engine implementations for different AI backends.
"""
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
import platform
import subprocess


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
        self.model_path = model_path
        self.device = device
        self.logger = logging.getLogger(self.__class__.__name__)
        self.class_names = []

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


class HardwareDetector:
    """
    Detects hardware capabilities and available AI backends.
    """
    
    @staticmethod
    def detect_hardware() -> Dict[str, Any]:
        """
        Detect hardware capabilities and available backends.
        
        Returns:
            Dictionary with hardware information
        """
        hardware_info = {
            'platform': platform.system(),
            'processor': platform.processor(),
            'available_backends': []
        }
        
        # Check for NVIDIA GPU (CUDA)
        if HardwareDetector._has_nvidia_gpu():
            hardware_info['available_backends'].append('cuda')
        
        # Check for AMD GPU
        if HardwareDetector._has_amd_gpu():
            hardware_info['available_backends'].append('amd')
        
        # Check for Intel integrated GPU
        if HardwareDetector._has_intel_gpu():
            hardware_info['available_backends'].append('intel_gpu')
        
        # Check if Apple Silicon (for CoreML)
        if platform.machine() in ['arm64', 'aarch64'] and platform.system() == 'Darwin':
            hardware_info['available_backends'].append('apple_silicon')
        
        # Check for various AI libraries
        if HardwareDetector._is_onnxruntime_available():
            hardware_info['available_backends'].append('onnxruntime')
        
        if HardwareDetector._is_openvino_available():
            hardware_info['available_backends'].append('openvino')
        
        if HardwareDetector._is_pytorch_available():
            hardware_info['available_backends'].append('pytorch')
        
        if HardwareDetector._is_safetensors_available():
            hardware_info['available_backends'].append('safetensors')
        
        if HardwareDetector._is_coreml_available():
            hardware_info['available_backends'].append('coreml')
        
        return hardware_info
    
    @staticmethod
    def _has_nvidia_gpu() -> bool:
        """Check if NVIDIA GPU is available."""
        try:
            result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    @staticmethod
    def _has_amd_gpu() -> bool:
        """Check if AMD GPU is available (basic check)."""
        try:
            # On Linux, check for AMD GPUs
            if platform.system() == 'Linux':
                result = subprocess.run(['lspci'], capture_output=True, text=True)
                return 'AMD' in result.stdout or 'Advanced Micro Devices' in result.stdout
            return False
        except:
            return False
    
    @staticmethod
    def _has_intel_gpu() -> bool:
        """Check if Intel iGPU is available."""
        try:
            if platform.system() == 'Linux':
                result = subprocess.run(['lspci'], capture_output=True, text=True)
                return 'Intel' in result.stdout and 'VGA' in result.stdout
            elif platform.system() == 'Darwin':  # macOS
                result = subprocess.run(['system_profiler', 'SPDisplaysDataType'], capture_output=True, text=True)
                return 'Intel' in result.stdout
            return False
        except:
            return False
    
    @staticmethod
    def _is_onnxruntime_available() -> bool:
        """Check if ONNX Runtime is available."""
        try:
            import onnxruntime
            return True
        except ImportError:
            return False
    
    @staticmethod
    def _is_openvino_available() -> bool:
        """Check if OpenVINO is available."""
        try:
            from openvino.runtime import Core
            return True
        except ImportError:
            return False
    
    @staticmethod
    def _is_pytorch_available() -> bool:
        """Check if PyTorch is available."""
        try:
            import torch
            return True
        except ImportError:
            return False
    
    @staticmethod
    def _is_safetensors_available() -> bool:
        """Check if safetensors is available."""
        try:
            from safetensors.torch import load_file
            return True
        except ImportError:
            return False
    
    @staticmethod
    def _is_coreml_available() -> bool:
        """Check if CoreML is available (macOS only)."""
        if platform.system() != 'Darwin':
            return False
        
        try:
            import coremltools
            return True
        except ImportError:
            return False


def get_optimal_engine_for_hardware(model_paths: List[str], preferred_device: str = "auto") -> Tuple[str, str]:
    """
    Determine the optimal engine and device based on hardware capabilities.
    
    Args:
        model_paths: List of available model paths
        preferred_device: Preferred device ('auto', 'cpu', 'gpu', 'cuda', etc.)
        
    Returns:
        Tuple of (engine_name, device) to use
    """
    hardware_info = HardwareDetector.detect_hardware()
    available_backends = hardware_info['available_backends']
    
    # If user specified a specific device, honor it if possible
    if preferred_device != 'auto':
        # Determine engine based on model format
        for path in model_paths:
            if path.endswith('.onnx'):
                return 'onnx', preferred_device
            elif path.endswith('.xml'):
                return 'openvino', preferred_device
            elif path.endswith('.pt'):
                return 'pytorch', preferred_device
            elif path.endswith('.safetensors'):
                return 'safetensors', preferred_device
            elif path.endswith('.mlmodel'):
                return 'coreml', preferred_device
    
    # Auto-detect best backend based on hardware
    if 'apple_silicon' in available_backends and any(p.endswith('.mlmodel') for p in model_paths):
        # Use CoreML on Apple Silicon
        return 'coreml', 'neural_engine'
    elif 'cuda' in available_backends and any(p.endswith('.onnx') for p in model_paths):
        # Use ONNX with CUDA
        return 'onnx', 'cuda:0'
    elif 'nvidia_gpu' in available_backends and any(p.endswith('.pt') for p in model_paths):
        # Use PyTorch with CUDA
        return 'pytorch', 'cuda:0'
    elif 'intel_gpu' in available_backends and any(p.endswith('.xml') for p in model_paths):
        # Use OpenVINO with iGPU
        return 'openvino', 'GPU'
    elif any(p.endswith('.xml') for p in model_paths):
        # Use OpenVINO with CPU if available
        return 'openvino', 'CPU'
    elif any(p.endswith('.onnx') for p in model_paths):
        # Use ONNX with CPU as fallback
        return 'onnx', 'cpu'
    elif any(p.endswith('.pt') for p in model_paths):
        # Use PyTorch with CPU as fallback
        return 'pytorch', 'cpu'
    elif any(p.endswith('.safetensors') for p in model_paths):
        # Use safetensors with CPU as fallback
        return 'safetensors', 'cpu'
    else:
        # No compatible models found
        raise ValueError("No compatible model formats found for available hardware")