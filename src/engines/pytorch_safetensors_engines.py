"""
PyTorch and SafeTensors inference engines.
"""
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
import os

from src.engines.base_engine import InferenceEngine


class PyTorchEngine(InferenceEngine):
    """
    PyTorch inference engine.
    Supports NVIDIA GPU acceleration.
    """
    
    def __init__(self, model_path: str, device: str = "cpu", **kwargs):
        """
        Initialize the PyTorch inference engine.
        
        Args:
            model_path: Path to the .pt model file
            device: Device to run inference on ('cpu', 'cuda:0', 'cuda:1', etc.)
            **kwargs: Additional parameters
        """
        super().__init__(model_path, device, **kwargs)
        
        # Import PyTorch
        try:
            import torch
        except ImportError as e:
            raise ImportError("PyTorch is not installed. Please install with: pip install torch") from e
        
        self.torch = torch  # Store reference for later use
        
        # Map device string to PyTorch device
        if device.startswith("cuda") and torch.cuda.is_available():
            self.device = torch.device(device)
        else:
            self.device = torch.device('cpu')
        
        try:
            # Load the model
            self.model = torch.load(model_path, map_location=self.device)
            
            # Set model to evaluation mode (important for inference)
            self.model.eval()
            
            # Move model to specified device
            self.model.to(self.device)
            
            # Extract class names from model if available (common for Ultralytics models)
            if hasattr(self.model, 'names') and self.model.names:
                self.class_names = self.model.names
            elif 'names' in kwargs:
                self.class_names = kwargs['names']
            else:
                self.class_names = []
            
            self.logger.info(f"PyTorch engine initialized with model: {model_path}")
            self.logger.info(f"Using device: {self.device}")
            self.logger.info(f"Model on device: {next(self.model.parameters()).device}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize PyTorch engine with model {model_path}: {str(e)}") from e
    
    def predict(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Run inference on an image using PyTorch.
        
        Args:
            image: Input image as numpy array (H, W, C)
            
        Returns:
            List of inference results
        """
        with torch.no_grad():  # Disable gradient computation for inference
            # Convert numpy array to PyTorch tensor
            # PyTorch expects (N, C, H, W) format
            if len(image.shape) == 3:  # Add batch dimension
                image = np.expand_dims(image, axis=0)
            
            # Transpose from (N, H, W, C) to (N, C, H, W) if needed
            if image.shape[-1] in [3, 1]:  # Channels are last
                image = np.transpose(image, (0, 3, 1, 2))
            
            # Convert to tensor and move to device
            tensor = torch.from_numpy(image).float().to(self.device)
            
            # Normalize if needed (common for PyTorch models)
            # This depends on the specific model, but many expect values in [0, 1] or normalized
            if tensor.max() > 1.0:
                tensor = tensor / 255.0  # Normalize from [0, 255] to [0, 1]
            
            # Run inference
            try:
                results = self.model(tensor)
                
                # Convert results back to numpy arrays
                if isinstance(results, torch.Tensor):
                    # Single output
                    return [results.cpu().numpy()]
                elif isinstance(results, (list, tuple)):
                    # Multiple outputs
                    return [r.cpu().numpy() if isinstance(r, torch.Tensor) else r for r in results]
                elif isinstance(results, dict):
                    # Dictionary output (like in some detection models)
                    return [v.cpu().numpy() if isinstance(v, torch.Tensor) else v for v in results.values()]
                else:
                    # Unknown format, try to convert to numpy
                    return [np.array(results)]
                    
            except Exception as e:
                raise RuntimeError(f"PyTorch inference failed: {str(e)}") from e
    
    def get_class_names(self) -> List[str]:
        """
        Get the class names for the model.
        
        Returns:
            List of class names
        """
        return self.class_names
    
    def warm_up(self):
        """
        Warm up the model by running a dummy inference.
        """
        # Create a dummy input with a reasonable size
        # Use model's expected input dimensions if possible, otherwise default
        dummy_shape = (1, 3, 640, 640)  # Common for detection models
        dummy_input = torch.randn(dummy_shape).to(self.device)
        
        try:
            with torch.no_grad():
                _ = self.model(dummy_input)
            self.logger.info("PyTorch engine warmed up successfully")
        except Exception as e:
            self.logger.error(f"Failed to warm up PyTorch engine: {e}")


class SafeTensorsEngine(InferenceEngine):
    """
    SafeTensors inference engine.
    Safer and faster loading than PyTorch.
    """
    
    def __init__(self, model_path: str, device: str = "cpu", **kwargs):
        """
        Initialize the SafeTensors inference engine.
        
        Args:
            model_path: Path to the .safetensors model file
            device: Device to run inference on ('cpu', 'cuda:0', 'cuda:1', etc.)
            **kwargs: Additional parameters
        """
        super().__init__(model_path, device, **kwargs)
        
        # Import required libraries
        try:
            import torch
            from safetensors.torch import load_file
        except ImportError as e:
            raise ImportError("PyTorch and safetensors are not installed. Please install with: pip install torch safetensors") from e
        
        self.torch = torch  # Store reference for later use
        self.load_file = load_file  # Store reference to load function
        
        # Map device string to PyTorch device
        if device.startswith("cuda") and torch.cuda.is_available():
            self.device = torch.device(device)
        else:
            self.device = torch.device('cpu')
        
        # Load the model weights from the SafeTensors file
        try:
            # Load the state dict from the safetensors file
            state_dict = self.load_file(model_path)
            
            # Create the model architecture (this is the tricky part - we need the model class)
            # Since the model class isn't stored in the safetensors file, we need it to be provided
            if 'model_class' not in kwargs:
                raise ValueError("model_class must be provided in kwargs for SafeTensors engine")
            
            model_class = kwargs['model_class']
            self.model = model_class()  # Initialize the model
            
            # Load the state dict into the model
            self.model.load_state_dict(state_dict)
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Move model to specified device
            self.model.to(self.device)
            
            # Extract class names if provided
            if 'names' in kwargs:
                self.class_names = kwargs['names']
            else:
                self.class_names = []
            
            self.logger.info(f"SafeTensors engine initialized with model: {model_path}")
            self.logger.info(f"Using device: {self.device}")
            self.logger.info(f"Model on device: {next(self.model.parameters()).device}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize SafeTensors engine with model {model_path}: {str(e)}") from e
    
    def predict(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Run inference on an image using SafeTensors model.
        
        Args:
            image: Input image as numpy array (H, W, C)
            
        Returns:
            List of inference results
        """
        with torch.no_grad():  # Disable gradient computation for inference
            # Convert numpy array to PyTorch tensor
            # PyTorch expects (N, C, H, W) format
            if len(image.shape) == 3:  # Add batch dimension
                image = np.expand_dims(image, axis=0)
            
            # Transpose from (N, H, W, C) to (N, C, H, W) if needed
            if image.shape[-1] in [3, 1]:  # Channels are last
                image = np.transpose(image, (0, 3, 1, 2))
            
            # Convert to tensor and move to device
            tensor = torch.from_numpy(image).float().to(self.device)
            
            # Normalize if needed (common for PyTorch models)
            if tensor.max() > 1.0:
                tensor = tensor / 255.0  # Normalize from [0, 255] to [0, 1]
            
            # Run inference
            try:
                results = self.model(tensor)
                
                # Convert results back to numpy arrays
                if isinstance(results, torch.Tensor):
                    # Single output
                    return [results.cpu().numpy()]
                elif isinstance(results, (list, tuple)):
                    # Multiple outputs
                    return [r.cpu().numpy() if isinstance(r, torch.Tensor) else r for r in results]
                elif isinstance(results, dict):
                    # Dictionary output (like in some detection models)
                    return [v.cpu().numpy() if isinstance(v, torch.Tensor) else v for v in results.values()]
                else:
                    # Unknown format, try to convert to numpy
                    return [np.array(results)]
                    
            except Exception as e:
                raise RuntimeError(f"SafeTensors inference failed: {str(e)}") from e
    
    def get_class_names(self) -> List[str]:
        """
        Get the class names for the model.
        
        Returns:
            List of class names
        """
        return self.class_names
    
    def warm_up(self):
        """
        Warm up the model by running a dummy inference.
        """
        # Create a dummy input with a reasonable size
        dummy_shape = (1, 3, 640, 640)  # Common for detection models
        dummy_input = torch.randn(dummy_shape).to(self.device)
        
        try:
            with torch.no_grad():
                _ = self.model(dummy_input)
            self.logger.info("SafeTensors engine warmed up successfully")
        except Exception as e:
            self.logger.error(f"Failed to warm up SafeTensors engine: {e}")