"""
ONNX and OpenVINO inference engines.
"""
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
import json
import os

from src.engines.base_engine import InferenceEngine


class OnnxEngine(InferenceEngine):
    """
    ONNX Runtime inference engine.
    Supports NVIDIA CUDA, AMD/Intel GPU, and CPU.
    """
    
    def __init__(self, model_path: str, device: str = "cpu", **kwargs):
        """
        Initialize the ONNX inference engine.
        
        Args:
            model_path: Path to the .onnx model file
            device: Device to run inference on ('cpu', 'cuda:0', 'cuda:1', etc.)
            **kwargs: Additional parameters
        """
        super().__init__(model_path, device, **kwargs)
        
        # Import ONNX Runtime
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError("ONNX Runtime is not installed. Please install with: pip install onnxruntime or onnxruntime-gpu") from e
        
        # Set execution providers based on device
        if device.startswith("cuda"):
            # Extract GPU ID from device string like "cuda:0"
            gpu_id = int(device.split(":")[1]) if ":" in device else 0
            providers = [
                ('CUDAExecutionProvider', {
                    'device_id': gpu_id,
                }),
                'CPUExecutionProvider'
            ]
        else:
            # CPU execution
            providers = ['CPUExecutionProvider']
        
        # Create inference session
        try:
            self.session = ort.InferenceSession(
                model_path,
                providers=providers,
                # Additional optimization options
                sess_options=ort.SessionOptions()
            )
            
            # Get input and output names
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            # Extract class names from model metadata if available
            self.class_names = self._extract_class_names()
            
            self.logger.info(f"ONNX engine initialized with model: {model_path}")
            self.logger.info(f"Using device: {device}")
            self.logger.info(f"Input shape: {self.input_shape}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize ONNX engine with model {model_path}: {str(e)}") from e
    
    def _extract_class_names(self) -> List[str]:
        """
        Extract class names from model metadata.
        
        Returns:
            List of class names
        """
        try:
            # Try to get class names from model metadata
            meta = self.session.get_modelmeta()
            
            # Look for class names in custom metadata
            if 'names' in meta.custom_metadata_map:
                names_str = meta.custom_metadata_map['names']
                # Parse the names string (format may vary depending on how model was exported)
                try:
                    # If it's a JSON string
                    names_dict = json.loads(names_str)
                    return list(names_dict.values()) if isinstance(names_dict, dict) else names_dict
                except json.JSONDecodeError:
                    # If it's not JSON, try to parse as a simple string list
                    return [name.strip() for name in names_str.split(',')]
            
            # If not found in metadata, try other common locations
            # This varies by model export method, so we have to try different approaches
            return []  # Default to empty list if not found
            
        except Exception as e:
            self.logger.warning(f"Could not extract class names from model metadata: {e}")
            return []
    
    def predict(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Run inference on an image using ONNX Runtime.
        
        Args:
            image: Input image as numpy array (H, W, C)
            
        Returns:
            List of inference results
        """
        # Prepare input - ensure correct shape and format
        # ONNX models expect (N, C, H, W) format
        if len(image.shape) == 3:  # Add batch dimension
            image = np.expand_dims(image, axis=0)
        
        # Transpose from (N, H, W, C) to (N, C, H, W) if needed
        if image.shape[-1] in [3, 1]:  # Channels are last
            image = np.transpose(image, (0, 3, 1, 2))
        
        # Convert to float32 if needed
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        
        # Run inference
        try:
            results = self.session.run(self.output_names, {self.input_name: image})
            return results
        except Exception as e:
            raise RuntimeError(f"ONNX inference failed: {str(e)}") from e
    
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
        # Create a dummy input with the expected input shape
        # Replace 'N' with 1 for batch size, keep other dimensions
        input_shape = list(self.input_shape)
        if input_shape[0] == 'N' or input_shape[0] is None:
            input_shape[0] = 1
        
        # For ONNX models, input is typically (N, C, H, W)
        if len(input_shape) == 4:
            dummy_input = np.random.random(input_shape).astype(np.float32)
        else:
            # If shape is different, create a simple random array
            dummy_input = np.random.random(input_shape).astype(np.float32)
        
        # Run a dummy inference
        try:
            _ = self.session.run(self.output_names, {self.input_name: dummy_input})
            self.logger.info("ONNX engine warmed up successfully")
        except Exception as e:
            self.logger.error(f"Failed to warm up ONNX engine: {e}")


class OpenVinoEngine(InferenceEngine):
    """
    OpenVINO inference engine.
    Supports CPU optimization and integrated GPU (iGPU).
    """
    
    def __init__(self, model_path: str, device: str = "CPU", **kwargs):
        """
        Initialize the OpenVINO inference engine.
        
        Args:
            model_path: Path to the OpenVINO model (directory containing .xml and .bin files)
                        or path to .xml file
            device: Device to run inference on ('CPU', 'GPU', 'AUTO', etc.)
            **kwargs: Additional parameters
        """
        super().__init__(model_path, device, **kwargs)
        
        # Import OpenVINO
        try:
            from openvino.runtime import Core
        except ImportError as e:
            raise ImportError("OpenVINO is not installed. Please install with: pip install openvino") from e
        
        # Initialize OpenVINO Core
        self.core = Core()
        
        # Load the model - model_path can be directory or .xml file
        xml_path = model_path
        if os.path.isdir(model_path):
            # If model_path is a directory, look for .xml file inside
            for file in os.listdir(model_path):
                if file.endswith('.xml'):
                    xml_path = os.path.join(model_path, file)
                    break
        
        if not xml_path.endswith('.xml'):
            raise ValueError(f"Model path should point to OpenVINO .xml file or directory containing it, got: {model_path}")
        
        try:
            # Read the model
            self.model = self.core.read_model(model=xml_path)
            
            # Compile the model for the specified device
            self.compiled_model = self.core.compile_model(
                model=self.model,
                device_name=device.upper()
            )
            
            # Get input and output information
            self.input_layer = self.compiled_model.input(0)
            self.output_layers = list(self.compiled_model.outputs)
            
            # Extract class names if available in model metadata
            self.class_names = self._extract_class_names()
            
            self.logger.info(f"OpenVINO engine initialized with model: {xml_path}")
            self.logger.info(f"Using device: {device}")
            self.logger.info(f"Model inputs: {self.input_layer.shape}")
            self.logger.info(f"Model outputs: {[out.shape for out in self.output_layers]}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenVINO engine with model {xml_path}: {str(e)}") from e
    
    def _extract_class_names(self) -> List[str]:
        """
        Extract class names from OpenVINO model metadata.
        
        Returns:
            List of class names
        """
        try:
            # For OpenVINO models, we often have to extract metadata differently
            # Check if model has metadata about class names
            if hasattr(self.model, 'get_rt_info'):
                rt_info = self.model.get_rt_info()
                # This varies by how the model was exported, so we check common keys
                for key in ['names', 'class_names', 'classes']:
                    if key in rt_info:
                        names = rt_info[key]
                        if isinstance(names, list):
                            return names
                        elif isinstance(names, dict):
                            return list(names.values())
            
            return []  # Default to empty list if not found
        except Exception as e:
            self.logger.warning(f"Could not extract class names from OpenVINO model metadata: {e}")
            return []
    
    def predict(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Run inference on an image using OpenVINO.
        
        Args:
            image: Input image as numpy array (H, W, C)
            
        Returns:
            List of inference results
        """
        # Prepare input - ensure correct shape and format
        # OpenVINO models typically expect (N, C, H, W) format
        if len(image.shape) == 3:  # Add batch dimension
            image = np.expand_dims(image, axis=0)
        
        # Transpose from (N, H, W, C) to (N, C, H, W) if needed
        if image.shape[-1] in [3, 1]:  # Channels are last
            image = np.transpose(image, (0, 3, 1, 2))
        
        # Convert to float32 if needed
        if image.dtype != np.float32:
            image = image.astype(np.float32)
            
        # Run inference
        try:
            results = self.compiled_model(image)
            # Convert to list of numpy arrays
            output_results = [results[out] for out in self.output_layers]
            return output_results
        except Exception as e:
            raise RuntimeError(f"OpenVINO inference failed: {str(e)}") from e
    
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
        # Create a dummy input with the expected input shape
        input_shape = list(self.input_layer.shape)
        if input_shape[0] == -1:  # Dynamic batch size
            input_shape[0] = 1
        
        # Generate random input in the expected shape
        dummy_input = np.random.random(input_shape).astype(np.float32)
        
        # Run a dummy inference
        try:
            _ = self.compiled_model(dummy_input)
            self.logger.info("OpenVINO engine warmed up successfully")
        except Exception as e:
            self.logger.error(f"Failed to warm up OpenVINO engine: {e}")