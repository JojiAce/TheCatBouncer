"""
CoreML inference engine.
"""
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import platform

from src.engines.base_engine import InferenceEngine


class CoreMLEngine(InferenceEngine):
    """
    CoreML inference engine for Apple Silicon optimization.
    """
    
    def __init__(self, model_path: str, device: str = "auto", **kwargs):
        """
        Initialize the CoreML inference engine.
        Only supported on macOS with Apple Silicon.
        
        Args:
            model_path: Path to the .mlmodel file
            device: Device to run inference on ('auto', 'neural_engine', 'cpu', etc.)
            **kwargs: Additional parameters
        """
        super().__init__(model_path, device, **kwargs)
        
        # Check platform compatibility
        if platform.system() != 'Darwin':
            raise RuntimeError("CoreML engine is only supported on macOS")
        
        # Import CoreML tools
        try:
            import coremltools as ct
            from PIL import Image
        except ImportError as e:
            raise ImportError("CoreML is not installed. Please install with: pip install coremltools") from e
        
        self.ct = ct
        self.Image = Image  # Store PIL Image for later use
        
        # Determine compute units based on device parameter
        if device == 'neural_engine' or 'neural' in device.lower():
            compute_units = ct.ComputeUnit.ANE  # Apple Neural Engine
        elif device == 'cpu':
            compute_units = ct.ComputeUnit.CPU_ONLY
        elif device == 'gpu':
            compute_units = ct.ComputeUnit.CPU_AND_GPU
        else:  # auto
            compute_units = ct.ComputeUnit.ALL  # Let CoreML decide
        
        try:
            # Load the model with specified compute units
            self.model = ct.models.MLModel(
                model_path, 
                compute_units=compute_units
            )
            
            # Get model description to extract input/output info if needed
            self.model_desc = self.model._spec.description
            
            # Extract class names if available in model metadata
            self.class_names = self._extract_class_names()
            
            # Find the input name
            self.input_name = self.model_desc.input[0].name
            
            self.logger.info(f"CoreML engine initialized with model: {model_path}")
            self.logger.info(f"Using compute units: {compute_units}")
            self.logger.info(f"Input name: {self.input_name}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize CoreML engine with model {model_path}: {str(e)}") from e
    
    def _extract_class_names(self) -> List[str]:
        """
        Extract class names from CoreML model metadata.
        
        Returns:
            List of class names
        """
        try:
            # CoreML models may store class names in the specification
            # Check the output description for class labels
            for output in self.model_desc.output:
                if output.type.HasField('dictionaryType'):
                    # This is a classification output with possible class labels
                    if output.type.dictionaryType.HasField('stringKeyType'):
                        # Class names might be stored in the model's user defined metadata
                        # This is dependent on how the model was converted
                        user_defined_metadata = getattr(self.model._spec, 'userDefinedMetadata', {})
                        if 'class_labels' in user_defined_metadata:
                            labels_str = user_defined_metadata['class_labels']
                            # Parse the labels string (format may vary)
                            try:
                                # If it's a JSON string
                                import json
                                return json.loads(labels_str)
                            except json.JSONDecodeError:
                                # If it's a comma-separated string
                                return [label.strip() for label in labels_str.split(',')]
            
            # If not found in the standard location, check model metadata
            if hasattr(self.model._spec, 'description') and self.model._spec.description:
                # Look for custom metadata
                if hasattr(self.model._spec.description, 'metadata'):
                    metadata = self.model._spec.description.metadata
                    if hasattr(metadata, 'shortDescription'):
                        # Sometimes class names are in the description
                        pass  # Not a reliable source
            
            return []  # Default to empty list if not found
        except Exception as e:
            self.logger.warning(f"Could not extract class names from CoreML model metadata: {e}")
            return []
    
    def predict(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Run inference on an image using CoreML.
        
        Args:
            image: Input image as numpy array (H, W, C)
            
        Returns:
            List of inference results
        """
        try:
            # Convert numpy array to PIL Image
            # CoreML expects PIL Image as input
            if len(image.shape) == 3:
                # Convert BGR (OpenCV) to RGB if needed
                if image.shape[2] == 3:
                    image_rgb = image[:, :, ::-1]  # BGR to RGB
                else:
                    image_rgb = image
                
                # Convert to PIL Image
                pil_image = self.Image.fromarray(image_rgb.astype('uint8'), 'RGB')
            else:
                raise ValueError(f"Expected 3D image array, got shape {image.shape}")
            
            # Run inference
            results = self.model.predict({self.input_name: pil_image})
            
            # Convert results to list of numpy arrays
            # The format depends on the model, but typically returns a dict
            output_list = []
            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    output_list.append(value)
                elif isinstance(value, list):
                    # Convert list items to numpy arrays
                    output_list.extend([np.array(item) if not isinstance(item, np.ndarray) else item for item in value])
                elif hasattr(value, '__len__') and not isinstance(value, str):
                    # Convert other sequence types
                    try:
                        output_list.append(np.array(value))
                    except:
                        # If conversion fails, skip this output
                        continue
                else:
                    # Single value, convert to array
                    output_list.append(np.array([value]))
            
            return output_list
        except Exception as e:
            raise RuntimeError(f"CoreML inference failed: {str(e)}") from e
    
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
        try:
            # Create a dummy image for warmup
            dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            pil_image = self.Image.fromarray(dummy_image.astype('uint8'), 'RGB')
            
            # Run dummy inference
            _ = self.model.predict({self.input_name: pil_image})
            self.logger.info("CoreML engine warmed up successfully")
        except Exception as e:
            self.logger.error(f"Failed to warm up CoreML engine: {e}")