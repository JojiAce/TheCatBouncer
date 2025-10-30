"""
Inference engine factory and model format conversion utilities.
"""
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple, Callable
from pathlib import Path
import platform

from src.engines.base_engine import InferenceEngine, get_optimal_engine_for_hardware


def create_inference_engine(
    model_paths: List[str],
    device: str = "auto",
    engine_type: Optional[str] = None,
    safetensors_options: Optional[Dict[str, Any]] = None,
) -> Tuple[InferenceEngine, List[str]]:
    """
    Factory function to create the appropriate inference engine based on model path and hardware.

    Args:
        model_paths: List of available model paths
        device: Device to run inference on ('auto', 'cpu', 'gpu', 'cuda', etc.)
        engine_type: Specific engine type to use (optional, overrides auto-detection)
        safetensors_options: Additional keyword arguments forwarded to ``SafeTensorsEngine``
            (e.g. ``{"model_builder": callable}``).

    Returns:
        Tuple of (engine_instance, class_names)
    """
    # Import engines inside the function to avoid circular imports
    from src.engines.onnx_openvino_engines import OnnxEngine, OpenVinoEngine
    from src.engines.pytorch_safetensors_engines import PyTorchEngine, SafeTensorsEngine
    from src.engines.coreml_engine import CoreMLEngine
    
    if engine_type is None:
        # Auto-detect the best engine and device based on available models and hardware
        engine_type, device = get_optimal_engine_for_hardware(model_paths, device)
    
    # Find the appropriate model path for the selected engine type
    model_path = None
    for path in model_paths:
        path_obj = Path(path)
        if (engine_type == 'onnx' and path_obj.suffix.lower() == '.onnx') or \
           (engine_type == 'openvino' and (path_obj.suffix.lower() == '.xml' or path_obj.is_dir())) or \
           (engine_type == 'pytorch' and path_obj.suffix.lower() == '.pt') or \
           (engine_type == 'safetensors' and path_obj.suffix.lower() == '.safetensors') or \
           (engine_type == 'coreml' and path_obj.suffix.lower() == '.mlmodel'):
            model_path = path
            break
    
    if model_path is None:
        raise ValueError(f"No compatible model found for engine type '{engine_type}' in provided paths: {model_paths}")
    
    # Prepare engine specific options
    safetensors_options = safetensors_options or {}

    # Create the appropriate engine
    if engine_type == 'onnx':
        engine = OnnxEngine(model_path, device)
    elif engine_type == 'openvino':
        engine = OpenVinoEngine(model_path, device)
    elif engine_type == 'pytorch':
        engine = PyTorchEngine(model_path, device)
    elif engine_type == 'safetensors':
        if not safetensors_options:
            raise ValueError(
                "SafeTensors engine requires 'safetensors_options' specifying how to build the model architecture."
            )
        engine = SafeTensorsEngine(model_path, device, **safetensors_options)
    elif engine_type == 'coreml':
        engine = CoreMLEngine(model_path, device)
    else:
        raise ValueError(f"Unknown engine type: {engine_type}")
    
    # Run warmup
    try:
        engine.warm_up()
    except Exception as e:
        logging.warning(f"Engine warmup failed: {e}")
    
    return engine, engine.get_class_names()


# For the SafeTensors engine to work properly, we'd need to also create a utility module
# that can handle model architecture loading, but for now we'll just define a fallback:
def create_safetensors_engine_with_architecture(model_path: str, architecture_func: Callable, device: str = "cpu", names: Optional[List[str]] = None) -> Tuple[InferenceEngine, List[str]]:
    """
    Create a SafeTensors engine with explicit model architecture.
    
    Args:
        model_path: Path to the .safetensors model file
        architecture_func: Function that returns the model architecture (class or instance)
        device: Device to run inference on
        names: Class names for the model
        
    Returns:
        Tuple of (engine_instance, class_names)
    """
    from src.engines.pytorch_safetensors_engines import SafeTensorsEngine
    
    # Get the model class from the architecture function
    model_class = architecture_func()
    
    # Create engine with model class
    engine = SafeTensorsEngine(
        model_path, 
        device, 
        model_class=model_class, 
        names=names or []
    )
    
    # Run warmup
    try:
        engine.warm_up()
    except Exception as e:
        logging.warning(f"Engine warmup failed: {e}")
    
    return engine, engine.get_class_names()


def convert_model_format(source_path: str, target_format: str, target_path: str) -> bool:
    """
    Convert model from one format to another.
    
    Args:
        source_path: Path to source model
        target_format: Target format ('onnx', 'openvino', 'safetensors', 'coreml')
        target_path: Path to save converted model
        
    Returns:
        True if conversion was successful, False otherwise
    """
    source_path_obj = Path(source_path)
    target_path_obj = Path(target_path)
    
    # Create target directory if it doesn't exist
    target_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    # Basic format conversion - in a real implementation this would use actual conversion tools
    try:
        if source_path_obj.suffix.lower() == '.pt' and target_format == 'onnx':
            # PyTorch to ONNX conversion
            import torch
            model = torch.load(source_path)
            model.eval()
            
            # Create dummy input for tracing
            dummy_input = torch.randn(1, 3, 640, 640)
            
            # Export to ONNX
            torch.onnx.export(
                model,
                dummy_input,
                target_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output']
            )
            
            return True
        elif source_path_obj.suffix.lower() == '.onnx' and target_format == 'openvino':
            # ONNX to OpenVINO conversion
            from openvino.tools import mo
            from openvino.runtime import serialize

            model = mo.convert_model(source_path)
            target_xml = target_path_obj.with_suffix('.xml') if target_path_obj.suffix != '.xml' else target_path_obj
            target_bin = target_xml.with_suffix('.bin')
            serialize(model, str(target_xml), str(target_bin))
            return True
        elif source_path_obj.suffix.lower() == '.pt' and target_format == 'safetensors':
            # PyTorch to SafeTensors conversion
            import torch
            from safetensors.torch import save_file

            loaded = torch.load(source_path, map_location='cpu')
            if hasattr(loaded, 'state_dict'):
                state_dict = loaded.state_dict()
            elif isinstance(loaded, dict):
                state_dict = loaded
            else:
                raise TypeError("Unsupported PyTorch model format for SafeTensors export")

            save_file(state_dict, target_path)
            return True
        elif source_path_obj.suffix.lower() == '.onnx' and target_format == 'coreml':
            # ONNX to CoreML conversion
            import coremltools as ct
            
            # Load ONNX model
            onnx_model = ct.convert(source_path, convert_to='mlprogram')
            onnx_model.save(target_path)
            return True
        else:
            # Unsupported conversion
            logging.error(f"Unsupported conversion from {source_path_obj.suffix} to {target_format}")
            return False
    except Exception as e:
        logging.error(f"Model conversion failed: {e}")
        return False


def validate_model_format(model_path: str) -> Tuple[bool, str]:
    """
    Validate that a model file is in a valid format.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    path = Path(model_path)
    
    if not path.exists():
        return False, f"Model path does not exist: {model_path}"
    
    # Check file extension to determine format
    valid_extensions = ['.onnx', '.xml', '.bin', '.pt', '.safetensors', '.mlmodel']
    if path.suffix.lower() not in valid_extensions and not path.is_dir():
        return False, f"Invalid model format. Expected one of {valid_extensions}, got {path.suffix}"
    
    # For OpenVINO, check both .xml and .bin files exist if it's a directory
    if path.is_dir():
        # Check if this looks like an OpenVINO model directory
        xml_files = list(path.glob('*.xml'))
        if not xml_files:
            return False, f"Directory {model_path} does not contain .xml files for OpenVINO model"
        
        # For each .xml file, check if corresponding .bin exists
        for xml_file in xml_files:
            bin_file = xml_file.with_suffix('.bin')
            if not bin_file.exists():
                return False, f"Missing .bin file for OpenVINO model: {bin_file}"
    
    # Additional validation based on extension would go here
    # For now, just return True for file existence
    return True, ""


def get_model_info(model_path: str) -> Optional[Dict[str, Any]]:
    """
    Get information about a model file.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Dictionary with model information or None if could not be determined
    """
    try:
        path = Path(model_path)
        
        info = {
            'path': str(path),
            'size_mb': path.stat().st_size / (1024 * 1024),
            'last_modified': path.stat().st_mtime
        }
        
        # Try to determine the model type based on extension
        if path.suffix.lower() == '.onnx':
            import onnxruntime as ort
            session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            info['engine_type'] = 'onnx'
            info['inputs'] = [{'name': inp.name, 'shape': inp.shape} for inp in session.get_inputs()]
            info['outputs'] = [{'name': out.name, 'shape': out.shape} for out in session.get_outputs()]
            
        elif path.suffix.lower() == '.xml' or path.is_dir():
            # This might be an OpenVINO model
            from openvino.runtime import Core
            core = Core()
            model = core.read_model(model=model_path if path.is_file() else next(path.glob('*.xml')))
            info['engine_type'] = 'openvino'
            # Add input/output information
            info['inputs'] = [{'shape':inp.shape} for inp in model.inputs]
            info['outputs'] = [{'shape':out.shape} for out in model.outputs]
            
        elif path.suffix.lower() == '.pt':
            import torch
            # PyTorch models don't typically have easy way to get input/output shapes without loading
            state_dict = torch.load(model_path, map_location='cpu')
            info['engine_type'] = 'pytorch'
            info['state_dict_keys'] = list(state_dict.keys())
            
        elif path.suffix.lower() == '.safetensors':
            from safetensors.torch import load_file
            tensors = load_file(model_path)
            info['engine_type'] = 'safetensors'
            info['tensors'] = {name: tensor.shape for name, tensor in tensors.items()}
            
        elif path.suffix.lower() == '.mlmodel':
            import coremltools as ct
            mlmodel = ct.models.MLModel(model_path)
            info['engine_type'] = 'coreml'
            info['inputs'] = [inp.name for inp in mlmodel._spec.description.input]
            info['outputs'] = [out.name for out in mlmodel._spec.description.output]
            
        return info
    except Exception as e:
        logging.error(f"Could not get model info for {model_path}: {e}")
        return None