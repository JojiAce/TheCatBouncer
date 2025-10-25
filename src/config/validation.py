"""
Configuration validation helpers.
"""
from typing import List, Tuple
import os
from pathlib import Path


def validate_model_path(model_path: str) -> Tuple[bool, str]:
    """
    Validate that the model path exists and is accessible.
    
    Args:
        model_path: Path to the model
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not model_path:
        return False, "Model path is empty"
    
    path = Path(model_path)
    if not path.exists():
        return False, f"Model path does not exist: {model_path}"
    
    # For OpenVINO models, check that both .xml and .bin files exist
    if model_path.endswith('_openvino_model') or model_path.endswith('.xml'):
        xml_path = path.with_suffix('.xml') if path.suffix != '.xml' else path
        bin_path = xml_path.with_suffix('.bin')
        if not xml_path.exists():
            return False, f"OpenVINO model XML file does not exist: {xml_path}"
        if not bin_path.exists():
            return False, f"OpenVINO model BIN file does not exist: {bin_path}"
    
    return True, ""


def validate_camera_index(index: int) -> Tuple[bool, str]:
    """
    Validate camera index.
    
    Args:
        index: Camera index to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if index < 0:
        return False, "Camera index must be non-negative"
    
    # Try to open the camera to test if it's available
    # This requires importing cv2, but we'll just check validity here
    return True, ""


def validate_resolution(width: int, height: int) -> Tuple[bool, str]:
    """
    Validate resolution values.
    
    Args:
        width: Width in pixels
        height: Height in pixels
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if width <= 0 or height <= 0:
        return False, "Resolution width and height must be positive"
    
    if width > 7680 or height > 4320:  # 8K max resolution check
        return False, "Resolution is too high (max 7680x4320)"
    
    return True, ""


def validate_brightness_threshold(threshold: int) -> Tuple[bool, str]:
    """
    Validate brightness threshold value.
    
    Args:
        threshold: Brightness threshold (0-255)
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not (0 <= threshold <= 255):
        return False, "Brightness threshold must be between 0 and 255"
    
    return True, ""


def validate_percentage(percentage: float) -> Tuple[bool, str]:
    """
    Validate percentage value.
    
    Args:
        percentage: Percentage value (0.0-1.0)
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not (0.0 <= percentage <= 1.0):
        return False, "Percentage must be between 0.0 and 1.0"
    
    return True, ""


def validate_hsv_values(h: int, s: int, v: int) -> Tuple[bool, str]:
    """
    Validate HSV values.
    
    Args:
        h: Hue value (0-180)
        s: Saturation value (0-255)
        v: Value/Brightness value (0-255)
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not (0 <= h <= 180):
        return False, f"Hue value must be between 0 and 180, got {h}"
    
    if not (0 <= s <= 255):
        return False, f"Saturation value must be between 0 and 255, got {s}"
    
    if not (0 <= v <= 255):
        return False, f"Value/Brightness value must be between 0 and 255, got {v}"
    
    return True, ""


def validate_time_format(time_str: str) -> Tuple[bool, str]:
    """
    Validate time format HH:MM.
    
    Args:
        time_str: Time string in HH:MM format
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not time_str or len(time_str) != 5 or time_str[2] != ':':
        return False, f"Time format must be HH:MM, got '{time_str}'"
    
    try:
        hours, minutes = time_str.split(':')
        h = int(hours)
        m = int(minutes)
        
        if not (0 <= h <= 23):
            return False, f"Hours must be between 0 and 23, got {h}"
        
        if not (0 <= m <= 59):
            return False, f"Minutes must be between 0 and 59, got {m}"
        
        return True, ""
    except ValueError:
        return False, f"Time format must be HH:MM with numeric values, got '{time_str}'"


def validate_email_format(email: str) -> Tuple[bool, str]:
    """
    Basic email format validation.
    
    Args:
        email: Email address to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not email:
        return False, "Email address is required"
    
    if '@' not in email or '.' not in email.split('@')[-1]:
        return False, f"Invalid email format: {email}"
    
    return True, ""


def validate_ip_format(ip: str) -> Tuple[bool, str]:
    """
    Basic IP address format validation.
    
    Args:
        ip: IP address to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not ip:
        return False, "IP address is required"
    
    parts = ip.split('.')
    if len(parts) != 4:
        return False, f"Invalid IP format: {ip}"
    
    for part in parts:
        try:
            num = int(part)
            if not (0 <= num <= 255):
                return False, f"IP octet must be between 0 and 255, got {num}"
        except ValueError:
            return False, f"IP octets must be numeric, got '{part}'"
    
    return True, ""


def validate_path_exists(path: str) -> Tuple[bool, str]:
    """
    Validate that a path exists.
    
    Args:
        path: Path to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not path:
        return False, "Path is required"
    
    if not os.path.exists(path):
        return False, f"Path does not exist: {path}"
    
    return True, ""


def validate_path_writable(path: str) -> Tuple[bool, str]:
    """
    Validate that a path is writable.
    
    Args:
        path: Path to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not path:
        return False, "Path is required"
    
    path_obj = Path(path)
    
    # Create the directory if it doesn't exist
    path_obj.mkdir(parents=True, exist_ok=True)
    
    # Check if we can write to the directory
    test_file = path_obj / ".write_test"
    try:
        test_file.touch()
        test_file.unlink()  # Remove the test file
        return True, ""
    except (PermissionError, OSError) as e:
        return False, f"Path is not writable: {path} - {str(e)}"


def validate_confidence_threshold(threshold: float) -> Tuple[bool, str]:
    """
    Validate confidence threshold.
    
    Args:
        threshold: Confidence threshold (0.0-1.0)
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not (0.0 <= threshold <= 1.0):
        return False, "Confidence threshold must be between 0.0 and 1.0"
    
    return True, ""


def validate_device(device: str) -> Tuple[bool, str]:
    """
    Validate inference device.
    
    Args:
        device: Device name
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    valid_devices = ["cpu", "gpu", "cuda:0", "cuda:1", "cuda:2", "cuda:3"]
    if device not in valid_devices:
        return False, f"Device must be one of {valid_devices}, got '{device}'"
    
    return True, ""


def validate_engine(engine: str) -> Tuple[bool, str]:
    """
    Validate inference engine.
    
    Args:
        engine: Engine name
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    valid_engines = ["onnx", "openvino", "pt", "safetensor", "coreml"]
    if engine not in valid_engines:
        return False, f"Engine must be one of {valid_engines}, got '{engine}'"
    
    return True, ""


def validate_percentage_range(percentage: float) -> Tuple[bool, str]:
    """
    Validate a percentage value in the range 0.0 to 1.0.
    
    Args:
        percentage: Percentage value to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not (0.0 <= percentage <= 1.0):
        return False, f"Percentage must be between 0.0 and 1.0, got {percentage}"
    
    return True, ""