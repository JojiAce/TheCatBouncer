"""
Color analysis system with HSV and LLM-based analysis.
"""
import cv2
import numpy as np
import json
import requests
from pathlib import Path
import logging
from typing import Dict, Any, Optional, Tuple, Union
import time

from src.interfaces.monitoring import ColorAnalyzer


class ColorAnalyzer(ColorAnalyzer):
    """
    Color analysis system with HSV analysis and LLM-based analysis.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the color analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Extract configuration values
        self.analysis_method = config.get('analysis_method', 'hsv').lower()
        self.lower_black_hsv = tuple(config.get('lower_black_hsv', (0, 0, 0)))
        self.upper_black_hsv = tuple(config.get('upper_black_hsv', (180, 255, 60)))
        self.black_pixel_threshold = config.get('black_pixel_threshold', 0.5)
        
        # LLM configuration
        self.ollama_host = config.get('ollama_host', 'http://localhost:11434')
        self.ollama_model = config.get('ollama_model', 'llava')
        self.ollama_prompt = config.get('ollama_prompt', 
                                       'Is the cat in this image black? Answer with only "yes" or "no".')
        
        # Validate HSV values
        if len(self.lower_black_hsv) != 3 or len(self.upper_black_hsv) != 3:
            raise ValueError("HSV values must be tuples of 3 values (H, S, V)")
        
        if not (0 <= self.lower_black_hsv[0] <= 180 and 0 <= self.upper_black_hsv[0] <= 180):
            raise ValueError("Hue values must be between 0 and 180")
        
        if not (0 <= self.lower_black_hsv[1] <= 255 and 0 <= self.upper_black_hsv[1] <= 255):
            raise ValueError("Saturation values must be between 0 and 255")
        
        if not (0 <= self.lower_black_hsv[2] <= 255 and 0 <= self.upper_black_hsv[2] <= 255):
            raise ValueError("Value/Brightness values must be between 0 and 255")
        
        self.logger.info(f"Color analyzer initialized with method: {self.analysis_method}")
    
    def analyze_color(self, image_path: str, bbox: Dict[str, int]) -> bool:
        """
        Analyze the color of an object in an image using the configured method.
        
        Args:
            image_path: Path to the image file
            bbox: Bounding box coordinates (x1, y1, x2, y2)
            
        Returns:
            True if the color matches the target, False otherwise
        """
        try:
            if self.analysis_method == 'hsv':
                return self._analyze_color_hsv(image_path, bbox)
            elif self.analysis_method == 'llm':
                return self._analyze_color_llm(image_path, bbox)
            elif self.analysis_method == 'combined':
                # Use both methods and return True if either succeeds
                hsv_result = self._analyze_color_hsv(image_path, bbox)
                llm_result = self._analyze_color_llm(image_path, bbox)
                self.logger.info(f"Combined analysis - HSV: {hsv_result}, LLM: {llm_result}")
                return hsv_result or llm_result
            else:
                self.logger.error(f"Unknown analysis method: {self.analysis_method}")
                return False
        except Exception as e:
            self.logger.error(f"Error in color analysis: {e}")
            return False
    
    def _analyze_color_hsv(self, image_path: str, bbox: Dict[str, int]) -> bool:
        """
        Analyze color using HSV analysis.
        
        Args:
            image_path: Path to the image file
            bbox: Bounding box coordinates (x1, y1, x2, y2)
            
        Returns:
            True if the color matches the target, False otherwise
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                self.logger.error(f"Could not load image: {image_path}")
                return False
            
            # Extract ROI (region of interest) using the bounding box
            x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
            roi = image[y1:y2, x1:x2]
            
            if roi.size == 0:
                self.logger.warning(f"Empty ROI for bbox: {bbox}")
                return False
            
            # Convert ROI to HSV color space
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # Create a mask for the target color range
            mask = cv2.inRange(hsv_roi, self.lower_black_hsv, self.upper_black_hsv)
            
            # Calculate the percentage of pixels in the target color range
            total_pixels = roi.shape[0] * roi.shape[1]
            color_pixels = cv2.countNonZero(mask)
            percentage = color_pixels / total_pixels if total_pixels > 0 else 0
            
            result = percentage >= self.black_pixel_threshold
            
            self.logger.info(f"HSV analysis result: {percentage:.2%} of pixels "
                           f"in target range (threshold: {self.black_pixel_threshold:.2%}), "
                           f"result: {result}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in HSV color analysis: {e}")
            return False
    
    def _analyze_color_llm(self, image_path: str, bbox: Dict[str, int]) -> bool:
        """
        Analyze color using LLM (Ollama).
        
        Args:
            image_path: Path to the image file
            bbox: Bounding box coordinates (x1, y1, x2, y2)
            
        Returns:
            True if the color matches the target, False otherwise
        """
        try:
            # Extract ROI (region of interest) using the bounding box
            image = cv2.imread(image_path)
            if image is None:
                self.logger.error(f"Could not load image: {image_path}")
                return False
            
            x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
            roi = image[y1:y2, x1:x2]
            
            if roi.size == 0:
                self.logger.warning(f"Empty ROI for bbox: {bbox}")
                return False
            
            # Encode the ROI image to base64 for the API request
            _, buffer = cv2.imencode('.jpg', roi, [cv2.IMWRITE_JPEG_QUALITY, 85])
            roi_base64 = buffer.tobytes()
            
            # Prepare the API request
            url = f"{self.ollama_host}/api/generate"
            payload = {
                "model": self.ollama_model,
                "prompt": self.ollama_prompt,
                "images": [roi_base64.decode('utf-8')],
                "stream": False
            }
            
            # Make the API request
            self.logger.info(f"Sending request to Ollama at {self.ollama_host}")
            response = requests.post(url, json=payload, timeout=30)
            
            if response.status_code != 200:
                self.logger.error(f"Ollama API request failed with status {response.status_code}: {response.text}")
                return False
            
            # Parse response
            response_data = response.json()
            response_text = response_data.get('response', '').strip().lower()
            
            self.logger.info(f"Ollama response: {response_text[:50]}...")  # Log first 50 chars
            
            # Determine result based on response
            result = 'yes' in response_text or 'true' in response_text
            
            self.logger.info(f"LLM analysis result: {result}")
            
            return result
            
        except requests.exceptions.ConnectionError:
            self.logger.error(f"Could not connect to Ollama at {self.ollama_host}. Using fallback method.")
            # Fallback to HSV analysis if LLM is unavailable
            if 'x1' in bbox and 'y1' in bbox and 'x2' in bbox and 'y2' in bbox:
                return self._analyze_color_hsv(image_path, bbox)
            else:
                return False
        except Exception as e:
            self.logger.error(f"Error in LLM color analysis: {e}")
            # Fallback to HSV analysis if LLM fails
            if 'x1' in bbox and 'y1' in bbox and 'x2' in bbox and 'y2' in bbox:
                return self._analyze_color_hsv(image_path, bbox)
            else:
                return False
    
    def analyze_multiple_color_ranges(self, image_path: str, bbox: Dict[str, int], 
                                    color_ranges: list) -> Dict[str, float]:
        """
        Analyze multiple color ranges in the same image ROI.
        
        Args:
            image_path: Path to the image file
            bbox: Bounding box coordinates (x1, y1, x2, y2)
            color_ranges: List of tuples (name, lower_hsv, upper_hsv)
            
        Returns:
            Dictionary with color name and percentage of pixels for each range
        """
        results = {}
        
        try:
            # Load image and extract ROI
            image = cv2.imread(image_path)
            if image is None:
                self.logger.error(f"Could not load image: {image_path}")
                return results
            
            x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
            roi = image[y1:y2, x1:x2]
            
            if roi.size == 0:
                self.logger.warning(f"Empty ROI for bbox: {bbox}")
                return results
            
            # Convert ROI to HSV
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            total_pixels = roi.shape[0] * roi.shape[1]
            
            # Analyze each color range
            for name, lower_hsv, upper_hsv in color_ranges:
                mask = cv2.inRange(hsv_roi, lower_hsv, upper_hsv)
                color_pixels = cv2.countNonZero(mask)
                percentage = color_pixels / total_pixels if total_pixels > 0 else 0
                results[name] = percentage
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in multiple color range analysis: {e}")
            return results


def analyze_cat_color(image_path: str, config: Dict[str, Any]) -> bool:
    """
    Convenience function to analyze cat color using the configuration.
    
    Args:
        image_path: Path to the image containing the cat
        config: Configuration dictionary for color analysis
        
    Returns:
        True if the cat is identified as the target color, False otherwise
    """
    analyzer = ColorAnalyzer(config)
    # This assumes we have bbox coordinates, for this example we'll use the full image
    # In practice, this would be replaced with actual object detection bbox
    bbox = {'x1': 0, 'y1': 0, 'x2': -1, 'y2': -1}  # Placeholder values
    
    # Get actual dimensions of the image
    image = cv2.imread(image_path)
    if image is not None:
        height, width = image.shape[:2]
        bbox = {'x1': 0, 'y1': 0, 'x2': width, 'y2': height}
    
    return analyzer.analyze_color(image_path, bbox)


def get_dominant_colors(image_path: str, k: int = 5) -> np.ndarray:
    """
    Get the dominant colors in an image using K-means clustering.
    
    Args:
        image_path: Path to the image file
        k: Number of dominant colors to find
        
    Returns:
        Array of dominant colors in RGB format
    """
    image = cv2.imread(image_path)
    if image is None:
        return np.array([])
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Reshape the image to be a list of pixels
    pixels = image.reshape((-1, 3))
    
    # Convert to float type
    pixels = np.float32(pixels)
    
    # Define criteria and apply k-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Convert back to 8-bit values
    centers = np.uint8(centers)
    
    return centers


def compare_colors(color1: tuple, color2: tuple, threshold: int = 30) -> bool:
    """
    Compare two colors in RGB space.
    
    Args:
        color1: First color as (R, G, B) tuple
        color2: Second color as (R, G, B) tuple
        threshold: Maximum distance between colors to consider them similar
        
    Returns:
        True if colors are similar within the threshold, False otherwise
    """
    if len(color1) != 3 or len(color2) != 3:
        raise ValueError("Colors must be RGB tuples with 3 values")
    
    distance = np.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(color1, color2)))
    return distance <= threshold