"""
Unit tests for color analysis system.
"""
import pytest
import numpy as np
import cv2
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.managers.color_analyzer import ColorAnalyzer, analyze_cat_color, get_dominant_colors, compare_colors


def test_color_analyzer_initialization():
    """Test ColorAnalyzer initialization."""
    config = {
        'analysis_method': 'hsv',
        'lower_black_hsv': (0, 0, 0),
        'upper_black_hsv': (180, 255, 60),
        'black_pixel_threshold': 0.5,
        'ollama_host': 'http://localhost:11434',
        'ollama_model': 'llava',
        'ollama_prompt': 'Is the cat in this image black?'
    }
    
    analyzer = ColorAnalyzer(config)
    
    assert analyzer.analysis_method == 'hsv'
    assert analyzer.lower_black_hsv == (0, 0, 0)
    assert analyzer.upper_black_hsv == (180, 255, 60)
    assert analyzer.black_pixel_threshold == 0.5


def test_color_analyzer_invalid_hsv():
    """Test ColorAnalyzer with invalid HSV values."""
    config = {
        'analysis_method': 'hsv',
        'lower_black_hsv': (0, 0),  # Invalid length
        'upper_black_hsv': (180, 255, 60),
        'black_pixel_threshold': 0.5
    }
    
    try:
        analyzer = ColorAnalyzer(config)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass  # Expected behavior


def test_analyze_color_hsv():
    """Test HSV-based color analysis."""
    config = {
        'analysis_method': 'hsv',
        'lower_black_hsv': (0, 0, 0),
        'upper_black_hsv': (180, 255, 60),
        'black_pixel_threshold': 0.5
    }
    
    analyzer = ColorAnalyzer(config)
    
    # Create a test image with a black area
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        # Create test image: half black, half white
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_image[:50, :, :] = 0  # Black top half
        test_image[50:, :, :] = 255  # White bottom half
        cv2.imwrite(tmp.name, test_image)
        
        result = analyzer.analyze_color(tmp.name, {'x1': 0, 'y1': 0, 'x2': 100, 'y2': 100})
        
        # Since 50% of the image is black and threshold is 50%, result should be True
        assert result == True
        
        # Clean up
        Path(tmp.name).unlink()


def test_analyze_color_hsv_with_different_threshold():
    """Test HSV-based color analysis with different thresholds."""
    config = {
        'analysis_method': 'hsv',
        'lower_black_hsv': (0, 0, 0),
        'upper_black_hsv': (180, 255, 60),
        'black_pixel_threshold': 0.8  # 80% threshold
    }
    
    analyzer = ColorAnalyzer(config)
    
    # Create a test image that is 50% black
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_image[:50, :, :] = 0  # Black top half
        test_image[50:, :, :] = 255  # White bottom half
        cv2.imwrite(tmp.name, test_image)
        
        result = analyzer.analyze_color(tmp.name, {'x1': 0, 'y1': 0, 'x2': 100, 'y2': 100})
        
        # Since only 50% is black but threshold is 80%, result should be False
        assert result == False
        
        # Clean up
        Path(tmp.name).unlink()


def test_analyze_multiple_color_ranges():
    """Test multiple color range analysis."""
    config = {
        'analysis_method': 'hsv',
        'lower_black_hsv': (0, 0, 0),
        'upper_black_hsv': (180, 255, 60),
        'black_pixel_threshold': 0.5
    }
    
    analyzer = ColorAnalyzer(config)
    
    # Create a test image with different colored regions
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        # Red region
        test_image[:50, :50, 2] = 255  # Red
        # Green region
        test_image[:50, 50:, 1] = 255  # Green
        # Blue region
        test_image[50:, :50, 0] = 255  # Blue
        # White region
        test_image[50:, 50:] = 255  # White
        
        cv2.imwrite(tmp.name, test_image)
        
        # Define color ranges to analyze
        color_ranges = [
            ('red', (0, 50, 50), (10, 255, 255)),
            ('green', (50, 50, 50), (70, 255, 255)),
            ('blue', (100, 50, 50), (130, 255, 255))
        ]
        
        results = analyzer.analyze_multiple_color_ranges(
            tmp.name, 
            {'x1': 0, 'y1': 0, 'x2': 100, 'y2': 100}, 
            color_ranges
        )
        
        # Each color should be roughly 25% of the image
        assert 'red' in results
        assert 'green' in results
        assert 'blue' in results
        assert 0.20 < results['red'] < 0.30  # Should be around 25%
        assert 0.20 < results['green'] < 0.30  # Should be around 25%
        assert 0.20 < results['blue'] < 0.30  # Should be around 25%
        
        # Clean up
        Path(tmp.name).unlink()


@patch('requests.post')
def test_analyze_color_llm(mock_post):
    """Test LLM-based color analysis with mocked API call."""
    config = {
        'analysis_method': 'llm',
        'lower_black_hsv': (0, 0, 0),
        'upper_black_hsv': (180, 255, 60),
        'black_pixel_threshold': 0.5,
        'ollama_host': 'http://localhost:11434',
        'ollama_model': 'llava',
        'ollama_prompt': 'Is the cat in this image black?'
    }
    
    # Mock the API response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {'response': 'yes, the cat is black'}
    mock_post.return_value = mock_response
    
    analyzer = ColorAnalyzer(config)
    
    # Create a test image
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(tmp.name, test_image)
        
        result = analyzer.analyze_color(tmp.name, {'x1': 0, 'y1': 0, 'x2': 100, 'y2': 100})
        
        # The API returns 'yes', so result should be True
        assert result == True
        
        # Clean up
        Path(tmp.name).unlink()


@patch('requests.post')
def test_analyze_color_llm_fallback(mock_post):
    """Test LLM-based color analysis fallback to HSV when LLM fails."""
    config = {
        'analysis_method': 'llm',
        'lower_black_hsv': (0, 0, 0),
        'upper_black_hsv': (180, 255, 60),
        'black_pixel_threshold': 0.5,
        'ollama_host': 'http://localhost:11434',
        'ollama_model': 'llava',
        'ollama_prompt': 'Is the cat in this image black?'
    }
    
    # Mock the API to raise an exception
    mock_post.side_effect = Exception("API Error")
    
    analyzer = ColorAnalyzer(config)
    
    # Create a test image that should pass HSV analysis
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(tmp.name, test_image)
        
        result = analyzer.analyze_color(tmp.name, {'x1': 0, 'y1': 0, 'x2': 100, 'y2': 100})
        
        # Should fall back to HSV analysis and return True (black image with black threshold)
        assert result == True
        
        # Clean up
        Path(tmp.name).unlink()


def test_analyze_cat_color_convenience_function():
    """Test the analyze_cat_color convenience function."""
    config = {
        'analysis_method': 'hsv',
        'lower_black_hsv': (0, 0, 0),
        'upper_black_hsv': (180, 255, 60),
        'black_pixel_threshold': 0.5
    }
    
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        # Create a black image
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(tmp.name, test_image)
        
        result = analyze_cat_color(tmp.name, config)
        
        # Should return True as the image is black
        assert result == True
        
        # Clean up
        Path(tmp.name).unlink()


def test_get_dominant_colors():
    """Test dominant color extraction."""
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        # Create an image with distinct colors
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_image[:50, :50] = [255, 0, 0]    # Red
        test_image[:50, 50:] = [0, 255, 0]    # Green
        test_image[50:, :50] = [0, 0, 255]    # Blue
        test_image[50:, 50:] = [255, 255, 255] # White
        cv2.imwrite(tmp.name, test_image)
        
        dominant_colors = get_dominant_colors(tmp.name, k=4)
        
        # Should find 4 dominant colors
        assert len(dominant_colors) == 4
        
        # Clean up
        Path(tmp.name).unlink()


def test_compare_colors():
    """Test color comparison function."""
    # Compare identical colors
    assert compare_colors((255, 0, 0), (255, 0, 0)) == True
    
    # Compare similar colors within threshold
    assert compare_colors((255, 0, 0), (250, 5, 5), threshold=10) == True
    
    # Compare different colors outside threshold
    assert compare_colors((255, 0, 0), (0, 255, 0), threshold=10) == False


if __name__ == "__main__":
    test_color_analyzer_initialization()
    test_color_analyzer_invalid_hsv()
    test_analyze_color_hsv()
    test_analyze_color_hsv_with_different_threshold()
    test_analyze_multiple_color_ranges()
    test_analyze_cat_color_convenience_function()
    test_get_dominant_colors()
    test_compare_colors()
    print("All color analysis system tests passed!")