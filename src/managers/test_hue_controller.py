"""
Unit tests for smart home integration (Hue controller).
"""
import pytest
import requests
from unittest.mock import Mock, patch, MagicMock
import json

from src.managers.hue_controller import HueController, get_available_lights


def test_hue_controller_initialization():
    """Test HueController initialization."""
    config = {
        'bridge_ip': '192.168.1.100',
        'app_key': 'test_app_key',
        'light_ids': ['1', '2', '3'],
        'brightness': 200,
        'saturation': 100,
        'hue': 15000
    }
    
    # Mock the test connection to succeed
    with patch.object(HueController, '_test_connection', return_value=True):
        controller = HueController(config)
        
        assert controller.bridge_ip == '192.168.1.100'
        assert controller.app_key == 'test_app_key'
        assert controller.light_ids == ['1', '2', '3']
        assert controller.default_brightness == 200
        assert controller.is_active == True


def test_hue_controller_initialization_without_credentials():
    """Test HueController initialization without credentials (should be inactive)."""
    config = {
        'bridge_ip': '',
        'app_key': '',
        'light_ids': ['1', '2', '3']
    }
    
    controller = HueController(config)
    
    assert controller.is_active == False


def test_test_connection():
    """Test the connection test functionality."""
    config = {
        'bridge_ip': '192.168.1.100',
        'app_key': 'test_app_key',
        'light_ids': ['1']
    }
    
    # Mock successful connection response
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        controller = HueController(config)
        
        # Should succeed due to mocked response
        assert controller.is_active == True


def test_send_put_request():
    """Test the PUT request sending functionality."""
    config = {
        'bridge_ip': '192.168.1.100',
        'app_key': 'test_app_key',
        'light_ids': ['1']
    }
    
    with patch('requests.get') as mock_get, patch('requests.put') as mock_put:
        # Mock successful connection test
        conn_response = Mock()
        conn_response.status_code = 200
        mock_get.return_value = conn_response
        
        # Mock successful PUT request
        put_response = Mock()
        put_response.status_code = 200
        mock_put.return_value = put_response
        
        controller = HueController(config)
        
        # Set to active since _test_connection is mocked
        controller.is_active = True
        controller.base_url = f"https://{config['bridge_ip']}/clip/v2/resource/light"
        controller.headers = {"hue-application-key": config['app_key']}
        
        payload = {"on": {"on": True}}
        success = controller._send_put_request('1', payload)
        
        assert success == True
        mock_put.assert_called_once()


def test_set_lights_on():
    """Test turning lights on."""
    config = {
        'bridge_ip': '192.168.1.100',
        'app_key': 'test_app_key',
        'light_ids': ['1', '2'],
        'brightness': 200,
        'saturation': 100,
        'hue': 15000
    }
    
    with patch('requests.get') as mock_get, patch('requests.put') as mock_put:
        # Mock successful connection test
        conn_response = Mock()
        conn_response.status_code = 200
        mock_get.return_value = conn_response
        
        # Mock successful PUT requests
        put_response = Mock()
        put_response.status_code = 200
        mock_put.return_value = put_response
        
        controller = HueController(config)
        
        # Set to active since _test_connection is mocked
        controller.is_active = True
        controller.base_url = f"https://{config['bridge_ip']}/clip/v2/resource/light"
        controller.headers = {"hue-application-key": config['app_key']}
        
        success = controller.set_lights_on()
        
        assert success == True
        # Should make 2 calls for 2 lights
        assert mock_put.call_count == 2


def test_set_lights_off():
    """Test turning lights off."""
    config = {
        'bridge_ip': '192.168.1.100',
        'app_key': 'test_app_key',
        'light_ids': ['1', '2']
    }
    
    with patch('requests.get') as mock_get, patch('requests.put') as mock_put:
        # Mock successful connection test
        conn_response = Mock()
        conn_response.status_code = 200
        mock_get.return_value = conn_response
        
        # Mock successful PUT requests
        put_response = Mock()
        put_response.status_code = 200
        mock_put.return_value = put_response
        
        controller = HueController(config)
        
        # Set to active since _test_connection is mocked
        controller.is_active = True
        controller.base_url = f"https://{config['bridge_ip']}/clip/v2/resource/light"
        controller.headers = {"hue-application-key": config['app_key']}
        
        success = controller.set_lights_off()
        
        assert success == True
        # Should make 2 calls for 2 lights
        assert mock_put.call_count == 2


def test_is_connected():
    """Test the is_connected method."""
    config = {
        'bridge_ip': '192.168.1.100',
        'app_key': 'test_app_key',
        'light_ids': ['1']
    }
    
    # Mock successful connection test
    with patch.object(HueController, '_test_connection', return_value=True):
        controller = HueController(config)
        assert controller.is_connected() == True


def test_hsv_to_xy_conversion():
    """Test HSV to XY color conversion."""
    config = {
        'bridge_ip': '192.168.1.100',
        'app_key': 'test_app_key',
        'light_ids': ['1']
    }
    
    with patch.object(HueController, '_test_connection', return_value=True):
        controller = HueController(config)
        
        # Test white color (saturation = 0)
        white_xy = controller._hsv_to_xy(0, 0, 200)
        assert 'x' in white_xy
        assert 'y' in white_xy
        
        # Test a colored value
        color_xy = controller._hsv_to_xy(15000, 200, 200)
        assert 'x' in color_xy
        assert 'y' in color_xy


def test_set_brightness():
    """Test setting brightness."""
    config = {
        'bridge_ip': '192.168.1.100',
        'app_key': 'test_app_key',
        'light_ids': ['1'],
        'brightness': 200
    }
    
    with patch('requests.get') as mock_get, patch('requests.put') as mock_put:
        # Mock successful connection test
        conn_response = Mock()
        conn_response.status_code = 200
        mock_get.return_value = conn_response
        
        # Mock successful PUT request
        put_response = Mock()
        put_response.status_code = 200
        mock_put.return_value = put_response
        
        controller = HueController(config)
        
        # Set to active since _test_connection is mocked
        controller.is_active = True
        controller.base_url = f"https://{config['bridge_ip']}/clip/v2/resource/light"
        controller.headers = {"hue-application-key": config['app_key']}
        
        success = controller.set_brightness(brightness=150)
        
        assert success == True


def test_flash_lights():
    """Test flashing lights."""
    config = {
        'bridge_ip': '192.168.1.100',
        'app_key': 'test_app_key',
        'light_ids': ['1'],
        'brightness': 200
    }
    
    with patch('requests.get') as mock_get, \
         patch('requests.put') as mock_put, \
         patch('time.sleep') as mock_sleep:
        
        # Mock successful connection test
        conn_response = Mock()
        conn_response.status_code = 200
        mock_get.return_value = conn_response
        
        # Mock successful PUT requests
        put_response = Mock()
        put_response.status_code = 200
        mock_put.return_value = put_response
        
        controller = HueController(config)
        
        # Set to active since _test_connection is mocked
        controller.is_active = True
        controller.base_url = f"https://{config['bridge_ip']}/clip/v2/resource/light"
        controller.headers = {"hue-application-key": config['app_key']}
        
        # Test with a short duration to avoid long test execution
        success = controller.flash_lights(duration=0.2)
        
        # For 0.2s duration with 0.5s interval, there should be 0 full cycles
        # So it would just execute the on/off once
        assert success == True


@patch('requests.get')
def test_get_available_lights(mock_get):
    """Test getting available lights from bridge."""
    # Mock successful response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        'data': [
            {
                'id': '1',
                'metadata': {'name': 'Living Room'},
                'on': {'on': True},
                'dimming': {'brightness': 80},
                'type': 'light'
            }
        ]
    }
    mock_get.return_value = mock_response
    
    lights = get_available_lights('192.168.1.100', 'test_app_key')
    
    assert lights is not None
    assert len(lights) == 1
    assert lights[0]['id'] == '1'
    assert lights[0]['name'] == 'Living Room'


if __name__ == "__main__":
    test_hue_controller_initialization()
    test_hue_controller_initialization_without_credentials()
    test_set_lights_on()
    test_set_lights_off()
    test_is_connected()
    test_hsv_to_xy_conversion()
    test_set_brightness()
    test_flash_lights()
    test_get_available_lights()
    print("All smart home integration tests passed!")