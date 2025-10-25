"""
Smart home integration with Philips Hue support.
"""
import requests
import logging
import json
import time
from typing import Dict, Any, List, Optional
from pathlib import Path

from src.interfaces.monitoring import SmartHomeController


class HueController(SmartHomeController):
    """
    Controls Philips Hue lights directly via the local API.
    Features connection retry logic and graceful operation when bridge is unavailable.
    """
    
    # Disable SSL warnings for local bridge communication
    from requests.packages.urllib3.exceptions import InsecureRequestWarning
    requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Hue controller.
        
        Args:
            config: Configuration dictionary with Hue settings
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Extract configuration values
        self.bridge_ip = config.get('bridge_ip', '')
        self.app_key = config.get('app_key', '')
        self.light_ids = config.get('light_ids', [])
        self.default_brightness = config.get('brightness', 254)
        self.default_saturation = config.get('saturation', 0)
        self.default_hue = config.get('hue', 14910)
        self.retry_attempts = config.get('retry_attempts', 3)
        self.retry_delay = config.get('retry_delay', 1)
        
        # Base URL for Hue API v2
        if self.bridge_ip and self.app_key:
            self.base_url = f"https://{self.bridge_ip}/clip/v2/resource/light"
            self.headers = {
                "hue-application-key": self.app_key,
            }
            self.is_active = self._test_connection()
        else:
            self.base_url = None
            self.headers = None
            self.is_active = False
            self.logger.warning("Hue bridge IP or app key not provided. Hue control is disabled.")
    
    def _test_connection(self) -> bool:
        """
        Test connection to the Hue bridge.
        
        Returns:
            True if connection is successful, False otherwise
        """
        if not self.bridge_ip or not self.app_key:
            self.logger.error("Hue bridge IP or app key is missing.")
            return False
        
        self.logger.info(f"Testing connection to Hue bridge at {self.bridge_ip}...")
        
        for attempt in range(self.retry_attempts):
            try:
                # Try to get info about the first light in the list, or just the first light if none specified
                if self.light_ids:
                    test_light_id = str(self.light_ids[0])
                else:
                    # If no specific lights defined, try to get any light
                    response = requests.get(
                        f"https://{self.bridge_ip}/clip/v2/resource/device",
                        headers=self.headers,
                        verify=False,
                        timeout=5
                    )
                    if response.status_code == 200:
                        self.logger.info("Hue bridge connection test successful.")
                        return True
                    else:
                        self.logger.error(f"Could not retrieve light information: {response.status_code}, {response.text}")
                        return False
                
                response = requests.get(
                    f"{self.base_url}/{test_light_id}",
                    headers=self.headers,
                    verify=False,
                    timeout=5
                )
                
                if response.status_code == 200:
                    self.logger.info("Hue bridge connection test successful.")
                    return True
                elif response.status_code == 404:
                    self.logger.error(f"Light ID {test_light_id} not found on bridge.")
                    return False
                else:
                    self.logger.error(f"Connection test failed with status {response.status_code}. Response: {response.text}")
                    if attempt < self.retry_attempts - 1:
                        time.sleep(self.retry_delay)
                        continue
                    return False
                    
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Connection test failed: {e}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay)
                    continue
                return False
            except Exception as e:
                self.logger.error(f"Unexpected error during connection test: {e}")
                return False
        
        return False
    
    def _send_put_request(self, light_id: str, payload: dict) -> bool:
        """
        Send a PUT request to a specific light.
        
        Args:
            light_id: ID of the light to control
            payload: The command payload
            
        Returns:
            True if request was successful, False otherwise
        """
        if not self.is_active:
            self.logger.warning("Hue controller is not active, skipping command.")
            return False
        
        if not self.base_url or not self.headers:
            self.logger.error("Hue controller not properly initialized.")
            return False
        
        url = f"{self.base_url}/{light_id}"
        
        for attempt in range(self.retry_attempts):
            try:
                response = requests.put(
                    url, 
                    headers=self.headers, 
                    data=json.dumps(payload), 
                    verify=False, 
                    timeout=10
                )
                
                if response.status_code in [200, 207]:
                    self.logger.debug(f"Command sent successfully to light {light_id}. Payload: {payload}")
                    return True
                elif response.status_code == 404:
                    self.logger.error(f"Light ID {light_id} not found on bridge.")
                    return False
                else:
                    self.logger.error(f"Failed to send command to light {light_id}. "
                                    f"Status: {response.status_code}, Response: {response.text}")
                    if attempt < self.retry_attempts - 1:
                        time.sleep(self.retry_delay)
                        continue
                    return False
                    
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Request failed for light {light_id}: {e}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay)
                    continue
                return False
            except Exception as e:
                self.logger.error(f"Unexpected error sending command to light {light_id}: {e}")
                return False
        
        return False
    
    def set_lights_on(self, light_ids: Optional[List[str]] = None, 
                     brightness: Optional[int] = None, 
                     saturation: Optional[int] = None, 
                     hue: Optional[int] = None) -> bool:
        """
        Turn the specified lights on with the desired settings.
        
        Args:
            light_ids: List of light IDs to control. If None, uses configured light IDs.
            brightness: Brightness level (0-254). If None, uses default.
            saturation: Saturation level (0-254). If None, uses default.
            hue: Hue value (0-65535). If None, uses default.
            
        Returns:
            True if all commands were sent successfully, False otherwise
        """
        if light_ids is None:
            light_ids = self.light_ids
        if brightness is None:
            brightness = self.default_brightness
        if saturation is None:
            saturation = self.default_saturation
        if hue is None:
            hue = self.default_hue
        
        if not light_ids:
            self.logger.warning("No light IDs provided or configured.")
            return False
        
        # Normalize brightness from 0-254 scale to 0-100 percentage for API v2
        brightness_percent = min(100, max(0, int((brightness / 254) * 100)))
        
        payload = {
            "on": {"on": True},
            "dimming": {"brightness": brightness_percent}
        }
        
        # Only add color properties if hue and saturation are provided
        if hue is not None and saturation is not None:
            # Convert HSV to xy for API v2 (simplified conversion)
            xy = self._hsv_to_xy(hue, saturation, brightness)
            payload["color"] = {"xy": xy}
        
        success_count = 0
        for light_id in light_ids:
            light_id = str(light_id).strip()
            if self._send_put_request(light_id, payload):
                success_count += 1
        
        success = success_count == len(light_ids)
        if success:
            self.logger.info(f"Successfully turned on lights: {light_ids}")
        else:
            self.logger.warning(f"Only {success_count}/{len(light_ids)} lights were turned on successfully")
        
        return success
    
    def set_lights_off(self, light_ids: Optional[List[str]] = None) -> bool:
        """
        Turn the specified lights off.
        
        Args:
            light_ids: List of light IDs to control. If None, uses configured light IDs.
            
        Returns:
            True if all commands were sent successfully, False otherwise
        """
        if light_ids is None:
            light_ids = self.light_ids
        
        if not light_ids:
            self.logger.warning("No light IDs provided or configured.")
            return False
        
        payload = {"on": {"on": False}}
        
        success_count = 0
        for light_id in light_ids:
            light_id = str(light_id).strip()
            if self._send_put_request(light_id, payload):
                success_count += 1
        
        success = success_count == len(light_ids)
        if success:
            self.logger.info(f"Successfully turned off lights: {light_ids}")
        else:
            self.logger.warning(f"Only {success_count}/{len(light_ids)} lights were turned off successfully")
        
        return success
    
    def _hsv_to_xy(self, h: int, s: int, v: int) -> Dict[str, float]:
        """
        Simplified conversion from HSV to CIE xy coordinates.
        This is an approximation and may not match all Hue bulbs perfectly.
        
        Args:
            h: Hue (0-65535 scale used by Hue)
            s: Saturation (0-254)
            v: Value/Brightness (0-254)
            
        Returns:
            Dictionary with x and y coordinates
        """
        # For a more accurate conversion, one would need to account for the specific
        # gamut of each Hue bulb model, but this provides a reasonable approximation.
        # For white light, use standard coordinates
        if s == 0:  # No saturation = white light
            return {"x": 0.3127, "y": 0.3290}  # D65 white point
        
        # For colored light, use a basic conversion
        # This is a simplified approach - in practice, each bulb has its own gamut
        # Convert hue from Hue's 0-65535 scale to 0-360 degrees
        hue_degrees = (h / 65535.0) * 360.0
        
        # Basic approximation based on hue
        if 0 <= hue_degrees < 60:  # Red to Yellow
            x = 0.64
            y = 0.33
        elif 60 <= hue_degrees < 120:  # Yellow to Green
            x = 0.41
            y = 0.51
        elif 120 <= hue_degrees < 180:  # Green to Cyan
            x = 0.20
            y = 0.70
        elif 180 <= hue_degrees < 240:  # Cyan to Blue
            x = 0.15
            y = 0.06
        elif 240 <= hue_degrees < 300:  # Blue to Magenta
            x = 0.12
            y = 0.04
        else:  # Magenta to Red
            x = 0.68
            y = 0.32
        
        return {"x": x, "y": y}
    
    def is_connected(self) -> bool:
        """
        Check if the smart home system is connected.
        
        Returns:
            True if connected, False otherwise
        """
        return self.is_active
    
    def set_brightness(self, light_ids: Optional[List[str]] = None, brightness: int = 100) -> bool:
        """
        Set brightness for specified lights.
        
        Args:
            light_ids: List of light IDs to control. If None, uses configured light IDs.
            brightness: Brightness level (0-254)
            
        Returns:
            True if all commands were sent successfully, False otherwise
        """
        if light_ids is None:
            light_ids = self.light_ids
        
        if not light_ids:
            self.logger.warning("No light IDs provided or configured.")
            return False
        
        # Normalize brightness for API v2
        brightness_percent = min(100, max(0, int((brightness / 254) * 100)))
        
        payload = {"dimming": {"brightness": brightness_percent}}
        
        success_count = 0
        for light_id in light_ids:
            light_id = str(light_id).strip()
            if self._send_put_request(light_id, payload):
                success_count += 1
        
        success = success_count == len(light_ids)
        if success:
            self.logger.info(f"Successfully set brightness to {brightness} for lights: {light_ids}")
        else:
            self.logger.warning(f"Only {success_count}/{len(light_ids)} lights had brightness adjusted successfully")
        
        return success
    
    def set_color(self, light_ids: Optional[List[str]] = None, hue: int = 14910, 
                 saturation: int = 0) -> bool:
        """
        Set color for specified lights.
        
        Args:
            light_ids: List of light IDs to control. If None, uses configured light IDs.
            hue: Hue value (0-65535)
            saturation: Saturation level (0-254)
            
        Returns:
            True if all commands were sent successfully, False otherwise
        """
        if light_ids is None:
            light_ids = self.light_ids
        
        if not light_ids:
            self.logger.warning("No light IDs provided or configured.")
            return False
        
        # Convert HSV to xy coordinates
        brightness = self.default_brightness  # Use current brightness
        xy = self._hsv_to_xy(hue, saturation, brightness)
        
        payload = {"color": {"xy": xy}}
        
        success_count = 0
        for light_id in light_ids:
            light_id = str(light_id).strip()
            if self._send_put_request(light_id, payload):
                success_count += 1
        
        success = success_count == len(light_ids)
        if success:
            self.logger.info(f"Successfully set color for lights: {light_ids}")
        else:
            self.logger.warning(f"Only {success_count}/{len(light_ids)} lights had color set successfully")
        
        return success
    
    def flash_lights(self, light_ids: Optional[List[str]] = None, 
                    duration: float = 2.0, 
                    brightness: int = 254) -> bool:
        """
        Flash lights on and off for a specified duration.
        
        Args:
            light_ids: List of light IDs to control. If None, uses configured light IDs.
            duration: Duration to flash in seconds
            brightness: Brightness level for flashing (0-254)
            
        Returns:
            True if flashing completed successfully, False otherwise
        """
        if light_ids is None:
            light_ids = self.light_ids
        
        if not light_ids:
            self.logger.warning("No light IDs provided or configured.")
            return False
        
        # Flash pattern: on for 0.5s, off for 0.5s
        flash_interval = 0.5
        cycles = int(duration / (flash_interval * 2))
        
        for cycle in range(cycles):
            # Turn lights on
            if not self.set_lights_on(light_ids, brightness=brightness):
                self.logger.error(f"Failed to turn lights on during flash cycle {cycle}")
                return False
            
            time.sleep(flash_interval)
            
            # Turn lights off
            if not self.set_lights_off(light_ids):
                self.logger.error(f"Failed to turn lights off during flash cycle {cycle}")
                return False
            
            time.sleep(flash_interval)
        
        self.logger.info(f"Completed {cycles} flash cycles for lights: {light_ids}")
        return True


def get_available_lights(bridge_ip: str, app_key: str) -> Optional[List[Dict[str, Any]]]:
    """
    Get a list of available lights from the Hue bridge.
    
    Args:
        bridge_ip: IP address of the Hue bridge
        app_key: Application key for the Hue bridge
        
    Returns:
        List of light information, or None if failed
    """
    try:
        headers = {"hue-application-key": app_key}
        response = requests.get(
            f"https://{bridge_ip}/clip/v2/resource/light",
            headers=headers,
            verify=False,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            lights = []
            
            for resource in data.get('data', []):
                light_info = {
                    'id': resource.get('id', ''),
                    'name': resource.get('metadata', {}).get('name', 'Unknown'),
                    'on': resource.get('on', {}).get('on', False),
                    'dimming': resource.get('dimming', {}).get('brightness', 100),
                    'type': resource.get('type', 'light')
                }
                lights.append(light_info)
            
            return lights
        else:
            logging.error(f"Failed to get lights from bridge: {response.status_code}, {response.text}")
            return None
            
    except Exception as e:
        logging.error(f"Error getting available lights: {e}")
        return None