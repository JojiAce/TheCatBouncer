"""
Configuration manager with type-safe loading and validation.
"""
import configparser
from pathlib import Path
from typing import Any, Dict, Optional, Union
import os
from .config_data import (
    GeneralConfig, CameraConfig, ImageRecognitionConfig, ColorAnalysisConfig,
    PhilipsHueConfig, ActionsConfig, StorageManagementConfig, NASConfig,
    NotificationsConfig, TimeManagementConfig, PathsConfig, TriggerConfig
)


class ConfigManager:
    """
    Configuration manager with type-safe configuration loading and validation.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration file. Defaults to 'config.ini' in current directory.
        """
        self.config_path = config_path or Path("config.ini")
        self.config_parser = configparser.ConfigParser()
        self.config_data = GeneralConfig()
        
        # Create a default config if it doesn't exist
        if not self.config_path.exists():
            self._create_default_config()
        
        # Load configuration
        self.load()
        
    def _create_default_config(self):
        """Create a default configuration file."""
        # Create default config using the GeneralConfig dataclass values
        config = configparser.ConfigParser()
        
        # Time Management section
        config['TimeManagement'] = {
            'start_time': self.config_data.time_management.start_time,
            'end_time': self.config_data.time_management.end_time,
            'data_management_time': self.config_data.time_management.data_management_time
        }
        
        # Philips Hue section
        config['PhilipsHue'] = {
            'bridge_ip': self.config_data.philips_hue.bridge_ip,
            'app_key': self.config_data.philips_hue.app_key,
            'light_ids': ','.join(self.config_data.philips_hue.light_ids)
        }
        
        # Camera section
        low_res_str = f"{self.config_data.camera.low_resolution[0]},{self.config_data.camera.low_resolution[1]}"
        high_res_str = f"{self.config_data.camera.high_resolution[0]},{self.config_data.camera.high_resolution[1]}"
        config['Camera'] = {
            'camera_index': str(self.config_data.camera.camera_index),
            'low_resolution': low_res_str,
            'high_resolution': high_res_str,
            'fps_low': str(self.config_data.camera.fps_low),
            'fps_high': str(self.config_data.camera.fps_high)
        }
        
        # Trigger section
        config['Trigger'] = {
            'brightness_threshold': str(self.config_data.trigger.brightness_threshold),
            'brightness_pixel_percentage': str(self.config_data.trigger.brightness_pixel_percentage)
        }
        
        # Paths section
        config['Paths'] = {
            'base_storage_path': self.config_data.paths.base_storage_path,
            'sound_directory': self.config_data.paths.sound_directory
        }
        
        # NAS section
        config['NAS'] = {
            'nas_ip': self.config_data.nas.nas_ip,
            'nas_user': self.config_data.nas.nas_user,
            'nas_destination_path': self.config_data.nas.nas_destination_path,
            'nas_windows_share': self.config_data.nas.nas_windows_share
        }
        
        # Image Recognition section
        config['ImageRecognition'] = {
            'yolo_model_path': self.config_data.image_recognition.yolo_model_path,
            'inference_device': self.config_data.image_recognition.inference_device,
            'cat_class_id': str(self.config_data.image_recognition.cat_class_id),
            'cat_confidence_threshold': str(self.config_data.image_recognition.cat_confidence_threshold)
        }
        
        # Color Analysis section
        lower_hsv_str = f"{self.config_data.color_analysis.lower_black_hsv[0]},{self.config_data.color_analysis.lower_black_hsv[1]},{self.config_data.color_analysis.lower_black_hsv[2]}"
        upper_hsv_str = f"{self.config_data.color_analysis.upper_black_hsv[0]},{self.config_data.color_analysis.upper_black_hsv[1]},{self.config_data.color_analysis.upper_black_hsv[2]}"
        config['ColorAnalysis'] = {
            'lower_black_hsv': lower_hsv_str,
            'upper_black_hsv': upper_hsv_str,
            'black_pixel_threshold': str(self.config_data.color_analysis.black_pixel_threshold),
            'analysis_method': self.config_data.color_analysis.analysis_method,
            'ollama_host': self.config_data.color_analysis.ollama_host,
            'ollama_model': self.config_data.color_analysis.ollama_model,
            'ollama_prompt': self.config_data.color_analysis.ollama_prompt
        }
        
        # Actions section
        config['Actions'] = {
            'intruder_light_minutes': str(self.config_data.actions.intruder_light_minutes),
            'show_live_window': str(self.config_data.actions.show_live_window).lower()
        }
        
        # Storage Management section
        config['StorageManagement'] = {
            'min_free_space_gb': str(self.config_data.storage_management.min_free_space_gb),
            'max_file_age_days': str(self.config_data.storage_management.max_file_age_days)
        }
        
        # Notifications section
        config['Notifications'] = {
            'enabled': str(self.config_data.notifications.enabled).lower(),
            'service': self.config_data.notifications.service,
            'email_subject': self.config_data.notifications.email_subject,
            'email_from': self.config_data.notifications.email_from,
            'email_to': self.config_data.notifications.email_to,
            'smtp_server': self.config_data.notifications.smtp_server,
            'smtp_port': str(self.config_data.notifications.smtp_port),
            'smtp_user': self.config_data.notifications.smtp_user,
            'smtp_password': self.config_data.notifications.smtp_password,
            'telegram_bot_token': self.config_data.notifications.telegram_bot_token,
            'telegram_chat_id': self.config_data.notifications.telegram_chat_id
        }
        
        # Backend section
        config['Backend'] = {
            'cpu_workers': str(self.config_data.cpu_workers),
            'engine': self.config_data.inference_engine
        }
        
        # Write the default configuration to file
        with open(self.config_path, 'w') as configfile:
            config.write(configfile)
    
    def load(self) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Returns:
            Dictionary containing configuration data
        """
        # Read the configuration file
        self.config_parser.read(self.config_path)
        
        # Update the config_data based on loaded values
        self._load_camera_config()
        self._load_image_recognition_config()
        self._load_color_analysis_config()
        self._load_philips_hue_config()
        self._load_actions_config()
        self._load_storage_management_config()
        self._load_nas_config()
        self._load_notifications_config()
        self._load_time_management_config()
        self._load_paths_config()
        self._load_trigger_config()
        self._load_backend_config()
        
        # Validate configuration after loading
        if not self.validate():
            raise ValueError("Configuration validation failed")
        
        # Return a dictionary representation
        return self._to_dict()
    
    def _load_camera_config(self):
        """Load camera configuration from config file."""
        if 'Camera' in self.config_parser:
            camera_section = self.config_parser['Camera']
            
            # Parse low resolution (format: width,height)
            low_res_str = camera_section.get('low_resolution', '640,480')
            try:
                low_res = tuple(map(int, low_res_str.split(',')))
                if len(low_res) == 2:
                    self.config_data.camera.low_resolution = low_res
            except ValueError:
                pass  # Use default
            
            # Parse high resolution (format: width,height)
            high_res_str = camera_section.get('high_resolution', '1920,1080')
            try:
                high_res = tuple(map(int, high_res_str.split(',')))
                if len(high_res) == 2:
                    self.config_data.camera.high_resolution = high_res
            except ValueError:
                pass  # Use default
            
            # Parse other values
            self.config_data.camera.camera_index = camera_section.getint('camera_index', fallback=self.config_data.camera.camera_index)
            self.config_data.camera.fps_low = camera_section.getint('fps_low', fallback=self.config_data.camera.fps_low)
            self.config_data.camera.fps_high = camera_section.getint('fps_high', fallback=self.config_data.camera.fps_high)
    
    def _load_image_recognition_config(self):
        """Load image recognition configuration from config file."""
        if 'ImageRecognition' in self.config_parser:
            ir_section = self.config_parser['ImageRecognition']
            
            self.config_data.image_recognition.yolo_model_path = ir_section.get('yolo_model_path', fallback=self.config_data.image_recognition.yolo_model_path)
            self.config_data.image_recognition.inference_device = ir_section.get('inference_device', fallback=self.config_data.image_recognition.inference_device)
            self.config_data.image_recognition.cat_class_id = ir_section.getint('cat_class_id', fallback=self.config_data.image_recognition.cat_class_id)
            self.config_data.image_recognition.cat_confidence_threshold = ir_section.getfloat('cat_confidence_threshold', fallback=self.config_data.image_recognition.cat_confidence_threshold)
    
    def _load_color_analysis_config(self):
        """Load color analysis configuration from config file."""
        if 'ColorAnalysis' in self.config_parser:
            ca_section = self.config_parser['ColorAnalysis']
            
            # Parse lower HSV (format: h,s,v)
            lower_hsv_str = ca_section.get('lower_black_hsv', '0,0,0')
            try:
                lower_hsv = tuple(map(int, lower_hsv_str.split(',')))
                if len(lower_hsv) == 3:
                    self.config_data.color_analysis.lower_black_hsv = lower_hsv
            except ValueError:
                pass  # Use default
            
            # Parse upper HSV (format: h,s,v)
            upper_hsv_str = ca_section.get('upper_black_hsv', '180,255,60')
            try:
                upper_hsv = tuple(map(int, upper_hsv_str.split(',')))
                if len(upper_hsv) == 3:
                    self.config_data.color_analysis.upper_black_hsv = upper_hsv
            except ValueError:
                pass  # Use default
            
            # Parse other values
            self.config_data.color_analysis.black_pixel_threshold = ca_section.getfloat('black_pixel_threshold', fallback=self.config_data.color_analysis.black_pixel_threshold)
            self.config_data.color_analysis.analysis_method = ca_section.get('analysis_method', fallback=self.config_data.color_analysis.analysis_method)
            self.config_data.color_analysis.ollama_host = ca_section.get('ollama_host', fallback=self.config_data.color_analysis.ollama_host)
            self.config_data.color_analysis.ollama_model = ca_section.get('ollama_model', fallback=self.config_data.color_analysis.ollama_model)
            self.config_data.color_analysis.ollama_prompt = ca_section.get('ollama_prompt', fallback=self.config_data.color_analysis.ollama_prompt)
    
    def _load_philips_hue_config(self):
        """Load Philips Hue configuration from config file."""
        if 'PhilipsHue' in self.config_parser:
            hue_section = self.config_parser['PhilipsHue']
            
            self.config_data.philips_hue.bridge_ip = hue_section.get('bridge_ip', fallback=self.config_data.philips_hue.bridge_ip)
            self.config_data.philips_hue.app_key = hue_section.get('app_key', fallback=self.config_data.philips_hue.app_key)
            
            # Parse light IDs
            light_ids_str = hue_section.get('light_ids', '')
            if light_ids_str:
                self.config_data.philips_hue.light_ids = [id.strip() for id in light_ids_str.split(',') if id.strip()]
    
    def _load_actions_config(self):
        """Load actions configuration from config file."""
        if 'Actions' in self.config_parser:
            actions_section = self.config_parser['Actions']
            
            self.config_data.actions.intruder_light_minutes = actions_section.getfloat('intruder_light_minutes', fallback=self.config_data.actions.intruder_light_minutes)
            self.config_data.actions.show_live_window = actions_section.getboolean('show_live_window', fallback=self.config_data.actions.show_live_window)
    
    def _load_storage_management_config(self):
        """Load storage management configuration from config file."""
        if 'StorageManagement' in self.config_parser:
            sm_section = self.config_parser['StorageManagement']
            
            self.config_data.storage_management.min_free_space_gb = sm_section.getint('min_free_space_gb', fallback=self.config_data.storage_management.min_free_space_gb)
            self.config_data.storage_management.max_file_age_days = sm_section.getint('max_file_age_days', fallback=self.config_data.storage_management.max_file_age_days)
    
    def _load_nas_config(self):
        """Load NAS configuration from config file."""
        if 'NAS' in self.config_parser:
            nas_section = self.config_parser['NAS']
            
            self.config_data.nas.nas_ip = nas_section.get('nas_ip', fallback=self.config_data.nas.nas_ip)
            self.config_data.nas.nas_user = nas_section.get('nas_user', fallback=self.config_data.nas.nas_user)
            self.config_data.nas.nas_destination_path = nas_section.get('nas_destination_path', fallback=self.config_data.nas.nas_destination_path)
            self.config_data.nas.nas_windows_share = nas_section.get('nas_windows_share', fallback=self.config_data.nas.nas_windows_share)
    
    def _load_notifications_config(self):
        """Load notifications configuration from config file."""
        if 'Notifications' in self.config_parser:
            notif_section = self.config_parser['Notifications']
            
            self.config_data.notifications.enabled = notif_section.getboolean('enabled', fallback=self.config_data.notifications.enabled)
            self.config_data.notifications.service = notif_section.get('service', fallback=self.config_data.notifications.service)
            self.config_data.notifications.email_subject = notif_section.get('email_subject', fallback=self.config_data.notifications.email_subject)
            self.config_data.notifications.email_from = notif_section.get('email_from', fallback=self.config_data.notifications.email_from)
            self.config_data.notifications.email_to = notif_section.get('email_to', fallback=self.config_data.notifications.email_to)
            self.config_data.notifications.smtp_server = notif_section.get('smtp_server', fallback=self.config_data.notifications.smtp_server)
            self.config_data.notifications.smtp_port = notif_section.getint('smtp_port', fallback=self.config_data.notifications.smtp_port)
            self.config_data.notifications.smtp_user = notif_section.get('smtp_user', fallback=self.config_data.notifications.smtp_user)
            self.config_data.notifications.smtp_password = notif_section.get('smtp_password', fallback=self.config_data.notifications.smtp_password)
            self.config_data.notifications.telegram_bot_token = notif_section.get('telegram_bot_token', fallback=self.config_data.notifications.telegram_bot_token)
            self.config_data.notifications.telegram_chat_id = notif_section.get('telegram_chat_id', fallback=self.config_data.notifications.telegram_chat_id)
    
    def _load_time_management_config(self):
        """Load time management configuration from config file."""
        if 'TimeManagement' in self.config_parser:
            tm_section = self.config_parser['TimeManagement']
            
            self.config_data.time_management.start_time = tm_section.get('start_time', fallback=self.config_data.time_management.start_time)
            self.config_data.time_management.end_time = tm_section.get('end_time', fallback=self.config_data.time_management.end_time)
            self.config_data.time_management.data_management_time = tm_section.get('data_management_time', fallback=self.config_data.time_management.data_management_time)
    
    def _load_paths_config(self):
        """Load paths configuration from config file."""
        if 'Paths' in self.config_parser:
            paths_section = self.config_parser['Paths']
            
            self.config_data.paths.base_storage_path = paths_section.get('base_storage_path', fallback=self.config_data.paths.base_storage_path)
            self.config_data.paths.sound_directory = paths_section.get('sound_directory', fallback=self.config_data.paths.sound_directory)
    
    def _load_trigger_config(self):
        """Load trigger configuration from config file."""
        if 'Trigger' in self.config_parser:
            trigger_section = self.config_parser['Trigger']
            
            self.config_data.trigger.brightness_threshold = trigger_section.getint('brightness_threshold', fallback=self.config_data.trigger.brightness_threshold)
            self.config_data.trigger.brightness_pixel_percentage = trigger_section.getfloat('brightness_pixel_percentage', fallback=self.config_data.trigger.brightness_pixel_percentage)
    
    def _load_backend_config(self):
        """Load backend configuration from config file."""
        if 'Backend' in self.config_parser:
            backend_section = self.config_parser['Backend']
            
            self.config_data.cpu_workers = backend_section.getint('cpu_workers', fallback=self.config_data.cpu_workers)
            self.config_data.inference_engine = backend_section.get('engine', fallback=self.config_data.inference_engine)
    
    def save(self, config: Optional[Dict[str, Any]] = None):
        """
        Save configuration to file.
        
        Args:
            config: Configuration data to save. If None, saves current config_data.
        """
        if config is not None:
            # If a config dict is provided, update our config_data accordingly
            # This would require implementing the reverse of load methods
            # For now, we'll just save the current config_data
            pass
        
        # Write the current configuration to file
        config = configparser.ConfigParser()
        
        # Create sections and populate with current values
        config['TimeManagement'] = {
            'start_time': self.config_data.time_management.start_time,
            'end_time': self.config_data.time_management.end_time,
            'data_management_time': self.config_data.time_management.data_management_time
        }
        
        config['PhilipsHue'] = {
            'bridge_ip': self.config_data.philips_hue.bridge_ip,
            'app_key': self.config_data.philips_hue.app_key,
            'light_ids': ','.join(self.config_data.philips_hue.light_ids)
        }
        
        # Low and high resolution as strings
        low_res_str = f"{self.config_data.camera.low_resolution[0]},{self.config_data.camera.low_resolution[1]}"
        high_res_str = f"{self.config_data.camera.high_resolution[0]},{self.config_data.camera.high_resolution[1]}"
        config['Camera'] = {
            'camera_index': str(self.config_data.camera.camera_index),
            'low_resolution': low_res_str,
            'high_resolution': high_res_str,
            'fps_low': str(self.config_data.camera.fps_low),
            'fps_high': str(self.config_data.camera.fps_high)
        }
        
        config['Trigger'] = {
            'brightness_threshold': str(self.config_data.trigger.brightness_threshold),
            'brightness_pixel_percentage': str(self.config_data.trigger.brightness_pixel_percentage)
        }
        
        config['Paths'] = {
            'base_storage_path': self.config_data.paths.base_storage_path,
            'sound_directory': self.config_data.paths.sound_directory
        }
        
        config['NAS'] = {
            'nas_ip': self.config_data.nas.nas_ip,
            'nas_user': self.config_data.nas.nas_user,
            'nas_destination_path': self.config_data.nas.nas_destination_path,
            'nas_windows_share': self.config_data.nas.nas_windows_share
        }
        
        config['ImageRecognition'] = {
            'yolo_model_path': self.config_data.image_recognition.yolo_model_path,
            'inference_device': self.config_data.image_recognition.inference_device,
            'cat_class_id': str(self.config_data.image_recognition.cat_class_id),
            'cat_confidence_threshold': str(self.config_data.image_recognition.cat_confidence_threshold)
        }
        
        # HSV values as strings
        lower_hsv_str = f"{self.config_data.color_analysis.lower_black_hsv[0]},{self.config_data.color_analysis.lower_black_hsv[1]},{self.config_data.color_analysis.lower_black_hsv[2]}"
        upper_hsv_str = f"{self.config_data.color_analysis.upper_black_hsv[0]},{self.config_data.color_analysis.upper_black_hsv[1]},{self.config_data.color_analysis.upper_black_hsv[2]}"
        config['ColorAnalysis'] = {
            'lower_black_hsv': lower_hsv_str,
            'upper_black_hsv': upper_hsv_str,
            'black_pixel_threshold': str(self.config_data.color_analysis.black_pixel_threshold),
            'analysis_method': self.config_data.color_analysis.analysis_method,
            'ollama_host': self.config_data.color_analysis.ollama_host,
            'ollama_model': self.config_data.color_analysis.ollama_model,
            'ollama_prompt': self.config_data.color_analysis.ollama_prompt
        }
        
        config['Actions'] = {
            'intruder_light_minutes': str(self.config_data.actions.intruder_light_minutes),
            'show_live_window': str(self.config_data.actions.show_live_window).lower()
        }
        
        config['StorageManagement'] = {
            'min_free_space_gb': str(self.config_data.storage_management.min_free_space_gb),
            'max_file_age_days': str(self.config_data.storage_management.max_file_age_days)
        }
        
        config['Notifications'] = {
            'enabled': str(self.config_data.notifications.enabled).lower(),
            'service': self.config_data.notifications.service,
            'email_subject': self.config_data.notifications.email_subject,
            'email_from': self.config_data.notifications.email_from,
            'email_to': self.config_data.notifications.email_to,
            'smtp_server': self.config_data.notifications.smtp_server,
            'smtp_port': str(self.config_data.notifications.smtp_port),
            'smtp_user': self.config_data.notifications.smtp_user,
            'smtp_password': self.config_data.notifications.smtp_password,
            'telegram_bot_token': self.config_data.notifications.telegram_bot_token,
            'telegram_chat_id': self.config_data.notifications.telegram_chat_id
        }
        
        config['Backend'] = {
            'cpu_workers': str(self.config_data.cpu_workers),
            'engine': self.config_data.inference_engine
        }
        
        # Write to file
        with open(self.config_path, 'w') as configfile:
            config.write(configfile)
    
    def validate(self) -> bool:
        """
        Validate the configuration.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        # Check that required values are present and valid
        validation_errors = []
        
        # Validate camera configuration
        if self.config_data.camera.camera_index < 0:
            validation_errors.append("Camera index must be non-negative")
        
        if len(self.config_data.camera.low_resolution) != 2 or any(x <= 0 for x in self.config_data.camera.low_resolution):
            validation_errors.append("Low resolution must be a tuple of positive integers (width, height)")
        
        if len(self.config_data.camera.high_resolution) != 2 or any(x <= 0 for x in self.config_data.camera.high_resolution):
            validation_errors.append("High resolution must be a tuple of positive integers (width, height)")
        
        if self.config_data.camera.fps_low <= 0:
            validation_errors.append("FPS low must be positive")
        
        if self.config_data.camera.fps_high <= 0:
            validation_errors.append("FPS high must be positive")
        
        # Validate image recognition configuration
        if not self.config_data.image_recognition.yolo_model_path:
            validation_errors.append("YOLO model path is required")
        
        if self.config_data.image_recognition.inference_device not in ["cpu", "gpu", "cuda:0", "cuda:1"]:
            validation_errors.append("Inference device must be one of: cpu, gpu, cuda:0, cuda:1")
        
        if self.config_data.image_recognition.cat_class_id < 0:
            validation_errors.append("Cat class ID must be non-negative")
        
        if not (0.0 <= self.config_data.image_recognition.cat_confidence_threshold <= 1.0):
            validation_errors.append("Cat confidence threshold must be between 0.0 and 1.0")
        
        # Validate color analysis configuration
        if len(self.config_data.color_analysis.lower_black_hsv) != 3 or any(x < 0 or (i < 2 and x > 180) or (i == 2 and x > 255) for i, x in enumerate(self.config_data.color_analysis.lower_black_hsv)):
            validation_errors.append("Lower black HSV must be a tuple of 3 values (H: 0-180, S&V: 0-255)")
        
        if len(self.config_data.color_analysis.upper_black_hsv) != 3 or any(x < 0 or (i < 2 and x > 180) or (i == 2 and x > 255) for i, x in enumerate(self.config_data.color_analysis.upper_black_hsv)):
            validation_errors.append("Upper black HSV must be a tuple of 3 values (H: 0-180, S&V: 0-255)")
        
        if not (0.0 <= self.config_data.color_analysis.black_pixel_threshold <= 1.0):
            validation_errors.append("Black pixel threshold must be between 0.0 and 1.0")
        
        if self.config_data.color_analysis.analysis_method not in ["hsv", "llm"]:
            validation_errors.append("Analysis method must be either 'hsv' or 'llm'")
        
        # Validate actions configuration
        if self.config_data.actions.intruder_light_minutes <= 0:
            validation_errors.append("Intruder light minutes must be positive")
        
        # Validate storage management configuration
        if self.config_data.storage_management.min_free_space_gb <= 0:
            validation_errors.append("Min free space GB must be positive")
        
        if self.config_data.storage_management.max_file_age_days <= 0:
            validation_errors.append("Max file age days must be positive")
        
        # Validate backend configuration
        if self.config_data.cpu_workers <= 0:
            validation_errors.append("CPU workers must be positive")
        
        if self.config_data.inference_engine not in ["onnx", "openvino", "pt", "safetensor", "coreml"]:
            validation_errors.append("Inference engine must be one of: onnx, openvino, pt, safetensor, coreml")
        
        # Validate time management configuration (basic format check)
        try:
            start_time_parts = self.config_data.time_management.start_time.split(':')
            if len(start_time_parts) != 2 or not (0 <= int(start_time_parts[0]) <= 23) or not (0 <= int(start_time_parts[1]) <= 59):
                validation_errors.append("Start time must be in HH:MM format")
        except (ValueError, IndexError):
            validation_errors.append("Start time must be in HH:MM format")
        
        try:
            end_time_parts = self.config_data.time_management.end_time.split(':')
            if len(end_time_parts) != 2 or not (0 <= int(end_time_parts[0]) <= 23) or not (0 <= int(end_time_parts[1]) <= 59):
                validation_errors.append("End time must be in HH:MM format")
        except (ValueError, IndexError):
            validation_errors.append("End time must be in HH:MM format")
        
        # Print validation errors if any
        if validation_errors:
            for error in validation_errors:
                print(f"Configuration validation error: {error}")
            return False
        
        return True
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key using dot notation (e.g., 'camera.camera_index').
        
        Args:
            key: Configuration key in dot notation
            default: Default value if key is not found
            
        Returns:
            Configuration value
        """
        # Split the key by dots to navigate the config structure
        parts = key.split('.')
        
        # Start with the config_data object
        current = self.config_data
        
        # Navigate through the parts
        for part in parts:
            try:
                # Handle attributes of dataclass objects
                current = getattr(current, part)
            except AttributeError:
                # If the attribute doesn't exist, return the default
                return default
        
        return current
    
    def set(self, key: str, value: Any):
        """
        Set a configuration value by key using dot notation (e.g., 'camera.camera_index').
        
        Args:
            key: Configuration key in dot notation
            value: Configuration value to set
        """
        # Split the key by dots to navigate the config structure
        parts = key.split('.')
        
        # Start with the config_data object
        current = self.config_data
        
        # Navigate to the parent of the target attribute
        for part in parts[:-1]:
            try:
                current = getattr(current, part)
            except AttributeError:
                # If the path doesn't exist, we can't set the value
                raise AttributeError(f"Configuration path '{'.'.join(parts[:-1])}' does not exist")
        
        # Set the final attribute
        setattr(current, parts[-1], value)
    
    def _to_dict(self) -> Dict[str, Any]:
        """
        Convert the configuration data to a dictionary.
        
        Returns:
            Dictionary representation of the configuration
        """
        return {
            'camera': {
                'camera_index': self.config_data.camera.camera_index,
                'low_resolution': self.config_data.camera.low_resolution,
                'high_resolution': self.config_data.camera.high_resolution,
                'fps_low': self.config_data.camera.fps_low,
                'fps_high': self.config_data.camera.fps_high
            },
            'image_recognition': {
                'yolo_model_path': self.config_data.image_recognition.yolo_model_path,
                'inference_device': self.config_data.image_recognition.inference_device,
                'cat_class_id': self.config_data.image_recognition.cat_class_id,
                'cat_confidence_threshold': self.config_data.image_recognition.cat_confidence_threshold
            },
            'color_analysis': {
                'lower_black_hsv': self.config_data.color_analysis.lower_black_hsv,
                'upper_black_hsv': self.config_data.color_analysis.upper_black_hsv,
                'black_pixel_threshold': self.config_data.color_analysis.black_pixel_threshold,
                'analysis_method': self.config_data.color_analysis.analysis_method,
                'ollama_host': self.config_data.color_analysis.ollama_host,
                'ollama_model': self.config_data.color_analysis.ollama_model,
                'ollama_prompt': self.config_data.color_analysis.ollama_prompt
            },
            'philips_hue': {
                'bridge_ip': self.config_data.philips_hue.bridge_ip,
                'app_key': self.config_data.philips_hue.app_key,
                'light_ids': self.config_data.philips_hue.light_ids
            },
            'actions': {
                'intruder_light_minutes': self.config_data.actions.intruder_light_minutes,
                'show_live_window': self.config_data.actions.show_live_window
            },
            'storage_management': {
                'min_free_space_gb': self.config_data.storage_management.min_free_space_gb,
                'max_file_age_days': self.config_data.storage_management.max_file_age_days
            },
            'nas': {
                'nas_ip': self.config_data.nas.nas_ip,
                'nas_user': self.config_data.nas.nas_user,
                'nas_destination_path': self.config_data.nas.nas_destination_path,
                'nas_windows_share': self.config_data.nas.nas_windows_share
            },
            'notifications': {
                'enabled': self.config_data.notifications.enabled,
                'service': self.config_data.notifications.service,
                'email_subject': self.config_data.notifications.email_subject,
                'email_from': self.config_data.notifications.email_from,
                'email_to': self.config_data.notifications.email_to,
                'smtp_server': self.config_data.notifications.smtp_server,
                'smtp_port': self.config_data.notifications.smtp_port,
                'smtp_user': self.config_data.notifications.smtp_user,
                'smtp_password': self.config_data.notifications.smtp_password,
                'telegram_bot_token': self.config_data.notifications.telegram_bot_token,
                'telegram_chat_id': self.config_data.notifications.telegram_chat_id
            },
            'time_management': {
                'start_time': self.config_data.time_management.start_time,
                'end_time': self.config_data.time_management.end_time,
                'data_management_time': self.config_data.time_management.data_management_time
            },
            'paths': {
                'base_storage_path': self.config_data.paths.base_storage_path,
                'sound_directory': self.config_data.paths.sound_directory
            },
            'trigger': {
                'brightness_threshold': self.config_data.trigger.brightness_threshold,
                'brightness_pixel_percentage': self.config_data.trigger.brightness_pixel_percentage
            },
            'backend': {
                'cpu_workers': self.config_data.cpu_workers,
                'engine': self.config_data.inference_engine
            }
        }