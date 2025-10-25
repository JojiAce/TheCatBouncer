"""
Configuration data classes with type hints.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class CameraConfig:
    """Camera configuration settings."""
    camera_index: int = 0
    low_resolution: tuple = (640, 480)
    high_resolution: tuple = (1920, 1080)
    fps_low: int = 5
    fps_high: int = 30


@dataclass
class ImageRecognitionConfig:
    """Image recognition configuration settings."""
    yolo_model_path: str = "yolo11_openvino_model_paths/yolo11s_openvino_model"
    inference_device: str = "gpu"  # cpu, gpu, cuda:0
    cat_class_id: int = 15
    cat_confidence_threshold: float = 0.8


@dataclass
class ColorAnalysisConfig:
    """Color analysis configuration settings."""
    lower_black_hsv: tuple = (0, 0, 0)
    upper_black_hsv: tuple = (180, 255, 60)
    black_pixel_threshold: float = 0.5
    analysis_method: str = "hsv"  # hsv, llm
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llava"
    ollama_prompt: str = "Is the cat in this image black? Answer with only \"yes\" or \"no\"."


@dataclass
class PhilipsHueConfig:
    """Philips Hue configuration settings."""
    bridge_ip: str = ""
    app_key: str = ""
    light_ids: List[str] = field(default_factory=list)


@dataclass
class ActionsConfig:
    """Actions configuration settings."""
    intruder_light_minutes: float = 4.0
    show_live_window: bool = True


@dataclass
class StorageManagementConfig:
    """Storage management configuration settings."""
    min_free_space_gb: int = 10
    max_file_age_days: int = 2


@dataclass
class NASConfig:
    """NAS backup configuration settings."""
    nas_ip: str = ""
    nas_user: str = ""
    nas_destination_path: str = "/path/to/backup/folder"
    nas_windows_share: str = "YOUR_WINDOWS_SHARE_NAME"


@dataclass
class NotificationsConfig:
    """Notifications configuration settings."""
    enabled: bool = False
    service: str = "email"  # email, telegram
    email_subject: str = "Intruder Alert!"
    email_from: str = "your_email@example.com"
    email_to: str = "recipient@example.com"
    smtp_server: str = "smtp.example.com"
    smtp_port: int = 587
    smtp_user: str = "your_email@example.com"
    smtp_password: str = "your_app_password"
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""


@dataclass
class TimeManagementConfig:
    """Time management configuration settings."""
    start_time: str = "18:00"
    end_time: str = "07:00"
    data_management_time: str = "07:10"


@dataclass
class PathsConfig:
    """Paths configuration settings."""
    base_storage_path: str = "CatDetectorData"
    sound_directory: str = "cat_scare_sound"


@dataclass
class TriggerConfig:
    """Trigger configuration settings."""
    brightness_threshold: int = 220
    brightness_pixel_percentage: float = 0.01


@dataclass
class GeneralConfig:
    """General application configuration."""
    camera: CameraConfig = field(default_factory=CameraConfig)
    image_recognition: ImageRecognitionConfig = field(default_factory=ImageRecognitionConfig)
    color_analysis: ColorAnalysisConfig = field(default_factory=ColorAnalysisConfig)
    philips_hue: PhilipsHueConfig = field(default_factory=PhilipsHueConfig)
    actions: ActionsConfig = field(default_factory=ActionsConfig)
    storage_management: StorageManagementConfig = field(default_factory=StorageManagementConfig)
    nas: NASConfig = field(default_factory=NASConfig)
    notifications: NotificationsConfig = field(default_factory=NotificationsConfig)
    time_management: TimeManagementConfig = field(default_factory=TimeManagementConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    trigger: TriggerConfig = field(default_factory=TriggerConfig)
    
    # Performance settings
    cpu_workers: int = 4
    inference_engine: str = "onnx"  # onnx, openvino, pt, safetensor, coreml