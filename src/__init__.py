"""
TheCatBouncer - AI-powered pet access control system

Top-level package initialization.
"""

__version__ = "2.0.0"
__author__ = "JojiAce"
__license__ = "MIT"

# Import key components for easy access (only those without problematic dependencies at import time)
from .config.config_data import GeneralConfig
from .config.config_manager import ConfigManager
from .interfaces.config import ConfigManager as ConfigManagerInterface
from .interfaces.inference import InferenceEngine as InferenceEngineInterface
from .interfaces.monitoring import ColorAnalyzer as ColorAnalyzerInterface
from .interfaces.scheduling import ScheduleManager as ScheduleManagerInterface

__all__ = [
    # Configuration
    'GeneralConfig',
    'ConfigManager',
    'ConfigManagerInterface',
    
    # Interfaces
    'InferenceEngineInterface',
    'ColorAnalyzerInterface',
    'ScheduleManagerInterface',
]