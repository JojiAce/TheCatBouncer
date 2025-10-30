"""
Abstract base classes for configuration management.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ConfigData:
    """Base configuration data class."""
    pass


class ConfigManager(ABC):
    """
    Abstract base class for configuration management.
    """
    
    @abstractmethod
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration file
        """
        pass

    @abstractmethod
    def load(self) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Returns:
            Dictionary containing configuration data
        """
        pass

    @abstractmethod
    def save(self, config: Dict[str, Any]):
        """
        Save configuration to file.
        
        Args:
            config: Configuration data to save
        """
        pass

    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.
        
        Args:
            key: Configuration key
            default: Default value if key is not found
            
        Returns:
            Configuration value
        """
        pass

    @abstractmethod
    def set(self, key: str, value: Any):
        """
        Set a configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
        """
        pass

    @abstractmethod
    def validate(self) -> bool:
        """
        Validate the configuration.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        pass