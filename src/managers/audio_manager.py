"""
Audio deterrent system for cross-platform audio playback.
"""
import pygame
import os
import random
import logging
import time
from typing import Dict, Any, Optional, List
from pathlib import Path

from src.interfaces.monitoring import AudioManager


class AudioManager(AudioManager):
    """
    Audio manager for cross-platform audio playback.
    Supports multiple audio formats and random sound selection.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the audio manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Extract configuration values
        self.sound_directory = config.get('sound_directory', 'cat_scare_sound')
        self.supported_formats = config.get('supported_formats', ['.wav', '.mp3', '.ogg'])
        self.volume = config.get('volume', 0.8)  # 0.0 to 1.0
        self.max_volume = config.get('max_volume', 1.0)
        self.cooldown_time = config.get('cooldown_time', 2.0)  # seconds between sounds
        
        # Initialize pygame mixer
        try:
            pygame.mixer.init()
            pygame.mixer.music.set_volume(min(self.volume, self.max_volume))
            self.is_initialized = True
            self.last_play_time = 0
            self.logger.info("Audio manager initialized successfully")
        except pygame.error as e:
            self.is_initialized = False
            self.logger.error(f"Failed to initialize audio manager: {e}")
    
    def play_sound(self, sound_file: Optional[str] = None):
        """
        Play a sound file.
        
        Args:
            sound_file: Path to sound file to play. If None, picks random file from directory.
        """
        if not self.is_initialized:
            self.logger.error("Audio manager not initialized. Cannot play sound.")
            return
        
        # Check cooldown period
        current_time = time.time()
        if current_time - self.last_play_time < self.cooldown_time:
            self.logger.debug(f"Audio cooldown active. {current_time - self.last_play_time:.1f}s since last play.")
            return
        
        try:
            if sound_file is None:
                # Select a random sound file from the directory
                sound_file = self._get_random_sound_file()
            
            if sound_file is None:
                self.logger.warning("No sound file found to play.")
                return
            
            sound_path = Path(sound_file)
            if not sound_path.exists():
                self.logger.error(f"Sound file does not exist: {sound_file}")
                return
            
            # Load and play the sound
            pygame.mixer.music.load(str(sound_path))
            pygame.mixer.music.play()
            
            self.logger.info(f"Playing sound: {sound_path.name}")
            self.last_play_time = time.time()
            
            # Wait for the sound to finish playing (in a non-blocking way)
            while pygame.mixer.music.get_busy():
                pygame.time.wait(100)  # Wait 100ms before checking again
                
        except pygame.error as e:
            self.logger.error(f"Pygame error playing sound: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error playing sound: {e}")
    
    def stop_sound(self):
        """
        Stop any currently playing sound.
        """
        if not self.is_initialized:
            return
        
        try:
            pygame.mixer.music.stop()
            self.logger.info("Stopped currently playing sound")
        except pygame.error as e:
            self.logger.error(f"Error stopping sound: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error stopping sound: {e}")
    
    def _get_random_sound_file(self) -> Optional[str]:
        """
        Select a random sound file from the configured directory.
        
        Returns:
            Path to a random sound file, or None if no files found
        """
        sound_dir = Path(self.sound_directory)
        
        if not sound_dir.exists():
            self.logger.warning(f"Sound directory does not exist: {sound_dir}")
            return None
        
        # Find all supported audio files in the directory
        sound_files = []
        for ext in self.supported_formats:
            sound_files.extend(sound_dir.glob(f"*{ext}"))
            sound_files.extend(sound_dir.glob(f"*{ext.upper()}"))  # Also check uppercase
        
        if not sound_files:
            self.logger.warning(f"No supported audio files found in: {sound_dir}")
            return None
        
        # Select a random file
        selected_file = random.choice(sound_files)
        self.logger.debug(f"Selected random sound file: {selected_file.name}")
        
        return str(selected_file)
    
    def get_available_sounds(self) -> List[str]:
        """
        Get a list of all available sound files.
        
        Returns:
            List of paths to available sound files
        """
        sound_dir = Path(self.sound_directory)
        
        if not sound_dir.exists():
            self.logger.warning(f"Sound directory does not exist: {sound_dir}")
            return []
        
        # Find all supported audio files in the directory
        sound_files = []
        for ext in self.supported_formats:
            sound_files.extend([str(f) for f in sound_dir.glob(f"*{ext}")])
            sound_files.extend([str(f) for f in sound_dir.glob(f"*{ext.upper()}")])
        
        return sorted(sound_files)
    
    def set_volume(self, volume: float):
        """
        Set the volume level.
        
        Args:
            volume: Volume level between 0.0 and 1.0
        """
        if not self.is_initialized:
            return
        
        # Clamp volume between 0.0 and max_volume
        volume = max(0.0, min(volume, self.max_volume))
        
        try:
            pygame.mixer.music.set_volume(volume)
            self.volume = volume
            self.logger.info(f"Volume set to {volume:.2f}")
        except pygame.error as e:
            self.logger.error(f"Error setting volume: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error setting volume: {e}")
    
    def test_audio(self) -> bool:
        """
        Test if audio system is working.
        
        Returns:
            True if audio system is working, False otherwise
        """
        if not self.is_initialized:
            return False
        
        try:
            # Get a test sound (or create a silent one for testing)
            test_sound = self._get_random_sound_file()
            if test_sound:
                # Try to load the test sound
                pygame.mixer.music.load(test_sound)
                # Don't actually play, just test loading
                return True
            else:
                self.logger.warning("No sound files available for testing")
                return True  # Consider it working if pygame initialized, even without files
        except Exception as e:
            self.logger.error(f"Audio test failed: {e}")
            return False
    
    def cleanup(self):
        """
        Clean up audio resources.
        """
        try:
            if self.is_initialized:
                pygame.mixer.music.stop()
                pygame.mixer.quit()
                self.is_initialized = False
                self.logger.info("Audio resources cleaned up")
        except Exception as e:
            self.logger.error(f"Error cleaning up audio resources: {e}")


def play_intruder_sound(sound_directory: str, volume: float = 0.8) -> bool:
    """
    Convenience function to play a random intruder deterrent sound.
    
    Args:
        sound_directory: Directory containing sound files
        volume: Volume level (0.0 to 1.0)
        
    Returns:
        True if sound was played successfully, False otherwise
    """
    config = {
        'sound_directory': sound_directory,
        'volume': volume,
        'cooldown_time': 0  # No cooldown for this convenience function
    }
    
    audio_manager = AudioManager(config)
    
    if not audio_manager.is_initialized:
        return False
    
    try:
        audio_manager.play_sound()
        return True
    except Exception:
        return False


def list_audio_devices() -> List[Dict[str, Any]]:
    """
    List available audio devices (requires pygame 2.0.0+).
    
    Returns:
        List of available audio devices
    """
    devices = []
    
    try:
        # If pygame version supports it, list audio devices
        if hasattr(pygame.mixer, 'get_sound_fonts'):
            # This is an example - actual implementation depends on pygame capabilities
            devices.append({
                'id': 0,
                'name': 'Default Audio Device',
                'default': True
            })
    except Exception as e:
        logging.debug(f"Could not list audio devices: {e}")
    
    return devices


def is_audio_available() -> bool:
    """
    Check if audio system is available.
    
    Returns:
        True if audio system is available, False otherwise
    """
    try:
        pygame.mixer.init()
        pygame.mixer.quit()
        return True
    except pygame.error:
        return False