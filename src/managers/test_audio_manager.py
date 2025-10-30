"""
Unit tests for audio deterrent system.
"""
import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.managers.audio_manager import AudioManager, play_intruder_sound, is_audio_available


def test_audio_manager_initialization():
    """Test AudioManager initialization."""
    config = {
        'sound_directory': 'test_sounds',
        'volume': 0.8,
        'cooldown_time': 2.0
    }
    
    # Mock pygame initialization to succeed
    with patch('pygame.mixer.init'), \
         patch('pygame.mixer.music.set_volume'), \
         patch('src.managers.audio_manager.is_audio_available', return_value=True):
        
        manager = AudioManager(config)
        
        assert manager.sound_directory == 'test_sounds'
        assert manager.volume == 0.8
        assert manager.cooldown_time == 2.0
        assert manager.is_initialized == True


def test_get_random_sound_file():
    """Test getting a random sound file."""
    # Create a temporary directory with test audio files
    with tempfile.TemporaryDirectory() as temp_dir:
        sound_dir = Path(temp_dir)
        
        # Create some test audio files
        wav_file = sound_dir / "test1.wav"
        mp3_file = sound_dir / "test2.mp3"
        ogg_file = sound_dir / "test3.ogg"
        
        wav_file.touch()
        mp3_file.touch()
        ogg_file.touch()
        
        config = {
            'sound_directory': str(sound_dir),
            'supported_formats': ['.wav', '.mp3', '.ogg'],
            'volume': 0.8
        }
        
        with patch('pygame.mixer.init'), \
             patch('pygame.mixer.music.set_volume'), \
             patch('src.managers.audio_manager.is_audio_available', return_value=True):
            
            manager = AudioManager(config)
            
            # Test that it finds a file
            random_file = manager._get_random_sound_file()
            assert random_file is not None
            assert Path(random_file).exists()


def test_get_random_sound_file_empty_directory():
    """Test getting a random sound file from empty directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = {
            'sound_directory': temp_dir,
            'supported_formats': ['.wav', '.mp3', '.ogg'],
            'volume': 0.8
        }
        
        with patch('pygame.mixer.init'), \
             patch('pygame.mixer.music.set_volume'), \
             patch('src.managers.audio_manager.is_audio_available', return_value=True):
            
            manager = AudioManager(config)
            
            # Should return None when no files exist
            random_file = manager._get_random_sound_file()
            assert random_file is None


@patch('pygame.mixer.init')
@patch('pygame.mixer.music.set_volume')
@patch('pygame.mixer.music.load')
@patch('pygame.mixer.music.play')
@patch('pygame.mixer.music.get_busy')
def test_play_sound_success(mock_get_busy, mock_play, mock_load, mock_set_volume, mock_init):
    """Test playing a sound successfully."""
    # Set up mocks
    mock_get_busy.return_value = False  # Sound finishes immediately
    
    config = {
        'sound_directory': 'test_sounds',
        'volume': 0.8
    }
    
    with patch('src.managers.audio_manager.is_audio_available', return_value=True):
        manager = AudioManager(config)
        manager.is_initialized = True  # Override for this test
        
        # Create a temporary sound file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            manager.play_sound(tmp_path)
            
            # Verify the sound was loaded and played
            mock_load.assert_called_once_with(tmp_path)
            mock_play.assert_called_once()
        finally:
            Path(tmp_path).unlink()


@patch('pygame.mixer.init')
@patch('pygame.mixer.music.set_volume')
def test_play_sound_file_not_found(mock_set_volume, mock_init):
    """Test playing a sound when file doesn't exist."""
    config = {
        'sound_directory': 'test_sounds',
        'volume': 0.8
    }
    
    with patch('src.managers.audio_manager.is_audio_available', return_value=True):
        manager = AudioManager(config)
        manager.is_initialized = True  # Override for this test
        
        # Try to play non-existent file
        manager.play_sound("/nonexistent/file.wav")


def test_stop_sound():
    """Test stopping a sound."""
    with patch('pygame.mixer.init'), \
         patch('pygame.mixer.music.set_volume'), \
         patch('pygame.mixer.music.stop'), \
         patch('src.managers.audio_manager.is_audio_available', return_value=True):
        
        config = {
            'sound_directory': 'test_sounds',
            'volume': 0.8
        }
        
        manager = AudioManager(config)
        manager.stop_sound()


def test_get_available_sounds():
    """Test getting available sounds."""
    # Create a temporary directory with test audio files
    with tempfile.TemporaryDirectory() as temp_dir:
        sound_dir = Path(temp_dir)
        
        # Create some test audio files
        wav_file = sound_dir / "test1.wav"
        mp3_file = sound_dir / "test2.mp3"
        
        wav_file.touch()
        mp3_file.touch()
        
        config = {
            'sound_directory': str(sound_dir),
            'supported_formats': ['.wav', '.mp3'],
            'volume': 0.8
        }
        
        with patch('pygame.mixer.init'), \
             patch('pygame.mixer.music.set_volume'), \
             patch('src.managers.audio_manager.is_audio_available', return_value=True):
            
            manager = AudioManager(config)
            
            available_sounds = manager.get_available_sounds()
            
            assert len(available_sounds) == 2
            assert str(wav_file) in available_sounds
            assert str(mp3_file) in available_sounds


def test_set_volume():
    """Test setting volume."""
    with patch('pygame.mixer.init'), \
         patch('pygame.mixer.music.set_volume'), \
         patch('src.managers.audio_manager.is_audio_available', return_value=True):
        
        config = {
            'sound_directory': 'test_sounds',
            'volume': 0.8,
            'max_volume': 1.0
        }
        
        manager = AudioManager(config)
        manager.set_volume(0.5)
        
        assert manager.volume == 0.5


def test_play_intruder_sound():
    """Test the play_intruder_sound convenience function."""
    with patch('pygame.mixer.init'), \
         patch('pygame.mixer.music.set_volume'), \
         patch('pygame.mixer.music.load'), \
         patch('pygame.mixer.music.play'), \
         patch('pygame.mixer.music.get_busy', return_value=False), \
         patch('src.managers.audio_manager.is_audio_available', return_value=True):
        
        # Create a temporary directory with a sound file
        with tempfile.TemporaryDirectory() as temp_dir:
            sound_file = Path(temp_dir) / "test.wav"
            sound_file.touch()
            
            # Create a subdirectory to represent the sound directory
            sound_dir = Path(temp_dir) / "sounds"
            sound_dir.mkdir()
            (sound_dir / "intruder.wav").touch()
            
            result = play_intruder_sound(str(sound_dir))
            
            assert result == True


def test_audio_cooldown():
    """Test audio cooldown functionality."""
    with patch('pygame.mixer.init'), \
         patch('pygame.mixer.music.set_volume'), \
         patch('pygame.mixer.music.load'), \
         patch('pygame.mixer.music.play'), \
         patch('pygame.mixer.music.get_busy', return_value=False), \
         patch('time.time', return_value=1000.0), \
         patch('src.managers.audio_manager.is_audio_available', return_value=True):
        
        config = {
            'sound_directory': 'test_sounds',
            'volume': 0.8,
            'cooldown_time': 5.0  # 5 second cooldown
        }
        
        manager = AudioManager(config)
        manager.is_initialized = True  # Override for this test
        
        # Create a temporary sound file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Play sound first time
            manager.play_sound(tmp_path)
            assert manager.last_play_time == 1000.0
            
            # Try to play again immediately (should be blocked by cooldown)
            with patch('time.time', return_value=1000.1):  # Only 0.1 seconds later
                manager.play_sound(tmp_path)
                # The second call should not trigger a new play due to cooldown
        finally:
            Path(tmp_path).unlink()


if __name__ == "__main__":
    test_audio_manager_initialization()
    test_get_random_sound_file()
    test_get_random_sound_file_empty_directory()
    test_play_sound_success()
    test_stop_sound()
    test_get_available_sounds()
    test_set_volume()
    test_play_intruder_sound()
    test_audio_cooldown()
    print("All audio deterrent system tests passed!")