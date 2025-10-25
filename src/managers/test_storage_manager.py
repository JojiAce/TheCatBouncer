"""
Unit tests for data management and storage system.
"""
import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json
import datetime

from src.managers.storage_manager import (
    StorageManager,
    BackupService,
    create_data_backup_scheduler,
    get_old_files,
    clean_old_files
)


def test_storage_manager_initialization():
    """Test StorageManager initialization."""
    config = {
        'base_storage_path': 'test_storage',
        'min_free_space_gb': 5,
        'max_file_age_days': 3
    }
    
    manager = StorageManager(config)
    
    assert manager.base_storage_path.name == 'test_storage'
    assert manager.min_free_space_gb == 5
    assert manager.max_file_age_days == 3


def test_store_detection():
    """Test storing detection data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = {
            'base_storage_path': temp_dir,
            'min_free_space_gb': 5,
            'max_file_age_days': 3
        }
        
        manager = StorageManager(config)
        
        # Create a temporary image file
        test_image = Path(temp_dir) / 'test_image.jpg'
        test_image.write_text('dummy image content')
        
        # Store detection
        detection_data = {
            'image_path': str(test_image),
            'confidence': 0.95,
            'class_name': 'cat',
            'bbox_coords': [100, 100, 200, 200]
        }
        
        stored_path = manager.store_detection(detection_data)
        
        stored_path_obj = Path(stored_path)
        assert stored_path_obj.exists()
        assert (stored_path_obj / 'metadata.json').exists()
        
        # Verify metadata content
        with open(stored_path_obj / 'metadata.json', 'r') as f:
            metadata = json.load(f)
            assert metadata['detection_data']['confidence'] == 0.95
            assert metadata['detection_data']['class_name'] == 'cat'


def test_cleanup_old_data():
    """Test cleaning up old data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = {
            'base_storage_path': temp_dir,
            'min_free_space_gb': 5,
            'max_file_age_days': 0  # Remove all data for this test
        }
        
        manager = StorageManager(config)
        
        # Create a detection folder from the past
        past_date = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
        past_time = datetime.datetime.now().strftime('%H-%M-%S')
        old_detection_path = Path(temp_dir) / 'detections' / past_date / past_time
        old_detection_path.mkdir(parents=True, exist_ok=True)
        
        # Create a dummy file in the old detection folder
        dummy_file = old_detection_path / 'dummy.txt'
        dummy_file.write_text('dummy content')
        
        assert dummy_file.exists()
        
        # Run cleanup
        manager.cleanup_old_data()
        
        # Old data should be removed
        assert not old_detection_path.exists()


def test_get_storage_stats():
    """Test getting storage statistics."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = {
            'base_storage_path': temp_dir,
            'min_free_space_gb': 5,
            'max_file_age_days': 3
        }
        
        manager = StorageManager(config)
        
        stats = manager.get_storage_stats()
        
        assert 'total_gb' in stats
        assert 'used_gb' in stats
        assert 'free_gb' in stats
        assert 'file_count' in stats
        assert 'sufficient_space' in stats


def test_backup_service_initialization():
    """Test BackupService initialization."""
    config = {
        'nas_ip': '192.168.1.100',
        'nas_user': 'test_user',
        'nas_destination_path': '/backups/test',
        'nas_windows_share': 'TestShare',
        'backup_enabled': True,
        'backup_time': '02:00',
        'backup_interval_days': 1
    }
    
    service = BackupService(config)
    
    assert service.nas_ip == '192.168.1.100'
    assert service.backup_enabled == True
    assert service.backup_time == '02:00'


@patch('subprocess.run')
def test_backup_with_rsync_success(mock_run):
    """Test successful backup with rsync."""
    # Mock successful rsync command
    mock_result = Mock()
    mock_result.returncode = 0
    mock_result.stdout = 'transferred 1,000,000 bytes'
    mock_result.stderr = ''
    mock_run.return_value = mock_result
    
    config = {
        'nas_ip': '192.168.1.100',
        'nas_user': 'test_user',
        'nas_destination_path': '/backups/test',
        'nas_windows_share': 'TestShare',
        'backup_enabled': True
    }
    
    service = BackupService(config)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        success = service._backup_with_rsync([temp_dir], 'test_backup')
        assert success == True


@patch('platform.system', return_value='Linux')
def test_perform_backup_linux(mock_system):
    """Test perform backup on Linux."""
    config = {
        'nas_ip': '192.168.1.100',
        'nas_user': 'test_user',
        'nas_destination_path': '/backups/test',
        'nas_windows_share': 'TestShare',
        'backup_enabled': True
    }
    
    service = BackupService(config)
    
    # Mock the _backup_with_rsync method
    with patch.object(service, '_backup_with_rsync', return_value=True):
        with tempfile.TemporaryDirectory() as temp_dir:
            success = service.perform_backup([temp_dir], 'test_backup')
            assert success == True


@patch('platform.system', return_value='Windows')
def test_perform_backup_windows(mock_system):
    """Test perform backup on Windows."""
    config = {
        'nas_ip': '192.168.1.100',
        'nas_user': 'test_user',
        'nas_destination_path': '/backups/test',
        'nas_windows_share': 'TestShare',
        'backup_enabled': True
    }
    
    service = BackupService(config)
    
    # Mock the _backup_with_robocopy method
    with patch.object(service, '_backup_with_robocopy', return_value=True):
        with tempfile.TemporaryDirectory() as temp_dir:
            success = service.perform_backup([temp_dir], 'test_backup')
            assert success == True


def test_get_old_files():
    """Test getting old files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create a recent file
        recent_file = temp_path / 'recent.txt'
        recent_file.write_text('recent content')
        
        # Create an old file by setting its modification time
        old_file = temp_path / 'old.txt'
        old_file.write_text('old content')
        # Set the modification time to 2 days ago
        old_time = (datetime.datetime.now() - datetime.timedelta(days=2)).timestamp()
        os.utime(old_file, (old_time, old_time))
        
        # Get files older than 1 day
        old_files = get_old_files(temp_path, 1)
        
        assert len(old_files) == 1
        assert old_file in old_files


def test_clean_old_files():
    """Test cleaning old files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create some test files
        file1 = temp_path / 'file1.txt'
        file2 = temp_path / 'file2.txt'
        file1.write_text('content1')
        file2.write_text('content2')
        
        # Verify files exist
        assert file1.exists()
        assert file2.exists()
        
        # Clean the files
        deleted_count = clean_old_files([file1, file2])
        
        assert deleted_count == 2
        assert not file1.exists()
        assert not file2.exists()


def test_verify_backup():
    """Test backup verification."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        config = {
            'nas_ip': '192.168.1.100',
            'nas_user': 'test_user',
            'nas_destination_path': '/backups/test',
            'nas_windows_share': 'TestShare',
            'backup_enabled': True
        }
        
        service = BackupService(config)
        
        # Create a test backup directory with content
        backup_dir = temp_path / 'test_backup'
        backup_dir.mkdir()
        (backup_dir / 'test_file.txt').write_text('test content')
        
        # Verify the backup
        is_valid = service.verify_backup(str(backup_dir))
        assert is_valid == True
        
        # Verify a non-existent backup
        is_valid = service.verify_backup('/nonexistent/backup')
        assert is_valid == False


if __name__ == "__main__":
    test_storage_manager_initialization()
    test_store_detection()
    test_cleanup_old_data()
    test_get_storage_stats()
    test_backup_service_initialization()
    test_perform_backup_linux()
    test_perform_backup_windows()
    test_get_old_files()
    test_clean_old_files()
    test_verify_backup()
    print("All data management and storage system tests passed!")