"""
Data management and storage system with organized file storage and cross-platform backup.
"""
import os
import shutil
import datetime
import json
import logging
import subprocess
import platform
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import tempfile

from src.interfaces.monitoring import DataManagerInterface


class StorageManager(DataManagerInterface):
    """
    Storage manager for organized file storage with date-based folder organization and metadata.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the storage manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Extract configuration values
        self.base_storage_path = Path(config.get('base_storage_path', 'CatDetectorData'))
        self.min_free_space_gb = config.get('min_free_space_gb', 10)
        self.max_file_age_days = config.get('max_file_age_days', 2)
        
        # Create base directory
        self.base_storage_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Storage manager initialized. Base path: {self.base_storage_path}, "
                        f"Min free space: {self.min_free_space_gb}GB, "
                        f"Max file age: {self.max_file_age_days} days")
    
    def store_detection(self, detection_data: Dict[str, Any]) -> str:
        """
        Store detection data with organized folder structure and metadata.
        
        Args:
            detection_data: Detection information to store
            
        Returns:
            Path where data was stored
        """
        try:
            # Create timestamp-based folder structure
            now = datetime.datetime.now()
            date_folder = self.base_storage_path / 'detections' / now.strftime('%Y-%m-%d')
            time_folder = date_folder / now.strftime('%H-%M-%S')
            
            # Create the folder structure
            time_folder.mkdir(parents=True, exist_ok=True)
            
            # Store image if provided
            image_path = None
            if 'image_path' in detection_data and detection_data['image_path']:
                src_path = Path(detection_data['image_path'])
                if src_path.exists():
                    image_name = src_path.name
                    dest_image_path = time_folder / image_name
                    shutil.copy2(src_path, dest_image_path)
                    image_path = str(dest_image_path)
            
            # Create metadata file
            metadata = {
                'timestamp': now.isoformat(),
                'detection_data': detection_data,
                'image_path': image_path,
                'original_image_path': detection_data.get('image_path', None)
            }
            
            metadata_path = time_folder / 'metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            self.logger.info(f"Detection stored at: {time_folder}")
            return str(time_folder)
            
        except Exception as e:
            self.logger.error(f"Error storing detection: {e}")
            raise
    
    def cleanup_old_data(self):
        """
        Clean up old data based on retention policies.
        """
        self.logger.info("Starting cleanup of old data...")
        
        # Get all detection folders
        detections_dir = self.base_storage_path / 'detections'
        if not detections_dir.exists():
            self.logger.info("No detections directory found, nothing to clean up.")
            return
        
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=self.max_file_age_days)
        cleaned_count = 0
        
        for date_dir in detections_dir.iterdir():
            if date_dir.is_dir():
                # Check date from folder name
                try:
                    date_obj = datetime.datetime.strptime(date_dir.name, '%Y-%m-%d')
                    if date_obj < cutoff_date:
                        # Remove the entire date folder
                        shutil.rmtree(date_dir)
                        self.logger.info(f"Removed old detection data: {date_dir}")
                        cleaned_count += 1
                        continue
                except ValueError:
                    # Folder name doesn't match date format, skip
                    pass
                
                # Check individual time folders within date folder
                for time_dir in date_dir.iterdir():
                    if time_dir.is_dir():
                        try:
                            # Extract date and time from folder name
                            parts = time_dir.name.split('-')
                            if len(parts) == 3:  # HH-MM-SS format
                                date_str = date_dir.name
                                time_str = time_dir.name
                                datetime_str = f"{date_str}_{time_str}"
                                folder_datetime = datetime.datetime.strptime(datetime_str, '%Y-%m-%d_%H-%M-%S')
                                
                                if folder_datetime < cutoff_date:
                                    shutil.rmtree(time_dir)
                                    self.logger.info(f"Removed old detection data: {time_dir}")
                                    cleaned_count += 1
                        except ValueError:
                            # Folder name doesn't match expected format, skip
                            continue
        
        self.logger.info(f"Cleanup completed. Removed {cleaned_count} old detection folders.")
        
        # Also clean up other potentially old data
        self._cleanup_old_logs()
        self._cleanup_old_videos()
    
    def _cleanup_old_logs(self):
        """Clean up old log files."""
        logs_dir = self.base_storage_path / 'logs'
        if not logs_dir.exists():
            return
        
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=self.max_file_age_days)
        cleaned_count = 0
        
        for log_file in logs_dir.rglob('*.log'):
            try:
                file_time = datetime.datetime.fromtimestamp(log_file.stat().st_mtime)
                if file_time < cutoff_date:
                    log_file.unlink()
                    self.logger.info(f"Removed old log file: {log_file}")
                    cleaned_count += 1
            except Exception as e:
                self.logger.error(f"Error cleaning up log file {log_file}: {e}")
        
        self.logger.info(f"Log cleanup completed. Removed {cleaned_count} old log files.")
    
    def _cleanup_old_videos(self):
        """Clean up old video files."""
        videos_dir = self.base_storage_path / 'videos'
        if not videos_dir.exists():
            return
        
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=self.max_file_age_days)
        cleaned_count = 0
        
        for video_file in videos_dir.rglob('*'):
            if video_file.is_file() and video_file.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                try:
                    file_time = datetime.datetime.fromtimestamp(video_file.stat().st_mtime)
                    if file_time < cutoff_date:
                        video_file.unlink()
                        self.logger.info(f"Removed old video file: {video_file}")
                        cleaned_count += 1
                except Exception as e:
                    self.logger.error(f"Error cleaning up video file {video_file}: {e}")
        
        self.logger.info(f"Video cleanup completed. Removed {cleaned_count} old video files.")
    
    def backup_data(self):
        """
        Back up data to external storage.
        This is a placeholder; actual backup implementation would depend on BackupService.
        """
        self.logger.info("Backup operation initiated through StorageManager")
        # In a real implementation, this would coordinate with BackupService
        # For now, we'll implement the actual backup functionality in BackupService
    
    def _check_disk_space(self) -> bool:
        """
        Check if disk space is above the minimum threshold.
        
        Returns:
            True if sufficient space is available, False otherwise
        """
        total, used, free = shutil.disk_usage(self.base_storage_path)
        free_gb = free / (1024**3)  # Convert to GB
        
        return free_gb >= self.min_free_space_gb
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics.
        
        Returns:
            Dictionary with storage statistics
        """
        total, used, free = shutil.disk_usage(self.base_storage_path)
        total_gb = total / (1024**3)
        used_gb = used / (1024**3)
        free_gb = free / (1024**3)
        
        # Count files
        file_count = sum(len(files) for _, _, files in os.walk(self.base_storage_path))
        
        stats = {
            'total_gb': round(total_gb, 2),
            'used_gb': round(used_gb, 2),
            'free_gb': round(free_gb, 2),
            'file_count': file_count,
            'min_free_space_gb': self.min_free_space_gb,
            'sufficient_space': free_gb >= self.min_free_space_gb
        }
        
        return stats


class BackupService:
    """
    Backup service for cross-platform backups using rsync (Linux/macOS) or robocopy (Windows).
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the backup service.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Extract configuration values
        self.nas_ip = config.get('nas_ip', '')
        self.nas_user = config.get('nas_user', '')
        self.nas_destination_path = config.get('nas_destination_path', '/backups/thecatbouncer')
        self.nas_windows_share = config.get('nas_windows_share', 'Backups')
        self.backup_enabled = config.get('backup_enabled', True)
        self.backup_time = config.get('backup_time', '02:00')  # Format: HH:MM
        self.backup_interval_days = config.get('backup_interval_days', 1)
        
        self.logger.info(f"Backup service initialized for {platform.system()} platform")
    
    def perform_backup(self, source_directories: List[str], backup_name: Optional[str] = None) -> bool:
        """
        Perform backup of specified directories to NAS.
        
        Args:
            source_directories: List of directories to backup
            backup_name: Name for the backup (defaults to timestamp)
            
        Returns:
            True if backup was successful, False otherwise
        """
        if not self.backup_enabled:
            self.logger.info("Backup is disabled, skipping backup operation.")
            return True
        
        if not self.nas_ip or not self.nas_user:
            self.logger.error("NAS IP or user not configured, cannot perform backup.")
            return False
        
        if not backup_name:
            backup_name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        
        self.logger.info(f"Starting backup to NAS at {self.nas_ip}")
        
        # Determine the appropriate backup command for the platform
        system = platform.system().lower()
        
        if system == 'windows':
            success = self._backup_with_robocopy(source_directories, backup_name)
        elif system in ['linux', 'darwin']:  # Linux or macOS
            success = self._backup_with_rsync(source_directories, backup_name)
        else:
            self.logger.error(f"Unsupported platform for backup: {system}")
            return False
        
        if success:
            self.logger.info("Backup completed successfully.")
        else:
            self.logger.error("Backup failed.")
        
        return success
    
    def _backup_with_rsync(self, source_directories: List[str], backup_name: str) -> bool:
        """
        Perform backup using rsync (Linux/macOS).
        
        Args:
            source_directories: List of directories to backup
            backup_name: Name for the backup
            
        Returns:
            True if backup was successful, False otherwise
        """
        try:
            # Create the backup destination path with timestamp
            destination = f"{self.nas_user}@{self.nas_ip}:{self.nas_destination_path}/{backup_name}"
            
            # rsync flags:
            # -a: archive mode (preserves permissions, timestamps, etc.)
            # -v: verbose
            # -z: compress during transfer
            # --delete: delete files at destination that don't exist at source
            # --progress: show progress
            base_cmd = ["rsync", "-avz", "--delete", "--progress"]
            
            success_count = 0
            for source_dir in source_directories:
                if not Path(source_dir).exists():
                    self.logger.warning(f"Source directory does not exist: {source_dir}")
                    continue
                
                cmd = base_cmd + [source_dir, destination]
                
                self.logger.info(f"Running rsync command: {' '.join(cmd)}")
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=3600  # 1 hour timeout
                )
                
                if result.returncode == 0:
                    self.logger.info(f"Successfully backed up {source_dir}")
                    success_count += 1
                else:
                    self.logger.error(f"rsync failed for {source_dir}: {result.stderr}")
            
            return success_count == len(source_directories)
            
        except subprocess.TimeoutExpired:
            self.logger.error("Backup timed out")
            return False
        except Exception as e:
            self.logger.error(f"Error during rsync backup: {e}")
            return False
    
    def _backup_with_robocopy(self, source_directories: List[str], backup_name: str) -> bool:
        """
        Perform backup using robocopy (Windows).
        
        Args:
            source_directories: List of directories to backup
            backup_name: Name for the backup
            
        Returns:
            True if backup was successful, False otherwise
        """
        try:
            # For Windows, create the destination path with share name
            destination = f"\\\\{self.nas_ip}\\{self.nas_windows_share}\\{backup_name}"
            
            # robocopy flags:
            # /MIR: mirror directory (equivalent to rsync -a --delete)
            # /R:n: retry n times (default 1 million which is effectively infinite)
            # /W:m: wait m seconds between retries (default 30)
            # /LOG: log output to file
            base_cmd = ["robocopy", "/MIR", "/R:3", "/W:10"]
            
            success_count = 0
            for source_dir in source_directories:
                if not Path(source_dir).exists():
                    self.logger.warning(f"Source directory does not exist: {source_dir}")
                    continue
                
                cmd = base_cmd + [source_dir, destination]
                
                self.logger.info(f"Running robocopy command: {' '.join(cmd)}")
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=3600  # 1 hour timeout
                )
                
                # robocopy returns different exit codes:
                # 0: no files copied (no changes)
                # 1-3: files copied
                # 4-7: some files copied, some failed
                # 8-15: failures occurred
                if result.returncode < 8:
                    self.logger.info(f"Successfully backed up {source_dir}")
                    success_count += 1
                else:
                    self.logger.error(f"robocopy failed for {source_dir}: {result.stderr}")
            
            return success_count == len(source_directories)
            
        except subprocess.TimeoutExpired:
            self.logger.error("Backup timed out")
            return False
        except Exception as e:
            self.logger.error(f"Error during robocopy backup: {e}")
            return False
    
    def schedule_backup(self, storage_manager: StorageManager) -> bool:
        """
        Schedule the backup to run at the specified time.
        
        Args:
            storage_manager: Storage manager instance to backup
        """
        self.logger.info(f"Backup scheduler started, checking for backup time: {self.backup_time}")
        
        while True:
            now = datetime.datetime.now()
            current_time_str = now.strftime('%H:%M')
            
            if current_time_str == self.backup_time:
                self.logger.info("Backup time reached, starting backup...")
                
                # Backup the base storage directory
                success = self.perform_backup([str(storage_manager.base_storage_path)])
                
                if success:
                    self.logger.info("Scheduled backup completed successfully")
                else:
                    self.logger.error("Scheduled backup failed")
                
                # Wait until the next day to avoid running multiple times
                time.sleep(60)  # Wait one minute to avoid running again in same minute
            
            time.sleep(30)  # Check every 30 seconds
    
    def verify_backup(self, backup_path: str) -> bool:
        """
        Verify the integrity of a backup.
        
        Args:
            backup_path: Path to the backup to verify
            
        Returns:
            True if backup is valid, False otherwise
        """
        try:
            backup_path_obj = Path(backup_path)
            if not backup_path_obj.exists():
                self.logger.error(f"Backup path does not exist: {backup_path}")
                return False
            
            # For now, just check if it's a directory and not empty
            if backup_path_obj.is_dir():
                return any(backup_path_obj.iterdir())
            
            # If it's a file, we might need to check its integrity based on format
            return backup_path_obj.stat().st_size > 0
            
        except Exception as e:
            self.logger.error(f"Error verifying backup: {e}")
            return False


def create_data_backup_scheduler(config: Dict[str, Any], storage_manager: StorageManager):
    """
    Create a data backup scheduler that runs at the specified time.
    
    Args:
        config: Backup configuration
        storage_manager: Storage manager instance to backup
    """
    backup_service = BackupService(config)
    backup_service.schedule_backup(storage_manager)


def get_old_files(directory: Path, days: int) -> List[Path]:
    """
    Get files older than specified number of days.
    
    Args:
        directory: Directory to search
        days: Number of days
        
    Returns:
        List of file paths older than specified days
    """
    cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days)
    old_files = []
    
    for file_path in directory.rglob('*'):
        if file_path.is_file():
            try:
                file_time = datetime.datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_time < cutoff_date:
                    old_files.append(file_path)
            except Exception as e:
                logging.error(f"Error checking file {file_path}: {e}")
    
    return old_files


def clean_old_files(file_list: List[Path]) -> int:
    """
    Delete the specified files and return the count of deleted files.
    
    Args:
        file_list: List of file paths to delete
        
    Returns:
        Number of files deleted
    """
    deleted_count = 0
    for file_path in file_list:
        try:
            if file_path.exists():
                file_path.unlink()
                deleted_count += 1
                logging.info(f"Deleted old file: {file_path}")
        except Exception as e:
            logging.error(f"Error deleting file {file_path}: {e}")
    
    return deleted_count