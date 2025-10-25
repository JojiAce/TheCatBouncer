"""
Main application and CLI interface for TheCatBouncer.
"""
import argparse
import sys
import signal
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import json
import datetime

from src.config.config_manager import ConfigManager
from src.managers.camera_manager import CameraManager
from src.managers.passive_monitor import PassiveMonitor
from src.managers.active_analyzer import ActiveAnalyzer
from src.managers.color_analyzer import ColorAnalyzer
from src.managers.hue_controller import HueController
from src.managers.audio_manager import AudioManager
from src.managers.notification_service import NotificationService
from src.managers.storage_manager import StorageManager, BackupService
from src.managers.lifecycle_manager import ScheduleManager, LifecycleManager
from src.managers.performance_monitor import PerformanceMonitor, LoadBalancer
from src.managers.logging_service import LoggingService, get_global_logger
from src.engines.engine_factory import create_inference_engine


class TheCatBouncerApp:
    """
    Main application class for TheCatBouncer.
    Orchestrates all components and manages the detection workflow.
    """
    
    def __init__(self, config_path: str = "config.ini", cli_args: Optional[Dict] = None):
        """
        Initialize the main application.
        
        Args:
            config_path: Path to the configuration file
            cli_args: Command line arguments
        """
        self.config_path = config_path
        self.cli_args = cli_args or {}
        self.logger = get_global_logger()
        
        # Initialize configuration
        self.config_manager = ConfigManager(Path(config_path))
        self.config = self.config_manager._to_dict()  # Get the full config
        
        # Override with CLI args if provided
        if cli_args:
            self._apply_cli_overrides()
        
        # Initialize components
        self.lifecycle_manager = None
        self.schedule_manager = None
        self.camera_manager = None
        self.passive_monitor = None
        self.active_analyzer = None
        self.color_analyzer = None
        self.hue_controller = None
        self.audio_manager = None
        self.notification_service = None
        self.storage_manager = None
        self.backup_service = None
        self.performance_monitor = None
        self.load_balancer = None
        self.inference_engine = None
        self.class_names = []
        
        # Application state
        self.running = False
        self.shutdown_requested = False
        
        self.logger.info("TheCatBouncer application initialized")
    
    def _apply_cli_overrides(self):
        """Apply command line argument overrides to configuration."""
        if self.cli_args.get('live_preview') is not None:
            self.config['actions']['show_live_window'] = self.cli_args['live_preview']
        
        if self.cli_args.get('device'):
            self.config['image_recognition']['inference_device'] = self.cli_args['device']
        
        self.logger.info("Applied CLI overrides to configuration")
    
    def setup_components(self):
        """Initialize all application components."""
        self.logger.info("Setting up application components...")
        
        try:
            # Initialize lifecycle manager
            self.lifecycle_manager = LifecycleManager(self.config.get('time_management', {}))
            
            # Initialize schedule manager
            self.schedule_manager = ScheduleManager(self.config.get('time_management', {}))
            
            # Initialize storage manager
            self.storage_manager = StorageManager(self.config.get('storage_management', {}))
            
            # Initialize backup service
            self.backup_service = BackupService(self.config.get('nas', {}))
            
            # Initialize performance monitor
            self.performance_monitor = PerformanceMonitor()
            self.performance_monitor.start_monitoring()
            
            # Initialize load balancer
            self.load_balancer = LoadBalancer()
            
            # Initialize and load inference engine
            model_path = self.config['image_recognition']['yolo_model_path']
            device = self.config['image_recognition']['inference_device']
            
            # Determine model paths based on the provided path
            model_dir = Path(model_path)
            model_paths = []
            
            if model_dir.is_dir():
                # Look for model files in the directory
                for ext in ['.onnx', '.xml', '.pt', '.safetensors', '.mlmodel']:
                    model_files = list(model_dir.glob(f'*{ext}'))
                    model_paths.extend([str(f) for f in model_files])
            else:
                model_paths = [model_path]
            
            if not model_paths:
                raise ValueError(f"No model files found at {model_path}")
            
            self.inference_engine, self.class_names = create_inference_engine(
                model_paths, 
                device
            )
            
            # Initialize color analyzer
            self.color_analyzer = ColorAnalyzer(self.config.get('color_analysis', {}))
            
            # Initialize audio manager
            self.audio_manager = AudioManager(self.config.get('paths', {}))
            
            # Initialize notification service
            self.notification_service = NotificationService(self.config.get('notifications', {}))
            
            # Initialize Hue controller
            self.hue_controller = HueController(self.config.get('philips_hue', {}))
            
            # Initialize passive monitor
            passive_config = self.config.get('passiv_analyzer', {})
            passive_config.update(self.config.get('camera', {}))
            passive_config.update(self.config.get('trigger', {}))
            self.passive_monitor = PassiveMonitor(passive_config)
            
            # Initialize active analyzer
            active_config = self.config.get('active', {})
            active_config.update(self.config.get('image_recognition', {}))
            active_config.update(self.config.get('detection', {}))
            active_config.update(self.config.get('paths', {}))
            active_config.update(self.config.get('actions', {}))
            self.active_analyzer = ActiveAnalyzer(active_config, self.inference_engine)
            
            # Register components with lifecycle manager
            self.lifecycle_manager.register_component('storage', self.storage_manager)
            self.lifecycle_manager.register_component('hue', self.hue_controller)
            self.lifecycle_manager.register_component('audio', self.audio_manager)
            self.lifecycle_manager.register_component('notifications', self.notification_service)
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to setup components: {e}")
            raise
    
    def run(self):
        """Run the main application loop."""
        self.logger.info("Starting TheCatBouncer application...")
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.running = True
        self.lifecycle_manager.start_application()
        
        try:
            while self.running and not self.shutdown_requested:
                # Check if we're within the scheduled active time
                if not self.schedule_manager.is_within_schedule():
                    self.logger.info("Outside active hours, sleeping...")
                    time.sleep(60)  # Check again in 1 minute
                    continue
                
                # Check if it's time for data management/maintenance
                if self.schedule_manager.is_maintenance_time():
                    self._run_maintenance()
                
                # Run passive monitoring phase
                self.logger.info("Starting passive monitoring phase...")
                trigger_detected = self._run_passive_phase()
                
                if not trigger_detected:
                    self.logger.info("No trigger detected, continuing passive monitoring...")
                    continue
                
                # If lights are connected, turn them on during active analysis
                if self.hue_controller.is_connected():
                    self.hue_controller.set_lights_on(
                        light_ids=self.config['philips_hue'].get('light_ids', []),
                        brightness=self.config['philips_hue'].get('brightness', 254),
                        saturation=self.config['philips_hue'].get('saturation', 0),
                        hue=self.config['philips_hue'].get('hue', 14910)
                    )
                
                self.logger.info("Trigger detected! Starting active analysis...")
                
                # Run active analysis phase
                detection_path = self._run_active_phase()
                
                if detection_path:
                    self.logger.info("Active analysis successful, starting color analysis...")
                    
                    # Run color analysis
                    is_cat_black = self._run_color_analysis(detection_path)
                    
                    if is_cat_black:
                        self.logger.info("OWN cat detected. Returning to passive monitoring...")
                        if self.hue_controller.is_connected():
                            self.hue_controller.set_lights_off(
                                light_ids=self.config['philips_hue'].get('light_ids', [])
                            )
                        time.sleep(10)  # Wait before going back to passive
                    else:
                        self.logger.info("INTRUDER detected! Activating deterrents...")
                        self._handle_intruder(detection_path)
                        
                        if self.hue_controller.is_connected():
                            self.hue_controller.set_lights_off(
                                light_ids=self.config['philips_hue'].get('light_ids', [])
                            )
                        
                        # Wait before returning to passive monitoring
                        time.sleep(5)
                else:
                    self.logger.info("Active analysis not successful, returning to passive monitoring...")
                    if self.hue_controller.is_connected():
                        self.hue_controller.set_lights_off(
                            light_ids=self.config['philips_hue'].get('light_ids', [])
                        )
                    time.sleep(5)
        
        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt")
        except Exception as e:
            self.logger.error(f"Error in main application loop: {e}")
            self.lifecycle_manager.handle_error(e, "main_loop")
        finally:
            self.stop()
    
    def _signal_handler(self, signum, frame):
        """Handle system signals for graceful shutdown."""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True
    
    def _run_passive_phase(self) -> bool:
        """Run the passive monitoring phase."""
        try:
            self.passive_monitor.start_monitoring()
            return self.passive_monitor.is_triggered()
        except Exception as e:
            self.logger.error(f"Error in passive phase: {e}")
            return False
    
    def _run_active_phase(self) -> Optional[str]:
        """Run the active analysis phase."""
        try:
            return self.active_analyzer.start_analysis()
        except Exception as e:
            self.logger.error(f"Error in active phase: {e}")
            return None
    
    def _run_color_analysis(self, detection_path: str) -> bool:
        """Run the color analysis phase."""
        try:
            # Define bounding box (for now, using full image)
            image_path = Path(detection_path) / "frame.jpg"
            if not image_path.exists():
                self.logger.error(f"Detection image not found: {image_path}")
                return False
            
            # In a real implementation, we'd get the actual bounding box coordinates
            # For now, we'll analyze the full image
            bbox = {
                'x1': 0,
                'y1': 0,
                'x2': -1,  # Will be set to image width
                'y2': -1   # Will be set to image height
            }
            
            # Get actual image dimensions
            import cv2
            image = cv2.imread(str(image_path))
            if image is not None:
                height, width = image.shape[:2]
                bbox = {'x1': 0, 'y1': 0, 'x2': width, 'y2': height}
            
            return self.color_analyzer.analyze_color(str(image_path), bbox)
        except Exception as e:
            self.logger.error(f"Error in color analysis: {e}")
            return False
    
    def _handle_intruder(self, detection_path: str):
        """Handle intruder detection with deterrents and notifications."""
        try:
            # Send notification about intruder
            intruder_image_path = Path(detection_path) / 'frame.jpg'
            self.notification_service.send_notification(
                message=f"Intruder detected at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                image_path=str(intruder_image_path)
            )
            
            # Activate deterrents
            duration_minutes = float(self.config['actions'].get('intruder_light_minutes', 4))
            sound_config = {'folder': self.config['paths'].get('sound_directory', 'cat_scare_sound')}
            
            # Play deterrent sound
            self.audio_manager.play_sound()
            
            # Keep lights on for specified duration (in a separate thread to not block)
            import threading
            def turn_off_lights_after_delay():
                time.sleep(duration_minutes * 60)  # Convert minutes to seconds
                if self.hue_controller.is_connected():
                    self.hue_controller.set_lights_off(
                        light_ids=self.config['philips_hue'].get('light_ids', [])
                    )
            
            light_thread = threading.Thread(target=turn_off_lights_after_delay, daemon=True)
            light_thread.start()
            
            self.logger.info("Intruder deterrents activated")
            
        except Exception as e:
            self.logger.error(f"Error handling intruder: {e}")
    
    def _run_maintenance(self):
        """Run scheduled maintenance tasks."""
        self.logger.info("Running scheduled maintenance tasks...")
        
        try:
            # Clean up old data
            self.storage_manager.cleanup_old_data()
            
            # Run backups
            source_directories = [str(self.storage_manager.base_storage_path)]
            backup_success = self.backup_service.perform_backup(source_directories)
            
            if backup_success:
                self.logger.info("Maintenance tasks completed successfully")
            else:
                self.logger.warning("Maintenance tasks completed with some errors")
                
        except Exception as e:
            self.logger.error(f"Error during maintenance: {e}")
    
    def stop(self):
        """Stop the application gracefully."""
        self.logger.info("Stopping TheCatBouncer application...")
        
        self.running = False
        
        # Stop active components
        if hasattr(self, 'active_analyzer') and self.active_analyzer:
            self.active_analyzer.stop_analysis()
        
        if hasattr(self, 'passive_monitor') and self.passive_monitor:
            self.passive_monitor.stop_monitoring()
        
        # Stop performance monitoring
        if hasattr(self, 'performance_monitor') and self.performance_monitor:
            self.performance_monitor.stop_monitoring()
        
        # Stop all registered components via lifecycle manager
        if self.lifecycle_manager:
            self.lifecycle_manager.stop_application()
        
        # Turn off lights if they were on
        if hasattr(self, 'hue_controller') and self.hue_controller.is_connected():
            self.hue_controller.set_lights_off()
        
        self.logger.info("TheCatBouncer application stopped")


def create_cli_parser() -> argparse.ArgumentParser:
    """Create the command line interface parser."""
    parser = argparse.ArgumentParser(
        description="TheCatBouncer - AI-powered pet access control system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Run with default config.ini
  %(prog)s --live-preview    # Force enable live preview window
  %(prog)s --no-live-preview # Force disable live preview window
  %(prog)s -d cuda:0         # Use CUDA device for inference
  %(prog)s --device cpu      # Force CPU for inference
        """
    )
    
    # Configuration file option
    parser.add_argument(
        '-c', '--config',
        type=str,
        default='config.ini',
        help='Path to configuration file (default: config.ini)'
    )
    
    # Live preview options
    preview_group = parser.add_mutually_exclusive_group()
    preview_group.add_argument(
        '--live-preview',
        action='store_true',
        help='Force enable live preview window'
    )
    preview_group.add_argument(
        '--no-live-preview',
        action='store_true',
        help='Force disable live preview window'
    )
    
    # Device options
    parser.add_argument(
        '-d', '--device',
        type=str,
        choices=['cpu', 'gpu', 'cuda:0', 'cuda:1'],
        help='Force inference device (overrides config)'
    )
    
    # Verbose output
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    # Version
    parser.add_argument(
        '--version',
        action='version',
        version='TheCatBouncer 2.0'
    )
    
    return parser


def main():
    """Main entry point for the application."""
    parser = create_cli_parser()
    args = parser.parse_args()
    
    # Determine live preview setting
    live_preview = None
    if args.live_preview:
        live_preview = True
    elif args.no_live_preview:
        live_preview = False
    
    # Create CLI args dict
    cli_args = {
        'device': args.device,
        'live_preview': live_preview
    }
    
    # Setup basic logging before loading the full logging system
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )
    
    try:
        # Initialize the application
        app = TheCatBouncerApp(config_path=args.config, cli_args=cli_args)
        
        # If verbose, enable debug logging
        if args.verbose:
            app.logger.enable_debug_logging()
        
        # Setup all components
        app.setup_components()
        
        # Run the application
        app.run()
        
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()