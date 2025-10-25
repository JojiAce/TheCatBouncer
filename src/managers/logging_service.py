"""
Comprehensive logging system with structured logging and performance metrics tracking.
"""
import logging
import logging.handlers
import os
import sys
import json
import datetime
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union
import threading
import queue
import atexit

from src.interfaces.monitoring import DataManagerInterface


class StructuredFormatter(logging.Formatter):
    """
    Custom formatter that outputs structured log entries in JSON format.
    """
    
    def format(self, record):
        # Create structured log entry
        log_entry = {
            'timestamp': datetime.datetime.utcfromtimestamp(record.created).isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add any extra fields
        if hasattr(record, 'extra_data'):
            log_entry['extra_data'] = record.extra_data
        
        return json.dumps(log_entry)


class AsyncLogHandler(logging.Handler):
    """
    Asynchronous log handler that queues log records and processes them in a background thread.
    """
    
    def __init__(self, delegate_handler, max_queue_size=10000):
        super().__init__()
        self.delegate_handler = delegate_handler
        self.log_queue = queue.Queue(maxsize=max_queue_size)
        self.running = True
        
        # Start background thread
        self.thread = threading.Thread(target=self._log_worker, daemon=True)
        self.thread.start()
        
        # Register cleanup
        atexit.register(self._cleanup)
    
    def emit(self, record):
        try:
            # Add timestamp for async processing
            record.async_timestamp = time.time()
            self.log_queue.put_nowait(record)
        except queue.Full:
            # Drop the log if queue is full
            sys.stderr.write(f"Dropped log due to full queue: {record.getMessage()[:100]}...\n")
    
    def _log_worker(self):
        """Background thread to process log records."""
        while self.running:
            try:
                record = self.log_queue.get(timeout=1)
                self.delegate_handler.emit(record)
                self.log_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                # Handle errors in logging to prevent infinite loops
                sys.stderr.write(f"Error in log worker: {e}\n")
    
    def _cleanup(self):
        """Cleanup method to flush remaining logs."""
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)
        
        # Flush any remaining records
        while not self.log_queue.empty():
            try:
                record = self.log_queue.get_nowait()
                self.delegate_handler.emit(record)
            except queue.Empty:
                break


class LoggingService:
    """
    Comprehensive logging service with log rotation, performance tracking, and structured logging.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the logging service.
        
        Args:
            config: Configuration dictionary for logging settings
        """
        self.config = config or {}
        self.logger = logging.getLogger('TheCatBouncer')
        self.logger.setLevel(logging.DEBUG)
        
        # Extract configuration values
        self.log_level = self.config.get('log_level', 'INFO').upper()
        self.log_directory = Path(self.config.get('log_directory', 'logs'))
        self.rotation_when = self.config.get('rotation_when', 'midnight')
        self.rotation_interval = self.config.get('rotation_interval', 1)
        self.rotation_backup_count = self.config.get('rotation_backup_count', 7)
        self.async_logging = self.config.get('async_logging', True)
        self.include_performance_metrics = self.config.get('include_performance_metrics', True)
        self.include_debug_window = self.config.get('include_debug_window', False)
        
        # Create log directory
        self.log_directory.mkdir(parents=True, exist_ok=True)
        
        # Set up handlers
        self._setup_handlers()
        
        # Performance tracking
        self.performance_metrics = {}
        self.performance_lock = threading.Lock()
        
        # Initialize logging
        self.logger.info("Logging service initialized", extra={'component': 'LoggingService'})
    
    def _setup_handlers(self):
        """Set up logging handlers based on configuration."""
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create file handler with rotation
        log_file_path = self.log_directory / 'thecatbouncer.log'
        file_handler = logging.handlers.TimedRotatingFileHandler(
            filename=str(log_file_path),
            when=self.rotation_when,
            interval=self.rotation_interval,
            backupCount=self.rotation_backup_count,
            encoding='utf-8'
        )
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        
        # Set formatter based on configuration
        if self.config.get('structured_format', False):
            formatter = StructuredFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s [%(levelname)-5s] %(name)s: %(message)s'
            )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Set log levels
        file_handler.setLevel(getattr(logging, self.log_level))
        console_handler.setLevel(getattr(logging, self.log_level))
        
        # Add handlers
        if self.async_logging:
            async_file_handler = AsyncLogHandler(file_handler)
            async_console_handler = AsyncLogHandler(console_handler)
            self.logger.addHandler(async_file_handler)
            self.logger.addHandler(async_console_handler)
        else:
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
    
    def log_detection(self, detection_data: Dict[str, Any], image_path: Optional[str] = None):
        """
        Log a detection event with additional metadata.
        
        Args:
            detection_data: Detection information
            image_path: Path to the detection image
        """
        extra_data = {
            'event_type': 'detection',
            'detection_data': detection_data,
            'image_path': image_path
        }
        
        self.logger.info(f"Detection event: {detection_data.get('class_name', 'unknown')} "
                        f"with confidence {detection_data.get('confidence', 0)}",
                        extra={'extra_data': extra_data})
    
    def log_performance_metric(self, metric_name: str, value: Union[float, int, str], 
                             context: Optional[Dict[str, Any]] = None):
        """
        Log a performance metric.
        
        Args:
            metric_name: Name of the metric
            value: Value of the metric
            context: Additional context information
        """
        with self.performance_lock:
            if metric_name not in self.performance_metrics:
                self.performance_metrics[metric_name] = []
            
            metric_entry = {
                'timestamp': time.time(),
                'value': value,
                'context': context or {}
            }
            self.performance_metrics[metric_name].append(metric_entry)
        
        extra_data = {
            'metric_name': metric_name,
            'value': value,
            'context': context
        }
        
        self.logger.debug(f"Performance metric: {metric_name} = {value}",
                         extra={'extra_data': extra_data})
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get a summary of collected performance metrics.
        
        Returns:
            Dictionary with performance summary
        """
        with self.performance_lock:
            summary = {}
            
            for metric_name, values in self.performance_metrics.items():
                if values:
                    # Extract just the values for calculations
                    metric_values = [v['value'] for v in values if isinstance(v['value'], (int, float))]
                    
                    if metric_values:
                        summary[metric_name] = {
                            'count': len(metric_values),
                            'min': min(metric_values),
                            'max': max(metric_values),
                            'avg': sum(metric_values) / len(metric_values),
                            'latest': values[-1]['value'],
                            'latest_timestamp': values[-1]['timestamp']
                        }
            
            return summary
    
    def log_system_event(self, event_type: str, message: str, 
                        severity: str = 'INFO', 
                        context: Optional[Dict[str, Any]] = None):
        """
        Log a system event.
        
        Args:
            event_type: Type of event
            message: Event message
            severity: Event severity level
            context: Additional context information
        """
        extra_data = {
            'event_type': event_type,
            'context': context or {}
        }
        
        log_method = getattr(self.logger, severity.lower(), self.logger.info)
        log_method(f"[{event_type}] {message}", extra={'extra_data': extra_data})
    
    def log_error_with_context(self, error: Exception, context: Dict[str, Any], 
                             component: str = 'Unknown'):
        """
        Log an error with additional context information.
        
        Args:
            error: Exception that occurred
            context: Context information about when/where the error occurred
            component: Component where the error occurred
        """
        extra_data = {
            'error_type': type(error).__name__,
            'component': component,
            'context': context
        }
        
        self.logger.error(f"Error in {component}: {str(error)}", 
                         extra={'extra_data': extra_data}, 
                         exc_info=True)
    
    def enable_debug_logging(self):
        """Enable debug-level logging."""
        self.logger.setLevel(logging.DEBUG)
        for handler in self.logger.handlers:
            handler.setLevel(logging.DEBUG)
        
        self.logger.debug("Debug logging enabled")
    
    def disable_debug_logging(self):
        """Disable debug-level logging."""
        self.logger.setLevel(getattr(logging, self.log_level))
        for handler in self.logger.handlers:
            handler.setLevel(getattr(logging, self.log_level))
        
        self.logger.info("Debug logging disabled")
    
    def get_log_file_path(self) -> Path:
        """
        Get the path to the current log file.
        
        Returns:
            Path to the current log file
        """
        # Find the most recent log file with the base name
        base_name = 'thecatbouncer.log'
        log_files = list(self.log_directory.glob(f'{base_name}*'))
        
        if not log_files:
            return self.log_directory / base_name
        
        # Return the most recent file
        return max(log_files, key=os.path.getctime)
    
    def cleanup_old_logs(self, days_to_keep: int = 30):
        """
        Clean up log files older than the specified number of days.
        
        Args:
            days_to_keep: Number of days of logs to keep
        """
        cutoff_time = time.time() - (days_to_keep * 24 * 60 * 60)
        
        for log_file in self.log_directory.glob('thecatbouncer.log*'):
            if log_file.stat().st_mtime < cutoff_time:
                try:
                    log_file.unlink()
                    self.logger.info(f"Removed old log file: {log_file}")
                except OSError as e:
                    self.logger.error(f"Failed to remove old log file {log_file}: {e}")
    
    def flush_logs(self):
        """Flush all pending log messages."""
        for handler in self.logger.handlers:
            if hasattr(handler, 'flush'):
                handler.flush()
            elif hasattr(handler, 'delegate_handler'):
                # For async handlers, delegate to the underlying handler
                if hasattr(handler.delegate_handler, 'flush'):
                    handler.delegate_handler.flush()


def setup_logging(config: Optional[Dict[str, Any]] = None) -> LoggingService:
    """
    Set up the logging service with the given configuration.
    
    Args:
        config: Logging configuration
        
    Returns:
        Configured LoggingService instance
    """
    return LoggingService(config)


def log_function_call(func):
    """
    Decorator to log function calls with their arguments and return values.
    
    Args:
        func: Function to wrap
        
    Returns:
        Wrapped function
    """
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} returned: {result}")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} raised {type(e).__name__}: {e}")
            raise
    
    return wrapper


class PerformanceTracker:
    """
    Utility class for tracking performance metrics.
    """
    
    def __init__(self, logging_service: LoggingService):
        self.logging_service = logging_service
        self.start_times = {}
    
    def start_timer(self, name: str):
        """
        Start a timer with the given name.
        
        Args:
            name: Name of the timer
        """
        self.start_times[name] = time.time()
    
    def stop_timer(self, name: str, log: bool = True) -> float:
        """
        Stop a named timer and return the elapsed time.
        
        Args:
            name: Name of the timer to stop
            log: Whether to log the timing result
            
        Returns:
            Elapsed time in seconds
        """
        if name not in self.start_times:
            raise ValueError(f"Timer '{name}' not started")
        
        elapsed = time.time() - self.start_times[name]
        del self.start_times[name]
        
        if log:
            self.logging_service.log_performance_metric(
                f"{name}_duration_seconds", 
                elapsed,
                {"unit": "seconds"}
            )
        
        return elapsed


# Global logging service instance
_global_logging_service = None


def get_global_logger() -> LoggingService:
    """
    Get the global logging service instance.
    
    Returns:
        Global LoggingService instance
    """
    global _global_logging_service
    if _global_logging_service is None:
        _global_logging_service = LoggingService()
    return _global_logging_service