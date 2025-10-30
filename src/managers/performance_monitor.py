"""
Performance optimization and monitoring system.
"""
import time
import psutil
import threading
import logging
import GPUtil
from typing import Dict, Any, Optional, Callable, List
from collections import deque, defaultdict
import statistics
import multiprocessing as mp
from dataclasses import dataclass


@dataclass
class PerformanceMetrics:
    """Data class for performance metrics."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    gpu_percent: Optional[float]
    fps: Optional[float] = None
    latency_ms: Optional[float] = None
    throughput: Optional[float] = None


class PerformanceMonitor:
    """
    Performance monitor for real-time optimization with FPS, latency, and throughput tracking.
    Monitors CPU, GPU, and memory utilization with automatic bottleneck detection.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the performance monitor.
        
        Args:
            config: Configuration dictionary (optional)
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Performance tracking
        self.metrics_history = deque(maxlen=self.config.get('history_size', 1000))
        self.bottleneck_thresholds = self.config.get('bottleneck_thresholds', {
            'cpu_percent': 90.0,
            'memory_percent': 90.0,
            'gpu_percent': 95.0,
            'latency_ms': 100.0
        })
        
        # Performance optimization
        self.adaptive_threading_enabled = self.config.get('adaptive_threading_enabled', True)
        self.current_worker_count = self.config.get('initial_worker_count', 1)
        self.max_worker_count = self.config.get('max_worker_count', mp.cpu_count())
        
        # Monitoring state
        self.monitoring = False
        self.monitoring_thread = None
        
        # Performance statistics
        self.fps_history = deque(maxlen=30)  # Last 30 FPS readings
        self.latency_history = deque(maxlen=30)  # Last 30 latency readings
        
        self.logger.info("Performance monitor initialized")
    
    def start_monitoring(self):
        """Start performance monitoring in a background thread."""
        if self.monitoring:
            self.logger.warning("Performance monitoring already running")
            return
        
        self.monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)
        self.logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Internal monitoring loop running in background thread."""
        while self.monitoring:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Check for bottlenecks
                bottlenecks = self._detect_bottlenecks(metrics)
                if bottlenecks:
                    self._handle_bottleneck(bottlenecks, metrics)
                
                time.sleep(self.config.get('monitoring_interval', 1.0))
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(1.0)
    
    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics."""
        timestamp = time.time()
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=None)
        
        # Memory usage
        memory_percent = psutil.virtual_memory().percent
        
        # GPU usage (if available)
        gpu_percent = None
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu_percent = gpus[0].load * 100  # Use first GPU
        
        return PerformanceMetrics(
            timestamp=timestamp,
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            gpu_percent=gpu_percent
        )
    
    def _detect_bottlenecks(self, metrics: PerformanceMetrics) -> List[str]:
        """Detect performance bottlenecks based on thresholds."""
        bottlenecks = []
        
        if metrics.cpu_percent > self.bottleneck_thresholds['cpu_percent']:
            bottlenecks.append(f"CPU usage high: {metrics.cpu_percent:.1f}%")
        
        if metrics.memory_percent > self.bottleneck_thresholds['memory_percent']:
            bottlenecks.append(f"Memory usage high: {metrics.memory_percent:.1f}%")
        
        if (metrics.gpu_percent is not None and 
            metrics.gpu_percent > self.bottleneck_thresholds['gpu_percent']):
            bottlenecks.append(f"GPU usage high: {metrics.gpu_percent:.1f}%")
        
        if (metrics.latency_ms is not None and 
            metrics.latency_ms > self.bottleneck_thresholds['latency_ms']):
            bottlenecks.append(f"Latency high: {metrics.latency_ms:.1f}ms")
        
        return bottlenecks
    
    def _handle_bottleneck(self, bottlenecks: List[str], metrics: PerformanceMetrics):
        """Handle detected bottlenecks."""
        for bottleneck in bottlenecks:
            self.logger.warning(f"Performance bottleneck detected: {bottleneck}")
        
        # Adaptive threading optimization
        if self.adaptive_threading_enabled and bottlenecks:
            self._optimize_threading()
    
    def _optimize_threading(self):
        """Optimize threading based on current performance."""
        # Simple algorithm: reduce worker count if CPU usage is too high
        cpu_avg = self._get_average_cpu_usage()
        if cpu_avg > 85:
            if self.current_worker_count > 1:
                self.current_worker_count = max(1, self.current_worker_count - 1)
                self.logger.info(f"Reduced worker count to {self.current_worker_count} due to high CPU usage")
        elif cpu_avg < 60 and self.current_worker_count < self.max_worker_count:
            self.current_worker_count += 1
            self.logger.info(f"Increased worker count to {self.current_worker_count}")
    
    def _get_average_cpu_usage(self) -> float:
        """Get average CPU usage from history."""
        if not self.metrics_history:
            return 0.0
        
        cpu_percentages = [m.cpu_percent for m in self.metrics_history]
        return sum(cpu_percentages) / len(cpu_percentages)
    
    def record_frame_processing(self, processing_time_ms: float):
        """
        Record frame processing metrics for FPS and latency calculations.
        
        Args:
            processing_time_ms: Time taken to process a frame in milliseconds
        """
        fps = 1000.0 / processing_time_ms if processing_time_ms > 0 else 0
        self.fps_history.append(fps)
        self.latency_history.append(processing_time_ms)
    
    def get_current_performance(self) -> Dict[str, Any]:
        """
        Get current performance metrics.
        
        Returns:
            Dictionary with current performance information
        """
        if not self.metrics_history:
            return {}
        
        latest = self.metrics_history[-1]
        
        performance = {
            'timestamp': latest.timestamp,
            'cpu_percent': latest.cpu_percent,
            'memory_percent': latest.memory_percent,
            'gpu_percent': latest.gpu_percent,
            'worker_count': self.current_worker_count,
            'adaptive_threading_enabled': self.adaptive_threading_enabled
        }
        
        # Add FPS and latency if available
        if self.fps_history:
            performance['current_fps'] = self.fps_history[-1] if self.fps_history else 0
            performance['avg_fps'] = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
        
        if self.latency_history:
            performance['current_latency_ms'] = self.latency_history[-1] if self.latency_history else 0
            performance['avg_latency_ms'] = sum(self.latency_history) / len(self.latency_history) if self.latency_history else 0
        
        return performance
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """
        Get performance statistics over time.
        
        Returns:
            Dictionary with performance statistics
        """
        if not self.metrics_history:
            return {}
        
        # CPU statistics
        cpu_values = [m.cpu_percent for m in self.metrics_history]
        cpu_stats = {
            'min': min(cpu_values),
            'max': max(cpu_values),
            'avg': sum(cpu_values) / len(cpu_values),
            'current': self.metrics_history[-1].cpu_percent
        }
        
        # Memory statistics
        memory_values = [m.memory_percent for m in self.metrics_history]
        memory_stats = {
            'min': min(memory_values),
            'max': max(memory_values),
            'avg': sum(memory_values) / len(memory_values),
            'current': self.metrics_history[-1].memory_percent
        }
        
        # FPS statistics (if available)
        fps_stats = {}
        if self.fps_history:
            fps_stats = {
                'min': min(self.fps_history) if self.fps_history else 0,
                'max': max(self.fps_history) if self.fps_history else 0,
                'avg': sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0,
                'current': self.fps_history[-1] if self.fps_history else 0
            }
        
        # Latency statistics (if available)
        latency_stats = {}
        if self.latency_history:
            latency_values = list(self.latency_history)
            latency_stats = {
                'min': min(latency_values) if latency_values else 0,
                'max': max(latency_values) if latency_values else 0,
                'avg': sum(latency_values) / len(latency_values) if latency_values else 0,
                'current': self.latency_history[-1] if self.latency_history else 0
            }
        
        return {
            'cpu_stats': cpu_stats,
            'memory_stats': memory_stats,
            'fps_stats': fps_stats,
            'latency_stats': latency_stats,
            'worker_count': self.current_worker_count,
            'sample_count': len(self.metrics_history)
        }
    
    def register_performance_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """
        Register a callback function to be called with performance updates.
        
        Args:
            callback: Function to call with performance data
        """
        pass  # Implementation would depend on specific requirements
    
    def get_optimal_worker_count(self) -> int:
        """
        Get the recommended optimal worker count based on current performance.
        
        Returns:
            Recommended worker count
        """
        return self.current_worker_count


class PipelineOptimizer:
    """
    Pipeline optimizer for queue size optimization and frame buffer management.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the pipeline optimizer.
        
        Args:
            config: Configuration dictionary (optional)
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Queue optimization parameters
        self.target_queue_size = self.config.get('target_queue_size', 10)
        self.min_queue_size = self.config.get('min_queue_size', 2)
        self.max_queue_size = self.config.get('max_queue_size', 50)
        
        # Frame buffer parameters
        self.frame_buffer_size = self.config.get('frame_buffer_size', 30)
        
        # Performance tracking
        self.queue_size_history = deque(maxlen=100)
        self.buffer_usage_history = deque(maxlen=100)
        
        self.logger.info("Pipeline optimizer initialized")
    
    def optimize_queue_size(self, current_queue_size: int, processing_rate: float, 
                          input_rate: float) -> int:
        """
        Optimize queue size based on processing and input rates.
        
        Args:
            current_queue_size: Current queue size
            processing_rate: Rate at which items are processed (items/second)
            input_rate: Rate at which items are added to queue (items/second)
            
        Returns:
            Recommended queue size
        """
        # If input rate exceeds processing rate, we need a larger queue
        if input_rate > processing_rate:
            # Calculate needed buffer to handle the difference
            rate_diff = input_rate - processing_rate
            needed_buffer = int(rate_diff * 5)  # 5 seconds of buffer
            recommended_size = min(self.max_queue_size, 
                                 max(self.min_queue_size, needed_buffer))
        else:
            # If processing rate is higher, we can use smaller queue
            recommended_size = int(self.target_queue_size * (input_rate / processing_rate))
            recommended_size = min(self.target_queue_size, 
                                 max(self.min_queue_size, recommended_size))
        
        self.queue_size_history.append(recommended_size)
        self.logger.debug(f"Queue size optimization: {current_queue_size} -> {recommended_size} "
                         f"(input_rate: {input_rate:.2f}, processing_rate: {processing_rate:.2f})")
        
        return recommended_size
    
    def optimize_frame_buffer(self, current_buffer_size: int, fps: float, 
                            processing_latency_ms: float) -> int:
        """
        Optimize frame buffer size based on FPS and processing latency.
        
        Args:
            current_buffer_size: Current buffer size
            fps: Current frames per second
            processing_latency_ms: Processing latency in milliseconds
            
        Returns:
            Recommended buffer size
        """
        # Calculate how many frames we might need based on latency
        latency_seconds = processing_latency_ms / 1000.0
        frames_for_latency = max(1, int(fps * latency_seconds * 2))  # 2x for safety
        
        recommended_size = max(self.min_queue_size, 
                             min(self.max_queue_size, frames_for_latency))
        
        self.buffer_usage_history.append(recommended_size)
        self.logger.debug(f"Frame buffer optimization: {current_buffer_size} -> {recommended_size} "
                         f"(fps: {fps:.2f}, latency: {processing_latency_ms:.2f}ms)")
        
        return recommended_size
    
    def get_optimal_settings(self) -> Dict[str, Any]:
        """
        Get all current optimal settings.
        
        Returns:
            Dictionary with optimal settings
        """
        return {
            'recommended_queue_size': self._get_average_queue_size(),
            'recommended_buffer_size': self._get_average_buffer_size()
        }
    
    def _get_average_queue_size(self) -> int:
        """Get average recommended queue size from history."""
        if not self.queue_size_history:
            return self.target_queue_size
        return int(sum(self.queue_size_history) / len(self.queue_size_history))
    
    def _get_average_buffer_size(self) -> int:
        """Get average recommended buffer size from history."""
        if not self.buffer_usage_history:
            return self.frame_buffer_size
        return int(sum(self.buffer_usage_history) / len(self.buffer_usage_history))


def monitor_performance(func: Callable) -> Callable:
    """
    Decorator to monitor performance of a function.
    
    Args:
        func: Function to monitor
        
    Returns:
        Monitored function
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = (time.time() - start_time) * 1000  # Convert to ms
            logging.info(f"{func.__name__} executed in {execution_time:.2f}ms")
            return result
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000  # Convert to ms
            logging.error(f"{func.__name__} failed after {execution_time:.2f}ms: {e}")
            raise
    return wrapper


class LoadBalancer:
    """
    Load balancer for distributing work across available processing units.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the load balancer.
        
        Args:
            config: Configuration dictionary (optional)
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Determine available processing units
        self.cpu_count = mp.cpu_count()
        self.gpu_count = len(GPUtil.getGPUs())
        
        # Initialize load tracking
        self.load_history = defaultdict(list)
        self.current_load = defaultdict(float)
        
        self.logger.info(f"Load balancer initialized with {self.cpu_count} CPUs and {self.gpu_count} GPUs")
    
    def get_optimal_device(self, task_type: str = "general") -> str:
        """
        Get the optimal device for a given task based on current loads.
        
        Args:
            task_type: Type of task (affects device selection)
            
        Returns:
            Recommended device ('cpu', 'gpu', etc.)
        """
        # Simple load balancing algorithm
        cpu_load = self.current_load.get('cpu', 0)
        gpu_load = self.current_load.get('gpu', 0) if self.gpu_count > 0 else float('inf')
        
        if task_type == "compute_intensive" and self.gpu_count > 0:
            # Prefer GPU for compute-intensive tasks if available and not overloaded
            if gpu_load < cpu_load * 0.7:  # GPU is significantly less loaded
                return "gpu"
        
        # Otherwise choose the less loaded option
        return "gpu" if gpu_load < cpu_load and self.gpu_count > 0 else "cpu"
    
    def update_load(self, device: str, load_value: float):
        """
        Update the load value for a specific device.
        
        Args:
            device: Device identifier ('cpu', 'gpu', etc.)
            load_value: Load value (0-100 for percentage)
        """
        self.current_load[device] = load_value
        self.load_history[device].append((time.time(), load_value))
        
        # Keep only recent history (last 100 entries)
        if len(self.load_history[device]) > 100:
            self.load_history[device] = self.load_history[device][-100:]
    
    def get_load_distribution(self) -> Dict[str, float]:
        """
        Get current load distribution across devices.
        
        Returns:
            Dictionary with device loads
        """
        return dict(self.current_load)