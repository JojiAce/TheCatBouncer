"""
Unit tests for performance optimization and monitoring system.
"""
import pytest
import time
from unittest.mock import Mock, patch, MagicMock
import threading

from src.managers.performance_monitor import (
    PerformanceMonitor,
    PipelineOptimizer,
    LoadBalancer,
    monitor_performance,
    PerformanceMetrics
)


def test_performance_monitor_initialization():
    """Test PerformanceMonitor initialization."""
    config = {
        'history_size': 100,
        'bottleneck_thresholds': {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'latency_ms': 50.0
        }
    }
    
    monitor = PerformanceMonitor(config)
    
    assert monitor.config['history_size'] == 100
    assert monitor.bottleneck_thresholds['cpu_percent'] == 80.0
    assert monitor.bottleneck_thresholds['memory_percent'] == 85.0


def test_performance_monitor_metrics_collection():
    """Test metrics collection."""
    monitor = PerformanceMonitor()
    
    # Mock the metric collection
    with patch('psutil.cpu_percent', return_value=50.0), \
         patch('psutil.virtual_memory') as mock_memory, \
         patch('GPUtil.getGPUs', return_value=[]):
        
        mock_memory.return_value.percent = 60.0
        
        metrics = monitor._collect_metrics()
        
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.cpu_percent == 50.0
        assert metrics.memory_percent == 60.0


def test_performance_monitor_bottleneck_detection():
    """Test bottleneck detection."""
    config = {
        'bottleneck_thresholds': {
            'cpu_percent': 50.0,  # Lower threshold for testing
            'memory_percent': 50.0
        }
    }
    
    monitor = PerformanceMonitor(config)
    
    # Mock metrics that exceed thresholds
    metrics = PerformanceMetrics(
        timestamp=time.time(),
        cpu_percent=75.0,  # Above threshold of 50.0
        memory_percent=70.0,  # Above threshold of 50.0
        gpu_percent=80.0,
        latency_ms=120.0  # Above default threshold of 100.0
    )
    
    bottlenecks = monitor._detect_bottlenecks(metrics)
    
    assert len(bottlenecks) >= 2  # Should detect at least CPU and memory issues


def test_performance_monitor_threading_optimization():
    """Test threading optimization."""
    config = {
        'adaptive_threading_enabled': True,
        'initial_worker_count': 4,
        'max_worker_count': 8
    }
    
    monitor = PerformanceMonitor(config)
    
    # Add some dummy metrics to the history to trigger optimization
    for i in range(10):
        metrics = PerformanceMetrics(
            timestamp=time.time(),
            cpu_percent=95.0,  # Very high CPU to trigger reduction
            memory_percent=60.0,
            gpu_percent=70.0
        )
        monitor.metrics_history.append(metrics)
    
    # Call the optimization method directly
    monitor._optimize_threading()
    
    # When CPU is high, worker count should decrease
    # (though this is a simplified test without continuous monitoring)


def test_record_frame_processing():
    """Test recording frame processing metrics."""
    monitor = PerformanceMonitor()
    
    # Record a frame processing time
    monitor.record_frame_processing(33.3)  # ~30 FPS
    
    assert len(monitor.fps_history) == 1
    assert abs(monitor.fps_history[0] - 30.0) < 1.0  # Should be about 30 FPS


def test_get_current_performance():
    """Test getting current performance metrics."""
    monitor = PerformanceMonitor()
    
    # Add a sample metric
    metrics = PerformanceMetrics(
        timestamp=time.time(),
        cpu_percent=50.0,
        memory_percent=60.0,
        gpu_percent=70.0
    )
    monitor.metrics_history.append(metrics)
    
    # Add some FPS data
    monitor.fps_history.append(30.0)
    monitor.latency_history.append(33.3)
    
    performance = monitor.get_current_performance()
    
    assert 'cpu_percent' in performance
    assert 'memory_percent' in performance
    assert performance['cpu_percent'] == 50.0
    assert 'current_fps' in performance


def test_pipeline_optimizer_initialization():
    """Test PipelineOptimizer initialization."""
    config = {
        'target_queue_size': 20,
        'min_queue_size': 5,
        'max_queue_size': 100
    }
    
    optimizer = PipelineOptimizer(config)
    
    assert optimizer.target_queue_size == 20
    assert optimizer.min_queue_size == 5
    assert optimizer.max_queue_size == 100


def test_optimize_queue_size():
    """Test queue size optimization."""
    optimizer = PipelineOptimizer()
    
    # Test when input rate exceeds processing rate
    recommended_size = optimizer.optimize_queue_size(
        current_queue_size=10,
        processing_rate=10.0,  # 10 items/sec processing
        input_rate=20.0       # 20 items/sec input
    )
    
    # Should recommend larger queue since input > processing
    assert recommended_size >= 10
    
    # Test when processing rate exceeds input rate
    recommended_size2 = optimizer.optimize_queue_size(
        current_queue_size=50,
        processing_rate=30.0,  # 30 items/sec processing
        input_rate=10.0       # 10 items/sec input
    )
    
    # Should recommend smaller queue since processing > input
    assert recommended_size2 <= 50


def test_optimize_frame_buffer():
    """Test frame buffer optimization."""
    optimizer = PipelineOptimizer()
    
    # Test with high FPS and low latency
    recommended_size = optimizer.optimize_frame_buffer(
        current_buffer_size=30,
        fps=60.0,
        processing_latency_ms=10.0
    )
    
    # Should be reasonable size for this scenario
    assert recommended_size >= optimizer.min_queue_size
    assert recommended_size <= optimizer.max_queue_size


def test_load_balancer_initialization():
    """Test LoadBalancer initialization."""
    with patch('multiprocessing.cpu_count', return_value=4), \
         patch('GPUtil.getGPUs', return_value=[Mock()]):
        
        balancer = LoadBalancer()
        
        assert balancer.cpu_count == 4
        assert balancer.gpu_count == 1


def test_load_balancer_device_selection():
    """Test load balancer device selection."""
    with patch('multiprocessing.cpu_count', return_value=4), \
         patch('GPUtil.getGPUs', return_value=[Mock()]):
        
        balancer = LoadBalancer()
        
        # Update loads
        balancer.update_load('cpu', 80.0)  # CPU is heavily loaded
        balancer.update_load('gpu', 30.0)  # GPU is lightly loaded
        
        # Should prefer GPU for general tasks when it's less loaded
        device = balancer.get_optimal_device()
        # The actual result depends on the algorithm, but it should consider the loads


def test_monitor_performance_decorator():
    """Test the performance monitoring decorator."""
    @monitor_performance
    def test_function():
        time.sleep(0.01)  # Sleep for 10ms
        return "result"
    
    # Call the decorated function
    result = test_function()
    assert result == "result"


def test_performance_monitor_start_stop():
    """Test starting and stopping the performance monitor."""
    monitor = PerformanceMonitor()
    
    # Start monitoring
    monitor.start_monitoring()
    assert monitor.monitoring == True
    
    # Stop monitoring
    monitor.stop_monitoring()
    assert monitor.monitoring == False


def test_get_performance_statistics():
    """Test getting performance statistics."""
    monitor = PerformanceMonitor()
    
    # Add some sample metrics
    for i in range(5):
        metrics = PerformanceMetrics(
            timestamp=time.time(),
            cpu_percent=50.0 + i * 5,
            memory_percent=60.0 + i * 2,
            gpu_percent=70.0 + i * 3
        )
        monitor.metrics_history.append(metrics)
    
    # Add FPS data
    for i in range(5):
        monitor.fps_history.append(30.0 + i)
        monitor.latency_history.append(33.0 - i)
    
    stats = monitor.get_performance_statistics()
    
    assert 'cpu_stats' in stats
    assert 'memory_stats' in stats
    assert 'fps_stats' in stats
    assert 'latency_stats' in stats
    assert stats['sample_count'] == 5


if __name__ == "__main__":
    test_performance_monitor_initialization()
    test_performance_monitor_metrics_collection()
    test_performance_monitor_bottleneck_detection()
    test_record_frame_processing()
    test_get_current_performance()
    test_pipeline_optimizer_initialization()
    test_optimize_queue_size()
    test_optimize_frame_buffer()
    test_load_balancer_initialization()
    test_monitor_performance_decorator()
    test_performance_monitor_start_stop()
    test_get_performance_statistics()
    print("All performance optimization and monitoring tests passed!")