"""
High-performance multi-threaded active analysis pipeline.
"""
import cv2
import numpy as np
import logging
import multiprocessing as mp
from multiprocessing import Queue, Process, Event
from typing import Dict, Any, List, Optional, Tuple
import time
import json
from pathlib import Path
import traceback
from dataclasses import dataclass

from src.interfaces.monitoring import ActiveAnalyzer
from src.engines.base_engine import InferenceEngine


@dataclass
class DetectionResult:
    """Data class for detection results."""
    image: np.ndarray
    detections: List[np.ndarray]
    inference_time_ms: float
    bbox_coords: Optional[Tuple[int, int, int, int]] = None
    confidence: Optional[float] = None
    class_id: Optional[int] = None
    class_name: Optional[str] = None


class ActiveAnalyzer(ActiveAnalyzer):
    """
    High-performance multi-threaded active analysis pipeline.
    Implements optimized multiprocessing pipeline with capture, preprocess, 
    inference, and postprocess stages.
    """
    
    def __init__(self, config: Dict[str, Any], inference_engine):
        """
        Initialize the active analyzer.
        
        Args:
            config: Configuration dictionary
            inference_engine: Inference engine instance to use
        """
        self.config = config
        self.inference_engine = inference_engine
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Extract configuration values
        self.camera_source = config.get('camera_source', 0)
        self.high_resolution = tuple(config.get('high_resolution', (1920, 1080)))
        self.fps_high = config.get('fps_high', 30)
        self.timeout_sec = config.get('timeout_sec', 60)
        self.cpu_workers = config.get('cpu_workers', 1)
        self.inference_device = config.get('inference_device', 'cpu')
        self.show_window = config.get('show_live_window', True)
        self.exit_key = config.get('exit_key', 'q')
        self.target_class_name = config.get('target_class_name', 'cat')
        self.min_confidence = config.get('min_confidence', 0.8)
        self.success_frame_folder = config.get('success_frame_folder', 'successful_detections')
        self.save_detection_images = config.get('save_detection_images', True)
        
        # Performance configuration
        self.capture_queue_size = config.get('capture_queue_size', 2)
        self.preprocess_queue_size = config.get('preprocess_queue_size', self.cpu_workers)
        self.inference_queue_size = config.get('inference_queue_size', self.cpu_workers)
        self.display_queue_size = config.get('display_queue_size', 2)
        
        self.logger.info(f"Active analyzer initialized with config: "
                        f"resolution={self.high_resolution}, "
                        f"fps={self.fps_high}, "
                        f"timeout={self.timeout_sec}s, "
                        f"workers={self.cpu_workers}, "
                        f"target={self.target_class_name} "
                        f"min_conf={self.min_confidence}")
    
    def start_analysis(self) -> Optional[str]:
        """
        Start the active analysis process using a multi-process pipeline.
        
        Returns:
            Path to detection result if successful, None otherwise
        """
        self.logger.info("Starting active analysis pipeline...")
        
        # Create queues for inter-process communication
        capture_queue = Queue(maxsize=self.capture_queue_size)
        preprocess_queue = Queue(maxsize=self.preprocess_queue_size)
        inference_queue = Queue(maxsize=self.inference_queue_size)
        result_queue = Queue(maxsize=1)  # Only need to return one successful result
        display_queue = Queue(maxsize=self.display_queue_size)
        error_queue = Queue()
        
        # Create events for process coordination
        stop_event = Event()
        success_event = Event()
        
        # Create processes for the pipeline
        processes = []
        
        # 1. Capture process
        capture_proc = Process(
            target=self._capture_process,
            args=(capture_queue, stop_event, error_queue)
        )
        processes.append(capture_proc)
        
        # 2. Preprocessing process
        preprocess_proc = Process(
            target=self._preprocess_process,
            args=(capture_queue, preprocess_queue, stop_event, error_queue)
        )
        processes.append(preprocess_proc)
        
        # 3. Inference processes (multiple workers for CPU)
        inference_workers = []
        num_workers = 1 if self.inference_device.lower() != 'cpu' else self.cpu_workers
        for i in range(num_workers):
            worker = Process(
                target=self._inference_process,
                args=(preprocess_queue, inference_queue, stop_event, error_queue, i)
            )
            processes.append(worker)
            inference_workers.append(worker)
        
        # 4. Postprocessing process (handles detection logic)
        postprocess_proc = Process(
            target=self._postprocess_process,
            args=(inference_queue, display_queue, result_queue, stop_event, 
                  success_event, error_queue)
        )
        processes.append(postprocess_proc)
        
        # 5. Display process (optional)
        if self.show_window:
            display_proc = Process(
                target=self._display_process,
                args=(display_queue, stop_event, error_queue)
            )
            processes.append(display_proc)
        
        # Start all processes
        self.logger.info(f"Starting {len(processes)} pipeline processes...")
        for proc in processes:
            proc.start()
        
        # Monitor the analysis
        start_time = time.time()
        result_path = None
        
        try:
            while not stop_event.is_set():
                # Check for errors from any subprocess
                if not error_queue.empty():
                    error_msg = error_queue.get()
                    self.logger.error(f"Error from subprocess: {error_msg}")
                    stop_event.set()
                    break
                
                # Check if we've achieved success
                if success_event.is_set():
                    try:
                        result_path = result_queue.get_nowait()
                        self.logger.info(f"SUCCESS: Detection achieved, result stored at: {result_path}")
                        stop_event.set()
                        break
                    except:
                        pass  # Queue might be empty temporarily
                
                # Check for timeout
                if time.time() - start_time > self.timeout_sec:
                    self.logger.info(f"Analysis timed out after {self.timeout_sec} seconds")
                    stop_event.set()
                    break
                
                # Small delay to prevent busy-waiting
                time.sleep(0.1)
        
        except KeyboardInterrupt:
            self.logger.info("Active analysis interrupted by user")
            stop_event.set()
        
        finally:
            # Stop all processes gracefully
            self.logger.info("Stopping pipeline processes...")
            for proc in processes:
                proc.join(timeout=5)
                if proc.is_alive():
                    self.logger.warning(f"Process {proc.name} still alive after timeout, terminating...")
                    proc.terminate()
                    proc.join(timeout=2)
            
            # Clean up any remaining processes
            for proc in processes:
                if proc.is_alive():
                    proc.kill()
        
        self.logger.info("Active analysis pipeline completed")
        return result_path
    
    def stop_analysis(self):
        """
        Stop the active analysis process.
        """
        self.logger.info("Stopping active analysis...")
        # This would typically involve setting a stop event
        # For now, this method serves as a placeholder
    
    def _capture_process(self, output_queue: Queue, stop_event: Event, error_queue: Queue):
        """
        Capture process: reads frames from camera and puts them in queue.
        
        Args:
            output_queue: Queue to put captured frames
            stop_event: Event to signal stop
            error_queue: Queue to report errors
        """
        try:
            # Initialize camera
            cap = cv2.VideoCapture(self.camera_source)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.high_resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.high_resolution[1])
            cap.set(cv2.CAP_PROP_FPS, self.fps_high)
            
            if not cap.isOpened():
                error_queue.put(f"capture_process: Cannot open camera {self.camera_source}")
                return
            
            self.logger.info(f"Capture process started with camera {self.camera_source}")
            
            frame_count = 0
            while not stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    self.logger.warning("Failed to capture frame, retrying...")
                    time.sleep(0.01)
                    continue
                
                # Only put frame in queue if there's space (avoid blocking)
                try:
                    if output_queue.full():
                        # Remove oldest frame to keep queue fresh
                        try:
                            output_queue.get_nowait()
                        except:
                            pass
                    output_queue.put(frame, block=False)
                    frame_count += 1
                except:
                    # Queue is full, skip this frame
                    continue
            
            cap.release()
            self.logger.info("Capture process ended")
            
        except Exception as e:
            error_msg = f"capture_process: {traceback.format_exc()}"
            error_queue.put(error_msg)
    
    def _preprocess_process(self, input_queue: Queue, output_queue: Queue, 
                            stop_event: Event, error_queue: Queue):
        """
        Preprocess process: converts frames to RGB and resizes as needed.
        
        Args:
            input_queue: Queue to get frames from
            output_queue: Queue to put preprocessed frames
            stop_event: Event to signal stop
            error_queue: Queue to report errors
        """
        try:
            self.logger.info("Preprocess process started")
            
            while not stop_event.is_set():
                try:
                    # Get frame from input queue
                    frame = input_queue.get(timeout=1)
                    
                    # Convert BGR to RGB
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Resize if needed to match model input requirements
                    # (This would be based on model requirements)
                    target_resolution = self.high_resolution
                    if rgb_frame.shape[1] != target_resolution[0] or rgb_frame.shape[0] != target_resolution[1]:
                        rgb_frame = cv2.resize(rgb_frame, target_resolution)
                    
                    # Put preprocessed frame in output queue
                    output_queue.put(rgb_frame)
                    
                except:
                    # Timeout likely means no frame available, continue
                    continue
            
            self.logger.info("Preprocess process ended")
            
        except Exception as e:
            error_msg = f"preprocess_process: {traceback.format_exc()}"
            error_queue.put(error_msg)
    
    def _inference_process(self, input_queue: Queue, output_queue: Queue, 
                          stop_event: Event, error_queue: Queue, worker_id: int):
        """
        Inference process: runs model inference on frames.
        
        Args:
            input_queue: Queue to get frames from
            output_queue: Queue to put inference results
            stop_event: Event to signal stop
            error_queue: Queue to report errors
            worker_id: ID of this worker process
        """
        try:
            self.logger.info(f"Inference process {worker_id} started")
            
            # Create a new instance of the inference engine for this process
            # (Important for some frameworks that don't work well with multiprocessing)
            current_engine = self.inference_engine  # In a real implementation, we would need to recreate this
            
            while not stop_event.is_set():
                try:
                    # Get frame from input queue
                    frame = input_queue.get(timeout=1)
                    
                    # Record inference start time
                    start_time = time.time()
                    
                    # Run inference
                    detections = current_engine.predict(frame)
                    
                    # Calculate inference time
                    inference_time = (time.time() - start_time) * 1000  # Convert to ms
                    
                    # Create detection result
                    result = DetectionResult(
                        image=frame,
                        detections=detections,
                        inference_time_ms=inference_time
                    )
                    
                    # Put result in output queue
                    output_queue.put(result)
                    
                except:
                    # Timeout likely means no frame available, continue
                    continue
            
            self.logger.info(f"Inference process {worker_id} ended")
            
        except Exception as e:
            error_msg = f"inference_process_{worker_id}: {traceback.format_exc()}"
            error_queue.put(error_msg)
    
    def _postprocess_process(self, input_queue: Queue, display_queue: Queue, 
                            result_queue: Queue, stop_event: Event, 
                            success_event: Event, error_queue: Queue):
        """
        Postprocess process: analyzes detection results and handles success logic.
        
        Args:
            input_queue: Queue to get detection results from
            display_queue: Queue to send frames for display
            result_queue: Queue to send successful result path
            stop_event: Event to signal stop
            success_event: Event to signal successful detection
            error_queue: Queue to report errors
        """
        try:
            self.logger.info("Postprocess process started")
            
            while not stop_event.is_set() and not success_event.is_set():
                try:
                    # Get detection result from input queue
                    result: DetectionResult = input_queue.get(timeout=1)
                    
                    # Check for target object in detections
                    target_found = self._check_for_target_object(result)
                    
                    if target_found:
                        # Save the successful detection
                        detection_path = self._save_successful_detection(result)
                        if detection_path:
                            result_queue.put(detection_path)
                            success_event.set()
                            break
                    
                    # Send frame to display queue for visualization (if enabled)
                    try:
                        if not display_queue.full():
                            # Add detection annotations to frame for display
                            annotated_frame = self._annotate_frame(
                                result.image, 
                                result.detections, 
                                result.inference_time_ms
                            )
                            display_queue.put_nowait(annotated_frame)
                    except:
                        # Display queue is full, skip this frame
                        pass
                        
                except:
                    # Timeout likely means no result available, continue
                    continue
            
            self.logger.info("Postprocess process ended")
            
        except Exception as e:
            error_msg = f"postprocess_process: {traceback.format_exc()}"
            error_queue.put(error_msg)
    
    def _display_process(self, input_queue: Queue, stop_event: Event, error_queue: Queue):
        """
        Display process: shows frames in a window with detection annotations.
        
        Args:
            input_queue: Queue to get frames from
            stop_event: Event to signal stop
            error_queue: Queue to report errors
        """
        try:
            self.logger.info("Display process started")
            
            window_name = "Active Analysis"
            cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
            
            while not stop_event.is_set():
                try:
                    # Get frame from input queue
                    frame = input_queue.get(timeout=0.1)  # Short timeout for responsive UI
                    
                    # Display frame
                    cv2.imshow(window_name, frame)
                    
                    # Check for exit key
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord(self.exit_key) or key == 27:  # 'q' or ESC
                        stop_event.set()
                        break
                        
                except:
                    # Timeout means no frame available, continue
                    continue
            
            cv2.destroyAllWindows()
            self.logger.info("Display process ended")
            
        except Exception as e:
            error_msg = f"display_process: {traceback.format_exc()}"
            error_queue.put(error_msg)
    
    def _check_for_target_object(self, result: DetectionResult) -> bool:
        """
        Check if the target object (e.g., cat) is detected with sufficient confidence.
        
        Args:
            result: Detection result to analyze
            
        Returns:
            True if target object is detected with sufficient confidence, False otherwise
        """
        # Get class names from the engine
        class_names = self.inference_engine.get_class_names()
        
        # Process detections - this depends on the model format
        # Common YOLO format: [batch, [x1, y1, x2, y2, confidence, class_id], ...]
        try:
            # Check if we have detections
            if not result.detections or not result.detections[0]:
                return False
            
            detections = result.detections[0]  # Assuming batch size of 1
            
            # Look for target class with sufficient confidence
            for det in detections:
                if len(det) >= 6:  # x1, y1, x2, y2, conf, cls_id
                    x1, y1, x2, y2, conf, cls_id = det[:6]
                    
                    # Convert to integers for class_id
                    cls_id = int(cls_id)
                    conf = float(conf)
                    
                    # Check if this is our target class and meets confidence threshold
                    class_name = class_names[cls_id] if cls_id < len(class_names) else f"Class_{cls_id}"
                    
                    if (class_name.lower() == self.target_class_name.lower() and 
                        conf >= self.min_confidence):
                        
                        # Update result with detection info
                        result.bbox_coords = (int(x1), int(y1), int(x2), int(y2))
                        result.confidence = conf
                        result.class_id = cls_id
                        result.class_name = class_name
                        
                        self.logger.info(f"Target {self.target_class_name} detected "
                                       f"with confidence {conf:.2f} "
                                       f"at coordinates {result.bbox_coords}")
                        return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking for target object: {e}")
            return False
    
    def _annotate_frame(self, frame: np.ndarray, detections: List[np.ndarray], 
                       inference_time: float) -> np.ndarray:
        """
        Annotate frame with detection bounding boxes and inference information.
        
        Args:
            frame: Original frame to annotate
            detections: Detection results
            inference_time: Inference time in milliseconds
            
        Returns:
            Annotated frame
        """
        # Convert frame back to BGR for OpenCV
        annotated_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Get class names from the engine
        class_names = self.inference_engine.get_class_names()
        
        # Draw detections
        if detections and detections[0] is not None:
            for det in detections[0]:  # Assuming batch size of 1
                if len(det) >= 6:  # x1, y1, x2, y2, conf, cls_id
                    x1, y1, x2, y2, conf, cls_id = det[:6]
                    
                    # Only draw if confidence is above threshold
                    if conf >= 0.5:  # Show all detections with confidence > 0.5
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        cls_id = int(cls_id)
                        
                        # Get class name
                        if cls_id < len(class_names):
                            label = f"{class_names[cls_id]}: {conf:.2f}"
                        else:
                            label = f"Class_{cls_id}: {conf:.2f}"
                        
                        # Draw bounding box
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw performance info
        fps = 1000 / inference_time if inference_time > 0 else 0
        perf_text = f"Latency: {inference_time:.1f}ms (FPS: {fps:.1f})"
        cv2.putText(annotated_frame, perf_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return annotated_frame
    
    def _save_successful_detection(self, result: DetectionResult) -> Optional[str]:
        """
        Save the successful detection with image and metadata.
        
        Args:
            result: Successful detection result to save
            
        Returns:
            Path to saved detection folder, or None if saving failed
        """
        try:
            if not self.save_detection_images:
                return None
            
            # Create detection folder with timestamp
            detection_time = time.strftime("%Y-%m-%d_%H-%M-%S")
            detection_folder = Path(self.success_frame_folder) / detection_time
            detection_folder.mkdir(parents=True, exist_ok=True)
            
            # Save the frame
            frame_path = detection_folder / "frame.jpg"
            # Convert RGB to BGR for saving with OpenCV
            bgr_image = cv2.cvtColor(result.image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(frame_path), bgr_image)
            
            # Save detection metadata
            metadata = {
                "detection_time": detection_time,
                "bbox_coords": result.bbox_coords,
                "confidence": float(result.confidence) if result.confidence else None,
                "class_id": result.class_id,
                "class_name": result.class_name,
                "inference_time_ms": result.inference_time_ms,
                "resolution": self.high_resolution,
                "target_class": self.target_class_name,
                "min_confidence": self.min_confidence
            }
            
            metadata_path = detection_folder / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Successful detection saved to: {detection_folder}")
            return str(detection_folder)
            
        except Exception as e:
            self.logger.error(f"Error saving successful detection: {e}")
            return None