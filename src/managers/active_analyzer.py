"""
High-performance multi-threaded active analysis pipeline.
"""
import cv2
import numpy as np
import logging
from multiprocessing import Queue, Process, Event
from typing import Dict, Any, List, Optional, Tuple
import time
import json
from pathlib import Path
import traceback
import importlib
import queue
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


@dataclass
class EngineSpec:
    """Serializable specification for spawning inference engine instances."""

    module: str
    class_name: str
    init_args: Tuple[Any, ...]
    init_kwargs: Dict[str, Any]

    def create_engine(self) -> InferenceEngine:
        engine_module = importlib.import_module(self.module)
        engine_cls = getattr(engine_module, self.class_name)
        return engine_cls(*self.init_args, **self.init_kwargs)


def _capture_worker(camera_source: int, resolution: Tuple[int, int], fps: int,
                    output_queue: Queue, stop_event: Event, error_queue: Queue,
                    logger_name: str) -> None:
    logger = logging.getLogger(logger_name)
    try:
        cap = cv2.VideoCapture(camera_source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        cap.set(cv2.CAP_PROP_FPS, fps)

        if not cap.isOpened():
            error_queue.put(f"capture_process: Cannot open camera {camera_source}")
            return

        logger.info(f"Capture process started with camera {camera_source}")

        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to capture frame, retrying...")
                time.sleep(0.01)
                continue

            try:
                if output_queue.full():
                    try:
                        output_queue.get_nowait()
                    except queue.Empty:
                        pass
                output_queue.put(frame, block=False)
            except queue.Full:
                continue

        cap.release()
        logger.info("Capture process ended")

    except Exception:
        error_msg = f"capture_process: {traceback.format_exc()}"
        error_queue.put(error_msg)


def _preprocess_worker(resolution: Tuple[int, int], input_queue: Queue,
                       output_queue: Queue, stop_event: Event, error_queue: Queue,
                       logger_name: str) -> None:
    logger = logging.getLogger(logger_name)
    try:
        logger.info("Preprocess process started")

        while not stop_event.is_set():
            try:
                frame = input_queue.get(timeout=1)
            except queue.Empty:
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if (rgb_frame.shape[1] != resolution[0] or
                    rgb_frame.shape[0] != resolution[1]):
                rgb_frame = cv2.resize(rgb_frame, resolution)

            output_queue.put(rgb_frame)

        logger.info("Preprocess process ended")

    except Exception:
        error_msg = f"preprocess_process: {traceback.format_exc()}"
        error_queue.put(error_msg)


def _inference_worker(engine_spec: EngineSpec, input_queue: Queue,
                      output_queue: Queue, stop_event: Event, error_queue: Queue,
                      worker_id: int, logger_name: str) -> None:
    logger = logging.getLogger(logger_name)
    try:
        logger.info(f"Inference process {worker_id} starting engine initialization")
        inference_engine = engine_spec.create_engine()
        logger.info(f"Inference process {worker_id} started")

        while not stop_event.is_set():
            try:
                frame = input_queue.get(timeout=1)
            except queue.Empty:
                continue

            start_time = time.time()
            detections = inference_engine.predict(frame)
            inference_time = (time.time() - start_time) * 1000

            result = DetectionResult(
                image=frame,
                detections=detections,
                inference_time_ms=inference_time
            )

            output_queue.put(result)

        logger.info(f"Inference process {worker_id} ended")

    except Exception:
        error_msg = f"inference_process_{worker_id}: {traceback.format_exc()}"
        error_queue.put(error_msg)


def _postprocess_worker(class_names: List[str], target_class_name: str,
                        min_confidence: float, save_detection_images: bool,
                        success_frame_folder: str, resolution: Tuple[int, int],
                        input_queue: Queue, display_queue: Queue,
                        result_queue: Queue, stop_event: Event,
                        success_event: Event, error_queue: Queue,
                        logger_name: str) -> None:
    logger = logging.getLogger(logger_name)
    try:
        logger.info("Postprocess process started")

        while not stop_event.is_set() and not success_event.is_set():
            try:
                result: DetectionResult = input_queue.get(timeout=1)
            except queue.Empty:
                continue

            target_found = ActiveAnalyzer._check_for_target_object(
                result,
                class_names,
                target_class_name,
                min_confidence,
                logger_name
            )

            if target_found:
                detection_path = ActiveAnalyzer._save_successful_detection(
                    result,
                    save_detection_images,
                    success_frame_folder,
                    resolution,
                    target_class_name,
                    min_confidence,
                    logger_name
                )
                if detection_path:
                    result_queue.put(detection_path)
                    success_event.set()
                    break

            if display_queue is not None and not display_queue.full():
                annotated_frame = ActiveAnalyzer._annotate_frame(
                    result.image,
                    result.detections,
                    result.inference_time_ms,
                    class_names
                )
                try:
                    display_queue.put_nowait(annotated_frame)
                except queue.Full:
                    pass

        logger.info("Postprocess process ended")

    except Exception:
        error_msg = f"postprocess_process: {traceback.format_exc()}"
        error_queue.put(error_msg)


def _display_worker(input_queue: Queue, stop_event: Event, error_queue: Queue,
                    logger_name: str, exit_key: str) -> None:
    logger = logging.getLogger(logger_name)
    try:
        logger.info("Display process started")

        window_name = "Active Analysis"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

        while not stop_event.is_set():
            try:
                frame = input_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(exit_key) or key == 27:
                stop_event.set()
                break

        cv2.destroyAllWindows()
        logger.info("Display process ended")

    except Exception:
        error_msg = f"display_process: {traceback.format_exc()}"
        error_queue.put(error_msg)


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

        self._engine_spec = self._build_engine_spec(inference_engine)

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
        display_queue = Queue(maxsize=self.display_queue_size) if self.show_window else None
        error_queue = Queue()

        # Create events for process coordination
        stop_event = Event()
        success_event = Event()

        if self._engine_spec is None:
            self.logger.error("Inference engine cannot be serialized for multiprocessing. Aborting analysis.")
            return None

        # Create processes for the pipeline
        processes = []

        # 1. Capture process
        capture_proc = Process(
            target=_capture_worker,
            args=(
                self.camera_source,
                self.high_resolution,
                self.fps_high,
                capture_queue,
                stop_event,
                error_queue,
                self.logger.name
            )
        )
        processes.append(capture_proc)

        # 2. Preprocessing process
        preprocess_proc = Process(
            target=_preprocess_worker,
            args=(
                self.high_resolution,
                capture_queue,
                preprocess_queue,
                stop_event,
                error_queue,
                self.logger.name
            )
        )
        processes.append(preprocess_proc)

        # 3. Inference processes (multiple workers for CPU)
        inference_workers = []
        num_workers = 1 if self.inference_device.lower() != 'cpu' else self.cpu_workers
        for i in range(num_workers):
            worker = Process(
                target=_inference_worker,
                args=(
                    self._engine_spec,
                    preprocess_queue,
                    inference_queue,
                    stop_event,
                    error_queue,
                    i,
                    self.logger.name
                )
            )
            processes.append(worker)
            inference_workers.append(worker)

        # 4. Postprocessing process (handles detection logic)
        class_names = self.inference_engine.get_class_names()
        postprocess_proc = Process(
            target=_postprocess_worker,
            args=(
                class_names,
                self.target_class_name,
                self.min_confidence,
                self.save_detection_images,
                self.success_frame_folder,
                self.high_resolution,
                inference_queue,
                display_queue,
                result_queue,
                stop_event,
                success_event,
                error_queue,
                self.logger.name
            )
        )
        processes.append(postprocess_proc)

        # 5. Display process (optional)
        if self.show_window:
            display_proc = Process(
                target=_display_worker,
                args=(
                    display_queue,
                    stop_event,
                    error_queue,
                    self.logger.name,
                    self.exit_key
                )
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

    def _build_engine_spec(self, engine: Optional[InferenceEngine]) -> Optional[EngineSpec]:
        if engine is None:
            self.logger.error("No inference engine provided; cannot start analysis.")
            return None

        spawn_params = getattr(engine, "get_spawn_params", None)
        init_args: Tuple[Any, ...]
        init_kwargs: Dict[str, Any]

        try:
            if callable(spawn_params):
                params = spawn_params()
                raw_args: Any = ()
                raw_kwargs: Dict[str, Any] = {}

                if isinstance(params, dict):
                    raw_kwargs = params
                elif isinstance(params, tuple):
                    if len(params) == 2 and isinstance(params[1], dict):
                        raw_args = params[0]
                        raw_kwargs = params[1]
                    else:
                        raw_args = params
                elif params is None:
                    raw_args = ()
                else:
                    raw_args = (params,)

                if isinstance(raw_args, (list, tuple)):
                    init_args = tuple(raw_args)
                elif raw_args in (None, ()):  # type: ignore[comparison-overlap]
                    init_args = ()
                else:
                    init_args = (raw_args,)

                init_kwargs = dict(raw_kwargs)
            else:
                model_path = getattr(engine, "model_path", None)
                init_args = (model_path,) if model_path is not None else ()
                init_kwargs = {}
                device = getattr(engine, "device", None)
                if device is not None:
                    init_kwargs['device'] = device

                extra_kwargs = getattr(engine, "spawn_kwargs", None)
                if isinstance(extra_kwargs, dict):
                    init_kwargs.update(extra_kwargs)

            return EngineSpec(
                module=engine.__class__.__module__,
                class_name=engine.__class__.__name__,
                init_args=init_args,
                init_kwargs=init_kwargs
            )
        except Exception as exc:
            self.logger.error(f"Failed to serialize inference engine for multiprocessing: {exc}")
            return None

    @staticmethod
    def _check_for_target_object(result: DetectionResult, class_names: List[str],
                                 target_class_name: str, min_confidence: float,
                                 logger_name: str) -> bool:
        logger = logging.getLogger(logger_name)
        try:
            if not result.detections or not result.detections[0]:
                return False

            detections = result.detections[0]
            for det in detections:
                if len(det) < 6:
                    continue

                x1, y1, x2, y2, conf, cls_id = det[:6]
                cls_id = int(cls_id)
                conf = float(conf)

                class_name = class_names[cls_id] if cls_id < len(class_names) else f"Class_{cls_id}"

                if class_name.lower() == target_class_name.lower() and conf >= min_confidence:
                    result.bbox_coords = (int(x1), int(y1), int(x2), int(y2))
                    result.confidence = conf
                    result.class_id = cls_id
                    result.class_name = class_name

                    logger.info(
                        f"Target {target_class_name} detected with confidence {conf:.2f} "
                        f"at coordinates {result.bbox_coords}"
                    )
                    return True

            return False

        except Exception as exc:
            logger.error(f"Error checking for target object: {exc}")
            return False

    @staticmethod
    def _annotate_frame(frame: np.ndarray, detections: List[np.ndarray],
                       inference_time: float, class_names: List[str]) -> np.ndarray:
        annotated_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if detections and detections[0] is not None:
            for det in detections[0]:
                if len(det) < 6:
                    continue

                x1, y1, x2, y2, conf, cls_id = det[:6]

                if conf < 0.5:
                    continue

                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cls_id = int(cls_id)

                if cls_id < len(class_names):
                    label = f"{class_names[cls_id]}: {conf:.2f}"
                else:
                    label = f"Class_{cls_id}: {conf:.2f}"

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        fps = 1000 / inference_time if inference_time > 0 else 0
        perf_text = f"Latency: {inference_time:.1f}ms (FPS: {fps:.1f})"
        cv2.putText(annotated_frame, perf_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return annotated_frame

    @staticmethod
    def _save_successful_detection(result: DetectionResult, save_detection_images: bool,
                                   success_frame_folder: str, resolution: Tuple[int, int],
                                   target_class_name: str, min_confidence: float,
                                   logger_name: str) -> Optional[str]:
        logger = logging.getLogger(logger_name)
        try:
            if not save_detection_images:
                return None

            detection_time = time.strftime("%Y-%m-%d_%H-%M-%S")
            detection_folder = Path(success_frame_folder) / detection_time
            detection_folder.mkdir(parents=True, exist_ok=True)

            frame_path = detection_folder / "frame.jpg"
            bgr_image = cv2.cvtColor(result.image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(frame_path), bgr_image)

            metadata = {
                "detection_time": detection_time,
                "bbox_coords": result.bbox_coords,
                "confidence": float(result.confidence) if result.confidence else None,
                "class_id": result.class_id,
                "class_name": result.class_name,
                "inference_time_ms": result.inference_time_ms,
                "resolution": resolution,
                "target_class": target_class_name,
                "min_confidence": min_confidence
            }

            metadata_path = detection_folder / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Successful detection saved to: {detection_folder}")
            return str(detection_folder)

        except Exception as exc:
            logger.error(f"Error saving successful detection: {exc}")
            return None
