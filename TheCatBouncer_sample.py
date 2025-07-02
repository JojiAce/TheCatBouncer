# -*- coding: utf-8 -*-

import os
import cv2
import datetime
import time
import logging
import shutil
import requests
import numpy as np
import pygame
import random
import platform
import subprocess
import threading
from pathlib import Path
from ultralytics import YOLO
from openvino.runtime import Core
import configparser
import psutil
import yaml
import argparse

# Disable SSL warnings
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

# =============================================================================
# LOAD CONFIGURATION
# =============================================================================
config = configparser.ConfigParser()
config_path = Path(__file__).parent / 'config.ini'
config.read(config_path)

# --- Time Management ---
START_TIME = datetime.datetime.strptime(config.get('TimeManagement', 'start_time'), '%H:%M').time()
END_TIME = datetime.datetime.strptime(config.get('TimeManagement', 'end_time'), '%H:%M').time()
DATA_MANAGEMENT_TIME = datetime.datetime.strptime(config.get('TimeManagement', 'data_management_time'), '%H:%M').time()

# --- Philips Hue ---
HUE_BRIDGE_IP = config.get('PhilipsHue', 'bridge_ip')
HUE_APP_KEY = config.get('PhilipsHue', 'app_key')
HUE_LIGHT_IDS = [item.strip() for item in config.get('PhilipsHue', 'light_ids').split(',')]

# --- Camera ---
CAMERA_INDEX = config.getint('Camera', 'camera_index')
LOW_RES = tuple(map(int, config.get('Camera', 'low_resolution').split(',')))
HIGH_RES = tuple(map(int, config.get('Camera', 'high_resolution').split(',')))
FPS_LOW = config.getint('Camera', 'fps_low')
FPS_HIGH = config.getint('Camera', 'fps_high')

# --- Trigger ---
BRIGHTNESS_THRESHOLD = config.getint('Trigger', 'brightness_threshold')
BRIGHTNESS_PIXEL_PERCENTAGE = config.getfloat('Trigger', 'brightness_pixel_percentage')

# --- Paths ---
SCRIPT_DIRECTORY = Path(__file__).parent
BASE_SAVE_PATH = SCRIPT_DIRECTORY / config.get('Paths', 'base_storage_path')
VIDEO_PATH_DEBUG = BASE_SAVE_PATH / "VideoRecordings_Debug"
LOG_PATH_DAILY = BASE_SAVE_PATH / "DailyLogs"
LOG_PATH_YOLO = BASE_SAVE_PATH / "Cat_detection_information_debugging"
LOG_PATH_SYSTEM = BASE_SAVE_PATH / "SystemLogs"
SCARE_SOUND_DIRECTORY = SCRIPT_DIRECTORY / config.get('Paths', 'sound_directory')

# --- NAS Configuration ---
NAS_IP = config.get('NAS', 'nas_ip')
NAS_USER = config.get('NAS', 'nas_user')
NAS_TARGET_PATH = config.get('NAS', 'nas_destination_path')
NAS_WINDOWS_SHARE = config.get('NAS', 'nas_windows_share')

# --- Image Recognition ---
import torch
import importlib

YOLO_MODEL_PATH = config.get('ImageRecognition', 'yolo_model_path')
CAT_CLASS_ID = config.getint('ImageRecognition', 'cat_class_id')
CAT_CONFIDENCE_THRESHOLD = config.getfloat('ImageRecognition', 'cat_confidence_threshold')

# --- Color Analysis ---
LOWER_BLACK_HSV = tuple(map(int, config.get('ColorAnalysis', 'lower_black_hsv').split(',')))
UPPER_BLACK_HSV = tuple(map(int, config.get('ColorAnalysis', 'upper_black_hsv').split(',')))
BLACK_PIXEL_PERCENTAGE_THRESHOLD = config.getfloat('ColorAnalysis', 'black_pixel_threshold')

# --- Actions ---
INTRUDER_LIGHT_ON_MINUTES = config.getint('Actions', 'intruder_light_minutes')
# This line reads the setting from config.ini and controls whether the window is displayed.
SHOW_LIVE_DEBUG_WINDOW = config.getboolean('Actions', 'show_live_window')

# --- Storage Management ---
MIN_FREE_SPACE_GB = config.getint('StorageManagement', 'min_free_space_gb')
MAX_FILE_AGE_DAYS = config.getint('StorageManagement', 'max_file_age_days')

# =============================================================================
# HELPER FUNCTIONS & LOGGING
# =============================================================================
def setup_environment():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [SETUP] - %(message)s', handlers=[logging.StreamHandler()])
    logging.info("Starting environment setup...")
    logging.info(f"Base storage path will be: {BASE_SAVE_PATH}")
    all_paths = {"Data-Base": BASE_SAVE_PATH, "Video-Debug": VIDEO_PATH_DEBUG, "Daily-Logs": LOG_PATH_DAILY, "YOLO-Logs": LOG_PATH_YOLO, "System-Logs": LOG_PATH_SYSTEM, "Sound-Files": SCARE_SOUND_DIRECTORY}
    setup_successful = True
    for name, path in all_paths.items():
        try:
            os.makedirs(path, exist_ok=True)
            logging.info(f"Folder '{name}' at '{path}' is ready.")
        except OSError as e:
            logging.error(f"FATAL: Could not create folder '{name}' at '{path}'. Error: {e}")
            setup_successful = False
    if not setup_successful:
        logging.error("Setup failed. Program cannot run safely. Exiting.")
        return False
    system_log_filename = LOG_PATH_SYSTEM / f"general_system_log_{datetime.date.today().strftime('%d.%m.%Y')}.log"
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(system_log_filename, encoding='utf-8'), logging.StreamHandler()])
    logging.info(f"Storage location for data: {BASE_SAVE_PATH}")
    logging.info("System logging for today started and all folders successfully checked/created.")
    return True

def get_daily_log_filename():
    return LOG_PATH_DAILY / f"log_for_{datetime.date.today().strftime('%d.%m.%Y')}_VideoRecordings_LivingRoom_for_Debugging.log"

def log_daily_event(message):
    try:
        with open(get_daily_log_filename(), 'a', encoding='utf-8') as f:
            f.write(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
    except Exception as e:
        logging.error(f"Error writing to daily log: {e}")

def get_video_filename():
    today_str = datetime.date.today().strftime('%d.%m.%Y')
    now_str = datetime.datetime.now().strftime('%H-%M')
    count = 1
    while True:
        full_path = VIDEO_PATH_DEBUG / f"{today_str}_{now_str}_{count}_VideoRecording_LivingRoom_for_Debugging.mp4"
        if not os.path.exists(full_path):
            return str(full_path)
        count += 1

yolo_detection_folder = None
def log_yolo_detection(frame, box, event_start_time):
    global yolo_detection_folder
    try:
        if yolo_detection_folder is None:
            yolo_detection_folder = LOG_PATH_YOLO / event_start_time.strftime('%d.%m.%Y_%H-%M-%S')
            os.makedirs(yolo_detection_folder, exist_ok=True)
            logging.info(f"New YOLO debug folder created: {yolo_detection_folder}")
        now = datetime.datetime.now()
        image_path = yolo_detection_folder / f"frame_{now.strftime('%H-%M-%S-%f')}.jpg"
        cv2.imwrite(str(image_path), frame)
        log_path = yolo_detection_folder / "detection_coordinates.log"
        x1, y1, x2, y2 = map(int, box)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"Frame at {now.strftime('%H:%M:%S.%f')}: Box(x1,y1,x2,y2) = ({x1},{y1},{x2},{y2})\n")
    except Exception as e:
        logging.error(f"Error saving YOLO detection: {e}")

def is_active_time():
    now = datetime.datetime.now().time()
    if START_TIME > END_TIME: return now >= START_TIME or now <= END_TIME
    else: return START_TIME <= now <= END_TIME

# =============================================================================
# HARDWARE INTERACTION (SIMPLIFIED)
# =============================================================================
def philips_hue_control(state: bool):
    if not all([HUE_BRIDGE_IP, HUE_APP_KEY, HUE_LIGHT_IDS]): logging.error("Philips Hue configuration is incomplete."); return False
    all_successful, status_text = True, "TURNED ON" if state else "TURNED OFF"
    for item_id in HUE_LIGHT_IDS:
        url = f"https://{HUE_BRIDGE_IP}/clip/v2/resource/light/{item_id}"
        headers = {'hue-application-key': HUE_APP_KEY}; body = {"on": {"on": state}}
        try:
            response = requests.put(url, json=body, headers=headers, verify=False, timeout=5)
            response.raise_for_status()
            logging.info(f"ACTION: Philips Hue device '{item_id}' successfully {status_text}.")
            log_daily_event(f"Hue device '{item_id}' was {status_text}.")
        except requests.exceptions.RequestException as e:
            logging.error(f"Error controlling Hue device '{item_id}': {e}"); all_successful = False
    return all_successful

def play_scare_sound():
    if not os.path.isdir(SCARE_SOUND_DIRECTORY): logging.error(f"Sound folder not found: {SCARE_SOUND_DIRECTORY}"); return False
    sound_files = [f for f in os.listdir(SCARE_SOUND_DIRECTORY) if f.endswith(('.mp3', '.wav'))]
    if not sound_files: logging.warning(f"No sound files found in '{SCARE_SOUND_DIRECTORY}'."); return False
    try:
        sound_file = random.choice(sound_files)
        pygame.mixer.init()
        pygame.mixer.music.load(SCARE_SOUND_DIRECTORY / sound_file); pygame.mixer.music.play()
        logging.info(f"ACTION: Playing sound '{sound_file}'.")
        log_daily_event(f"Played sound '{sound_file}'.")
        return True
    except pygame.error as e: logging.error(f"Error playing sound: {e}"); return False

# =============================================================================
# CORE LOGIC & MAIN WORKFLOW
# =============================================================================
def get_camera_backend():
    """
    Returns the operating system-specific camera backend for OpenCV.
    For macOS, CAP_AVFOUNDATION is used explicitly to avoid stability issues.
    """
    if platform.system() == "Windows":
        return cv2.CAP_DSHOW
    elif platform.system() == "Darwin": # macOS
        return cv2.CAP_AVFOUNDATION
    else:
        return cv2.CAP_ANY # Linux and others

class ThreadedVideoStream:
    """Camera object that runs the camera stream in a separate thread"""
    def __init__(self, src=0, name="ThreadedVideoStream"):
        self.stream = cv2.VideoCapture(src, get_camera_backend())
        if not self.stream.isOpened():
            logging.error(f"[{name}] Could not open camera at index {src}")
            self.grabbed = False
            return

        (self.grabbed, self.frame) = self.stream.read()
        self.name = name
        self.stopped = False

    def start(self):
        t = threading.Thread(target=self.update, name=self.name, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while not self.stopped:
            (self.grabbed, self.frame) = self.stream.read()
        self.stream.release()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

    def set(self, propId, value):
        if self.stream.isOpened():
            self.stream.set(propId, value)

    def get(self, propId):
        if self.stream.isOpened():
            return self.stream.get(propId)
        return None

    def isOpened(self):
        return self.stream.isOpened()

def is_light_triggered(frame):
    if frame is None:
        return False
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bright_pixels = np.sum(gray_frame > BRIGHTNESS_THRESHOLD)
    percentage = (bright_pixels / (frame.shape[0] * frame.shape[1]))
    if percentage > BRIGHTNESS_PIXEL_PERCENTAGE:
        logging.info(f"MOTION SENSOR LIGHT DETECTED! Brightness: {percentage:.2%}"); return True
    return False

def analyze_cat_color(frame, box):
    x1, y1, x2, y2 = map(int, box)
    cat_roi = frame[y1:y2, x1:x2]
    if cat_roi.size == 0: return 'unknown'
    hsv_roi = cv2.cvtColor(cat_roi, cv2.COLOR_BGR2HSV)
    black_mask = cv2.inRange(hsv_roi, LOWER_BLACK_HSV, UPPER_BLACK_HSV)
    black_percentage = np.sum(black_mask > 0) / (cat_roi.shape[0] * cat_roi.shape[1]) if (cat_roi.shape[0] * cat_roi.shape[1]) > 0 else 0
    logging.info(f"Color analysis result: {black_percentage:.2%} of pixels in the cat area are 'black'.")
    log_daily_event(f"Color analysis: {black_percentage:.2%} of pixels are in the defined black range.")
    return 'black' if black_percentage >= BLACK_PIXEL_PERCENTAGE_THRESHOLD else 'white'

def passive_monitoring(cap):
    """
    Phase A: Passively monitors for brightness changes.
    Takes a persistent camera object instead of creating a new one.
    """
    logging.info(f"PASSIVE MONITORING started. Resolution: {LOW_RES}, FPS: {FPS_LOW}")
    triggered = False
    
    if not cap.isOpened():
        logging.error("[Passive] Error: Passed camera object is not open.")
        time.sleep(10)
        return False
        
    # Adjust camera settings for passive monitoring
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, LOW_RES[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, LOW_RES[1])
    cap.set(cv2.CAP_PROP_FPS, FPS_LOW)
    time.sleep(1) # Give the camera time to apply the settings

    while is_active_time():
        frame = cap.read() # Read frame from the threaded stream
        if frame is None:
            logging.warning("[Passive] Could not read frame.");
            time.sleep(1)
            continue
        if is_light_triggered(frame):
            triggered = True
            break
        # Short pause to relieve the CPU
        time.sleep(1 / FPS_LOW)
        
    logging.info("[Passive] Passive monitoring finished.")
    # IMPORTANT: cap.release() is no longer called here
    return triggered

def active_analysis(cap, model, class_names, window_name):
    """
    Phase B: Performs the analysis.
    Takes a persistent camera object and a window name.
    """
    global yolo_detection_folder
    yolo_detection_folder = None
    event_start_time = datetime.datetime.now()
    logging.info("ACTIVE ANALYSIS started.")
    philips_hue_control(True)

    out, detected_cat_type = None, None
    frames_to_write = []

    try:
        if not cap.isOpened():
            logging.error("[Active] Error: Passed camera object is not open.")
            philips_hue_control(False)
            return None

        # Adjust camera settings for active analysis
        logging.info(f"Requesting high resolution: {HIGH_RES}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, HIGH_RES[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HIGH_RES[1])
        cap.set(cv2.CAP_PROP_FPS, FPS_HIGH)
        time.sleep(2) # Camera warmup

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Log the actual resolution
        logging.info(f"Actual resolution received from camera: {width}x{height}")

        if width == 0 or height == 0:
             logging.error(f"[Active] Error: Camera resolution is invalid ({width}x{height}). Stopping analysis.")
             philips_hue_control(False)
             return None

        video_filename, fourcc = get_video_filename(), cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_filename, fourcc, FPS_HIGH, (width, height))
        log_daily_event(f"Video recording started: {os.path.basename(video_filename)}")

        analysis_duration, start_time_analysis = 30, time.time()
        prev_frame_time, new_frame_time = 0, 0

        while time.time() - start_time_analysis < analysis_duration:
            frame = cap.read()
            if frame is None:
                logging.warning("[Active] Frame from camera was None, skipping.")
                time.sleep(0.01) # Short pause to avoid CPU spinning
                continue

            # =========================================================================
            #  1. Inference and creation of a COMPLETE detection list
            # =========================================================================
            all_detections = []
            if BACKEND == 'torch':
                results = model(frame, verbose=False, device=INFERENCE_DEVICE)
                for r in results:
                    if hasattr(r, 'boxes') and r.boxes:
                        for box in r.boxes:
                            all_detections.append({
                                'xyxy': box.xyxy[0].tolist(),
                                'conf': box.conf.item(),
                                'cls': box.cls.item()
                            })
            elif BACKEND == 'openvino':
                input_layer = model.inputs[0]
                input_shape = input_layer.shape
                input_height, input_width = input_shape[2], input_shape[3]
                resized_frame = cv2.resize(frame, (input_width, input_height))
                input_tensor = resized_frame.astype(np.float32) / 255.0
                input_tensor = input_tensor.transpose(2, 0, 1)
                input_tensor = np.expand_dims(input_tensor, 0)
                
                results = model([input_tensor])
                output_tensor = results[model.outputs[0]].transpose((0, 2, 1))
                
                boxes, confidences, class_ids = [], [], []
                original_h, original_w = frame.shape[:2]
                x_factor, y_factor = original_w / input_width, original_h / input_height

                for row in output_tensor[0]:
                    cx, cy, w, h = row[:4]
                    class_scores = row[4:]
                    class_id, confidence = np.argmax(class_scores), class_scores[np.argmax(class_scores)]
                    if confidence > 0.1:
                        w_s, h_s, x_s, y_s = int(w * x_factor), int(h * y_factor), int((cx * x_factor) - w * x_factor / 2), int((cy * y_factor) - h * y_factor / 2)
                        boxes.append([x_s, y_s, w_s, h_s])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
                
                display_indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.1, 0.5)
                if len(display_indices) > 0:
                    for i in display_indices.flatten():
                        x, y, w, h = boxes[i]
                        all_detections.append({
                            'xyxy': [x, y, x + w, y + h],
                            'conf': confidences[i],
                            'cls': class_ids[i]
                        })

            # =========================================================================
            #  2. Actions based on HIGH-confidence detections
            # =========================================================================
            for detection in all_detections:
                if detected_cat_type is None and detection['cls'] == CAT_CLASS_ID and detection['conf'] > CAT_CONFIDENCE_THRESHOLD:
                    log_yolo_detection(frame, detection['xyxy'], event_start_time)
                    logging.info(f"CAT DETECTED with {detection['conf']:.2%} confidence!")
                    log_daily_event(f"Cat detected in frame with {detection['conf']:.2%} confidence.")
                    detected_cat_type = analyze_cat_color(frame, detection['xyxy'])
                    if detected_cat_type == 'black': handle_own_cat()
                    elif detected_cat_type == 'white': handle_intruder_cat()
            
            # =========================================================================
            #  3. Display ALL detected objects in the debug window
            # =========================================================================
            annotated_frame = frame.copy()
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time > 0 else 0
            prev_frame_time = new_frame_time

            # This check ensures that cv2.imshow is only called
            # if the window was actually created in the main loop.
            if SHOW_LIVE_DEBUG_WINDOW:
                if all_detections:
                    for det in all_detections:
                        x1, y1, x2, y2 = [int(c) for c in det['xyxy']]
                        conf, cls_id = det['conf'], int(det['cls'])
                        color = (0, 255, 0) if cls_id == CAT_CLASS_ID and conf > CAT_CONFIDENCE_THRESHOLD else (255, 0, 0)
                        label = f"{class_names[cls_id]}: {conf:.2f}" if cls_id < len(class_names) else "Unknown"
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                cv2.putText(annotated_frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # Use the passed window name
                cv2.imshow(window_name, annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): logging.info("Live view terminated by user."); break
            
            frames_to_write.append(annotated_frame)
            
        if detected_cat_type is None:
            log_daily_event("No cat detected within the analysis period.")
            philips_hue_control(False)
            
    finally:
        if out and frames_to_write:
            logging.info(f"Writing {len(frames_to_write)} cached frames to video file...")
            for frame_to_write in frames_to_write:
                out.write(frame_to_write)
            logging.info("Video file successfully written.")
        # IMPORTANT: cv2.destroyAllWindows() is no longer called here
        if out: 
            out.release()
        logging.info("[Active] Active analysis finished.")
    
    return detected_cat_type

def handle_own_cat():
    logging.info("Own cat detected. Turning off lights.")
    log_daily_event("Action: Own cat detected. Lights are being turned off.")
    philips_hue_control(False)

def handle_intruder_cat():
    logging.warning("Intruder detected! Playing deterrent sound.")
    log_daily_event("Action: Intruder detected. Sound is being played.")
    play_scare_sound()

# =============================================================================
# DATA MANAGEMENT (WITH NAS BACKUP)
# =============================================================================
def data_management_tasks():
    logging.info("Starting daily data management tasks...")
    is_windows = platform.system() == "Windows"
    placeholders_present = (NAS_IP == "YOUR_NAS_IP" or NAS_USER == "YOUR_NAS_USER" or NAS_TARGET_PATH == "/path/to/backup/folder" or (is_windows and NAS_WINDOWS_SHARE == "YOUR_WINDOWS_SHARE"))

    if placeholders_present:
        logging.warning("NAS configuration seems to contain placeholders. Skipping backup.")
    else:
        logging.info(f"Starting backup of '{BASE_SAVE_PATH}' to NAS ({NAS_IP})...")
        log_daily_event(f"[Data Management] Starting backup to NAS: {NAS_IP}")
        current_os, backup_successful = platform.system(), False
        try:
            timeout_seconds = 900
            if current_os in ["Linux", "Darwin"]:
                target = f"{NAS_USER}@{NAS_IP}:{NAS_TARGET_PATH}"
                command = ["rsync", "-avz", "--remove-source-files", str(BASE_SAVE_PATH) + "/", target]
                result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=timeout_seconds)
                logging.info("rsync backup completed successfully.")
                backup_successful = True
            elif current_os == "Windows":
                win_target_path = NAS_TARGET_PATH.replace("/", "\\").lstrip("\\")
                unc_path = f"\\\\{NAS_IP}\\{NAS_WINDOWS_SHARE}\\{win_target_path}"
                command = ["robocopy", str(BASE_SAVE_PATH), unc_path, "/E", "/MOV"]
                result = subprocess.run(command, capture_output=True, text=True, timeout=timeout_seconds)
                if result.returncode < 8:
                    logging.info(f"robocopy backup successful (Return Code: {result.returncode}).")
                    backup_successful = True
                else:
                    logging.error(f"robocopy failed with Return Code: {result.returncode}\n{result.stderr}")
        except Exception as e:
            logging.error(f"An unexpected error occurred during backup: {e}")
        log_daily_event("[Data Management] Backup to NAS successful." if backup_successful else "[Data Management] ERROR: Backup to NAS failed.")

    logging.info("Checking local disk space...")
    try:
        drive = Path(BASE_SAVE_PATH).anchor
        _, _, free = shutil.disk_usage(drive)
        free_gb = free / (1024**3)
        logging.info(f"Available disk space on drive '{drive}': {free_gb:.2f} GB.")
        if free_gb < MIN_FREE_SPACE_GB:
            logging.warning(f"Disk space below {MIN_FREE_SPACE_GB} GB. Deleting local files older than {MAX_FILE_AGE_DAYS} days.")
            cutoff, items_deleted = time.time() - (MAX_FILE_AGE_DAYS * 86400), 0
            for folder in [VIDEO_PATH_DEBUG, LOG_PATH_YOLO, LOG_PATH_DAILY, LOG_PATH_SYSTEM]:
                if not folder.exists(): continue
                for item in folder.glob('**/*'):
                    try:
                        if item.is_file() and item.stat().st_mtime < cutoff:
                            os.remove(item); logging.info(f"Locally deleted: {item}"); items_deleted += 1
                    except Exception as e: logging.error(f"Could not delete local file: {item}. Error: {e}")
            logging.warning(f"Local cleanup completed. {items_deleted} files deleted.")
    except Exception as e: logging.error(f"Error during storage cleanup: {e}")
    logging.info("Data management tasks finished.")

# =============================================================================
# MAIN
# =============================================================================
def main():
    # --- Add arguments for the preview window ---
    parser = argparse.ArgumentParser(description="Automatic Cat Bouncer.")
    parser.add_argument('-d', '--device', type=str, help="Overrides the inference device (e.g., 'cpu', 'gpu', 'cuda:0').")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--live-preview', action='store_true', help="Forces the display of the live preview window (overrides config.ini).")
    group.add_argument('--no-live-preview', action='store_true', help="Disables the live preview window (overrides config.ini).")
    args = parser.parse_args()

    if not setup_environment():
        time.sleep(10); return

    # --- Logic for prioritizing window display ---
    # Priority 1: Command line argument. Priority 2: config.ini.
    global SHOW_LIVE_DEBUG_WINDOW
    if args.live_preview:
        SHOW_LIVE_DEBUG_WINDOW = True
        logging.info("Live preview window FORCED by command line argument --live-preview.")
    elif args.no_live_preview:
        SHOW_LIVE_DEBUG_WINDOW = False
        logging.info("Live preview window DISABLED by command line argument --no-live-preview.")
    else:
        logging.info(f"Setting for preview window from config.ini used: {SHOW_LIVE_DEBUG_WINDOW}")

    
    # Device setting
    if args.device:
        inference_device_config = args.device
        logging.info(f"Inference device set to '{inference_device_config}' by command line argument.")
    else:
        inference_device_config = config.get('ImageRecognition', 'inference_device', fallback='cpu')
        logging.info(f"Inference device loaded from config.ini: '{inference_device_config}'.")

    # Set global variables for backend and device
    global INFERENCE_DEVICE, BACKEND
    raw_device = inference_device_config.lower()
    OV_AVAILABLE = importlib.util.find_spec('openvino.runtime') is not None

    if raw_device == 'gpu':
        if torch.cuda.is_available():
            INFERENCE_DEVICE = 'cuda:0'
            BACKEND = 'torch'
            logging.info("CUDA GPU found - using 'cuda:0'.")
        elif OV_AVAILABLE:
            if platform.system() == "Darwin":
                INFERENCE_DEVICE = 'CPU'
                logging.info("No CUDA GPU, but OpenVINO available on macOS - using iGPU via 'CPU'.")
            else:
                INFERENCE_DEVICE = 'GPU'
                logging.info("No CUDA GPU, but OpenVINO available - using iGPU via 'GPU'.")
            BACKEND = 'openvino'
        else:
            INFERENCE_DEVICE = 'cpu'
            BACKEND = 'torch'
            logging.warning("inference_device='gpu' configured, but neither CUDA nor OpenVINO found. Falling back to 'cpu'.")
    elif raw_device.startswith('cuda'):
        INFERENCE_DEVICE = raw_device
        BACKEND = 'torch'
        logging.info(f"Specific CUDA device '{INFERENCE_DEVICE}' will be used.")
    elif raw_device == 'openvino':
        INFERENCE_DEVICE = 'GPU'
        BACKEND = 'openvino'
        logging.info(f"OpenVINO backend explicitly configured for device: {INFERENCE_DEVICE}")
    else:
        INFERENCE_DEVICE = 'cpu'
        BACKEND = 'torch'
        logging.info("Using 'cpu' for inference.")

    logging.info(f"Final Inference-Device: {INFERENCE_DEVICE}, Backend: {BACKEND}")

    # Load model
    model_for_inference, class_names = None, []
    try:
        if BACKEND == 'openvino':
            core = Core()
            model_dir_name = Path(YOLO_MODEL_PATH).name
            model_size_prefix = model_dir_name.split('_')[0]
            openvino_model_xml_path = Path(YOLO_MODEL_PATH) / f"{model_size_prefix}.xml"
            metadata_path = Path(YOLO_MODEL_PATH) / "metadata.yaml"
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    class_names = [n for _, n in sorted(yaml.safe_load(f)['names'].items())]
                logging.info(f"{len(class_names)} class names extracted from metadata.yaml.")
            else: logging.warning(f"metadata.yaml not found at {metadata_path}.")
            ov_model = core.read_model(model=openvino_model_xml_path)
            model_for_inference = core.compile_model(model=ov_model, device_name=INFERENCE_DEVICE)
            logging.info(f"OpenVINO model '{openvino_model_xml_path}' loaded on '{INFERENCE_DEVICE}'.")
        else:
            model_for_inference = YOLO(YOLO_MODEL_PATH, task='detect')
            class_names = list(model_for_inference.names.values())
            logging.info(f"Model with Torch backend loaded. Device = {INFERENCE_DEVICE}")
    except Exception as e:
        logging.error(f"FATAL: Could not load YOLO model: {e}"); return

    logging.info("Program started. Waiting for activity time window...")
    last_data_management_check_date = datetime.date.today() - datetime.timedelta(days=1)

    # Create persistent camera object
    cap = ThreadedVideoStream(src=CAMERA_INDEX).start()
    if not cap.isOpened():
        logging.error("FATAL: Camera could not be initialized. Program will be terminated.")
        time.sleep(10)
        return

    try:
        while True:
            now = datetime.datetime.now()
            if now.time() >= DATA_MANAGEMENT_TIME and last_data_management_check_date != now.date():
                data_management_tasks()
                last_data_management_check_date = now.date()

            if is_active_time():
                logging.info("Activity time window reached. Starting monitoring cycle.")
                if passive_monitoring(cap):
                    
                    window_name = f"YOLO Live Detection ({YOLO_MODEL_PATH})"
                    
                    # It is checked here whether the window should be displayed according to the final setting.
                    if SHOW_LIVE_DEBUG_WINDOW:
                        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    
                    try:
                        # The analysis is always performed, but the window is only displayed if SHOW_LIVE_DEBUG_WINDOW is true.
                        detected_cat_type = active_analysis(cap, model_for_inference, class_names, window_name)
                        
                        if detected_cat_type == 'white':
                            logging.info(f"Intruder detected. Lights will stay on for {INTRUDER_LIGHT_ON_MINUTES} minutes.")
                            time.sleep(INTRUDER_LIGHT_ON_MINUTES * 60)
                            logging.info("Timer for intruder expired. Turning off lights.")
                            philips_hue_control(False)
                    finally:
                        # The window is only destroyed if it was previously created.
                        if SHOW_LIVE_DEBUG_WINDOW:
                            cv2.destroyWindow(window_name)

                if is_active_time():
                    logging.info("Waiting 10 seconds before the next monitoring cycle.")
                    time.sleep(10)
            else:
                time.sleep(60)
    except KeyboardInterrupt:
        logging.info("Program terminated by user.")
    finally:
        if cap:
            cap.stop()
        # Close all windows safely at the end
        cv2.destroyAllWindows()
        logging.info("Resources released. Program terminated.")

if __name__ == "__main__":
    main()
