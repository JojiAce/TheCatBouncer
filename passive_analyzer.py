import cv2
import time
import logging
from typing import Callable
import numpy as np

# Logging so konfigurieren, dass es mit dem Hauptskript übereinstimmt
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

def brightness_check_strategy(frame: np.ndarray, config: dict) -> bool:
    """
    Standard-Strategie: Prüft, ob ein ausreichender Prozentsatz der Pixel hell genug ist.
    """
    brightness_threshold = config['brightness_threshold']
    pixel_percentage = config['brightness_pixel_percentage']

    # Konvertiere zu Graustufen für Helligkeitsanalyse
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Zähle die Pixel, die über dem Helligkeitsschwellenwert liegen
    bright_pixels = np.sum(gray_frame > brightness_threshold)
    
    total_pixels = gray_frame.size
    current_percentage = bright_pixels / total_pixels
    
    return current_percentage >= pixel_percentage

def run_passive_analysis(
    camera_src: str, 
    camera_config: dict, 
    trigger_config: dict,
    brightness_strategy: Callable[[np.ndarray, dict], bool] = brightness_check_strategy
) -> bool:
    """
    Überwacht einen Kamerastream auf eine signifikante Helligkeitsänderung.

    Args:
        camera_src (str): Die Kameraquelle (z.B. '0').
        camera_config (dict): Konfiguration für Auflösung und FPS.
        trigger_config (dict): Konfiguration für die Trigger-Logik.
        brightness_strategy (Callable): Eine Funktion, die einen Frame und die Konfig
                                        entgegennimmt und True zurückgibt, wenn er "hell" ist.

    Returns:
        bool: True, wenn der Trigger ausgelöst wurde.
    """
    res = tuple(map(int, camera_config['low_resolution'].split(',')))
    fps = int(camera_config['fps_low'])
    
    darkness_thresh = trigger_config['darkness_threshold']
    trigger_frame_count = trigger_config['trigger_frame_count']

    cap = cv2.VideoCapture(int(camera_src))
    if not cap.isOpened():
        logger.error(f"[Passive] Kann Kameraquelle nicht öffnen: {camera_src}")
        return False
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, res[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])
    cap.set(cv2.CAP_PROP_FPS, fps)
    
    state = "WAITING_FOR_DARKNESS"
    bright_frame_counter = 0
    
    logger.info(f"[Passive] Starte passive Analyse. Zustand: {state}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("[Passive] Kein Frame von Kamera empfangen.")
                time.sleep(1)
                continue

            # Berechne die durchschnittliche Helligkeit für die Dunkelheitsprüfung
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            avg_brightness = np.mean(gray_frame)

            if state == "WAITING_FOR_DARKNESS":
                if avg_brightness < darkness_thresh:
                    logger.info(f"[Passive] Dunkelheit erkannt (Avg Helligkeit: {avg_brightness:.1f}). Wechsle zu Zustand: WAITING_FOR_BRIGHTNESS")
                    state = "WAITING_FOR_BRIGHTNESS"
                    bright_frame_counter = 0
            
            elif state == "WAITING_FOR_BRIGHTNESS":
                # Prüfe, ob der Frame nach der gewählten Strategie "hell" ist
                if brightness_strategy(frame, trigger_config):
                    bright_frame_counter += 1
                    logger.debug(f"[Passive] Heller Frame erkannt. Zähler: {bright_frame_counter}/{trigger_frame_count}")
                    if bright_frame_counter >= trigger_frame_count:
                        logger.info("[Passive] TRIGGER! Stabile Helligkeit erkannt. Beende passive Analyse.")
                        return True
                else:
                    # Wenn ein dunkler Frame dazwischenkommt, zurück zum Anfang
                    logger.debug("[Passive] Helligkeit nicht stabil, setze Zähler zurück.")
                    bright_frame_counter = 0
                    # Optional: Man könnte hier auch wieder in den WAITING_FOR_DARKNESS Zustand gehen
                    # if avg_brightness < darkness_thresh:
                    #     state = "WAITING_FOR_DARKNESS"

            # Verhindere eine zu hohe CPU-Last
            time.sleep(1 / fps)

    except KeyboardInterrupt:
        logger.info("[Passive] Passive Analyse durch Benutzer unterbrochen.")
        return False
    finally:
        cap.release()
        logger.info("[Passive] Kamera freigegeben.")
