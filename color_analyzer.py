import cv2
import numpy as np
import json
from pathlib import Path
import logging

# Logging so konfigurieren, dass es mit dem Hauptskript übereinstimmt
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

def analyze_cat_color(event_folder_path: str, color_config: dict) -> bool:
    """
    Analysiert den Farbinhalt des erkannten Objekts in einem bestimmten Bild.

    Args:
        event_folder_path (str): Der Pfad zum Ordner, der das Bild und die Bounding-Box-Daten enthält.
        color_config (dict): Ein Dictionary mit den HSV-Werten und dem Schwellenwert.

    Returns:
        bool: True, wenn der prozentuale Anteil der Zielfarbe den Schwellenwert überschreitet, sonst False.
    """
    logger.info(f"Starte Farbanalyse für Ereignis-Ordner: {event_folder_path}")
    event_folder = Path(event_folder_path)

    # Pfade zu Bild und Bounding-Box-Daten definieren
    image_path = event_folder / 'frame.jpg'
    bbox_path = event_folder / 'bbox.json'

    # Prüfen, ob beide Dateien existieren
    if not image_path.exists() or not bbox_path.exists():
        logger.error(f"Fehler: Bild- oder Bounding-Box-Datei nicht im Ordner {event_folder} gefunden.")
        return False

    try:
        # Bild und Bounding-Box-Koordinaten laden
        image = cv2.imread(str(image_path))
        with open(bbox_path, 'r') as f:
            bbox_data = json.load(f)
        
        x1, y1, x2, y2 = bbox_data['x1'], bbox_data['y1'], bbox_data['x2'], bbox_data['y2']

        # Nur den Bereich innerhalb der Bounding Box (Region of Interest, ROI) ausschneiden
        cat_roi = image[y1:y2, x1:x2]
        if cat_roi.size == 0:
            logger.warning("Bounding Box hat eine Größe von 0. Analyse nicht möglich.")
            return False

        # Konvertiere den ROI in den HSV-Farbraum für eine robustere Farberkennung
        hsv_roi = cv2.cvtColor(cat_roi, cv2.COLOR_BGR2HSV)

        # Farbwerte aus der Konfiguration extrahieren
        lower_hsv = np.array(color_config['lower_hsv'])
        upper_hsv = np.array(color_config['upper_hsv'])
        threshold = color_config['pixel_threshold']

        # Eine Maske erstellen, die nur die Pixel im definierten Farbbereich enthält
        mask = cv2.inRange(hsv_roi, lower_hsv, upper_hsv)

        # Berechne den prozentualen Anteil der "schwarzen" Pixel
        total_pixels_in_roi = cat_roi.shape[0] * cat_roi.shape[1]
        color_pixels = cv2.countNonZero(mask)
        percentage = color_pixels / total_pixels_in_roi

        is_black = percentage >= threshold
        
        # Detaillierte Log-Ausgabe für das Debugging
        logger.info(f"Farb-Analyse Ergebnis: {percentage:.2%} der Pixel in der Bounding Box sind im Ziel-Farbbereich.")
        logger.info(f"Schwellenwert ist {threshold:.2%}. Ergebnis: {'Katze ist schwarz.' if is_black else 'Katze ist nicht schwarz.'}")

        return is_black

    except Exception as e:
        logger.error(f"Ein Fehler ist während der Farbanalyse aufgetreten: {e}")
        return False

