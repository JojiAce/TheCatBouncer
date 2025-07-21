import cv2
import time
import logging
from pathlib import Path
import datetime
import random
import os


# Pygame wird für die Audio-Wiedergabe benötigt.
# Installation mit: pip install pygame
try:
    import pygame
except ImportError:
    pygame = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

def play_intruder_sound(sound_folder_path: str):
    """
    Spielt eine zufällige Sound-Datei aus einem Ordner ab.
    """
    if not pygame:
        logger.error("[Audio] Pygame-Bibliothek nicht gefunden. Audio kann nicht abgespielt werden. (pip install pygame)")
        return

    sound_folder = Path(sound_folder_path)
    if not sound_folder.is_dir():
        logger.warning(f"[Audio] Sound-Ordner '{sound_folder_path}' nicht gefunden.")
        return

    # Finde alle unterstützten Audio-Dateien
    supported_formats = ['.wav', '.mp3', '.ogg']
    sound_files = [f for f in sound_folder.iterdir() if f.suffix.lower() in supported_formats]

    if not sound_files:
        logger.warning(f"[Audio] Keine Sound-Dateien in '{sound_folder_path}' gefunden.")
        return

    # Wähle eine zufällige Datei aus
    chosen_sound = random.choice(sound_files)
    logger.info(f"[Audio] Wähle zufälligen Sound aus: {chosen_sound.name}")

    try:
        pygame.mixer.init()
        pygame.mixer.music.load(str(chosen_sound))
        pygame.mixer.music.play()
        logger.info(f"[Audio] Spiele jetzt '{chosen_sound.name}' ab.")
    except Exception as e:
        logger.error(f"[Audio] Fehler beim Abspielen der Sound-Datei: {e}")

def handle_intruder_event(
    duration_minutes: float, 
    camera_src: str, 
    camera_config: dict, 
    video_save_folder: str,
    sound_config: dict
):
    """
    Startet die "Intruder"-Aktionen: Videoaufnahme und Sound-Wiedergabe.

    Args:
        duration_minutes (float): Dauer der Aktionen in Minuten.
        camera_src (str): Die Kameraquelle.
        camera_config (dict): Konfiguration für Auflösung und FPS.
        video_save_folder (str): Der Hauptordner zum Speichern der Videos.
        sound_config (dict): Konfiguration für den Sound-Ordner.
    """
    # Schritt 1: Starte die Sound-Wiedergabe (läuft im Hintergrund)
    play_intruder_sound(sound_config.get('folder', 'cat_scare_sound'))

    # Schritt 2: Starte die Videoaufnahme für die definierte Dauer
    duration_sec = duration_minutes * 60
    logger.info(f"[Recorder] Starte 'Intruder'-Videoaufnahme für {duration_minutes} Minuten.")

    res = tuple(map(int, camera_config['high_resolution'].split(',')))
    fps = int(camera_config['fps_high'])

    cap = cv2.VideoCapture(int(camera_src))
    if not cap.isOpened():
        logger.error("[Recorder] Kann Kameraquelle für Aufnahme nicht öffnen.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, res[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])
    cap.set(cv2.CAP_PROP_FPS, fps)

    # Dateipfad für das Video erstellen
    folder = Path(video_save_folder)
    folder.mkdir(parents=True, exist_ok=True)
    filename = f"intruder_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    filepath = folder / filename

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(filepath), fourcc, fps, res)

    end_time = time.time() + duration_sec
    logger.info(f"[Recorder] Aufnahme läuft... wird gespeichert in: {filepath}")

    try:
        while time.time() < end_time:
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
            # Optional: cv2.waitKey(1) kann hier nötig sein, um den Stream flüssig zu halten
    finally:
        logger.info("[Recorder] Aufnahme beendet.")
        cap.release()
        writer.release()
        # Stoppe die Musik, falls sie noch läuft
        if pygame and pygame.mixer.get_init():
            pygame.mixer.music.stop()
            pygame.mixer.quit()
