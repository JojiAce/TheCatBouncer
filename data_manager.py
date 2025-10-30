import logging
import configparser
from pathlib import Path
import shutil
import datetime
import subprocess
import platform
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

def delete_old_data(config: configparser.ConfigParser):
    """Löscht alte Daten, um Speicherplatz freizugeben."""
    logger.info("Starte Überprüfung auf alte Daten...")
    try:
        storage_config = dict(config.items('StorageManagement'))
        min_free_gb = float(storage_config.get('min_free_space_gb', 10))
        max_age_days = int(storage_config.get('max_file_age_days', 2))
        
        # Ordner, die aufgeräumt werden sollen
        folders_to_clean = [
            config.get('Detection', 'success_frame_folder', fallback='successful_detections'),
            config.get('Actions', 'intruder_video_folder', fallback='intruder_recordings'),
            'logs',
            'videos'
        ]

        # 1. Prüfe freien Speicherplatz
        total, used, free = shutil.disk_usage("/")
        free_gb = free // (2**30)
        logger.info(f"Aktueller freier Speicherplatz: {free_gb} GB (Minimum: {min_free_gb} GB)")

        # 2. Lösche alte Dateien, wenn nötig
        now = datetime.datetime.now()
        cutoff_date = now - datetime.timedelta(days=max_age_days)
        
        if free_gb < min_free_gb:
            logger.warning("Freier Speicherplatz ist niedrig. Beginne mit dem Löschen alter Dateien.")
            deletion_needed = True
        else:
            logger.info("Ausreichend Speicherplatz vorhanden. Lösche nur Dateien, die älter als das Maximum sind.")
            deletion_needed = False

        for folder_name in folders_to_clean:
            folder = Path(folder_name)
            if not folder.is_dir():
                continue

            for item in sorted(folder.iterdir(), key=os.path.getmtime):
                try:
                    item_mtime = datetime.datetime.fromtimestamp(item.stat().st_mtime)
                    if item_mtime < cutoff_date or deletion_needed:
                        logger.info(f"Lösche alten Eintrag: {item} (Erstellt: {item_mtime.strftime('%Y-%m-%d')})")
                        if item.is_dir():
                            shutil.rmtree(item)
                        else:
                            item.unlink()
                        
                        # Wenn wir nur wegen niedrigem Speicherplatz löschen, prüfen wir erneut
                        if deletion_needed:
                            _, _, free = shutil.disk_usage("/")
                            if (free // (2**30)) >= min_free_gb:
                                logger.info("Ausreichend Speicherplatz wurde freigegeben. Stoppe vorzeitiges Löschen.")
                                deletion_needed = False

                except Exception as e:
                    logger.error(f"Fehler beim Löschen von {item}: {e}")

    except Exception as e:
        logger.error(f"Fehler im Datenmanagement (Löschen): {e}")


def backup_to_nas(config: configparser.ConfigParser):
    """Kopiert die Ereignisse des heutigen Tages auf ein NAS."""
    logger.info("Starte NAS-Backup für die heutigen Ereignisse...")
    try:
        nas_config = dict(config.items('NAS'))
        nas_ip = nas_config.get('nas_ip')
        nas_user = nas_config.get('nas_user')
        if not all([nas_ip, nas_user]):
            logger.error("NAS-Konfiguration unvollständig. Überspringe Backup.")
            return

        # Nur die Ordner mit den wichtigen Ereignissen sichern
        source_folders = [
            config.get('Detection', 'success_frame_folder', fallback='successful_detections'),
            config.get('Actions', 'intruder_video_folder', fallback='intruder_recordings')
        ]
        
        today_str = datetime.datetime.now().strftime('%Y-%m-%d')
        
        for folder_name in source_folders:
            source_path = Path(folder_name) / today_str
            if not source_path.is_dir():
                logger.info(f"Keine heutigen Ereignisse in '{source_path}' gefunden. Überspringe.")
                continue

            system = platform.system()
            command = []

            if system == "Windows":
                nas_share = nas_config.get('nas_windows_share')
                if not nas_share: 
                    logger.error("`nas_windows_share` fehlt in der Konfiguration für Windows. Überspringe Backup.")
                    continue
                # UNC-Pfad für Windows
                destination = f"\\\\{nas_ip}\\{nas_share}"
                # Robocopy ist das robusteste Werkzeug für Windows
                # /E: Kopiert Unterverzeichnisse, auch leere. /MIR: Spiegelt, löscht also alte Dateien am Ziel.
                command = ["robocopy", str(source_path), f"{destination}\\{source_path.name}", "/MIR"]
            
            else: # Linux & macOS
                destination_path = nas_config.get('nas_destination_path')
                if not destination_path:
                    logger.error("`nas_destination_path` fehlt in der Konfiguration. Überspringe Backup.")
                    continue
                destination = f"{nas_user}@{nas_ip}:{destination_path}"
                # rsync ist der Standard für Linux/macOS
                # -a: Archivmodus, -v: ausführlich, -z: komprimieren
                command = ["rsync", "-avz", "--delete", str(source_path), destination]
                logger.info("Hinweis: Für rsync wird ein passwortloser SSH-Zugang (via SSH-Key) zum NAS empfohlen.")

            logger.info(f"Führe Backup-Befehl aus: {' '.join(command)}")
            result = subprocess.run(command, capture_output=True, text=True)

            if result.returncode == 0 or (system == "Windows" and result.returncode < 8): # Robocopy hat spezielle Exit-Codes
                logger.info(f"Backup von '{source_path}' erfolgreich abgeschlossen.")
            else:
                logger.error(f"Backup von '{source_path}' fehlgeschlagen. Return Code: {result.returncode}")
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")

    except Exception as e:
        logger.error(f"Fehler im Datenmanagement (NAS-Backup): {e}")


def run_data_management(config: configparser.ConfigParser):
    """Führt alle Datenmanagement-Aufgaben aus."""
    logger.info("=== Starte tägliches Datenmanagement ===")
    
    # 1. Alte Daten löschen, um Platz zu schaffen
    delete_old_data(config)
    
    # 2. Heutige Daten auf NAS sichern
    backup_to_nas(config)
    
    logger.info("=== Tägliches Datenmanagement abgeschlossen ===")
