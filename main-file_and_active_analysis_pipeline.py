import cv2
import time
import datetime
import multiprocessing as mp
import configparser
from pathlib import Path
import os
import traceback
import logging
import json

# Eigene Module importieren
from NewPickerEnigneforActive import get_model_paths_from_config
from inference_factory import get_inference_engine
from color_analyzer import analyze_cat_color
from llm_color_analyzer import analyze_color_with_llm
from passive_analyzer import run_passive_analysis
from hue_controller import HueController
from utility_recorder import handle_intruder_event
from data_manager import run_data_management
from notifier import Notifier

# Logging konfigurieren
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

# --- HELFERFUNKTION FÜR ZEITMANAGEMENT ---
def is_within_schedule(start_str: str, end_str: str) -> bool:
    """
    Prüft, ob die aktuelle Uhrzeit innerhalb des definierten Zeitfensters liegt.
    Behandelt auch Zeitfenster, die über Mitternacht gehen (z.B. 18:00 - 07:00).
    """
    try:
        now = datetime.datetime.now().time()
        start_time = datetime.datetime.strptime(start_str, '%H:%M').time()
        end_time = datetime.datetime.strptime(end_str, '%H:%M').time()

        if start_time <= end_time:
            return start_time <= now <= end_time
        else:
            return start_time <= now or now <= end_time
    except ValueError:
        logger.error(f"Ungültiges Zeitformat in der Konfiguration. Bitte HH:MM verwenden.")
        return False

# --- PROZESS-DEFINITIONEN ---
def capture_proc(capture_q, src, res, fps, stop_event, error_q):
    """Liest Frames von der Kamera und legt sie in eine Queue."""
    try:
        cap = cv2.VideoCapture(int(src))
        if not cap.isOpened():
            raise IOError(f"Kann Kameraquelle nicht öffnen: {src}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, res[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])
        cap.set(cv2.CAP_PROP_FPS, fps)
        logger.info(f"Kamera {src} gestartet mit {res[0]}x{res[1]} @ {fps} FPS.")
        
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                logger.warning("Kein Frame von der Kamera empfangen. Stream-Ende?")
                time.sleep(0.1)
                continue
            
            if capture_q.full():
                capture_q.get_nowait()
            capture_q.put_nowait(frame)

    except Exception as e:
        error_q.put(f"capture_proc: {traceback.format_exc()}")
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        logger.info("Capture-Prozess beendet.")


def preprocess_proc(capture_q, preprocess_q, stop_event, target_res, error_q):
    """Holt Frames, konvertiert und skaliert sie."""
    try:
        while not stop_event.is_set():
            frame = capture_q.get(timeout=5)
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if img.shape[:2] != target_res:
                img = cv2.resize(img, target_res, interpolation=cv2.INTER_AREA)
            preprocess_q.put(img)
    except mp.queues.Empty:
        pass
    except Exception as e:
        error_q.put(f"preprocess_proc: {traceback.format_exc()}")
    finally:
        logger.info("Preprocess-Prozess beendet.")


def inference_proc(preprocess_q, result_q, stop_event, engine, error_q):
    """Führt Inferenz mit der VORGELADENEN Engine aus."""
    try:
        while not stop_event.is_set():
            img = preprocess_q.get(timeout=5)
            t0 = time.time()
            dets = engine.predict(img)
            infer_ms = (time.time() - t0) * 1000
            result_q.put((img, dets, infer_ms))
    except mp.queues.Empty:
        pass
    except Exception as e:
        error_q.put(f"inference_proc: {traceback.format_exc()}")
    finally:
        logger.info("Inference-Prozess beendet.")


def postprocess_proc(result_q, display_q, log_q, stop_event, success_event, result_path_q, error_q, detection_config, class_names):
    """
    Prüft auf das Zielobjekt, speichert bei Erfolg Frame und Bounding Box 
    und meldet den Pfad zurück.
    """
    target_name = detection_config['target_class_name']
    min_conf = detection_config['min_confidence']
    save_folder = Path(detection_config['success_frame_folder'])

    if not class_names:
        logger.error("Keine Klassennamen erhalten! Kann nicht nach Namen suchen.")
        return

    def draw_boxes_and_stats(img, dets, infer_ms):
        frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if dets and isinstance(dets, list) and dets[0] is not None:
             for *box, conf, cls_id in dets[0]:
                if conf >= 0.5:
                    x1, y1, x2, y2 = box
                    label = f"{class_names[int(cls_id)]}: {conf:.2f}"
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        fps_val = 1000 / infer_ms if infer_ms > 0 else 0
        text = f"Latency: {infer_ms:.1f}ms (FPS: {fps_val:.1f})"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return frame

    try:
        while not stop_event.is_set():
            img, dets, infer_ms = result_q.get(timeout=5)
            
            if dets and dets[0] is not None:
                for *box, conf, cls_id in dets[0]:
                    if class_names[int(cls_id)] == target_name and conf >= min_conf:
                        logger.info(f"ERFOLG! '{target_name}' mit Konfidenz {conf:.2f} gefunden.")
                        
                        event_time = datetime.datetime.now()
                        event_folder = save_folder / event_time.strftime('%Y-%m-%d') / event_time.strftime('%H-%M-%S_%f')
                        event_folder.mkdir(parents=True, exist_ok=True)

                        bgr_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(str(event_folder / 'frame.jpg'), bgr_image)
                        
                        x1, y1, x2, y2 = map(int, box)
                        bbox_data = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'confidence': f"{conf:.2f}"}
                        with open(event_folder / 'bbox.json', 'w') as f:
                            json.dump(bbox_data, f, indent=4)
                        
                        logger.info(f"Erfolgreiche Detektion gespeichert unter: {event_folder}")

                        result_path_q.put(str(event_folder))
                        success_event.set()
                        break 
            
            log_q.put((datetime.datetime.now(), infer_ms))
            if not display_q.full():
                annotated = draw_boxes_and_stats(img, dets, infer_ms)
                display_q.put_nowait(annotated)

            if success_event.is_set():
                break

    except Exception as e:
        error_q.put(f"postprocess_proc: {traceback.format_exc()}")
    finally:
        logger.info("Postprocess-Prozess beendet.")


def display_writer_proc(display_q, video_path, show_window, display_res, fps, stop_event, error_q, exit_key):
    writer = None
    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        writer = cv2.VideoWriter(video_path, fourcc, fps, display_res)
        
        while not stop_event.is_set():
            frame = display_q.get(timeout=5)
            if writer:
                writer.write(frame)
            if show_window:
                cv2.imshow('Active Analysis', frame)
                if cv2.waitKey(1) & 0xFF == ord(exit_key):
                    stop_evt.set()
                    break
    except mp.queues.Empty:
        pass
    except Exception as e:
        error_q.put(f"display_writer_proc: {traceback.format_exc()}")
    finally:
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        logger.info("Display/Writer-Prozess beendet.")


def logger_proc(log_q, stop_event, log_file, error_q):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    buffer = []
    try:
        with open(log_file, 'a') as f:
            f.write("timestamp,latency_ms,fps\n")
            while not stop_event.is_set() or not log_q.empty():
                try:
                    timestamp, infer_ms = log_q.get(timeout=1)
                    fps = 1000 / infer_ms if infer_ms > 0 else 0
                    buffer.append(f"{timestamp.isoformat()},{infer_ms:.1f},{fps:.1f}\n")
                    if len(buffer) >= 100:
                        f.writelines(buffer)
                        buffer.clear()
                except mp.queues.Empty:
                    continue
    except Exception as e:
        error_q.put(f"logger_proc: {traceback.format_exc()}")
    finally:
        if buffer:
            with open(log_file, 'a') as f:
                f.writelines(buffer)
        logger.info("Logger-Prozess beendet.")


# --- HAUPT-ORCHESTRIERUNGSFUNKTIONEN ---
def run_active_analysis(config, inference_engine, class_names) -> str | None:
    """
    Startet die Pipeline und gibt bei Erfolg den Pfad zum Ereignis-Ordner zurück.
    """
    CAMERA_SRC = config.get('Input', 'camera_src', fallback='0')
    HIGH_W, HIGH_H = map(int, config.get('Active', 'high_resolution').split(','))
    ACTIVE_RES = (HIGH_W, HIGH_H)
    FPS_HIGH = config.getint('Active', 'fps_high')
    ACTIVE_TIMEOUT_SEC = config.getint('Active', 'timeout_sec', fallback=60)
    CPU_WORKERS = config.getint('Backend', 'cpu_workers', fallback=1)
    inference_device = config.get('Backend', 'inference_device', fallback='cpu')
    SHOW_WINDOW = config.getboolean('Debug', 'window', fallback=True)
    EXIT_KEY = config.get('Debug', 'exit_key', fallback='q')

    detection_config = {
        'target_class_name': config.get('Detection', 'target_class_name'),
        'min_confidence': config.getfloat('Detection', 'min_confidence'),
        'success_frame_folder': config.get('Detection', 'success_frame_folder')
    }

    today = datetime.datetime.now().strftime('%Y-%m-%d')
    LOG_PATH = Path('logs') / today / 'analysis.log'
    VIDEO_PATH = Path('videos') / today / f'output_{datetime.datetime.now().strftime("%H-%M-%S")}.mp4'

    cap_q = mp.Queue(maxsize=1)
    pre_q = mp.Queue(maxsize=CPU_WORKERS)
    res_q = mp.Queue(maxsize=CPU_WORKERS)
    disp_q = mp.Queue(maxsize=1)
    log_q = mp.Queue(maxsize=1000)
    error_q = mp.Queue()
    stop_evt = mp.Event()
    success_event = mp.Event()
    result_path_q = mp.Queue(maxsize=1)

    procs = [
        mp.Process(target=capture_proc, args=(cap_q, CAMERA_SRC, ACTIVE_RES, FPS_HIGH, stop_evt, error_q)),
        mp.Process(target=preprocess_proc, args=(cap_q, pre_q, stop_evt, ACTIVE_RES, error_q)),
    ]
    
    workers = 1 if inference_device.lower() == 'gpu' else CPU_WORKERS
    for _ in range(workers):
        procs.append(mp.Process(target=inference_proc, args=(pre_q, res_q, stop_evt, inference_engine, error_q)))
        
    procs += [
        mp.Process(target=postprocess_proc, args=(res_q, disp_q, log_q, stop_evt, success_event, result_path_q, error_q, detection_config, class_names)),
        mp.Process(target=display_writer_proc, args=(disp_q, str(VIDEO_PATH), SHOW_WINDOW, ACTIVE_RES, FPS_HIGH, stop_evt, error_q, EXIT_KEY)),
        mp.Process(target=logger_proc, args=(log_q, stop_evt, str(LOG_PATH), error_q)),
    ]

    logger.info(f"Starte {len(procs)} Analyse-Prozesse...")
    for p in procs:
        p.start()

    start_time = time.time()
    try:
        while not stop_evt.is_set():
            if not error_q.empty():
                err = error_q.get()
                logger.error(f"FATALER FEHLER IN EINEM SUBPROZESS:\n{err}")
                stop_evt.set()
                break
            
            if success_event.is_set():
                logger.info("ERFOLG: Zielobjekt mit erforderlicher Konfidenz gefunden. Beende aktive Analyse.")
                stop_evt.set()
                break

            if time.time() - start_time > ACTIVE_TIMEOUT_SEC:
                logger.info(f"TIMEOUT: Zielobjekt nicht innerhalb von {ACTIVE_TIMEOUT_SEC} Sekunden gefunden. Beende aktive Analyse.")
                stop_evt.set()
                break
            
            time.sleep(0.5)
    except KeyboardInterrupt:
        logger.info("Strg+C erkannt. Beende Anwendung.")
        stop_evt.set()

    logger.info("Warte auf das Beenden aller Prozesse...")
    for p in procs:
        p.join(timeout=10)
        if p.is_alive():
            logger.warning(f"Prozess {p.name} (PID: {p.pid}) konnte nicht ordnungsgemäß beendet werden. Terminiere...")
            p.terminate()
    logger.info("Alle Prozesse beendet.")

    if success_event.is_set():
        try:
            return result_path_q.get_nowait()
        except mp.queues.Empty:
            logger.error("Erfolgs-Event wurde gesetzt, aber kein Ergebnis-Pfad gefunden.")
            return None
    else:
        return None

# --- HAUPTEINSTIEGSPUNKT ---
if __name__ == '__main__':
    config = configparser.ConfigParser()
    config_path = Path(__file__).parent / 'config.ini'
    if not config_path.exists():
        logger.error(f"Konfigurationsdatei nicht gefunden: {config_path}")
        exit()
    config.read(config_path)

    try:
        # --- EINMALIGE INITIALISIERUNG ---
        
        # 1. KI-Modell laden
        model_paths = get_model_paths_from_config(str(config_path))
        engine_name = config.get('Backend', 'engine')
        device = config.get('Backend', 'inference_device', fallback='cpu')
        engine_config = {
            'engine_name': engine_name, 'model_paths': model_paths, 'device': device
        }
        inference_engine_instance, class_names = get_inference_engine(engine_config)
        logger.info(f"Modell erfolgreich geladen. Gefundene Klassen: {len(class_names)}")
        if not class_names:
            raise ValueError("Das geladene Modell enthält keine Klassennamen.")

        # 2. Hue Controller initialisieren
        hue_config = dict(config.items('PhilipsHue'))
        hue_controller = HueController(hue_config.get('bridge_ip'), hue_config.get('app_key'))
        
        # 3. Notifier initialisieren
        notification_config = dict(config.items('Notifications'))
        notifier = Notifier(notification_config)

        # 4. Zeitmanagement-Konfiguration lesen
        time_config = dict(config.items('TimeManagement'))
        start_time_str = time_config.get('start_time', '00:00')
        end_time_str = time_config.get('end_time', '23:59')
        data_management_time_str = time_config.get('data_management_time')
        
        camera_src = config.get('Input', 'camera_src', fallback='0')
        data_management_done_today = False

        # --- DIE AUTONOME HAUPTSCHLEIFE ---
        while True:
            now = datetime.datetime.now()
            
            # Reset des "Datenmanagement erledigt"-Flags um Mitternacht
            if now.time().hour == 0 and now.time().minute == 0 and data_management_done_today:
                logger.info("Neuer Tag. Setze Datenmanagement-Flag zurück.")
                data_management_done_today = False

            # --- DATENMANAGEMENT ---
            if data_management_time_str:
                data_management_time = datetime.datetime.strptime(data_management_time_str, '%H:%M').time()
                if now.time() >= data_management_time and not data_management_done_today:
                    logger.info(f"Es ist {now.time().strftime('%H:%M')}, Zeit für das Datenmanagement.")
                    run_data_management(config)
                    data_management_done_today = True

            # Prüfen, ob wir uns im aktiven Zeitfenster befinden
            if is_within_schedule(start_time_str, end_time_str):
                
                # --- PHASE 1: PASSIVE ANALYSE ---
                logger.info("Innerhalb des Zeitplans. Starte passive Analyse und warte auf Helligkeits-Trigger...")
                
                passive_cam_config = dict(config.items('PassivAnalyzerCamera'))
                passive_trigger_config = dict(config.items('PassivAnalyzerTrigger'))
                
                trigger_detected = run_passive_analysis(
                    camera_src=camera_src,
                    camera_config=passive_cam_config,
                    trigger_config=passive_trigger_config
                )

                if not trigger_detected:
                    logger.info("Passive Analyse wurde ohne Trigger beendet. Beende Hauptprogramm.")
                    break 

                # --- PHASE 2: AKTIVE ANALYSE ---
                if hue_controller.is_active:
                    hue_controller.set_lights_on(
                        light_ids=hue_config.get('light_ids', '').split(','),
                        brightness=int(hue_config.get('brightness', 254)),
                        saturation=int(hue_config.get('saturation', 0)),
                        hue=int(hue_config.get('hue', 14910))
                    )

                logger.info("Trigger erkannt! Starte aktive Analyse...")
                event_path = run_active_analysis(config, inference_engine_instance, class_names)

                # --- PHASE 3: FARBANALYSE & AKTIONEN ---
                if event_path:
                    logger.info("Aktive Analyse erfolgreich. Starte Farbanalyse.")
                    try:
                        method = config.get('ColorAnalysis', 'analysis_method', fallback='hsv').lower()
                        is_cat_black = False

                        if method == 'hsv':
                            hsv_config = {
                                'lower_hsv': [int(v) for v in config.get('ColorAnalysis', 'lower_black_hsv').split(',')],
                                'upper_hsv': [int(v) for v in config.get('ColorAnalysis', 'upper_black_hsv').split(',')],
                                'pixel_threshold': config.getfloat('ColorAnalysis', 'black_pixel_threshold')
                            }
                            is_cat_black = analyze_cat_color(event_path, hsv_config)
                        elif method == 'llm':
                            llm_config = {
                                'host': config.get('ColorAnalysis', 'ollama_host', fallback='http://localhost:11434'),
                                'model': config.get('ColorAnalysis', 'ollama_model'),
                                'prompt': config.get('ColorAnalysis', 'ollama_prompt')
                            }
                            is_cat_black = analyze_color_with_llm(event_path, llm_config)
                        
                        if is_cat_black:
                            logger.info("VOLLER ERFOLG! Zielobjekt (schwarze Katze) gefunden.")
                            if hue_controller.is_active:
                                hue_controller.set_lights_off(hue_config.get('light_ids', '').split(','))
                            logger.info("Kehre nach Erfolg zur passiven Analyse zurück...")
                            time.sleep(10)
                            continue
                        else:
                            logger.info("INTRUDER! Nicht-schwarze Katze erkannt. Starte Gegenmaßnahmen.")
                            intruder_image_path = Path(event_path) / 'frame.jpg'
                            notifier.send(
                                message=f"Intruder (nicht-schwarze Katze) um {datetime.datetime.now().strftime('%H:%M:%S')} erkannt!",
                                image_path=str(intruder_image_path)
                            )
                            actions_config = dict(config.items('Actions'))
                            sound_config = {'folder': actions_config.get('sound_folder', 'cat_scare_sound')}
                            
                            handle_intruder_event(
                                duration_minutes=float(actions_config.get('intruder_light_minutes', 1)),
                                camera_src=camera_src,
                                camera_config=dict(config.items('Active')),
                                video_save_folder=actions_config.get('intruder_video_folder', 'intruder_recordings'),
                                sound_config=sound_config
                            )
                            
                            if hue_controller.is_active:
                                hue_controller.set_lights_off(hue_config.get('light_ids', '').split(','))
                            logger.info("Intruder-Protokoll beendet. Kehre zur passiven Analyse zurück.")
                            time.sleep(5)
                            continue
                    except Exception as e:
                        logger.error(f"Fehler bei der Farbanalyse: {e}")
                else:
                    logger.info("Aktive Analyse nicht erfolgreich (Timeout).")
                    if hue_controller.is_active:
                        hue_controller.set_lights_off(hue_config.get('light_ids', '').split(','))
                    logger.info("Kehre zur passiven Analyse zurück.")
                    time.sleep(5)
                    continue
            else:
                logger.info(f"Außerhalb des Zeitplans ({start_time_str} - {end_time_str}). Gehe in den Ruhezustand.")
                time.sleep(300)

    except KeyboardInterrupt:
        logger.info("Hauptprogramm durch Benutzer (Strg+C) beendet.")
    except Exception as e:
        logger.error(f"Ein unerwarteter Fehler im Hauptprozess ist aufgetreten: {traceback.format_exc()}")
    finally:
        if 'hue_controller' in locals() and hue_controller.is_active:
            logger.info("Schalte Lichter beim Herunterfahren aus...")
            hue_controller.set_lights_off(hue_config.get('light_ids', '').split(','))
        logger.info("Anwendung wird heruntergefahren.")