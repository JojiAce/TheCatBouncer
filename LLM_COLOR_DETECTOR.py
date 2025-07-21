import ollama
import base64
from pathlib import Path
import logging

# Logging so konfigurieren, dass es mit dem Hauptskript übereinstimmt
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

def encode_image_to_base64(filepath: Path) -> str:
    """Liest ein Bild und kodiert es als Base64-String."""
    try:
        with open(filepath, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Fehler beim Kodieren des Bildes {filepath}: {e}")
        return ""

def analyze_color_with_llm(event_folder_path: str, llm_config: dict) -> bool:
    """
    Verwendet ein lokales LLM (via Ollama), um die Farbe der Katze im Bild zu bestimmen.

    Args:
        event_folder_path (str): Der Pfad zum Ordner, der das Bild enthält.
        llm_config (dict): Ein Dictionary mit den Ollama-Parametern (host, model, prompt).

    Returns:
        bool: True, wenn das LLM mit "yes" antwortet, sonst False.
    """
    logger.info(f"Starte LLM-Farbanalyse für Ereignis-Ordner: {event_folder_path}")
    event_folder = Path(event_folder_path)
    image_path = event_folder / 'frame.jpg'

    if not image_path.exists():
        logger.error(f"Fehler: Bilddatei nicht im Ordner {event_folder} gefunden.")
        return False

    base64_image = encode_image_to_base64(image_path)
    if not base64_image:
        return False

    # Konfiguration aus dem Dictionary auslesen
    host_url = llm_config.get('host', 'http://localhost:11434')
    model_name = llm_config.get('model', 'llava')
    prompt_text = llm_config.get('prompt', 'Is the cat in this image black? Answer with only "yes" or "no".')

    try:
        # AKTUALISIERT: Expliziten Ollama-Client erstellen
        logger.info(f"Verbinde mit Ollama-Host: {host_url}")
        client = ollama.Client(host=host_url)

        logger.info(f"Sende Anfrage an Ollama mit Modell '{model_name}'...")
        response = client.chat(
            model=model_name,
            messages=[
                {
                    'role': 'user',
                    'content': prompt_text,
                    'images': [base64_image]
                }
            ]
        )
        
        answer = response['message']['content'].strip().lower()
        logger.info(f"Antwort vom LLM erhalten: '{answer}'")

        if 'yes' in answer:
            logger.info("LLM hat die Farbe als schwarz bestätigt.")
            return True
        else:
            logger.info("LLM hat die Farbe nicht als schwarz bestätigt.")
            return False

    except Exception as e:
        logger.error(f"Fehler bei der Kommunikation mit dem Ollama-Server unter {host_url}: {e}")
        logger.error(f"Stellen Sie sicher, dass Ollama läuft und das Modell '{model_name}' heruntergeladen ist (z.B. mit 'ollama run {model_name}').")
        return False