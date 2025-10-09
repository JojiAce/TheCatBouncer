import configparser
import logging
from pathlib import Path
from typing import List


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Mapping from engine names to supported file extensions
MODEL_FILETYPES = {
    'onnx':        ['onnx'],
    'onnxslim':    ['onnx'],  # onnxslim uses .onnx
    'openvino':    ['xml', 'bin'],
    'pt':          ['pt'],
    'safetensor':  ['safetensors'],
    'coreml':      ['mlmodel']
}

# Supported model size codes
ALLOWED_SIZES = ['n', 's', 'm', 'l', 'x']


def get_model_paths(
    engine: str,
    model_size: str,
    model_generation: str,
    inference_model_main_folder: str,
    folder_layout: str = 'nested'
) -> List[Path]:
    """
    Sucht und liefert alle Modell-Dateipfade basierend auf der Konfiguration.

    Ordnerstruktur (nested):
      inference_model_main_folder/
        └── <model_generation>/
            └── <model_generation>_<model_size>/
                └── <model_generation>_<model_size>_<engine>/
                    └── Modell-Dateien

    Args:
        engine: Name der Inferenz-Engine (z.B. 'onnx', 'onnxslim', 'openvino', 'pt', 'safetensor', 'coreml').
        model_size: Model-Größe (einer der 'n', 's', 'm', 'l', 'x').
        model_generation: Modell-Generation (z.B. 'YOLO11').
        inference_model_main_folder: Pfad zum Hauptordner, der alle Generationen enthält.
        folder_layout: 'nested' oder 'flat'. Bestimmt die Suchebene.

    Returns:
        Eine Liste mit Pfaden aller gefundenen Modell-Dateien.

    Raises:
        ValueError: Wenn engine oder model_size nicht unterstützt werden.
        FileNotFoundError: Wenn die erwarteten Ordner/Dateien nicht gefunden werden.
    """
    # Validate engine
    if engine not in MODEL_FILETYPES:
        logger.error(f"Unbekannte Engine '{engine}'. Verfügbare: {list(MODEL_FILETYPES.keys())}")
        raise ValueError(f"Engine '{engine}' not supported.")

    # Validate model_size
    if model_size not in ALLOWED_SIZES:
        logger.error(f"Unbekannte model_size '{model_size}'. Verfügbare: {ALLOWED_SIZES}")
        raise ValueError(f"Model size '{model_size}' not supported.")

    # Determine extensions for this engine
    extensions = MODEL_FILETYPES[engine]

    base_path = Path(inference_model_main_folder)
    if not base_path.exists():
        logger.error(f"Hauptordner nicht gefunden: {base_path}")
        raise FileNotFoundError(f"Inference model folder '{base_path}' does not exist.")

    model_paths: List[Path] = []

    search_base: Path
    if folder_layout == 'nested':
        # Verschachtelte Struktur mit Hauptordner, Generation, Size und Engine
        gen_dir = base_path / model_generation
        size_dir = gen_dir / f"{model_generation}_{model_size}"
        engine_dir = size_dir / f"{model_generation}_{model_size}_{engine}"

        logger.debug(f"Suche im verschachtelten Pfad: {engine_dir}")
        if not engine_dir.exists():
            logger.error(f"Erwarteter Unterordner fehlt: {engine_dir}")
            raise FileNotFoundError(
                f"Expected folder structure not found at '{engine_dir}'. "
                "Check 'model_generation', 'model_size' and 'engine'."
            )

        search_base = engine_dir
        # Suche alle Dateien mit den passenden Endungen
        for ext in extensions:
            found = list(engine_dir.glob(f"*.{ext}"))
            logger.info(f"{len(found)} Dateien mit Endung '.{ext}' gefunden in {engine_dir}")
            model_paths.extend(found)

    elif folder_layout == 'flat':
        # Flache Struktur: alle Modelle direkt im Hauptordner
        logger.debug(f"Suche flach im Hauptordner: {base_path}")
        search_base = base_path
        for ext in extensions:
            found = list(base_path.glob(f"*.{ext}"))
            logger.info(f"{len(found)} Dateien mit Endung '.{ext}' gefunden in {base_path}")
            model_paths.extend(found)

    else:
        logger.error(f"Unbekanntes folder_layout: {folder_layout}")
        raise ValueError(f"Unknown folder_layout '{folder_layout}'. Use 'nested' or 'flat'.")

    if not model_paths:
        logger.error("Keine Modelldateien gefunden.")
        raise FileNotFoundError(
            f"No model files with extensions {extensions} found under '{search_base}'."
        )

    # Sortierung und Rückgabe
    model_paths = sorted(model_paths)
    logger.info(f"Insgesamt {len(model_paths)} Modell-Dateien gefunden.")
    return model_paths


def get_model_paths_from_config(config_path: str = 'config.ini') -> List[Path]:
    """
    Liest die Konfiguration und ruft get_model_paths auf.

    Args:
        config_path: Pfad zur config.ini.

    Returns:
        List[Path]: Gefundene Modell-Dateien.
    """
    # Konfigurations Datei (INI) laden
    config = configparser.ConfigParser()
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Konfigurationsdatei nicht gefunden: {config_path}")

    config.read(config_path)


    try:
        engine = config['Backend']['engine']
        model_size = config['Backend']['model_size']
        model_generation = config['Backend']['model_generation']
        inference_model_main_folder = config['Backend']['inference_model_main_folder']
        folder_layout = config['Backend'].get('model_folder_layout', 'nested')
    except KeyError as e:
        logger.error(f"Fehlender Konfigurationsschlüssel: {e}")
        raise

    return get_model_paths(
        engine=engine,
        model_size=model_size,
        model_generation=model_generation,
        inference_model_main_folder=inference_model_main_folder,
        folder_layout=folder_layout
    )


# -----------------------------------------------------------------------------
# "model_files = get_model_paths_from_config()" liefert die modellpfade für die
# aktive Pipeline.

if __name__ == '__main__':
    try:
        paths = get_model_paths_from_config()
        for p in paths:
            print(p)
    except Exception:
        logger.exception("Fehler beim Abrufen der Model-Pfade.")


