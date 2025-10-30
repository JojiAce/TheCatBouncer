# inference_factory.py

import logging
from importlib import import_module
from functools import partial
from pathlib import Path
from typing import Callable, Iterable, List, Dict, Any, Tuple
import numpy as np
import cv2
import json

# --- Voraussetzungen ---
# Versuche, die notwendigen Bibliotheken zu importieren.
try:
    import onnxruntime
except ImportError:
    onnxruntime = None
try:
    from openvino.runtime import Core as OpenVinoCore
except ImportError:
    OpenVinoCore = None
try:
    import torch
except ImportError:
    torch = None
try:
    from safetensors.torch import load_file as load_safetensors
except ImportError:
    load_safetensors = None
try:
    import coremltools as ct
    from PIL import Image
except ImportError:
    ct = None
    Image = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)


def _import_from_string(path: str) -> Callable:
    """Lädt ein Objekt basierend auf einem Import-String."""
    if not isinstance(path, str) or not path.strip():
        raise ValueError("Ein gültiger Import-Pfad als String wird benötigt.")

    normalized = path.replace(":", ".").strip()
    module_path, _, attr = normalized.rpartition(".")
    if not module_path:
        raise ValueError(f"Der Import-Pfad '{path}' muss ein Modul und ein Attribut enthalten.")

    try:
        module = import_module(module_path)
    except ImportError as exc:
        raise ImportError(f"Kann Modul '{module_path}' nicht importieren (für '{path}').") from exc

    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise ImportError(f"Attribut '{attr}' wurde in Modul '{module_path}' nicht gefunden.") from exc


def _ensure_callable(candidate: Any, description: str) -> Callable:
    if not callable(candidate):
        raise TypeError(f"{description} muss aufrufbar sein, erhalten: {type(candidate)!r}.")
    return candidate


def _parse_optional_iterable(value: Any) -> Iterable:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return value
    if isinstance(value, dict):
        return value.values()
    if isinstance(value, str):
        try:
            data = json.loads(value)
        except json.JSONDecodeError:
            logger.warning("Konnte Klassennamen nicht als JSON interpretieren. Verwende Roh-String.")
            return [value]
        return data if isinstance(data, (list, tuple)) else [value]
    return [value]


def _create_safetensors_builder(config: Dict[str, Any]) -> Callable[[], Any]:
    """Erzeugt einen Builder für Safetensors-Modelle basierend auf der Konfiguration."""
    builder = config.get('safetensors_model_builder') or config.get('model_builder')
    model_class = config.get('safetensors_model_class') or config.get('model_class')

    if builder:
        if isinstance(builder, str):
            builder_callable = _import_from_string(builder)
        else:
            builder_callable = builder
        builder_callable = _ensure_callable(builder_callable, 'safetensors_model_builder')
        return builder_callable

    if model_class:
        if isinstance(model_class, str):
            model_class_obj = _import_from_string(model_class)
        else:
            model_class_obj = model_class
        model_class_obj = _ensure_callable(model_class_obj, 'safetensors_model_class')

        args = config.get('safetensors_model_args') or config.get('model_args') or ()
        kwargs = config.get('safetensors_model_kwargs') or config.get('model_kwargs') or {}

        if isinstance(args, str):
            try:
                args = tuple(json.loads(args))
            except json.JSONDecodeError as exc:
                raise ValueError("safetensors_model_args muss eine Liste oder JSON-Liste sein.") from exc
        if isinstance(kwargs, str):
            try:
                kwargs = json.loads(kwargs)
            except json.JSONDecodeError as exc:
                raise ValueError("safetensors_model_kwargs muss ein Dict oder JSON-Dict sein.") from exc

        if not isinstance(args, (list, tuple)):
            raise TypeError("safetensors_model_args muss eine Liste oder ein Tuple sein.")
        if not isinstance(kwargs, dict):
            raise TypeError("safetensors_model_kwargs muss ein Dict sein.")

        return partial(model_class_obj, *args, **kwargs)

    raise ValueError(
        "Für Safetensors-Modelle muss entweder 'safetensors_model_builder' oder 'safetensors_model_class'"
        " (optional mit Args/Kwargs) in der Konfiguration angegeben werden."
    )

# =============================================================================
#  1. WRAPPER-KLASSEN (Die "Universal-Adapter")
# =============================================================================

class OnnxEngine:
    """Wrapper für ONNX-Runtime-Modelle."""
    def __init__(self, model_path: Path, device: str):
        logger.info(f"Lade ONNX-Modell von: {model_path}")
        if onnxruntime is None:
            raise ImportError("onnxruntime nicht installiert. Bitte installieren mit 'pip install onnxruntime' oder 'onnxruntime-gpu'.")
        
        providers = {'gpu': 'CUDAExecutionProvider', 'cpu': 'CPUExecutionProvider'}
        if device.lower() not in providers:
            raise ValueError(f"Ungültiges Gerät für ONNX: {device}. Wähle 'cpu' oder 'gpu'.")
        provider = providers[device.lower()]

        self.session = onnxruntime.InferenceSession(str(model_path), providers=[provider])
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        # Extrahiere Klassennamen aus den Metadaten des Modells
        self.class_names = []
        meta = self.session.get_modelmeta()
        if 'names' in meta.custom_metadata_map:
            try:
                names_dict = json.loads(meta.custom_metadata_map['names'])
                self.class_names = list(names_dict.values())
            except Exception as e:
                logger.warning(f"Konnte Klassennamen aus ONNX-Metadaten nicht parsen: {e}")
        
        if not self.class_names:
            logger.warning("Keine Klassennamen im ONNX-Modell gefunden. Vergleiche basieren auf IDs.")
        
        logger.info("ONNX-Modell erfolgreich geladen.")

    def predict(self, image: np.ndarray) -> List[np.ndarray]:
        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)
        return self.session.run(self.output_names, {self.input_name: image})


class OpenVinoEngine:
    """Wrapper für OpenVINO-Modelle."""
    def __init__(self, model_paths: List[Path], device: str):
        xml_path = next((p for p in model_paths if p.suffix == '.xml'), None)
        if not xml_path:
            raise FileNotFoundError("Keine .xml-Datei für OpenVINO gefunden.")
        
        bin_path = xml_path.with_suffix('.bin')
        if not bin_path.exists():
            raise FileNotFoundError(f"Die .bin-Datei '{bin_path.name}', die zu '{xml_path.name}' gehört, wurde nicht gefunden.")

        logger.info(f"Lade OpenVINO-Modell von: {xml_path}")
        if OpenVinoCore is None:
            raise ImportError("openvino ist nicht installiert. Bitte installieren mit 'pip install openvino-dev'.")

        core = OpenVinoCore()
        model = core.read_model(model=xml_path)
        
        # Extrahiere Klassennamen, falls in Metadaten vorhanden
        self.class_names = []
        if 'names' in model.get_rt_info():
            names = model.get_rt_info()["names"]
            if isinstance(names, dict):
                self.class_names = list(names.values())
            elif isinstance(names, (list, tuple)):
                self.class_names = list(names)

        self.compiled_model = core.compile_model(model=model, device_name=device.upper())
        self.input_layer = self.compiled_model.input(0)
        logger.info("OpenVINO-Modell erfolgreich geladen.")

    def predict(self, image: np.ndarray) -> List[np.ndarray]:
        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)
        results = self.compiled_model([image])
        return [results[next(iter(results))]]


class PyTorchEngine:
    """Wrapper für PyTorch-Modelle (.pt)."""
    def __init__(self, model_path: Path, device: str):
        logger.info(f"Lade PyTorch-Modell von: {model_path}")
        if torch is None:
            raise ImportError("torch ist nicht installiert. Bitte installieren mit 'pip install torch'.")
        
        self.device = torch.device(device)
        self.model = torch.load(model_path, map_location=self.device)
        self.model.to(self.device).eval()

        # Extrahiere Klassennamen (typisch für Ultralytics-Modelle)
        self.class_names = getattr(self.model, 'names', [])
        if not self.class_names:
            logger.warning("Keine 'names'-Attribut im PyTorch-Modell gefunden.")

        logger.info("PyTorch-Modell erfolgreich geladen.")

    def predict(self, image: np.ndarray) -> Any:
        tensor = torch.from_numpy(image).permute(2, 0, 1).float().to(self.device)
        tensor /= 255.0
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        with torch.no_grad():
            return self.model(tensor)


class SafetensorsEngine:
    """Wrapper für Safetensors-Modelle (.safetensors)."""

    def __init__(self, model_path: Path, device: str, *, model_builder: Callable[[], Any], names: Iterable[str] | None = None):
        logger.info(f"Lade Safetensors-Modell von: {model_path}")
        if torch is None or load_safetensors is None:
            raise ImportError("torch und safetensors müssen installiert sein.")
        if model_builder is None:
            raise ValueError("Für Safetensors-Modelle muss ein 'model_builder' angegeben werden.")

        self.device = torch.device(device)
        model = model_builder()
        if not hasattr(model, 'load_state_dict'):
            raise TypeError("Der model_builder muss eine torch.nn.Module Instanz zurückgeben.")

        load_safetensors(str(model_path), model)
        self.model = model.to(self.device).eval()

        inferred_names = getattr(self.model, 'names', [])
        provided_names = list(names) if names else []
        self.class_names = provided_names or inferred_names
        logger.info("Safetensors-Modell erfolgreich geladen.")

    def predict(self, image: np.ndarray) -> Any:
        tensor = torch.from_numpy(image).permute(2, 0, 1).float().to(self.device)
        tensor /= 255.0
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        with torch.no_grad():
            return self.model(tensor)


class CoreMLEngine:
    """Wrapper für CoreML-Modelle (.mlmodel)."""
    def __init__(self, model_path: Path, device: str):
        logger.info(f"Lade CoreML-Modell von: {model_path}")
        if ct is None or Image is None:
            raise ImportError("coremltools und Pillow müssen installiert sein.")

        compute_units = ct.models.compute_units.CPU_ONLY if device.lower() == 'cpu' else ct.models.compute_units.ALL
        self.model = ct.models.MLModel(str(model_path), compute_units=compute_units)
        self.input_name = self.model.get_spec().description.input[0].name
        
        # Extrahiere Klassennamen aus den Metadaten
        self.class_names = []
        try:
            # Annahme: Der Output, der die Klassen enthält, hat einen 'stringVector'
            output_description = self.model.get_spec().description.output[0]
            if output_description.type.HasField('stringVectorType'):
                self.class_names = list(output_description.type.stringVectorType.vector)
        except Exception as e:
            logger.warning(f"Konnte Klassennamen aus CoreML-Modell nicht extrahieren: {e}")

        logger.info(f"CoreML-Modell erfolgreich geladen. Input-Name: '{self.input_name}'.")

    def predict(self, image: np.ndarray) -> Dict:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        return self.model.predict({self.input_name: pil_image})


# =============================================================================
#  2. DIE FACTORY-FUNKTION
# =============================================================================

def get_inference_engine(config: Dict[str, Any]) -> Tuple[Any, List[str]]:
    """
    Erstellt eine Engine-Instanz und gibt sie ZUSAMMEN mit den Klassennamen zurück.
    """
    engine_name = config.get('engine_name')
    model_paths = config.get('model_paths')
    device = config.get('device')

    if not model_paths:
        raise ValueError("Keine Modell-Pfade im Konfigurations-Dictionary gefunden.")
    
    logger.info(f"Initialisiere Engine '{engine_name}' auf Gerät '{device}'.")

    engine_instance = None
    if engine_name in ('onnx', 'onnxslim'):
        path = next((p for p in model_paths if p.suffix == '.onnx'), None)
        if not path: raise FileNotFoundError("Keine .onnx Datei gefunden.")
        engine_instance = OnnxEngine(path, device)

    elif engine_name == 'openvino':
        engine_instance = OpenVinoEngine(model_paths, device)
        
    elif engine_name == 'pt':
        path = next((p for p in model_paths if p.suffix == '.pt'), None)
        if not path: raise FileNotFoundError("Keine .pt Datei gefunden.")
        engine_instance = PyTorchEngine(path, device)
    
    elif engine_name == 'safetensor':
        path = next((p for p in model_paths if p.suffix == '.safetensors'), None)
        if not path: raise FileNotFoundError("Keine .safetensors Datei gefunden.")

        builder = _create_safetensors_builder(config)
        names = list(_parse_optional_iterable(config.get('class_names')))

        class_names_path = config.get('class_names_path')
        if class_names_path and not names:
            try:
                with open(class_names_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    names = list(data.values())
                elif isinstance(data, list):
                    names = list(data)
                else:
                    logger.warning("class_names_path enthält ein unbekanntes Format. Ignoriere Datei.")
            except FileNotFoundError:
                logger.error(f"Klassennamen-Datei '{class_names_path}' wurde nicht gefunden.")
            except json.JSONDecodeError as exc:
                logger.error(f"Konnte Klassennamen aus '{class_names_path}' nicht parsen: {exc}")

        engine_instance = SafetensorsEngine(path, device, model_builder=builder, names=names)

    elif engine_name == 'coreml':
        path = next((p for p in model_paths if p.suffix == '.mlmodel'), None)
        if not path: raise FileNotFoundError("Keine .mlmodel Datei gefunden.")
        engine_instance = CoreMLEngine(path, device)

    else:
        raise ValueError(f"Unbekannte oder nicht unterstützte Engine in der Konfiguration: '{engine_name}'")

    if not engine_instance:
        raise RuntimeError(f"Engine '{engine_name}' konnte nicht initialisiert werden.")

    # Gib die Engine-Instanz und die Klassennamen zurück
    return engine_instance, getattr(engine_instance, 'class_names', [])
