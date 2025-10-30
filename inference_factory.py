# inference_factory.py

import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
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

# --- Platzhalter für Modell-Architektur (für Safetensors) ---
# WICHTIG: Diese Zeile muss für die Verwendung von .safetensors-Dateien angepasst werden.
try:
    # from my_yolo_model_file import MyYoloModelClass
    MyYoloModelClass = None
except ImportError:
    MyYoloModelClass = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

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
    def __init__(self, model_path: Path, device: str):
        logger.info(f"Lade Safetensors-Modell von: {model_path}")
        if torch is None or load_safetensors is None:
            raise ImportError("torch und safetensors müssen installiert sein.")
        if MyYoloModelClass is None:
            raise NotImplementedError("Keine Modell-Klasse importiert. Bitte passen Sie 'inference_factory.py' an und definieren Sie 'MyYoloModelClass'.")

        self.device = torch.device(device)
        self.model = MyYoloModelClass()
        load_safetensors(str(model_path), self.model)
        self.model.to(self.device).eval()
        
        # Extrahiere Klassennamen
        self.class_names = getattr(self.model, 'names', [])
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
        engine_instance = SafetensorsEngine(path, device)

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
