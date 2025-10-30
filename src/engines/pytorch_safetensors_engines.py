"""PyTorch and SafeTensors inference engine implementations."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from src.engines.base_engine import InferenceEngine


class PyTorchEngine(InferenceEngine):
    """Inference engine that executes PyTorch models."""

    def __init__(self, model_path: str, device: str = "cpu", **kwargs: Any) -> None:
        super().__init__(model_path, device, **kwargs)

        try:
            import torch
        except ImportError as exc:  # pragma: no cover - guarded import
            raise ImportError(
                "PyTorch is not installed. Please install it with 'pip install torch'."
            ) from exc

        self._torch = torch

        if device.startswith("cuda") and torch.cuda.is_available():
            self.device = torch.device(device)
        else:
            self.device = torch.device("cpu")

        try:
            self.model = torch.load(model_path, map_location=self.device)
            self.model.eval()
            self.model.to(self.device)

            if hasattr(self.model, "names") and getattr(self.model, "names"):
                self.class_names = list(self.model.names)
            elif "names" in kwargs and isinstance(kwargs["names"], (list, tuple)):
                self.class_names = list(kwargs["names"])

            self.logger.info("PyTorch engine initialised for %s", model_path)
        except Exception as exc:  # pragma: no cover - defensive logging
            raise RuntimeError(
                f"Failed to initialise PyTorch engine with model {model_path}: {exc}"
            ) from exc

    def predict(self, image: np.ndarray) -> List[np.ndarray]:
        torch = self._torch

        with torch.no_grad():
            if image.ndim == 3:
                image = np.expand_dims(image, axis=0)

            if image.shape[-1] in (1, 3):
                image = np.transpose(image, (0, 3, 1, 2))

            tensor = torch.from_numpy(image).float().to(self.device)
            if tensor.max().item() > 1.0:
                tensor = tensor / 255.0

            results = self.model(tensor)

            if isinstance(results, torch.Tensor):
                return [results.detach().cpu().numpy()]
            if isinstance(results, (list, tuple)):
                output: List[np.ndarray] = []
                for value in results:
                    if isinstance(value, torch.Tensor):
                        output.append(value.detach().cpu().numpy())
                    else:
                        output.append(np.asarray(value))
                return output
            if isinstance(results, dict):
                return [
                    (value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value))
                    for value in results.values()
                ]

            return [np.asarray(results)]

    def warm_up(self) -> None:
        torch = self._torch
        dummy_input = torch.randn(1, 3, 640, 640, device=self.device)

        try:
            with torch.no_grad():
                _ = self.model(dummy_input)
            self.logger.info("PyTorch engine warm-up completed")
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.warning("PyTorch engine warm-up failed: %s", exc)


class SafeTensorsEngine(InferenceEngine):
    """Inference engine for models stored in the SafeTensors format."""

    def __init__(self, model_path: str, device: str = "cpu", **kwargs: Any) -> None:
        super().__init__(model_path, device, **kwargs)

        try:
            import torch
            from safetensors.torch import load_file
        except ImportError as exc:  # pragma: no cover - guarded import
            raise ImportError(
                "PyTorch and safetensors are required. Install them with 'pip install torch safetensors'."
            ) from exc

        self._torch = torch
        self._load_file = load_file

        if device.startswith("cuda") and torch.cuda.is_available():
            self.device = torch.device(device)
        else:
            self.device = torch.device("cpu")

        state_dict = self._load_file(model_path)
        model = self._instantiate_model(kwargs)

        try:
            model.load_state_dict(state_dict)
            model.eval()
            model.to(self.device)
        except Exception as exc:  # pragma: no cover - defensive logging
            raise RuntimeError(
                f"Failed to load SafeTensors weights from {model_path}: {exc}"
            ) from exc

        self.model = model

        names = kwargs.get("names")
        if isinstance(names, (list, tuple)):
            self.class_names = list(names)

        self.logger.info("SafeTensors engine initialised for %s", model_path)

    def _instantiate_model(self, kwargs: Dict[str, Any]):
        torch = self._torch

        if "model" in kwargs and kwargs["model"] is not None:
            model_candidate = kwargs["model"]
            if callable(model_candidate) and not isinstance(model_candidate, torch.nn.Module):
                model_candidate = model_candidate()
        else:
            builder: Optional[Any] = kwargs.get("model_builder") or kwargs.get("model_class")
            if builder is None:
                raise ValueError(
                    "SafeTensors engine requires 'model', 'model_builder' or 'model_class' in kwargs "
                    "to construct the network architecture."
                )

            if isinstance(builder, torch.nn.Module):
                model_candidate = builder
            elif isinstance(builder, type):
                model_candidate = builder()
            else:
                model_candidate = builder()

        if not isinstance(model_candidate, torch.nn.Module):
            raise TypeError(
                "SafeTensors model builder must return a torch.nn.Module instance."
            )

        return model_candidate

    def predict(self, image: np.ndarray) -> List[np.ndarray]:
        torch = self._torch

        with torch.no_grad():
            if image.ndim == 3:
                image = np.expand_dims(image, axis=0)

            if image.shape[-1] in (1, 3):
                image = np.transpose(image, (0, 3, 1, 2))

            tensor = torch.from_numpy(image).float().to(self.device)
            if tensor.max().item() > 1.0:
                tensor = tensor / 255.0

            results = self.model(tensor)

            if isinstance(results, torch.Tensor):
                return [results.detach().cpu().numpy()]
            if isinstance(results, (list, tuple)):
                output: List[np.ndarray] = []
                for value in results:
                    if isinstance(value, torch.Tensor):
                        output.append(value.detach().cpu().numpy())
                    else:
                        output.append(np.asarray(value))
                return output
            if isinstance(results, dict):
                return [
                    (value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value))
                    for value in results.values()
                ]

            return [np.asarray(results)]

    def warm_up(self) -> None:
        torch = self._torch
        dummy_input = torch.randn(1, 3, 640, 640, device=self.device)

        try:
            with torch.no_grad():
                _ = self.model(dummy_input)
            self.logger.info("SafeTensors engine warm-up completed")
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.warning("SafeTensors engine warm-up failed: %s", exc)

