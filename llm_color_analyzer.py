"""Utilities for validating cat colour via a local LLM service.

This module replaces the previous ad-hoc Ollama integration with a
feature-complete, well tested implementation that works with the official
Python client *and* plain HTTP fallbacks.  The goal is to make the colour check
reliable even on constrained edge devices where optional dependencies might not
be available.
"""

from __future__ import annotations

import base64
import importlib
import json
import logging
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

LOGGER = logging.getLogger(__name__)


def _ensure_logging_configured() -> None:
    """Attach a default handler if the host application has not done so."""

    if not LOGGER.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )


_ensure_logging_configured()


class LLMCommunicationError(RuntimeError):
    """Raised when the LLM backend cannot be reached or returns bad data."""


@dataclass(frozen=True)
class OllamaSettings:
    """Configuration container for the Ollama based local LLM service."""

    host: str = "http://localhost:11434"
    model: str = "llava"
    prompt: str = 'Is the cat in this image black? Answer only with "yes" or "no".'
    temperature: float = 0.0
    timeout: float = 30.0

    @classmethod
    def from_mapping(cls, data: Optional[Mapping[str, Any]]) -> "OllamaSettings":
        """Create settings from a mapping, applying defaults and validation."""

        raw = dict(data or {})
        host = str(
            raw.pop("host", None)
            or raw.pop("ollama_host", None)
            or cls.host
        ).strip()
        model = str(
            raw.pop("model", None)
            or raw.pop("ollama_model", None)
            or cls.model
        ).strip()
        prompt = str(
            raw.pop("prompt", None)
            or raw.pop("ollama_prompt", None)
            or cls.prompt
        )

        temperature = raw.pop("temperature", raw.pop("ollama_temperature", cls.temperature))
        timeout = raw.pop("timeout", raw.pop("ollama_timeout", cls.timeout))

        try:
            temperature_value = float(temperature)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive guard
            raise ValueError("Temperature must be numeric.") from exc

        try:
            timeout_value = float(timeout)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive guard
            raise ValueError("Timeout must be numeric.") from exc

        if not host:
            raise ValueError("A valid Ollama host must be provided.")
        if not model:
            raise ValueError("A valid Ollama model name must be provided.")

        parsed_host = urllib.parse.urlparse(host if host.startswith("http") else f"http://{host}")
        if not parsed_host.scheme or not parsed_host.netloc:
            raise ValueError(f"Invalid Ollama host URL: {host}")

        return cls(
            host=f"{parsed_host.scheme}://{parsed_host.netloc}".rstrip("/"),
            model=model,
            prompt=prompt,
            temperature=temperature_value,
            timeout=max(1.0, timeout_value),
        )

    def build_payload(self, encoded_image: str) -> Mapping[str, Any]:
        """Generate the chat payload understood by Ollama."""

        return {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": self.prompt,
                    "images": [encoded_image],
                }
            ],
            "options": {
                "temperature": self.temperature,
            },
            "stream": False,
        }


class OllamaColorAnalyzer:
    """Evaluate an event frame with a locally hosted LLM."""

    def __init__(self, settings: OllamaSettings):
        self._settings = settings
        self._ollama_module = self._load_optional_client()

    @staticmethod
    def _load_optional_client():
        """Return the optional Ollama python module if available."""

        module_spec = importlib.util.find_spec("ollama")
        if module_spec is None:
            LOGGER.debug("Python ollama client not installed; will use HTTP fallback.")
            return None
        return importlib.import_module("ollama")

    @staticmethod
    def _encode_image(path: Path) -> str:
        if not path.exists():
            raise FileNotFoundError(f"Image for LLM analysis not found: {path}")
        return base64.b64encode(path.read_bytes()).decode("utf-8")

    def _chat_via_client(self, payload: Mapping[str, Any]) -> Optional[str]:
        if self._ollama_module is None:
            return None

        try:
            client = self._ollama_module.Client(host=self._settings.host)
            response = client.chat(**payload)
            return self._extract_message(response)
        except Exception as exc:  # pragma: no cover - depends on external lib
            LOGGER.warning(
                "Ollama python client failed (%s). Falling back to HTTP interface.",
                exc,
            )
            return None

    def _chat_via_http(self, payload: Mapping[str, Any]) -> str:
        url = f"{self._settings.host}/api/chat"
        encoded_payload = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=encoded_payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self._settings.timeout) as response:
                if response.status != 200:
                    raise LLMCommunicationError(
                        f"Unexpected HTTP status {response.status} from Ollama backend."
                    )
                data = response.read().decode("utf-8")
        except urllib.error.URLError as exc:  # pragma: no cover - network failure
            raise LLMCommunicationError(str(exc)) from exc

        try:
            parsed = json.loads(data)
        except json.JSONDecodeError as exc:  # pragma: no cover - invalid server data
            raise LLMCommunicationError("Received invalid JSON from Ollama backend.") from exc

        return self._extract_message(parsed)

    @staticmethod
    def _extract_message(response: Mapping[str, Any]) -> str:
        message = response.get("message")
        if not isinstance(message, Mapping):
            raise LLMCommunicationError("Ollama response missing 'message' block.")
        content = message.get("content")
        if not isinstance(content, str):
            raise LLMCommunicationError("Ollama response did not contain textual content.")
        return content

    @staticmethod
    def _interpret_answer(answer: str) -> Optional[bool]:
        text = answer.strip().lower()
        if not text:
            return None

        affirmative_tokens = {"yes", "yeah", "yep", "affirmative", "true"}
        negative_tokens = {"no", "nope", "nah", "false"}

        tokens = re.findall(r"[a-z']+", text)
        decision: Optional[bool] = None
        for token in tokens:
            if token in affirmative_tokens:
                decision = True
            elif token in negative_tokens:
                decision = False

        if decision is not None:
            return decision

        if "not black" in text or "isn't black" in text or "is not black" in text:
            return False
        if "is black" in text or "looks black" in text or "appears black" in text:
            return True
        return None

    def classify(self, event_folder: Path) -> Optional[bool]:
        """Return True/False if the cat is black, None when uncertain."""

        image_path = event_folder / "frame.jpg"
        encoded_image = self._encode_image(image_path)
        payload = self._settings.build_payload(encoded_image)

        answer = self._chat_via_client(payload)
        if answer is None:
            answer = self._chat_via_http(payload)

        LOGGER.info("Antwort vom LLM erhalten: '%s'", answer)
        return self._interpret_answer(answer)


def analyze_color_with_llm(event_folder_path: str | Path, llm_config: Mapping[str, Any]) -> bool:
    """Public convenience helper used by the main pipeline."""

    settings = OllamaSettings.from_mapping(llm_config)
    analyzer = OllamaColorAnalyzer(settings)

    event_folder = Path(event_folder_path)
    LOGGER.info("Starte LLM-Farbanalyse für Ereignis-Ordner: %s", event_folder)

    try:
        decision = analyzer.classify(event_folder)
    except FileNotFoundError as exc:
        LOGGER.error(str(exc))
        return False
    except LLMCommunicationError as exc:
        LOGGER.error("Fehler bei der Kommunikation mit dem Ollama-Server unter %s: %s", settings.host, exc)
        return False

    if decision is None:
        LOGGER.warning(
            "LLM-Antwort konnte nicht eindeutig interpretiert werden. Behandle Ergebnis als 'nicht schwarz'."
        )
        return False

    if decision:
        LOGGER.info("LLM hat die Farbe als schwarz bestätigt.")
    else:
        LOGGER.info("LLM hat die Farbe nicht als schwarz bestätigt.")
    return decision


__all__ = [
    "LLMCommunicationError",
    "OllamaColorAnalyzer",
    "OllamaSettings",
    "analyze_color_with_llm",
]
