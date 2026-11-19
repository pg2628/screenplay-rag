"""Local Ollama answer provider."""

import json
import os
from typing import Optional
from urllib.request import Request, urlopen

from .prompts import Prompt


class OllamaProvider:
    """Send prompts to a local Ollama chat model."""

    def __init__(self, host: str | None = None, model: str | None = None):
        self.host = (
            host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        ).rstrip("/")
        self.model = model or os.environ.get("OLLAMA_MODEL")
        if not self.model:
            raise RuntimeError("OLLAMA_MODEL must name an installed Ollama model")
        self.timeout = float(os.environ.get("OLLAMA_TIMEOUT", "600"))

    def __call__(self, prompt: Prompt) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ],
            "stream": False,
        }
        request = Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                response_data = json.loads(response.read().decode("utf-8"))
        except TimeoutError as error:
            raise RuntimeError(
                f"Ollama chat request timed out after {self.timeout:g} seconds"
            ) from error
        except OSError as error:
            raise RuntimeError("Could not connect to the Ollama chat API") from error
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise RuntimeError("Ollama returned invalid JSON") from error

        try:
            answer = response_data["message"]["content"]
        except (KeyError, TypeError) as error:
            raise RuntimeError(
                "Ollama response did not contain message content"
            ) from error

        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError("Ollama response contained an empty answer")
        return answer
