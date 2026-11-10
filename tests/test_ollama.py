import json

import pytest

from screenplay_rag.ollama import OllamaProvider
from screenplay_rag.prompts import Prompt


class FakeResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return self.payload


def test_ollama_provider_sends_prompt_and_returns_answer(monkeypatch):
    requests = []

    def fake_urlopen(request, timeout):
        requests.append((request, timeout))
        return FakeResponse({"message": {"content": "A grounded answer."}})

    monkeypatch.setenv("OLLAMA_MODEL", "llama3.2")
    monkeypatch.setattr("screenplay_rag.ollama.urlopen", fake_urlopen)

    answer = OllamaProvider()(Prompt(system="System rules", user="User question"))

    request, timeout = requests[0]
    payload = json.loads(request.data.decode("utf-8"))
    assert answer == "A grounded answer."
    assert request.full_url == "http://localhost:11434/api/chat"
    assert timeout == 120
    assert payload["model"] == "llama3.2"
    assert payload["messages"] == [
        {"role": "system", "content": "System rules"},
        {"role": "user", "content": "User question"},
    ]


def test_ollama_provider_rejects_missing_model(monkeypatch):
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)

    with pytest.raises(RuntimeError, match="OLLAMA_MODEL"):
        OllamaProvider()


def test_ollama_provider_rejects_invalid_response(monkeypatch):
    monkeypatch.setattr(
        "screenplay_rag.ollama.urlopen",
        lambda request, timeout: FakeResponse({"message": {}}),
    )

    with pytest.raises(RuntimeError, match="message content"):
        OllamaProvider(model="llama3.2")(Prompt(system="", user=""))
