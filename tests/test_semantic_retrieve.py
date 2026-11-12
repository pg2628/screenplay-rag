import json

import pytest

from screenplay_rag.semantic_retrieve import OllamaSemanticRetriever


class FakeResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return self.payload


def test_semantic_retriever_batches_embeddings_and_ranks_scenes(monkeypatch):
    requests = []

    def fake_urlopen(request, timeout):
        requests.append((request, timeout))
        return FakeResponse(
            {
                "embeddings": [
                    [1.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                ]
            }
        )

    monkeypatch.setattr("screenplay_rag.semantic_retrieve.urlopen", fake_urlopen)
    retriever = OllamaSemanticRetriever(model="granite-embedding:278m")
    scenes = [
        {
            "scene_number": 1,
            "embedding_text": "The farmer meets the general.",
        },
        {
            "scene_number": 2,
            "embedding_text": "The family eats dinner.",
        },
    ]

    results = retriever("What happens at the farm?", scenes, top_k=2)

    request, timeout = requests[0]
    payload = json.loads(request.data.decode("utf-8"))
    assert [result["scene_number"] for result in results] == [1, 2]
    assert results[0]["score"] == 1.0
    assert results[1]["score"] == 0.0
    assert request.full_url == "http://localhost:11434/api/embed"
    assert timeout == 120
    assert payload["model"] == "granite-embedding:278m"
    assert payload["input"] == [
        "What happens at the farm?",
        "The farmer meets the general.",
        "The family eats dinner.",
    ]


def test_semantic_retriever_uses_environment_model(monkeypatch):
    monkeypatch.setenv("OLLAMA_EMBED_MODEL", "custom-embedding")

    retriever = OllamaSemanticRetriever()

    assert retriever.model == "custom-embedding"


def test_semantic_retriever_rejects_incomplete_response(monkeypatch):
    monkeypatch.setattr(
        "screenplay_rag.semantic_retrieve.urlopen",
        lambda request, timeout: FakeResponse({"embeddings": [[1.0, 0.0]]}),
    )

    with pytest.raises(RuntimeError, match="every input"):
        OllamaSemanticRetriever()("question", [{"embedding_text": "scene"}])