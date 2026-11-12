"""In-memory semantic retrieval using Ollama embeddings."""

import json
import math
import os
from urllib.request import Request, urlopen


class OllamaSemanticRetriever:
    """Rank screenplay scenes by cosine similarity from Ollama embeddings."""

    def __init__(self, host: str | None = None, model: str | None = None):
        self.host = (
            host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        ).rstrip("/")
        self.model = model or os.environ.get(
            "OLLAMA_EMBED_MODEL", "granite-embedding:278m"
        )

    def __call__(
        self,
        query: str,
        scenes: list[dict[str, object]],
        top_k: int = 3,
    ) -> list[dict[str, object]]:
        """Return the scenes most semantically relevant to ``query``."""
        if not query.strip() or not scenes or top_k <= 0:
            return []

        texts = [query] + [str(scene.get("embedding_text", "")) for scene in scenes]
        embeddings = self._embed(texts)
        query_embedding = embeddings[0]
        scores = [
            self._cosine_similarity(query_embedding, embedding)
            for embedding in embeddings[1:]
        ]
        ranked_indexes = sorted(
            range(len(scenes)), key=lambda index: scores[index], reverse=True
        )[:top_k]
        return [
            {**scenes[index], "score": float(scores[index])}
            for index in ranked_indexes
        ]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        payload = {"model": self.model, "input": texts}
        request = Request(
            f"{self.host}/api/embed",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=120) as response:
                response_data = json.loads(response.read().decode("utf-8"))
        except (OSError, TimeoutError) as error:
            raise RuntimeError("Could not connect to the Ollama embedding API") from error
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise RuntimeError("Ollama returned invalid embedding JSON") from error

        embeddings = response_data.get("embeddings") if isinstance(response_data, dict) else None
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise RuntimeError("Ollama response did not contain embeddings for every input")

        if not all(
            isinstance(embedding, list)
            and all(isinstance(value, (int, float)) for value in embedding)
            for embedding in embeddings
        ):
            raise RuntimeError("Ollama response contained invalid embeddings")
        return embeddings

    @staticmethod
    def _cosine_similarity(first: list[float], second: list[float]) -> float:
        if len(first) != len(second):
            raise RuntimeError("Ollama embeddings have inconsistent dimensions")
        first_norm = math.sqrt(sum(value * value for value in first))
        second_norm = math.sqrt(sum(value * value for value in second))
        if not first_norm or not second_norm:
            return 0.0
        dot_product = sum(left * right for left, right in zip(first, second))
        return dot_product / (first_norm * second_norm)