"""Persistent Chroma retrieval for screenplay scenes."""

import os
from pathlib import Path


class ChromaRetriever:
    """Retrieve screenplay scenes from a persistent Chroma collection."""

    def __init__(
        self,
        persist_directory: str = "data/chroma",
        collection_name: str = "screenplay_scenes",
        host: str | None = None,
        model: str | None = None,
        vector_store=None,
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.host = (
            host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        ).rstrip("/")
        self.model = model or os.environ.get(
            "OLLAMA_EMBED_MODEL", "granite-embedding:278m"
        )
        self.vector_store = vector_store or self._create_vector_store()

    def __call__(
        self,
        query: str,
        scenes: list[dict[str, object]],
        top_k: int = 3,
    ) -> list[dict[str, object]]:
        """Return scenes ranked by persistent vector similarity."""
        if not query.strip() or top_k <= 0:
            return []

        matches = self.vector_store.similarity_search_with_relevance_scores(
            query, k=top_k
        )
        results = []
        for document, score in matches:
            metadata = document.metadata
            text = str(metadata.get("text", document.page_content))
            heading = str(metadata.get("heading", "Untitled scene"))
            results.append(
                {
                    "scene_number": metadata.get("scene_number", "unknown"),
                    "heading": heading,
                    "source": metadata.get("source", "unknown"),
                    "text": text,
                    "embedding_text": f"{heading}\n{text}",
                    "score": float(score),
                }
            )
        return results

    def _create_vector_store(self):
        from langchain_chroma import Chroma
        from langchain_ollama import OllamaEmbeddings

        embeddings = OllamaEmbeddings(model=self.model, base_url=self.host)
        return Chroma(
            collection_name=self.collection_name,
            embedding_function=embeddings,
            persist_directory=self.persist_directory,
        )