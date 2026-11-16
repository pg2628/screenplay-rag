"""Hybrid retrieval by fusing lexical and semantic scene scores."""

import os
from collections.abc import Callable

from .retrieve import retrieve
from .semantic_retrieve import OllamaSemanticRetriever

Retriever = Callable[[str, list[dict[str, object]], int], list[dict[str, object]]]


class HybridRetriever:
    """Combine normalized TF-IDF and semantic scores for screenplay scenes."""

    def __init__(
        self,
        lexical_retriever: Retriever = retrieve,
        semantic_retriever: Retriever | None = None,
        semantic_weight: float | None = None,
    ):
        weight_text = os.environ.get("RAG_SEMANTIC_WEIGHT", "0.5")
        self.semantic_weight = (
            float(weight_text) if semantic_weight is None else semantic_weight
        )
        if not 0.0 <= self.semantic_weight <= 1.0:
            raise ValueError("semantic_weight must be between 0.0 and 1.0")
        self.lexical_retriever = lexical_retriever
        self.semantic_retriever = semantic_retriever or OllamaSemanticRetriever()

    def __call__(
        self,
        query: str,
        scenes: list[dict[str, object]],
        top_k: int = 3,
    ) -> list[dict[str, object]]:
        """Return the top scenes using weighted score fusion."""
        if not query.strip() or not scenes or top_k <= 0:
            return []

        candidate_count = len(scenes)
        lexical_results = self.lexical_retriever(query, scenes, candidate_count)
        semantic_results = self.semantic_retriever(query, scenes, candidate_count)
        lexical_scores = self._scores_by_scene(lexical_results)
        semantic_scores = self._scores_by_scene(semantic_results)
        lexical_scores = self._normalize(lexical_scores, scenes)
        semantic_scores = self._normalize(semantic_scores, scenes)

        ranked_scenes = sorted(
            scenes,
            key=lambda scene: (
                (1.0 - self.semantic_weight) * lexical_scores[self._scene_key(scene)]
                + self.semantic_weight * semantic_scores[self._scene_key(scene)]
            ),
            reverse=True,
        )
        return [
            {
                **scene,
                "score": (1.0 - self.semantic_weight)
                * lexical_scores[self._scene_key(scene)]
                + self.semantic_weight * semantic_scores[self._scene_key(scene)],
            }
            for scene in ranked_scenes[:top_k]
        ]

    @staticmethod
    def _scene_key(scene: dict[str, object]) -> tuple[object, ...]:
        return (
            scene.get("source", ""),
            scene.get("scene_number", ""),
            scene.get("heading", ""),
            scene.get("embedding_text", ""),
        )

    @classmethod
    def _scores_by_scene(
        cls, results: list[dict[str, object]]
    ) -> dict[tuple[object, ...], float]:
        return {
            cls._scene_key(result): float(result.get("score", 0.0))
            for result in results
        }

    @staticmethod
    def _normalize(
        scores: dict[tuple[object, ...], float], scenes: list[dict[str, object]]
    ) -> dict[tuple[object, ...], float]:
        values = [scores.get(HybridRetriever._scene_key(scene), 0.0) for scene in scenes]
        minimum = min(values)
        maximum = max(values)
        if minimum == maximum:
            return {HybridRetriever._scene_key(scene): 0.0 for scene in scenes}
        return {
            HybridRetriever._scene_key(scene): (
                scores.get(HybridRetriever._scene_key(scene), 0.0) - minimum
            )
            / (maximum - minimum)
            for scene in scenes
        }