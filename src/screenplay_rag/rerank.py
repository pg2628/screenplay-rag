"""Lightweight reranking for screenplay scene candidates."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfidfReranker:
    """Rerank candidate scenes using TF-IDF cosine similarity."""

    def __call__(
        self,
        query: str,
        candidates: list[dict[str, object]],
        top_k: int,
    ) -> list[dict[str, object]]:
        """Return the most relevant candidates in descending relevance."""
        if not query.strip() or not candidates or top_k <= 0:
            return []

        texts = [
            str(candidate.get("embedding_text", candidate.get("text", "")))
            for candidate in candidates
        ]
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([query, *texts])
        scores = cosine_similarity(vectors[0:1], vectors[1:]).ravel()

        ranked_indexes = sorted(
            range(len(candidates)),
            key=lambda index: scores[index],
            reverse=True,
        )[:top_k]

        return [
            {**candidates[index], "rerank_score": float(scores[index])}
            for index in ranked_indexes
        ]
