"""A small TF-IDF retrieval baseline for screenplay scenes."""

from .ingest import load_script
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def retrieve(
    query: str,
    scenes: list[dict[str, object]],
    top_k: int = 3,
) -> list[dict[str, object]]:
    """Return the scenes most relevant to ``query`` using TF-IDF."""
    if not query.strip() or not scenes or top_k <= 0:
        return []

    # Fit one vectorizer so the query and every scene share the same vocabulary.
    texts = [str(scene.get("embedding_text", "")) for scene in scenes]
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([query, *texts])

    # Cosine similarity measures the angle between the query and scene vectors.
    scores = cosine_similarity(vectors[0:1], vectors[1:]).ravel()
    ranked_indexes = sorted(
        range(len(scenes)), key=lambda index: scores[index], reverse=True
    )[:top_k]

    # Copy each scene before adding its score, leaving the input unchanged.
    return [
        {**scenes[index], "score": float(scores[index])} for index in ranked_indexes
    ]


def main() -> None:
    """Load the script, ask one question, and print the best matching scenes."""
    scenes = load_script("data/script.txt")
    query = input("Ask a question: ").strip()

    for scene in retrieve(query, scenes):
        print(f"\nScore: {scene['score']:.4f}")
        print(f"Scene: {scene.get('scene_number', '')}")
        print(f"Heading: {scene.get('heading', '')}")
        print(f"Text: {scene.get('text', '')}")


if __name__ == "__main__":
    main()
