import pytest

from screenplay_rag.rerank import TfidfReranker


SCENES = [
    {
        "scene_number": 1,
        "heading": "EXT. FARM - DAY",
        "embedding_text": "A farmer watches the road.",
    },
    {
        "scene_number": 2,
        "heading": "INT. HOUSE - NIGHT",
        "embedding_text": "The family eats dinner.",
    },
    {
        "scene_number": 3,
        "heading": "EXT. ROAD - DAY",
        "embedding_text": "A farmer walks along the road.",
    },
]


def test_tfidf_reranker_returns_relevance_order():
    results = TfidfReranker()("farm road", SCENES, top_k=2)

    assert [scene["scene_number"] for scene in results] == [1, 3]
    assert results[0]["rerank_score"] >= results[1]["rerank_score"]


@pytest.mark.parametrize(
    "query,candidates,top_k",
    [("", SCENES, 2), ("question", [], 2), ("question", SCENES, 0)],
)
def test_tfidf_reranker_handles_empty_inputs(query, candidates, top_k):
    assert TfidfReranker()(query, candidates, top_k) == []


def test_tfidf_reranker_handles_missing_embedding_fields():
    candidates = [
        {"scene_number": 1, "text": "farm road"},
        {"scene_number": 2, "heading": "House"},
    ]

    results = TfidfReranker()("farm", candidates, top_k=1)

    assert results[0]["scene_number"] == 1


def test_tfidf_reranker_top_k_larger_than_candidates():
    results = TfidfReranker()("farm", SCENES[:2], top_k=10)

    assert len(results) == 2


def test_tfidf_reranker_preserves_duplicate_scene_records():
    duplicates = [SCENES[0], SCENES[0].copy(), SCENES[1]]

    results = TfidfReranker()("farm", duplicates, top_k=3)

    assert len(results) == 3
    assert sum(result["scene_number"] == 1 for result in results) == 2
