import pytest

from screenplay_rag.hybrid_retrieve import HybridRetriever


SCENES = [
    {
        "scene_number": number,
        "heading": f"Scene {number}",
        "embedding_text": f"Scene text {number}",
    }
    for number in (1, 2, 3)
]


def scored_results(scores):
    return [
        {**scene, "score": score}
        for scene, score in zip(SCENES, scores)
    ]


def test_hybrid_retriever_fuses_normalized_scores_and_queries_full_corpus():
    calls = []

    def lexical(query, scenes, top_k):
        calls.append(("lexical", top_k))
        return scored_results([1.0, 0.5, 0.0])

    def semantic(query, scenes, top_k):
        calls.append(("semantic", top_k))
        return scored_results([0.0, 1.0, 0.5])

    retriever = HybridRetriever(
        lexical_retriever=lexical,
        semantic_retriever=semantic,
        semantic_weight=0.5,
    )

    results = retriever("question", SCENES, top_k=3)

    assert calls == [("lexical", 3), ("semantic", 3)]
    assert [scene["scene_number"] for scene in results] == [2, 1, 3]
    assert [scene["score"] for scene in results] == [0.75, 0.5, 0.25]


def test_hybrid_retriever_normalizes_negative_scores_before_filling_missing():
    def lexical(query, scenes, top_k):
        return [
            {**scenes[0], "score": -0.2},
            {**scenes[1], "score": -0.8},
        ]

    retriever = HybridRetriever(
        lexical_retriever=lexical,
        semantic_retriever=lambda query, scenes, top_k: [],
        semantic_weight=0.0,
    )

    results = retriever("question", SCENES)

    assert [scene["scene_number"] for scene in results] == [1, 2, 3]
    assert results[2]["score"] == 0.0


def test_hybrid_retriever_treats_missing_score_as_zero_after_normalization():
    def lexical(query, scenes, top_k):
        return [{**scenes[0]}, {**scenes[1], "score": 1.0}]

    retriever = HybridRetriever(
        lexical_retriever=lexical,
        semantic_retriever=lambda query, scenes, top_k: [],
        semantic_weight=0.0,
    )

    results = retriever("question", SCENES)

    assert results[0]["scene_number"] == 2
    assert results[1]["score"] == 0.0


def test_hybrid_retriever_preserves_duplicate_scene_records():
    duplicate_scenes = [SCENES[0], SCENES[0], SCENES[1]]

    def scored_retriever(query, scenes, top_k):
        return [{**scene, "score": 1.0} for scene in scenes]

    retriever = HybridRetriever(
        lexical_retriever=scored_retriever,
        semantic_retriever=scored_retriever,
    )

    results = retriever("question", duplicate_scenes, top_k=2)

    assert len(results) == 2
    assert [scene["scene_number"] for scene in results] == [1, 1]


def test_hybrid_weight_can_select_one_component():
    lexical = lambda query, scenes, top_k: scored_results([1.0, 0.5, 0.0])
    semantic = lambda query, scenes, top_k: scored_results([0.0, 1.0, 0.5])

    lexical_results = HybridRetriever(lexical, semantic, semantic_weight=0.0)(
        "question", SCENES
    )
    semantic_results = HybridRetriever(lexical, semantic, semantic_weight=1.0)(
        "question", SCENES
    )

    assert lexical_results[0]["scene_number"] == 1
    assert semantic_results[0]["scene_number"] == 2


def test_hybrid_retriever_reads_semantic_weight_from_environment(monkeypatch):
    monkeypatch.setenv("RAG_SEMANTIC_WEIGHT", "0.8")

    retriever = HybridRetriever(
        lexical_retriever=lambda query, scenes, top_k: [],
        semantic_retriever=lambda query, scenes, top_k: [],
    )

    assert retriever.semantic_weight == 0.8


@pytest.mark.parametrize("query, scenes, top_k", [("", SCENES, 3), ("question", [], 3), ("question", SCENES, 0)])
def test_hybrid_retriever_handles_empty_inputs(query, scenes, top_k):
    retriever = HybridRetriever(
        lexical_retriever=lambda query, scenes, top_k: pytest.fail("not called"),
        semantic_retriever=lambda query, scenes, top_k: pytest.fail("not called"),
    )

    assert retriever(query, scenes, top_k) == []


def test_hybrid_retriever_respects_top_k_bounds():
    retriever = HybridRetriever(
        lexical_retriever=lambda query, scenes, top_k: scored_results([1.0, 0.5, 0.0]),
        semantic_retriever=lambda query, scenes, top_k: scored_results([0.0, 0.5, 1.0]),
    )

    assert len(retriever("question", SCENES, top_k=1)) == 1
    assert len(retriever("question", SCENES, top_k=10)) == len(SCENES)


def test_hybrid_retriever_rejects_invalid_weight():
    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        HybridRetriever(semantic_weight=1.1)