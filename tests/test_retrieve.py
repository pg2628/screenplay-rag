import copy
from pathlib import Path

import pytest

from screenplay_rag.ingest import load_script
from screenplay_rag.retrieve import retrieve


@pytest.fixture(scope="module")
def scenes():
    script_path = Path(__file__).resolve().parents[1] / "data" / "script.txt"
    return load_script(str(script_path))


def test_day_two_returns_at_most_three_results(scenes):
    results = retrieve("What happens on Day Two?", scenes, top_k=3)
    assert len(results) <= 3


def test_day_two_top_result_is_scene_four(scenes):
    results = retrieve("What happens on Day Two?", scenes, top_k=3)
    assert results[0]["scene_number"] == 4


def test_final_scene_distinctive_query_returns_scene_six(scenes):
    results = retrieve(
        "What happens when Najjad shoots the general with the pistol?", scenes
    )
    assert results[0]["scene_number"] == 6


def test_scores_are_descending(scenes):
    results = retrieve("What happens on Day Two?", scenes)
    scores = [result["score"] for result in results]
    assert scores == sorted(scores, reverse=True)


def test_empty_question_returns_no_results(scenes):
    assert retrieve("", scenes) == []


def test_empty_scene_list_returns_no_results():
    assert retrieve("question", [], top_k=3) == []


def test_zero_top_k_returns_no_results(scenes):
    assert retrieve("question", scenes, top_k=0) == []


def test_retrieval_does_not_mutate_scene_dictionaries(scenes):
    original_scenes = copy.deepcopy(scenes)
    retrieve("What happens on Day Two?", scenes, top_k=3)
    assert scenes == original_scenes
