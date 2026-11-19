from screenplay_rag.grounding import (
    INVALID_CITATION_MESSAGE,
    MISSING_CITATION_MESSAGE,
    NO_EVIDENCE_MESSAGE,
    validate_answer,
)


SCENES = [
    {"scene_number": 4, "heading": "EXT. CRATER - DAY"},
    {"scene_number": 6, "heading": "INT. HOUSE - NIGHT"},
]


def test_validate_answer_accepts_retrieved_scene_citations():
    result = validate_answer("The battle begins. [Scene 4: EXT. CRATER - DAY]", SCENES)

    assert result.valid is True
    assert result.cited_scene_numbers == frozenset({"4"})


def test_validate_answer_rejects_missing_citations():
    result = validate_answer("The battle begins.", SCENES)

    assert result.valid is False
    assert result.reason == MISSING_CITATION_MESSAGE


def test_validate_answer_rejects_citations_outside_retrieved_evidence():
    result = validate_answer("The battle begins. [Scene 9]", SCENES)

    assert result.valid is False
    assert result.reason == INVALID_CITATION_MESSAGE


def test_validate_answer_rejects_missing_evidence():
    result = validate_answer("Nothing is known.", [])

    assert result.valid is False
    assert result.reason == NO_EVIDENCE_MESSAGE


def test_validate_answer_accepts_multiple_retrieved_citations():
    result = validate_answer("The scenes connect. [Scene 4] [Scene 6]", SCENES)

    assert result.valid is True
    assert result.cited_scene_numbers == frozenset({"4", "6"})