from screenplay_rag.prompts import build_prompt
from screenplay_rag.roles import ASSISTANT_DIRECTOR, CINEMATOGRAPHER


def test_build_prompt_contains_role_question_and_scene_evidence():
    prompt = build_prompt(
        ASSISTANT_DIRECTOR,
        "What production requirements does this scene have?",
        [
            {
                "scene_number": 4,
                "heading": "EXT. DESERT CRATER - DAY",
                "text": "The People prepare for another battle.",
            }
        ],
    )

    assert "Assistant Director" in prompt.system
    assert ASSISTANT_DIRECTOR.instructions in prompt.system
    assert "do not treat them as mandatory headings" in prompt.system
    assert "Answer naturally unless the user requests a specific format." in prompt.system
    assert "Cite every screenplay factual claim as [Scene N: HEADING]." in prompt.system
    assert "Include a compact Sources footer containing only scenes you cited." in prompt.system
    assert "Treat the retrieved screenplay evidence as authoritative." in prompt.system
    assert "Treat conversation history as context, not evidence." in prompt.system
    assert "especially Day Two and Day Three" in prompt.system
    assert "Label professional suggestions as Recommendations" in prompt.system
    assert "Distinguish screenplay facts" not in prompt.system
    assert "If the evidence is insufficient, say so plainly." in prompt.system
    assert "Do not invent cast" in prompt.system
    assert "What production requirements does this scene have?" in prompt.user
    assert "Scene 4: EXT. DESERT CRATER - DAY" in prompt.user
    assert "The People prepare for another battle." in prompt.user
    assert "Do not invent cast" not in prompt.user


def test_build_prompt_is_deterministic_for_empty_evidence():
    first = build_prompt(ASSISTANT_DIRECTOR, "What is known?", [])
    second = build_prompt(ASSISTANT_DIRECTOR, "What is known?", [])

    assert first == second
    assert "No screenplay scenes were retrieved." in first.user


def test_build_prompt_includes_cinematographer_guidance():
    prompt = build_prompt(
        CINEMATOGRAPHER,
        "How might this scene be photographed?",
        [],
    )

    assert "Cinematographer" in prompt.system
    assert CINEMATOGRAPHER.instructions in prompt.system
    assert "visual_intent" in prompt.system
    assert "Identify visual ideas as interpretations" in prompt.system
    assert "Do not invent locations" in prompt.system
    assert "not enough visual information" in prompt.system
