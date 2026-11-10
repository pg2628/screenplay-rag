from screenplay_rag.prompts import build_prompt
from screenplay_rag.roles import ASSISTANT_DIRECTOR


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
