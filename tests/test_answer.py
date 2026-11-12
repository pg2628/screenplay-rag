from screenplay_rag.answer import answer_question
from screenplay_rag.prompts import Prompt
from screenplay_rag.roles import ASSISTANT_DIRECTOR


class FakeAnswerProvider:
    def __init__(self):
        self.prompt = None

    def __call__(self, prompt: Prompt) -> str:
        self.prompt = prompt
        return "fake grounded answer"


def test_answer_question_retrieves_builds_prompt_and_returns_answer():
    provider = FakeAnswerProvider()
    scenes = [
        {
            "scene_number": 1,
            "heading": "EXT. FARM - EVENING",
            "text": "The farmer meets the general.",
            "embedding_text": "EXT. FARM - EVENING The farmer meets the general.",
        },
        {
            "scene_number": 2,
            "heading": "INT. HOUSE - NIGHT",
            "text": "The family eats dinner.",
            "embedding_text": "INT. HOUSE - NIGHT The family eats dinner.",
        },
    ]

    answer = answer_question(
        "What happens at the farm?",
        ASSISTANT_DIRECTOR,
        scenes,
        provider,
        top_k=1,
    )

    assert answer == "fake grounded answer"
    assert provider.prompt is not None
    assert "What happens at the farm?" in provider.prompt.user
    assert "Scene 1: EXT. FARM - EVENING" in provider.prompt.user
    assert "Scene 2: INT. HOUSE - NIGHT" not in provider.prompt.user
    assert ASSISTANT_DIRECTOR.instructions in provider.prompt.system


def test_answer_question_uses_injected_retriever():
    provider = FakeAnswerProvider()
    retrieved_scene = {
        "scene_number": 9,
        "heading": "EXT. STUDIO LOT - DAY",
        "text": "The crew prepares the set.",
    }
    calls = []

    def fake_retriever(question, scenes, top_k):
        calls.append((question, scenes, top_k))
        return [retrieved_scene]

    answer_question(
        "What is being prepared?",
        ASSISTANT_DIRECTOR,
        [],
        provider,
        top_k=1,
        retriever=fake_retriever,
    )

    assert calls == [("What is being prepared?", [], 1)]
    assert "Scene 9: EXT. STUDIO LOT - DAY" in provider.prompt.user
    assert "The crew prepares the set." in provider.prompt.user
