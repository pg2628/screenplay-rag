from screenplay_rag.assistant import run
from screenplay_rag.prompts import Prompt


class FakeAnswerProvider:
    def __init__(self):
        self.prompt = None

    def __call__(self, prompt: Prompt) -> str:
        self.prompt = prompt
        return "fake Assistant Director answer"


def test_run_uses_fake_provider_and_prints_answer():
    provider = FakeAnswerProvider()
    output = []

    answer = run(
        provider=provider,
        input_fn=lambda prompt: "What happens on Day Two?",
        output_fn=output.append,
    )

    assert answer == "fake Assistant Director answer"
    assert output == ["fake Assistant Director answer"]
    assert provider.prompt is not None
    assert "What happens on Day Two?" in provider.prompt.user
