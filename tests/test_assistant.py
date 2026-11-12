import pytest

from screenplay_rag.assistant import main, run
from screenplay_rag.prompts import Prompt
from screenplay_rag.roles import ASSISTANT_DIRECTOR, PRODUCER


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
        role=ASSISTANT_DIRECTOR,
        provider=provider,
        input_fn=lambda prompt: "What happens on Day Two?",
        output_fn=output.append,
    )

    assert answer == "fake Assistant Director answer"
    assert output == ["fake Assistant Director answer"]
    assert provider.prompt is not None
    assert "What happens on Day Two?" in provider.prompt.user


def test_main_selects_producer_role():
    provider = FakeAnswerProvider()
    output = []

    main(
        ["--role", "producer"],
        provider=provider,
        input_fn=lambda prompt: "What are the production risks?",
        output_fn=output.append,
    )

    assert output == ["fake Assistant Director answer"]
    assert PRODUCER.instructions in provider.prompt.system


def test_main_rejects_unknown_role(capsys):
    with pytest.raises(SystemExit):
        main(["--role", "cinematographer"])

    assert "invalid choice" in capsys.readouterr().err
