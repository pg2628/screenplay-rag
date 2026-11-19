import re

import pytest

from screenplay_rag.assistant import build_retriever, main, run, run_conversation
from screenplay_rag.prompts import Prompt
from screenplay_rag.roles import ASSISTANT_DIRECTOR, PRODUCER


class FakeAnswerProvider:
    def __init__(self):
        self.prompt = None

    def __call__(self, prompt: Prompt) -> str:
        self.prompt = prompt
        scene_number = re.search(r"Scene (\d+):", prompt.user).group(1)
        return f"fake Assistant Director answer [Scene {scene_number}]"


def test_run_uses_fake_provider_and_prints_answer():
    provider = FakeAnswerProvider()
    output = []

    answer = run(
        role=ASSISTANT_DIRECTOR,
        provider=provider,
        input_fn=lambda prompt: "What happens on Day Two?",
        output_fn=output.append,
    )

    assert answer.startswith("fake Assistant Director answer [Scene ")
    assert output == [answer]
    assert provider.prompt is not None
    assert "What happens on Day Two?" in provider.prompt.user


def test_main_selects_producer_role():
    provider = FakeAnswerProvider()
    output = []
    questions = iter(["What are the production risks?", "exit"])

    main(
        ["--role", "producer"],
        provider=provider,
        input_fn=lambda prompt: next(questions),
        output_fn=output.append,
    )

    assert output[0].startswith("fake Assistant Director answer [Scene ")
    assert PRODUCER.instructions in provider.prompt.system


def test_main_accepts_retriever_and_provider_flags():
    provider = FakeAnswerProvider()
    output = []
    questions = iter(["What are the production risks?", "exit"])
    retrieved = []

    def fake_retriever(query, scenes, top_k):
        retrieved.append((query, top_k))
        return scenes[:1]

    main(
        ["--role", "producer", "--retriever", "semantic", "--provider", "langchain"],
        provider=provider,
        retriever=fake_retriever,
        input_fn=lambda prompt: next(questions),
        output_fn=output.append,
    )

    assert output[0].startswith("fake Assistant Director answer [Scene ")
    assert retrieved == [("What are the production risks?", 3)]


def test_main_accepts_reranker_and_candidate_count_flags():
    provider = FakeAnswerProvider()
    retriever_calls = []
    reranker_calls = []

    class FakeReranker:
        def __call__(self, query, candidates, top_k):
            reranker_calls.append((query, len(candidates), top_k))
            return candidates[:1]

    reranker = FakeReranker()

    def retriever(query, scenes, top_k):
        retriever_calls.append((query, top_k))
        return scenes[:top_k]

    questions = iter(["What happens?", "exit"])

    main(
        [
            "--role",
            "producer",
            "--retriever",
            "tfidf",
            "--reranker",
            "tfidf",
            "--candidate-count",
            "8",
        ],
        provider=provider,
        retriever=retriever,
        reranker=reranker,
        candidate_count=8,
        input_fn=lambda prompt: next(questions),
        output_fn=lambda answer: None,
    )

    assert retriever_calls == [("What happens?", 8)]
    assert reranker_calls
    assert reranker_calls[0][0] == "What happens?"
    assert reranker_calls[0][1] <= 8
    assert reranker_calls[0][2] == 3


def test_build_retriever_uses_tfidf_baseline():
    assert build_retriever("tfidf").__name__ == "retrieve"


def test_main_rejects_unknown_role(capsys):
    with pytest.raises(SystemExit):
        main(["--role", "writer"])

    assert "invalid choice" in capsys.readouterr().err


def test_conversation_includes_recent_history_in_follow_up_prompt():
    class ConversationProvider:
        def __init__(self):
            self.prompts = []

        def __call__(self, prompt: Prompt) -> str:
            self.prompts.append(prompt)
            scene_number = re.search(r"Scene (\d+):", prompt.user).group(1)
            return f"answer {len(self.prompts)} [Scene {scene_number}]"

    provider = ConversationProvider()
    output = []
    questions = iter(["What happens on Day Two?", "What should a producer watch?", "quit"])

    run_conversation(
        role=PRODUCER,
        provider=provider,
        input_fn=lambda prompt: next(questions),
        output_fn=output.append,
    )

    assert output[0].startswith("answer 1 [Scene ")
    assert output[1].startswith("answer 2 [Scene ")
    assert "What happens on Day Two?" in provider.prompts[1].user
    assert "answer 1" in provider.prompts[1].user
    assert "What should a producer watch?" in provider.prompts[1].user
