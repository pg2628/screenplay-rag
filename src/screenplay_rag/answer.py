"""Orchestrate screenplay retrieval, prompt construction, and answer generation."""

from typing import Protocol

from .prompts import Prompt, build_prompt
from .retrieve import retrieve
from .roles import Role


class AnswerProvider(Protocol):
    """Callable interface for an answer-generating provider."""

    def __call__(self, prompt: Prompt) -> str: ...


def answer_question(
    question: str,
    role: Role,
    scenes: list[dict[str, object]],
    provider: AnswerProvider,
    top_k: int = 3,
) -> str:
    """Retrieve evidence, build a role prompt, and return the provider answer."""
    retrieved_scenes = retrieve(question, scenes, top_k=top_k)
    prompt = build_prompt(role, question, retrieved_scenes)
    return provider(prompt)
