"""Orchestrate screenplay retrieval, prompt construction, and answer generation."""

from collections.abc import Callable
from typing import Protocol

from .grounding import GroundingResult, NO_EVIDENCE_MESSAGE, validate_answer
from .prompts import Prompt, build_prompt
from .retrieve import retrieve
from .roles import Role


class AnswerProvider(Protocol):
    """Callable interface for an answer-generating provider."""

    def __call__(self, prompt: Prompt) -> str: ...


class AnswerValidator(Protocol):
    """Callable interface for validating generated answers."""

    def __call__(
        self, answer: str, retrieved_scenes: list[dict[str, object]]
    ) -> GroundingResult: ...


class AnswerReranker(Protocol):
    """Callable interface for reranking retrieved candidate scenes."""

    def __call__(
        self,
        query: str,
        candidates: list[dict[str, object]],
        top_k: int,
    ) -> list[dict[str, object]]: ...


def answer_question(
    question: str,
    role: Role,
    scenes: list[dict[str, object]],
    provider: AnswerProvider,
    top_k: int = 3,
    retriever: Callable[
        [str, list[dict[str, object]], int], list[dict[str, object]]
    ] | None = None,
    validator: AnswerValidator | None = None,
    reranker: AnswerReranker | None = None,
    candidate_count: int = 10,
) -> str:
    """Retrieve evidence, rerank it, generate an answer, and validate citations."""
    if reranker is not None and candidate_count <= 0:
        raise ValueError("candidate_count must be greater than zero")

    retrieval_function = retrieve if retriever is None else retriever
    retrieval_count = max(candidate_count, top_k) if reranker else top_k
    candidates = retrieval_function(question, scenes, retrieval_count)
    if not candidates:
        return NO_EVIDENCE_MESSAGE

    retrieved_scenes = reranker(question, candidates, top_k) if reranker else candidates
    if not retrieved_scenes:
        return NO_EVIDENCE_MESSAGE

    prompt = build_prompt(role, question, retrieved_scenes)
    answer = provider(prompt)
    validation_function = validate_answer if validator is None else validator
    validation = validation_function(answer, retrieved_scenes)
    return answer if validation.valid else validation.reason
