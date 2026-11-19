"""Interactive role-based screenplay assistant command-line workflow."""

import argparse
from collections.abc import Callable

from .answer import AnswerProvider, answer_question
from .chroma_retrieve import ChromaRetriever
from .hybrid_retrieve import HybridRetriever
from .ingest import load_script
from .langchain_ollama import LangChainOllamaProvider
from .ollama import OllamaProvider
from .prompts import Prompt
from .rerank import TfidfReranker
from .retrieve import retrieve
from .roles import ASSISTANT_DIRECTOR, ROLES, Role
from .semantic_retrieve import OllamaSemanticRetriever

MAX_HISTORY_TURNS = 6
RETRIEVER_NAMES = ("tfidf", "semantic", "chroma", "hybrid")
PROVIDER_NAMES = ("custom", "langchain")
RERANKER_NAMES = ("none", "tfidf")
Retriever = Callable[[str, list[dict[str, object]], int], list[dict[str, object]]]


def build_retriever(name: str) -> Retriever:
    """Build a retriever by its CLI-facing name."""
    if name == "tfidf":
        return retrieve
    if name == "semantic":
        return OllamaSemanticRetriever()
    if name == "chroma":
        return ChromaRetriever()
    if name == "hybrid":
        return HybridRetriever()
    raise ValueError(f"Unknown retriever: {name}")


def build_reranker(name: str):
    """Build a reranker by its CLI-facing name."""
    if name == "none":
        return None
    if name == "tfidf":
        return TfidfReranker()
    raise ValueError(f"Unknown reranker: {name}")


def build_provider(name: str) -> AnswerProvider:
    """Build an answer provider by its CLI-facing name."""
    if name == "custom":
        return OllamaProvider()
    if name == "langchain":
        return LangChainOllamaProvider()
    raise ValueError(f"Unknown provider: {name}")


class _ConversationProvider:
    def __init__(
        self,
        provider: AnswerProvider,
        history: list[tuple[str, str]],
        question: str,
    ):
        self.provider = provider
        self.history = history
        self.question = question

    def __call__(self, prompt: Prompt) -> str:
        recent_history = self.history[-MAX_HISTORY_TURNS:]
        if recent_history:
            history_text = "\n\n".join(
                f"User: {question}\nAssistant: {answer}"
                for question, answer in recent_history
            )
            user_text = (
                f"Conversation history:\n{history_text}\n\n"
                f"Current turn:\n{prompt.user}"
            )
            prompt = Prompt(system=prompt.system, user=user_text)

        answer = self.provider(prompt)
        self.history.append((self.question, answer))
        return answer


def run(
    role: Role = ASSISTANT_DIRECTOR,
    provider: AnswerProvider | None = None,
    retriever: Retriever | None = None,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
    reranker: object | None = None,
    candidate_count: int = 10,
) -> str:
    """Read one question, answer it in the selected role, and print it."""
    scenes = load_script("data/script.txt")
    question = input_fn("Ask a question: ").strip()
    answer = answer_question(
        question,
        role,
        scenes,
        provider or OllamaProvider(),
        retriever=retriever,
        reranker=reranker,
        candidate_count=candidate_count,
    )
    output_fn(answer)
    return answer


def run_conversation(
    role: Role = ASSISTANT_DIRECTOR,
    provider: AnswerProvider | None = None,
    retriever: Retriever | None = None,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
    reranker: object | None = None,
    candidate_count: int = 10,
) -> None:
    """Keep answering questions until the user exits or sends EOF."""
    scenes = load_script("data/script.txt")
    answer_provider = provider or OllamaProvider()
    history: list[tuple[str, str]] = []

    while True:
        try:
            question = input_fn("Ask a question (or type 'exit'): ").strip()
        except EOFError:
            break

        if question.lower() in {"exit", "quit"}:
            break

        answer = answer_question(
            question,
            role,
            scenes,
            _ConversationProvider(answer_provider, history, question),
            retriever=retriever,
            reranker=reranker,
            candidate_count=candidate_count,
        )
        output_fn(answer)


def main(
    argv: list[str] | None = None,
    provider: AnswerProvider | None = None,
    retriever: Retriever | None = None,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
    reranker: object | None = None,
    candidate_count: int | None = None,
) -> None:
    parser = argparse.ArgumentParser(description="Ask a screenplay role for guidance.")
    parser.add_argument("--role", choices=sorted(ROLES), required=True)
    parser.add_argument("--retriever", choices=RETRIEVER_NAMES, default="tfidf")
    parser.add_argument("--provider", choices=PROVIDER_NAMES, default="custom")
    parser.add_argument("--reranker", choices=RERANKER_NAMES, default="none")
    parser.add_argument("--candidate-count", type=int, default=10)
    arguments = parser.parse_args(argv)
    if arguments.candidate_count <= 0:
        parser.error("--candidate-count must be greater than zero")
    run_conversation(
        ROLES[arguments.role],
        provider or build_provider(arguments.provider),
        retriever or build_retriever(arguments.retriever),
        input_fn,
        output_fn,
        reranker if reranker is not None else build_reranker(arguments.reranker),
        arguments.candidate_count if candidate_count is None else candidate_count,
    )


if __name__ == "__main__":
    main()
