"""Interactive role-based screenplay assistant command-line workflow."""

import argparse
from collections.abc import Callable

from .answer import AnswerProvider, answer_question
from .chroma_retrieve import ChromaRetriever
from .ingest import load_script
from .langchain_ollama import LangChainOllamaProvider
from .ollama import OllamaProvider
from .prompts import Prompt
from .retrieve import retrieve
from .roles import ASSISTANT_DIRECTOR, ROLES, Role
from .semantic_retrieve import OllamaSemanticRetriever

MAX_HISTORY_TURNS = 6
RETRIEVER_NAMES = ("tfidf", "semantic", "chroma")
PROVIDER_NAMES = ("custom", "langchain")
Retriever = Callable[[str, list[dict[str, object]], int], list[dict[str, object]]]


def build_retriever(name: str) -> Retriever:
    """Build a retriever by its CLI-facing name."""
    if name == "tfidf":
        return retrieve
    if name == "semantic":
        return OllamaSemanticRetriever()
    if name == "chroma":
        return ChromaRetriever()
    raise ValueError(f"Unknown retriever: {name}")


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
    )
    output_fn(answer)
    return answer


def run_conversation(
    role: Role = ASSISTANT_DIRECTOR,
    provider: AnswerProvider | None = None,
    retriever: Retriever | None = None,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
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

        answer_question(
            question,
            role,
            scenes,
            _ConversationProvider(answer_provider, history, question),
            retriever=retriever,
        )
        output_fn(history[-1][1])


def main(
    argv: list[str] | None = None,
    provider: AnswerProvider | None = None,
    retriever: Retriever | None = None,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> None:
    parser = argparse.ArgumentParser(description="Ask a screenplay role for guidance.")
    parser.add_argument("--role", choices=sorted(ROLES), required=True)
    parser.add_argument("--retriever", choices=RETRIEVER_NAMES, default="tfidf")
    parser.add_argument("--provider", choices=PROVIDER_NAMES, default="custom")
    arguments = parser.parse_args(argv)
    run_conversation(
        ROLES[arguments.role],
        provider or build_provider(arguments.provider),
        retriever or build_retriever(arguments.retriever),
        input_fn,
        output_fn,
    )


if __name__ == "__main__":
    main()
