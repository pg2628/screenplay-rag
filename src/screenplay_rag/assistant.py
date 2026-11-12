"""Interactive Assistant Director command-line workflow."""

from typing import Optional
from collections.abc import Callable

from .answer import AnswerProvider, answer_question
from .ingest import load_script
from .ollama import OllamaProvider
from .roles import ASSISTANT_DIRECTOR


def run(
    provider: AnswerProvider | None = None,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> str:
    """Read one question, answer it as an Assistant Director, and print it."""
    scenes = load_script("data/script.txt")
    question = input_fn("Ask a question: ").strip()
    answer = answer_question(
        question,
        ASSISTANT_DIRECTOR,
        scenes,
        provider or OllamaProvider(),
    )
    output_fn(answer)
    return answer


def main() -> None:
    run()


if __name__ == "__main__":
    main()
