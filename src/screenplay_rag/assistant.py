"""Interactive role-based screenplay assistant command-line workflow."""

import argparse
from collections.abc import Callable

from .answer import AnswerProvider, answer_question
from .ingest import load_script
from .ollama import OllamaProvider
from .roles import ASSISTANT_DIRECTOR, ROLES, Role


def run(
    role: Role = ASSISTANT_DIRECTOR,
    provider: AnswerProvider | None = None,
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
    )
    output_fn(answer)
    return answer


def main(
    argv: list[str] | None = None,
    provider: AnswerProvider | None = None,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> None:
    parser = argparse.ArgumentParser(description="Ask a screenplay role for guidance.")
    parser.add_argument("--role", choices=sorted(ROLES), required=True)
    arguments = parser.parse_args(argv)
    run(ROLES[arguments.role], provider, input_fn, output_fn)


if __name__ == "__main__":
    main()
