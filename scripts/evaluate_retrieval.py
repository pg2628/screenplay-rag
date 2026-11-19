"""Compare TF-IDF, hybrid, Ollama semantic, and Chroma retrieval."""

import json
from collections.abc import Callable
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

from screenplay_rag.ingest import load_script
from screenplay_rag.chroma_retrieve import ChromaRetriever
from screenplay_rag.hybrid_retrieve import HybridRetriever
from screenplay_rag.rerank import TfidfReranker
from screenplay_rag.retrieve import retrieve
from screenplay_rag.semantic_retrieve import OllamaSemanticRetriever


CASES_PATH = REPO_ROOT / "evaluation" / "retrieval_cases.json"
SCRIPT_PATH = REPO_ROOT / "data" / "script.txt"
CHROMA_PATH = REPO_ROOT / "data" / "chroma"


def load_cases(path: Path) -> list[dict[str, object]]:
    """Load evaluation questions and expected scene numbers from JSON."""
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def evaluate_retriever(
    name: str,
    retriever: Callable[
        [str, list[dict[str, object]], int], list[dict[str, object]]
    ],
    scenes: list[dict[str, object]],
    cases: list[dict[str, object]],
) -> None:
    """Print per-question results and aggregate metrics for one retriever."""
    print(f"=== {name} ===")
    hit_at_1 = 0
    hit_at_3 = 0

    for case in cases:
        question = str(case["question"])
        expected = set(case["expected_scene_numbers"])
        results = retriever(question, scenes, 3)
        returned = [int(result["scene_number"]) for result in results]
        is_hit_at_1 = bool(returned) and returned[0] in expected
        is_hit_at_3 = bool(expected.intersection(returned))
        hit_at_1 += is_hit_at_1
        hit_at_3 += is_hit_at_3

        print(f"Question: {question}")
        print(f"Expected scenes: {sorted(expected)}")
        print(f"Returned scenes: {returned}")
        print(f"Hit@1: {'yes' if is_hit_at_1 else 'no'}")
        print(f"Hit@3: {'yes' if is_hit_at_3 else 'no'}")
        print()

    case_count = len(cases)
    print(f"Hit@1: {hit_at_1}/{case_count} ({hit_at_1 / case_count:.1%})")
    print(f"Hit@3: {hit_at_3}/{case_count} ({hit_at_3 / case_count:.1%})")
    print()


def main() -> None:
    scenes = load_script(str(SCRIPT_PATH))
    cases = load_cases(CASES_PATH)
    evaluate_retriever("TF-IDF", retrieve, scenes, cases)

    hybrid_retriever = HybridRetriever()
    evaluate_retriever("Hybrid", hybrid_retriever, scenes, cases)

    reranker = TfidfReranker()

    def hybrid_reranked(question: str, available_scenes: list[dict[str, object]], top_k: int):
        candidates = hybrid_retriever(question, available_scenes, max(top_k * 3, top_k))
        return reranker(question, candidates, top_k)

    evaluate_retriever("Hybrid + reranking", hybrid_reranked, scenes, cases)
    evaluate_retriever(
        "Ollama semantic",
        OllamaSemanticRetriever(),
        scenes,
        cases,
    )

    if CHROMA_PATH.is_dir() and any(CHROMA_PATH.iterdir()):
        print(f"Chroma index: available at {CHROMA_PATH}")
        evaluate_retriever(
            "Chroma semantic",
            ChromaRetriever(persist_directory=str(CHROMA_PATH)),
            scenes,
            cases,
        )
    else:
        print(f"Chroma index: unavailable at {CHROMA_PATH}; skipping")


if __name__ == "__main__":
    main()
