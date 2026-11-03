"""Evaluate the TF-IDF retriever against a small screenplay question set."""

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from screenplay_rag.ingest import load_script
from screenplay_rag.retrieve import retrieve


CASES_PATH = REPO_ROOT / "evaluation" / "retrieval_cases.json"
SCRIPT_PATH = REPO_ROOT / "data" / "script.txt"


def load_cases(path: Path) -> list[dict[str, object]]:
    """Load evaluation questions and expected scene numbers from JSON."""
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    scenes = load_script(str(SCRIPT_PATH))
    cases = load_cases(CASES_PATH)
    hit_at_1 = 0
    hit_at_3 = 0

    for case in cases:
        question = str(case["question"])
        expected = set(case["expected_scene_numbers"])
        results = retrieve(question, scenes, top_k=3)
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


if __name__ == "__main__":
    main()
