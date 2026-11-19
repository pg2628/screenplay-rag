"""Validate screenplay scene citations in generated answers."""

from dataclasses import dataclass
import re


CITATION_PATTERN = re.compile(r"\[Scene\s+(\d+)(?::[^\]]*)?\]", re.IGNORECASE)
NO_EVIDENCE_MESSAGE = (
    "I can't provide a grounded answer because no screenplay evidence was retrieved."
)
MISSING_CITATION_MESSAGE = (
    "I can't provide a grounded answer because the response did not cite screenplay evidence."
)
INVALID_CITATION_MESSAGE = (
    "I can't provide a grounded answer because the response cited scenes outside the retrieved evidence."
)


@dataclass(frozen=True)
class GroundingResult:
    """Describe whether an answer is supported by retrieved screenplay scenes."""

    valid: bool
    cited_scene_numbers: frozenset[str]
    reason: str


def validate_answer(
    answer: str,
    retrieved_scenes: list[dict[str, object]],
) -> GroundingResult:
    """Validate that every answer citation refers to retrieved evidence."""
    if not retrieved_scenes:
        return GroundingResult(False, frozenset(), NO_EVIDENCE_MESSAGE)

    cited_scene_numbers = frozenset(
        match.group(1) for match in CITATION_PATTERN.finditer(answer)
    )
    if not cited_scene_numbers:
        return GroundingResult(False, cited_scene_numbers, MISSING_CITATION_MESSAGE)

    retrieved_scene_numbers = {
        str(scene.get("scene_number")) for scene in retrieved_scenes
    }
    if not cited_scene_numbers.issubset(retrieved_scene_numbers):
        return GroundingResult(False, cited_scene_numbers, INVALID_CITATION_MESSAGE)

    return GroundingResult(True, cited_scene_numbers, "")