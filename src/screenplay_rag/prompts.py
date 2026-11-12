"""Deterministic prompt construction for screenplay roles."""

from dataclasses import dataclass

from .roles import Role


@dataclass(frozen=True)
class Prompt:
    """Separate stable role behavior from request-specific content."""

    system: str
    user: str


def build_prompt(
    role: Role,
    question: str,
    scenes: list[dict[str, object]],
) -> Prompt:
    """Build separate system and user prompts for a role and retrieved scenes."""
    output_sections = "\n".join(f"- {section}" for section in role.output_sections)
    grounding_rules = "\n".join(f"- {rule}" for rule in role.grounding_rules)
    abstention_rules = "\n".join(f"- {rule}" for rule in role.abstention_rules)

    evidence = []
    for scene in scenes:
        scene_number = scene.get("scene_number", "unknown")
        heading = scene.get("heading", "Untitled scene")
        text = scene.get("text", "")
        evidence.append(f"Scene {scene_number}: {heading}\n{text}")

    evidence_text = "\n\n".join(evidence) or "No screenplay scenes were retrieved."

    system = f"""You are acting as the {role.display_name}.

Role instructions:
{role.instructions}

Use these topics as guidance when relevant; do not treat them as mandatory headings:
{output_sections}

Answer naturally unless the user requests a specific format.
For screenplay facts, cite the scene number and heading from the evidence.
Clearly distinguish screenplay facts from professional recommendations.
If the evidence is insufficient, say so plainly.

Grounding rules:
{grounding_rules}

Abstention rules:
{abstention_rules}
"""

    user = f"""User question:
{question}

Retrieved screenplay evidence:
{evidence_text}
"""

    return Prompt(system=system, user=user)
