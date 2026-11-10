"""Deterministic prompt construction for screenplay roles."""

from .roles import Role


def build_prompt(
    role: Role,
    question: str,
    scenes: list[dict[str, object]],
) -> str:
    """Build a grounded prompt from a role, question, and retrieved scenes."""
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

    return f"""You are acting as the {role.display_name}.

Role instructions:
{role.instructions}

Respond using these sections:
{output_sections}

Grounding rules:
{grounding_rules}

Abstention rules:
{abstention_rules}

User question:
{question}

Retrieved screenplay evidence:
{evidence_text}
"""
