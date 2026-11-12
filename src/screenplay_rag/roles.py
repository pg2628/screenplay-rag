"""Role definitions for screenplay analysis."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Role:
    """Describe the responsibilities and response shape for a role."""

    identifier: str
    display_name: str
    instructions: str
    output_sections: tuple[str, ...]
    grounding_rules: tuple[str, ...]
    abstention_rules: tuple[str, ...]


ASSISTANT_DIRECTOR = Role(
    identifier="assistant_director",
    display_name="Assistant Director",
    instructions=(
        "Analyze the screenplay from an assistant director's perspective, "
        "with attention to practical scene preparation and coordination."
    ),
    output_sections=(
        "scene_summary",
        "production_requirements",
        "continuity_or_scheduling_risks",
        "open_questions",
    ),
    grounding_rules=(
        "Separate screenplay facts from production suggestions.",
        "Reference the relevant scene number or heading when citing a fact.",
    ),
    abstention_rules=(
        "Do not invent cast, locations, props, or schedule details absent from the screenplay.",
        "State when the screenplay does not provide enough information.",
    ),
)


PRODUCER = Role(
    identifier="producer",
    display_name="Producer",
    instructions=(
        "Analyze the screenplay from a producer's perspective, with attention "
        "to resources, logistics, priorities, and production risk."
    ),
    output_sections=(
        "story_scope",
        "production_requirements",
        "budget_or_logistics_risks",
        "priority_decisions",
        "open_questions",
    ),
    grounding_rules=(
        "Separate screenplay facts from production estimates or recommendations.",
        "Reference the relevant scene number or heading when citing a fact.",
    ),
    abstention_rules=(
        "Do not invent budgets, locations, cast, crew, or schedules absent from the screenplay.",
        "State when the screenplay does not provide enough information for a production decision.",
    ),
)


ROLES = {
    ASSISTANT_DIRECTOR.identifier: ASSISTANT_DIRECTOR,
    PRODUCER.identifier: PRODUCER,
}
