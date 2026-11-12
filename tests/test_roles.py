from screenplay_rag.roles import CINEMATOGRAPHER, ASSISTANT_DIRECTOR, PRODUCER, ROLES


def test_assistant_director_role_has_expected_configuration():
    assert ASSISTANT_DIRECTOR.identifier == "assistant_director"
    assert ASSISTANT_DIRECTOR.display_name == "Assistant Director"
    assert ASSISTANT_DIRECTOR.instructions
    assert ASSISTANT_DIRECTOR.output_sections
    assert ASSISTANT_DIRECTOR.grounding_rules
    assert ASSISTANT_DIRECTOR.abstention_rules


def test_assistant_director_is_registered_by_identifier():
    assert ROLES["assistant_director"] is ASSISTANT_DIRECTOR


def test_producer_role_is_registered_and_configured():
    assert PRODUCER.identifier == "producer"
    assert PRODUCER.display_name == "Producer"
    assert PRODUCER.instructions
    assert PRODUCER.output_sections
    assert PRODUCER.grounding_rules
    assert PRODUCER.abstention_rules
    assert ROLES["producer"] is PRODUCER


def test_cinematographer_role_is_registered_and_configured():
    assert CINEMATOGRAPHER.identifier == "cinematographer"
    assert CINEMATOGRAPHER.display_name == "Cinematographer"
    assert "framing" in CINEMATOGRAPHER.instructions
    assert "camera movement" in CINEMATOGRAPHER.instructions
    assert "lens perspective" in CINEMATOGRAPHER.instructions
    assert "lighting" in CINEMATOGRAPHER.instructions
    assert "color" in CINEMATOGRAPHER.instructions
    assert "atmosphere" in CINEMATOGRAPHER.instructions
    assert "visual continuity" in CINEMATOGRAPHER.instructions
    assert CINEMATOGRAPHER.output_sections
    assert CINEMATOGRAPHER.grounding_rules
    assert CINEMATOGRAPHER.abstention_rules
    assert ROLES["cinematographer"] is CINEMATOGRAPHER
