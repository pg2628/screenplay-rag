from screenplay_rag.roles import ASSISTANT_DIRECTOR, ROLES


def test_assistant_director_role_has_expected_configuration():
    assert ASSISTANT_DIRECTOR.identifier == "assistant_director"
    assert ASSISTANT_DIRECTOR.display_name == "Assistant Director"
    assert ASSISTANT_DIRECTOR.instructions
    assert ASSISTANT_DIRECTOR.output_sections
    assert ASSISTANT_DIRECTOR.grounding_rules
    assert ASSISTANT_DIRECTOR.abstention_rules


def test_assistant_director_is_registered_by_identifier():
    assert ROLES["assistant_director"] is ASSISTANT_DIRECTOR
