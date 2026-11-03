from pathlib import Path

import pytest

from screenplay_rag.ingest import load_script, parse_screenplay


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "data" / "script.txt"


@pytest.fixture(scope="module")
def scenes():
    return load_script(str(SCRIPT_PATH))


def test_parses_six_scenes(scenes):
    assert len(scenes) == 6


def test_scene_numbers(scenes):
    assert [scene["scene_number"] for scene in scenes] == [1, 2, 3, 4, 5, 6]


def test_scene_headings(scenes):
    headings = [scene["heading"] for scene in scenes]
    assert headings[0] == "EXT. FARM - EVENING"
    assert headings[-1] == "EXT. FARM - EVENING"
    assert headings[1] == "EXT. DESERT CRATER - DAY"


def test_second_scene_contains_expected_text(scenes):
    assert "Day One, 0-0" in scenes[1]["text"]


def test_every_scene_has_embedding_text(scenes):
    for scene in scenes:
        assert "embedding_text" in scene


def test_embedding_text_starts_with_heading(scenes):
    for scene in scenes:
        assert scene["embedding_text"].startswith(scene["heading"] + "\n")


def test_scene_text_excludes_heading(scenes):
    for scene in scenes:
        assert scene["heading"] not in scene["text"]


def test_preamble_is_not_in_first_scene():
    script = "Preamble text\nThis is not a scene.\n\nEXT. FARM - EVENING\nAction.\n"
    scenes = parse_screenplay(script)

    assert len(scenes) == 1
    assert "Preamble text" not in scenes[0]["text"]
    assert "This is not a scene" not in scenes[0]["text"]
