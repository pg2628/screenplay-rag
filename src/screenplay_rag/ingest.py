"""Small, scene-aware parser for a screenplay."""

from pathlib import Path
import re

SCENE_HEADING_RE = re.compile(
    r"^\s*(?:INT\.|EXT\.)(?:/(?:INT\.|EXT\.))?.*$",
    re.IGNORECASE,
)


def parse_screenplay(script: str) -> list[dict[str, object]]:
    """Split screenplay text into scenes without changing the scene text."""
    scenes = []
    current_heading = None
    current_text = []

    def save_scene() -> None:
        if current_heading is not None:
            scene_text = "".join(current_text)
            scenes.append(
                {
                    "scene_number": len(scenes) + 1,
                    "heading": current_heading,
                    "text": scene_text,
                    "embedding_text": f"{current_heading}\n{scene_text}",
                }
            )

    for line in script.splitlines(keepends=True):
        if SCENE_HEADING_RE.match(line):
            save_scene()
            current_heading = line.strip()
            current_text = []
        elif current_heading is not None:
            current_text.append(line)

    save_scene()
    return scenes


def load_script(path: str = "data/script.txt") -> list[dict[str, object]]:
    """Read a screenplay and return its scene chunks."""
    return parse_screenplay(Path(path).read_text(encoding="utf-8"))


if __name__ == "__main__":
    for scene in load_script():
        print(scene)
