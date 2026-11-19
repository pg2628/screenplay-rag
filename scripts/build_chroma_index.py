"""Build a persistent Chroma index from screenplay scenes."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from screenplay_rag.ingest import load_script


SCRIPT_PATH = REPO_ROOT / "data" / "script.txt"
PERSIST_DIRECTORY = REPO_ROOT / "data" / "chroma"
COLLECTION_NAME = "screenplay_scenes"


def build_index(
    script_path: Path = SCRIPT_PATH,
    persist_directory: Path = PERSIST_DIRECTORY,
) -> None:
    """Embed screenplay scenes and persist them in Chroma."""
    scenes = load_script(str(script_path))
    embeddings = OllamaEmbeddings(
        model="granite-embedding:278m",
        base_url="http://localhost:11434",
    )
    documents = [
        Document(
            page_content=str(scene["embedding_text"]),
            metadata={
                "scene_number": scene["scene_number"],
                "heading": scene["heading"],
                "source": str(script_path),
                "text": scene["text"],
            },
        )
        for scene in scenes
    ]
    ids = [f"{script_path}:{scene['scene_number']}" for scene in scenes]
    Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        ids=ids,
        collection_name=COLLECTION_NAME,
        persist_directory=str(persist_directory),
    )
    print(f"Indexed {len(documents)} scenes in {persist_directory}")


if __name__ == "__main__":
    build_index()