"""Run the screenplay assistant directly from the repository root."""

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).parent / "src"))

from screenplay_rag.assistant import main


if __name__ == "__main__":
    main()