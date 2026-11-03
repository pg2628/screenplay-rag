# Screenplay RAG

A learning-focused retrieval-augmented generation project built around screenplay data. The current proof of concept uses an original screenplay to parse scenes and retrieve relevant scenes for a question. Future Nolan scripts belong under `data/screenplays/`.

## Architecture

```text
screenplay
    -> scene-aware ingestion
    -> TF-IDF retrieval
    -> ranked scenes
```

## Current Status

- Scene-aware screenplay parser
- Heading-aware text prepared for embeddings
- TF-IDF and cosine-similarity retrieval
- Pytest test suite
- Retrieval evaluation workflow with Hit@1 and Hit@3 metrics

The current implementation retrieves screenplay passages. It does not yet generate answers with an LLM.

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the project and development dependencies:

```bash
python -m pip install -e ".[dev]"
```

Run the tests:

```bash
python -m pytest
```

## Usage

Run the interactive retrieval CLI:

```bash
python -m screenplay_rag.retrieve
```

Run the retrieval evaluation workflow:

```bash
python scripts/evaluate_retrieval.py
```

The evaluation workflow uses the cases in `evaluation/retrieval_cases.json` and reports the scenes returned by the TF-IDF retriever.

## Baseline

The current ten-question evaluation reports:

- Hit@1: 60%
- Hit@3: 100%

This means the correct scene is currently retrieved in the top three for every evaluation question, while it is ranked first for six of ten questions.

## Screenplay Copyright

`data/script.txt` is an original screenplay owned by the project author. The screenplay is included for this project and is not offered under an open-source license. It may not be reproduced, redistributed, or reused without permission from the author.

## Roadmap

- Compare semantic embeddings with the TF-IDF baseline
- Add reusable retrieval indexes
- Add grounded answer generation
- Add citations and abstention when the context is insufficient
- Support multiple screenplays
- Add IMSDb ingestion later
