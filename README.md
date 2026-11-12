# Screenplay RAG

A learning-focused retrieval-augmented generation project built around screenplay data. The current proof of concept uses an original screenplay to parse scenes and retrieve relevant scenes for a question. Future screenplays can belong under `data/screenplays/`.

## Architecture

```text
screenplay
    -> scene-aware ingestion
    -> TF-IDF retrieval
    -> role-specific system/user prompt
    -> local Ollama answer
```

## Current Status

- Scene-aware screenplay parser
- Heading-aware text prepared for embeddings
- TF-IDF and cosine-similarity retrieval
- Assistant Director and Producer roles
- Grounded prompts with screenplay evidence
- Local Ollama answer generation
- Pytest test suite
- Retrieval evaluation workflow with Hit@1 and Hit@3 metrics

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the runtime and test dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the tests:

```bash
python -m pytest
```

## Usage

Set the Ollama model and run the role-based assistant:

```bash
export OLLAMA_MODEL=llama3.2
python assistant.py --role assistant_director
```

Available roles:

```bash
python assistant.py --role assistant_director
python assistant.py --role producer
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

- Add source citations to generated answers
- Improve abstention and evidence validation
- Compare semantic embeddings with the TF-IDF baseline
- Add reusable retrieval indexes
- Support multiple screenplays
- Add IMSDb ingestion later
