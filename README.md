# Screenplay RAG

A learning-focused retrieval-augmented generation project built around screenplay data. The current proof of concept uses an original screenplay to parse scenes and retrieve relevant scenes for a question. Future screenplays can belong under `data/screenplays/`.

## Architecture

```text
screenplay
    -> scene-aware ingestion
    -> TF-IDF, Ollama semantic, or Chroma retrieval
    -> role-specific system/user prompt
    -> custom or LangChain Ollama answer
```

## Current Status

- Scene-aware screenplay parser
- Heading-aware text prepared for embeddings
- TF-IDF and cosine-similarity retrieval
- Ollama semantic retrieval with `granite-embedding:278m`
- Persistent Chroma retrieval
- Assistant Director, Producer, and Cinematographer roles
- Grounded prompts with screenplay evidence
- Custom and parallel LangChain Ollama answer providers
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
export OLLAMA_MODEL=granite4.2:3b
python assistant.py --role assistant_director
```

Available roles:

```bash
python assistant.py --role assistant_director
python assistant.py --role producer
python assistant.py --role cinematographer
```

Run the retrieval evaluation workflow:

```bash
python scripts/evaluate_retrieval.py
```

The evaluation workflow uses the cases in `evaluation/retrieval_cases.json` and compares TF-IDF, Ollama semantic retrieval, and Chroma retrieval. Set the embedding model before running it:

```bash
export OLLAMA_EMBED_MODEL=granite-embedding:278m
python scripts/evaluate_retrieval.py
```

Build the persistent Chroma index before running the Chroma comparison:

```bash
python scripts/build_chroma_index.py
```

## Retrieval Results

On the current ten-question evaluation set:

| Retriever | Hit@1 | Hit@3 |
| --- | ---: | ---: |
| TF-IDF | 60% | 100% |
| Ollama semantic | 80% | 90% |
| Chroma semantic | 80% | 90% |

Semantic retrieval ranks the expected scene first more often, while TF-IDF retrieves an expected scene within the top three more consistently on this small set. These results are a comparison checkpoint, not a universal conclusion.

## Screenplay Copyright

`data/script.txt` is an original screenplay owned by the project author. The screenplay is included for this project and is not offered under an open-source license. It may not be reproduced, redistributed, or reused without permission from the author.

## Roadmap

- Add source citations to generated answers
- Improve abstention and evidence validation
- Add selectable retrievers and providers to the assistant CLI
- Support multiple screenplays
- Add IMSDb ingestion later
