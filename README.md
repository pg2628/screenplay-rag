# Screenplay RAG

A learning-focused screenplay RAG project showcase built around scene-aware retrieval, reranking, role-specific prompts, local generation, and citation grounding.

## Architecture

```text
screenplay
    -> scene-aware ingestion
    -> TF-IDF, hybrid, Ollama semantic, or Chroma candidate retrieval
    -> optional TF-IDF reranking
    -> role-specific system/user prompt
    -> custom or LangChain Ollama answer
    -> citation grounding and abstention
```

## Current Status

- Scene-aware screenplay parser
- Heading-aware text prepared for embeddings
- TF-IDF and cosine-similarity retrieval
- Ollama semantic retrieval with `granite-embedding:278m`
- Hybrid TF-IDF and semantic score fusion
- Optional TF-IDF reranking over retrieved candidates
- Persistent Chroma retrieval
- Assistant Director, Producer, and Cinematographer roles
- Grounded prompts with screenplay evidence
- Citation validation and grounded abstention
- Custom and parallel LangChain Ollama answer providers
- Pytest test suite
- Retrieval evaluation workflow with Hit@1 and Hit@3 metrics

## Quick Demo

Start with the default TF-IDF retriever and custom Ollama provider:

```bash
export OLLAMA_MODEL=granite4.2:3b
screenplay-rag --role producer
```

Run the completed two-stage hybrid retrieval and reranking path:

```bash
screenplay-rag --role producer --retriever hybrid --reranker tfidf --candidate-count 10
```

The reranker reduces broader candidate scenes to the final `top_k` evidence used in the prompt. Answers cite retrieved scenes and abstain when evidence or citations are insufficient.

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the project and its development dependencies in editable mode:

```bash
python -m pip install -e ".[dev]"
```

Run the tests:

```bash
python -m pytest
```

## Usage

Set the Ollama model and run the role-based assistant:

```bash
export OLLAMA_MODEL=granite4.2:3b
screenplay-rag --role assistant_director
```

The assistant defaults to the custom Ollama provider and the TF-IDF baseline.
The custom provider waits up to 10 minutes for a response by default; set
`OLLAMA_TIMEOUT` to change the timeout in seconds.
Choose alternate pipeline components explicitly when comparing implementations:

```bash
screenplay-rag --role producer --retriever semantic --provider langchain
screenplay-rag --role cinematographer --retriever chroma --provider custom
screenplay-rag --role assistant_director --retriever hybrid --provider langchain
screenplay-rag --role producer --retriever hybrid --reranker tfidf --candidate-count 10
```

The assistant can also be started with `python -m screenplay_rag`.

Available retrievers are `tfidf`, `semantic`, `chroma`, and `hybrid`. Available
providers are `custom` and `langchain`. Reranking is optional and is enabled with
`--reranker tfidf`; `--candidate-count` controls the broader retrieval set.
Build the Chroma index before selecting `--retriever chroma`. Hybrid retrieval
uses equal lexical and semantic weight by default; set `RAG_SEMANTIC_WEIGHT` to
a value from `0.0` to `1.0` to adjust it:

```bash
export RAG_SEMANTIC_WEIGHT=0.7
```

Available roles:

```bash
screenplay-rag --role assistant_director
screenplay-rag --role producer
screenplay-rag --role cinematographer
```

Run the retrieval evaluation workflow:

```bash
python scripts/evaluate_retrieval.py
```

The evaluation workflow uses the cases in `evaluation/retrieval_cases.json` and compares TF-IDF, Ollama semantic, hybrid, hybrid-plus-reranking, and available Chroma retrieval. Set the embedding model before running it:

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

## Future Extensions

- Add answer-level evaluation for citation support and abstention quality
- Compare cross-encoder reranking with the current TF-IDF reranker
- Support multiple screenplays and screenplay-level metadata
- Add workflow orchestration after the linear RAG pipeline is measured
- Evaluate whether screened-source or API ingestion improves retrieval
