from screenplay_rag.chroma_retrieve import ChromaRetriever


class FakeDocument:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata


class FakeVectorStore:
    def __init__(self):
        self.calls = []

    def similarity_search_with_relevance_scores(self, query, k):
        self.calls.append((query, k))
        return [
            (
                FakeDocument(
                    "EXT. FARM - EVENING\nThe farmer meets the general.",
                    {
                        "scene_number": 1,
                        "heading": "EXT. FARM - EVENING",
                        "source": "data/script.txt",
                        "text": "The farmer meets the general.",
                    },
                ),
                0.91,
            )
        ]


def test_chroma_retriever_maps_store_results_to_scene_shape():
    vector_store = FakeVectorStore()
    retriever = ChromaRetriever(vector_store=vector_store)

    results = retriever("Who meets the general?", [], top_k=1)

    assert vector_store.calls == [("Who meets the general?", 1)]
    assert results == [
        {
            "scene_number": 1,
            "heading": "EXT. FARM - EVENING",
            "source": "data/script.txt",
            "text": "The farmer meets the general.",
            "embedding_text": "EXT. FARM - EVENING\nThe farmer meets the general.",
            "score": 0.91,
        }
    ]