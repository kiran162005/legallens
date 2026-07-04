"""
retrieve.py — loads the legal corpus, builds a dense embedding index,
and retrieves the most relevant statutory chunks for a given query.

Design notes:
- One corpus = one JSON file of atomic, section-level chunks (see corpus/*.json).
- We embed `text` only (not metadata) so retrieval is grounded in actual statutory
  language, not in act names or section numbers that could bias matching.
- With ~9 chunks today, a single dense FAISS search is sufficient. Hybrid
  BM25 + dense + RRF should be added once a document type has 30+ chunks,
  where lexical precision on exact terms/section numbers starts to matter.
"""

import json
import os
from dataclasses import dataclass

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

CORPUS_PATH = os.path.join(os.path.dirname(__file__), "..", "corpus", "ni_act_cheque_bounce.json")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, good enough for this corpus size


@dataclass
class RetrievedChunk:
    chunk_id: str
    section_number: str
    full_citation: str
    text: str
    score: float


class CorpusRetriever:
    def __init__(self, corpus_path: str = CORPUS_PATH, model_name: str = EMBED_MODEL_NAME):
        with open(corpus_path, "r") as f:
            self.chunks = json.load(f)

        if not self.chunks:
            raise ValueError(f"No chunks loaded from {corpus_path}")

        self.model = SentenceTransformer(model_name)
        self._build_index()

    def _build_index(self):
        texts = [c["text"] for c in self.chunks]
        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # cosine sim via inner product on normalized vectors
        self.index.add(embeddings.astype(np.float32))

    def search(self, query: str, top_k: int = 4) -> list[RetrievedChunk]:
        query_vec = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        scores, indices = self.index.search(query_vec.astype(np.float32), top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            results.append(
                RetrievedChunk(
                    chunk_id=chunk["chunk_id"],
                    section_number=chunk["section_number"],
                    full_citation=chunk["full_citation"],
                    text=chunk["text"],
                    score=float(score),
                )
            )
        return results

    def get_chunk_by_id(self, chunk_id: str) -> dict | None:
        for c in self.chunks:
            if c["chunk_id"] == chunk_id:
                return c
        return None


if __name__ == "__main__":
    # quick manual smoke test
    retriever = CorpusRetriever()
    test_query = "cheque was dishonoured due to insufficient funds, what is the punishment"
    results = retriever.search(test_query, top_k=3)
    for r in results:
        print(f"[{r.score:.3f}] {r.full_citation} — {r.text[:80]}...")
