"""
retrieve.py — ChromaDB-backed retrieval for LegalLens.

Replaces the original FAISS implementation.

Key improvements over FAISS:
1. Persistent storage — embeddings are written to disk on first load.
   Subsequent startups skip re-embedding and load instantly from disk.
2. Metadata filtering — before searching, we filter by applicable_document_type
   so a cheque_bounce query only searches 9 chunks instead of all 55.
3. Cleaner code — ChromaDB manages the vector-to-chunk mapping internally.

ChromaDB stores its data in chroma_db/ at the project root.
This folder is created automatically on first run. Add it to .gitignore.
"""

import json
import os
from dataclasses import dataclass

import chromadb
from chromadb.utils import embedding_functions

CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
COLLECTION_NAME = "legallens_corpus"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


@dataclass
class RetrievedChunk:
    chunk_id: str
    section_number: str
    full_citation: str
    text: str
    score: float


class CorpusRetriever:
    def __init__(
        self,
        corpus_path: str = None,
        model_name: str = EMBED_MODEL_NAME,
        corpus_dir: str = None,
    ):
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.ef,
            metadata={"hnsw:space": "cosine"},
        )

        if self.collection.count() == 0:
            print("ChromaDB collection empty - loading corpus files...")
            self._load_all_corpora(corpus_dir)
            print(f"Loaded {self.collection.count()} chunks into ChromaDB.")
        else:
            print(f"ChromaDB loaded from disk - {self.collection.count()} chunks ready.")

    def _load_all_corpora(self, corpus_dir: str = None):
        if corpus_dir is None:
            corpus_dir = os.path.join(os.path.dirname(__file__), "..", "corpus")

        corpus_files = [
            f for f in os.listdir(corpus_dir)
            if f.endswith(".json") and f != "source_manifest.json"
        ]

        all_ids, all_texts, all_metadatas = [], [], []

        for filename in corpus_files:
            filepath = os.path.join(corpus_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                chunks = json.load(f)

            for chunk in chunks:
                all_ids.append(chunk["chunk_id"])
                all_texts.append(chunk["text"])
                all_metadatas.append({
                    "act_name": chunk.get("act_name", ""),
                    "section_number": chunk.get("section_number", ""),
                    "section_title": chunk.get("section_title", ""),
                    "full_citation": chunk.get("full_citation", ""),
                    "applicable_document_type": chunk.get("applicable_document_type", ""),
                    "jurisdiction": chunk.get("jurisdiction", "national"),
                    "source_url": chunk.get("source_url", ""),
                })

        batch_size = 50
        for i in range(0, len(all_ids), batch_size):
            self.collection.add(
                ids=all_ids[i:i+batch_size],
                documents=all_texts[i:i+batch_size],
                metadatas=all_metadatas[i:i+batch_size],
            )

    def search(self, query: str, top_k: int = 4, doc_type: str = None) -> list[RetrievedChunk]:
        where_filter = None
        if doc_type:
            where_filter = {"applicable_document_type": {"$eq": doc_type}}

        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, self.collection.count()),
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        retrieved = []
        for i in range(len(results["ids"][0])):
            chunk_id = results["ids"][0][i]
            text = results["documents"][0][i]
            metadata = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            score = 1 - distance

            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    section_number=metadata.get("section_number", ""),
                    full_citation=metadata.get("full_citation", ""),
                    text=text,
                    score=round(score, 4),
                )
            )

        return retrieved

    def get_chunk_by_id(self, chunk_id: str) -> dict | None:
        result = self.collection.get(ids=[chunk_id], include=["documents", "metadatas"])
        if not result["ids"]:
            return None
        return {
            "chunk_id": chunk_id,
            "text": result["documents"][0],
            **result["metadatas"][0],
        }

    def reset_collection(self):
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.ef,
            metadata={"hnsw:space": "cosine"},
        )
        self._load_all_corpora()
        print(f"Collection reset. {self.collection.count()} chunks reloaded.")


if __name__ == "__main__":
    retriever = CorpusRetriever()
    print(f"\nTotal chunks in ChromaDB: {retriever.collection.count()}")
    results = retriever.search(
        "cheque dishonoured insufficient funds punishment",
        top_k=3,
        doc_type="cheque_bounce"
    )
    print("\nTop 3 results (filtered to cheque_bounce):")
    for r in results:
        print(f"  [{r.score:.3f}] {r.full_citation} - {r.text[:60]}...")