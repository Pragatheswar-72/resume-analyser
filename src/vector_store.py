"""FAISS-backed vector store over resume chunks.

Stores chunk embeddings in an in-memory FAISS index and returns the most
similar chunks for a query. Uses inner-product search on normalised vectors,
which is equivalent to cosine similarity.
"""

from dataclasses import dataclass
from typing import List

import numpy as np

from src.embeddings import embed_text, embed_texts


@dataclass
class SearchResult:
    """A retrieved chunk with its position and similarity score."""

    index: int          # position of the chunk in the original list
    text: str           # the chunk text
    score: float        # cosine similarity (higher is more relevant)


class VectorStore:
    """In-memory FAISS index over a list of text chunks."""

    def __init__(self, chunks: List[str]):
        import faiss

        if not chunks:
            raise ValueError("Cannot build a vector store with no chunks.")

        self.chunks = chunks
        vectors = embed_texts(chunks)
        self.dim = vectors.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(vectors)

    def search(self, query: str, k: int = 4) -> List[SearchResult]:
        """Return the top-k chunks most relevant to the query."""
        if not query or not query.strip():
            return []

        k = min(k, len(self.chunks))
        query_vec = embed_text(query).reshape(1, -1)
        scores, indices = self.index.search(query_vec, k)

        results: List[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS pads with -1 when fewer than k results
                continue
            results.append(
                SearchResult(index=int(idx), text=self.chunks[idx], score=float(score))
            )
        return results
