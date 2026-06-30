"""Unit tests for chunker and vector store (no API key required)."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.chunker import chunk_text
from src.vector_store import VectorStore


def test_chunker_short_text_single_chunk():
    text = "Python developer with FAISS experience"
    chunks = chunk_text(text, chunk_size=120, overlap=30)
    assert chunks == [text]


def test_chunker_splits_long_text_with_overlap():
    words = [f"w{i}" for i in range(300)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size=120, overlap=30)
    # More than one chunk, and each chunk is at most chunk_size words.
    assert len(chunks) > 1
    assert all(len(c.split()) <= 120 for c in chunks)
    # Consecutive chunks overlap (step = 90), so chunk 2 starts at word 90.
    assert chunks[1].split()[0] == "w90"


def test_chunker_empty_returns_empty_list():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_vector_store_returns_most_relevant_chunk():
    chunks = [
        "Experienced in Python, FAISS, and building RAG pipelines.",
        "Hobbies include hiking, cooking, and photography.",
        "Led a team of five engineers on a payments platform.",
    ]
    store = VectorStore(chunks)
    results = store.search("Which vector database skills are listed?", k=1)
    assert results, "expected at least one result"
    assert "FAISS" in results[0].text
