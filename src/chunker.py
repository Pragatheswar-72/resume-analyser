"""Split resume text into overlapping word chunks.

Smaller, focused chunks improve retrieval quality and let us cite specific
parts of the resume. Overlap keeps context from being split awkwardly at
chunk boundaries.
"""

from typing import List


def chunk_text(
    text: str, chunk_size: int = 120, overlap: int = 30
) -> List[str]:
    """Split text into overlapping chunks measured in words.

    Args:
        text: The full text to split.
        chunk_size: Target number of words per chunk.
        overlap: Number of words shared between consecutive chunks.

    Returns:
        A list of chunk strings (never empty unless the input is empty).
    """
    if not text or not text.strip():
        return []

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = text.split()
    if len(words) <= chunk_size:
        return [" ".join(words)]

    step = chunk_size - overlap
    chunks: List[str] = []
    for start in range(0, len(words), step):
        window = words[start : start + chunk_size]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + chunk_size >= len(words):
            break

    return chunks
