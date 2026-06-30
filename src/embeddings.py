"""Local text embeddings using sentence-transformers.

The model runs locally (CPU), so embedding costs nothing and needs no API
key. The model is loaded once and cached for the life of the process.
"""

from functools import lru_cache
from typing import List

import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _get_model():
    """Load and cache the embedding model (lazy import keeps startup fast)."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME)


def embed_texts(texts: List[str]) -> np.ndarray:
    """Embed a list of texts into a 2D float32 array (n_texts x dim).

    Vectors are L2-normalised so that inner-product search in FAISS is
    equivalent to cosine similarity.
    """
    if not texts:
        return np.zeros((0, 384), dtype="float32")

    model = _get_model()
    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vectors.astype("float32")


def embed_text(text: str) -> np.ndarray:
    """Embed a single text into a 1D float32 vector."""
    return embed_texts([text])[0]
