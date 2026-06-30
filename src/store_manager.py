"""In-memory registry of indexed resumes for the stateless API.

A client uploads a resume once (`POST /index`) and gets a `doc_id`; later
chat/analyse calls reference that id. Stores are kept in memory with a simple
LRU-style cap so the process doesn't grow unbounded.
"""

import threading
import uuid
from collections import OrderedDict
from typing import Optional

from src.chunker import chunk_text
from src.vector_store import VectorStore

_MAX_DOCS = 50
_lock = threading.Lock()
_stores: "OrderedDict[str, VectorStore]" = OrderedDict()


def index_resume(resume_text: str) -> str:
    """Chunk + embed + index a resume; return a doc_id handle."""
    chunks = chunk_text(resume_text)
    store = VectorStore(chunks)
    doc_id = uuid.uuid4().hex[:12]
    with _lock:
        _stores[doc_id] = store
        _stores.move_to_end(doc_id)
        while len(_stores) > _MAX_DOCS:
            _stores.popitem(last=False)  # evict oldest
    return doc_id


def get_store(doc_id: str) -> Optional[VectorStore]:
    with _lock:
        store = _stores.get(doc_id)
        if store is not None:
            _stores.move_to_end(doc_id)
        return store
