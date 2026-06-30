"""API endpoint tests using FastAPI's TestClient.

These cover routing, validation, and error handling without calling the LLM
(the /chat and /analyse happy paths need an API key, so we assert the
404/validation behaviour that is deterministic and free)."""

import io
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas

from api import app

client = TestClient(app)


def _make_pdf_bytes() -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.drawString(72, 720, "Test Candidate - Python and FAISS engineer")
    c.save()
    return buf.getvalue()


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_index_returns_doc_id():
    pdf = _make_pdf_bytes()
    r = client.post(
        "/index", files={"file": ("resume.pdf", pdf, "application/pdf")}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["doc_id"]
    assert body["characters"] > 0
    assert body["chunks"] >= 1


def test_chat_unknown_doc_id_is_404():
    r = client.post("/chat", json={"doc_id": "doesnotexist", "question": "hi"})
    assert r.status_code == 404


def test_chat_validation_error_is_422():
    # Missing required 'question' field.
    r = client.post("/chat", json={"doc_id": "abc"})
    assert r.status_code == 422
