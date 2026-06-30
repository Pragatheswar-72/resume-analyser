"""FastAPI backend for the Resume Analyser.

Exposes the same RAG/analysis logic as the Streamlit UI over a clean JSON
API, so the service can be consumed by any frontend, mobile app, or another
service. Endpoints:

    GET  /health           liveness check
    POST /index            upload a PDF resume -> returns a doc_id
    POST /chat             grounded answer + citations for a question
    POST /chat/stream      same, streamed token-by-token (SSE-style)
    POST /analyse          JD match score + gap analysis + citations
"""

import time

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.analyser import analyse
from src.llm import LLMError
from src.logging_config import get_logger, log_event
from src.pdf_loader import extract_text
from src.rag import answer_question, stream_answer
from src.store_manager import get_store, index_resume

logger = get_logger()
app = FastAPI(
    title="Resume Analyser API",
    description="RAG-powered resume ↔ job-description matching with citations.",
    version="1.0.0",
)


# --- Schemas -----------------------------------------------------------------
class IndexResponse(BaseModel):
    doc_id: str
    characters: int
    chunks: int


class Source(BaseModel):
    index: int
    text: str
    score: float


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cached: bool


class ChatRequest(BaseModel):
    doc_id: str = Field(..., description="doc_id returned by /index")
    question: str
    k: int = Field(4, ge=1, le=10)


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    usage: Usage


class AnalyseRequest(BaseModel):
    doc_id: str
    job_description: str
    k: int = Field(6, ge=1, le=12)


class AnalyseResponse(BaseModel):
    score: str
    gaps: str
    sources: list[Source]
    usage: Usage


# --- Helpers -----------------------------------------------------------------
def _sources(items) -> list[Source]:
    return [Source(index=s.index, text=s.text, score=s.score) for s in items]


def _usage(u) -> Usage:
    return Usage(
        prompt_tokens=u.prompt_tokens,
        completion_tokens=u.completion_tokens,
        total_tokens=u.total_tokens,
        cached=u.cached,
    )


def _require_store(doc_id: str):
    store = get_store(doc_id)
    if store is None:
        raise HTTPException(status_code=404, detail="Unknown doc_id. Call /index first.")
    return store


# --- Observability middleware -----------------------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
    log_event(
        logger,
        "request",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        latency_ms=elapsed_ms,
    )
    return response


# --- Routes ------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/index", response_model=IndexResponse)
async def index(file: UploadFile = File(...)):
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")
    try:
        text = extract_text(file.file)
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err))
    doc_id = index_resume(text)
    store = _require_store(doc_id)
    log_event(logger, "indexed", doc_id=doc_id, chars=len(text), chunks=len(store.chunks))
    return IndexResponse(doc_id=doc_id, characters=len(text), chunks=len(store.chunks))


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    store = _require_store(req.doc_id)
    try:
        result = answer_question(store, req.question, k=req.k)
    except LLMError as err:
        raise HTTPException(status_code=502, detail=str(err))
    return ChatResponse(
        answer=result.answer, sources=_sources(result.sources), usage=_usage(result.usage)
    )


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    store = _require_store(req.doc_id)
    sources, token_iter = stream_answer(store, req.question, k=req.k)

    def event_stream():
        # First line: the citations as a JSON event, then the streamed text.
        import json

        yield "event: sources\ndata: " + json.dumps(
            [s.__dict__ for s in sources]
        ) + "\n\n"
        try:
            for delta in token_iter:
                yield f"data: {delta}\n\n"
        except LLMError as err:
            yield f"event: error\ndata: {err}\n\n"
        yield "event: done\ndata: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/analyse", response_model=AnalyseResponse)
def analyse_endpoint(req: AnalyseRequest):
    store = _require_store(req.doc_id)
    try:
        result = analyse(store, req.job_description, k=req.k)
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err))
    except LLMError as err:
        raise HTTPException(status_code=502, detail=str(err))
    return AnalyseResponse(
        score=result.score_text,
        gaps=result.gaps_text,
        sources=_sources(result.sources),
        usage=_usage(result.usage),
    )
