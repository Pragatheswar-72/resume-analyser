"""RAG chat: retrieve relevant resume chunks, then answer from them only.

The answer is grounded in the retrieved excerpts and the excerpts are
returned alongside it so the UI can show citations.
"""

from dataclasses import dataclass
from typing import List

from typing import Iterator, Tuple

from src.llm import LLMResponse, generate, generate_stream
from src.vector_store import SearchResult, VectorStore

CHAT_PROMPT = """You are a resume assistant. Answer the user's question USING ONLY the
resume excerpts provided below. If the answer is not in the excerpts,
say "That information is not in the resume." Do not invent anything.

Resume excerpts:
{retrieved_chunks}

Question: {user_question}

Answer concisely, then list which excerpt number(s) you used."""


@dataclass
class RAGResult:
    answer: str
    sources: List[SearchResult]
    usage: LLMResponse


def _format_chunks(sources: List[SearchResult]) -> str:
    return "\n\n".join(
        f"[Excerpt {i + 1}] {s.text}" for i, s in enumerate(sources)
    )


def answer_question(store: VectorStore, question: str, k: int = 4) -> RAGResult:
    """Retrieve top-k chunks for the question and answer grounded in them."""
    sources = store.search(question, k=k)
    if not sources:
        # No retrievable content — return a grounded refusal without an API call.
        usage = LLMResponse(text="", prompt_tokens=0, completion_tokens=0, cached=True)
        return RAGResult(
            answer="That information is not in the resume.",
            sources=[],
            usage=usage,
        )

    prompt = CHAT_PROMPT.format(
        retrieved_chunks=_format_chunks(sources), user_question=question
    )
    usage = generate(prompt)
    return RAGResult(answer=usage.text, sources=sources, usage=usage)


def stream_answer(
    store: VectorStore, question: str, k: int = 4
) -> Tuple[List[SearchResult], Iterator[str]]:
    """Retrieve sources, then return them with a generator that streams the answer.

    Sources are resolved up front (local, instant) so the UI can show citations
    while the answer streams in.
    """
    sources = store.search(question, k=k)
    if not sources:
        def _refusal() -> Iterator[str]:
            yield "That information is not in the resume."

        return [], _refusal()

    prompt = CHAT_PROMPT.format(
        retrieved_chunks=_format_chunks(sources), user_question=question
    )
    return sources, generate_stream(prompt)
