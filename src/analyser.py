"""JD match score and gap analysis, grounded in retrieved resume chunks."""

from dataclasses import dataclass
from typing import List

from src.llm import LLMResponse, generate
from src.vector_store import SearchResult, VectorStore

SCORE_PROMPT = """You are a technical recruiter. Compare the resume excerpts to the job
description. Give a match score from 0 to 100 and a 2-sentence
justification. Base your judgment ONLY on the excerpts provided.

Job description:
{job_description}

Resume excerpts:
{retrieved_chunks}

Respond in this format:
Score: <number>/100
Justification: <2 sentences>"""

GAP_PROMPT = """List the skills, tools, or qualifications mentioned in the job
description that are NOT supported by the resume excerpts below.
Only list genuine gaps. Base it ONLY on the excerpts.

Job description:
{job_description}

Resume excerpts:
{retrieved_chunks}

Respond as a short bullet list of missing items."""


@dataclass
class AnalysisResult:
    score_text: str
    gaps_text: str
    sources: List[SearchResult]
    usage: LLMResponse  # combined token usage for both calls


def _format_chunks(sources: List[SearchResult]) -> str:
    return "\n\n".join(
        f"[Excerpt {i + 1}] {s.text}" for i, s in enumerate(sources)
    )


def analyse(store: VectorStore, job_description: str, k: int = 6) -> AnalysisResult:
    """Score the resume against the JD and list the gaps, with citations."""
    if not job_description or not job_description.strip():
        raise ValueError("Please paste a job description first.")

    sources = store.search(job_description, k=k)
    chunks = _format_chunks(sources)

    score = generate(
        SCORE_PROMPT.format(job_description=job_description, retrieved_chunks=chunks)
    )
    gaps = generate(
        GAP_PROMPT.format(job_description=job_description, retrieved_chunks=chunks)
    )

    combined = LLMResponse(
        text="",
        prompt_tokens=score.prompt_tokens + gaps.prompt_tokens,
        completion_tokens=score.completion_tokens + gaps.completion_tokens,
        cached=score.cached and gaps.cached,
    )
    return AnalysisResult(
        score_text=score.text,
        gaps_text=gaps.text,
        sources=sources,
        usage=combined,
    )
