# Resume Bullets & Talking Points — Resume Analyser

> Pick the version that fits your resume's space. Keep the metrics; recruiters
> scan for them.
>
> - **Live app:** https://resume-analyser-rnnelecskssselehhqkshp.streamlit.app/
> - **GitHub:** https://github.com/Pragatheswar-72/resume-analyser

## Project header (for the Projects section)

**Resume Analyser — RAG-powered Resume ↔ Job Matching**
Live: resume-analyser-rnnelecskssselehhqkshp.streamlit.app · Code: github.com/Pragatheswar-72/resume-analyser
*Python, FastAPI, FAISS, sentence-transformers, Google Gemini, Docker, Streamlit*

## Bullet options

**Full (2 lines):**
- Built a production RAG service that scores resume–job fit and answers questions
  with citations to the exact resume lines (zero hallucination), using local
  sentence-transformers embeddings + a FAISS vector index and Google Gemini.
- Shipped a FastAPI backend with streaming responses, an evaluation harness
  (100% retrieval hit-rate, LLM-as-judge), structured logging, rate-limit
  backoff, Docker packaging, and a pytest CI pipeline.

**Compact (1 line):**
- Built and deployed a RAG resume-analysis service (FastAPI, FAISS,
  sentence-transformers, Gemini) with citation-grounded answers, streaming,
  an evaluation harness, Docker, and CI — running at $0 on free-tier infra.

**Impact-first (1 line):**
- Designed a citation-grounded RAG pipeline (FAISS + Gemini) that prevents LLM
  hallucination by answering only from retrieved resume chunks; exposed it via a
  tested, containerized FastAPI service with streaming and evaluation.

## Skills line (add these keywords to your Skills section)

RAG · LLMs · Prompt Engineering · Embeddings · Vector Databases (FAISS) ·
sentence-transformers · Google Gemini · FastAPI · REST APIs · Streaming (SSE) ·
LLM Evaluation (LLM-as-judge) · Docker · pytest · CI/CD · Python

## 30-second pitch (for interviews / cover note)

"I built a resume analyser that uses retrieval-augmented generation: it chunks a
resume, embeds it locally with sentence-transformers, stores the vectors in
FAISS, and retrieves the most relevant chunks for any question or job
description. The LLM answers *only* from those retrieved chunks and cites them,
so it can't hallucinate. I exposed it as a FastAPI service with streaming, wrote
an evaluation harness to measure retrieval and answer quality, and packaged it
with Docker and CI — all on free-tier infrastructure."

## Be ready to explain (common interview questions)

- **What is RAG and why use it?** Retrieve relevant text, put it in the prompt,
  so answers are grounded in real data instead of the model's memory.
- **What is an embedding?** A vector capturing text meaning; similar meanings sit
  close together, enabling semantic search.
- **What does FAISS do?** Stores embeddings and finds the most similar ones fast
  (here, cosine similarity via normalized inner-product).
- **Why chunk the resume?** Retrieval and citations work better on small focused
  pieces than on one giant blob.
- **How do you prevent hallucination?** The prompt restricts the model to the
  retrieved excerpts and the UI shows them as citations.
- **How did you evaluate it?** Retrieval hit-rate@k on a golden dataset plus an
  LLM-as-judge scoring answers against reference answers.
- **How did you keep cost at $0?** Local embeddings, free LLM tier, and response
  caching to avoid repeat calls.
