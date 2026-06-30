"""Resume Analyser — Streamlit UI.

RAG web app: upload a resume (PDF) and paste a job description to get a
match score, a gap analysis, and a chat assistant grounded in the resume,
with citations to the exact resume excerpts used.
"""

import streamlit as st

from src.analyser import analyse
from src.llm import LLMError
from src.pdf_loader import extract_text
from src.rag import answer_question
from src.vector_store import SearchResult, VectorStore

st.set_page_config(page_title="Resume Analyser", page_icon="📄", layout="wide")

# --- Session state -----------------------------------------------------------
st.session_state.setdefault("total_tokens", 0)
st.session_state.setdefault("messages", [])  # chat history: list of dicts


@st.cache_resource(show_spinner="Indexing resume…")
def build_store(resume_text: str) -> VectorStore:
    """Chunk + embed + index the resume. Cached per unique resume text."""
    from src.chunker import chunk_text

    chunks = chunk_text(resume_text)
    return VectorStore(chunks)


def render_citations(sources: list[SearchResult]) -> None:
    if not sources:
        return
    with st.expander(f"📎 Citations ({len(sources)} excerpts used)"):
        for i, s in enumerate(sources, 1):
            st.markdown(f"**Excerpt {i}** · similarity `{s.score:.2f}`")
            st.write(s.text)


def add_tokens(usage) -> None:
    if not usage.cached:
        st.session_state["total_tokens"] += usage.total_tokens


# --- Sidebar -----------------------------------------------------------------
with st.sidebar:
    st.title("📄 Resume Analyser")
    uploaded_pdf = st.file_uploader("Upload resume (PDF)", type=["pdf"])
    job_description = st.text_area(
        "Paste the job description", height=180, placeholder="Paste the JD here..."
    )
    st.divider()
    st.metric("Approx tokens used (this session)", f"{st.session_state['total_tokens']:,}")
    with st.expander("How it works"):
        st.markdown(
            "1. Resume text is chunked and embedded **locally** "
            "(sentence-transformers).\n"
            "2. Chunks are stored in a **FAISS** vector index.\n"
            "3. Your question / the JD retrieves the top matching chunks (**RAG**).\n"
            "4. **Gemini** answers using only those chunks and cites them."
        )

# --- Build / cache the resume index -----------------------------------------
store = None
if uploaded_pdf is not None:
    try:
        resume_text = extract_text(uploaded_pdf)
        store = build_store(resume_text)
        st.success(f"Resume indexed — {len(resume_text):,} characters.")
    except ValueError as err:
        st.error(str(err))

if store is None:
    st.info("👈 Upload a PDF resume in the sidebar to get started.")
    st.stop()

# --- Tabs --------------------------------------------------------------------
chat_tab, analyse_tab = st.tabs(["💬 Chat with resume", "🎯 Match & Gaps"])

with chat_tab:
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            render_citations(msg.get("sources", []))

    if question := st.chat_input("Ask about the resume…"):
        st.session_state["messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking…"):
                    result = answer_question(store, question)
                add_tokens(result.usage)
                st.markdown(result.answer)
                render_citations(result.sources)
                st.session_state["messages"].append(
                    {
                        "role": "assistant",
                        "content": result.answer,
                        "sources": result.sources,
                    }
                )
            except LLMError as err:
                st.error(str(err))
        st.rerun()

with analyse_tab:
    st.write("Score the resume against the pasted job description and surface gaps.")
    if st.button("Analyse against JD", type="primary"):
        try:
            with st.spinner("Analysing…"):
                result = analyse(store, job_description)
            add_tokens(result.usage)
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Match score")
                st.markdown(result.score_text)
            with col2:
                st.subheader("Gaps")
                st.markdown(result.gaps_text)
            render_citations(result.sources)
        except ValueError as err:
            st.warning(str(err))
        except LLMError as err:
            st.error(str(err))
