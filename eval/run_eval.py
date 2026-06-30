"""RAG evaluation harness.

Measures two things on a golden Q&A dataset:

  1. Retrieval hit rate@k  — did the retrieved chunks contain the expected
     evidence keywords? (Pure retrieval quality, no LLM, no cost.)
  2. Answer correctness    — an LLM-as-judge scores each answer against the
     expected answer (0 or 1). Requires GEMINI_API_KEY.

Run:  python eval/run_eval.py            (full eval, calls the LLM)
      python eval/run_eval.py --retrieval-only   (no API key needed)
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.chunker import chunk_text
from src.llm import generate
from src.rag import answer_question
from src.vector_store import VectorStore

HERE = os.path.dirname(__file__)

JUDGE_PROMPT = """You are grading a question-answering system.
Question: {question}
Reference answer: {expected}
System answer: {actual}

Does the system answer match the reference answer in meaning? Reply with
exactly one word: PASS or FAIL."""


def load_store() -> VectorStore:
    with open(os.path.join(HERE, "sample_resume.txt"), encoding="utf-8") as f:
        text = f.read()
    return VectorStore(chunk_text(text))


def retrieval_hit(store: VectorStore, question: str, keywords: list[str], k: int = 4) -> bool:
    """True if expected keywords appear in retrieved chunks (or none expected)."""
    if not keywords:
        return True  # 'not in resume' cases have no evidence to retrieve
    retrieved = " ".join(s.text for s in store.search(question, k=k)).lower()
    return any(kw.lower() in retrieved for kw in keywords)


def judge(question: str, expected: str, actual: str) -> bool:
    verdict = generate(
        JUDGE_PROMPT.format(question=question, expected=expected, actual=actual)
    ).text.strip().upper()
    return verdict.startswith("PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-only", action="store_true")
    args = parser.parse_args()

    with open(os.path.join(HERE, "eval_dataset.json"), encoding="utf-8") as f:
        dataset = json.load(f)

    store = load_store()
    hits = correct = 0
    print(f"{'Q':<55}{'retrieval':<12}{'answer':<8}")
    print("-" * 75)

    for item in dataset:
        q = item["question"]
        hit = retrieval_hit(store, q, item["expected_keywords"])
        hits += int(hit)

        ans_mark = "-"
        if not args.retrieval_only:
            result = answer_question(store, q)
            ok = judge(q, item["expected_answer"], result.answer)
            correct += int(ok)
            ans_mark = "PASS" if ok else "FAIL"

        print(f"{q[:53]:<55}{'HIT' if hit else 'MISS':<12}{ans_mark:<8}")
        if not args.retrieval_only:
            time.sleep(13)  # stay within free-tier ~5 requests/min limit

    n = len(dataset)
    print("-" * 75)
    print(f"Retrieval hit rate: {hits}/{n} = {hits / n:.0%}")
    if not args.retrieval_only:
        print(f"Answer accuracy:    {correct}/{n} = {correct / n:.0%}")


if __name__ == "__main__":
    main()
