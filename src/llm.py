"""LLM client (Google Gemini) with caching and token estimation.

Generation is the only step that calls a paid/free-tier API. We:
  * read the API key from the environment or Streamlit secrets,
  * cache responses by prompt hash so repeated questions are free & instant,
  * estimate token usage for cost/usage tracking.
"""

import hashlib
import os
import time
from dataclasses import dataclass
from typing import Dict

from dotenv import load_dotenv

load_dotenv()

# gemini-2.5-flash is available on the free tier (~5 requests/min). Other
# models like gemini-2.0-flash require billing. Override via GEMINI_MODEL.
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Retry settings for transient rate-limit (429) errors.
_MAX_RETRIES = 3
_BACKOFF_SECONDS = 5

# Simple in-process cache: prompt-hash -> LLMResponse.
_CACHE: Dict[str, "LLMResponse"] = {}


class LLMError(RuntimeError):
    """Raised for missing API key or a failed generation call."""


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    cached: bool = False

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars per token)."""
    return max(1, len(text) // 4)


def get_api_key() -> str:
    """Return the Gemini API key from env or Streamlit secrets, or raise."""
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        try:  # Streamlit Cloud stores secrets here
            import streamlit as st

            key = st.secrets.get("GEMINI_API_KEY")
        except Exception:
            key = None
    if not key:
        raise LLMError(
            "No Gemini API key found. Add GEMINI_API_KEY to a .env file "
            "locally, or to Secrets when deployed."
        )
    return key


def _cache_key(prompt: str, model: str) -> str:
    return hashlib.sha256(f"{model}::{prompt}".encode("utf-8")).hexdigest()


def _is_rate_limit(exc: Exception) -> bool:
    return "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc) or "quota" in str(exc).lower()


def _call_with_retry(fn):
    """Call fn(), retrying with exponential backoff on rate-limit (429) errors."""
    last_exc = None
    for attempt in range(_MAX_RETRIES):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - inspected below
            last_exc = exc
            if _is_rate_limit(exc) and attempt < _MAX_RETRIES - 1:
                time.sleep(_BACKOFF_SECONDS * (2**attempt))
                continue
            raise
    raise last_exc  # pragma: no cover


def generate(prompt: str, model: str = DEFAULT_MODEL, temperature: float = 0.2) -> LLMResponse:
    """Generate text for a prompt, using the cache when possible."""
    key = _cache_key(prompt, model)
    if key in _CACHE:
        cached = _CACHE[key]
        return LLMResponse(
            text=cached.text,
            prompt_tokens=cached.prompt_tokens,
            completion_tokens=cached.completion_tokens,
            cached=True,
        )

    api_key = get_api_key()
    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        gen_model = genai.GenerativeModel(model)
        result = _call_with_retry(
            lambda: gen_model.generate_content(
                prompt,
                generation_config={"temperature": temperature},
            )
        )
        text = (result.text or "").strip()
    except Exception as exc:
        if _is_rate_limit(exc):
            raise LLMError(
                "Gemini free-tier rate limit hit. Wait a minute and try again, "
                "or set GEMINI_MODEL to a model with a higher free quota."
            ) from exc
        raise LLMError(f"The language model request failed: {exc}") from exc

    response = LLMResponse(
        text=text,
        prompt_tokens=_estimate_tokens(prompt),
        completion_tokens=_estimate_tokens(text),
        cached=False,
    )
    _CACHE[key] = response
    return response


def generate_stream(prompt: str, model: str = DEFAULT_MODEL, temperature: float = 0.2):
    """Yield generated text incrementally (token streaming).

    Yields string deltas as they arrive. If the response is already cached,
    yields the cached text in one chunk. The full response is cached on
    completion so later identical calls are instant.
    """
    key = _cache_key(prompt, model)
    if key in _CACHE:
        yield _CACHE[key].text
        return

    api_key = get_api_key()
    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        gen_model = genai.GenerativeModel(model)
        stream = gen_model.generate_content(
            prompt,
            generation_config={"temperature": temperature},
            stream=True,
        )
        parts = []
        for chunk in stream:
            delta = getattr(chunk, "text", "") or ""
            if delta:
                parts.append(delta)
                yield delta
    except Exception as exc:
        raise LLMError(f"The language model request failed: {exc}") from exc

    full_text = "".join(parts).strip()
    _CACHE[key] = LLMResponse(
        text=full_text,
        prompt_tokens=_estimate_tokens(prompt),
        completion_tokens=_estimate_tokens(full_text),
        cached=False,
    )
