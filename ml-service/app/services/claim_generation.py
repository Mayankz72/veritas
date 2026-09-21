"""Pluggable claim generation, restricted to retrieved evidence only.

The prompt/interface never lets a provider free-associate: every provider
receives the same list of retrieved passages and must produce claims
groundable in them. The default provider (`extractive`) needs no API key at
all, so the pipeline, tests, and CI never depend on a secret being present -
a real LLM provider is an opt-in upgrade via the LLM_PROVIDER env var.
"""

from __future__ import annotations

import difflib
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import httpx

from app.config import get_settings

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
CLAIM_PROMPT_TEMPLATE = (
    "You are extracting factual claims from a research paper section titled "
    '"{topic}". Using ONLY the passages below, write up to {max_claims} short, '
    "concise factual claims. Every claim must be directly supported by at least "
    "one passage - do not add outside knowledge or opinions. Return one claim "
    "per line, with no numbering, bullets, or extra commentary.\n\n"
    "{passages_block}"
)


@dataclass
class GeneratedClaim:
    text: str
    source_chunk_index: int  # best-matching index into the passages list


def _best_matching_passage_index(claim_text: str, passages: list[str]) -> int:
    scores = [
        difflib.SequenceMatcher(None, claim_text.lower(), passage.lower()).ratio()
        for passage in passages
    ]
    return max(range(len(scores)), key=lambda i: scores[i]) if scores else 0


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]


class LLMProvider(ABC):
    name: str
    # False for the keyless extractive baseline: callers that need free-form
    # completion (paper briefs, cross-paper synthesis) fall back to their own
    # extractive path instead of calling `complete`.
    supports_completion: bool = True

    @abstractmethod
    def generate_claims(
        self, topic: str, passages: list[str], max_claims: int
    ) -> list[GeneratedClaim]: ...

    def complete(self, prompt: str, *, json_mode: bool = False) -> str:
        raise NotImplementedError(f"{self.name} provider does not support free-form completion")


class ExtractiveProvider(LLMProvider):
    """No-API-key fallback: lifts a leading sentence from each top passage.

    Deterministic and offline. This is intentionally "dumber" than an LLM -
    it's the always-available baseline the rest of the pipeline (retrieval,
    grounding verification) is built and tested against.
    """

    name = "extractive"
    supports_completion = False

    def generate_claims(
        self, topic: str, passages: list[str], max_claims: int = 5
    ) -> list[GeneratedClaim]:
        claims: list[GeneratedClaim] = []
        for idx, passage in enumerate(passages):
            if len(claims) >= max_claims:
                break
            sentences = _split_sentences(passage)
            if not sentences:
                continue
            sentence = sentences[0]
            if len(sentence) < 40 and len(sentences) > 1:
                sentence = f"{sentence} {sentences[1]}"
            claims.append(GeneratedClaim(text=sentence, source_chunk_index=idx))
        return claims


class _PromptedProvider(LLMProvider):
    """Shared claim generation on top of a provider's `complete`."""

    def generate_claims(
        self, topic: str, passages: list[str], max_claims: int = 5
    ) -> list[GeneratedClaim]:
        prompt = self._build_prompt(topic, passages, max_claims)
        return self._parse_response_lines(self.complete(prompt), passages, max_claims)

    def _parse_response_lines(
        self, raw_text: str, passages: list[str], max_claims: int
    ) -> list[GeneratedClaim]:
        lines = [line.strip("-* \t") for line in raw_text.splitlines() if line.strip()]
        claims = []
        for line in lines[:max_claims]:
            claims.append(
                GeneratedClaim(
                    text=line,
                    source_chunk_index=_best_matching_passage_index(line, passages),
                )
            )
        return claims

    def _build_prompt(self, topic: str, passages: list[str], max_claims: int) -> str:
        passages_block = "\n\n".join(f"[Passage {i}] {p}" for i, p in enumerate(passages))
        return CLAIM_PROMPT_TEMPLATE.format(
            topic=topic, max_claims=max_claims, passages_block=passages_block
        )


class OpenAIProvider(_PromptedProvider):
    name = "openai"

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.model = model

    def complete(self, prompt: str, *, json_mode: bool = False) -> str:
        body: dict = {"model": self.model, "messages": [{"role": "user", "content": prompt}]}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        response = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=body,
            timeout=60.0,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


class AnthropicProvider(_PromptedProvider):
    name = "anthropic"

    def __init__(self, api_key: str, model: str = "claude-haiku-4-5-20251001"):
        self.api_key = api_key
        self.model = model

    def complete(self, prompt: str, *, json_mode: bool = False) -> str:
        response = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01"},
            json={
                "model": self.model,
                "max_tokens": 4096,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=60.0,
        )
        response.raise_for_status()
        return response.json()["content"][0]["text"]


# Gemini answers 503 "high demand" / 429 in bursts, and individual model
# names get retired for new accounts (gemini-2.0/2.5-flash 404 for us), so
# the provider tries a chain of models with short retries instead of
# failing a whole multi-paper analysis on one transient error.
GEMINI_RETRY_STATUSES = {429, 500, 502, 503, 504}
GEMINI_RETRY_DELAYS = (2.0, 5.0)
GEMINI_DEFAULT_MODELS = ("gemini-3.5-flash-lite", "gemini-3.1-flash-lite")


class GeminiProvider(_PromptedProvider):
    name = "gemini"

    def __init__(self, api_key: str, model: str | None = None):
        self.api_key = api_key
        preferred = [model] if model else []
        self.models = preferred + [m for m in GEMINI_DEFAULT_MODELS if m not in preferred]

    def complete(self, prompt: str, *, json_mode: bool = False) -> str:
        # temperature 0: a verification pipeline should return the same brief for the same paper.
        generation_config: dict = {"temperature": 0}
        if json_mode:
            generation_config["responseMimeType"] = "application/json"
        body: dict = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": generation_config,
        }

        last_error: Exception | None = None
        for model in self.models:
            for attempt in range(len(GEMINI_RETRY_DELAYS) + 1):
                try:
                    # Header (not ?key=) auth: the newer "AQ." keys only work this way.
                    response = httpx.post(
                        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                        headers={"x-goog-api-key": self.api_key},
                        json=body,
                        timeout=90.0,
                    )
                    if response.status_code in GEMINI_RETRY_STATUSES:
                        raise httpx.HTTPStatusError(
                            f"{response.status_code} from {model}",
                            request=response.request,
                            response=response,
                        )
                    response.raise_for_status()
                    parts = response.json()["candidates"][0]["content"]["parts"]
                    return "".join(part.get("text", "") for part in parts)
                except (httpx.HTTPStatusError, httpx.TransportError) as exc:
                    last_error = exc
                    retryable = isinstance(exc, httpx.TransportError) or (
                        exc.response.status_code in GEMINI_RETRY_STATUSES
                    )
                    if not retryable:
                        break  # e.g. 404 model retired - go straight to the next model
                    if attempt < len(GEMINI_RETRY_DELAYS):
                        time.sleep(GEMINI_RETRY_DELAYS[attempt])
        raise RuntimeError(f"All Gemini models failed ({self.models}): {last_error}")


class OllamaProvider(_PromptedProvider):
    """Local model via Ollama - no API key, but requires `ollama serve` running."""

    name = "ollama"

    def __init__(self, model: str = "llama3.1", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def complete(self, prompt: str, *, json_mode: bool = False) -> str:
        body: dict = {"model": self.model, "prompt": prompt, "stream": False}
        if json_mode:
            body["format"] = "json"
        response = httpx.post(f"{self.base_url}/api/generate", json=body, timeout=120.0)
        response.raise_for_status()
        return response.json()["response"]


def extract_supporting_quote(claim_text: str, passage_text: str) -> str:
    """Finds the minimal supporting span for a claim within its source
    passage: the exact substring if the claim already is one (true for the
    extractive provider), else the best-matching sentence."""
    if claim_text in passage_text:
        return claim_text

    sentences = _split_sentences(passage_text)
    if not sentences:
        return passage_text[:200]
    return max(
        sentences,
        key=lambda s: difflib.SequenceMatcher(None, claim_text.lower(), s.lower()).ratio(),
    )


def _setting(name: str) -> str | None:
    """Env var first, then the same key from ml-service/.env (pydantic-settings
    reads .env into Settings but does not export it to os.environ)."""
    return os.environ.get(name) or getattr(get_settings(), name.lower(), None)


def get_provider() -> LLMProvider:
    """Selects a provider from LLM_PROVIDER (default: extractive, no key needed)."""
    provider_name = (_setting("LLM_PROVIDER") or "extractive").lower()

    if provider_name == "openai":
        return OpenAIProvider(api_key=_setting("OPENAI_API_KEY") or "")
    if provider_name == "anthropic":
        return AnthropicProvider(api_key=_setting("ANTHROPIC_API_KEY") or "")
    if provider_name == "gemini":
        return GeminiProvider(
            api_key=_setting("GEMINI_API_KEY") or "", model=_setting("GEMINI_MODEL")
        )
    if provider_name == "ollama":
        return OllamaProvider(model=_setting("OLLAMA_MODEL") or "llama3.1")
    return ExtractiveProvider()
