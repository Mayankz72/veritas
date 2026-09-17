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
from abc import ABC, abstractmethod
from dataclasses import dataclass

import httpx

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

    @abstractmethod
    def generate_claims(
        self, topic: str, passages: list[str], max_claims: int
    ) -> list[GeneratedClaim]: ...


class ExtractiveProvider(LLMProvider):
    """No-API-key fallback: lifts a leading sentence from each top passage.

    Deterministic and offline. This is intentionally "dumber" than an LLM -
    it's the always-available baseline the rest of the pipeline (retrieval,
    grounding verification) is built and tested against.
    """

    name = "extractive"

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


class _PromptedProviderMixin:
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


class OpenAIProvider(_PromptedProviderMixin, LLMProvider):
    name = "openai"

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.model = model

    def generate_claims(
        self, topic: str, passages: list[str], max_claims: int = 5
    ) -> list[GeneratedClaim]:
        prompt = self._build_prompt(topic, passages, max_claims)
        response = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "messages": [{"role": "user", "content": prompt}]},
            timeout=30.0,
        )
        response.raise_for_status()
        text = response.json()["choices"][0]["message"]["content"]
        return self._parse_response_lines(text, passages, max_claims)


class AnthropicProvider(_PromptedProviderMixin, LLMProvider):
    name = "anthropic"

    def __init__(self, api_key: str, model: str = "claude-haiku-4-5-20251001"):
        self.api_key = api_key
        self.model = model

    def generate_claims(
        self, topic: str, passages: list[str], max_claims: int = 5
    ) -> list[GeneratedClaim]:
        prompt = self._build_prompt(topic, passages, max_claims)
        response = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01"},
            json={
                "model": self.model,
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30.0,
        )
        response.raise_for_status()
        text = response.json()["content"][0]["text"]
        return self._parse_response_lines(text, passages, max_claims)


class GeminiProvider(_PromptedProviderMixin, LLMProvider):
    name = "gemini"

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        self.api_key = api_key
        self.model = model

    def generate_claims(
        self, topic: str, passages: list[str], max_claims: int = 5
    ) -> list[GeneratedClaim]:
        prompt = self._build_prompt(topic, passages, max_claims)
        response = httpx.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
            params={"key": self.api_key},
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=30.0,
        )
        response.raise_for_status()
        text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
        return self._parse_response_lines(text, passages, max_claims)


class OllamaProvider(_PromptedProviderMixin, LLMProvider):
    """Local model via Ollama - no API key, but requires `ollama serve` running."""

    name = "ollama"

    def __init__(self, model: str = "llama3.1", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def generate_claims(
        self, topic: str, passages: list[str], max_claims: int = 5
    ) -> list[GeneratedClaim]:
        prompt = self._build_prompt(topic, passages, max_claims)
        response = httpx.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False},
            timeout=60.0,
        )
        response.raise_for_status()
        text = response.json()["response"]
        return self._parse_response_lines(text, passages, max_claims)


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


def get_provider() -> LLMProvider:
    """Selects a provider from LLM_PROVIDER (default: extractive, no key needed)."""
    provider_name = os.getenv("LLM_PROVIDER", "extractive").lower()

    if provider_name == "openai":
        return OpenAIProvider(api_key=os.environ["OPENAI_API_KEY"])
    if provider_name == "anthropic":
        return AnthropicProvider(api_key=os.environ["ANTHROPIC_API_KEY"])
    if provider_name == "gemini":
        return GeminiProvider(api_key=os.environ["GEMINI_API_KEY"])
    if provider_name == "ollama":
        return OllamaProvider(model=os.getenv("OLLAMA_MODEL", "llama3.1"))
    return ExtractiveProvider()
