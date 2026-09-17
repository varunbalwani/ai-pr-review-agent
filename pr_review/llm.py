"""LLM provider port. The Protocol is what lets a hosted provider replace
Ollama later without touching any call site (decision: LLM provider: local
Ollama, not Anthropic). OpenAICompatibleProvider is that swap: a hosted
OpenAI-compatible endpoint (e.g. requesty.ai), used instead of Ollama once
local inference proved too slow to serve 4 concurrent lens calls."""

import asyncio
import os
import time
from typing import Literal, Protocol, TypedDict

import httpx

Role = Literal["security", "quality", "test_coverage", "docs", "verify"]

_ROLE_ENV_KEY: dict[Role, str] = {
    "security": "AGENT_MODEL_SECURITY",
    "quality": "AGENT_MODEL_QUALITY",
    "test_coverage": "AGENT_MODEL_TEST_COVERAGE",
    "docs": "AGENT_MODEL_DOCS",
    "verify": "AGENT_MODEL_VERIFY",
}
_DEFAULT_MODEL = "qwen3:8b"


def model_for(role: Role, *, default: str = _DEFAULT_MODEL) -> str:
    """Resolve a role to its configured model, never hardcoded per-call."""
    return os.environ.get(_ROLE_ENV_KEY[role], default)


class LLMProvider(Protocol):
    async def complete(self, role: Role, prompt: str, *, system: str | None = None) -> str: ...


class LLMProviderWithUsage(Protocol):
    default_model: str

    async def complete_with_usage(
        self, role: Role, prompt: str, *, system: str | None = None
    ) -> tuple[str, "LLMUsage"]: ...


class LLMUsage(TypedDict):
    input_tokens: int
    output_tokens: int
    latency_ms: int


class OllamaProvider:
    def __init__(
        self,
        base_url: str | None = None,
        transport: httpx.BaseTransport | None = None,
    ):
        self.base_url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        self._transport = transport  # injectable for tests; None uses the real network
        self.default_model = _DEFAULT_MODEL  # audit-log accuracy: AuditingLLMProvider reads this

    async def complete(self, role: Role, prompt: str, *, system: str | None = None) -> str:
        content, _usage = await self.complete_with_usage(role, prompt, system=system)
        return content

    async def complete_with_usage(
        self, role: Role, prompt: str, *, system: str | None = None
    ) -> tuple[str, LLMUsage]:
        """Same call as complete(), but also returns token/latency usage
        (FR-9's audit trail needs this; the Protocol's complete() doesn't
        carry it so every existing call site stays untouched)."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        start = time.monotonic()
        async with httpx.AsyncClient(
            base_url=self.base_url, transport=self._transport, timeout=120
        ) as client:
            response = await client.post(
                "/api/chat",
                json={"model": model_for(role), "messages": messages, "stream": False},
            )
            response.raise_for_status()
            body = response.json()
        latency_ms = int((time.monotonic() - start) * 1000)

        usage: LLMUsage = {
            "input_tokens": body.get("prompt_eval_count", 0),
            "output_tokens": body.get("eval_count", 0),
            "latency_ms": latency_ms,
        }
        return body["message"]["content"], usage


_DEFAULT_OPENAI_MODEL = "openai/gpt-4o-mini"


class OpenAICompatibleProvider:
    """LLMProvider over any OpenAI-compatible /chat/completions endpoint
    (e.g. requesty.ai), auth'd with a bearer API key. Calls are gated by a
    semaphore, not fired raw: a zero/low-balance router account caps
    concurrent in-flight requests (as low as 1), and the 4 lens agents call
    complete() concurrently (agents.py's run_all_lenses) -- without this,
    request 2-4 come back 429 instead of queuing."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        default_model: str | None = None,
        transport: httpx.BaseTransport | None = None,
        max_concurrency: int | None = None,
    ):
        self.api_key = api_key or os.environ["OPENAI_API_KEY"]
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.default_model = default_model or os.environ.get(
            "OPENAI_DEFAULT_MODEL", _DEFAULT_OPENAI_MODEL
        )
        self._transport = transport  # injectable for tests; None uses the real network
        self._semaphore = asyncio.Semaphore(
            max_concurrency or int(os.environ.get("OPENAI_MAX_CONCURRENCY", "1"))
        )

    async def complete(self, role: Role, prompt: str, *, system: str | None = None) -> str:
        content, _usage = await self.complete_with_usage(role, prompt, system=system)
        return content

    async def complete_with_usage(
        self, role: Role, prompt: str, *, system: str | None = None
    ) -> tuple[str, LLMUsage]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        async with self._semaphore:
            start = time.monotonic()
            async with httpx.AsyncClient(
                base_url=self.base_url,
                transport=self._transport,
                timeout=120,
                headers={"Authorization": f"Bearer {self.api_key}"},
            ) as client:
                response = await client.post(
                    "/chat/completions",
                    json={
                        "model": model_for(role, default=self.default_model),
                        "messages": messages,
                    },
                )
                response.raise_for_status()
                body = response.json()
            latency_ms = int((time.monotonic() - start) * 1000)

        body_usage = body.get("usage", {})
        usage: LLMUsage = {
            "input_tokens": body_usage.get("prompt_tokens", 0),
            "output_tokens": body_usage.get("completion_tokens", 0),
            "latency_ms": latency_ms,
        }
        return body["choices"][0]["message"]["content"], usage
