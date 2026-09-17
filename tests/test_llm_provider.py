import json

import httpx
import pytest

from pr_review.llm import OllamaProvider, model_for


def test_model_for_defaults_to_qwen(monkeypatch):
    monkeypatch.delenv("AGENT_MODEL_SECURITY", raising=False)
    assert model_for("security") == "qwen3:8b"


def test_model_for_reads_config_key(monkeypatch):
    monkeypatch.setenv("AGENT_MODEL_VERIFY", "qwen3:14b")
    assert model_for("verify") == "qwen3:14b"


@pytest.mark.asyncio
async def test_ollama_provider_posts_chat_request_and_parses_reply():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json={"message": {"content": "looks fine"}})

    provider = OllamaProvider(
        base_url="http://localhost:11434", transport=httpx.MockTransport(handler)
    )
    reply = await provider.complete("quality", "review this diff", system="you are a reviewer")

    assert reply == "looks fine"
    assert captured["url"] == "http://localhost:11434/api/chat"
    assert captured["json"]["model"] == "qwen3:8b"
    assert captured["json"]["messages"][0] == {"role": "system", "content": "you are a reviewer"}


@pytest.mark.asyncio
async def test_complete_with_usage_returns_content_and_token_counts():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "message": {"content": "looks fine"},
                "prompt_eval_count": 42,
                "eval_count": 7,
            },
        )

    provider = OllamaProvider(
        base_url="http://localhost:11434", transport=httpx.MockTransport(handler)
    )
    content, usage = await provider.complete_with_usage("quality", "review this diff")

    assert content == "looks fine"
    assert usage["input_tokens"] == 42
    assert usage["output_tokens"] == 7
    assert usage["latency_ms"] >= 0


@pytest.mark.asyncio
async def test_complete_with_usage_defaults_token_counts_to_zero_when_ollama_omits_them():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"content": "ok"}})

    provider = OllamaProvider(
        base_url="http://localhost:11434", transport=httpx.MockTransport(handler)
    )
    _content, usage = await provider.complete_with_usage("quality", "x")

    assert usage["input_tokens"] == 0
    assert usage["output_tokens"] == 0
