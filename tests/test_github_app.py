import base64
import json
import time

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from pr_review.contracts import Finding
from pr_review.github_app import GitHubAppProvider

APP_ID = "12345"
REPO = "varunbalwani/In-memory-Pub-sub-in-Go"

_private_key_obj = rsa.generate_private_key(public_exponent=65537, key_size=2048)
PRIVATE_KEY = _private_key_obj.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.NoEncryption(),
).decode()
PUBLIC_KEY = _private_key_obj.public_key()


def _future_iso(seconds: int) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + seconds))


def _install_and_token_routes(captured: dict):
    def handler(request: httpx.Request) -> httpx.Response:
        auth = request.headers["authorization"]
        assert auth.startswith("Bearer ")
        token = auth.removeprefix("Bearer ")

        if request.url.path == f"/repos/{REPO}/installation":
            captured["app_jwt"] = token
            return httpx.Response(200, json={"id": 999})
        if request.url.path == "/app/installations/999/access_tokens":
            captured.setdefault("token_mint_count", 0)
            captured["token_mint_count"] += 1
            return httpx.Response(
                200, json={"token": "install-token-abc", "expires_at": _future_iso(3600)}
            )
        raise AssertionError(f"unexpected request {request.url.path}")

    return handler


def test_mint_jwt_is_signed_rs256_for_configured_app_id():
    provider = GitHubAppProvider(APP_ID, PRIVATE_KEY)
    token = provider._mint_jwt()
    decoded = jwt.decode(token, PUBLIC_KEY, algorithms=["RS256"])
    assert decoded["iss"] == APP_ID
    assert decoded["exp"] > decoded["iat"]


@pytest.mark.asyncio
async def test_installation_token_is_minted_via_app_jwt_then_cached():
    captured: dict = {}
    provider = GitHubAppProvider(
        APP_ID, PRIVATE_KEY, transport=httpx.MockTransport(_install_and_token_routes(captured))
    )

    first = await provider._installation_token(REPO)
    second = await provider._installation_token(REPO)

    assert first == second == "install-token-abc"
    assert captured["token_mint_count"] == 1  # cached, not re-minted
    jwt.decode(captured["app_jwt"], PUBLIC_KEY, algorithms=["RS256"])


@pytest.mark.asyncio
async def test_installation_token_is_reminted_once_expired():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == f"/repos/{REPO}/installation":
            return httpx.Response(200, json={"id": 999})
        if request.url.path == "/app/installations/999/access_tokens":
            captured.setdefault("token_mint_count", 0)
            captured["token_mint_count"] += 1
            return httpx.Response(
                200, json={"token": "install-token-abc", "expires_at": _future_iso(-10)}
            )
        raise AssertionError(f"unexpected request {request.url.path}")

    provider = GitHubAppProvider(APP_ID, PRIVATE_KEY, transport=httpx.MockTransport(handler))
    await provider._installation_token(REPO)
    await provider._installation_token(REPO)

    assert captured["token_mint_count"] == 2


@pytest.mark.asyncio
async def test_fetch_diff_returns_head_sha_and_files():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == f"/repos/{REPO}/installation":
            return httpx.Response(200, json={"id": 999})
        if path == "/app/installations/999/access_tokens":
            return httpx.Response(
                200, json={"token": "install-token-abc", "expires_at": _future_iso(3600)}
            )
        if path == f"/repos/{REPO}/pulls/7":
            return httpx.Response(200, json={"head": {"sha": "deadbeef"}})
        if path == f"/repos/{REPO}/pulls/7/files":
            captured["auth"] = request.headers["authorization"]
            return httpx.Response(
                200,
                json=[
                    {"filename": "a.py", "status": "modified", "patch": "@@ -1 +1 @@\n-a\n+b"},
                    {"filename": "b.py", "status": "added", "patch": "@@ -0,0 +1 @@\n+new"},
                ],
            )
        raise AssertionError(f"unexpected request {path}")

    provider = GitHubAppProvider(APP_ID, PRIVATE_KEY, transport=httpx.MockTransport(handler))
    diff = await provider.fetch_diff(REPO, 7)

    assert diff.head_sha == "deadbeef"
    assert [f.path for f in diff.files] == ["a.py", "b.py"]
    assert diff.files[0].status == "modified"
    assert captured["auth"] == "Bearer install-token-abc"


@pytest.mark.asyncio
async def test_fetch_file_decodes_base64_content():
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == f"/repos/{REPO}/installation":
            return httpx.Response(200, json={"id": 999})
        if path == "/app/installations/999/access_tokens":
            return httpx.Response(
                200, json={"token": "install-token-abc", "expires_at": _future_iso(3600)}
            )
        if path == f"/repos/{REPO}/contents/README.md":
            assert request.url.params["ref"] == "deadbeef"
            content = base64.b64encode(b"# Hello").decode()
            return httpx.Response(200, json={"content": content})
        raise AssertionError(f"unexpected request {path}")

    provider = GitHubAppProvider(APP_ID, PRIVATE_KEY, transport=httpx.MockTransport(handler))
    content = await provider.fetch_file(REPO, "README.md", "deadbeef")

    assert content == "# Hello"


@pytest.mark.asyncio
async def test_post_review_sends_batched_comments():
    captured: dict = {}
    finding = Finding(
        path="a.py",
        line=5,
        side="RIGHT",
        category="security",
        severity="high",
        confidence=0.9,
        rationale="hardcoded secret",
        fingerprint="abc123",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == f"/repos/{REPO}/installation":
            return httpx.Response(200, json={"id": 999})
        if path == "/app/installations/999/access_tokens":
            return httpx.Response(
                200, json={"token": "install-token-abc", "expires_at": _future_iso(3600)}
            )
        if path == f"/repos/{REPO}/pulls/7/reviews":
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"id": 1})
        raise AssertionError(f"unexpected request {path}")

    provider = GitHubAppProvider(APP_ID, PRIVATE_KEY, transport=httpx.MockTransport(handler))
    await provider.post_review(REPO, 7, [finding], summary="1 finding")

    assert captured["body"]["event"] == "COMMENT"
    assert captured["body"]["body"] == "1 finding"
    assert captured["body"]["comments"] == [
        {"path": "a.py", "line": 5, "side": "RIGHT", "body": "hardcoded secret"}
    ]


@pytest.mark.asyncio
async def test_list_review_comments_returns_raw_comments():
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == f"/repos/{REPO}/installation":
            return httpx.Response(200, json={"id": 999})
        if path == "/app/installations/999/access_tokens":
            return httpx.Response(
                200, json={"token": "install-token-abc", "expires_at": _future_iso(3600)}
            )
        if path == f"/repos/{REPO}/pulls/7/comments":
            return httpx.Response(
                200,
                json=[
                    {"id": 1, "path": "app.py", "line": 5, "commit_id": "deadbeef"},
                    {"id": 2, "path": "b.py", "line": 9, "commit_id": "deadbeef"},
                ],
            )
        raise AssertionError(f"unexpected request {path}")

    provider = GitHubAppProvider(APP_ID, PRIVATE_KEY, transport=httpx.MockTransport(handler))
    comments = await provider.list_review_comments(REPO, 7)

    assert len(comments) == 2
    assert comments[0]["path"] == "app.py"
    assert comments[1]["commit_id"] == "deadbeef"


@pytest.mark.asyncio
async def test_resolve_comment_sends_graphql_mutation():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == f"/repos/{REPO}/installation":
            return httpx.Response(200, json={"id": 999})
        if path == "/app/installations/999/access_tokens":
            return httpx.Response(
                200, json={"token": "install-token-abc", "expires_at": _future_iso(3600)}
            )
        if path == "/graphql":
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"data": {"resolveReviewThread": {"thread": {}}}})
        raise AssertionError(f"unexpected request {path}")

    provider = GitHubAppProvider(APP_ID, PRIVATE_KEY, transport=httpx.MockTransport(handler))
    await provider.resolve_comment(REPO, 7, "PRRT_abc123")

    assert captured["body"]["variables"] == {"id": "PRRT_abc123"}
    assert "resolveReviewThread" in captured["body"]["query"]
