import json
import os
import uuid

import asyncpg
import httpx
import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from pr_review.contracts import Diff, DiffFile, Finding
from pr_review.hitl import authenticate_approver, check_repo_ownership, create_app, reanchor_finding
from pr_review.orchestration import build_graph
from pr_review.storage import create_schema

REPO = "varunbalwani/In-memory-Pub-sub-in-Go"


@pytest.fixture
async def db_pool():
    pool = await asyncpg.create_pool(os.environ["DATABASE_URL"])
    async with pool.acquire() as conn:
        await create_schema(conn)
    yield pool
    await pool.close()


def _finding(fingerprint: str = "fp1", snippet: str = "os.system(x)", path: str = "app.py") -> Finding:
    return Finding(
        path=path,
        line=10,
        side="RIGHT",
        category="security",
        severity="high",
        confidence=0.5,
        rationale="command injection",
        fingerprint=fingerprint,
        snippet=snippet,
    )


# --- reanchor_finding: pure function, FR-7 ---


def test_reanchor_finding_keeps_line_when_snippet_still_at_same_position():
    diff = Diff(
        head_sha="newsha",
        files=[
            DiffFile(
                path="app.py", status="modified", patch="@@ -10,1 +10,2 @@\n+os.system(x)\n+more"
            )
        ],
    )
    assert reanchor_finding(_finding(), diff) == ("line", 10)


def test_reanchor_finding_resolves_new_line_when_snippet_moved():
    diff = Diff(
        head_sha="newsha",
        files=[DiffFile(path="app.py", status="modified", patch="@@ -10,1 +30,1 @@\n+os.system(x)")],
    )
    assert reanchor_finding(_finding(), diff) == ("line", 30)


def test_reanchor_finding_ignores_the_snippet_on_a_deleted_line():
    """A RIGHT-side finding must not anchor to text that only appears on a
    '-' (removed) line -- that line no longer exists in the current file."""
    diff = Diff(
        head_sha="newsha",
        files=[
            DiffFile(
                path="app.py", status="modified", patch="@@ -10,1 +10,1 @@\n-os.system(x)\n+safe()"
            )
        ],
    )
    assert reanchor_finding(_finding(), diff) == ("file_note", None)


def test_reanchor_finding_falls_back_to_file_note_when_snippet_gone():
    diff = Diff(
        head_sha="newsha",
        files=[
            DiffFile(
                path="app.py", status="modified", patch="@@ -10,1 +10,1 @@\n+something else entirely"
            )
        ],
    )
    assert reanchor_finding(_finding(), diff) == ("file_note", None)


def test_reanchor_finding_resolves_line_in_a_second_hunk_with_no_length_field():
    """GitHub emits the no-comma hunk-header form (@@ -N +M @@) for a
    single-line change; the header-reset must work per hunk, not just once."""
    diff = Diff(
        head_sha="newsha",
        files=[
            DiffFile(
                path="app.py",
                status="modified",
                patch="@@ -1,2 +1,2 @@\n context\n context\n@@ -10 +30 @@\n+os.system(x)",
            )
        ],
    )
    assert reanchor_finding(_finding(), diff) == ("line", 30)


def test_reanchor_finding_falls_back_to_file_note_when_file_removed():
    diff = Diff(head_sha="newsha", files=[DiffFile(path="other.py", status="modified", patch="+x")])
    assert reanchor_finding(_finding(), diff) == ("file_note", None)


# --- authenticate_approver / check_repo_ownership: FR-6/NFR-5 ---


@pytest.mark.asyncio
async def test_authenticate_approver_resolves_valid_token_to_login():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer good-token"
        return httpx.Response(200, json={"login": "varunbalwani"})

    login = await authenticate_approver("good-token", transport=httpx.MockTransport(handler))
    assert login == "varunbalwani"


@pytest.mark.asyncio
async def test_authenticate_approver_rejects_invalid_token():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"message": "Bad credentials"})

    with pytest.raises(PermissionError):
        await authenticate_approver("bad-token", transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_check_repo_ownership_true_for_write_permission():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == f"/repos/{REPO}/collaborators/varunbalwani/permission"
        return httpx.Response(200, json={"permission": "write"})

    assert await check_repo_ownership(
        "varunbalwani", REPO, "good-token", transport=httpx.MockTransport(handler)
    )


@pytest.mark.asyncio
async def test_check_repo_ownership_false_for_read_permission():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"permission": "read"})

    assert not await check_repo_ownership(
        "varunbalwani", REPO, "good-token", transport=httpx.MockTransport(handler)
    )


@pytest.mark.asyncio
async def test_check_repo_ownership_false_when_not_a_collaborator():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    assert not await check_repo_ownership(
        "varunbalwani", REPO, "good-token", transport=httpx.MockTransport(handler)
    )


@pytest.mark.asyncio
async def test_check_repo_ownership_fails_closed_on_forbidden():
    """GitHub returns 403 (not 404) when the caller's own token lacks enough
    access on the repo to even ask about collaborator permissions -- this
    must deny, not raise (fail closed, NFR-5)."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"message": "Must have push access"})

    assert not await check_repo_ownership(
        "varunbalwani", REPO, "good-token", transport=httpx.MockTransport(handler)
    )


# --- create_app: the endpoint end-to-end ---


class _StubVCS:
    def __init__(self, diff: Diff):
        self._diff = diff
        self.posted: list[tuple] = []

    async def fetch_diff(self, repo, pr_number):
        return self._diff

    async def post_review(self, repo, pr_number, findings, summary):
        self.posted.append((repo, pr_number, findings, summary))


def _github_handler(login: str = "varunbalwani", permission: str = "write"):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/user":
            return httpx.Response(200, json={"login": login})
        if request.url.path.endswith("/permission"):
            return httpx.Response(200, json={"permission": permission})
        raise AssertionError(f"unexpected request {request.url.path}")

    return handler


async def _paused_graph(repo: str = REPO):
    """Builds a graph already paused on one HITL finding, matching
    test_orchestration.py's pattern (verify always returns 0.5 -> hitl)."""

    class _StubLLM:
        async def complete(self, role, prompt, *, system=None):
            if role == "verify":
                return json.dumps({"confidence": 0.4})
            if role == "security":
                return json.dumps(
                    [
                        {
                            "path": "app.py",
                            "line": 10,
                            "side": "RIGHT",
                            "snippet": "os.system(x)",
                            "severity": "high",
                            "confidence": 0.4,
                            "rationale": "command injection",
                            "suggested_patch": None,
                        }
                    ]
                )
            return "[]"

    graph = build_graph(_StubLLM(), checkpointer=MemorySaver())
    diff = Diff(head_sha="oldsha", files=[DiffFile(path="app.py", status="modified", patch="+os.system(x)")])
    config = {"configurable": {"thread_id": f"{repo}#7#oldsha"}}
    paused = await graph.ainvoke(
        {"diff": diff, "findings": [], "to_post": [], "discarded": []}, config=config
    )
    fingerprint = paused["__interrupt__"][0].value["hitl_findings"][0]["fingerprint"]
    return graph, fingerprint


@pytest.mark.asyncio
async def test_pending_endpoint_returns_paused_findings_without_resuming(db_pool):
    graph, fingerprint = await _paused_graph()
    app = create_app(
        graph,
        _StubVCS(Diff(head_sha="newsha", files=[])),
        db_pool,
        transport=httpx.MockTransport(_github_handler()),
    )
    client = TestClient(app)

    response = client.get(
        "/hitl/pending",
        params={"repo": REPO, "pr_number": 7, "head_sha": "oldsha"},
        headers={"Authorization": "Bearer good-token"},
    )

    assert response.status_code == 200
    findings = response.json()["hitl_findings"]
    assert len(findings) == 1
    assert findings[0]["fingerprint"] == fingerprint

    # still paused -- reading pending state must not resume the run
    still_pending = client.get(
        "/hitl/pending",
        params={"repo": REPO, "pr_number": 7, "head_sha": "oldsha"},
        headers={"Authorization": "Bearer good-token"},
    )
    assert len(still_pending.json()["hitl_findings"]) == 1


@pytest.mark.asyncio
async def test_pending_endpoint_rejects_missing_token(db_pool):
    graph, _fingerprint = await _paused_graph()
    app = create_app(
        graph,
        _StubVCS(Diff(head_sha="newsha", files=[])),
        db_pool,
        transport=httpx.MockTransport(_github_handler()),
    )
    client = TestClient(app)

    response = client.get("/hitl/pending", params={"repo": REPO, "pr_number": 7, "head_sha": "oldsha"})

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_approve_endpoint_rejects_missing_token(db_pool):
    graph, fingerprint = await _paused_graph()
    app = create_app(
        graph,
        _StubVCS(Diff(head_sha="newsha", files=[])),
        db_pool,
        transport=httpx.MockTransport(_github_handler()),
    )
    client = TestClient(app)

    response = client.post(
        "/hitl/approve",
        json={"repo": REPO, "pr_number": 7, "head_sha": "oldsha", "approvals": {fingerprint: True}},
    )

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_approve_endpoint_rejects_approver_without_write_access(db_pool):
    graph, fingerprint = await _paused_graph()
    transport = httpx.MockTransport(_github_handler(permission="read"))
    app = create_app(graph, _StubVCS(Diff(head_sha="newsha", files=[])), db_pool, transport=transport)
    client = TestClient(app)

    response = client.post(
        "/hitl/approve",
        json={"repo": REPO, "pr_number": 7, "head_sha": "oldsha", "approvals": {fingerprint: True}},
        headers={"Authorization": "Bearer good-token"},
    )

    assert response.status_code == 403


@pytest.mark.asyncio
async def test_approve_endpoint_resumes_and_reanchors_to_line(db_pool):
    repo = f"varunbalwani/test-{uuid.uuid4()}"
    graph, fingerprint = await _paused_graph(repo)
    fresh_diff = Diff(
        head_sha="newsha",
        files=[
            DiffFile(
                path="app.py", status="modified", patch="@@ -10,1 +30,2 @@\n+os.system(x)\n+more"
            )
        ],
    )
    transport = httpx.MockTransport(_github_handler())
    vcs = _StubVCS(fresh_diff)
    app = create_app(graph, vcs, db_pool, transport=transport)

    # an async client sharing this test's event loop: TestClient runs the ASGI
    # app on its own thread/loop, which the db_pool fixture's asyncpg
    # connections (bound to this loop) can't safely be used from
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/hitl/approve",
            json={
                "repo": repo,
                "pr_number": 7,
                "head_sha": "oldsha",
                "approvals": {fingerprint: True},
            },
            headers={"Authorization": "Bearer good-token"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["head_sha"] == "newsha"
    assert len(body["to_post"]) == 1
    assert body["to_post"][0]["anchor"] == "line"
    assert body["to_post"][0]["finding"]["fingerprint"] == fingerprint
    assert body["to_post"][0]["finding"]["line"] == 30  # re-anchored, not the stale pre-pause line 10
    assert body["discarded"] == []

    # the gap this test now closes: approval must actually reach GitHub, not just
    # return the routed decision (previously /hitl/approve never called post_review)
    assert len(vcs.posted) == 1
    assert vcs.posted[0][0] == repo
    assert vcs.posted[0][1] == 7
    assert vcs.posted[0][2][0].fingerprint == fingerprint


@pytest.mark.asyncio
async def test_approve_endpoint_falls_back_to_file_note_when_line_gone(db_pool):
    repo = f"varunbalwani/test-{uuid.uuid4()}"
    graph, fingerprint = await _paused_graph(repo)
    fresh_diff = Diff(
        head_sha="newsha",
        files=[
            DiffFile(
                path="app.py", status="modified", patch="@@ -10,1 +10,1 @@\n+totally different code"
            )
        ],
    )
    transport = httpx.MockTransport(_github_handler())
    app = create_app(graph, _StubVCS(fresh_diff), db_pool, transport=transport)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/hitl/approve",
            json={
                "repo": repo,
                "pr_number": 7,
                "head_sha": "oldsha",
                "approvals": {fingerprint: True},
            },
            headers={"Authorization": "Bearer good-token"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["to_post"][0]["anchor"] == "file_note"
