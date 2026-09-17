import asyncio
import os
import uuid

import asyncpg
import pytest

from pr_review.contracts import Diff, DiffFile, Finding
from pr_review.storage import create_schema
from pr_review.worker import AuditingLLMProvider, post_findings


@pytest.fixture
async def db_conn():
    pool = await asyncpg.create_pool(os.environ["DATABASE_URL"])
    async with pool.acquire() as conn:
        await create_schema(conn)
        tx = conn.transaction()
        await tx.start()
        try:
            yield conn
        finally:
            await tx.rollback()
    await pool.close()


@pytest.fixture
async def db_pool():
    pool = await asyncpg.create_pool(os.environ["DATABASE_URL"])
    async with pool.acquire() as conn:
        await create_schema(conn)
    yield pool
    await pool.close()


class _StubOllama:
    """Each role gets its own fixed (content, usage) pair -- concurrent
    complete() calls must not cross-attribute usage between roles."""

    def __init__(self, replies: dict[str, tuple[str, dict]]):
        self._replies = replies
        self.default_model = "stub-model"

    async def complete_with_usage(self, role, prompt, *, system=None):
        await asyncio.sleep(0)  # force an interleaving point
        return self._replies[role]


@pytest.mark.asyncio
async def test_auditing_llm_provider_attributes_usage_correctly_under_concurrency(db_pool):
    repo = f"o/r-{uuid.uuid4()}"
    ollama = _StubOllama(
        {
            "security": ("sec reply", {"input_tokens": 10, "output_tokens": 1, "latency_ms": 5}),
            "quality": ("qual reply", {"input_tokens": 20, "output_tokens": 2, "latency_ms": 9}),
        }
    )
    llm = AuditingLLMProvider(ollama, db_pool, repo=repo, pr_number=1)

    security_result, quality_result = await asyncio.gather(
        llm.complete("security", "p1"), llm.complete("quality", "p2")
    )

    assert security_result == "sec reply"
    assert quality_result == "qual reply"

    async with db_pool.acquire() as conn:
        rows = {
            r["agent"]: r
            for r in await conn.fetch("SELECT * FROM audit_log WHERE repo = $1", repo)
        }
    assert rows["security"]["input_tokens"] == 10
    assert rows["security"]["output_tokens"] == 1
    assert rows["quality"]["input_tokens"] == 20
    assert rows["quality"]["output_tokens"] == 2


def _finding(fingerprint: str = "fp1") -> Finding:
    return Finding(
        path="app.py",
        line=1,
        side="RIGHT",
        category="security",
        severity="high",
        confidence=0.9,
        rationale="r",
        fingerprint=fingerprint,
    )


class _StubVCS:
    def __init__(self):
        self.posted: list[tuple] = []

    async def post_review(self, repo, pr_number, findings, summary):
        self.posted.append((repo, pr_number, findings, summary))


def _diff() -> Diff:
    """A diff where (app.py, line=1, side=RIGHT) -- _finding()'s position -- is valid."""
    return Diff(head_sha="sha1", files=[DiffFile(path="app.py", status="modified", patch="@@ -1,1 +1,1 @@\n+x")])


@pytest.mark.asyncio
async def test_post_findings_posts_once_and_skips_the_retry(db_conn):
    vcs = _StubVCS()
    finding = _finding()

    await post_findings(db_conn, vcs, repo="o/r", pr_number=1, head_sha="sha1", findings=[finding], diff=_diff())
    await post_findings(db_conn, vcs, repo="o/r", pr_number=1, head_sha="sha1", findings=[finding], diff=_diff())

    assert len(vcs.posted) == 1  # FR-8: retry with the same key never double-posts
    assert vcs.posted[0][2] == [finding]


@pytest.mark.asyncio
async def test_post_findings_does_not_call_vcs_when_nothing_to_post(db_conn):
    vcs = _StubVCS()

    await post_findings(db_conn, vcs, repo="o/r", pr_number=1, head_sha="sha1", findings=[], diff=_diff())

    assert vcs.posted == []


@pytest.mark.asyncio
async def test_post_findings_drops_finding_with_invalid_diff_position(db_conn):
    """A finding whose (path, line, side) isn't an actual diff position (FR-8's
    live 422 failure mode) is dropped unclaimed, not claimed-then-posted."""
    vcs = _StubVCS()
    finding = _finding().model_copy(update={"line": 999})

    await post_findings(db_conn, vcs, repo="o/r", pr_number=1, head_sha="sha1", findings=[finding], diff=_diff())

    assert vcs.posted == []
    row = await db_conn.fetchrow(
        "SELECT 1 FROM posted_findings WHERE repo = $1 AND fingerprint = $2", "o/r", finding.fingerprint
    )
    assert row is None  # not claimed -- stays retriable, not permanently orphaned


@pytest.mark.asyncio
async def test_post_findings_rolls_back_claims_when_vcs_post_fails(db_conn):
    """A post_review failure (422, network error) must not leave findings
    permanently claimed -- ASSUMPTION-cdc6fa50's data-loss scenario."""

    class _FailingVCS:
        async def post_review(self, repo, pr_number, findings, summary):
            raise RuntimeError("simulated 422")

    finding = _finding()
    with pytest.raises(RuntimeError):
        await post_findings(
            db_conn, _FailingVCS(), repo="o/r", pr_number=1, head_sha="sha1", findings=[finding], diff=_diff()
        )

    row = await db_conn.fetchrow(
        "SELECT 1 FROM posted_findings WHERE repo = $1 AND fingerprint = $2", "o/r", finding.fingerprint
    )
    assert row is None  # rolled back -- retriable, not orphaned
