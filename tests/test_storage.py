import os

import asyncpg
import pytest

from pr_review.contracts import Finding
from pr_review.storage import claim_finding, create_schema, record_llm_call, record_outcome


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


def _finding(fingerprint: str = "fp1") -> Finding:
    return Finding(
        path="app.py",
        line=1,
        side="RIGHT",
        category="security",
        severity="medium",
        confidence=0.9,
        rationale="r",
        fingerprint=fingerprint,
    )


@pytest.mark.asyncio
async def test_claim_finding_is_idempotent_on_retry(db_conn):
    first = await claim_finding(
        db_conn, repo="o/r", pr_number=1, head_sha="sha1", finding=_finding()
    )
    second = await claim_finding(
        db_conn, repo="o/r", pr_number=1, head_sha="sha1", finding=_finding()
    )

    assert first is True
    assert second is False


@pytest.mark.asyncio
async def test_claim_finding_treats_different_head_sha_as_distinct(db_conn):
    first = await claim_finding(
        db_conn, repo="o/r", pr_number=1, head_sha="sha1", finding=_finding()
    )
    second = await claim_finding(
        db_conn, repo="o/r", pr_number=1, head_sha="sha2", finding=_finding()
    )

    assert first is True
    assert second is True


@pytest.mark.asyncio
async def test_record_llm_call_stores_metadata_with_no_raw_content_column(db_conn):
    await record_llm_call(
        db_conn,
        model="qwen3:8b",
        agent="security",
        repo="o/r",
        pr_number=1,
        input_tokens=100,
        output_tokens=20,
        latency_ms=350,
        decision="auto_post",
    )

    row = await db_conn.fetchrow("SELECT * FROM audit_log WHERE repo = 'o/r'")
    assert row["model"] == "qwen3:8b"
    assert row["agent"] == "security"
    assert row["input_tokens"] == 100
    assert row["output_tokens"] == 20
    assert row["latency_ms"] == 350
    assert row["decision"] == "auto_post"
    assert set(row.keys()) == {
        "id",
        "model",
        "agent",
        "repo",
        "pr_number",
        "input_tokens",
        "output_tokens",
        "latency_ms",
        "decision",
        "recorded_at",
    }


@pytest.mark.asyncio
async def test_record_llm_call_allows_no_decision_for_lens_calls(db_conn):
    await record_llm_call(
        db_conn,
        model="qwen3:8b",
        agent="security",
        repo="o/r",
        pr_number=1,
        input_tokens=100,
        output_tokens=20,
        latency_ms=350,
    )

    row = await db_conn.fetchrow("SELECT decision FROM audit_log WHERE repo = 'o/r'")
    assert row["decision"] is None


@pytest.mark.asyncio
async def test_record_outcome_captures_pr_merge_and_comment_resolution(db_conn):
    await record_outcome(
        db_conn, repo="o/r", pr_number=1, kind="pr", comment_id=None, status="merged"
    )
    await record_outcome(
        db_conn, repo="o/r", pr_number=1, kind="comment", comment_id="PRRT_1", status="resolved"
    )

    rows = await db_conn.fetch("SELECT kind, comment_id, status FROM outcomes ORDER BY kind")
    assert [dict(r) for r in rows] == [
        {"kind": "comment", "comment_id": "PRRT_1", "status": "resolved"},
        {"kind": "pr", "comment_id": None, "status": "merged"},
    ]
