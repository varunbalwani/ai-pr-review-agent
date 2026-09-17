"""Postgres storage for TASK-storage: findings idempotency (FR-8), an LLM
call audit trail with no raw content (FR-9), and PR/comment outcome capture
(FR-10). create_schema is idempotent (CREATE TABLE IF NOT EXISTS) so both
the app and tests can call it freely against a fresh Tiger Cloud service."""

from typing import Literal, Protocol

from pr_review.contracts import Finding

Kind = Literal["pr", "comment"]


class _Executor(Protocol):
    async def execute(self, query: str, *args) -> str: ...
    async def fetchrow(self, query: str, *args): ...


_SCHEMA = """
CREATE TABLE IF NOT EXISTS posted_findings (
    id BIGSERIAL PRIMARY KEY,
    repo TEXT NOT NULL,
    pr_number INTEGER NOT NULL,
    head_sha TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    path TEXT NOT NULL,
    line INTEGER NOT NULL,
    category TEXT NOT NULL,
    severity TEXT NOT NULL,
    posted_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (repo, pr_number, head_sha, fingerprint)
);

CREATE TABLE IF NOT EXISTS audit_log (
    id BIGSERIAL PRIMARY KEY,
    model TEXT NOT NULL,
    agent TEXT NOT NULL,
    repo TEXT NOT NULL,
    pr_number INTEGER NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    latency_ms INTEGER NOT NULL,
    decision TEXT,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS outcomes (
    id BIGSERIAL PRIMARY KEY,
    repo TEXT NOT NULL,
    pr_number INTEGER NOT NULL,
    kind TEXT NOT NULL,
    comment_id TEXT,
    status TEXT NOT NULL,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


async def create_schema(conn: _Executor) -> None:
    await conn.execute(_SCHEMA)


async def claim_finding(
    conn: _Executor, *, repo: str, pr_number: int, head_sha: str, finding: Finding
) -> bool:
    """FR-8: re-checks (repo, pr_number, head_sha, fingerprint) immediately
    before writing so retries never double-post. True the first time this
    key is claimed, False on every retry."""
    row = await conn.fetchrow(
        """
        INSERT INTO posted_findings (repo, pr_number, head_sha, fingerprint, path, line, category, severity)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        ON CONFLICT (repo, pr_number, head_sha, fingerprint) DO NOTHING
        RETURNING id
        """,
        repo,
        pr_number,
        head_sha,
        finding.fingerprint,
        finding.path,
        finding.line,
        finding.category,
        finding.severity,
    )
    return row is not None


async def record_llm_call(
    conn: _Executor,
    *,
    model: str,
    agent: str,
    repo: str,
    pr_number: int,
    input_tokens: int,
    output_tokens: int,
    latency_ms: int,
    decision: str | None = None,
) -> None:
    """FR-9: model/tokens/latency/agent/repo/pr_number/timestamp/decision
    only -- no raw diff or completion content. decision is only known once
    the verify step routes a finding, so lens-agent calls have none."""
    await conn.execute(
        """
        INSERT INTO audit_log (model, agent, repo, pr_number, input_tokens, output_tokens, latency_ms, decision)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """,
        model,
        agent,
        repo,
        pr_number,
        input_tokens,
        output_tokens,
        latency_ms,
        decision,
    )


async def record_outcome(
    conn: _Executor,
    *,
    repo: str,
    pr_number: int,
    kind: Kind,
    comment_id: str | None,
    status: str,
) -> None:
    """FR-10: PR outcome (merged/closed) when kind='pr', comment resolution
    (resolved/dismissed) when kind='comment'."""
    await conn.execute(
        """
        INSERT INTO outcomes (repo, pr_number, kind, comment_id, status)
        VALUES ($1, $2, $3, $4, $5)
        """,
        repo,
        pr_number,
        kind,
        comment_id,
        status,
    )
