"""ARQ worker: runs a queued review job end to end. review_pr fetches the
diff, runs the LangGraph orchestration (analyze -> verify -> route), and --
if the run didn't pause for HITL -- executes the posting step that
DECISION-b3356b1f explicitly deferred out of TASK-hitl: claim each to_post
finding (FR-8) before posting it, so a retried job never double-posts.
AuditingLLMProvider wraps the real LLMProvider to record one audit_log row
per call (FR-9/AC-4) -- usage travels back with the call's own return
value (complete_with_usage), never through shared state, so concurrent
lens calls can't cross-attribute tokens to the wrong agent.

NFR-4 caveat: the checkpointer this worker uses is an in-process
MemorySaver (TASK-storage's Postgres-backed saver per DECISION-4fed71f4
doesn't exist yet), so a HITL pause does not survive a worker restart --
AC-2 is only verifiable within one continuous worker run.

ASSUMPTION-cdc6fa50 is narrowed, not retired: claim_finding and post_review
now run in one DB transaction, so a post failure (422, network error) rolls
the claims back and the findings stay retriable. Only a process crash in the
gap between post_review returning and the transaction committing still
permanently burns the claim -- two-phase commit would be needed to close
that, and Phase 1 doesn't have it.

post_findings also drops any finding whose (path, line, side) isn't an
actual position in the current diff before claiming it, since GitHub's
create-review call is all-or-nothing: one invalid position previously
caused an entire batch (12 valid findings included) to 422 and, before the
transaction fix above, get silently and permanently orphaned in
posted_findings. An unanchorable finding is dropped unclaimed rather than
posted as a file-level note -- it stays out of this run's review but isn't
consumed, so a future run (or a fix to the finding it came from) can still
post it."""

import re

import asyncpg
from arq import ArqRedis
from arq.connections import RedisSettings
from langgraph.checkpoint.memory import MemorySaver

from pr_review.contracts import Diff, Finding, VCSProvider
from pr_review.llm import LLMProviderWithUsage, model_for
from pr_review.orchestration import build_graph
from pr_review.storage import claim_finding, record_llm_call

_HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")


class AuditingLLMProvider:
    """Implements LLMProvider (complete() -> str) for one job's lifetime,
    bound to a single (repo, pr_number) so call sites need no audit-specific
    arguments. Each call records its own audit row from its own usage
    return value -- no attribute is shared or mutated across calls."""

    def __init__(self, base: LLMProviderWithUsage, pool: asyncpg.Pool, *, repo: str, pr_number: int):
        self._base = base
        self._pool = pool
        self._repo = repo
        self._pr_number = pr_number

    async def complete(self, role, prompt, *, system=None) -> str:
        content, usage = await self._base.complete_with_usage(role, prompt, system=system)
        async with self._pool.acquire() as conn:  # own connection per call: calls run concurrently
            await record_llm_call(
                conn,
                model=model_for(role, default=self._base.default_model),
                agent=role,
                repo=self._repo,
                pr_number=self._pr_number,
                input_tokens=usage["input_tokens"],
                output_tokens=usage["output_tokens"],
                latency_ms=usage["latency_ms"],
            )
        return content


def _diff_positions(diff: Diff) -> set[tuple[str, int, str]]:
    """Every (path, line, side) GitHub will accept as a review comment
    position in the current diff: a '-' line only exists on LEFT, a '+' or
    context line only exists on RIGHT."""
    positions: set[tuple[str, int, str]] = set()
    for file in diff.files:
        old_line = new_line = 0
        for raw_line in file.patch.splitlines():
            header = _HUNK_HEADER.match(raw_line)
            if header:
                old_line, new_line = int(header.group(1)), int(header.group(2))
                continue
            marker = raw_line[0] if raw_line else " "
            if marker == "+":
                positions.add((file.path, new_line, "RIGHT"))
                new_line += 1
            elif marker == "-":
                positions.add((file.path, old_line, "LEFT"))
                old_line += 1
            else:
                positions.add((file.path, new_line, "RIGHT"))
                old_line += 1
                new_line += 1
    return positions


async def post_findings(
    conn,
    vcs: VCSProvider,
    *,
    repo: str,
    pr_number: int,
    head_sha: str,
    findings: list[Finding],
    diff: Diff,
) -> None:
    """FR-8: claim each finding immediately before posting so a retried job
    with the same (repo, pr_number, head_sha, fingerprint) never double-posts.
    A single batched review, per FR-5/AC-1. Findings whose position isn't
    valid in `diff` are dropped unclaimed first -- see module docstring."""
    positions = _diff_positions(diff)
    postable = [finding for finding in findings if (finding.path, finding.line, finding.side) in positions]
    async with conn.transaction():
        to_post = [
            finding
            for finding in postable
            if await claim_finding(conn, repo=repo, pr_number=pr_number, head_sha=head_sha, finding=finding)
        ]
        if to_post:
            summary = f"{len(to_post)} finding(s) from automated review"
            await vcs.post_review(repo, pr_number, to_post, summary)


_MEMORY_CHECKPOINTER = MemorySaver()  # NFR-4 caveat: process-lifetime only, see module docstring


async def review_pr(ctx: dict, repo: str, pr_number: int, head_sha: str) -> None:
    pool: asyncpg.Pool = ctx["db_pool"]
    vcs: VCSProvider = ctx["vcs"]
    llm_provider: LLMProviderWithUsage = ctx["llm_provider"]

    llm = AuditingLLMProvider(llm_provider, pool, repo=repo, pr_number=pr_number)
    graph = build_graph(llm, checkpointer=_MEMORY_CHECKPOINTER)
    diff = await vcs.fetch_diff(repo, pr_number)
    config = {"configurable": {"thread_id": f"{repo}#{pr_number}#{head_sha}"}}

    result = await graph.ainvoke(
        {"diff": diff, "findings": [], "to_post": [], "discarded": []}, config=config
    )
    if "__interrupt__" in result:
        return  # paused for HITL -- /hitl/approve resumes this thread_id later

    async with pool.acquire() as conn:
        await post_findings(
            conn,
            vcs,
            repo=repo,
            pr_number=pr_number,
            head_sha=diff.head_sha,
            findings=result["to_post"],
            diff=diff,
        )


async def enqueue_review(redis: ArqRedis, repo: str, pr_number: int, head_sha: str) -> None:
    await redis.enqueue_job("review_pr", repo, pr_number, head_sha)


class WorkerSettings:
    functions = [review_pr]
    redis_settings = RedisSettings()
