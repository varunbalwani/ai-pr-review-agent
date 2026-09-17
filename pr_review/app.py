"""Assembles the whole Phase 1 system as one process: the FastAPI webhook
+ HITL endpoints, and an ARQ worker run in-process (not arq's separate CLI)
as a background asyncio task on the same event loop.

Running the worker in-process, not as a separate `arq` CLI process, is
required for HITL to work at all under NFR-4's current limitation: the
graph handed to the HITL endpoint and the graph review_pr builds per job
must share the exact same MemorySaver *object* (worker._MEMORY_CHECKPOINTER)
for a resume to see the pause it's resuming -- that only holds within one
process. A real deployment gets a separate worker process once TASK-storage's
Postgres-backed checkpointer (DECISION-4fed71f4) exists; until then, this
is a deliberate single-process Phase 1 shape, not an oversight.

Every environment read and the private-key file open happen inside
_lifespan, not at import time, so `import pr_review.app` succeeds with no
environment configured at all -- only actually starting the app requires
DATABASE_URL, GITHUB_APP_ID, GITHUB_APP_PRIVATE_KEY_PATH, GITHUB_WEBHOOK_SECRET,
Redis, and OPENAI_API_KEY (OPENAI_BASE_URL/OPENAI_DEFAULT_MODEL optional --
default is a hosted OpenAI-compatible endpoint, not local Ollama).

Run with: uvicorn pr_review.app:app --env-file .env --host 0.0.0.0 --port 8000
"""

import asyncio
import os
from contextlib import asynccontextmanager

import asyncpg
from arq.connections import RedisSettings, create_pool as create_redis_pool
from arq.worker import Worker
from fastapi import FastAPI

from pr_review import hitl, webhook
from pr_review.github_app import GitHubAppProvider
from pr_review.llm import OpenAICompatibleProvider
from pr_review.orchestration import build_graph
from pr_review.worker import _MEMORY_CHECKPOINTER, enqueue_review, review_pr


class _ResumeOnlyLLM:
    """Bound into the graph passed to the HITL endpoint. Never actually
    called: resuming a paused run re-enters LangGraph's 'route' node
    directly from its checkpoint -- 'analyze'/'verify' already ran before
    the pause and are never replayed."""

    async def complete(self, role, prompt, *, system=None) -> str:
        raise NotImplementedError("resume must never re-run analyze/verify")


def _merge_routes(app: FastAPI, *sub_apps: FastAPI) -> None:
    """Appends each sub-app's own routes into `app` -- webhook's
    /webhooks/github and hitl's /hitl/approve share no path prefix, so
    this is safe without mounting. Each sub-app's own auto-added
    docs/openapi routes are skipped; `app` already has its own copy."""
    default_paths = {route.path for route in FastAPI().routes}
    for sub_app in sub_apps:
        for route in sub_app.routes:
            if getattr(route, "path", None) not in default_paths:
                app.router.routes.append(route)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    with open(os.environ["GITHUB_APP_PRIVATE_KEY_PATH"]) as key_file:
        vcs = GitHubAppProvider(os.environ["GITHUB_APP_ID"], key_file.read())
    llm_provider = OpenAICompatibleProvider()

    db_pool = await asyncpg.create_pool(os.environ["DATABASE_URL"])
    redis_settings = RedisSettings()
    redis = await create_redis_pool(redis_settings)

    async def enqueue(repo: str, pr_number: int, head_sha: str) -> None:
        await enqueue_review(redis, repo, pr_number, head_sha)

    async def worker_startup(ctx: dict) -> None:
        ctx["db_pool"] = db_pool
        ctx["vcs"] = vcs
        ctx["llm_provider"] = llm_provider

    worker = Worker(
        functions=[review_pr],
        redis_settings=redis_settings,
        on_startup=worker_startup,
        handle_signals=False,  # uvicorn owns signal handling in this process
    )

    webhook_app = webhook.create_app(os.environ["GITHUB_WEBHOOK_SECRET"], enqueue)
    hitl_graph = build_graph(_ResumeOnlyLLM(), checkpointer=_MEMORY_CHECKPOINTER)
    hitl_app = hitl.create_app(hitl_graph, vcs, db_pool)
    _merge_routes(app, webhook_app, hitl_app)

    worker_task = asyncio.create_task(worker.async_run())
    try:
        yield
    finally:
        await worker.close()  # signals the run loop to stop and drains in-flight jobs
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass
        await redis.close()
        await db_pool.close()


app = FastAPI(lifespan=_lifespan)
