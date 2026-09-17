"""HITL approval endpoint (FR-6, NFR-5): a GitHub OAuth user token
identifies the approver, then an ownership check confirms they have
write access to the specific repo they're approving for -- scoped per
NFR-5, not a blanket admin flag. On approval, the paused LangGraph run
resumes and each to_post finding is re-anchored against a freshly fetched
diff (FR-7): its snippet still present at that path keeps it line-level,
otherwise it falls back to a file-level note. A fresh diff also means the
returned head_sha is always current, never the stale one the run paused
on. Posting to GitHub reuses worker.post_findings so the same FR-8
idempotency claim (claim_finding before post) applies here as it does on
review_pr's own auto-post path -- a retried /hitl/approve call never
double-posts. Both endpoints take head_sha to select the run's thread
(repo#pr_number#head_sha, matching worker.review_pr) rather than just
repo#pr_number, so a new commit landing on the same PR gets its own
independent thread instead of bleeding stale checkpoint state into --
or being unresolvably ambiguous with -- an still-paused earlier run."""

import re
from typing import Literal

import asyncpg
import httpx
from fastapi import FastAPI, HTTPException, Request
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from pr_review.contracts import Diff, Finding, VCSProvider
from pr_review.worker import post_findings

Anchor = Literal["line", "file_note"]

_OWNER_PERMISSIONS = {"admin", "maintain", "write"}
_HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _find_line(patch: str, side: str, normalized_snippet: str) -> int | None:
    """Walks a unified-diff hunk, tracking old/new line numbers, and returns
    the line number (on the side the finding is anchored to) whose content
    contains the snippet -- a '-' line only counts for LEFT, a '+' or
    context line only counts for RIGHT, since those are the lines that
    still exist on that side of the diff. Returns the first match if the
    snippet appears more than once on that side -- a genuinely duplicated
    snippet anchors to whichever occurrence comes first, not an error."""
    old_line = new_line = 0
    for raw_line in patch.splitlines():
        header = _HUNK_HEADER.match(raw_line)
        if header:
            old_line, new_line = int(header.group(1)), int(header.group(2))
            continue
        marker, content = (raw_line[0], raw_line[1:]) if raw_line else (" ", "")
        matches = normalized_snippet in _normalize(content)
        if marker == "+":
            if side == "RIGHT" and matches:
                return new_line
            new_line += 1
        elif marker == "-":
            if side == "LEFT" and matches:
                return old_line
            old_line += 1
        else:
            if matches:
                return new_line if side == "RIGHT" else old_line
            old_line += 1
            new_line += 1
    return None


async def authenticate_approver(token: str, *, transport: httpx.BaseTransport | None = None) -> str:
    """FR-6/NFR-5 authentication: resolve a GitHub OAuth user token to its login."""
    async with httpx.AsyncClient(
        base_url="https://api.github.com", transport=transport, timeout=10
    ) as client:
        response = await client.get(
            "/user",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
        )
    if response.status_code == 401:
        raise PermissionError("invalid or expired GitHub token")
    response.raise_for_status()
    return response.json()["login"]


async def check_repo_ownership(
    login: str, repo: str, token: str, *, transport: httpx.BaseTransport | None = None
) -> bool:
    """NFR-5 ownership check: the approver must have write access to *this* repo, not just a valid token."""
    async with httpx.AsyncClient(
        base_url="https://api.github.com", transport=transport, timeout=10
    ) as client:
        response = await client.get(
            f"/repos/{repo}/collaborators/{login}/permission",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
        )
    if response.status_code != 200:
        return False  # fail closed: 404 (not a collaborator), 403 (caller's own token lacks access), etc.
    return response.json()["permission"] in _OWNER_PERMISSIONS


def reanchor_finding(finding: Finding, diff: Diff) -> tuple[Anchor, int | None]:
    """FR-7: resolves the finding's *current* line by locating its snippet
    in the current diff, since content survives line-number drift on
    resume even when the number doesn't (DECISION-0d2b1dcb). file_note
    (line None) when the snippet no longer exists at that path/side."""
    if not finding.snippet:
        return "file_note", None
    normalized_snippet = _normalize(finding.snippet)
    for file in diff.files:
        if file.path != finding.path:
            continue
        line = _find_line(file.patch, finding.side, normalized_snippet)
        if line is not None:
            return "line", line
    return "file_note", None


def create_app(
    graph: CompiledStateGraph,
    vcs: VCSProvider,
    pool: asyncpg.Pool,
    *,
    transport: httpx.BaseTransport | None = None,
) -> FastAPI:
    app = FastAPI()

    @app.get("/hitl/pending")
    async def pending(repo: str, pr_number: int, head_sha: str, request: Request):
        """Read-only companion to /hitl/approve: lets an approver see what's
        paused before deciding, without resuming the run (aget_state doesn't
        touch the checkpoint, unlike Command(resume=...)). head_sha selects
        which run's thread to inspect -- the thread that paused, since a
        newer commit on the same PR gets its own independent thread/run."""
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="missing bearer token")
        token = auth_header.removeprefix("Bearer ")

        try:
            login = await authenticate_approver(token, transport=transport)
        except PermissionError:
            raise HTTPException(status_code=401, detail="invalid token")
        if not await check_repo_ownership(login, repo, token, transport=transport):
            raise HTTPException(status_code=403, detail="not authorized for this repo")

        config = {"configurable": {"thread_id": f"{repo}#{pr_number}#{head_sha}"}}
        state = await graph.aget_state(config)
        findings = [
            finding
            for task in state.tasks
            for interrupt in task.interrupts
            for finding in interrupt.value.get("hitl_findings", [])
        ]
        return {"hitl_findings": findings}

    @app.post("/hitl/approve")
    async def approve(request: Request):
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="missing bearer token")
        token = auth_header.removeprefix("Bearer ")

        try:
            login = await authenticate_approver(token, transport=transport)
        except PermissionError:
            raise HTTPException(status_code=401, detail="invalid token")

        body = await request.json()
        repo = body["repo"]
        pr_number = body["pr_number"]
        head_sha = body["head_sha"]
        approvals: dict[str, bool] = body["approvals"]

        if not await check_repo_ownership(login, repo, token, transport=transport):
            raise HTTPException(status_code=403, detail="not authorized for this repo")

        config = {"configurable": {"thread_id": f"{repo}#{pr_number}#{head_sha}"}}
        result = await graph.ainvoke(Command(resume=approvals), config=config)

        diff = await vcs.fetch_diff(repo, pr_number)
        reanchored = []
        to_post_response = []
        for finding in result["to_post"]:
            anchor, line = reanchor_finding(finding, diff)
            if anchor == "line":
                finding = finding.model_copy(update={"line": line})
            reanchored.append(finding)
            to_post_response.append({"finding": finding.model_dump(mode="json"), "anchor": anchor})

        async with pool.acquire() as conn:
            await post_findings(
                conn,
                vcs,
                repo=repo,
                pr_number=pr_number,
                head_sha=diff.head_sha,
                findings=reanchored,
                diff=diff,
            )

        return {
            "head_sha": diff.head_sha,
            "to_post": to_post_response,
            "discarded": [finding.model_dump(mode="json") for finding in result["discarded"]],
        }

    return app
