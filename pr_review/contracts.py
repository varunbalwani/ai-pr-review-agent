"""Shared contracts every Phase 1 component depends on: the Finding schema
lens agents emit, and the VCSProvider port that abstracts the GitHub App
(and any future host) behind the same interface."""

from typing import Literal, Protocol

from pydantic import BaseModel


class Finding(BaseModel):
    path: str
    line: int
    side: Literal["LEFT", "RIGHT"]
    category: Literal["security", "quality", "test_coverage", "docs"]
    severity: Literal["low", "medium", "high"]
    confidence: float
    rationale: str
    suggested_patch: str | None = None
    fingerprint: str  # hash(path, normalized_snippet, category) — not line number
    snippet: str | None = None  # normalized text the fingerprint was hashed from; re-anchoring (FR-7) needs it back


class DiffFile(BaseModel):
    path: str
    status: Literal["added", "modified", "removed", "renamed"]
    patch: str


class Diff(BaseModel):
    head_sha: str
    files: list[DiffFile]


class VCSProvider(Protocol):
    async def fetch_diff(self, repo: str, pr_number: int) -> Diff: ...
    async def fetch_file(self, repo: str, path: str, ref: str) -> str: ...
    async def post_review(
        self, repo: str, pr_number: int, findings: list[Finding], summary: str
    ) -> None: ...
    async def resolve_comment(self, repo: str, pr_number: int, comment_id: str) -> None: ...
