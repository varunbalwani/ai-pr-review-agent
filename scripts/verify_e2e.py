#!/usr/bin/env python3
"""TASK-e2e-verification's gate: checks observable evidence for AC-1..AC-6
against a real PR you've already driven through the running system (opened
a planted-issue PR, pushed a second commit, retried a job, etc. -- this
script verifies outcomes, it does not orchestrate the scenario).

AC-1/AC-2/AC-5 depend on live PR state this script can't conjure on its
own -- it checks what's observable (comments exist, the claimed-finding
count matches the posted-comment count, audit rows are populated) and
prints the rest (each comment's path/line/commit_id) for you to judge
against what you actually planted or pushed.

AC-6 (cross-repo, no hardcoding) is verified by running this script twice,
once per repo -- nothing in the script or the system it's checking is
tied to one repo.

Usage:
  python scripts/verify_e2e.py --repo OWNER/REPO --pr-number N [--head-sha SHA]

Exits non-zero on any failed check, and non-zero (not a silent pass) if
required environment/arguments are missing.
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo isn't packaged/installed

import asyncpg
import httpx

from pr_review.github_app import GitHubAppProvider


def _fail(message: str) -> None:
    print(f"FAIL: {message}")


def _ok(message: str) -> None:
    print(f"OK:   {message}")


async def _check_audit_log(pool: asyncpg.Pool, repo: str, pr_number: int) -> bool:
    """AC-4: one row per LLM call, model/tokens/latency populated, no raw
    diff/completion content column exists on the table at all."""
    passed = True
    async with pool.acquire() as conn:
        columns = {
            r["column_name"]
            for r in await conn.fetch(
                "SELECT column_name FROM information_schema.columns WHERE table_name = 'audit_log'"
            )
        }
        forbidden = columns & {"prompt", "completion", "diff", "raw_content", "content"}
        if forbidden:
            _fail(f"audit_log has raw-content-shaped column(s): {forbidden}")
            passed = False
        else:
            _ok("audit_log schema carries no raw diff/completion content column")

        rows = await conn.fetch(
            "SELECT agent, model, input_tokens, output_tokens, latency_ms FROM audit_log "
            "WHERE repo = $1 AND pr_number = $2",
            repo,
            pr_number,
        )
    if not rows:
        _fail(f"no audit_log rows for {repo}#{pr_number} -- was a review actually run?")
        return False

    by_agent: dict[str, int] = {}
    for row in rows:
        by_agent[row["agent"]] = by_agent.get(row["agent"], 0) + 1
        if row["model"] is None or row["input_tokens"] is None or row["output_tokens"] is None or row["latency_ms"] is None:
            _fail(f"audit_log row for agent={row['agent']} has a null model/tokens/latency field: {dict(row)}")
            passed = False
    if passed:
        _ok(f"audit_log has {len(rows)} row(s) with model/tokens/latency populated: {by_agent}")

    if "docs" in by_agent:
        print(
            "NOTE (AC-5): a 'docs' audit row exists for this PR -- the docs lens *did* run. "
            "Confirm that's expected (the diff touches public surface) or it's a false positive."
        )
    else:
        print(
            "NOTE (AC-5): no 'docs' audit row for this PR -- the docs lens was skipped "
            "(agents.py returns before calling the LLM when nothing public-facing changed). "
            "Confirm that matches what you pushed."
        )
    return passed


async def _check_claims_match_posted_comments(
    pool: asyncpg.Pool, repo: str, pr_number: int, comment_count: int
) -> bool:
    """AC-3: retrying the same job must never double-post. The UNIQUE
    constraint on posted_findings makes a DB-level duplicate structurally
    impossible, so the check that actually matters is whether every
    claimed finding made it to GitHub: ASSUMPTION-cdc6fa50's crash-drop gap
    (a claim that succeeds but the post never happens) would show up here
    as fewer GitHub comments than claimed rows, not as a DB duplicate."""
    async with pool.acquire() as conn:
        claimed = await conn.fetch(
            "SELECT fingerprint, head_sha FROM posted_findings WHERE repo = $1 AND pr_number = $2",
            repo,
            pr_number,
        )
    if not claimed:
        _fail(f"no posted_findings rows for {repo}#{pr_number} -- nothing was claimed for posting")
        return False
    if len(claimed) != comment_count:
        _fail(
            f"{len(claimed)} finding(s) claimed for posting but {comment_count} comment(s) exist on "
            f"GitHub -- mismatch suggests ASSUMPTION-cdc6fa50's claim-then-crash gap or a duplicate post"
        )
        return False
    _ok(f"{len(claimed)} finding(s) claimed for posting, matching {comment_count} posted GitHub comment(s)")
    return True


async def _check_review_comments(vcs: GitHubAppProvider, repo: str, pr_number: int) -> list[dict] | None:
    """AC-1/AC-2: inline comments exist, and their commit_id is printed
    per-comment (not asserted equal to a single 'correct' sha -- that
    depends on which push you're verifying against) so you can confirm
    re-anchoring/staleness by eye against what you actually pushed."""
    comments = await vcs.list_review_comments(repo, pr_number)
    if not comments:
        _fail(f"no inline review comments found on {repo}#{pr_number}")
        return None
    _ok(f"{len(comments)} inline review comment(s) found:")
    for comment in comments:
        print(f"      path={comment['path']} line={comment.get('line')} commit_id={comment['commit_id']}")
    return comments


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="owner/repo, e.g. varunbalwani/In-memory-Pub-sub-in-Go")
    parser.add_argument("--pr-number", required=True, type=int)
    args = parser.parse_args()

    database_url = os.environ.get("DATABASE_URL")
    app_id = os.environ.get("GITHUB_APP_ID")
    key_path = os.environ.get("GITHUB_APP_PRIVATE_KEY_PATH")
    if not database_url or not app_id or not key_path:
        _fail("DATABASE_URL, GITHUB_APP_ID, GITHUB_APP_PRIVATE_KEY_PATH must all be set")
        return 1

    with open(key_path) as key_file:
        vcs = GitHubAppProvider(app_id, key_file.read())
    pool = await asyncpg.create_pool(database_url)

    try:
        comments = await _check_review_comments(vcs, args.repo, args.pr_number)
        results = [
            comments is not None,
            await _check_claims_match_posted_comments(
                pool, args.repo, args.pr_number, len(comments) if comments else 0
            ),
            await _check_audit_log(pool, args.repo, args.pr_number),
        ]
    except httpx.HTTPStatusError as exc:
        _fail(f"GitHub API request failed: {exc}")
        return 1
    except asyncpg.PostgresError as exc:
        _fail(f"database query failed: {exc}")
        return 1
    finally:
        await pool.close()

    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
