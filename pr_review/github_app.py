"""GitHub App auth: JWT minting and per-repo installation-token caching,
behind the VCSProvider port (decision: GitHub App only, no PAT). Endpoints
that GitHub only exposes via GraphQL (thread resolution) go through the
same authenticated client as the REST calls."""

import base64
import calendar
import time

import httpx
import jwt

from pr_review.contracts import Diff, DiffFile, Finding

_JWT_CLOCK_SKEW_SECONDS = 60
_JWT_TTL_SECONDS = 600
_TOKEN_REFRESH_MARGIN_SECONDS = 60


class GitHubAppProvider:
    """Implements VCSProvider using GitHub App JWT auth. One instance serves
    any repo the App is installed on; installation tokens are minted and
    cached per repo, not tied to a single hardcoded installation."""

    def __init__(
        self,
        app_id: str,
        private_key: str,
        base_url: str = "https://api.github.com",
        transport: httpx.BaseTransport | None = None,
    ):
        self.app_id = app_id
        self.private_key = private_key
        self.base_url = base_url
        self._transport = transport
        self._installation_tokens: dict[str, tuple[str, float]] = {}

    def _mint_jwt(self) -> str:
        now = int(time.time())
        payload = {
            "iat": now - _JWT_CLOCK_SKEW_SECONDS,
            "exp": now + _JWT_TTL_SECONDS,
            "iss": self.app_id,
        }
        return jwt.encode(payload, self.private_key, algorithm="RS256")

    def _client(self, token: str) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self.base_url,
            transport=self._transport,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
            },
            timeout=30,
        )

    async def _installation_token(self, repo: str) -> str:
        cached = self._installation_tokens.get(repo)
        if cached and cached[1] - _TOKEN_REFRESH_MARGIN_SECONDS > time.time():
            return cached[0]

        app_jwt = self._mint_jwt()
        async with self._client(app_jwt) as client:
            installation = await client.get(f"/repos/{repo}/installation")
            installation.raise_for_status()
            installation_id = installation.json()["id"]

            minted = await client.post(f"/app/installations/{installation_id}/access_tokens")
            minted.raise_for_status()
            body = minted.json()

        token = body["token"]
        expires_at = calendar.timegm(time.strptime(body["expires_at"], "%Y-%m-%dT%H:%M:%SZ"))
        self._installation_tokens[repo] = (token, expires_at)
        return token

    async def fetch_diff(self, repo: str, pr_number: int) -> Diff:
        token = await self._installation_token(repo)
        async with self._client(token) as client:
            pr = await client.get(f"/repos/{repo}/pulls/{pr_number}")
            pr.raise_for_status()
            head_sha = pr.json()["head"]["sha"]

            files_response = await client.get(f"/repos/{repo}/pulls/{pr_number}/files")
            files_response.raise_for_status()
            files = [
                DiffFile(path=f["filename"], status=f["status"], patch=f.get("patch", ""))
                for f in files_response.json()
            ]
        return Diff(head_sha=head_sha, files=files)

    async def fetch_file(self, repo: str, path: str, ref: str) -> str:
        token = await self._installation_token(repo)
        async with self._client(token) as client:
            response = await client.get(f"/repos/{repo}/contents/{path}", params={"ref": ref})
            response.raise_for_status()
        return base64.b64decode(response.json()["content"]).decode()

    async def post_review(
        self, repo: str, pr_number: int, findings: list[Finding], summary: str
    ) -> None:
        token = await self._installation_token(repo)
        comments = [
            {
                "path": finding.path,
                "line": finding.line,
                "side": finding.side,
                "body": finding.rationale,
            }
            for finding in findings
        ]
        async with self._client(token) as client:
            response = await client.post(
                f"/repos/{repo}/pulls/{pr_number}/reviews",
                json={"event": "COMMENT", "body": summary, "comments": comments},
            )
            response.raise_for_status()

    async def list_review_comments(self, repo: str, pr_number: int) -> list[dict]:
        """Not part of VCSProvider -- verification-only (scripts/verify_e2e.py)
        read of the raw inline review comments GitHub actually recorded."""
        token = await self._installation_token(repo)
        async with self._client(token) as client:
            response = await client.get(f"/repos/{repo}/pulls/{pr_number}/comments")
            response.raise_for_status()
        return response.json()

    async def resolve_comment(self, repo: str, pr_number: int, comment_id: str) -> None:
        token = await self._installation_token(repo)
        async with self._client(token) as client:
            response = await client.post(
                "/graphql",
                json={
                    "query": "mutation($id: ID!) { resolveReviewThread(input: {threadId: $id}) "
                    "{ thread { id } } }",
                    "variables": {"id": comment_id},
                },
            )
            response.raise_for_status()
