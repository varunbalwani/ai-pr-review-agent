"""GitHub webhook receiver: HMAC verification on the raw body, event
filtering, then hands off to whatever enqueues the review job. The ARQ/Redis
wiring itself lives outside this module so the HTTP + auth logic is testable
without a running queue."""

import hashlib
import hmac
from typing import Awaitable, Callable

from fastapi import FastAPI, Header, HTTPException, Request

EnqueueFn = Callable[[str, int, str], Awaitable[None]]

_HANDLED_ACTIONS = {"opened", "synchronize", "reopened"}


def verify_signature(secret: str, payload: bytes, signature_header: str | None) -> bool:
    if not signature_header or not signature_header.startswith("sha256="):
        return False
    expected = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature_header.removeprefix("sha256="))


def create_app(webhook_secret: str, enqueue: EnqueueFn) -> FastAPI:
    app = FastAPI()

    @app.post("/webhooks/github")
    async def receive_webhook(
        request: Request,
        x_hub_signature_256: str | None = Header(default=None),
        x_github_event: str | None = Header(default=None),
    ):
        body = await request.body()
        if not verify_signature(webhook_secret, body, x_hub_signature_256):
            raise HTTPException(status_code=401, detail="invalid signature")

        if x_github_event != "pull_request":
            return {"status": "ignored", "reason": "unhandled event type"}

        payload = await request.json()
        action = payload.get("action")
        if action not in _HANDLED_ACTIONS:
            return {"status": "ignored", "reason": f"unhandled action: {action}"}

        repo = payload["repository"]["full_name"]
        pr_number = payload["pull_request"]["number"]
        head_sha = payload["pull_request"]["head"]["sha"]

        await enqueue(repo, pr_number, head_sha)
        return {"status": "queued"}

    return app
