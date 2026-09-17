import hashlib
import hmac
import json

from fastapi.testclient import TestClient

from pr_review.webhook import create_app

SECRET = "test-secret"


def _sign(body: bytes) -> str:
    return "sha256=" + hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest()


def _pr_payload(action: str = "opened") -> bytes:
    return json.dumps(
        {
            "action": action,
            "repository": {"full_name": "varunbalwani/In-memory-Pub-sub-in-Go"},
            "pull_request": {"number": 7, "head": {"sha": "deadbeef"}},
        }
    ).encode()


def _client(enqueued: list):
    async def enqueue(repo, pr_number, head_sha):
        enqueued.append((repo, pr_number, head_sha))

    return TestClient(create_app(SECRET, enqueue))


def test_valid_pull_request_event_enqueues_job():
    enqueued = []
    body = _pr_payload("opened")
    response = _client(enqueued).post(
        "/webhooks/github",
        content=body,
        headers={
            "X-Hub-Signature-256": _sign(body),
            "X-GitHub-Event": "pull_request",
            "Content-Type": "application/json",
        },
    )
    assert response.status_code == 200
    assert response.json() == {"status": "queued"}
    assert enqueued == [("varunbalwani/In-memory-Pub-sub-in-Go", 7, "deadbeef")]


def test_invalid_signature_is_rejected():
    enqueued = []
    body = _pr_payload("opened")
    response = _client(enqueued).post(
        "/webhooks/github",
        content=body,
        headers={
            "X-Hub-Signature-256": "sha256=" + "0" * 64,
            "X-GitHub-Event": "pull_request",
        },
    )
    assert response.status_code == 401
    assert enqueued == []


def test_missing_signature_is_rejected():
    enqueued = []
    body = _pr_payload("opened")
    response = _client(enqueued).post(
        "/webhooks/github", content=body, headers={"X-GitHub-Event": "pull_request"}
    )
    assert response.status_code == 401
    assert enqueued == []


def test_non_pull_request_event_is_ignored():
    enqueued = []
    body = _pr_payload("opened")
    response = _client(enqueued).post(
        "/webhooks/github",
        content=body,
        headers={"X-Hub-Signature-256": _sign(body), "X-GitHub-Event": "issue_comment"},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "ignored"
    assert enqueued == []


def test_unhandled_pull_request_action_is_ignored():
    enqueued = []
    body = _pr_payload("closed")
    response = _client(enqueued).post(
        "/webhooks/github",
        content=body,
        headers={"X-Hub-Signature-256": _sign(body), "X-GitHub-Event": "pull_request"},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "ignored"
    assert enqueued == []
