# Product specification — pr-review-pipeline

> Status: draft. A coding agent must not implement product code until this specification is approved through Genesis.

## Problem

Manually reviewing every PR for security issues, quality problems, missing tests, and doc drift
is slow and inconsistent. The goal is an automated, multi-lens PR review system that posts
high-confidence findings inline on GitHub PRs and routes uncertain findings to a human, without
spamming false positives or duplicate comments.

## Users

- The repo owner/maintainer (Varun) — installs the GitHub App, approves HITL-queued findings.
- Any contributor whose PR triggers automated review comments — sees inline findings, is not
  otherwise an actor in the system.

## Functional requirements

- FR-1: On any PR opened/updated on a GitHub repo where the App is installed, the system fetches
  the diff and PR metadata via GitHub App auth (no PAT).
- FR-2: Four lens agents (security, quality, test_coverage, docs) run in parallel over the diff
  and each emit zero or more `Finding` objects per the shared schema (path, line, side, category,
  severity, confidence, rationale, suggested_patch, fingerprint).
- FR-3: The docs lens fires only when the diff touches public API surface, config schema, CLI
  flags, or files referenced by README/docs.
- FR-4: Every raw finding passes through a single verify step that assigns a confidence score;
  routing (auto-post / HITL / discard) is a pure function of that score — no second gate.
- FR-5: Findings above the auto-post threshold are posted as inline PR comments in one batched
  review; findings below the discard floor are logged but not surfaced; findings in between pause
  the run via `interrupt()` and wait for human approval.
- FR-6: A human approver can approve or reject a HITL-queued finding through an authenticated,
  authorized, ownership-checked endpoint; approval resumes the paused run.
- FR-7: On resume, findings re-anchor by fingerprint against the current diff — post normally if
  the line still exists, else fall back to a file-level note in the summary; never post against a
  stale `head_sha`.
- FR-8: The post node is idempotent — re-checks `(repo, pr_number, head_sha, fingerprint)`
  immediately before writing so retries never double-post.
- FR-9: Every LLM call is recorded to an audit table with model, input/output tokens, latency,
  agent, repo, pr_number, timestamp, decision — no raw diff/completion content by default.
- FR-10: PR outcome (merged/closed) and comment resolution (resolved/dismissed) are captured for
  later analysis; v1 does not use this data to change agent behavior.

## Non-functional requirements

- NFR-1: Webhook requests are verified via HMAC over the raw request body before any processing.
- NFR-2: LLM calls run against a locally-hosted Ollama model (`qwen3:8b`) through a provider
  abstraction (mirroring `VCSProvider`) so a hosted provider (e.g. Anthropic) can be swapped in
  later without rewriting call sites.
- NFR-3: Models are addressed by config key (`AGENT_MODEL_SECURITY`/`_QUALITY`/`_TEST_COVERAGE`/
  `_DOCS`/`_VERIFY`), never hardcoded, even though all five currently point at the same local
  model.
- NFR-4: The system must not block ARQ worker concurrency while a finding is paused for human
  approval — `interrupt()` persists a checkpoint and returns the job.
- NFR-5: The HITL approval endpoint enforces authentication, authorization, and an ownership
  check (an approver is scoped to the installation/repo they're approving for).

## Constraints

- GitHub App only (not a PAT) for VCS auth.
- Single GitHub account/tenant; GitHub host only (no Bitbucket/GitLab in v1).
- Self-hosted webhook receiver. Phase 1 runs on a local dev machine behind a tunnel
  (ngrok/cloudflared) — no hosting account yet.
- LLM inference is local Ollama (`qwen3:8b`) — no Anthropic API key available.
- Storage: Tiger Cloud Postgres (`pgvectorscale` reserved for Phase 2 RAG).
- Queue: Redis + ARQ.
- Orchestration: LangGraph with a checkpointed graph for HITL pause/resume.
- No call-graph/blast-radius agent in v1 — the quality lens uses diff context only in Phase 1.

## Non-goals

- No RAG-backed context retrieval in Phase 1 (Phase 2).
- No custom dashboard app in Phase 1 (Phase 2).
- No behavior adaptation from outcome data (confidence recalibration, prompt tuning) in v1.
- No Bitbucket/GitLab support.
- No multi-tenant/org-wide installation management.

## Acceptance criteria

- AC-1: A PR opened against `github.com/varunbalwani/In-memory-Pub-sub-in-Go` with a planted
  security issue, a quality bug, and an untested change produces a single batched review with
  correctly-anchored inline comments for each, posted only after HMAC-verified webhook processing.
- AC-2: Pushing a second commit after a finding is HITL-queued, then approving it, results in
  either a correctly re-anchored inline comment or a file-level fallback note in the summary —
  never a comment on a stale `head_sha`.
- AC-3: Retrying the same ARQ job does not produce a duplicate posted comment for the same
  `(repo, pr_number, head_sha, fingerprint)`.
- AC-4: The audit table contains one row per LLM call with model/tokens/latency populated and no
  raw diff or completion text.
- AC-5: The docs lens does not fire on a PR that touches only internal implementation with no
  public-surface change.
- AC-6: The system correctly processes a PR from a repo other than the test repo, with no
  repo-specific hardcoding (general-purpose install).

## Risks

- A single 8B local model is used for both a lens agent and the verify step, so the confidence
  gate is not an independent check of the same model's own output — likely miscalibrated
  confidence scores. Mitigation: treat thresholds as tunable, expect to widen the HITL band
  initially; revisit if a stronger or second model becomes available.
- Local-only deployment (Phase 1) means the webhook is only reachable while the dev machine and
  tunnel are up — GitHub webhook deliveries during downtime may be lost. Mitigation: acceptable
  for personal/dev use; revisit before relying on this as always-on infrastructure.
- No pgvectorscale/RAG in Phase 1 means lens agents reason from diff context alone, increasing
  the risk of missing findings that depend on broader codebase context.

## Open questions

- None blocking Phase 1. Phase 2 hosting and dashboard decisions are deferred.
