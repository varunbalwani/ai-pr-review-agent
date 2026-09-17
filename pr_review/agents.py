"""Four lens agents mapping a diff to zero or more Finding objects (FR-2).
Each lens sends the diff to its configured model and parses the JSON reply;
fingerprint is computed here from the snippet the model quotes rather than
trusted from model output, since Finding.fingerprint must stay stable
across line-number shifts on resume (contracts.py). The docs lens only
runs when the diff touches public surface (FR-3)."""

import asyncio
import hashlib
import json
import logging
import re
from typing import Literal

from pydantic import ValidationError

from pr_review.contracts import Diff, Finding
from pr_review.llm import LLMProvider

logger = logging.getLogger(__name__)

LensCategory = Literal["security", "quality", "test_coverage", "docs"]

_ALL_CATEGORIES: tuple[LensCategory, ...] = ("security", "quality", "test_coverage", "docs")

_SYSTEM_PROMPTS: dict[LensCategory, str] = {
    "security": (
        "You are a security reviewer. Inspect the diff for vulnerabilities: "
        "injection, hardcoded secrets, auth/authz gaps, unsafe deserialization, "
        "SSRF, path traversal. Ignore style issues."
    ),
    "quality": (
        "You are a code quality reviewer. Inspect the diff for correctness bugs, "
        "unhandled edge cases, and maintainability problems. Ignore style/formatting."
    ),
    "test_coverage": (
        "You are a test coverage reviewer. Inspect the diff for new or changed "
        "logic that lacks a corresponding test, or tests that don't exercise the "
        "new behavior."
    ),
    "docs": (
        "You are a docs reviewer. Inspect the diff for public API, config, or CLI "
        "surface changes that are not reflected in README/docs."
    ),
}

_OUTPUT_INSTRUCTIONS = (
    "Respond with a JSON array only, no prose, no markdown fences. Each element: "
    '{"path": str, "line": int, "side": "LEFT"|"RIGHT", "snippet": str, '
    '"severity": "low"|"medium"|"high", "confidence": number 0-1, '
    '"rationale": str, "suggested_patch": str|null}. '
    "Return [] if there is nothing to report."
)

_PUBLIC_SURFACE_PATH_PATTERNS = (
    re.compile(r"(^|/)README(\.\w+)?$", re.IGNORECASE),
    re.compile(r"(^|/)docs/"),
    re.compile(r"cli", re.IGNORECASE),
    re.compile(r"\.(ya?ml|toml|ini|cfg)$"),
)
_PUBLIC_SYMBOL_PATTERN = re.compile(r"^\+\s*(def|class)\s+([A-Za-z_]\w*)")


def _touches_public_surface(diff: Diff) -> bool:
    for file in diff.files:
        if any(pattern.search(file.path) for pattern in _PUBLIC_SURFACE_PATH_PATTERNS):
            return True
        for line in file.patch.splitlines():
            match = _PUBLIC_SYMBOL_PATTERN.match(line)
            if match and not match.group(2).startswith("_"):
                return True
    return False


def _normalize_snippet(snippet: str) -> str:
    return " ".join(snippet.split())


def _fingerprint(path: str, snippet: str, category: str) -> str:
    key = f"{path}|{_normalize_snippet(snippet)}|{category}"
    return hashlib.sha256(key.encode()).hexdigest()


def _render_diff(diff: Diff) -> str:
    return "\n\n".join(f"--- {file.path} ({file.status}) ---\n{file.patch}" for file in diff.files)


_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _strip_markdown_fence(text: str) -> str:
    """Extracts fenced content by regex match, not prefix/suffix trim: a
    weaker model sometimes appends prose after the closing fence, which
    removesuffix("```") can't see past -- leaving that trailing text in
    the string to break json.loads() with 'Extra data' past the array."""
    match = _FENCE_PATTERN.search(text)
    return match.group(1).strip() if match else text.strip()


async def run_lens(category: LensCategory, diff: Diff, llm: LLMProvider) -> list[Finding]:
    if category == "docs" and not _touches_public_surface(diff):
        return []

    prompt = f"{_render_diff(diff)}\n\n{_OUTPUT_INSTRUCTIONS}"
    reply = await llm.complete(category, prompt, system=_SYSTEM_PROMPTS[category])
    raw_findings = json.loads(_strip_markdown_fence(reply))

    findings = []
    for raw in raw_findings:
        try:
            findings.append(
                Finding(
                    path=raw["path"],
                    line=raw["line"],
                    side=raw["side"],
                    category=category,
                    severity=raw["severity"],
                    confidence=raw["confidence"],
                    rationale=raw["rationale"],
                    suggested_patch=raw.get("suggested_patch"),
                    fingerprint=_fingerprint(raw["path"], raw["snippet"], category),
                    snippet=_normalize_snippet(raw["snippet"]),
                )
            )
        except (KeyError, ValidationError):
            # A weaker/free model doesn't always match the requested schema
            # exactly; one malformed finding shouldn't drop the whole lens.
            logger.warning("%s lens: dropping malformed finding: %r", category, raw)

    return findings


async def run_all_lenses(diff: Diff, llm: LLMProvider) -> list[Finding]:
    results = await asyncio.gather(*(run_lens(category, diff, llm) for category in _ALL_CATEGORIES))
    return [finding for lens_findings in results for finding in lens_findings]
