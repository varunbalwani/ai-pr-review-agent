import json

import pytest

from pr_review.agents import _fingerprint, run_all_lenses, run_lens
from pr_review.contracts import Diff, DiffFile


class _StubLLM:
    def __init__(self, reply: str | list[str]):
        self._replies = reply if isinstance(reply, list) else [reply]
        self.calls: list[tuple[str, str]] = []

    async def complete(self, role, prompt, *, system=None):
        self.calls.append((role, prompt))
        return self._replies[len(self.calls) - 1] if len(self._replies) > 1 else self._replies[0]


def _diff(*files: DiffFile) -> Diff:
    return Diff(head_sha="deadbeef", files=list(files))


@pytest.mark.asyncio
async def test_run_lens_parses_json_reply_into_findings():
    reply = json.dumps(
        [
            {
                "path": "app.py",
                "line": 10,
                "side": "RIGHT",
                "snippet": "os.system(user_input)",
                "severity": "high",
                "confidence": 0.9,
                "rationale": "command injection",
                "suggested_patch": None,
            }
        ]
    )
    llm = _StubLLM(reply)
    diff = _diff(DiffFile(path="app.py", status="modified", patch="+os.system(user_input)"))

    findings = await run_lens("security", diff, llm)

    assert len(findings) == 1
    finding = findings[0]
    assert finding.category == "security"
    assert finding.path == "app.py"
    assert finding.rationale == "command injection"
    assert finding.fingerprint == _fingerprint("app.py", "os.system(user_input)", "security")
    assert finding.snippet == "os.system(user_input)"


@pytest.mark.asyncio
async def test_run_lens_returns_empty_list_for_empty_reply():
    llm = _StubLLM("[]")
    diff = _diff(DiffFile(path="app.py", status="modified", patch="+pass"))

    findings = await run_lens("quality", diff, llm)

    assert findings == []


@pytest.mark.asyncio
async def test_run_lens_strips_markdown_fences():
    reply = "```json\n[]\n```"
    llm = _StubLLM(reply)
    diff = _diff(DiffFile(path="app.py", status="modified", patch="+pass"))

    findings = await run_lens("quality", diff, llm)

    assert findings == []


@pytest.mark.asyncio
async def test_docs_lens_skips_llm_call_when_no_public_surface_touched():
    llm = _StubLLM("[]")
    diff = _diff(DiffFile(path="pr_review/internal.py", status="modified", patch="+    x = 1"))

    findings = await run_lens("docs", diff, llm)

    assert findings == []
    assert llm.calls == []


@pytest.mark.asyncio
async def test_docs_lens_runs_when_readme_touched():
    reply = json.dumps(
        [
            {
                "path": "README.md",
                "line": 3,
                "side": "RIGHT",
                "snippet": "## New flag",
                "severity": "low",
                "confidence": 0.6,
                "rationale": "flag undocumented",
                "suggested_patch": None,
            }
        ]
    )
    llm = _StubLLM(reply)
    diff = _diff(DiffFile(path="README.md", status="modified", patch="+## New flag"))

    findings = await run_lens("docs", diff, llm)

    assert len(findings) == 1
    assert llm.calls != []


@pytest.mark.asyncio
async def test_docs_lens_runs_when_public_function_added():
    llm = _StubLLM("[]")
    diff = _diff(
        DiffFile(path="pr_review/api.py", status="modified", patch="+def public_entrypoint():\n+    pass")
    )

    await run_lens("docs", diff, llm)

    assert llm.calls != []


@pytest.mark.asyncio
async def test_docs_lens_ignores_private_function_added():
    llm = _StubLLM("[]")
    diff = _diff(
        DiffFile(path="pr_review/api.py", status="modified", patch="+def _helper():\n+    pass")
    )

    findings = await run_lens("docs", diff, llm)

    assert findings == []
    assert llm.calls == []


def test_fingerprint_is_stable_across_whitespace_and_sensitive_to_category():
    a = _fingerprint("app.py", "os.system(x)", "security")
    b = _fingerprint("app.py", "  os.system(x)  ", "security")
    c = _fingerprint("app.py", "os.system(x)", "quality")

    assert a == b
    assert a != c


@pytest.mark.asyncio
async def test_run_all_lenses_gathers_findings_from_every_category():
    empty = json.dumps([])
    llm = _StubLLM(empty)
    diff = _diff(DiffFile(path="app.py", status="modified", patch="+pass"))

    findings = await run_all_lenses(diff, llm)

    assert findings == []
    called_roles = {role for role, _ in llm.calls}
    assert called_roles == {"security", "quality", "test_coverage"}  # docs skipped, no LLM call
