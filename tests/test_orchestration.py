import json

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from pr_review.contracts import Diff, DiffFile, Finding
from pr_review.orchestration import build_graph, route_finding, verify_finding

_SECURITY_REPLY = json.dumps(
    [
        {
            "path": "app.py",
            "line": 1,
            "side": "RIGHT",
            "snippet": "x = 1",
            "severity": "medium",
            "confidence": 0.5,
            "rationale": "placeholder",
            "suggested_patch": None,
        }
    ]
)


class _StubLLM:
    """Every lens except security reports nothing; verify always returns a
    fixed, configurable confidence so tests can drive routing directly."""

    def __init__(self, verify_confidence: float = 0.95):
        self.verify_calls = 0
        self._verify_confidence = verify_confidence

    async def complete(self, role, prompt, *, system=None):
        if role == "verify":
            self.verify_calls += 1
            return json.dumps({"confidence": self._verify_confidence})
        if role == "security":
            return _SECURITY_REPLY
        return "[]"


def _diff() -> Diff:
    return Diff(
        head_sha="deadbeef", files=[DiffFile(path="app.py", status="modified", patch="+x = 1")]
    )


def test_route_finding_is_a_pure_function_of_confidence():
    assert route_finding(0.9, auto_post_threshold=0.85, discard_floor=0.3) == "auto_post"
    assert route_finding(0.1, auto_post_threshold=0.85, discard_floor=0.3) == "discard"
    assert route_finding(0.5, auto_post_threshold=0.85, discard_floor=0.3) == "hitl"
    assert route_finding(0.85, auto_post_threshold=0.85, discard_floor=0.3) == "auto_post"
    assert route_finding(0.3, auto_post_threshold=0.85, discard_floor=0.3) == "hitl"


@pytest.mark.asyncio
async def test_verify_finding_overwrites_confidence_from_verify_step():
    finding = Finding(
        path="app.py",
        line=1,
        side="RIGHT",
        category="security",
        severity="low",
        confidence=0.5,
        rationale="placeholder",
        fingerprint="fp1",
    )

    async def fake_complete(role, prompt, *, system=None):
        assert role == "verify"
        return json.dumps({"confidence": 0.42})

    llm = _StubLLM()
    llm.complete = fake_complete
    verified = await verify_finding(finding, _diff(), llm)

    assert verified.confidence == 0.42
    assert verified.path == finding.path


@pytest.mark.asyncio
async def test_graph_auto_posts_high_confidence_finding(monkeypatch):
    monkeypatch.setenv("AUTO_POST_THRESHOLD", "0.85")
    monkeypatch.setenv("DISCARD_FLOOR", "0.3")
    llm = _StubLLM(verify_confidence=0.95)
    graph = build_graph(llm, checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "pr-1"}}

    result = await graph.ainvoke(
        {"diff": _diff(), "findings": [], "to_post": [], "discarded": []}, config=config
    )

    assert len(result["to_post"]) == 1
    assert result["to_post"][0].confidence == 0.95
    assert result["discarded"] == []
    assert "__interrupt__" not in result


@pytest.mark.asyncio
async def test_graph_pauses_on_mid_confidence_and_resumes_without_reanalyzing(monkeypatch):
    monkeypatch.setenv("AUTO_POST_THRESHOLD", "0.85")
    monkeypatch.setenv("DISCARD_FLOOR", "0.3")
    llm = _StubLLM(verify_confidence=0.5)
    graph = build_graph(llm, checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "pr-2"}}

    paused = await graph.ainvoke(
        {"diff": _diff(), "findings": [], "to_post": [], "discarded": []}, config=config
    )
    assert "__interrupt__" in paused
    fingerprint = paused["__interrupt__"][0].value["hitl_findings"][0]["fingerprint"]
    assert llm.verify_calls == 1

    resumed = await graph.ainvoke(Command(resume={fingerprint: True}), config=config)

    assert len(resumed["to_post"]) == 1
    assert resumed["discarded"] == []
    assert llm.verify_calls == 1  # resume did not re-run analyze/verify


@pytest.mark.asyncio
async def test_graph_discards_below_floor_finding_without_pausing(monkeypatch):
    monkeypatch.setenv("AUTO_POST_THRESHOLD", "0.85")
    monkeypatch.setenv("DISCARD_FLOOR", "0.3")
    llm = _StubLLM(verify_confidence=0.1)
    graph = build_graph(llm, checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "pr-4"}}

    result = await graph.ainvoke(
        {"diff": _diff(), "findings": [], "to_post": [], "discarded": []}, config=config
    )

    assert "__interrupt__" not in result
    assert result["to_post"] == []
    assert len(result["discarded"]) == 1


@pytest.mark.asyncio
async def test_graph_discards_rejected_hitl_finding(monkeypatch):
    monkeypatch.setenv("AUTO_POST_THRESHOLD", "0.85")
    monkeypatch.setenv("DISCARD_FLOOR", "0.3")
    llm = _StubLLM(verify_confidence=0.5)
    graph = build_graph(llm, checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "pr-3"}}

    paused = await graph.ainvoke(
        {"diff": _diff(), "findings": [], "to_post": [], "discarded": []}, config=config
    )
    fingerprint = paused["__interrupt__"][0].value["hitl_findings"][0]["fingerprint"]

    resumed = await graph.ainvoke(Command(resume={fingerprint: False}), config=config)

    assert resumed["to_post"] == []
    assert len(resumed["discarded"]) == 1
