"""LangGraph orchestration: fan out to the four lens agents (already
parallel via run_all_lenses), verify every raw finding through one
confidence step (FR-4: routing is a pure function of that score, no
second gate), then route to auto-post / HITL / discard (FR-5). A HITL
batch pauses the run via interrupt() and resumes once approvals arrive,
without re-running analyze/verify (NFR-4's non-blocking half). build_graph
takes any BaseCheckpointSaver so a persistent one (Postgres, TASK-storage)
drops in later -- NFR-4's durability half (a checkpoint surviving the job
returning) is unproven here with MemorySaver and is TASK-storage's job.
Actually posting to GitHub, the FR-8 idempotency check, and logging
discarded findings (FR-5, FR-9) belong to TASK-storage/TASK-hitl, not
here -- this graph's output is the routed finding buckets."""

import asyncio
import json
import os
from typing import Literal, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from pr_review.agents import _strip_markdown_fence, run_all_lenses
from pr_review.contracts import Diff, Finding
from pr_review.llm import LLMProvider

Route = Literal["auto_post", "hitl", "discard"]

_VERIFY_SYSTEM_PROMPT = (
    "You are the confidence-verification step for an automated PR reviewer. "
    "Given one raw finding and the diff it came from, output the single JSON "
    'object {"confidence": number 0-1}: your best estimate that this finding '
    "is correct and worth surfacing. No prose, no markdown fences."
)


def auto_post_threshold() -> float:
    return float(os.environ.get("AUTO_POST_THRESHOLD", "0.5"))


def discard_floor() -> float:
    return float(os.environ.get("DISCARD_FLOOR", "0.3"))


def route_finding(
    confidence: float, *, auto_post_threshold: float, discard_floor: float
) -> Route:
    """Pure function of the verified confidence score -- FR-4's 'no second gate'."""
    if confidence >= auto_post_threshold:
        return "auto_post"
    if confidence < discard_floor:
        return "discard"
    return "hitl"


def _patch_for(diff: Diff, path: str) -> str:
    for file in diff.files:
        if file.path == path:
            return file.patch
    return ""


async def verify_finding(finding: Finding, diff: Diff, llm: LLMProvider) -> Finding:
    prompt = (
        f"Finding: {finding.model_dump_json(exclude={'confidence'})}\n\n"
        f"Diff context for {finding.path}:\n{_patch_for(diff, finding.path)}"
    )
    reply = await llm.complete("verify", prompt, system=_VERIFY_SYSTEM_PROMPT)
    confidence = json.loads(_strip_markdown_fence(reply))["confidence"]
    return finding.model_copy(update={"confidence": confidence})


class ReviewState(TypedDict):
    diff: Diff
    findings: list[Finding]
    to_post: list[Finding]
    discarded: list[Finding]


def build_graph(llm: LLMProvider, checkpointer: BaseCheckpointSaver) -> CompiledStateGraph:
    async def analyze(state: ReviewState) -> dict:
        findings = await run_all_lenses(state["diff"], llm)
        return {"findings": findings}

    async def verify(state: ReviewState) -> dict:
        verified = await asyncio.gather(
            *(verify_finding(finding, state["diff"], llm) for finding in state["findings"])
        )
        return {"findings": list(verified)}

    async def route(state: ReviewState) -> dict:
        threshold = auto_post_threshold()
        floor = discard_floor()
        to_post: list[Finding] = []
        discarded: list[Finding] = []
        hitl: list[Finding] = []
        for finding in state["findings"]:
            decision = route_finding(
                finding.confidence, auto_post_threshold=threshold, discard_floor=floor
            )
            if decision == "auto_post":
                to_post.append(finding)
            elif decision == "discard":
                discarded.append(finding)
            else:
                hitl.append(finding)

        if hitl:
            approvals: dict[str, bool] = interrupt(
                {"hitl_findings": [finding.model_dump(mode="json") for finding in hitl]}
            )
            for finding in hitl:
                if approvals.get(finding.fingerprint):
                    to_post.append(finding)
                else:
                    discarded.append(finding)

        return {"to_post": to_post, "discarded": discarded}

    graph = StateGraph(ReviewState)
    graph.add_node("analyze", analyze)
    graph.add_node("verify", verify)
    graph.add_node("route", route)
    graph.set_entry_point("analyze")
    graph.add_edge("analyze", "verify")
    graph.add_edge("verify", "route")
    graph.add_edge("route", END)
    return graph.compile(checkpointer=checkpointer)
