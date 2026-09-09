from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart

from app.config import Settings
from app.harness.context_management import ResearchContextManager, SemanticContextSummary
from app.models.domain import new_id
from app.storage.artifacts import ArtifactStore


class _Registry:
    def __init__(self, question: str):
        self.run = SimpleNamespace(question=question)
        self.events = []

    def get(self, _run_id):
        return self.run

    async def emit(self, event):
        self.events.append(event)


@pytest.fixture
def context_env(tmp_path):
    settings = Settings(
        data_dir=tmp_path,
        llm_context_window=10_000,
        llm_max_tokens=1_000,
        context_safety_margin_tokens=500,
        context_warn_fraction=0.10,
        context_compact_target_fraction=0.30,
        context_keep_tool_pairs=1,
    )
    artifacts = ArtifactStore(settings, "run_context")
    evidence = artifacts.put("evidence_bundle", {"evidence_ids": ["ev_1"]})
    registry = _Registry("What problem does this paper address?")
    deps = SimpleNamespace(
        artifacts=artifacts,
        registry=registry,
        run_id="run_context",
        target_document_id="doc_target",
    )
    ctx = SimpleNamespace(deps=deps, usage=SimpleNamespace(requests=0))
    return settings, artifacts, evidence, registry, ctx


@pytest.mark.asyncio
async def test_compaction_clears_bulky_tool_payload_and_retains_artifact_references(context_env):
    settings, artifacts, evidence, registry, ctx = context_env

    async def summarize(_transcript, _ctx):
        return SemanticContextSummary(
            findings=["Evidence was persisted."],
            unresolved_questions=["Canonical triples are not yet available."],
            current_intent="Assess remaining scientific state.",
        )

    manager = ResearchContextManager(settings, summary_builder=summarize)
    messages = [
        ModelRequest(parts=[UserPromptPart(content="What problem does this paper address?")]),
        ModelResponse(parts=[ToolCallPart(tool_name="inspect_artifact", args={}, tool_call_id="call_old")]),
        ModelRequest(parts=[ToolReturnPart(tool_name="inspect_artifact", tool_call_id="call_old", content="X" * 20_000)]),
        ModelResponse(parts=[TextPart(content="I will continue scientific research.")]),
        ModelResponse(parts=[TextPart(content="Operational discussion " * 1_500)]),
        ModelResponse(parts=[ToolCallPart(tool_name="inspect_scientific_state", args={}, tool_call_id="call_new")]),
        ModelRequest(parts=[ToolReturnPart(tool_name="inspect_scientific_state", tool_call_id="call_new", content="Y" * 4_000)]),
    ]

    compacted = await manager.compact(messages, ctx)

    rendered = "\n".join(str(message) for message in compacted)
    assert "X" * 1_000 not in rendered
    assert evidence.id in rendered
    assert "What problem does this paper address?" in rendered
    assert any(isinstance(message, ModelRequest) and any("Research working memory" in str(part) for part in message.parts) for message in compacted)
    assert artifacts.find_latest("context_compaction_receipt") is not None
    assert registry.events[-1].action == "context_compacted"


@pytest.mark.asyncio
async def test_context_manager_is_noop_without_operator_declared_context_window(context_env):
    settings, _artifacts, _evidence, _registry, ctx = context_env
    settings.llm_context_window = None
    manager = ResearchContextManager(settings)
    messages = [ModelRequest(parts=[UserPromptPart(content="question")]), ModelResponse(parts=[TextPart(content="Z" * 30_000)])]
    assert await manager.compact(messages, ctx) == messages


@pytest.mark.asyncio
async def test_summary_failure_falls_back_to_host_facts_without_scientific_payload(context_env):
    settings, artifacts, evidence, _registry, ctx = context_env

    async def fail(_transcript, _ctx):
        raise RuntimeError("summary backend unavailable")

    manager = ResearchContextManager(settings, summary_builder=fail)
    messages = [
        ModelRequest(parts=[UserPromptPart(content="question")]),
        ModelResponse(parts=[TextPart(content="A" * 20_000)]),
        ModelResponse(parts=[TextPart(content="B" * 20_000)]),
    ]
    compacted = await manager.compact(messages, ctx)
    receipt = artifacts.get(artifacts.find_latest("context_compaction_receipt")["id"])
    assert receipt["summary_fallback_used"] is True
    rendered = "\n".join(str(message) for message in compacted)
    assert evidence.id in rendered
    assert "summary backend unavailable" not in rendered
