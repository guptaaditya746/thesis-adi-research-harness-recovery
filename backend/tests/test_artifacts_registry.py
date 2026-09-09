import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest

from app.models.domain import AgentTraceEvent, ExperimentRun
from app.harness.events import TraceLineage, make_pydantic_event_handler
from app.storage.artifacts import ArtifactStore
from app.storage.registry import RunRegistry
from app.telemetry.tracing import Telemetry


def test_artifacts_are_immutable_idempotent_envelopes(settings):
    store = ArtifactStore(settings, "run_test")
    ref = store.put("evidence_span", {"text": "source"}, artifact_id="ev_fixed", creator_agent_id="agent_x")
    same = store.put("evidence_span", {"text": "source"}, artifact_id="ev_fixed", creator_agent_id="agent_x")
    assert ref.id == same.id == "ev_fixed"
    env = store.get_envelope("ev_fixed")
    assert env.run_id == "run_test"
    assert env.creator_agent_id == "agent_x"
    assert env.content_sha256
    with pytest.raises(FileExistsError):
        store.put("evidence_span", {"text": "changed"}, artifact_id="ev_fixed")


def test_registry_persists_full_event_trace_and_recovery(settings):
    registry = RunRegistry(settings)
    run = ExperimentRun(workspace_id="default", document_ids=["d"], question="q", status="running", resumable=True)
    registry.create(run)
    event = AgentTraceEvent(run_id=run.id, type="decision", action="choose", title="Decision", payload={"reason":"gap"})
    asyncio.run(registry.emit(event))
    reloaded = RunRegistry(settings)
    assert reloaded.get(run.id).question == "q"
    assert reloaded.events(run.id)[0].payload["reason"] == "gap"
    assert run.id in {x.id for x in reloaded.recoverable()}


def test_artifact_index_retains_all_concurrent_writes(settings):
    store = ArtifactStore(settings, "run_concurrent_artifacts")

    def write(index: int) -> str:
        return store.put("test_artifact", {"index": index}, artifact_id=f"artifact_{index}").id

    with ThreadPoolExecutor(max_workers=8) as executor:
        artifact_ids = list(executor.map(write, range(32)))

    assert {item["id"] for item in store.list("test_artifact")} == set(artifact_ids)


def test_registry_broadcasts_live_events_to_each_subscriber(settings):
    registry = RunRegistry(settings)
    run = ExperimentRun(workspace_id="default", document_ids=["d"], question="q", status="running")
    registry.create(run)

    async def scenario():
        history_a, queue_a = registry.subscribe(run.id)
        history_b, queue_b = registry.subscribe(run.id)
        assert history_a == history_b == []
        first = AgentTraceEvent(run_id=run.id, type="decision", action="first", title="First")
        await registry.emit(first)
        assert (await queue_a.get()).id == first.id
        assert (await queue_b.get()).id == first.id

        registry.unsubscribe(run.id, queue_a)
        second = AgentTraceEvent(run_id=run.id, type="decision", action="second", title="Second")
        await registry.emit(second)
        assert (await queue_b.get()).id == second.id
        assert queue_a.empty()
        registry.unsubscribe(run.id, queue_b)

    asyncio.run(scenario())


def test_pydantic_tool_events_preserve_tool_call_hierarchy(settings):
    registry = RunRegistry(settings)
    run = ExperimentRun(workspace_id="default", document_ids=["d"], question="q", status="running")
    registry.create(run)
    lineage = TraceLineage()

    class Part:
        tool_name = "spawn_specialist"
        tool_call_id = "call-1"
        args = {"role": "evidence selector"}
        content = {"task_status": "completed"}

    class FunctionToolCallEvent:
        part = Part()

    class FunctionToolResultEvent:
        part = Part()

    class Context:
        agent_name = "research_director"

    async def stream():
        yield FunctionToolCallEvent()
        yield FunctionToolResultEvent()

    async def scenario():
        handler = make_pydantic_event_handler(
            registry, run.id, Telemetry(settings), lineage=lineage,
        )
        await handler(Context(), stream())

    asyncio.run(scenario())
    call, result = registry.events(run.id)
    assert result.parent_event_id == call.id
    assert lineage.parent_for_tool_call("call-1") is None
