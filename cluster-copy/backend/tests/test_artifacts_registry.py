import asyncio

import pytest

from app.models.domain import AgentTraceEvent, ExperimentRun
from app.storage.artifacts import ArtifactStore
from app.storage.registry import RunRegistry


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
