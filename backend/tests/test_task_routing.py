import asyncio
from types import SimpleNamespace

from app.agents.factory import AgentFactory
from app.agents.model_factory import model_metadata
from app.agents.routing import TaskModelRouter
from app.models.domain import AgentSpec, ModelProfile, SpecialistResult
from app.storage.artifacts import ArtifactStore


def make_spec(scope: str, profile: ModelProfile = ModelProfile.BALANCED) -> AgentSpec:
    return AgentSpec(
        role="test_specialist",
        objective="Test bounded task routing.",
        instructions="Perform the assigned bounded task.",
        task_scope=scope,
        model_profile=profile,
        creation_reason="Regression test",
    )


def test_known_task_scopes_receive_host_enforced_profiles(settings):
    router = TaskModelRouter()

    assert router.route_specialist("evidence_selection", ModelProfile.REASONING).effective_profile == ModelProfile.FAST
    assert router.route_specialist("edc_define", ModelProfile.FAST).effective_profile == ModelProfile.REASONING
    assert router.route_specialist("edc_canonicalize", ModelProfile.REASONING).effective_profile == ModelProfile.FAST
    assert router.route_specialist("verification", ModelProfile.BALANCED).effective_profile == ModelProfile.CRITIC
    assert router.route_specialist("other", ModelProfile.BALANCED).effective_profile == ModelProfile.BALANCED


def test_fast_profile_uses_existing_litellm_alias(settings):
    settings.litellm_fast_model = "research-fast"
    settings.litellm_model = "qwen-chat"

    assert model_metadata(settings, ModelProfile.FAST).model == "research-fast"
    assert model_metadata(settings, ModelProfile.REASONING).model == "qwen-chat"


def test_edc_canonicalizer_service_route_uses_fast_profile():
    router = TaskModelRouter()

    assert router.profile_for_service("edc_canonicalize", ModelProfile.REASONING) == ModelProfile.FAST


def test_effective_route_is_persisted_as_a_non_scientific_decision(settings):
    store = ArtifactStore(settings, "run_route_decision")

    class Emitter:
        async def emit(self, **_kwargs):
            return None

    factory = AgentFactory.__new__(AgentFactory)
    factory.deps = SimpleNamespace(settings=settings, artifacts=store)
    factory.emitter = Emitter()

    routed = asyncio.run(factory._apply_model_route(make_spec("verification", ModelProfile.FAST)))

    assert routed.model_profile == ModelProfile.CRITIC
    decisions = store.list("model_routing_decision")
    assert len(decisions) == 1
    payload = store.get(decisions[0]["id"])
    assert payload["requested_profile"] == "fast"
    assert payload["effective_profile"] == "critic"


def test_fanout_dispatches_independent_specialists_concurrently_and_records_metrics(settings):
    store = ArtifactStore(settings, "run_fanout_metrics")

    class Registry:
        def get(self, _run_id):
            return SimpleNamespace(state=SimpleNamespace(delegate_calls=0))

    class Emitter:
        async def emit(self, **_kwargs):
            return None

    factory = AgentFactory.__new__(AgentFactory)
    factory.deps = SimpleNamespace(
        settings=settings,
        registry=Registry(),
        run_id="run_fanout_metrics",
        artifacts=store,
    )
    factory.emitter = Emitter()
    active = 0
    max_active = 0

    async def fake_run(spec):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.03)
        active -= 1
        return SpecialistResult(task_status="completed", task_scope=spec.task_scope, summary="done")

    factory.run = fake_run
    specs = [make_spec("evidence_selection"), make_spec("document_analysis")]
    outputs = asyncio.run(factory.run_fanout(specs))

    assert len(outputs) == 2
    assert max_active == 2
    records = store.list("fanout_execution")
    assert len(records) == 1
    payload = store.get(records[0]["id"])
    assert payload["dispatched_specialist_ids"] == [spec.id for spec in specs]
    assert payload["task_scopes"] == ["evidence_selection", "document_analysis"]
