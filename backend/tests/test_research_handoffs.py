import asyncio
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from pydantic_ai.exceptions import UnexpectedModelBehavior

from app.agents.factory import AgentFactory
from app.agents.toolsets import _ontology_tools, persist_candidate_evidence_from_bundle, read_retrieval_candidate_bundle
from app.harness.contracts import CorrectableToolInputError, validate_specialist_spec
from app.harness.context_management import ResearchContextManager
from app.harness.runtime import DIRECTOR_INSTRUCTIONS
from app.models.domain import AgentSpec, EvidenceBundle, EvidenceSpan, OntologyProposalInput, RetrievalCandidateBundle, SpecialistResult, SubOntologyArtifact
from app.services.ontology import OntologyService
from app.storage.artifacts import ArtifactStore
from app.tools.research import persist_retrieval_candidate_bundle


def make_span(text: str = "The paper evaluates the proposed method.") -> EvidenceSpan:
    return EvidenceSpan(
        document_id="doc_1",
        page=2,
        block_id="block_2",
        text=text,
        retrieval_method="sqlite_fts5_bm25",
        retrieval_query="research question",
    )


def make_deps(settings, store):
    return SimpleNamespace(settings=settings, artifacts=store)


def make_spec(**updates) -> AgentSpec:
    values = {
        "role": "specialist",
        "objective": "Inspect scientific evidence.",
        "instructions": "Select relevant source evidence.",
        "creation_reason": "Regression test",
    }
    values.update(updates)
    return AgentSpec(**values)


def test_retrieval_candidates_are_persisted_as_a_bundle_not_evidence(settings):
    store = ArtifactStore(settings, "run_handoff")
    deps = make_deps(settings, store)

    ref = persist_retrieval_candidate_bundle(deps, "research question", [make_span()])

    bundle = RetrievalCandidateBundle.model_validate(store.get(ref.id))
    assert ref.kind == "retrieval_candidate_bundle"
    assert bundle.spans[0].id not in {item["id"] for item in store.list()}
    assert store.list("evidence_span") == []


def test_retrieval_candidate_bundle_persistence_returns_an_artifact_ref(settings):
    store = ArtifactStore(settings, "run_candidate_ref")
    deps = make_deps(settings, store)

    ref = persist_retrieval_candidate_bundle(deps, "research question", [make_span()])

    assert ref is not None
    assert ref.id
    assert ref.kind == "retrieval_candidate_bundle"


def test_candidate_bundle_read_is_bounded_and_typed(settings):
    store = ArtifactStore(settings, "run_handoff_read")
    deps = make_deps(settings, store)
    ref = persist_retrieval_candidate_bundle(deps, "q", [make_span("x" * 1000)])

    result = read_retrieval_candidate_bundle(deps, ref.id, preview_chars=80)

    assert result["artifact_id"] == ref.id
    assert result["candidate_count"] == 1
    candidate = result["candidates"][0]
    assert len(candidate["preview"]) <= 80
    assert candidate["preview_truncated"] is True
    assert candidate["candidate_id"]
    assert "text" not in candidate
    assert "source_text_sha256" not in candidate
    assert "start_char" not in candidate
    assert "end_char" not in candidate


def test_truncated_preview_is_never_an_evidence_span(settings):
    store = ArtifactStore(settings, "run_preview_contract")
    deps = make_deps(settings, store)
    ref = persist_retrieval_candidate_bundle(deps, "q", [make_span("verbatim source " * 100)])

    preview = read_retrieval_candidate_bundle(deps, ref.id, preview_chars=40)["candidates"][0]

    assert "…" in preview["preview"]
    assert preview["preview_truncated"] is True
    assert preview["candidate_id"]
    with pytest.raises(ValidationError):
        EvidenceSpan.model_validate(preview)


def test_selecting_candidate_by_id_persists_exact_full_span(settings):
    store = ArtifactStore(settings, "run_candidate_persist")
    span = make_span("verbatim source " * 100)

    class Retrieval:
        def validate_span(self, _doc, candidate):
            return candidate.text == span.text, "unexpected source mismatch"

    class Deps:
        def __init__(self):
            self.document_ids = ["doc_1"]
            self.artifacts = store
            self.retrieval = Retrieval()
            self.settings = settings

        def ensure_not_cancelled(self):
            return None

        class Pdf:
            @staticmethod
            def load(_doc_id):
                return object()
        pdf = Pdf()
    deps = Deps()

    bundle_ref = persist_retrieval_candidate_bundle(deps, "q", [span])
    result = persist_candidate_evidence_from_bundle(deps, "evidence_selector", bundle_ref.id, [span.id], "sufficient")
    evidence = EvidenceBundle.model_validate(store.get(result["artifact_ref"]["id"]))

    assert evidence.spans[0].id == span.id
    assert evidence.spans[0].text == span.text
    assert "…" not in evidence.spans[0].text


def test_candidate_evidence_identity_ignores_sufficiency_summary(settings):
    store = ArtifactStore(settings, "run_candidate_idempotent")
    span = make_span()

    class Retrieval:
        @staticmethod
        def validate_span(_doc, candidate):
            return candidate.text == span.text, "unexpected source mismatch"

    deps = SimpleNamespace(
        document_ids=["doc_1"], artifacts=store, retrieval=Retrieval(), settings=settings,
        pdf=SimpleNamespace(load=lambda _doc_id: object()),
    )
    bundle_ref = persist_retrieval_candidate_bundle(deps, "q", [span])

    first = persist_candidate_evidence_from_bundle(deps, "selector", bundle_ref.id, [span.id], "first wording")
    second = persist_candidate_evidence_from_bundle(deps, "selector", bundle_ref.id, [span.id, span.id], "different wording")

    assert first["artifact_ref"]["id"] == second["artifact_ref"]["id"]
    assert EvidenceBundle.model_validate(store.get(first["artifact_ref"]["id"])).sufficiency_summary == ""


def test_specialist_prompt_contains_reference_not_scientific_payload(settings):
    store = ArtifactStore(settings, "run_prompt")
    deps = make_deps(settings, store)
    ref = persist_retrieval_candidate_bundle(deps, "q", [make_span("PRIVATE SOURCE PAYLOAD")])
    factory = AgentFactory.__new__(AgentFactory)
    factory.deps = deps

    prompt = factory._task_prompt(make_spec(input_artifact_ids=[ref.id]))

    assert ref.id in prompt
    assert "retrieval_candidate_bundle" in prompt
    assert "PRIVATE SOURCE PAYLOAD" not in prompt


def test_edc_specialist_is_blocked_until_evidence_and_ontology_are_persisted(settings):
    store = ArtifactStore(settings, "run_edc_guard")
    deps = make_deps(settings, store)
    evidence_ref = store.put("evidence_bundle", EvidenceBundle(spans=[make_span()]))

    from app.harness.contracts import ScientificStateBlocked

    with pytest.raises(ScientificStateBlocked, match="ontology_snapshot"):
        validate_specialist_spec(deps, make_spec(required_capabilities=["edc"], input_artifact_ids=[evidence_ref.id]))

    ontology_ref = store.put("ontology_snapshot", SubOntologyArtifact(turtle_path="snapshot.ttl", selected_elements=[]))
    validate_specialist_spec(
        deps, make_spec(required_capabilities=["edc"], input_artifact_ids=[evidence_ref.id, ontology_ref.id])
    )


def test_wrong_artifact_reference_error_is_bounded_and_payload_free(settings):
    store = ArtifactStore(settings, "run_bounded_error")
    deps = make_deps(settings, store)
    candidate_ref = persist_retrieval_candidate_bundle(deps, "q", [make_span("secret source " * 5000)])

    from app.harness.contracts import InvalidArtifactKind, require_kind

    with pytest.raises(InvalidArtifactKind) as exc_info:
        require_kind(deps, candidate_ref.id, "ontology_snapshot")
    message = str(exc_info.value)
    assert len(message) <= 512
    assert "secret source" not in message
    assert "expected ontology_snapshot, got retrieval_candidate_bundle" in message


def test_verification_requires_candidate_triples_and_evidence(settings):
    store = ArtifactStore(settings, "run_verification_guard")
    deps = make_deps(settings, store)
    evidence_ref = store.put("evidence_bundle", EvidenceBundle(spans=[make_span()]))

    from app.harness.contracts import ScientificStateBlocked

    with pytest.raises(ScientificStateBlocked, match="candidate_canonical_triples"):
        validate_specialist_spec(deps, make_spec(required_capabilities=["verification"], input_artifact_ids=[evidence_ref.id]))

    ontology_ref = store.put(
        "ontology_snapshot", SubOntologyArtifact(turtle_path="snapshot.ttl", selected_elements=[])
    )
    open_ref = store.put(
        "edc_open_triples",
        {"edc_execution_id": "edc_guard", "evidence_bundle_artifact_id": evidence_ref.id, "triples": []},
        parent_artifact_ids=[evidence_ref.id],
    )
    definitions_ref = store.put(
        "edc_relation_definitions",
        {"edc_execution_id": "edc_guard", "open_triples_artifact_id": open_ref.id, "definitions": []},
        parent_artifact_ids=[open_ref.id],
    )
    candidate_ref = store.put("candidate_canonical_triples", [], parent_artifact_ids=[
        open_ref.id, definitions_ref.id, evidence_ref.id, ontology_ref.id,
    ])
    validate_specialist_spec(
        deps,
        make_spec(required_capabilities=["verification"], input_artifact_ids=[evidence_ref.id, candidate_ref.id]),
    )


def test_missing_verification_state_returns_blocked_specialist_result(settings):
    store = ArtifactStore(settings, "run_blocked_specialist")
    deps = SimpleNamespace(
        settings=settings,
        artifacts=store,
        ensure_not_cancelled=lambda: None,
    )
    factory = AgentFactory.__new__(AgentFactory)
    factory.deps = deps
    expected = SpecialistResult(
        task_status="blocked", task_scope="verification", summary="blocked",
        blocked_requirements=["candidate_canonical_triples"],
    )
    factory._blocked_result = lambda spec, exc, **kwargs: expected

    result = asyncio.run(factory.run(make_spec(task_scope="verification", required_capabilities=["verification"])))

    assert result.task_status == "blocked"
    assert result.blocked_requirements == ["candidate_canonical_triples"]


def test_generation_token_limit_failure_is_not_profile_escalated(settings):
    store = ArtifactStore(settings, "run_generation_limit")

    class Registry:
        def __init__(self):
            self.run = SimpleNamespace(state=SimpleNamespace(delegate_calls=0, active_agent_ids=[]))

        def get(self, _run_id):
            return self.run

        def persist(self, _run):
            return None

    class Telemetry:
        @staticmethod
        def span(*_args, **_kwargs):
            return nullcontext(SimpleNamespace(set_attribute=lambda *_args: None))

    class Emitter:
        async def emit(self, **_kwargs):
            return None

    class FailingAgent:
        calls = 0

        async def run(self, *_args, **_kwargs):
            self.calls += 1
            raise UnexpectedModelBehavior(
                "Model token limit (12288) exceeded before any response was generated. "
                "Increase the `max_tokens` model setting, or simplify the prompt to result in a shorter response that will fit within the limit."
            )

    registry = Registry()
    deps = SimpleNamespace(
        settings=settings, artifacts=store, registry=registry, run_id="run_generation_limit", telemetry=Telemetry(),
        ensure_not_cancelled=lambda: None,
    )
    factory = AgentFactory.__new__(AgentFactory)
    factory.deps = deps
    factory.emitter = Emitter()
    failing_agent = FailingAgent()
    factory.build = lambda _spec: failing_agent

    result = asyncio.run(factory.run(make_spec()))

    assert result.task_status == "partial"
    assert failing_agent.calls == 1
    recoveries = [item for item in store.list("recovery")]
    assert recoveries
    assert store.get(recoveries[-1]["id"])["action"] == "generation_token_limit"


def test_ontology_batch_tools_persist_independent_resources_and_decisions(settings):
    store = ArtifactStore(settings, "run_ontology_batch")
    ontology = OntologyService(settings)
    deps = SimpleNamespace(
        settings=settings,
        artifacts=store,
        ontology=ontology,
        run_id="run_ontology_batch",
        registry=SimpleNamespace(get=lambda _run_id: SimpleNamespace(question="What problem does this paper address?")),
    )
    toolset = _ontology_tools(deps, "ontology_agent")
    inputs = [
        OntologyProposalInput(
            proposal_type=proposal_type,
            label=label, definition=definition, rationale="Required for the research question.",
        )
        for label, proposal_type, definition in [
            ("ResearchProblem", "class", "A research problem addressed by a publication."),
            ("addressesProblem", "object_property", "A publication addresses a research problem."),
            ("focusesOn", "object_property", "A publication focuses on a topic."),
        ]
    ]

    created = toolset.tools["create_ontology_proposals"].function(inputs)
    proposal_ids = [ref["id"] for ref in created["proposal_refs"]]
    accepted = toolset.tools["accept_ontology_proposals_for_run"].function(
        proposal_ids, "No DBLP predicate captures these domain relations.", []
    )

    assert created["proposal_count"] == 3
    assert len(proposal_ids) == len(set(proposal_ids)) == 3
    for proposal_id in proposal_ids:
        assert store.get(proposal_id)["uri"].startswith("https://example.org/thesis/run/run_ontology_batch/ontology#")
    assert accepted["accepted_count"] == 3
    assert len(accepted["decision_refs"]) == 3
    assert len(store.list("ontology_acceptance_decision")) == 3


def test_ontology_snapshot_specialist_budget_is_feasible(settings):
    store = ArtifactStore(settings, "run_ontology_budget")
    deps = make_deps(settings, store)

    with pytest.raises(CorrectableToolInputError, match="max_tool_calls >= 4"):
        validate_specialist_spec(
            deps,
            make_spec(task_scope="ontology_snapshot", required_capabilities=["ontology"], max_tool_calls=3),
        )

    validate_specialist_spec(
        deps,
        make_spec(task_scope="ontology_snapshot", required_capabilities=["ontology"], max_tool_calls=4),
    )


def test_director_ontology_guidance_never_requests_one_proposal_for_many_resources():
    guidance = DIRECTOR_INSTRUCTIONS.lower()
    assert "exactly one ontologyproposal" not in guidance
    assert "each ontologyproposal represents exactly one rdf resource" in guidance
    assert "create separately typed proposal entries" in guidance


def test_goal_compass_is_factual_and_does_not_prescribe_an_action(settings):
    store = ArtifactStore(settings, "run_goal_compass")
    evidence = store.put("evidence_bundle", EvidenceBundle(spans=[make_span()]))
    deps = SimpleNamespace(
        artifacts=store,
        target_document_id="doc_1",
        registry=SimpleNamespace(get=lambda _run_id: SimpleNamespace(question="What problem does this paper address?")),
        run_id="run_goal_compass",
    )

    compass = ResearchContextManager(settings).goal_compass(deps)

    assert evidence.id in compass
    assert "accepted canonical triples" in compass
    assert "Decide the next research action autonomously" in compass
    assert "run EDC next" not in compass


def test_candidate_lineage_resolves_exact_scientific_inputs(settings):
    store = ArtifactStore(settings, "run_lineage")
    evidence_ref = store.put("evidence_bundle", EvidenceBundle(spans=[make_span()]))
    ontology_ref = store.put(
        "ontology_snapshot", SubOntologyArtifact(turtle_path="snapshot.ttl", selected_elements=[])
    )
    open_ref = store.put("edc_open_triples", {"edc_execution_id": "edc_test", "evidence_bundle_artifact_id": evidence_ref.id, "triples": []}, parent_artifact_ids=[evidence_ref.id])
    defs_ref = store.put("edc_relation_definitions", {"edc_execution_id": "edc_test", "open_triples_artifact_id": open_ref.id, "definitions": []}, parent_artifact_ids=[open_ref.id])
    candidate_ref = store.put("candidate_canonical_triples", [], parent_artifact_ids=[open_ref.id, defs_ref.id, evidence_ref.id, ontology_ref.id])

    from app.harness.contracts import resolve_candidate_lineage

    lineage = resolve_candidate_lineage(make_deps(settings, store), candidate_ref.id)
    assert lineage.evidence_bundle_id == evidence_ref.id
    assert lineage.ontology_snapshot_id == ontology_ref.id
    assert lineage.open_triples_id == open_ref.id
    assert lineage.relation_definitions_id == defs_ref.id


def test_model_invented_artifact_id_is_rejected_without_profile_retry(settings):
    store = ArtifactStore(settings, "run_invented_artifact_id")

    class Registry:
        def __init__(self):
            self.run = SimpleNamespace(state=SimpleNamespace(delegate_calls=0, active_agent_ids=[]))
        def get(self, _run_id):
            return self.run
        def persist(self, _run):
            pass

    class Emitter:
        async def emit(self, **_kwargs):
            return SimpleNamespace(id="event")

    class InventingAgent:
        calls = 0
        async def run(self, *_args, **_kwargs):
            self.calls += 1
            return SimpleNamespace(output=SpecialistResult(
                task_scope="ontology_snapshot",
                summary="Created the ontology snapshot.",
                artifact_ids=["default_ontology_snapshot"],
            ))

    deps = SimpleNamespace(
        settings=settings,
        artifacts=store,
        registry=Registry(),
        run_id="run_invented_artifact_id",
        telemetry=SimpleNamespace(span=lambda *_args, **_kwargs: nullcontext(
            SimpleNamespace(set_attribute=lambda *_args: None)
        )),
        trace_lineage=SimpleNamespace(),
        ensure_not_cancelled=lambda: None,
    )
    factory = AgentFactory.__new__(AgentFactory)
    factory.deps = deps
    factory.emitter = Emitter()
    inventing_agent = InventingAgent()
    factory.build = lambda _spec: inventing_agent

    result = asyncio.run(factory.run(make_spec(task_scope="ontology_snapshot")))

    assert result.task_status == "partial"
    assert "default_ontology_snapshot" not in result.artifact_ids
    assert inventing_agent.calls == 1
    recovery = store.get(store.find_latest("recovery")["id"])
    assert recovery["action"] == "invalid_artifact_reference"

    repeated = asyncio.run(factory.run(make_spec(task_scope="ontology_snapshot")))

    assert repeated.task_status == "blocked"
    assert "already failed" in repeated.blocked_reason
    assert inventing_agent.calls == 1


def test_backend_collects_persisted_specialist_artifacts_and_updates_run_state(settings):
    store = ArtifactStore(settings, "run_backend_artifact_ids")
    spec = make_spec(task_scope="ontology_snapshot")
    snapshot_ref = store.put(
        "ontology_snapshot",
        SubOntologyArtifact(turtle_path="snapshot.ttl", selected_elements=[]),
        creator_agent_id=spec.id,
    )
    state = SimpleNamespace(
        artifact_refs=[], ontology_snapshot_id=None, verification_artifact_ids=[]
    )
    registry = SimpleNamespace(
        get=lambda _run_id: SimpleNamespace(state=state), persist=lambda _run: None
    )
    factory = AgentFactory.__new__(AgentFactory)
    factory.deps = SimpleNamespace(artifacts=store, registry=registry, run_id="run_backend_artifact_ids")

    output = factory._validated_output(spec, SpecialistResult(
        task_scope="ontology_snapshot", summary="Snapshot persisted."
    ))
    factory._record_validated_output_state(output.artifact_ids)

    assert output.artifact_ids == [snapshot_ref.id]
    assert state.ontology_snapshot_id == snapshot_ref.id
    assert [ref.id for ref in state.artifact_refs] == [snapshot_ref.id]
