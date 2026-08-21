from types import SimpleNamespace

import pytest

from app.agents.factory import AgentFactory
from app.agents.toolsets import read_retrieval_candidate_bundle
from app.harness.contracts import validate_specialist_spec
from app.models.domain import AgentSpec, EvidenceBundle, EvidenceSpan, RetrievalCandidateBundle
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


def test_candidate_bundle_read_is_bounded_and_typed(settings):
    store = ArtifactStore(settings, "run_handoff_read")
    deps = make_deps(settings, store)
    ref = persist_retrieval_candidate_bundle(deps, "q", [make_span("x" * 1000)])

    result = read_retrieval_candidate_bundle(deps, ref.id, preview_chars=80)

    assert result["artifact_id"] == ref.id
    assert result["candidate_count"] == 1
    assert len(result["candidates"][0]["text"]) <= 80
    assert result["candidates"][0]["id"]


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


def test_edc_requires_evidence_bundle(settings):
    store = ArtifactStore(settings, "run_edc_guard")
    deps = make_deps(settings, store)
    candidate_ref = persist_retrieval_candidate_bundle(deps, "q", [make_span()])

    with pytest.raises(ValueError, match="evidence_bundle"):
        validate_specialist_spec(deps, make_spec(required_capabilities=["edc"], input_artifact_ids=[candidate_ref.id]))


def test_verification_requires_candidate_triples_and_evidence(settings):
    store = ArtifactStore(settings, "run_verification_guard")
    deps = make_deps(settings, store)
    evidence_ref = store.put("evidence_bundle", EvidenceBundle(spans=[make_span()]))

    with pytest.raises(ValueError, match="candidate_canonical_triples"):
        validate_specialist_spec(deps, make_spec(required_capabilities=["verification"], input_artifact_ids=[evidence_ref.id]))

    candidate_ref = store.put("candidate_canonical_triples", [])
    validate_specialist_spec(
        deps,
        make_spec(required_capabilities=["verification"], input_artifact_ids=[evidence_ref.id, candidate_ref.id]),
    )
