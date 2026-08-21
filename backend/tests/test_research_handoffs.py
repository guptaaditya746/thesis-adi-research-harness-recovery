from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.agents.factory import AgentFactory
from app.agents.toolsets import persist_candidate_evidence_from_bundle, read_retrieval_candidate_bundle
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
