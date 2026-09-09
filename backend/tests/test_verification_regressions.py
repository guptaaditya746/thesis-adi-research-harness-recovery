import asyncio
from types import SimpleNamespace

from app.agents import verifier as compatibility_verifier
from app.agents.toolsets import _edc_tools, _verification_tools
from app.models.domain import (
    CanonicalTriple,
    CanonicalizationDecision,
    CandidateTriple,
    EDCCanonicalizationResult,
    EDCOpenTriples,
    EDCRelationDefinitions,
    EvidenceBundle,
    EvidenceSpan,
    RelationDefinition,
    SubOntologyArtifact,
    TripleVerification,
    ValidationReport,
    VerificationDimensions,
    VerificationStatus,
)
from app.services.verification import TripleVerificationService, VerificationJudgement, VerificationOutput
from app.storage.artifacts import ArtifactStore


def span(span_id: str) -> EvidenceSpan:
    return EvidenceSpan(
        id=span_id, document_id="doc_1", page=1, block_id="block_1", text="The paper addresses a problem.",
        retrieval_method="sqlite_fts5_bm25", retrieval_query="problem",
    )


def triple(triple_id: str, evidence_ids: list[str]) -> CanonicalTriple:
    return CanonicalTriple(
        id=triple_id, subject="Paper", predicate="addresses", object="Problem",
        predicate_uri="https://example.org/ontology#addresses", evidence_ids=evidence_ids,
        canonicalization="llm_verified",
    )


def accepted_check(triple_id: str, evidence_ids: list[str]) -> TripleVerification:
    return TripleVerification(
        triple_id=triple_id, status=VerificationStatus.ACCEPTED, evidence_ids=evidence_ids, verifier_agent_id="test",
        explanation="supported",
        dimensions=VerificationDimensions(
            evidence_support="yes", ontology_compatibility="yes", direction_correct="yes",
            rdf_structural_validity="yes", shacl_conformance="yes", negation_modality_ok="yes",
            entity_normalization_quality="good",
        ),
    )


def test_verification_service_preserves_input_order_with_interleaved_prefailures(settings):
    service = TripleVerificationService.__new__(TripleVerificationService)

    class Agent:
        async def run(self, _prompt):
            return SimpleNamespace(output=VerificationOutput(judgements=[
                VerificationJudgement(
                    triple_id="a", evidence_support="yes", ontology_compatibility="yes", direction_correct="yes",
                    negation_modality_ok="yes", entity_normalization_quality="good", explanation="a",
                ),
                VerificationJudgement(
                    triple_id="c", evidence_support="yes", ontology_compatibility="yes", direction_correct="yes",
                    negation_modality_ok="yes", entity_normalization_quality="good", explanation="c",
                ),
            ]))

    service.agent = Agent()
    values = [triple("a", ["ev_a"]), triple("b", ["missing"]), triple("c", ["ev_c"])]
    checks = asyncio.run(service.verify(values, EvidenceBundle(spans=[span("ev_a"), span("ev_c")])))

    assert [check.triple_id for check in checks] == ["a", "b", "c"]
    assert checks[1].status == VerificationStatus.REJECTED


def test_compatibility_verifier_pairs_by_triple_id_not_position(monkeypatch, settings):
    values = [triple("a", ["ev_a"]), triple("b", ["missing"]), triple("c", ["ev_c"])]
    checks = [accepted_check("b", ["missing"]), accepted_check("a", ["ev_a"]), accepted_check("c", ["ev_c"])]

    class Service:
        def __init__(self, _settings):
            pass

        async def verify(self, _triples, _bundle):
            return checks

    monkeypatch.setattr(compatibility_verifier, "TripleVerificationService", Service)
    output = asyncio.run(compatibility_verifier.verify_triples_with_agent(
        settings, values, EvidenceBundle(spans=[span("ev_a"), span("ev_c")])
    ))

    assert [item.verification.triple_id for item in output if item.verification] == ["a", "b", "c"]


def test_verification_tool_persists_candidate_lineage_without_name_error(settings):
    store = ArtifactStore(settings, "run_verification_lineage")
    evidence_ref = store.put("evidence_bundle", EvidenceBundle(spans=[span("ev_1")]))
    open_ref = store.put(
        "edc_open_triples", EDCOpenTriples(edc_execution_id="edc_1", evidence_bundle_artifact_id=evidence_ref.id),
        parent_artifact_ids=[evidence_ref.id],
    )
    definitions_ref = store.put(
        "edc_relation_definitions", EDCRelationDefinitions(edc_execution_id="edc_1", open_triples_artifact_id=open_ref.id),
        parent_artifact_ids=[open_ref.id],
    )
    snapshot_ref = store.put("ontology_snapshot", SubOntologyArtifact(turtle_path="unused.ttl", selected_elements=[]))
    candidate = triple("candidate_1", ["ev_1"])
    candidate_ref = store.put(
        "candidate_canonical_triples", [candidate],
        parent_artifact_ids=[open_ref.id, definitions_ref.id, evidence_ref.id, snapshot_ref.id],
    )

    deps = SimpleNamespace(
        settings=settings, artifacts=store,
        verification=SimpleNamespace(verify=lambda triples, *_args, **_kwargs: _async_value([accepted_check(triples[0].id, triples[0].evidence_ids)])),
        validation=SimpleNamespace(validate=lambda *_args: ValidationReport(rdf_valid=True, shacl_conforms=True, shacl_report="ok")),
    )
    tool = _verification_tools(deps, "verification_agent").tools["verify_candidate_canonical_triples"].function
    result = asyncio.run(tool(candidate_ref.id))

    assert result["accepted_count"] == 1
    for ref_key in ("verification_ref", "validation_ref", "canonical_triples_ref", "rejected_triples_ref"):
        assert candidate_ref.id in store.get_envelope(result[ref_key]["id"]).parent_artifact_ids


def test_edc_persists_canonicalized_subset_with_unresolved_diagnostics(settings):
    store = ArtifactStore(settings, "run_partial_edc")
    evidence_ref = store.put("evidence_bundle", EvidenceBundle(spans=[span("ev_1")]))
    open = EDCOpenTriples(
        edc_execution_id="edc_2", evidence_bundle_artifact_id=evidence_ref.id,
        triples=[CandidateTriple(id="open_1", subject="Paper", predicate="addresses", object="Problem", evidence_ids=["ev_1"])],
    )
    open_ref = store.put("edc_open_triples", open, parent_artifact_ids=[evidence_ref.id])
    definitions = EDCRelationDefinitions(
        edc_execution_id="edc_2", open_triples_artifact_id=open_ref.id,
        definitions=[RelationDefinition(relation="addresses", definition="addresses", open_triple_ids=["open_1"])],
    )
    definitions_ref = store.put("edc_relation_definitions", definitions, parent_artifact_ids=[open_ref.id])
    snapshot_ref = store.put("ontology_snapshot", SubOntologyArtifact(turtle_path="unused.ttl", selected_elements=[]))
    candidate = triple("candidate_2", ["ev_1"])
    decision = CanonicalizationDecision(
        open_triple_id="open_1", open_predicate="addresses", relation_definition="addresses",
        decision_reason="matched", selected_uri=candidate.predicate_uri, selected_label="addresses",
        outcome="successful_canonicalization", evidence_ids=["ev_1"],
    )

    async def canonicalize(*_args, **_kwargs):
        return [decision], [candidate], [], ["open_unresolved"]

    deps = SimpleNamespace(
        settings=settings, artifacts=store, run_id="run_partial_edc",
        edc=SimpleNamespace(canonicalize=canonicalize),
        ontology=SimpleNamespace(relation_catalog=lambda _snapshot: []),
        registry=SimpleNamespace(get=lambda _run_id: SimpleNamespace(question="What problem?")),
    )
    tool = _edc_tools(deps, "edc_agent").tools["retry_edc_canonicalization"].function
    result = asyncio.run(tool(open_ref.id, definitions_ref.id, snapshot_ref.id))

    assert result["status"] == "partial"
    candidate_ref = result["candidate_ref"]
    assert candidate_ref is not None
    canonicalization_ref = result["result_ref"]
    assert canonicalization_ref["id"] in store.get_envelope(candidate_ref["id"]).parent_artifact_ids


async def _async_value(value):
    return value
