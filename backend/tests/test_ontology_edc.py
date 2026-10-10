import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from pydantic_ai.exceptions import UnexpectedModelBehavior

from app.agents.toolsets import _edc_tools
from app.models.domain import (
    CandidateTriple,
    CanonicalTriple,
    CanonicalRelationCandidate,
    CanonicalizationDecision,
    EDCOpenTriples,
    EDCRelationDefinitions,
    EvidenceBundle,
    EvidenceSpan,
    OntologyElement,
    OntologyProposal,
    OntologyProposalStatus,
    RelationDefinition,
)
from app.services.edc import CanonicalizationChoice, EDCService, ExtractionOutput
from app.services.ontology import DBLP, OntologyService
from app.storage.artifacts import ArtifactStore


def proposal(service: OntologyService, run_id: str, label: str, proposal_type: str, **updates) -> OntologyProposal:
    values = {
        "proposed_by": "agent",
        "question": "What problem does this paper address?",
        "type": proposal_type,
        "uri": service.make_proposal_uri(run_id, label),
        "label": label,
        "definition": f"The subject {label} the object.",
        "rationale": "Test fixture.",
        "status": OntologyProposalStatus.ACCEPTED_FOR_RUN,
    }
    values.update(updates)
    return OntologyProposal(**values)


def test_run_overlay_cannot_mutate_dblp_namespace(settings, tmp_path: Path):
    service = OntologyService(settings)
    item = OntologyProposal(
        proposed_by="agent", question="q", type="object_property", uri=DBLP + "inventedRelation",
        label="invented relation", definition="x relates to y", rationale="test",
        status=OntologyProposalStatus.ACCEPTED_FOR_RUN,
    )
    with pytest.raises(ValueError, match="DBLP namespace"):
        service.build_snapshot([], [item], tmp_path / "active.ttl", run_id="r1")


def test_valid_extension_is_run_local(settings, tmp_path: Path):
    service = OntologyService(settings)
    uri = service.make_proposal_uri("r1", "uses method")
    item = OntologyProposal(
        proposed_by="agent", question="q", type="object_property", uri=uri, label="uses method",
        definition="The subject uses the method in the object.", rationale="paper evidence",
        status=OntologyProposalStatus.ACCEPTED_FOR_RUN,
    )
    snapshot = service.build_snapshot([], [item], tmp_path / "active.ttl", run_id="r1")
    assert not snapshot.validation_warnings
    assert any(x.uri == uri for x in service.relation_catalog(snapshot))
    assert snapshot.seed_ontology_sha256 == service.seed_sha256
    assert snapshot.snapshot_ttl_sha256 == snapshot.content_sha256
    assert snapshot.snapshot_builder_version


def test_class_proposal_cannot_declare_property_fields(settings):
    service = OntologyService(settings)
    with pytest.raises(ValidationError, match="Class proposals cannot declare domain or range"):
        proposal(service, "r1", "ResearchProblem", "class", domain="https://example.org/Paper")


def test_prose_mentions_have_no_hidden_ontology_side_effects(settings, tmp_path: Path):
    service = OntologyService(settings)
    item = proposal(
        service,
        "r1",
        "ResearchProblem",
        "class",
        definition="A problem concept. Related prose may mention addressesProblem, focusesOn, and arguesFor.",
    )
    snapshot = service.build_snapshot([], [item], tmp_path / "active.ttl", run_id="r1")

    assert {x.uri for x in service.class_catalog(snapshot)} >= {item.uri}
    labels = {x.label for x in service.relation_catalog(snapshot)}
    assert "addressesProblem" not in labels
    assert "focusesOn" not in labels
    assert "arguesFor" not in labels


def test_independent_resources_require_independent_typed_proposals(settings, tmp_path: Path):
    service = OntologyService(settings)
    props = [
        proposal(service, "r1", "addressesProblem", "object_property"),
        proposal(service, "r1", "focusesOn", "object_property"),
        proposal(service, "r1", "arguesFor", "object_property"),
        proposal(service, "r1", "hasPaperType", "object_property"),
        proposal(service, "r1", "Problem", "class"),
    ]
    snapshot = service.build_snapshot([], props, tmp_path / "active.ttl", run_id="r1")
    relation_uris = {x.uri for x in service.relation_catalog(snapshot)}
    class_uris = {x.uri for x in service.class_catalog(snapshot)}

    assert {p.uri for p in props[:-1]} <= relation_uris
    assert props[-1].uri in class_uris


def test_accepted_property_and_class_proposals_survive_catalog_reconstruction(settings, tmp_path: Path):
    service = OntologyService(settings)
    relation = proposal(service, "r1", "addressesProblem", "object_property")
    data = proposal(service, "r1", "hasConfidence", "datatype_property", range="http://www.w3.org/2001/XMLSchema#decimal")
    concept = proposal(service, "r1", "ResearchProblem", "class")
    snapshot = service.build_snapshot([], [relation, data, concept], tmp_path / "active.ttl", run_id="r1")

    assert {relation.uri, data.uri} <= {x.uri for x in service.relation_catalog(snapshot)}
    assert concept.uri in {x.uri for x in service.class_catalog(snapshot)}


def test_edc_candidate_ranking_is_deterministic_non_embedding(settings):
    service = EDCService(settings)
    triple = CandidateTriple(subject="Paper", predicate="authored by", object="Ada", evidence_ids=["ev1"])
    relations = [
        OntologyElement(uri=DBLP+"authoredBy", label="authoredBy", kind="property", description="The publication is authored by the creator."),
        OntologyElement(uri=DBLP+"doi", label="doi", kind="property", description="A Digital Object Identifier."),
    ]
    ranked = service.rank_candidates(triple, "The subject publication was authored by the object creator.", relations)
    assert ranked[0].uri == DBLP + "authoredBy"
    assert any("label_similarity" in reason for reason in ranked[0].reasons)


def test_extract_assigns_a_unique_open_triple_id_for_each_proposition():
    class FakeExtractor:
        async def run(self, _prompt):
            # Simulates the recurrent model error: copying an EvidenceSpan ID as
            # every proposition's id.
            return SimpleNamespace(output=ExtractionOutput(triples=[
                CandidateTriple(id="ev_same", subject="Paper", predicate="reports", object="A", evidence_ids=["ev_same"]),
                CandidateTriple(id="ev_same", subject="Paper", predicate="reports", object="B", evidence_ids=["ev_same"]),
            ]))

    edc = EDCService.__new__(EDCService)
    edc.extract_agent = FakeExtractor()
    bundle = EvidenceBundle(spans=[EvidenceSpan(id="ev_same", document_id="doc", page=1, text="source")])

    triples = asyncio.run(edc.extract(bundle, question="What does the paper report?"))

    assert len({triple.id for triple in triples}) == 2
    assert {tuple(triple.evidence_ids) for triple in triples} == {("ev_same",)}


def test_open_triple_artifact_rejects_duplicate_proposition_ids():
    triple = CandidateTriple(id="open_duplicate", subject="Paper", predicate="reports", object="A", evidence_ids=["ev1"])

    with pytest.raises(ValidationError, match="must be unique"):
        EDCOpenTriples(
            edc_execution_id="edc_duplicate",
            evidence_bundle_artifact_id="evidence_bundle_1",
            triples=[triple, triple.model_copy(update={"object": "B"})],
        )


def test_canonicalization_rejects_asserted_domain_or_range_mismatch():
    service = EDCService.__new__(EDCService)
    triple = CandidateTriple(
        id="open_author_claim", subject="Authors", subject_type="Creator",
        predicate="acknowledges limitation", object="Hard to interpret", object_type="LimitationStatement",
        evidence_ids=["ev1"],
    )
    relation = OntologyElement(
        uri="https://example.org/acknowledgesLimitation", label="acknowledgesLimitation", kind="property",
        domain="https://dblp.org/rdf/schema#Publication",
        range="https://example.org/LimitationStatement",
    )

    async def decide(*_args, **_kwargs):
        return CanonicalizationDecision(
            open_triple_id=triple.id, open_predicate=triple.predicate, relation_definition="A publication acknowledges a limitation.",
            candidates=[CanonicalRelationCandidate(uri=relation.uri, label=relation.label, score=0.9)],
            selected_uri=relation.uri, selected_label=relation.label, decision_reason="fixture selected", evidence_ids=triple.evidence_ids,
            outcome="successful_canonicalization",
        ), None

    service._decide = decide
    service.rank_candidates = lambda *_args, **_kwargs: [CanonicalRelationCandidate(uri=relation.uri, label=relation.label, score=0.9)]
    decisions, candidates, _proposals, unresolved = asyncio.run(service.canonicalize(
        [triple], [RelationDefinition(relation=triple.predicate, definition="A publication acknowledges a limitation.")], [relation],
        run_id="run", question="q", ontology_snapshot_id="ontology_snapshot_1",
    ))

    assert candidates == []
    assert unresolved == [triple.id]
    assert decisions[0].selected_uri is None
    assert "domain/range" in decisions[0].decision_reason


def test_materialized_problem_relations_enter_existing_shortlist(settings, tmp_path: Path):
    ontology = OntologyService(settings)
    relations = [
        proposal(ontology, "r1", "addressesProblem", "object_property", definition="A publication addresses a research problem."),
        proposal(ontology, "r1", "focusesOn", "object_property", definition="A publication focuses on a topic or problem."),
        proposal(ontology, "r1", "arguesFor", "object_property", definition="A publication argues for a claim."),
    ]
    snapshot = ontology.build_snapshot([], relations, tmp_path / "active.ttl", run_id="r1")
    catalog = ontology.relation_catalog(snapshot)
    edc = EDCService(settings)
    cases = [
        ("addresses the problem that", "The subject addresses the research problem expressed by the object.", "addressesProblem"),
        ("focuses on", "The subject focuses on the topic expressed by the object.", "focusesOn"),
        ("argues for", "The subject argues for the claim expressed by the object.", "arguesFor"),
    ]

    for predicate, definition, label in cases:
        ranked = edc.rank_candidates(CandidateTriple(subject="Paper", predicate=predicate, object="Claim", evidence_ids=["ev1"]), definition, catalog)
        candidate = next(item for item in ranked if item.label == label)
        assert candidate.score >= edc.CANONICALIZATION_SHORTLIST_THRESHOLD


def test_canonicalization_decisions_expose_outcome_diagnostics(settings):
    edc = EDCService.__new__(EDCService)
    edc.CANONICALIZATION_SHORTLIST_THRESHOLD = EDCService.CANONICALIZATION_SHORTLIST_THRESHOLD
    triple = CandidateTriple(subject="Paper", predicate="unrelated phrase", object="Thing", evidence_ids=["ev1"])

    no_candidates, _ = asyncio.run(edc._decide(triple, "A relation with no matching catalog term.", [], [], "r1", "q", "edc"))
    weak = CanonicalRelationCandidate(uri="https://example.org/weak", label="weak", score=edc.CANONICALIZATION_SHORTLIST_THRESHOLD - 0.01)
    below, _ = asyncio.run(edc._decide(triple, "A relation with no matching catalog term.", [weak], [], "r1", "q", "edc"))

    class FakeVerifier:
        def __init__(self, selected_uri):
            self.selected_uri = selected_uri

        async def run(self, _prompt):
            return SimpleNamespace(output=CanonicalizationChoice(selected_uri=self.selected_uri, reason="fixture verdict"))

    accepted_relation = OntologyElement(uri="https://example.org/relation", label="relatedTo", kind="property")
    shortlist = [CanonicalRelationCandidate(uri=accepted_relation.uri, label=accepted_relation.label, score=edc.CANONICALIZATION_SHORTLIST_THRESHOLD)]
    edc.verify_agent = FakeVerifier(None)
    rejected, _ = asyncio.run(edc._decide(triple, "A relation with a candidate.", shortlist, [accepted_relation], "r1", "q", "edc"))
    edc.verify_agent = FakeVerifier(accepted_relation.uri)
    accepted, _ = asyncio.run(edc._decide(triple, "A relation with a candidate.", shortlist, [accepted_relation], "r1", "q", "edc"))

    assert no_candidates.outcome == "no_candidates"
    assert no_candidates.candidates == []
    assert below.outcome == "candidates_below_shortlist_threshold"
    assert rejected.outcome == "semantic_verifier_rejection"
    assert accepted.outcome == "successful_canonicalization"


def test_legacy_run_edc_is_not_model_visible(settings):
    toolset = _edc_tools(SimpleNamespace(), "edc")
    assert set(toolset.tools) == {"run_edc_pipeline", "retry_edc_canonicalization"}


def test_canonicalize_reuses_existing_phase_artifacts_with_a_new_snapshot(settings, tmp_path: Path):
    store = ArtifactStore(settings, "run_phase_reuse")
    ontology = OntologyService(settings)
    evidence = store.put("evidence_bundle", EvidenceBundle(spans=[]))
    open_ref = store.put(
        "edc_open_triples",
        EDCOpenTriples(edc_execution_id="edc_existing", evidence_bundle_artifact_id=evidence.id, triples=[]),
        parent_artifact_ids=[evidence.id],
    )
    definitions = store.put(
        "edc_relation_definitions",
        EDCRelationDefinitions(edc_execution_id="edc_existing", open_triples_artifact_id=open_ref.id, definitions=[]),
        parent_artifact_ids=[open_ref.id],
    )
    newer_snapshot = ontology.build_snapshot(
        [], [proposal(ontology, "run_phase_reuse", "addressesProblem", "object_property")],
        tmp_path / "newer.ttl", run_id="run_phase_reuse",
    )
    snapshot_ref = store.put("ontology_snapshot", newer_snapshot)

    class FakeEdc:
        def __init__(self):
            self.calls = []

        async def canonicalize(self, open_triples, relation_definitions, relations, **kwargs):
            self.calls.append((open_triples, relation_definitions, relations, kwargs))
            return [], [], [], []

    fake = FakeEdc()
    deps = SimpleNamespace(
        artifacts=store,
        ontology=ontology,
        edc=fake,
        run_id="run_phase_reuse",
        registry=SimpleNamespace(get=lambda _run_id: SimpleNamespace(question="What problem does this paper address?")),
    )
    toolset = _edc_tools(deps, "edc")

    result = asyncio.run(toolset.tools["retry_edc_canonicalization"].function(open_ref.id, definitions.id, snapshot_ref.id))

    assert result["status"] == "blocked"
    assert len(fake.calls) == 1
    assert fake.calls[0][0] == []
    assert fake.calls[0][1] == []
    assert fake.calls[0][3]["ontology_snapshot_id"] == snapshot_ref.id


def test_generation_limit_persists_bounded_canonicalization_attempt(settings, tmp_path: Path):
    store = ArtifactStore(settings, "run_canonicalization_limit")
    ontology = OntologyService(settings)
    evidence = store.put("evidence_bundle", EvidenceBundle(spans=[]))
    open_ref = store.put(
        "edc_open_triples",
        EDCOpenTriples(edc_execution_id="edc_limit", evidence_bundle_artifact_id=evidence.id, triples=[]),
        parent_artifact_ids=[evidence.id],
    )
    definitions = store.put(
        "edc_relation_definitions",
        EDCRelationDefinitions(edc_execution_id="edc_limit", open_triples_artifact_id=open_ref.id, definitions=[]),
        parent_artifact_ids=[open_ref.id],
    )
    snapshot = store.put("ontology_snapshot", ontology.build_snapshot([], [], tmp_path / "snapshot.ttl", run_id="run_canonicalization_limit"))

    class FailingEdc:
        async def canonicalize(self, *_args, **_kwargs):
            raise UnexpectedModelBehavior("Model token limit (2048) exceeded before any response was generated. " + "x" * 2000)

    deps = SimpleNamespace(
        settings=settings, artifacts=store, ontology=ontology, edc=FailingEdc(), run_id="run_canonicalization_limit",
        registry=SimpleNamespace(get=lambda _run_id: SimpleNamespace(question="What limitation is acknowledged?")),
    )
    result = asyncio.run(_edc_tools(deps, "edc").tools["retry_edc_canonicalization"].function(open_ref.id, definitions.id, snapshot.id))

    assert result["status"] == "blocked"
    assert result["failure_kind"] == "model_generation_limit"
    attempt = store.get(result["attempt_ref"]["id"])
    assert attempt["failure_kind"] == "model_generation_limit"
    assert len(attempt["message"]) <= 512
    assert attempt["source_artifact_ids"] == [open_ref.id, definitions.id, evidence.id, snapshot.id]


def test_edc_pipeline_persists_all_phase_artifacts(settings, tmp_path: Path):
    store = ArtifactStore(settings, "run_pipeline")
    ontology = OntologyService(settings)
    evidence_ref = store.put("evidence_bundle", EvidenceBundle(spans=[]))
    snapshot_ref = store.put("ontology_snapshot", ontology.build_snapshot([], [], tmp_path / "snapshot.ttl", run_id="run_pipeline"))
    open_triple = CandidateTriple(id="open_pipeline", subject="Paper", predicate="addresses", object="Problem", evidence_ids=[])
    canonical = CanonicalTriple(subject="Paper", predicate="addressesProblem", object="Problem", predicate_uri="https://example.org/addressesProblem")

    class FakeEdc:
        async def extract(self, *_args, **_kwargs):
            return [open_triple]

        async def define(self, _triples):
            return [RelationDefinition(relation="addresses", definition="A publication addresses a problem.")]

        async def canonicalize(self, *_args, **_kwargs):
            return [], [canonical], [], []

    deps = SimpleNamespace(
        artifacts=store, ontology=ontology, edc=FakeEdc(), run_id="run_pipeline",
        registry=SimpleNamespace(get=lambda _run_id: SimpleNamespace(question="What problem does this paper address?")),
    )
    result = asyncio.run(_edc_tools(deps, "edc").tools["run_edc_pipeline"].function(evidence_ref.id, snapshot_ref.id))

    assert result["status"] == "completed"
    assert result["triple_count"] == 1
    assert result["definition_count"] == 1
    assert set(result["phase_artifact_refs"]) == {"edc_open_triples", "edc_relation_definitions", "edc_canonicalization"}


def test_edc_pipeline_auto_extends_snapshot_and_retries_unmatched_relations(settings, tmp_path: Path):
    store = ArtifactStore(settings, "run_auto_extend")
    ontology = OntologyService(settings)
    evidence_ref = store.put("evidence_bundle", EvidenceBundle(spans=[]))
    snapshot_ref = store.put(
        "ontology_snapshot",
        ontology.build_snapshot([], [], tmp_path / "empty.ttl", run_id="run_auto_extend"),
    )
    open_triple = CandidateTriple(
        id="open_reported", subject="Paper", predicate="reported", object="A finding", evidence_ids=["ev_1"]
    )
    relation_proposal = OntologyProposal(
        proposed_by="edc", question="What was found?", type="object_property",
        uri=ontology.make_proposal_uri("run_auto_extend", "reported"), label="reported",
        definition="The subject reports the finding in the object.", rationale="No active relation matched.",
        evidence_ids=["ev_1"],
    )

    class FakeEdc:
        calls = 0

        async def extract(self, *_args, **_kwargs):
            return [open_triple]

        async def define(self, _triples):
            return [RelationDefinition(relation="reported", definition=relation_proposal.definition)]

        async def canonicalize(self, _triples, _definitions, relations, **kwargs):
            self.calls += 1
            matched = next((relation for relation in relations if relation.label == "reported"), None)
            if matched is None:
                return [], [], [relation_proposal], [open_triple.id]
            return [], [CanonicalTriple(
                subject=open_triple.subject, predicate=matched.label, object=open_triple.object,
                predicate_uri=matched.uri, evidence_ids=open_triple.evidence_ids,
                ontology_snapshot_id=kwargs["ontology_snapshot_id"], canonicalization="exact",
            )], [], []

    fake = FakeEdc()
    deps = SimpleNamespace(
        artifacts=store, ontology=ontology, edc=fake, run_id="run_auto_extend",
        registry=SimpleNamespace(get=lambda _run_id: SimpleNamespace(question="What was found?")),
    )

    result = asyncio.run(
        _edc_tools(deps, "edc").tools["run_edc_pipeline"].function(evidence_ref.id, snapshot_ref.id)
    )

    assert fake.calls == 2
    assert result["status"] == "completed"
    assert len(store.get(result["candidate_ref"]["id"])) == 1
    assert result["auto_extension_snapshot_ref"]["id"]
    assert len(store.list("ontology_acceptance_decision")) == 1
    assert result["candidate_ref"] is not None
    execution = store.get(result["pipeline_execution_ref"]["id"])
    assert execution["edc_execution_id"] == result["edc_execution_id"]
    assert set(execution["phase_durations_ms"]) == {"extract", "define", "canonicalize", "total"}


def test_question_conditioned_extraction_prompt_preserves_relevant_fixture_claims(settings):
    relevant = CandidateTriple(subject="Paper", predicate="addresses", object="research problem", evidence_ids=["ev_problem"])
    supporting = CandidateTriple(subject="Authors", predicate="evaluate", object="proposed method", evidence_ids=["ev_support"])

    class FakeExtractAgent:
        def __init__(self):
            self.prompt = ""

        async def run(self, prompt):
            self.prompt = prompt
            return SimpleNamespace(output=ExtractionOutput(triples=[relevant, supporting]))

    fake = FakeExtractAgent()
    edc = EDCService.__new__(EDCService)
    edc.extract_agent = fake
    bundle = EvidenceBundle(spans=[
        EvidenceSpan(document_id="doc", page=1, text="The paper addresses a research problem.", retrieval_method="test", retrieval_query="q", id="ev_problem"),
        EvidenceSpan(document_id="doc", page=2, text="The authors evaluate the proposed method.", retrieval_method="test", retrieval_query="q", id="ev_support"),
        EvidenceSpan(document_id="doc", page=3, text="Funding: Example Council.", retrieval_method="test", retrieval_query="q", id="ev_funding"),
    ])

    output = asyncio.run(edc.extract(bundle, question="What problem does this paper address?"))

    assert [(triple.subject, triple.predicate, triple.object) for triple in output] == [
        ("Paper", "addresses", "research problem"),
        ("Authors", "evaluate", "proposed method"),
    ]
    assert len({triple.id for triple in output}) == 2
    assert "materially relevant to answering the research question" in edc.extract_agent.prompt
    assert "Do not extract unrelated facts merely because they occur" in edc.extract_agent.prompt
    assert "Funding" in edc.extract_agent.prompt  # Available if a future question asks about funding.
