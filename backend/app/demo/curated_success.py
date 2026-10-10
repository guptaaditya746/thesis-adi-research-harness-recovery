"""Create an evidence-backed successful run for demonstrations.

This module does not call a model and does not represent benchmark performance.
It materializes a small human-curated example through the same artifact contracts
used by the agentic harness so the existing UI can demonstrate provenance,
ontology lineage, verification, and the triple map reliably.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

from app.config import Settings
from app.harness.completion import ScientificCompletionService
from app.models.domain import (
    AgentTraceEvent,
    CandidateTriple,
    CanonicalRelationCandidate,
    CanonicalizationDecision,
    CanonicalTriple,
    EDCCanonicalizationResult,
    EDCOpenTriples,
    EDCRelationDefinitions,
    EvidenceBundle,
    EvidenceSpan,
    ExperimentRun,
    GoldAnnotation,
    GoldTriple,
    OntologyProposal,
    OntologyProposalStatus,
    RelationDefinition,
    ResearchState,
    TripleVerification,
    VerificationDimensions,
    VerificationStatus,
    new_id,
)
from app.services.evaluation import EvaluationService
from app.services.ontology import DBLP, OntologyService
from app.services.pdf import PDFService
from app.services.retrieval import RetrievalService
from app.services.validation import StructuralValidationService
from app.storage.artifacts import ArtifactStore
from app.storage.registry import RunRegistry


DEMO_QUESTION = "[CURATED DEMO] Who authored the paper, and what two KBQA challenges does it identify?"
PAPER_TITLE = "Knowledge Base Question Answering: A Semantic Parsing Perspective"
AUTHORS = ("Yu Gu", "Vardaan Pahuja", "Gong Cheng", "Yu Su")
CHALLENGES = ("schema-level complexity", "fact-level complexity")
AUTHORED_BY = f"{DBLP}authoredBy"


def _inclusive_excerpt(text: str, start: str, end: str) -> str:
    start_at = text.find(start)
    end_at = text.find(end, start_at)
    if start_at < 0 or end_at < 0:
        raise ValueError(f"The curated demo markers were not found: {start!r} … {end!r}")
    return text[start_at : end_at + len(end)]


def _evidence(document_id: str, page: int, text: str, purpose: str) -> EvidenceSpan:
    return EvidenceSpan(
        document_id=document_id,
        page=page,
        text=text,
        source_text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        retrieval_method="curated_demo_exact_match",
        retrieval_query=purpose,
        discovering_agent_id="curated_demo_builder",
        score=1.0,
        reason_selected="Human-curated presentation example; exact text is revalidated against DocumentIR.",
    )


def create_curated_success(settings: Settings, document_id: str) -> ExperimentRun:
    """Persist one transparent successful run backed by exact source text."""
    settings.ensure_dirs()
    pdf = PDFService(settings)
    document = pdf.load(document_id)
    page_path = settings.data_dir / "documents" / document_id / "page_0001.txt"
    page_text = page_path.read_text(encoding="utf-8")
    workspace_marker = page_path.parent / "workspace_id.txt"
    workspace_id = workspace_marker.read_text(encoding="utf-8").strip() if workspace_marker.exists() else "default"

    author_text = _inclusive_excerpt(page_text, PAPER_TITLE, AUTHORS[-1])
    challenge_text = _inclusive_excerpt(
        page_text,
        "We identify and attribute this to two unique challenges of KBQA,",
        "schema-level complexity and fact-level complexity.",
    )
    author_span = _evidence(document_id, 1, author_text, "paper title and authors")
    challenge_span = _evidence(document_id, 1, challenge_text, "two unique challenges of KBQA")
    retrieval = RetrievalService(settings)
    for span in (author_span, challenge_span):
        valid, reason = retrieval.validate_span(document, span)
        if not valid:
            raise ValueError(f"Curated evidence no longer matches the document: {reason}")

    run = ExperimentRun(
        workspace_id=workspace_id,
        document_ids=[document_id],
        target_document_id=document_id,
        question=DEMO_QUESTION,
        status="running",
        active_actor="curated_demo_builder",
        state=ResearchState(
            run_id="pending",
            workspace_id=workspace_id,
            question=DEMO_QUESTION,
            document_ids=[document_id],
            status="running",
        ),
        gold=GoldAnnotation(question=DEMO_QUESTION),
    )
    run.state.run_id = run.id
    registry = RunRegistry(settings)
    registry.create(run)
    artifacts = ArtifactStore(settings, run.id)

    author_ref = artifacts.put(
        "evidence_span", author_span, label="Exact title and author passage",
        artifact_id=author_span.id, creator_agent_id="curated_demo_builder",
    )
    challenge_ref = artifacts.put(
        "evidence_span", challenge_span, label="Exact KBQA challenge passage",
        artifact_id=challenge_span.id, creator_agent_id="curated_demo_builder",
    )
    bundle = EvidenceBundle(
        spans=[author_span, challenge_span],
        sufficiency_summary="Exact page-one evidence covers all six curated triples.",
    )
    bundle_ref = artifacts.put(
        "evidence_bundle",
        bundle,
        label="Curated exact-match evidence",
        creator_agent_id="curated_demo_builder",
        parent_artifact_ids=[author_ref.id, challenge_ref.id],
    )

    proposal_uri = f"https://example.org/thesis/run/{run.id}/ontology#identifiesChallenge"
    proposal = OntologyProposal(
        proposed_by="curated_demo_builder",
        question=DEMO_QUESTION,
        type="object_property",
        uri=proposal_uri,
        label="identifies challenge",
        definition="Relates a research publication to a challenge it explicitly identifies.",
        domain=f"{DBLP}Publication",
        rationale="The DBLP seed describes bibliographic relations but has no relation for research challenges.",
        evidence_ids=[challenge_span.id],
        status=OntologyProposalStatus.ACCEPTED_FOR_RUN,
    )
    proposal_ref = artifacts.put(
        "ontology_proposal", proposal, label="Accepted run-local demo relation", creator_agent_id="curated_demo_builder"
    )
    ontology = OntologyService(settings)
    snapshot = ontology.build_snapshot(
        [AUTHORED_BY, f"{DBLP}Publication", f"{DBLP}Creator"],
        [proposal],
        settings.data_dir / "runs" / run.id / "ontology" / "curated_demo.ttl",
        run_id=run.id,
    ).model_copy(update={"accepted_proposal_artifact_ids": [proposal_ref.id]})
    snapshot_ref = artifacts.put(
        "ontology_snapshot",
        snapshot,
        label="DBLP plus accepted curated demo relation",
        creator_agent_id="curated_demo_builder",
        parent_artifact_ids=[proposal_ref.id],
    )

    open_triples = [
        CandidateTriple(
            subject=PAPER_TITLE,
            predicate="authored by",
            object=author,
            subject_type="Publication",
            object_type="Creator",
            evidence_ids=[author_span.id],
            extraction_note="Human-curated exact-match presentation fixture.",
        )
        for author in AUTHORS
    ] + [
        CandidateTriple(
            subject=PAPER_TITLE,
            predicate="identifies challenge",
            object=challenge,
            subject_type="Publication",
            object_type="ResearchChallenge",
            evidence_ids=[challenge_span.id],
            extraction_note="Human-curated exact-match presentation fixture.",
        )
        for challenge in CHALLENGES
    ]
    edc_execution_id = new_id("edc_demo")
    open_result = EDCOpenTriples(
        edc_execution_id=edc_execution_id,
        evidence_bundle_artifact_id=bundle_ref.id,
        triples=open_triples,
    )
    open_ref = artifacts.put(
        "edc_open_triples",
        open_result,
        label="Curated open triples",
        creator_agent_id="curated_demo_builder",
        parent_artifact_ids=[bundle_ref.id],
    )
    definitions = [
        RelationDefinition(
            relation="authored by",
            definition="The publication is authored by the creator.",
            open_triple_ids=[triple.id for triple in open_triples if triple.predicate == "authored by"],
        ),
        RelationDefinition(
            relation="identifies challenge",
            definition=proposal.definition,
            open_triple_ids=[triple.id for triple in open_triples if triple.predicate == "identifies challenge"],
        ),
    ]
    definitions_payload = EDCRelationDefinitions(
        edc_execution_id=edc_execution_id,
        open_triples_artifact_id=open_ref.id,
        definitions=definitions,
    )
    definitions_ref = artifacts.put(
        "edc_relation_definitions",
        definitions_payload,
        label="Curated relation definitions",
        creator_agent_id="curated_demo_builder",
        parent_artifact_ids=[open_ref.id],
    )

    definition_by_predicate = {definition.relation: definition for definition in definitions}
    candidates: list[CanonicalTriple] = []
    decisions: list[CanonicalizationDecision] = []
    for triple in open_triples:
        predicate_uri = AUTHORED_BY if triple.predicate == "authored by" else proposal_uri
        definition = definition_by_predicate[triple.predicate]
        decision = CanonicalizationDecision(
            open_triple_id=triple.id,
            open_predicate=triple.predicate,
            relation_definition=definition.definition,
            candidates=[CanonicalRelationCandidate(
                uri=predicate_uri,
                label=triple.predicate,
                score=1.0,
                reasons=["Curated exact semantic match for the presentation fixture."],
                domain_compatible=True,
                range_compatible=True,
            )],
            selected_uri=predicate_uri,
            selected_label=triple.predicate,
            decision_reason="Human-curated exact semantic match; not a benchmark model decision.",
            evidence_ids=triple.evidence_ids,
            ontology_proposal_id=proposal.proposal_id if predicate_uri == proposal_uri else None,
            outcome="successful_canonicalization",
        )
        decisions.append(decision)
        candidates.append(CanonicalTriple(
            subject=triple.subject,
            predicate=triple.predicate,
            object=triple.object,
            predicate_uri=predicate_uri,
            subject_type=f"{DBLP}Publication",
            object_type=f"{DBLP}Creator" if predicate_uri == AUTHORED_BY else None,
            original_open_triple=triple,
            canonicalization_decision_id=decision.id,
            relation_definition_id=definition.id,
            evidence_ids=triple.evidence_ids,
            ontology_snapshot_id=snapshot_ref.id,
            confidence_optional=1.0,
            producing_agent_id="curated_demo_builder",
            run_id=run.id,
            canonicalization="exact",
        ))

    canonicalization = EDCCanonicalizationResult(
        edc_execution_id=edc_execution_id,
        open_triples_artifact_id=open_ref.id,
        relation_definitions_artifact_id=definitions_ref.id,
        evidence_bundle_artifact_id=bundle_ref.id,
        ontology_snapshot_artifact_id=snapshot_ref.id,
        candidates=candidates,
        decisions=decisions,
    )
    canonicalization_ref = artifacts.put(
        "edc_canonicalization",
        canonicalization,
        label="Curated canonicalization decisions",
        creator_agent_id="curated_demo_builder",
        parent_artifact_ids=[open_ref.id, definitions_ref.id, bundle_ref.id, snapshot_ref.id],
    )
    candidate_ref = artifacts.put(
        "candidate_canonical_triples",
        candidates,
        label="Curated canonical triple candidates",
        creator_agent_id="curated_demo_builder",
        parent_artifact_ids=[bundle_ref.id, snapshot_ref.id, open_ref.id, definitions_ref.id, canonicalization_ref.id],
    )

    report = StructuralValidationService().validate(candidates, snapshot)
    if not report.rdf_valid or not report.shacl_conforms:
        raise RuntimeError(f"The curated demo no longer passes structural validation: {report.shacl_report}")
    verification_id = new_id("verification")
    checks = [TripleVerification(
        triple_id=triple.id,
        status=VerificationStatus.ACCEPTED,
        dimensions=VerificationDimensions(
            evidence_support="yes",
            ontology_compatibility="yes",
            direction_correct="yes",
            rdf_structural_validity="yes",
            shacl_conformance="yes",
            negation_modality_ok="yes",
            entity_normalization_quality="good",
        ),
        explanation="Curated exact-match fixture checked against source text and the materialized ontology snapshot.",
        evidence_ids=triple.evidence_ids,
        verifier_agent_id="curated_demo_auditor",
        candidate_artifact_id=candidate_ref.id,
        evidence_bundle_artifact_id=bundle_ref.id,
        ontology_snapshot_artifact_id=snapshot_ref.id,
        edc_execution_id=edc_execution_id,
    ) for triple in candidates]
    verification_ref = artifacts.put(
        "verification",
        checks,
        artifact_id=verification_id,
        label="Deterministic curated-demo verification",
        creator_agent_id="curated_demo_auditor",
        parent_artifact_ids=[candidate_ref.id, bundle_ref.id, snapshot_ref.id],
    )
    validation_ref = artifacts.put(
        "structural_validation",
        report,
        label="Curated-demo RDF/SHACL validation",
        creator_agent_id="curated_demo_auditor",
        parent_artifact_ids=[candidate_ref.id, snapshot_ref.id],
    )
    accepted = [triple.model_copy(update={
        "verification_status": VerificationStatus.ACCEPTED,
        "verification_record_id": check.id,
        "verifier_artifact_ids": [verification_ref.id, validation_ref.id],
    }) for triple, check in zip(candidates, checks, strict=True)]
    canonical_ref = artifacts.put(
        "canonical_triples",
        accepted,
        label="Accepted curated demonstration triples",
        creator_agent_id="curated_demo_auditor",
        parent_artifact_ids=[candidate_ref.id, verification_ref.id, validation_ref.id, bundle_ref.id, snapshot_ref.id],
    )
    artifacts.put(
        "rejected_triples",
        [],
        label="No rejected triples in curated fixture",
        creator_agent_id="curated_demo_auditor",
        parent_artifact_ids=[candidate_ref.id, verification_ref.id, validation_ref.id, bundle_ref.id, snapshot_ref.id],
    )

    deps = SimpleNamespace(artifacts=artifacts, retrieval=retrieval, pdf=pdf)
    completion = ScientificCompletionService().attempt(deps, canonical_ref.id)
    if not completion.accepted:
        raise RuntimeError(f"Curated demo failed the host completion contract: {completion.reason}")
    artifacts.put(
        "completion_attempt",
        completion,
        label="Accepted curated demonstration completion",
        creator_agent_id="research_harness",
        parent_artifact_ids=[canonical_ref.id],
    )

    run.final_triples = accepted
    run.final_answer = (
        "Curated demonstration result (not a benchmark model run): the paper is authored by "
        f"{', '.join(AUTHORS)}, and identifies {CHALLENGES[0]} and {CHALLENGES[1]} as unique KBQA challenges."
    )
    run.gold = GoldAnnotation(
        question=DEMO_QUESTION,
        triples=[GoldTriple(subject=t.subject, predicate=t.predicate, object=t.object) for t in accepted],
        evidence_texts=[author_text, challenge_text],
    )
    run.evaluation = EvaluationService().evaluate(
        accepted,
        shacl_conforms=report.shacl_conforms,
        gold=run.gold,
        edc_reached=True,
        ontology_inspected=True,
        ontology_snapshot_created=True,
        candidate_triples_produced=True,
        verification_reached=True,
        canonical_triples_produced=True,
        completion_attempts=1,
    )
    run.evaluation.notes.append("Human-curated deterministic presentation fixture; exclude from benchmark aggregates.")
    artifacts.put("evaluation", run.evaluation, label="Curated demo evaluation", creator_agent_id="research_harness")
    run.events = [
        AgentTraceEvent(run_id=run.id, type="evidence", actor="curated_demo_builder", action="exact_match", status="completed", title="Exact evidence selected", summary="Two page-one spans were matched verbatim to the parsed document."),
        AgentTraceEvent(run_id=run.id, type="ontology", actor="curated_demo_builder", action="canonicalize", status="completed", title="Relations canonicalized", summary="DBLP authoredBy and one explicit run-local challenge relation were materialized."),
        AgentTraceEvent(run_id=run.id, type="validation", actor="curated_demo_auditor", action="verify", status="completed", title="Evidence and structure verified", summary="All curated triples passed exact-source and RDF/SHACL checks."),
        AgentTraceEvent(run_id=run.id, type="run", actor="research_harness", action="complete", status="completed", title="Curated demonstration completed", summary="Six accepted triples are ready for the presentation UI."),
    ]
    run.status = "completed"
    run.active_actor = None
    run.resumable = False
    run.error = None
    if run.state:
        run.state.status = "completed"
        run.state.canonical_triple_ids = [triple.id for triple in accepted]
        run.state.evidence_ids = [author_span.id, challenge_span.id]
        run.state.ontology_snapshot_id = snapshot_ref.id
    registry.persist(run)
    return run
