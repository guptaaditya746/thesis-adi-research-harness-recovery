from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:16]}"


def stable_hash(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

SpecialistCapability = Literal[
    "documents", "ontology", "sparql", "edc", "verification",
    "entity_normalization", "artifacts", "shell", "filesystem",
]


class ModelProfile(StrEnum):
    FAST = "fast"
    BALANCED = "balanced"
    REASONING = "reasoning"
    CRITIC = "critic"
    EXTRACTOR = "extractor"


class ModelCallMetadata(BaseModel):
    provider: str = "litellm-openai-compatible"
    model: str
    profile: ModelProfile
    temperature: float = 0.0
    max_tokens: int = 4096
    reasoning_settings: dict[str, Any] = Field(default_factory=dict)
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: float | None = None
    retry_number: int = 0
    finish_reason: str | None = None
    request_id: str | None = None


class ArtifactRef(BaseModel):
    id: str
    kind: str
    path: str | None = None
    label: str | None = None
    content_sha256: str | None = None


class ArtifactEnvelope(BaseModel):
    artifact_id: str
    artifact_type: str
    run_id: str
    creator_agent_id: str
    parent_artifact_ids: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    schema_version: str = "1.0"
    content_sha256: str
    model_metadata: ModelCallMetadata | None = None
    content: Any


class DecisionRecord(BaseModel):
    id: str = Field(default_factory=lambda: new_id("decision"))
    action: str
    reason: str
    missing_information: str | None = None
    confidence: Literal["low", "medium", "high"] = "medium"
    artifact_ids: list[str] = Field(default_factory=list)


class AgentSpec(BaseModel):
    id: str = Field(default_factory=lambda: new_id("agent"))
    role: str
    objective: str
    instructions: str
    required_capabilities: list[SpecialistCapability] = Field(default_factory=list)
    input_artifact_ids: list[str] = Field(default_factory=list)
    output_schema: str = "specialist_result"
    model_profile: ModelProfile = ModelProfile.BALANCED
    context_budget: int = 12_000
    max_tool_calls: int = 8
    parent_agent_id: str = "research_director"
    creation_reason: str


class SpecialistResult(BaseModel):
    summary: str
    findings: list[str] = Field(default_factory=list)
    artifact_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    recommended_next_actions: list[str] = Field(default_factory=list)
    decision: DecisionRecord | None = None


class SourcePosition(BaseModel):
    page: int | None = None
    section_id: str | None = None
    block_id: str | None = None
    paragraph_index: int | None = None
    start_char: int | None = None
    end_char: int | None = None
    bbox: list[float] | None = None


class DocumentMetadata(BaseModel):
    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    doi: str | None = None
    year: int | None = None
    venue: str | None = None


class DocumentBlock(BaseModel):
    id: str
    page: int
    section_id: str | None = None
    paragraph_index: int | None = None
    text: str
    start_char: int | None = None
    end_char: int | None = None
    bbox: list[float] | None = None


class DocumentSection(BaseModel):
    id: str
    heading: str
    level: int = 1
    page_start: int
    page_end: int
    text: str
    block_ids: list[str] = Field(default_factory=list)


class DocumentReference(BaseModel):
    id: str
    raw_text: str
    doi: str | None = None
    title: str | None = None


class DocumentStatus(StrEnum):
    PENDING = "pending"
    PARSING = "parsing"
    READY = "ready"
    FAILED = "failed"


class ScientificDocument(BaseModel):
    """Versioned parser-neutral DocumentIR persisted for each paper."""
    id: str = Field(default_factory=lambda: new_id("doc"))
    filename: str
    sha256: str
    page_count: int
    created_at: datetime = Field(default_factory=utc_now)
    status: DocumentStatus = DocumentStatus.READY
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    sections: list[DocumentSection] = Field(default_factory=list)
    blocks: list[DocumentBlock] = Field(default_factory=list)
    references: list[DocumentReference] = Field(default_factory=list)
    tables: list[dict[str, Any]] = Field(default_factory=list)
    figures: list[dict[str, Any]] = Field(default_factory=list)
    captions: list[str] = Field(default_factory=list)
    source_positions: list[SourcePosition] = Field(default_factory=list)
    workspace: str
    parse_engine: str = "pymupdf"
    parser_version: str | None = None
    parse_warnings: list[str] = Field(default_factory=list)
    sidecar_manifest: str | None = None
    error: str | None = None


class EvidenceSpan(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: new_id("ev"))
    document_id: str
    page: int
    section_id: str | None = None
    block_id: str | None = None
    paragraph_index: int | None = None
    text: str
    source_text_sha256: str | None = None
    start_char: int | None = None
    end_char: int | None = None
    bbox: list[float] | None = None
    retrieval_method: str
    retrieval_query: str | None = None
    discovering_agent_id: str = "research_director"
    created_at: datetime = Field(default_factory=utc_now)
    score: float | None = None
    reason_selected: str | None = None

    @field_validator("text")
    @classmethod
    def evidence_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Evidence text may not be blank")
        return value


class RetrievalCandidateBundle(BaseModel):
    """Immutable orientation results; candidates are not accepted evidence."""

    query: str
    spans: list[EvidenceSpan] = Field(default_factory=list)
    document_ids: list[str] = Field(default_factory=list)
    retrieval_method: str = "sqlite_fts5_bm25"


class EvidenceBundle(BaseModel):
    id: str = Field(default_factory=lambda: new_id("bundle"))
    spans: list[EvidenceSpan]
    sufficiency_summary: str = ""


class QuestionAnalysis(BaseModel):
    normalized_question: str
    intent: str
    key_entities: list[str] = Field(default_factory=list)
    key_concepts: list[str] = Field(default_factory=list)
    information_needs: list[str] = Field(default_factory=list)
    retrieval_terms: list[str] = Field(default_factory=list)
    decision_summary: str


class OntologyElement(BaseModel):
    uri: str
    label: str
    kind: Literal["class", "property"]
    description: str | None = None
    domain: str | None = None
    range: str | None = None
    parents: list[str] = Field(default_factory=list)
    equivalents: list[str] = Field(default_factory=list)
    source: Literal["seed", "run_overlay"] = "seed"


class OntologyProposalStatus(StrEnum):
    PROPOSED = "proposed"
    ACCEPTED_FOR_RUN = "accepted_for_run"
    REJECTED = "rejected"
    DEPRECATED = "deprecated"


class OntologyProposal(BaseModel):
    proposal_id: str = Field(default_factory=lambda: new_id("ontprop"))
    proposed_by: str = "research_director"
    question: str
    type: Literal["class", "property"]
    uri: str
    label: str
    definition: str
    domain: str | None = None
    range: str | None = None
    rationale: str
    evidence_ids: list[str] = Field(default_factory=list)
    status: OntologyProposalStatus = OntologyProposalStatus.PROPOSED


class SubOntologyArtifact(BaseModel):
    id: str = Field(default_factory=lambda: new_id("onto"))
    turtle_path: str
    seed_schema_version: str = "2024-06-14"
    selected_elements: list[OntologyElement]
    dependency_uris: list[str] = Field(default_factory=list)
    proposals: list[OntologyProposal] = Field(default_factory=list)
    validation_warnings: list[str] = Field(default_factory=list)
    content_sha256: str | None = None


class SPARQLArtifact(BaseModel):
    id: str = Field(default_factory=lambda: new_id("sparql"))
    query: str
    purpose: str
    expected_variables: list[str] = Field(default_factory=list)
    semantic_terms: list[str] = Field(default_factory=list)
    retrieval_hints: list[str] = Field(default_factory=list)
    valid: bool = False
    validation_error: str | None = None
    preview_rows: list[dict[str, str]] = Field(default_factory=list)
    repair_attempts: list[str] = Field(default_factory=list)


class CandidateTriple(BaseModel):
    id: str = Field(default_factory=lambda: new_id("open"))
    subject: str
    predicate: str
    object: str
    subject_type: str | None = None
    object_type: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    extraction_note: str | None = None


class RelationDefinition(BaseModel):
    relation: str
    definition: str


class CanonicalRelationCandidate(BaseModel):
    uri: str
    label: str
    score: float
    reasons: list[str] = Field(default_factory=list)
    domain_compatible: bool | None = None
    range_compatible: bool | None = None


class CanonicalizationDecision(BaseModel):
    id: str = Field(default_factory=lambda: new_id("canon"))
    open_triple_id: str
    open_predicate: str
    relation_definition: str
    candidates: list[CanonicalRelationCandidate] = Field(default_factory=list)
    selected_uri: str | None = None
    selected_label: str | None = None
    rejected_uris: list[str] = Field(default_factory=list)
    decision_reason: str
    evidence_ids: list[str] = Field(default_factory=list)
    ontology_proposal_id: str | None = None


class SupportType(StrEnum):
    DIRECT = "direct"
    INFERRED = "inferred"
    WEAK = "weak"


class VerificationStatus(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    UNCERTAIN = "uncertain"


class CanonicalTriple(BaseModel):
    id: str = Field(default_factory=lambda: new_id("triple"))
    subject: str
    predicate: str
    object: str
    predicate_uri: str | None = None
    subject_type: str | None = None
    object_type: str | None = None
    original_open_triple: CandidateTriple | None = None
    canonicalization_decision_id: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    support_type: SupportType = SupportType.DIRECT
    verification_status: VerificationStatus = VerificationStatus.UNCERTAIN
    verifier_artifact_ids: list[str] = Field(default_factory=list)
    ontology_snapshot_id: str | None = None
    confidence_optional: float | None = None
    producing_agent_id: str = "edc"
    run_id: str | None = None
    canonicalization: Literal["exact", "deterministic_match", "llm_verified", "open_relation"] = "open_relation"


class VerificationDimensions(BaseModel):
    evidence_support: Literal["yes", "no", "uncertain"]
    ontology_compatibility: Literal["yes", "no", "uncertain"]
    direction_correct: Literal["yes", "no", "uncertain"]
    rdf_structural_validity: Literal["yes", "no", "uncertain"]
    shacl_conformance: Literal["yes", "no", "not_applicable", "uncertain"]
    negation_modality_ok: Literal["yes", "no", "uncertain"]
    entity_normalization_quality: Literal["good", "weak", "uncertain"]


class TripleVerification(BaseModel):
    id: str = Field(default_factory=lambda: new_id("verify"))
    triple_id: str
    status: VerificationStatus
    dimensions: VerificationDimensions
    explanation: str
    evidence_ids: list[str]
    verifier_agent_id: str


class RejectedTriple(BaseModel):
    triple: CanonicalTriple
    reason: str
    verification_artifact_ids: list[str] = Field(default_factory=list)


# Compatibility wrapper retained for API/frontend consumers; final scientific product is CanonicalTriple.
class ProvenanceRecord(BaseModel):
    triple_id: str
    document_id: str | None = None
    document_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str]
    inference_type: Literal["direct", "aggregated", "inferred", "ontology_derived", "uncertain"]
    method: str


class VerifiedTriple(BaseModel):
    triple: CanonicalTriple
    status: Literal["supported", "weakly_supported", "unsupported", "conflicting"]
    explanation: str
    provenance: ProvenanceRecord
    verification: TripleVerification | None = None


class ValidationReport(BaseModel):
    rdf_valid: bool
    shacl_conforms: bool
    shacl_report: str
    rejected_triple_ids: list[str] = Field(default_factory=list)
    structural_errors: list[str] = Field(default_factory=list)


class EntityResolutionCandidate(BaseModel):
    canonical_name: str
    score: float
    source: str
    external_id: str | None = None


class EntityNormalizationDecision(BaseModel):
    original_mention: str
    canonical_name: str
    external_ids: dict[str, str] = Field(default_factory=dict)
    aliases: list[str] = Field(default_factory=list)
    candidates: list[EntityResolutionCandidate] = Field(default_factory=list)
    method: str


class EvaluationResult(BaseModel):
    exact_precision: float | None = None
    exact_recall: float | None = None
    exact_f1: float | None = None
    normalized_precision: float | None = None
    normalized_recall: float | None = None
    normalized_f1: float | None = None
    subject_accuracy: float | None = None
    predicate_accuracy: float | None = None
    object_accuracy: float | None = None
    evidence_recall: float | None = None
    evidence_precision: float | None = None
    provenance_coverage: float = 0.0
    evidence_support_accuracy: float | None = None
    supported_ratio: float = 0.0
    shacl_conformance: bool = False
    tool_actions: int = 0
    subagents_spawned: int = 0
    retries: int = 0
    model_escalations: int = 0
    notes: list[str] = Field(default_factory=list)


class GoldTriple(BaseModel):
    subject: str
    predicate: str
    object: str


class GoldAnnotation(BaseModel):
    question: str
    triples: list[GoldTriple] = Field(default_factory=list)
    evidence_texts: list[str] = Field(default_factory=list)


class Hypothesis(BaseModel):
    id: str = Field(default_factory=lambda: new_id("hyp"))
    statement: str
    status: Literal["open", "supported", "rejected", "uncertain"] = "open"
    evidence_ids: list[str] = Field(default_factory=list)


class RecoveryAction(BaseModel):
    id: str = Field(default_factory=lambda: new_id("recovery"))
    level: int
    action: str
    trigger: str
    agent_id: str | None = None
    from_profile: ModelProfile | None = None
    to_profile: ModelProfile | None = None
    retry_number: int = 0
    success: bool | None = None
    artifact_ids: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)


class AgentTraceEvent(BaseModel):
    id: str = Field(default_factory=lambda: new_id("evt"))
    run_id: str
    type: Literal[
        "run", "agent", "delegate", "tool", "artifact", "evidence", "ontology", "sparql", "triple",
        "validation", "answer", "decision", "recovery", "model", "message", "persistence", "system"
    ] = "system"
    actor: str = "research_harness"
    action: str
    status: Literal["started", "completed", "failed", "info"] = "info"
    title: str
    summary: str = ""
    artifact_id: str | None = None
    parent_event_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=utc_now)


class ResearchState(BaseModel):
    run_id: str
    workspace_id: str
    question: str
    current_goal: str = "Produce evidence-backed canonicalized triples relevant to the research question."
    current_plan: list[str] = Field(default_factory=list)
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    active_agent_ids: list[str] = Field(default_factory=list)
    document_ids: list[str] = Field(default_factory=list)
    ontology_snapshot_id: str | None = None
    ontology_proposal_ids: list[str] = Field(default_factory=list)
    sparql_artifact_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    rejected_evidence_ids: list[str] = Field(default_factory=list)
    open_triple_ids: list[str] = Field(default_factory=list)
    canonical_triple_ids: list[str] = Field(default_factory=list)
    rejected_triple_ids: list[str] = Field(default_factory=list)
    verification_artifact_ids: list[str] = Field(default_factory=list)
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)
    context_budget: int = 45_000
    tool_budget: int = 28
    token_budget: int = 64_000
    cost_budget: float | None = None
    model_calls: int = 0
    tool_calls: int = 0
    delegate_calls: int = 0
    status: Literal["queued", "running", "completed", "partial", "failed", "cancelled"] = "queued"


class PartialResearchResult(BaseModel):
    supported_findings: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    rejected_triple_ids: list[str] = Field(default_factory=list)
    failure_reason: str
    recommended_recovery_option: str


class ResearchConclusion(BaseModel):
    answer: str = ""
    answer_summary: str = ""
    triple_artifact_id: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    unresolved_issues: list[str] = Field(default_factory=list)
    completion_reason: str
    status: Literal["completed", "partial"] = "completed"


class ExperimentRun(BaseModel):
    id: str = Field(default_factory=lambda: new_id("run"))
    workspace_id: str
    document_ids: list[str]
    question: str
    status: Literal["queued", "running", "completed", "partial", "failed", "cancelled"] = "queued"
    active_actor: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    state: ResearchState | None = None
    events: list[AgentTraceEvent] = Field(default_factory=list)
    final_triples: list[CanonicalTriple] = Field(default_factory=list)
    rejected_triples: list[RejectedTriple] = Field(default_factory=list)
    final_answer: str | None = None
    evaluation: EvaluationResult | None = None
    gold: GoldAnnotation | None = None
    langtrace_trace_id: str | None = None
    resumable: bool = False
    error: str | None = None


class StartRunRequest(BaseModel):
    workspace_id: str = "default"
    document_ids: list[str]
    question: str
    gold: GoldAnnotation | None = None
