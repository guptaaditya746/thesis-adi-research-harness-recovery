from __future__ import annotations

import hashlib
from uuid import uuid4
from typing import Any

from pydantic_ai import FunctionToolset

from app.harness.contracts import (
    SUPPORTED_CAPABILITIES,
    ArtifactIntegrityError,
    InvalidArtifactKind,
    ScientificStateBlocked,
    load_payload,
    metadata,
    require_kind,
    resolve_candidate_lineage,
    bounded_error,
)
from app.harness.deps import HarnessDeps
from app.models.domain import (
    CanonicalTriple,
    EDCOpenTriples,
    EDCRelationDefinitions,
    EDCCanonicalizationResult,
    EvidenceBundle,
    EvidenceSpan,
    OntologyProposal,
    OntologyAcceptanceDecision,
    OntologyProposalStatus,
    RetrievalCandidateBundle,
    RetrievalCandidatePreview,
    SPARQLArtifact,
    SubOntologyArtifact,
    VerificationStatus,
    stable_hash,
)


class CorrectableToolInputError(ValueError):
    """A tool-input/provenance mistake the model can correct by rereading state."""

    def __init__(self, message: str) -> None:
        super().__init__(bounded_error(message))


def persist_candidate_evidence_from_bundle(
    deps, agent_id: str, bundle_artifact_id: str, candidate_ids: list[str], sufficiency_summary: str,
) -> dict:
    if not candidate_ids:
        raise CorrectableToolInputError("Select at least one candidate_id from the retrieval candidate bundle.")
    bundle = load_payload(deps, bundle_artifact_id, "retrieval_candidate_bundle", RetrievalCandidateBundle)
    by_id = {span.id: span for span in bundle.spans}
    unknown = [candidate_id for candidate_id in candidate_ids if candidate_id not in by_id]
    if unknown:
        raise CorrectableToolInputError(
            f"Unknown candidate_id(s): {unknown}. Reread the retrieval candidate bundle and select existing IDs."
        )
    normalized_candidate_ids = sorted(set(candidate_ids))
    spans = [by_id[candidate_id] for candidate_id in normalized_candidate_ids]
    if len(spans) > deps.settings.max_evidence_spans:
        raise CorrectableToolInputError(
            f"At most {deps.settings.max_evidence_spans} evidence spans may be persisted per bundle."
        )
    stable: list[EvidenceSpan] = []
    refs = []
    for span in spans:
        _check_doc(deps, span.document_id)
        valid, reason = deps.retrieval.validate_span(deps.pdf.load(span.document_id), span)
        if not valid:
            raise CorrectableToolInputError(
                f"Candidate {span.id} failed source-resolution validation: {reason}. "
                "Reread the source and select a different candidate; do not reconstruct its text."
            )
        stable.append(span)
        refs.append(deps.artifacts.put(
            "evidence_span", span, label=f"Evidence {span.document_id} p.{span.page}",
            artifact_id=span.id, creator_agent_id=agent_id,
        ).model_dump(mode="json"))
    # Candidate selection, not model wording, determines the immutable evidence identity.
    evidence_artifact_id = "evidence_bundle_" + stable_hash({
        "candidate_bundle_artifact_id": bundle_artifact_id,
        "candidate_ids": normalized_candidate_ids,
    })[:16]
    evidence = EvidenceBundle(id=evidence_artifact_id, spans=stable, sufficiency_summary="")
    ref = deps.artifacts.put(
        "evidence_bundle", evidence, label="Selected immutable evidence", creator_agent_id=agent_id,
        parent_artifact_ids=[bundle_artifact_id, *[x["id"] for x in refs]],
        artifact_id=evidence_artifact_id,
    )
    return {"artifact_ref": ref.model_dump(mode="json"), "evidence_refs": refs, "bundle": evidence.model_dump(mode="json"),
            "sufficiency_summary": sufficiency_summary}


def build_specialist_toolsets(deps: HarnessDeps, capability_names: list[str], agent_id: str) -> list[FunctionToolset]:
    unknown = sorted(set(capability_names) - SUPPORTED_CAPABILITIES)
    if unknown:
        raise ValueError(f"Unknown specialist capabilities: {', '.join(unknown)}")
    return [builder(deps, agent_id) for name, builder in _BUILDERS.items() if name in capability_names]


def read_retrieval_candidate_bundle(deps, artifact_id: str, offset: int = 0, limit: int = 8, preview_chars: int = 240) -> dict:
    """Read bounded candidate metadata; full source text remains in document tools."""
    bundle = load_payload(deps, artifact_id, "retrieval_candidate_bundle", RetrievalCandidateBundle)
    start = max(0, offset)
    stop = start + max(1, min(limit, 20))
    chars = max(32, min(preview_chars, 500))
    candidates = []
    for span in bundle.spans[start:stop]:
        truncated = len(span.text) > chars
        candidates.append(RetrievalCandidatePreview(
            candidate_id=span.id,
            preview=span.text if not truncated else span.text[:chars - 1] + "…",
            preview_truncated=truncated,
            document_id=span.document_id,
            page=span.page,
            section_id=span.section_id,
            block_id=span.block_id,
            score=span.score,
        ).model_dump(mode="json"))
    return {
        "artifact_id": artifact_id,
        "query": bundle.query,
        "document_ids": bundle.document_ids,
        "retrieval_method": bundle.retrieval_method,
        "candidate_count": len(bundle.spans),
        "offset": start,
        "limit": stop - start,
        "candidates": candidates,
        "provenance_status": "retrieval_candidates_only",
    }


def _document_tools(deps: HarnessDeps, agent_id: str) -> FunctionToolset:
    ts = FunctionToolset(instructions=(
        "Use lexical/structural retrieval only. Retrieval candidate previews are not evidence and do not contain source payloads. "
        "Read the candidate bundle, select candidates by candidate_id, then call persist_candidate_evidence. "
        "Never construct or edit EvidenceSpan provenance fields, and never pass preview text to persist_evidence."
    ))

    @ts.tool_plain
    def list_documents() -> list[dict]:
        deps.ensure_not_cancelled()
        return [deps.retrieval.inspect_metadata(deps.pdf.load(x)) for x in deps.document_ids]

    @ts.tool_plain
    def read_retrieval_candidates(artifact_id: str, offset: int = 0, limit: int = 8, preview_chars: int = 240) -> dict:
        return read_retrieval_candidate_bundle(deps, artifact_id, offset, limit, preview_chars)

    @ts.tool_plain
    def persist_candidate_evidence(bundle_artifact_id: str, candidate_ids: list[str], sufficiency_summary: str) -> dict:
        """Persist exact spans selected by ID; never accept model-reconstructed preview text."""
        return persist_candidate_evidence_from_bundle(deps, agent_id, bundle_artifact_id, candidate_ids, sufficiency_summary)

    @ts.tool_plain
    def search_documents(query: str, document_ids: list[str] | None = None, limit_per_document: int = 8) -> list[dict]:
        deps.ensure_not_cancelled()
        ids = document_ids or [deps.target_document_id]
        out = []
        for doc_id in ids:
            if doc_id not in deps.document_ids:
                continue
            out.extend(x.model_dump(mode="json") for x in deps.retrieval.search(deps.pdf.load(doc_id), query, min(limit_per_document, 20), agent_id=agent_id))
        return sorted(out, key=lambda x: -(x.get("score") or 0.0))[:50]

    @ts.tool_plain
    def search_sections(document_id: str, query: str, limit: int = 10) -> list[dict]:
        _check_doc(deps, document_id)
        return [x.model_dump(mode="json") for x in deps.retrieval.search_sections(deps.pdf.load(document_id), query, limit, agent_id=agent_id)]

    @ts.tool_plain
    def search_exact_phrase(document_id: str, phrase: str, limit: int = 12) -> list[dict]:
        _check_doc(deps, document_id)
        return [x.model_dump(mode="json") for x in deps.retrieval.find_phrase(deps.pdf.load(document_id), phrase, limit, agent_id=agent_id)]

    @ts.tool_plain
    def search_entities(document_id: str, entity: str, limit: int = 12) -> list[dict]:
        _check_doc(deps, document_id)
        return [x.model_dump(mode="json") for x in deps.retrieval.search_entities(deps.pdf.load(document_id), entity, limit, agent_id=agent_id)]

    @ts.tool_plain
    def search_references(document_id: str, query: str, limit: int = 20) -> list[dict]:
        _check_doc(deps, document_id)
        return deps.retrieval.search_references(deps.pdf.load(document_id), query, limit)

    @ts.tool_plain
    def search_headings(document_id: str, query: str, limit: int = 20) -> list[dict]:
        _check_doc(deps, document_id)
        return deps.retrieval.search_headings(deps.pdf.load(document_id), query, limit)

    @ts.tool_plain
    def list_sections(document_id: str) -> list[dict]:
        _check_doc(deps, document_id)
        return deps.retrieval.list_sections(deps.pdf.load(document_id))

    @ts.tool_plain
    def inspect_document_metadata(document_id: str) -> dict:
        _check_doc(deps, document_id)
        return deps.retrieval.inspect_metadata(deps.pdf.load(document_id))

    @ts.tool_plain
    def read_page(document_id: str, page: int) -> str:
        _check_doc(deps, document_id)
        return _bounded(deps, deps.retrieval.read_page(deps.pdf.load(document_id), page))

    @ts.tool_plain
    def read_section(document_id: str, section_id: str) -> str:
        _check_doc(deps, document_id)
        return _bounded(deps, deps.retrieval.read_section(deps.pdf.load(document_id), section_id))

    @ts.tool_plain
    def read_block(document_id: str, block_id: str) -> dict:
        _check_doc(deps, document_id)
        return deps.retrieval.read_block(deps.pdf.load(document_id), block_id).model_dump(mode="json")

    @ts.tool_plain
    def get_nearby_blocks(document_id: str, block_id: str, radius: int = 2) -> list[dict]:
        _check_doc(deps, document_id)
        return [x.model_dump(mode="json") for x in deps.retrieval.nearby_blocks(deps.pdf.load(document_id), block_id, min(radius, 5))]

    @ts.tool_plain
    def expand_context(evidence: EvidenceSpan, paragraphs: int = 2) -> dict:
        _check_doc(deps, evidence.document_id)
        return deps.retrieval.expand(deps.pdf.load(evidence.document_id), evidence, min(paragraphs, 4)).model_dump(mode="json")

    @ts.tool_plain
    def persist_evidence(spans: list[EvidenceSpan], sufficiency_summary: str) -> dict:
        deps.ensure_not_cancelled()
        if len(spans) > deps.settings.max_evidence_spans:
            raise CorrectableToolInputError(f"At most {deps.settings.max_evidence_spans} evidence spans may be persisted per bundle")
        chars = sum(len(x.text) for x in spans)
        if chars > deps.settings.max_evidence_chars:
            raise CorrectableToolInputError(
                f"Evidence bundle is {chars} chars; narrow it below MAX_EVIDENCE_CHARS={deps.settings.max_evidence_chars}. "
                "Evidence text is never truncated because provenance must remain exact."
            )
        seen = set()
        stable: list[EvidenceSpan] = []
        refs = []
        for span in spans:
            if span.id in seen:
                continue
            seen.add(span.id)
            _check_doc(deps, span.document_id)
            valid, reason = deps.retrieval.validate_span(deps.pdf.load(span.document_id), span)
            if not valid:
                raise CorrectableToolInputError(f"Evidence {span.id} failed source-resolution validation: {reason}")
            stable.append(span)
            refs.append(deps.artifacts.put(
                "evidence_span", span, label=f"Evidence {span.document_id} p.{span.page}",
                artifact_id=span.id, creator_agent_id=agent_id,
            ).model_dump(mode="json"))
        bundle = EvidenceBundle(spans=stable, sufficiency_summary=sufficiency_summary)
        ref = deps.artifacts.put(
            "evidence_bundle", bundle, label="Selected immutable evidence",
            creator_agent_id=agent_id, parent_artifact_ids=[x["id"] for x in refs],
        )
        return {"artifact_ref": ref.model_dump(mode="json"), "evidence_refs": refs, "bundle": bundle.model_dump(mode="json")}

    return ts


def _ontology_tools(deps: HarnessDeps, agent_id: str) -> FunctionToolset:
    ts = FunctionToolset(instructions=(
        "DBLP is immutable. Use EDC relation definitions as the primary query terms before materializing a snapshot. "
        "Any missing relation/class must be represented as an explicit run-local OntologyProposal; never mint a dblp: URI."
    ))

    @ts.tool_plain
    def search_ontology(terms: list[str], kind: str | None = None, limit: int = 40) -> list[dict]:
        return [x.model_dump(mode="json") for x in deps.ontology.search(terms, min(limit, 80), kind)]

    @ts.tool_plain
    def inspect_ontology_term(uri: str) -> dict:
        return deps.ontology.inspect(uri).model_dump(mode="json")

    @ts.tool_plain
    def search_ontology_for_edc_definitions(edc_relation_definitions_artifact_id: str, limit: int = 40) -> list[dict]:
        definitions = load_payload(deps, edc_relation_definitions_artifact_id, "edc_relation_definitions", EDCRelationDefinitions)
        terms = [f"{x.relation} {x.definition}" for x in definitions.definitions]
        return [x.model_dump(mode="json") for x in deps.ontology.search(terms, min(limit, 80), "property")]

    @ts.tool_plain
    def create_ontology_proposal(
        label: str, definition: str, proposal_type: str, rationale: str,
        domain: str | None = None, range: str | None = None, evidence_ids: list[str] | None = None,
    ) -> dict:
        ptype = "class" if proposal_type == "class" else "property"
        proposal = OntologyProposal(
            proposed_by=agent_id, question=deps.registry.get(deps.run_id).question, type=ptype,
            uri=deps.ontology.make_proposal_uri(deps.run_id, label), label=label, definition=definition,
            domain=domain, range=range, rationale=rationale, evidence_ids=evidence_ids or [],
        )
        ref = deps.artifacts.put("ontology_proposal", proposal, label=f"Proposed {ptype}: {label}", creator_agent_id=agent_id)
        return {"artifact_ref": ref.model_dump(mode="json"), "proposal": proposal.model_dump(mode="json")}

    @ts.tool_plain
    def accept_ontology_proposal_for_run(
        proposal_artifact_id: str,
        decision_reason: str,
        competing_dblp_candidates: list[str] | None = None,
    ) -> dict:
        require_kind(deps, proposal_artifact_id, "ontology_proposal")
        raw = deps.artifacts.get(proposal_artifact_id)
        proposal = OntologyProposal.model_validate(raw).model_copy(update={"status": OntologyProposalStatus.ACCEPTED_FOR_RUN})
        ref = deps.artifacts.put(
            "ontology_proposal", proposal, label=f"Accepted for run: {proposal.label}", creator_agent_id=agent_id,
            parent_artifact_ids=[proposal_artifact_id],
        )
        decision = deps.artifacts.put(
            "ontology_acceptance_decision", OntologyAcceptanceDecision(
                proposal_artifact_id=proposal_artifact_id, reason=decision_reason,
                deciding_actor=agent_id, competing_dblp_candidates=competing_dblp_candidates or [],
            ),
            creator_agent_id=agent_id, parent_artifact_ids=[ref.id],
        )
        return {"proposal_ref": ref.model_dump(mode="json"), "decision_ref": decision.model_dump(mode="json"), "proposal": proposal.model_dump(mode="json")}

    @ts.tool_plain
    def materialize_ontology_snapshot(selected_uris: list[str], accepted_proposal_artifact_ids: list[str] | None = None) -> dict:
        proposal_ids = accepted_proposal_artifact_ids or []
        proposals = []
        for x in proposal_ids:
            require_kind(deps, x, "ontology_proposal")
            proposal = OntologyProposal.model_validate(deps.artifacts.get(x))
            if proposal.status != OntologyProposalStatus.ACCEPTED_FOR_RUN:
                raise ScientificStateBlocked("Ontology proposals must be explicitly accepted before snapshot materialization.", requirements=("ontology_snapshot",))
            proposals.append(proposal)
        digest = hashlib.sha256(("|".join(sorted(selected_uris)) + "|" + "|".join(sorted(proposal_ids))).encode()).hexdigest()[:12]
        path = deps.artifacts.root / f"ontology_snapshot_{digest}.ttl"
        snapshot = deps.ontology.build_snapshot(selected_uris, proposals, path, run_id=deps.run_id)
        snapshot = snapshot.model_copy(update={"accepted_proposal_artifact_ids": sorted(proposal_ids)})
        ref = deps.artifacts.put(
            "ontology_snapshot", snapshot, label="Active question-specific ontology snapshot", creator_agent_id=agent_id,
            parent_artifact_ids=proposal_ids,
        )
        return {"artifact_ref": ref.model_dump(mode="json"), "snapshot": snapshot.model_dump(mode="json")}

    return ts


def _sparql_tools(deps: HarnessDeps, agent_id: str) -> FunctionToolset:
    ts = FunctionToolset(instructions="SPARQL is a semantic intent/schema pattern. It never retrieves facts directly from raw PDFs and must not imply a persistent instance KG.")

    @ts.tool_plain
    def validate_sparql(artifact: SPARQLArtifact, ontology_snapshot_artifact_id: str) -> dict:
        snapshot = SubOntologyArtifact.model_validate(deps.artifacts.get(ontology_snapshot_artifact_id))
        checked = deps.sparql.validate_and_preview(artifact, snapshot)
        ref = deps.artifacts.put("sparql", checked, label="Inspectible SPARQL semantic pattern", creator_agent_id=agent_id,
                                 parent_artifact_ids=[ontology_snapshot_artifact_id])
        return {"artifact_ref": ref.model_dump(mode="json"), "sparql": checked.model_dump(mode="json")}

    return ts


def _edc_tools(deps: HarnessDeps, agent_id: str) -> FunctionToolset:
    ts = FunctionToolset(instructions=(
        "EDC is internally Extract -> Define -> Canonicalize. Extract requires evidence_bundle; "
        "Define requires edc_open_triples; Canonicalize requires exact open triples, definitions, and ontology_snapshot. "
        "Persisted artifacts are the only scientific handoff."
    ))

    @ts.tool_plain
    async def run_edc_extract(evidence_bundle_artifact_id: str, schema_hint_uris: list[str] | None = None) -> dict:
        bundle = load_payload(deps, evidence_bundle_artifact_id, "evidence_bundle", EvidenceBundle)
        hints = []
        for uri in (schema_hint_uris or [])[:20]:
            try:
                hints.append(deps.ontology.inspect(uri))
            except KeyError:
                continue
        execution_id = f"edc_{uuid4().hex[:16]}"
        triples = await deps.edc.extract(bundle, question=deps.registry.get(deps.run_id).question, schema_hints=hints)
        payload = EDCOpenTriples(edc_execution_id=execution_id, evidence_bundle_artifact_id=evidence_bundle_artifact_id, triples=triples)
        ref = deps.artifacts.put("edc_open_triples", payload, creator_agent_id=agent_id, parent_artifact_ids=[evidence_bundle_artifact_id])
        return {"status": "completed", "edc_execution_id": execution_id, "artifact_ref": ref.model_dump(mode="json"), "triple_count": len(triples)}

    @ts.tool_plain
    async def run_edc_define(edc_open_triples_artifact_id: str) -> dict:
        open_payload = load_payload(deps, edc_open_triples_artifact_id, "edc_open_triples", EDCOpenTriples)
        definitions = await deps.edc.define(open_payload.triples)
        payload = EDCRelationDefinitions(
            edc_execution_id=open_payload.edc_execution_id,
            open_triples_artifact_id=edc_open_triples_artifact_id,
            definitions=definitions,
        )
        ref = deps.artifacts.put("edc_relation_definitions", payload, creator_agent_id=agent_id, parent_artifact_ids=[edc_open_triples_artifact_id])
        return {"status": "completed", "edc_execution_id": payload.edc_execution_id, "artifact_ref": ref.model_dump(mode="json"), "definition_count": len(definitions)}

    @ts.tool_plain
    async def run_edc_canonicalize(
        edc_open_triples_artifact_id: str,
        edc_relation_definitions_artifact_id: str,
        ontology_snapshot_artifact_id: str,
    ) -> dict:
        open_payload = load_payload(deps, edc_open_triples_artifact_id, "edc_open_triples", EDCOpenTriples)
        definitions_payload = load_payload(deps, edc_relation_definitions_artifact_id, "edc_relation_definitions", EDCRelationDefinitions)
        snapshot = load_payload(deps, ontology_snapshot_artifact_id, "ontology_snapshot", SubOntologyArtifact)
        if definitions_payload.open_triples_artifact_id != edc_open_triples_artifact_id or definitions_payload.edc_execution_id != open_payload.edc_execution_id:
            raise ScientificStateBlocked("Relation definitions do not belong to this EDC execution.", requirements=("edc_relation_definitions",))
        open_meta = metadata(deps, edc_open_triples_artifact_id)
        if not open_meta.parent_artifact_ids:
            raise ScientificStateBlocked("EDC open triples have no evidence_bundle lineage.", requirements=("evidence_bundle",))
        evidence_bundle_artifact_id = open_meta.parent_artifact_ids[0]
        require_kind(deps, evidence_bundle_artifact_id, "evidence_bundle")
        decisions, candidates, proposals, unresolved = await deps.edc.canonicalize(
            open_payload.triples, definitions_payload.definitions, deps.ontology.relation_catalog(snapshot),
            run_id=deps.run_id, question=deps.registry.get(deps.run_id).question, producing_agent_id=agent_id,
            ontology_snapshot_id=ontology_snapshot_artifact_id,
        )
        proposal_refs = [
            deps.artifacts.put("ontology_proposal", p, label=f"EDC ontology proposal: {p.label}", creator_agent_id=agent_id).model_dump(mode="json")
            for p in proposals
        ]
        payload = EDCCanonicalizationResult(
            edc_execution_id=open_payload.edc_execution_id,
            open_triples_artifact_id=edc_open_triples_artifact_id,
            relation_definitions_artifact_id=edc_relation_definitions_artifact_id,
            evidence_bundle_artifact_id=evidence_bundle_artifact_id,
            ontology_snapshot_artifact_id=ontology_snapshot_artifact_id,
            candidates=candidates, decisions=decisions, unresolved_open_triple_ids=unresolved,
            ontology_proposal_artifact_ids=[x["id"] for x in proposal_refs],
        )
        parents = [edc_open_triples_artifact_id, edc_relation_definitions_artifact_id, evidence_bundle_artifact_id, ontology_snapshot_artifact_id]
        result_ref = deps.artifacts.put("edc_canonicalization", payload, creator_agent_id=agent_id, parent_artifact_ids=parents)
        candidate_ref = None
        if candidates and not unresolved:
            candidate_ref = deps.artifacts.put(
                "candidate_canonical_triples", candidates, creator_agent_id=agent_id,
                parent_artifact_ids=[*parents, result_ref.id],
            )
        return {
            "status": "completed" if candidate_ref else "blocked",
            "edc_execution_id": open_payload.edc_execution_id,
            "result_ref": result_ref.model_dump(mode="json"),
            "candidate_ref": candidate_ref.model_dump(mode="json") if candidate_ref else None,
            "ontology_proposal_refs": proposal_refs,
            "unresolved_open_triple_ids": unresolved,
        }

    # Compatibility wrapper for existing callers; the scientific contract is the phase tools above.
    @ts.tool_plain
    async def run_edc(evidence_bundle_artifact_id: str, ontology_snapshot_artifact_id: str) -> dict:
        extract = await run_edc_extract(evidence_bundle_artifact_id)
        define = await run_edc_define(extract["artifact_ref"]["id"])
        return await run_edc_canonicalize(extract["artifact_ref"]["id"], define["artifact_ref"]["id"], ontology_snapshot_artifact_id)

    return ts


def _verification_tools(deps: HarnessDeps, agent_id: str) -> FunctionToolset:
    ts = FunctionToolset(instructions="Evidence support, ontology compatibility, direction, RDF validity and SHACL conformance are separate verification dimensions. SHACL is not truth.")

    @ts.tool_plain
    async def verify_candidate_canonical_triples(candidate_canonical_triples_artifact_id: str) -> dict:
        lineage = resolve_candidate_lineage(deps, candidate_canonical_triples_artifact_id)
        open_payload = load_payload(deps, lineage.open_triples_id, "edc_open_triples", EDCOpenTriples)
        require_kind(deps, candidate_canonical_triples_artifact_id, "candidate_canonical_triples")
        try:
            triples = [CanonicalTriple.model_validate(x) for x in deps.artifacts.get(candidate_canonical_triples_artifact_id)]
        except Exception as exc:
            raise ArtifactIntegrityError("candidate_canonical_triples payload violates its stored schema") from exc
        bundle = load_payload(deps, lineage.evidence_bundle_id, "evidence_bundle", EvidenceBundle)
        snapshot = load_payload(deps, lineage.ontology_snapshot_id, "ontology_snapshot", SubOntologyArtifact)
        if any(t.ontology_snapshot_id not in {None, lineage.ontology_snapshot_id} for t in triples):
            raise ScientificStateBlocked("Candidate triples reference a different ontology snapshot.", requirements=("ontology_snapshot",))
        checks = await deps.verification.verify(triples, bundle, snapshot, verifier_agent_id=agent_id)
        report = deps.validation.validate(triples, snapshot)
        checks = [x.model_copy(update={
            "candidate_artifact_id": candidate_canonical_triples_artifact_id,
            "evidence_bundle_artifact_id": lineage.evidence_bundle_id,
            "ontology_snapshot_artifact_id": lineage.ontology_snapshot_id,
            "edc_execution_id": open_payload.edc_execution_id,
            "dimensions": x.dimensions.model_copy(update={
                "rdf_structural_validity": "yes" if report.rdf_valid else "no",
                "shacl_conformance": "yes" if report.shacl_conforms else "uncertain",
            }),
        }) for x in checks]
        by_check = {x.triple_id: x for x in checks}
        accepted: list[CanonicalTriple] = []
        rejected = []
        for triple in triples:
            check = by_check.get(triple.id)
            if not check:
                rejected.append({"triple": triple.model_dump(mode="json"), "reason": "No verifier result"})
                continue
            # RDF/SHACL remain recorded separately. In non-strict mode SHACL failure cannot prove a factual triple false.
            status = check.status
            if deps.settings.shacl_strict and not report.shacl_conforms:
                status = VerificationStatus.UNCERTAIN
            updated = triple.model_copy(update={
                "verification_status": status,
                "verifier_artifact_ids": [],
                "verification_record_id": check.id,
                "ontology_snapshot_id": lineage.ontology_snapshot_id,
            })
            if status == VerificationStatus.ACCEPTED:
                accepted.append(updated)
            else:
                rejected.append({"triple": updated.model_dump(mode="json"), "reason": check.explanation})
        checks_ref = deps.artifacts.put("verification", checks, creator_agent_id=agent_id,
                                       parent_artifact_ids=[canonical_triples_artifact_id, lineage.evidence_bundle_id, lineage.ontology_snapshot_id])
        report_ref = deps.artifacts.put("structural_validation", report, creator_agent_id=agent_id,
                                       parent_artifact_ids=[canonical_triples_artifact_id, lineage.ontology_snapshot_id])
        accepted = [x.model_copy(update={"verifier_artifact_ids": [checks_ref.id, report_ref.id]}) for x in accepted]
        accepted_ref = deps.artifacts.put(
            "canonical_triples", accepted, label="Accepted evidence-backed canonicalized triples", creator_agent_id=agent_id,
            parent_artifact_ids=[canonical_triples_artifact_id, checks_ref.id, report_ref.id, lineage.evidence_bundle_id, lineage.ontology_snapshot_id],
        )
        rejected_ref = deps.artifacts.put("rejected_triples", rejected, creator_agent_id=agent_id,
                                         parent_artifact_ids=[canonical_triples_artifact_id, checks_ref.id, report_ref.id, lineage.evidence_bundle_id, lineage.ontology_snapshot_id])
        return {
            "canonical_triples_ref": accepted_ref.model_dump(mode="json"), "rejected_triples_ref": rejected_ref.model_dump(mode="json"),
            "verification_ref": checks_ref.model_dump(mode="json"), "validation_ref": report_ref.model_dump(mode="json"),
            "accepted_count": len(accepted), "rejected_or_uncertain_count": len(rejected),
        }

    @ts.tool_plain
    async def verify_canonical_triples(canonical_triples_artifact_id: str, evidence_bundle_artifact_id: str | None = None, ontology_snapshot_artifact_id: str | None = None) -> dict:
        """Legacy name; supplied evidence/ontology IDs are ignored in favour of candidate lineage."""
        return await verify_candidate_canonical_triples(canonical_triples_artifact_id)

    return ts


def _entity_tools(deps: HarnessDeps, agent_id: str) -> FunctionToolset:
    ts = FunctionToolset()

    @ts.tool_plain
    def normalize_entity(mention: str, aliases: dict[str, str] | None = None, known_entities: list[str] | None = None,
                         external_ids: dict[str, str] | None = None) -> dict:
        decision = deps.entities.normalize(mention, aliases=aliases, known_entities=known_entities, external_ids=external_ids)
        ref = deps.artifacts.put("entity_normalization", decision, creator_agent_id=agent_id)
        return {"artifact_ref": ref.model_dump(mode="json"), "decision": decision.model_dump(mode="json")}

    return ts


def _artifact_tools(deps: HarnessDeps, agent_id: str) -> FunctionToolset:
    ts = FunctionToolset(instructions="Prefer artifact references over copying large transcripts into agent context.")

    @ts.tool_plain
    def list_artifacts(kind: str | None = None) -> list[dict]:
        return deps.artifacts.list(kind)

    @ts.tool_plain
    def inspect_artifact(artifact_id: str) -> Any:
        return deps.artifacts.get(artifact_id)

    return ts


def _check_doc(deps: HarnessDeps, document_id: str) -> None:
    deps.ensure_not_cancelled()
    if document_id not in deps.document_ids:
        raise ValueError(f"Document {document_id} is not in this run")


def _bounded(deps: HarnessDeps, text: str) -> str:
    if len(text) <= deps.settings.max_tool_output_chars:
        return text
    return text[:deps.settings.max_tool_output_chars] + "\n[TOOL OUTPUT TRUNCATED: use narrower read/search]"


_BUILDERS = {
    "documents": _document_tools,
    "ontology": _ontology_tools,
    "sparql": _sparql_tools,
    "edc": _edc_tools,
    "verification": _verification_tools,
    "entity_normalization": _entity_tools,
    "artifacts": _artifact_tools,
}
