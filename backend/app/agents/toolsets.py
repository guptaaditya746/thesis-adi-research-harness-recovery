from __future__ import annotations

import hashlib
from typing import Any

from pydantic_ai import FunctionToolset

from app.harness.contracts import SUPPORTED_CAPABILITIES
from app.harness.deps import HarnessDeps
from app.models.domain import (
    CanonicalTriple,
    EvidenceBundle,
    EvidenceSpan,
    OntologyProposal,
    OntologyProposalStatus,
    RetrievalCandidateBundle,
    RetrievalCandidatePreview,
    SPARQLArtifact,
    SubOntologyArtifact,
    VerificationStatus,
)


class CorrectableToolInputError(ValueError):
    """A tool-input/provenance mistake the model can correct by rereading state."""


def persist_candidate_evidence_from_bundle(
    deps, agent_id: str, bundle_artifact_id: str, candidate_ids: list[str], sufficiency_summary: str,
) -> dict:
    if not candidate_ids:
        raise CorrectableToolInputError("Select at least one candidate_id from the retrieval candidate bundle.")
    envelope = deps.artifacts.get_envelope(bundle_artifact_id)
    if envelope.artifact_type != "retrieval_candidate_bundle":
        raise CorrectableToolInputError(
            f"Artifact {bundle_artifact_id} is {envelope.artifact_type}, not retrieval_candidate_bundle."
        )
    bundle = RetrievalCandidateBundle.model_validate(envelope.content)
    by_id = {span.id: span for span in bundle.spans}
    unknown = [candidate_id for candidate_id in candidate_ids if candidate_id not in by_id]
    if unknown:
        raise CorrectableToolInputError(
            f"Unknown candidate_id(s): {unknown}. Reread the retrieval candidate bundle and select existing IDs."
        )
    spans = [by_id[candidate_id] for candidate_id in dict.fromkeys(candidate_ids)]
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
    evidence = EvidenceBundle(spans=stable, sufficiency_summary=sufficiency_summary)
    ref = deps.artifacts.put(
        "evidence_bundle", evidence, label="Selected immutable evidence", creator_agent_id=agent_id,
        parent_artifact_ids=[bundle_artifact_id, *[x["id"] for x in refs]],
    )
    return {"artifact_ref": ref.model_dump(mode="json"), "evidence_refs": refs, "bundle": evidence.model_dump(mode="json")}


def build_specialist_toolsets(deps: HarnessDeps, capability_names: list[str], agent_id: str) -> list[FunctionToolset]:
    unknown = sorted(set(capability_names) - SUPPORTED_CAPABILITIES)
    if unknown:
        raise ValueError(f"Unknown specialist capabilities: {', '.join(unknown)}")
    return [builder(deps, agent_id) for name, builder in _BUILDERS.items() if name in capability_names]


def read_retrieval_candidate_bundle(deps, artifact_id: str, offset: int = 0, limit: int = 8, preview_chars: int = 240) -> dict:
    """Read bounded candidate metadata; full source text remains in document tools."""
    envelope = deps.artifacts.get_envelope(artifact_id)
    if envelope.artifact_type != "retrieval_candidate_bundle":
        raise ValueError(f"Artifact {artifact_id} is {envelope.artifact_type}, not retrieval_candidate_bundle")
    bundle = RetrievalCandidateBundle.model_validate(envelope.content)
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
        ids = document_ids or deps.document_ids
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
    ts = FunctionToolset(instructions="DBLP is immutable. Any missing relation/class must be represented as an explicit run-local OntologyProposal; never mint a dblp: URI.")

    @ts.tool_plain
    def search_ontology(terms: list[str], kind: str | None = None, limit: int = 40) -> list[dict]:
        return [x.model_dump(mode="json") for x in deps.ontology.search(terms, min(limit, 80), kind)]

    @ts.tool_plain
    def inspect_ontology_term(uri: str) -> dict:
        return deps.ontology.inspect(uri).model_dump(mode="json")

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
    def accept_ontology_proposal_for_run(proposal_artifact_id: str, decision_reason: str) -> dict:
        raw = deps.artifacts.get(proposal_artifact_id)
        proposal = OntologyProposal.model_validate(raw).model_copy(update={"status": OntologyProposalStatus.ACCEPTED_FOR_RUN})
        ref = deps.artifacts.put(
            "ontology_proposal", proposal, label=f"Accepted for run: {proposal.label}", creator_agent_id=agent_id,
            parent_artifact_ids=[proposal_artifact_id],
        )
        decision = deps.artifacts.put(
            "decision", {"action": "accept_ontology_proposal_for_run", "reason": decision_reason, "proposal_id": proposal.proposal_id},
            creator_agent_id=agent_id, parent_artifact_ids=[ref.id],
        )
        return {"proposal_ref": ref.model_dump(mode="json"), "decision_ref": decision.model_dump(mode="json"), "proposal": proposal.model_dump(mode="json")}

    @ts.tool_plain
    def materialize_ontology_snapshot(selected_uris: list[str], accepted_proposal_artifact_ids: list[str] | None = None) -> dict:
        proposal_ids = accepted_proposal_artifact_ids or []
        proposals = [OntologyProposal.model_validate(deps.artifacts.get(x)) for x in proposal_ids]
        digest = hashlib.sha256(("|".join(sorted(selected_uris)) + "|" + "|".join(sorted(proposal_ids))).encode()).hexdigest()[:12]
        path = deps.artifacts.root / f"ontology_snapshot_{digest}.ttl"
        snapshot = deps.ontology.build_snapshot(selected_uris, proposals, path, run_id=deps.run_id)
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
    ts = FunctionToolset(instructions="EDC is allowed only after evidence has been persisted. Feed it EvidenceSpan bundles, never whole PDFs.")

    @ts.tool_plain
    async def run_edc(evidence_bundle_artifact_id: str, ontology_snapshot_artifact_id: str) -> dict:
        bundle = EvidenceBundle.model_validate(deps.artifacts.get(evidence_bundle_artifact_id))
        snapshot = SubOntologyArtifact.model_validate(deps.artifacts.get(ontology_snapshot_artifact_id))
        result = await deps.edc.run(
            bundle, deps.ontology.relation_catalog(snapshot), run_id=deps.run_id,
            question=deps.registry.get(deps.run_id).question, producing_agent_id=agent_id,
        )
        parent_ids = [evidence_bundle_artifact_id, ontology_snapshot_artifact_id]
        open_ref = deps.artifacts.put("edc_open_triples", result.open_triples, creator_agent_id=agent_id, parent_artifact_ids=parent_ids)
        defs_ref = deps.artifacts.put("edc_relation_definitions", result.definitions, creator_agent_id=agent_id, parent_artifact_ids=[open_ref.id])
        decisions_ref = deps.artifacts.put("edc_canonicalization_decisions", result.canonicalization_decisions, creator_agent_id=agent_id,
                                          parent_artifact_ids=[open_ref.id, defs_ref.id, ontology_snapshot_artifact_id])
        canon_ref = deps.artifacts.put("candidate_canonical_triples", result.canonical_triples, creator_agent_id=agent_id,
                                      parent_artifact_ids=[open_ref.id, decisions_ref.id, evidence_bundle_artifact_id])
        proposal_refs = [
            deps.artifacts.put("ontology_proposal", p, label=f"EDC unknown relation: {p.label}", creator_agent_id=agent_id,
                               parent_artifact_ids=[decisions_ref.id]).model_dump(mode="json")
            for p in result.ontology_proposals
        ]
        return {
            "open_triples_ref": open_ref.model_dump(mode="json"), "definitions_ref": defs_ref.model_dump(mode="json"),
            "decisions_ref": decisions_ref.model_dump(mode="json"), "canonical_triples_ref": canon_ref.model_dump(mode="json"),
            "ontology_proposal_refs": proposal_refs,
        }

    return ts


def _verification_tools(deps: HarnessDeps, agent_id: str) -> FunctionToolset:
    ts = FunctionToolset(instructions="Evidence support, ontology compatibility, direction, RDF validity and SHACL conformance are separate verification dimensions. SHACL is not truth.")

    @ts.tool_plain
    async def verify_canonical_triples(canonical_triples_artifact_id: str, evidence_bundle_artifact_id: str, ontology_snapshot_artifact_id: str) -> dict:
        triples = [CanonicalTriple.model_validate(x) for x in deps.artifacts.get(canonical_triples_artifact_id)]
        bundle = EvidenceBundle.model_validate(deps.artifacts.get(evidence_bundle_artifact_id))
        snapshot = SubOntologyArtifact.model_validate(deps.artifacts.get(ontology_snapshot_artifact_id))
        checks = await deps.verification.verify(triples, bundle, verifier_agent_id=agent_id)
        report = deps.validation.validate(triples, snapshot)
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
                "ontology_snapshot_id": snapshot.id,
            })
            if status == VerificationStatus.ACCEPTED:
                accepted.append(updated)
            else:
                rejected.append({"triple": updated.model_dump(mode="json"), "reason": check.explanation})
        checks_ref = deps.artifacts.put("verification", checks, creator_agent_id=agent_id,
                                       parent_artifact_ids=[canonical_triples_artifact_id, evidence_bundle_artifact_id])
        report_ref = deps.artifacts.put("structural_validation", report, creator_agent_id=agent_id,
                                       parent_artifact_ids=[canonical_triples_artifact_id, ontology_snapshot_artifact_id])
        accepted = [x.model_copy(update={"verifier_artifact_ids": [checks_ref.id, report_ref.id]}) for x in accepted]
        accepted_ref = deps.artifacts.put(
            "canonical_triples", accepted, label="Accepted evidence-backed canonicalized triples", creator_agent_id=agent_id,
            parent_artifact_ids=[canonical_triples_artifact_id, checks_ref.id, report_ref.id],
        )
        rejected_ref = deps.artifacts.put("rejected_triples", rejected, creator_agent_id=agent_id,
                                         parent_artifact_ids=[canonical_triples_artifact_id, checks_ref.id, report_ref.id])
        return {
            "canonical_triples_ref": accepted_ref.model_dump(mode="json"), "rejected_triples_ref": rejected_ref.model_dump(mode="json"),
            "verification_ref": checks_ref.model_dump(mode="json"), "validation_ref": report_ref.model_dump(mode="json"),
            "accepted_count": len(accepted), "rejected_or_uncertain_count": len(rejected),
        }

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
