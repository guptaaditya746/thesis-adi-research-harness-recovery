"""Compatibility wrapper around the thesis verification service."""
from __future__ import annotations

from app.models.domain import CanonicalTriple, EvidenceBundle, ProvenanceRecord, VerifiedTriple
from app.services.verification import TripleVerificationService


async def verify_triples_with_agent(settings, triples: list[CanonicalTriple], bundle: EvidenceBundle) -> list[VerifiedTriple]:
    checks = await TripleVerificationService(settings).verify(triples, bundle)
    by_id = {x.id: x for x in bundle.spans}
    output: list[VerifiedTriple] = []
    for triple, check in zip(triples, checks):
        doc_ids = sorted({by_id[e].document_id for e in triple.evidence_ids if e in by_id})
        status = "supported" if check.status == "accepted" else "unsupported" if check.status == "rejected" else "weakly_supported"
        output.append(VerifiedTriple(
            triple=triple, status=status, explanation=check.explanation,
            provenance=ProvenanceRecord(
                triple_id=triple.id, document_id=doc_ids[0] if len(doc_ids) == 1 else None,
                document_ids=doc_ids, evidence_ids=triple.evidence_ids,
                inference_type="direct" if status == "supported" else "uncertain", method="independent_evidence_audit",
            ),
            verification=check,
        ))
    return output
