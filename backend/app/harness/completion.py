"""Strict scientific completion validation behind one Director-facing operation."""

from __future__ import annotations

from app.harness.contracts import ArtifactNotFound, InvalidArtifactKind, ScientificStateBlocked, resolve_candidate_lineage
from app.models.domain import CanonicalTriple, CompletionAttempt, EvidenceSpan, ScientificStateSnapshot


class ScientificCompletionService:
    """Validate a claimed canonical-triples artifact without prescribing a next action."""

    @staticmethod
    def scientific_state(artifacts) -> ScientificStateSnapshot:
        return ScientificStateSnapshot(
            evidence_bundle=bool(artifacts.list("evidence_bundle")),
            edc_open_triples=bool(artifacts.list("edc_open_triples")),
            edc_relation_definitions=bool(artifacts.list("edc_relation_definitions")),
            ontology_snapshot=bool(artifacts.list("ontology_snapshot")),
            candidate_canonical_triples=bool(artifacts.list("candidate_canonical_triples")),
            verification_report=bool(artifacts.list("verification")),
            canonical_triples=bool(artifacts.list("canonical_triples")),
        )

    def attempt(self, deps, canonical_triples_artifact_id: str) -> CompletionAttempt:
        state = self.scientific_state(deps.artifacts)
        try:
            final_meta = deps.artifacts.get_metadata(canonical_triples_artifact_id)
            if final_meta.artifact_type != "canonical_triples":
                return self._reject(state)
            parent_meta = {deps.artifacts.get_metadata(parent).artifact_type: parent for parent in final_meta.parent_artifact_ids}
            candidate_id = parent_meta.get("candidate_canonical_triples")
            verification_id = parent_meta.get("verification")
            if not candidate_id or not verification_id:
                return self._reject(state)
            lineage = resolve_candidate_lineage(deps, candidate_id)
            required_parents = {candidate_id, verification_id, lineage.evidence_bundle_id, lineage.ontology_snapshot_id}
            if not required_parents.issubset(set(final_meta.parent_artifact_ids)):
                return self._reject(state)
            triples = [CanonicalTriple.model_validate(item) for item in deps.artifacts.get(canonical_triples_artifact_id)]
            if not triples:
                return self._reject(state)
            for triple in triples:
                if not triple.evidence_ids or not triple.predicate_uri or triple.canonicalization == "open_relation":
                    return self._reject(state)
                if triple.verification_status != "accepted" or verification_id not in triple.verifier_artifact_ids:
                    return self._reject(state)
                if triple.ontology_snapshot_id != lineage.ontology_snapshot_id:
                    return self._reject(state)
                for evidence_id in triple.evidence_ids:
                    evidence_meta = deps.artifacts.get_metadata(evidence_id)
                    if evidence_meta.artifact_type != "evidence_span":
                        return self._reject(state)
                    span = EvidenceSpan.model_validate(deps.artifacts.get(evidence_id))
                    valid, _reason = deps.retrieval.validate_span(deps.pdf.load(span.document_id), span)
                    if not valid:
                        return self._reject(state)
        except (KeyError, ArtifactNotFound, InvalidArtifactKind, ScientificStateBlocked):
            return self._reject(state)
        except Exception:
            return CompletionAttempt(
                accepted=False,
                reason="scientific_completion_integrity_failure",
                scientific_state=state,
            )
        return CompletionAttempt(
            accepted=True,
            reason="scientific_completion_accepted",
            canonical_triples_artifact_id=canonical_triples_artifact_id,
            scientific_state=state,
        )

    @staticmethod
    def _reject(state: ScientificStateSnapshot) -> CompletionAttempt:
        return CompletionAttempt(
            accepted=False,
            reason="scientific_completion_requirements_not_met",
            scientific_state=state,
        )
