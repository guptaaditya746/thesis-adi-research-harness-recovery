from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from app.agents.model_factory import build_model
from app.agents.routing import TaskModelRouter
from app.config import Settings
from app.models.domain import (
    CanonicalTriple, EvidenceBundle, ModelProfile, SubOntologyArtifact, TripleVerification, VerificationDimensions, VerificationStatus,
)


class VerificationJudgement(BaseModel):
    triple_id: str
    evidence_support: str
    ontology_compatibility: str
    direction_correct: str
    negation_modality_ok: str
    entity_normalization_quality: str
    explanation: str


class VerificationOutput(BaseModel):
    judgements: list[VerificationJudgement] = Field(default_factory=list)


class TripleVerificationService:
    def __init__(self, settings: Settings):
        self.settings = settings
        router = TaskModelRouter()
        self.agent = Agent(
            build_model(settings, router.profile_for_service("verification", ModelProfile.CRITIC)),
            output_type=VerificationOutput,
            retries={"tools": settings.llm_output_retries, "output": settings.llm_output_retries},
            model_settings={"temperature": settings.llm_temperature, "max_tokens": settings.llm_max_tokens},
            instructions=(
                "Act as an independent evidence auditor. Judge each supplied canonical triple only against the exact "
                "EvidenceSpan text supplied for that triple. Check relation direction, negation/modality, ontology meaning, "
                "and entity normalization. Do not infer missing facts. Use yes/no/uncertain for support/compatibility/direction/negation. "
                "Use good/weak/uncertain for entity normalization."
            ),
        )

    async def verify(
        self,
        triples: list[CanonicalTriple],
        bundle: EvidenceBundle,
        ontology: SubOntologyArtifact | None = None,
        *,
        verifier_agent_id: str = "independent_triple_verifier",
    ) -> list[TripleVerification]:
        span_by_id = {x.id: x for x in bundle.spans}
        eligible: list[CanonicalTriple] = []
        checks_by_triple_id: dict[str, TripleVerification] = {}
        for triple in triples:
            if not triple.evidence_ids or any(eid not in span_by_id for eid in triple.evidence_ids):
                checks_by_triple_id[triple.id] = self._failed_missing_evidence(triple, verifier_agent_id)
            else:
                eligible.append(triple)
        if not eligible:
            return [checks_by_triple_id[triple.id] for triple in triples]
        relation_by_uri = {x.uri: x for x in (ontology.selected_elements if ontology else [])}
        chunks = []
        for t in eligible:
            evidence = "\n\n".join(
                f"[{eid}] {span_by_id[eid].document_id} page {span_by_id[eid].page}: {span_by_id[eid].text}"
                for eid in t.evidence_ids
            )
            chunks.append(
                f"TRIPLE_ID={t.id}\nTRIPLE=({t.subject}, {t.predicate}, {t.object})\n"
                f"PREDICATE_URI={t.predicate_uri or 'OPEN_RELATION'}\n"
                f"ONTOLOGY_RELATION={relation_by_uri.get(t.predicate_uri).description if t.predicate_uri in relation_by_uri else 'unresolved'}\n"
                f"EVIDENCE:\n{evidence}"
            )
        result = await self.agent.run("\n\n---\n\n".join(chunks))
        by_id = {x.triple_id: x for x in result.output.judgements}
        for triple in eligible:
            j = by_id.get(triple.id)
            if not j:
                checks_by_triple_id[triple.id] = self._uncertain(
                    triple, verifier_agent_id, "Verifier did not return a judgement for this triple."
                )
                continue
            dims = VerificationDimensions(
                evidence_support=self._yn(j.evidence_support),
                ontology_compatibility=self._yn(j.ontology_compatibility),
                direction_correct=self._yn(j.direction_correct),
                rdf_structural_validity="uncertain",
                shacl_conformance="uncertain",
                negation_modality_ok=self._yn(j.negation_modality_ok),
                entity_normalization_quality=self._entity(j.entity_normalization_quality),
            )
            status = self._status(dims)
            checks_by_triple_id[triple.id] = TripleVerification(
                triple_id=triple.id, status=status, dimensions=dims, explanation=j.explanation,
                evidence_ids=triple.evidence_ids, verifier_agent_id=verifier_agent_id,
            )
        # Keep the public service result stable in caller input order. Callers must
        # still use triple_id as the integrity key, rather than positional pairing.
        return [checks_by_triple_id[triple.id] for triple in triples]

    @staticmethod
    def _status(d: VerificationDimensions) -> VerificationStatus:
        hard = [d.evidence_support, d.ontology_compatibility, d.direction_correct, d.negation_modality_ok]
        if "no" in hard:
            return VerificationStatus.REJECTED
        if all(x == "yes" for x in hard) and d.entity_normalization_quality != "weak":
            return VerificationStatus.ACCEPTED
        return VerificationStatus.UNCERTAIN

    @staticmethod
    def _yn(value: str) -> str:
        v = value.strip().lower()
        return v if v in {"yes", "no", "uncertain"} else "uncertain"

    @staticmethod
    def _entity(value: str) -> str:
        v = value.strip().lower()
        return v if v in {"good", "weak", "uncertain"} else "uncertain"

    @staticmethod
    def _failed_missing_evidence(t: CanonicalTriple, verifier: str) -> TripleVerification:
        return TripleVerification(
            triple_id=t.id, status=VerificationStatus.REJECTED,
            dimensions=VerificationDimensions(
                evidence_support="no", ontology_compatibility="uncertain", direction_correct="uncertain",
                rdf_structural_validity="uncertain", shacl_conformance="uncertain", negation_modality_ok="uncertain",
                entity_normalization_quality="uncertain",
            ),
            explanation="Triple has missing or unresolved EvidenceSpan IDs.", evidence_ids=t.evidence_ids,
            verifier_agent_id=verifier,
        )

    @staticmethod
    def _uncertain(t: CanonicalTriple, verifier: str, explanation: str) -> TripleVerification:
        return TripleVerification(
            triple_id=t.id, status=VerificationStatus.UNCERTAIN,
            dimensions=VerificationDimensions(
                evidence_support="uncertain", ontology_compatibility="uncertain", direction_correct="uncertain",
                rdf_structural_validity="uncertain", shacl_conformance="uncertain", negation_modality_ok="uncertain",
                entity_normalization_quality="uncertain",
            ),
            explanation=explanation, evidence_ids=t.evidence_ids, verifier_agent_id=verifier,
        )
