from __future__ import annotations

import re
from uuid import uuid4
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from app.agents.model_factory import build_model
from app.config import Settings
from app.models.domain import (
    CandidateTriple,
    CanonicalRelationCandidate,
    CanonicalTriple,
    CanonicalizationDecision,
    EvidenceBundle,
    ModelProfile,
    OntologyElement,
    OntologyProposal,
    RelationDefinition,
)


class ExtractionOutput(BaseModel):
    triples: list[CandidateTriple] = Field(default_factory=list)


class DefinitionOutput(BaseModel):
    definitions: list[RelationDefinition] = Field(default_factory=list)


class CanonicalizationChoice(BaseModel):
    selected_uri: str | None = None
    reason: str


@dataclass
class EDCResult:
    edc_execution_id: str
    evidence_bundle_artifact_id: str | None
    ontology_snapshot_artifact_id: str | None
    open_triples: list[CandidateTriple]
    definitions: list[RelationDefinition]
    canonicalization_decisions: list[CanonicalizationDecision]
    canonical_triples: list[CanonicalTriple]
    ontology_proposals: list[OntologyProposal]
    unresolved_open_triple_ids: list[str]


class EDCService:
    """EDC adapted for evidence-first, non-embedding thesis extraction.

    Reuses the upstream EDC decomposition (Extract -> Define -> Canonicalize) while
    replacing embedding schema retrieval with a deterministic shortlist followed by
    bounded LLM verification. The optional embedding ablation flag is intentionally
    not implemented in the base path.
    """
    def __init__(self, settings: Settings):
        self.settings = settings
        retries = {"tools": settings.llm_output_retries, "output": settings.llm_output_retries}
        self.extract_agent = Agent(
            build_model(settings, ModelProfile.EXTRACTOR),
            output_type=ExtractionOutput,
            retries=retries,
            model_settings={"temperature": settings.llm_temperature, "max_tokens": settings.llm_max_tokens},
            instructions=(
                "EDC Extract stage. Extract only factual subject-predicate-object triples directly supported by the "
                "provided immutable EvidenceSpan records. Preserve the evidence IDs that support each triple. "
                "Do not infer unsupported facts or use ontology relations as evidence. Keep predicates open/natural at this stage."
            ),
        )
        self.define_agent = Agent(
            build_model(settings, ModelProfile.BALANCED),
            output_type=DefinitionOutput,
            retries=retries,
            model_settings={"temperature": settings.llm_temperature, "max_tokens": settings.llm_max_tokens},
            instructions=(
                "EDC Define stage. Define each open predicate in general subject-to-object terms. Preserve directionality. "
                "Definitions describe the relation semantics; they do not add facts."
            ),
        )
        self.verify_agent = Agent(
            build_model(settings, ModelProfile.BALANCED),
            output_type=CanonicalizationChoice,
            retries=retries,
            model_settings={"temperature": settings.llm_temperature, "max_tokens": min(settings.llm_max_tokens, 1400)},
            instructions=(
                "EDC Canonicalize verification stage. Choose a relation ONLY from the supplied deterministic shortlist when "
                "its meaning and subject/object direction match the open predicate in the evidence. Return selected_uri=null "
                "when no candidate is semantically equivalent. Never invent a URI."
            ),
        )

    async def extract(
        self,
        bundle: EvidenceBundle,
        *,
        question: str,
        schema_hints: list[OntologyElement] | None = None,
    ) -> list[CandidateTriple]:
        if not bundle.spans:
            return []
        hints = "\n".join(
            f"- {r.label}: {r.description or ''} (domain={r.domain or '?'}, range={r.range or '?'})"
            for r in (schema_hints or [])[:20]
        ) or "(No schema hints supplied; keep predicates open.)"
        extraction = await self.extract_agent.run(
            f"Research question (scope only): {question}\n\n"
            "Optional DBLP schema hints (orientation only; do not force these predicates):\n"
            f"{hints}\n\nEvidence:\n{self._evidence_prompt(bundle)}"
        )
        allowed_evidence_ids = {s.id for s in bundle.spans}
        out: list[CandidateTriple] = []
        for triple in extraction.output.triples:
            valid_ids = [eid for eid in triple.evidence_ids if eid in allowed_evidence_ids]
            if valid_ids:
                out.append(triple.model_copy(update={"evidence_ids": valid_ids}))
        return out

    async def define(self, open_triples: list[CandidateTriple]) -> list[RelationDefinition]:
        distinct = sorted({t.predicate for t in open_triples})
        if not distinct:
            return []
        defined = await self.define_agent.run(
            "Open predicates:\n" + "\n".join(f"- {x}" for x in distinct)
        )
        by_rel = {self._norm(x.relation): x for x in defined.output.definitions}
        return [
            (by_rel.get(
                self._norm(rel),
                RelationDefinition(relation=rel, definition=f"Relation expressed by '{rel}' between subject and object."),
            )).model_copy(update={
                "open_triple_ids": [t.id for t in open_triples if self._norm(t.predicate) == self._norm(rel)]
            })
            for rel in distinct
        ]

    async def canonicalize(
        self,
        open_triples: list[CandidateTriple],
        definitions: list[RelationDefinition],
        relations: list[OntologyElement],
        *,
        run_id: str,
        question: str,
        producing_agent_id: str = "edc",
        ontology_snapshot_id: str,
    ) -> tuple[list[CanonicalizationDecision], list[CanonicalTriple], list[OntologyProposal], list[str]]:
        def_map = {self._norm(x.relation): x.definition for x in definitions}
        decisions: list[CanonicalizationDecision] = []
        canonical: list[CanonicalTriple] = []
        proposals: list[OntologyProposal] = []
        unresolved: list[str] = []
        for triple in open_triples:
            definition = def_map.get(self._norm(triple.predicate), triple.predicate)
            definition_id = next((x.id for x in definitions if self._norm(x.relation) == self._norm(triple.predicate)), None)
            ranked = self.rank_candidates(triple, definition, relations, top_k=5)
            decision, proposal = await self._decide(triple, definition, ranked, relations, run_id, question, producing_agent_id)
            decisions.append(decision)
            if proposal:
                proposals.append(proposal)
            selected = next((r for r in relations if r.uri == decision.selected_uri), None) if decision.selected_uri else None
            if not selected:
                unresolved.append(triple.id)
                continue
            canonical.append(CanonicalTriple(
                subject=triple.subject, predicate=selected.label, object=triple.object,
                predicate_uri=selected.uri, subject_type=triple.subject_type, object_type=triple.object_type,
                original_open_triple=triple, canonicalization_decision_id=decision.id,
                relation_definition_id=definition_id,
                evidence_ids=triple.evidence_ids, producing_agent_id=producing_agent_id, run_id=run_id,
                ontology_snapshot_id=ontology_snapshot_id,
                canonicalization=("exact" if self._norm(selected.label) == self._norm(triple.predicate) else "llm_verified"),
                confidence_optional=ranked[0].score if ranked else None,
            ))
        return decisions, canonical, proposals, unresolved

    async def run(
        self,
        bundle: EvidenceBundle,
        relations: list[OntologyElement],
        *,
        run_id: str,
        question: str,
        producing_agent_id: str = "edc",
    ) -> EDCResult:
        execution_id = f"edc_{uuid4().hex[:16]}"
        open_triples = await self.extract(bundle, question=question, schema_hints=relations)
        definitions = await self.define(open_triples)
        decisions, canonical, proposals, unresolved = await self.canonicalize(
            open_triples, definitions, relations, run_id=run_id, question=question,
            producing_agent_id=producing_agent_id, ontology_snapshot_id="",
        )
        return EDCResult(execution_id, None, None, open_triples, definitions, decisions, canonical, proposals, unresolved)

    def rank_candidates(
        self,
        triple: CandidateTriple,
        definition: str,
        relations: list[OntologyElement],
        top_k: int = 5,
    ) -> list[CanonicalRelationCandidate]:
        """Non-embedding relation shortlist: lexical/fuzzy/definition/domain-range/hierarchy cues."""
        pnorm = self._norm(triple.predicate)
        dtokens = set(self._tokens(definition))
        ranked: list[CanonicalRelationCandidate] = []
        for rel in relations:
            lnorm = self._norm(rel.label)
            label_ratio = SequenceMatcher(None, pnorm, lnorm).ratio()
            exact = 1.0 if pnorm == lnorm else 0.0
            rel_tokens = set(self._tokens((rel.description or "") + " " + rel.label))
            overlap = len(dtokens & rel_tokens) / max(1, len(dtokens | rel_tokens))
            token_overlap = len(set(self._tokens(triple.predicate)) & set(self._tokens(rel.label))) / max(
                1, len(set(self._tokens(triple.predicate)) | set(self._tokens(rel.label)))
            )
            domain_ok = self._type_compatible(triple.subject_type, rel.domain)
            range_ok = self._type_compatible(triple.object_type, rel.range)
            hierarchy_bonus = 0.03 if rel.parents else 0.0
            score = 0.45 * label_ratio + 0.22 * overlap + 0.18 * token_overlap + 0.15 * exact + hierarchy_bonus
            reasons = [
                f"label_similarity={label_ratio:.3f}", f"definition_token_overlap={overlap:.3f}",
                f"predicate_token_overlap={token_overlap:.3f}",
            ]
            if domain_ok is True:
                score += 0.05; reasons.append("domain_compatible")
            elif domain_ok is False:
                score -= 0.18; reasons.append("domain_incompatible")
            if range_ok is True:
                score += 0.05; reasons.append("range_compatible")
            elif range_ok is False:
                score -= 0.18; reasons.append("range_incompatible")
            score = max(0.0, min(1.0, score))
            if score >= 0.18 or exact:
                ranked.append(CanonicalRelationCandidate(
                    uri=rel.uri, label=rel.label, score=score, reasons=reasons,
                    domain_compatible=domain_ok, range_compatible=range_ok,
                ))
        ranked.sort(key=lambda x: (-x.score, x.label.lower()))
        return ranked[:top_k]

    async def _decide(
        self,
        triple: CandidateTriple,
        definition: str,
        ranked: list[CanonicalRelationCandidate],
        relations: list[OntologyElement],
        run_id: str,
        question: str,
        producing_agent_id: str,
    ) -> tuple[CanonicalizationDecision, OntologyProposal | None]:
        exact = next((x for x in ranked if self._norm(x.label) == self._norm(triple.predicate)), None)
        if exact:
            return CanonicalizationDecision(
                open_triple_id=triple.id, open_predicate=triple.predicate, relation_definition=definition,
                candidates=ranked, selected_uri=exact.uri, selected_label=exact.label,
                rejected_uris=[x.uri for x in ranked if x.uri != exact.uri], decision_reason="Exact normalized label match.",
                evidence_ids=triple.evidence_ids,
            ), None

        selected_uri: str | None = None
        reason = "No deterministic candidate was strong enough to verify."
        shortlist = [x for x in ranked if x.score >= 0.28]
        if shortlist:
            catalog = {r.uri: r for r in relations}
            prompt = [
                f"Open triple: ({triple.subject}, {triple.predicate}, {triple.object})",
                f"Relation definition: {definition}",
                "Candidate relations:",
            ]
            for c in shortlist:
                rel = catalog[c.uri]
                prompt.append(
                    f"- URI={c.uri}\n  label={c.label}\n  definition={rel.description or ''}\n  domain={rel.domain or '?'}\n  range={rel.range or '?'}\n  deterministic_score={c.score:.3f}"
                )
            result = await self.verify_agent.run("\n".join(prompt))
            if result.output.selected_uri in {x.uri for x in shortlist}:
                selected_uri = result.output.selected_uri
                reason = result.output.reason
            else:
                reason = result.output.reason or reason

        selected_label = next((x.label for x in ranked if x.uri == selected_uri), None)
        proposal = None
        proposal_id = None
        if selected_uri is None:
            from app.services.ontology import RUN_OVERLAY_BASE
            slug = re.sub(r"[^A-Za-z0-9_]+", "_", triple.predicate.strip()).strip("_") or "relation"
            proposal = OntologyProposal(
                proposed_by=producing_agent_id,
                question=question,
                type="property",
                uri=f"{RUN_OVERLAY_BASE}{run_id}/ontology#{slug}",
                label=triple.predicate,
                definition=definition,
                rationale="No active DBLP/run-local relation passed deterministic shortlist plus LLM semantic verification.",
                evidence_ids=triple.evidence_ids,
            )
            proposal_id = proposal.proposal_id
        return CanonicalizationDecision(
            open_triple_id=triple.id,
            open_predicate=triple.predicate,
            relation_definition=definition,
            candidates=ranked,
            selected_uri=selected_uri,
            selected_label=selected_label,
            rejected_uris=[x.uri for x in ranked if x.uri != selected_uri],
            decision_reason=reason,
            evidence_ids=triple.evidence_ids,
            ontology_proposal_id=proposal_id,
        ), proposal

    @staticmethod
    def _type_compatible(entity_type: str | None, ontology_type: str | None) -> bool | None:
        if not entity_type or not ontology_type:
            return None
        a = EDCService._norm(entity_type)
        b = EDCService._norm(ontology_type.rsplit("#", 1)[-1].rsplit("/", 1)[-1])
        return a == b or a in b or b in a

    @staticmethod
    def _evidence_prompt(bundle: EvidenceBundle) -> str:
        return "\n\n".join(
            f"[{s.id}] document={s.document_id} page={s.page} section={s.section_id or '-'} block={s.block_id or '-'}\n{s.text}"
            for s in bundle.spans
        )

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    @staticmethod
    def _norm(text: str) -> str:
        return " ".join(EDCService._tokens(text))
