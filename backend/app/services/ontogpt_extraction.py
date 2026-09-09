from __future__ import annotations

import asyncio
from pathlib import Path

from ontogpt.engines.spires_engine import SPIRESEngine
from ontogpt.io.template_loader import get_template_details

from app.config import Settings
from app.models.domain import CandidateTriple, EvidenceBundle, new_id


class OntoGPTExtractionService:
    """Schema-guided, evidence-local candidate-triple extraction via OntoGPT."""

    _TEMPLATE = Path(__file__).parent.parent / "ontogpt_templates" / "thesis_research.yaml"

    def __init__(self, settings: Settings):
        self.settings = settings

    async def extract(self, bundle: EvidenceBundle, *, question: str) -> list[CandidateTriple]:
        return await asyncio.to_thread(self._extract_sync, bundle, question)

    def _extract_sync(self, bundle: EvidenceBundle, question: str) -> list[CandidateTriple]:
        template = get_template_details(str(self._TEMPLATE))
        engine = SPIRESEngine(
            template_details=template,
            model=self.settings.model_for_profile("extractor"),
            api_base=self.settings.litellm_api_base,
            model_provider="openai",
            temperature=self.settings.llm_temperature,
            max_text_length=self.settings.simple_passage_max_chars,
            system_message=(
                "Extract only statements explicitly supported by the supplied passage and materially relevant to this "
                f"research question: {question}. Do not infer facts, combine separate passages, or invent citations."
            ),
        )
        # OntoGPT's engine creates a LiteLLM client but does not forward its engine api_key field.
        # Supplying it here keeps the existing gateway credential server-side.
        engine.client.api_key = self.settings.litellm_api_key
        triples: list[CandidateTriple] = []
        for span in bundle.spans:
            try:
                result = engine.extract_from_text(span.text)
            except BaseException as exc:
                # OntoGPT's CLI-oriented client can raise SystemExit for fatal provider failures.
                raise RuntimeError(f"OntoGPT extraction failed for evidence {span.id}: {type(exc).__name__}") from exc
            payload = result.extracted_object.model_dump() if result.extracted_object else {}
            for item in payload.get("triples", []) or []:
                subject = str(item.get("subject") or "").strip()
                predicate = str(item.get("predicate") or "").strip()
                object_ = str(item.get("object") or "").strip()
                if subject and predicate and object_:
                    triples.append(CandidateTriple(
                        id=new_id("open"), subject=subject, predicate=predicate, object=object_,
                        subject_type=self._optional(item.get("subject_type")),
                        object_type=self._optional(item.get("object_type")), evidence_ids=[span.id],
                        extraction_note="OntoGPT SPIRES schema-guided extraction",
                    ))
        return triples

    @staticmethod
    def _optional(value: object) -> str | None:
        text = str(value or "").strip()
        return text or None
