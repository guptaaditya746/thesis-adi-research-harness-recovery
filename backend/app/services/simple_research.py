from __future__ import annotations

import asyncio
import hashlib
import os
import tempfile
import time
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4

from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.config import Settings
from app.models.domain import EvidenceSpan
from app.services.pdf import PDFService
from app.services.retrieval import RetrievalService


def _now() -> datetime:
    return datetime.now(UTC)


class SimpleResearchRequest(BaseModel):
    question: str
    document_ids: list[str]
    model: str | None = None
    reasoning_effort: Literal["low", "medium", "high"] | None = None


class SimpleCitation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    document_id: str
    page: int
    quote: str


class SimpleEntityType(StrEnum):
    PAPER = "ResearchPaper"
    METHOD = "ResearchMethod"
    TASK = "ResearchTask"
    DATASET = "Dataset"
    METRIC = "Metric"
    RESULT = "Result"
    CLAIM = "Claim"
    LIMITATION = "Limitation"
    ENTITY = "Entity"


class SimplePredicate(StrEnum):
    ADDRESSES_PROBLEM = "addresses_problem"
    PROPOSES_METHOD = "proposes_method"
    USES_DATASET = "uses_dataset"
    EVALUATES_ON = "evaluates_on"
    REPORTS_RESULT = "reports_result"
    IMPROVES = "improves"
    COMPARES_WITH = "compares_with"
    SUPPORTS_CLAIM = "supports_claim"
    HAS_LIMITATION = "has_limitation"
    DESCRIBES = "describes"


SIMPLE_ONTOLOGY_NAMESPACE = "https://example.org/thesis/simple-research#"
SIMPLE_PREDICATE_LABELS: dict[SimplePredicate, str] = {
    SimplePredicate.ADDRESSES_PROBLEM: "addresses problem",
    SimplePredicate.PROPOSES_METHOD: "proposes method",
    SimplePredicate.USES_DATASET: "uses dataset",
    SimplePredicate.EVALUATES_ON: "evaluates on",
    SimplePredicate.REPORTS_RESULT: "reports result",
    SimplePredicate.IMPROVES: "improves",
    SimplePredicate.COMPARES_WITH: "compares with",
    SimplePredicate.SUPPORTS_CLAIM: "supports claim",
    SimplePredicate.HAS_LIMITATION: "has limitation",
    SimplePredicate.DESCRIBES: "describes",
}


class SimpleTriple(BaseModel):
    id: str = Field(default_factory=lambda: f"triple_{uuid4().hex[:16]}")
    subject: str = Field(min_length=1, max_length=300)
    subject_type: SimpleEntityType
    predicate: str
    predicate_key: SimplePredicate
    predicate_uri: str
    object: str = Field(min_length=1, max_length=300)
    object_type: SimpleEntityType
    evidence_ids: list[str]


class SimpleTripleDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject: str = Field(min_length=1, max_length=300)
    subject_type: SimpleEntityType
    predicate: SimplePredicate
    object: str = Field(min_length=1, max_length=300)
    object_type: SimpleEntityType
    supporting_passage_numbers: list[int]


class SimpleSynthesis(BaseModel):
    answer: str
    citations: list[SimpleCitation] = Field(default_factory=list)
    triples: list[SimpleTriple] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class SimpleSynthesisDraft(BaseModel):
    """Strict model output; durable IDs are assigned by the host."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1, max_length=4_000)
    supporting_passage_numbers: list[int]
    triples: list[SimpleTripleDraft] = Field(min_length=1, max_length=12)
    limitations: list[str]


class CompactSimpleSynthesisDraft(SimpleSynthesisDraft):
    """Small structured target used only by the reasoning-model profile."""

    answer: str = Field(min_length=1, max_length=800)
    triples: list[SimpleTripleDraft] = Field(min_length=1, max_length=3)
    limitations: list[str] = Field(default_factory=list, max_length=3)


class SimpleExecutionProfile(BaseModel):
    retrieval_per_document: int
    max_passages: int
    passage_max_chars: int
    requested_max_tokens: int
    reasoning_effort: Literal["low", "medium", "high"] | None = None


class SimpleModelRequestDiagnostic(BaseModel):
    """Compact request facts for diagnosing model failures without storing prompts."""

    stage: Literal["generation", "repair"]
    model: str
    estimated_prompt_tokens: int
    requested_max_tokens: int
    context_window_tokens: int | None = None
    timeout_seconds: float
    elapsed_ms: float | None = None
    outcome: Literal["pending", "completed", "timed_out", "failed"] = "pending"
    error_type: str | None = None
    finish_reason: str | None = None
    provider_prompt_tokens: int | None = None
    provider_completion_tokens: int | None = None
    visible_output_chars: int | None = None
    reasoning_content_chars: int | None = None


class SimpleResearchRun(BaseModel):
    id: str = Field(default_factory=lambda: f"simple_{uuid4().hex[:16]}")
    question: str
    document_ids: list[str]
    model: str
    model_mode: Literal["fast", "default", "reasoning", "custom"] = "custom"
    reasoning_effort: Literal["low", "medium", "high"] | None = None
    status: Literal["queued", "running", "completed", "failed", "cancelled"] = "queued"
    stage: Literal["queued", "retrieving", "generating", "validating", "repairing", "completed", "failed", "cancelled"] = "queued"
    evidence: list[EvidenceSpan] = Field(default_factory=list)
    result: SimpleSynthesis | None = None
    timings_ms: dict[str, float] = Field(default_factory=dict)
    execution_profile: SimpleExecutionProfile | None = None
    model_request_diagnostics: list[SimpleModelRequestDiagnostic] = Field(default_factory=list)
    model_calls: int = 0
    error: str | None = None
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


class SimpleRunStore:
    def __init__(self, settings: Settings):
        self.root = settings.data_dir / "simple_runs"
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, run: SimpleResearchRun) -> None:
        run.updated_at = _now()
        target = self.root / f"{run.id}.json"
        fd, temporary = tempfile.mkstemp(prefix=f".{run.id}.", dir=self.root)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(run.model_dump_json(indent=2))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def get(self, run_id: str) -> SimpleResearchRun:
        path = self.root / f"{run_id}.json"
        if not path.exists():
            raise KeyError(run_id)
        return SimpleResearchRun.model_validate_json(path.read_text(encoding="utf-8"))

    def list(self) -> list[SimpleResearchRun]:
        runs: list[SimpleResearchRun] = []
        for path in self.root.glob("simple_*.json"):
            try:
                runs.append(SimpleResearchRun.model_validate_json(path.read_text(encoding="utf-8")))
            except (OSError, ValidationError):
                continue
        return sorted(runs, key=lambda item: item.created_at, reverse=True)

    def mark_interrupted_failed(self) -> None:
        for run in self.list():
            if run.status in {"queued", "running"}:
                run.status = "failed"
                run.stage = "failed"
                run.error = "Server restarted before this simple research run completed."
                self.save(run)


class SimpleResearchService:
    """Two-call evidence-backed baseline using the existing LiteLLM gateway."""

    def __init__(self, settings: Settings, pdf: PDFService):
        self.settings = settings
        self.pdf = pdf
        self.retrieval = RetrievalService(settings)
        self.store = SimpleRunStore(settings)
        self.client = AsyncOpenAI(base_url=settings.litellm_api_base, api_key=settings.litellm_api_key or "not-required")
        self.cancelled: set[str] = set()

    def create(self, request: SimpleResearchRequest) -> SimpleResearchRun:
        question = request.question.strip()
        document_ids = list(dict.fromkeys(request.document_ids))
        if not question:
            raise ValueError("Research question may not be blank.")
        if not document_ids:
            raise ValueError("Select at least one document.")
        for document_id in document_ids:
            self.pdf.load(document_id)
        requested_model = request.model or "fast"
        model = {
            "fast": self.settings.litellm_fast_model,
            "default": self.settings.litellm_model,
            "reasoning": self.settings.model_reasoning or self.settings.litellm_model,
        }.get(requested_model, requested_model)
        allowed = {
            self.settings.litellm_fast_model,
            self.settings.litellm_model,
            *(value for value in (
                self.settings.model_balanced,
                self.settings.model_reasoning,
                self.settings.model_critic,
                self.settings.model_extractor,
            ) if value),
        }
        if model not in allowed:
            raise ValueError("Model must be one of the configured LiteLLM aliases.")
        model_mode: Literal["fast", "default", "reasoning", "custom"] = (
            requested_model if requested_model in {"fast", "default", "reasoning"} else "custom"
        )
        reasoning_effort = request.reasoning_effort if model_mode == "reasoning" else None
        profile = self._execution_profile_for(model_mode, reasoning_effort)
        run = SimpleResearchRun(
            question=question,
            document_ids=document_ids,
            model=model,
            model_mode=model_mode,
            reasoning_effort=reasoning_effort,
            execution_profile=profile,
        )
        self.store.save(run)
        return run

    async def execute(self, run_id: str) -> None:
        run = self.store.get(run_id)
        total_started = time.perf_counter()
        try:
            self._check_cancelled(run)
            run.status = "running"
            run.stage = "retrieving"
            self.store.save(run)
            started = time.perf_counter()
            run.evidence = await asyncio.to_thread(self._retrieve, run)
            run.timings_ms["retrieval"] = self._elapsed(started)
            if not run.evidence:
                raise ValueError("No relevant passages were retrieved from the selected documents.")

            self._check_cancelled(run)
            run.stage = "generating"
            self.store.save(run)
            started = time.perf_counter()
            run.model_calls += 1
            try:
                raw = await self._synthesize(run)
            finally:
                run.timings_ms["generation"] = self._elapsed(started)

            self._check_cancelled(run)
            run.stage = "validating"
            self.store.save(run)
            started = time.perf_counter()
            result, errors = self._parse_and_validate(
                raw, run.evidence, self._draft_model_for(run.model_mode),
            )
            run.timings_ms["validation"] = self._elapsed(started)

            if errors:
                run.stage = "repairing"
                self.store.save(run)
                started = time.perf_counter()
                run.model_calls += 1
                try:
                    raw = await self._repair(run, raw, errors)
                finally:
                    run.timings_ms["repair"] = self._elapsed(started)
                result, errors = self._parse_and_validate(
                    raw, run.evidence, self._draft_model_for(run.model_mode),
                )
            if errors or result is None:
                raise ValueError("Result validation failed after one repair: " + "; ".join(errors[:6]))

            run.result = result
            run.status = "completed"
            run.stage = "completed"
        except asyncio.CancelledError:
            run.status = "cancelled"
            run.stage = "cancelled"
            run.error = "Cancelled by user."
        except Exception as exc:  # noqa: BLE001 - this boundary persists a bounded run failure
            run.status = "failed"
            run.stage = "failed"
            run.error = self._bounded_error(exc)
        finally:
            run.timings_ms["total"] = self._elapsed(total_started)
            self.store.save(run)

    def cancel(self, run_id: str) -> SimpleResearchRun:
        run = self.store.get(run_id)
        if run.status not in {"completed", "failed", "cancelled"}:
            self.cancelled.add(run_id)
            run.status = "cancelled"
            run.stage = "cancelled"
            run.error = "Cancellation requested."
            self.store.save(run)
        return run

    def triple_map(self, run_id: str) -> dict[str, list[dict]]:
        run = self.store.get(run_id)
        triples = run.result.triples if run.result else []
        nodes: dict[str, dict] = {}
        edges: list[dict] = []
        for index, triple in enumerate(triples):
            subject_id = self._node_id(triple.subject)
            object_id = self._node_id(triple.object)
            nodes.setdefault(subject_id, {"id": subject_id, "label": triple.subject, "kind": "entity"})
            nodes.setdefault(object_id, {"id": object_id, "label": triple.object, "kind": "entity"})
            edges.append({
                "id": triple.id,
                "source": subject_id,
                "target": object_id,
                "label": triple.predicate,
                "predicate": triple.predicate,
                "status": "source_validated",
                "index": index,
            })
        return {"nodes": list(nodes.values()), "edges": edges}

    def _retrieve(self, run: SimpleResearchRun) -> list[EvidenceSpan]:
        spans: list[EvidenceSpan] = []
        profile = run.execution_profile or self._execution_profile_for(run.model_mode, run.reasoning_effort)
        per_document = max(1, min(profile.retrieval_per_document, 12))
        for document_id in run.document_ids:
            spans.extend(self.retrieval.search(
                self.pdf.load(document_id), run.question, per_document, agent_id="simple_research",
            ))
        unique: dict[tuple[str, int, str | None], EvidenceSpan] = {}
        for span in sorted(spans, key=lambda item: -(item.score or 0.0)):
            key = (span.document_id, span.page, span.block_id)
            if key not in unique:
                valid, _ = self.retrieval.validate_span(self.pdf.load(span.document_id), span)
                if not valid:
                    continue
                bounded_text = span.text[: profile.passage_max_chars]
                unique[key] = span.model_copy(update={
                    "text": bounded_text,
                    "source_text_sha256": hashlib.sha256(bounded_text.encode()).hexdigest(),
                    "start_char": None,
                    "end_char": None,
                    "retrieval_method": span.retrieval_method + "+bounded",
                })
        return list(unique.values())[: profile.max_passages]

    async def _synthesize(self, run: SimpleResearchRun) -> str:
        evidence = self._numbered_passages(run.evidence)
        compact_instruction = (
            " Return a direct answer and exactly one to three materially useful triples. "
            "Do not attempt exhaustive extraction or explain your reasoning."
            if run.model_mode == "reasoning" else ""
        )
        messages = [
            {"role": "system", "content": (
                "Answer only from supplied passages. Return JSON matching the supplied schema. "
                "Select only passage numbers you received; never output evidence IDs, document IDs, page numbers, or quotations. "
                "Every triple needs at least one supporting passage number. Keep the answer concise."
                + compact_instruction
            )},
            {"role": "user", "content": f"Question: {run.question}\n\nPassages:\n{evidence}"},
        ]
        return await self._completion(
            run.model, run.model_mode, run.reasoning_effort, messages,
            diagnostics=run.model_request_diagnostics, stage="generation",
        )

    async def _repair(self, run: SimpleResearchRun, raw: str, errors: list[str]) -> str:
        evidence = self._numbered_passages(run.evidence)
        messages = [
            {"role": "system", "content": (
                "Repair the JSON using only supplied numbered passages. Return only schema-valid JSON. "
                "Do not output evidence IDs, document IDs, page numbers, or quotations."
                + (
                    " Keep the direct answer and exactly one to three materially useful triples."
                    if run.model_mode == "reasoning" else ""
                )
            )},
            {"role": "user", "content": (
                f"Question: {run.question}\nValidation errors:\n- " + "\n- ".join(errors[:8])
                + f"\n\nPrevious output:\n{raw[:6000]}\n\nPassages:\n{evidence}"
            )},
        ]
        return await self._completion(
            run.model, run.model_mode, run.reasoning_effort, messages,
            diagnostics=run.model_request_diagnostics, stage="repair",
        )

    async def _completion(
        self,
        model: str,
        model_mode: Literal["fast", "default", "reasoning", "custom"],
        reasoning_effort: Literal["low", "medium", "high"] | None,
        messages: list[dict[str, str]],
        *,
        diagnostics: list[SimpleModelRequestDiagnostic] | None = None,
        stage: Literal["generation", "repair"] = "generation",
    ) -> str:
        max_tokens = self._max_tokens_for(model_mode, messages)
        request: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "simple_research_synthesis",
                    "strict": True,
                    "schema": self._draft_model_for(model_mode).model_json_schema(),
                },
            },
        }
        if model_mode == "reasoning":
            request["reasoning_effort"] = reasoning_effort or self.settings.simple_reasoning_effort
            # LiteLLM validates OpenAI-compatible parameters before forwarding to vLLM.
            # Its proxy accepts this request-scoped allowlist for provider extensions.
            request["extra_body"] = {"allowed_openai_params": ["reasoning_effort"]}
        diagnostic = SimpleModelRequestDiagnostic(
            stage=stage,
            model=model,
            estimated_prompt_tokens=self._estimate_prompt_tokens(messages),
            requested_max_tokens=max_tokens,
            context_window_tokens=self.settings.llm_context_window,
            timeout_seconds=self.settings.simple_request_timeout_seconds,
        )
        if diagnostics is not None:
            diagnostics.append(diagnostic)
        started = time.perf_counter()
        try:
            response = await asyncio.wait_for(
                self.client.chat.completions.create(**request),
                timeout=self.settings.simple_request_timeout_seconds,
            )
        except TimeoutError as exc:
            diagnostic.outcome = "timed_out"
            diagnostic.elapsed_ms = self._elapsed(started)
            diagnostic.error_type = "TimeoutError"
            raise TimeoutError(
                "Model request timed out after "
                f"{self.settings.simple_request_timeout_seconds:g} seconds before receiving a response "
                f"(model={model}; estimated_prompt_tokens={diagnostic.estimated_prompt_tokens}; "
                f"requested_max_tokens={max_tokens}; context_window_tokens="
                f"{self.settings.llm_context_window or 'unset'})."
            ) from exc
        except Exception as exc:
            diagnostic.outcome = "failed"
            diagnostic.elapsed_ms = self._elapsed(started)
            diagnostic.error_type = type(exc).__name__
            raise
        choice = response.choices[0]
        content = self._message_content(choice.message)
        reasoning_content = self._reasoning_content(choice.message)
        diagnostic.elapsed_ms = self._elapsed(started)
        diagnostic.finish_reason = choice.finish_reason or None
        diagnostic.visible_output_chars = len(content)
        diagnostic.reasoning_content_chars = len(reasoning_content)
        usage = getattr(response, "usage", None)
        diagnostic.provider_prompt_tokens = self._usage_value(usage, "prompt_tokens")
        diagnostic.provider_completion_tokens = self._usage_value(usage, "completion_tokens")
        if not content:
            diagnostic.outcome = "failed"
            diagnostic.error_type = "MissingFinalJson"
            finish_reason = diagnostic.finish_reason or "unknown"
            raise ValueError(
                "Model returned no final JSON "
                f"(finish_reason={finish_reason}; reasoning_content_chars={len(reasoning_content)}). "
                "The reasoning model likely exhausted its completion budget before emitting the final JSON."
            )
        diagnostic.outcome = "completed"
        return content

    @staticmethod
    def _parse_and_validate(
        raw: str,
        evidence: list[EvidenceSpan],
        draft_model: type[SimpleSynthesisDraft] = SimpleSynthesisDraft,
    ) -> tuple[SimpleSynthesis | None, list[str]]:
        try:
            draft = draft_model.model_validate_json(raw)
        except ValidationError as exc:
            return None, [f"Invalid JSON output: {SimpleResearchService._bounded_error(exc)}"]

        errors: list[str] = []
        answer_spans = SimpleResearchService._resolve_passage_numbers(
            draft.supporting_passage_numbers, evidence, "answer", errors,
        )
        triples: list[SimpleTriple] = []
        all_citation_spans = list(answer_spans)
        for index, triple in enumerate(draft.triples, start=1):
            triple_spans = SimpleResearchService._resolve_passage_numbers(
                triple.supporting_passage_numbers, evidence, f"triple {index}", errors,
            )
            all_citation_spans.extend(triple_spans)
            triples.append(SimpleTriple(
                subject=triple.subject,
                subject_type=triple.subject_type,
                predicate=SIMPLE_PREDICATE_LABELS[triple.predicate],
                predicate_key=triple.predicate,
                predicate_uri=SIMPLE_ONTOLOGY_NAMESPACE + triple.predicate.value,
                object=triple.object,
                object_type=triple.object_type,
                evidence_ids=[span.id for span in triple_spans],
            ))
        if errors:
            return None, errors

        unique_citations: dict[str, EvidenceSpan] = {}
        for span in all_citation_spans:
            unique_citations.setdefault(span.id, span)
        result = SimpleSynthesis(
            answer=draft.answer,
            citations=[SimpleResearchService._host_citation(span) for span in unique_citations.values()],
            triples=triples,
            limitations=draft.limitations,
        )
        return result, errors

    @staticmethod
    def _numbered_passages(evidence: list[EvidenceSpan]) -> str:
        return "\n\n".join(f"[Passage {index}]\n{span.text}" for index, span in enumerate(evidence, start=1))

    @staticmethod
    def _resolve_passage_numbers(
        numbers: list[int], evidence: list[EvidenceSpan], scope: str, errors: list[str],
    ) -> list[EvidenceSpan]:
        if not numbers:
            errors.append(f"{scope.capitalize()} must select at least one passage number.")
            return []
        spans: list[EvidenceSpan] = []
        for number in dict.fromkeys(numbers):
            if number < 1 or number > len(evidence):
                errors.append(f"{scope.capitalize()} selected unavailable passage number {number}.")
                continue
            spans.append(evidence[number - 1])
        return spans

    @staticmethod
    def _host_citation(span: EvidenceSpan) -> SimpleCitation:
        return SimpleCitation(
            evidence_id=span.id,
            document_id=span.document_id,
            page=span.page,
            quote=SimpleResearchService._host_excerpt(span.text),
        )

    @staticmethod
    def _host_excerpt(text: str, limit: int = 750) -> str:
        source = text.strip()
        if len(source) <= limit:
            return source
        sentence_end = max(source.rfind(". ", 180, limit), source.rfind("? ", 180, limit), source.rfind("! ", 180, limit))
        return source[: sentence_end + 1] if sentence_end >= 180 else source[:limit]

    def _execution_profile_for(
        self,
        model_mode: Literal["fast", "default", "reasoning", "custom"],
        requested_reasoning_effort: Literal["low", "medium", "high"] | None,
    ) -> SimpleExecutionProfile:
        if model_mode == "reasoning":
            return SimpleExecutionProfile(
                retrieval_per_document=max(1, min(self.settings.simple_reasoning_retrieval_per_document, 12)),
                max_passages=max(1, self.settings.simple_reasoning_max_passages),
                passage_max_chars=max(200, self.settings.simple_reasoning_passage_max_chars),
                requested_max_tokens=self.settings.simple_reasoning_llm_max_tokens,
                reasoning_effort=requested_reasoning_effort or self.settings.simple_reasoning_effort,
            )
        return SimpleExecutionProfile(
            retrieval_per_document=max(1, min(self.settings.simple_retrieval_per_document, 12)),
            max_passages=max(1, self.settings.simple_max_passages),
            passage_max_chars=max(200, self.settings.simple_passage_max_chars),
            requested_max_tokens=self._max_tokens_for(model_mode),
        )

    @staticmethod
    def _draft_model_for(
        model_mode: Literal["fast", "default", "reasoning", "custom"],
    ) -> type[SimpleSynthesisDraft]:
        return CompactSimpleSynthesisDraft if model_mode == "reasoning" else SimpleSynthesisDraft

    def _max_tokens_for(
        self, model_mode: Literal["fast", "default", "reasoning", "custom"], messages: list[dict[str, str]] | None = None,
    ) -> int:
        requested = {
            "fast": self.settings.simple_llm_max_tokens,
            "default": self.settings.simple_full_llm_max_tokens,
            "reasoning": self.settings.simple_reasoning_llm_max_tokens,
            "custom": self.settings.simple_llm_max_tokens,
        }[model_mode]
        context_window = self.settings.llm_context_window
        if not context_window:
            return requested
        prompt_tokens = self._estimate_prompt_tokens(messages or [])
        # Conservative OpenAI-chat estimate: prompt text, schema/provider overhead, then a final-response margin.
        available = context_window - prompt_tokens - 1_024
        if available < 256:
            raise ValueError(
                "Simple research request leaves too little completion headroom for the configured context window. "
                "Reduce retrieved passages or increase LLM_CONTEXT_WINDOW."
            )
        return min(requested, available)

    @staticmethod
    def _estimate_prompt_tokens(messages: list[dict[str, str]]) -> int:
        # Conservative, dependency-free estimate; includes small chat framing overhead.
        return sum((len(message.get("content", "")) + 3) // 4 + 4 for message in messages)

    @staticmethod
    def _usage_value(usage: Any, name: str) -> int | None:
        value = getattr(usage, name, None)
        return value if isinstance(value, int) else None

    @staticmethod
    def _message_content(message: Any) -> str:
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            return "".join(
                part.get("text", "") if isinstance(part, dict) else getattr(part, "text", "")
                for part in content
            ).strip()
        return ""

    @staticmethod
    def _reasoning_content(message: Any) -> str:
        direct = getattr(message, "reasoning_content", None)
        if isinstance(direct, str):
            return direct
        extra = getattr(message, "model_extra", None)
        value = extra.get("reasoning_content", "") if isinstance(extra, dict) else ""
        return value if isinstance(value, str) else ""

    def _check_cancelled(self, run: SimpleResearchRun) -> None:
        if run.id in self.cancelled:
            raise asyncio.CancelledError

    @staticmethod
    def _elapsed(started: float) -> float:
        return round((time.perf_counter() - started) * 1000, 2)

    @staticmethod
    def _bounded_error(error: object, limit: int = 600) -> str:
        text = str(error).replace("\x00", " ").strip()
        if not text:
            text = f"{type(error).__name__}: no error message was provided."
        return text if len(text) <= limit else text[: limit - 1] + "…"

    @staticmethod
    def _node_id(label: str) -> str:
        return "node_" + hashlib.sha256(label.strip().casefold().encode("utf-8")).hexdigest()[:16]
