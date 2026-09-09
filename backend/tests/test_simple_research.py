from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.models.domain import EvidenceSpan
from app.services.simple_research import (
    CompactSimpleSynthesisDraft,
    SimpleResearchRequest,
    SimpleResearchRun,
    SimpleResearchService,
    SimpleRunStore,
)


def span() -> EvidenceSpan:
    return EvidenceSpan(
        id="ev_source",
        document_id="doc_paper",
        page=4,
        text="The method reduces retrieval latency while preserving source attribution.",
        retrieval_method="test",
    )


def output(*, passage_numbers: list[int]) -> str:
    return json.dumps({
        "answer": "The method reduces retrieval latency.",
        "supporting_passage_numbers": passage_numbers,
        "triples": [{
            "subject": "method",
            "subject_type": "ResearchMethod",
            "predicate": "improves",
            "object": "retrieval latency",
            "object_type": "Metric",
            "supporting_passage_numbers": passage_numbers,
        }],
        "limitations": [],
    })


def test_host_maps_model_passage_numbers_to_exact_source_provenance() -> None:
    result, errors = SimpleResearchService._parse_and_validate(
        output(passage_numbers=[1]), [span()],
    )
    assert errors == []
    assert result is not None
    assert result.triples[0].id.startswith("triple_")
    assert result.triples[0].evidence_ids == ["ev_source"]
    assert result.triples[0].predicate_uri == "https://example.org/thesis/simple-research#improves"
    assert result.triples[0].subject_type == "ResearchMethod"
    assert result.triples[0].object_type == "Metric"
    assert result.citations[0].evidence_id == "ev_source"
    assert result.citations[0].document_id == "doc_paper"
    assert result.citations[0].page == 4
    assert result.citations[0].quote in span().text

    _, errors = SimpleResearchService._parse_and_validate(
        output(passage_numbers=[2]), [span()],
    )
    assert errors == [
        "Answer selected unavailable passage number 2.",
        "Triple 1 selected unavailable passage number 2.",
    ]


def test_simple_store_round_trips_and_marks_interrupted_runs(settings) -> None:
    store = SimpleRunStore(settings)
    run = SimpleResearchRun(question="Why?", document_ids=["doc"], model="fast", status="running")
    store.save(run)

    assert store.get(run.id).question == "Why?"
    store.mark_interrupted_failed()
    restored = store.get(run.id)
    assert restored.status == "failed"
    assert "restarted" in (restored.error or "").lower()


def test_symbolic_model_choice_resolves_to_configured_alias(settings) -> None:
    class ExistingDocuments:
        @staticmethod
        def load(document_id: str) -> object:
            if document_id != "doc":
                raise KeyError(document_id)
            return object()

    service = SimpleResearchService(settings, ExistingDocuments())
    fast = service.create(SimpleResearchRequest(question="Why?", document_ids=["doc"], model="fast"))
    full = service.create(SimpleResearchRequest(question="Why?", document_ids=["doc"], model="default"))
    reasoning = service.create(SimpleResearchRequest(
        question="Why?", document_ids=["doc"], model="reasoning", reasoning_effort="high",
    ))
    assert fast.model == settings.litellm_fast_model
    assert full.model == settings.litellm_model
    assert reasoning.model == (settings.model_reasoning or settings.litellm_model)
    assert reasoning.model_mode == "reasoning"
    assert reasoning.reasoning_effort == "high"
    assert reasoning.execution_profile is not None
    assert reasoning.execution_profile.max_passages == settings.simple_reasoning_max_passages
    assert reasoning.execution_profile.passage_max_chars == settings.simple_reasoning_passage_max_chars
    assert reasoning.execution_profile.reasoning_effort == "high"


def test_reasoning_profile_is_compact_while_fast_and_full_remain_unchanged(settings) -> None:
    service = SimpleResearchService.__new__(SimpleResearchService)
    service.settings = settings

    fast = service._execution_profile_for("fast", None)
    full = service._execution_profile_for("default", None)
    reasoning = service._execution_profile_for("reasoning", None)

    assert fast.max_passages == settings.simple_max_passages
    assert full.passage_max_chars == settings.simple_passage_max_chars
    assert reasoning.max_passages == settings.simple_reasoning_max_passages
    assert reasoning.passage_max_chars == settings.simple_reasoning_passage_max_chars
    assert reasoning.requested_max_tokens == settings.simple_reasoning_llm_max_tokens
    assert reasoning.reasoning_effort == settings.simple_reasoning_effort


@pytest.mark.asyncio
async def test_execute_repairs_at_most_once_and_persists_result(settings) -> None:
    service = SimpleResearchService.__new__(SimpleResearchService)
    service.settings = settings
    service.store = SimpleRunStore(settings)
    service.cancelled = set()
    service._retrieve = lambda run: [span()]

    async def synthesize(run):
        return output(passage_numbers=[99])

    async def repair(run, raw, errors):
        assert errors
        return output(passage_numbers=[1])

    service._synthesize = synthesize
    service._repair = repair
    run = SimpleResearchRun(question="What improves?", document_ids=["doc_paper"], model="fast")
    service.store.save(run)

    await service.execute(run.id)

    completed = service.store.get(run.id)
    assert completed.status == "completed"
    assert completed.model_calls == 2
    assert completed.result is not None


def test_strict_output_schema_requires_every_model_field() -> None:
    from app.services.simple_research import SimpleSynthesisDraft

    schema = SimpleSynthesisDraft.model_json_schema()
    assert set(schema["required"]) == {"answer", "supporting_passage_numbers", "triples", "limitations"}
    assert schema["additionalProperties"] is False


def test_reasoning_schema_rejects_more_than_three_triples() -> None:
    draft = json.loads(output(passage_numbers=[1]))
    draft["triples"] *= 4

    with pytest.raises(ValidationError):
        CompactSimpleSynthesisDraft.model_validate(draft)


def test_reasoning_mode_uses_its_own_completion_budget(settings) -> None:
    service = SimpleResearchService.__new__(SimpleResearchService)
    service.settings = settings
    assert service._max_tokens_for("fast") == settings.simple_llm_max_tokens
    assert service._max_tokens_for("default") == settings.simple_full_llm_max_tokens
    assert service._max_tokens_for("reasoning") == settings.simple_reasoning_llm_max_tokens


def test_reasoning_budget_leaves_headroom_in_declared_context_window(settings) -> None:
    service = SimpleResearchService.__new__(SimpleResearchService)
    service.settings = settings.model_copy(update={"llm_context_window": 8_192})
    budget = service._max_tokens_for("reasoning", [{"role": "user", "content": "x" * 8_000}])
    assert 256 <= budget < service.settings.simple_reasoning_llm_max_tokens


@pytest.mark.asyncio
async def test_reasoning_request_allows_the_provider_extension(settings) -> None:
    captured: dict = {}

    class Completions:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content='{}', reasoning_content='brief internal reasoning'),
                    finish_reason="stop",
                )],
                usage=SimpleNamespace(prompt_tokens=101, completion_tokens=202),
            )

    service = SimpleResearchService.__new__(SimpleResearchService)
    service.settings = settings
    service.client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))

    diagnostics = []
    assert await service._completion(
        "heavy-model", "reasoning", "high", [{"role": "user", "content": "test"}],
        diagnostics=diagnostics,
    ) == "{}"
    assert captured["max_tokens"] == settings.simple_reasoning_llm_max_tokens
    assert captured["reasoning_effort"] == "high"
    assert captured["extra_body"] == {"allowed_openai_params": ["reasoning_effort"]}
    assert diagnostics[0].provider_prompt_tokens == 101
    assert diagnostics[0].provider_completion_tokens == 202
    assert diagnostics[0].reasoning_content_chars == len("brief internal reasoning")
    assert diagnostics[0].finish_reason == "stop"


@pytest.mark.asyncio
async def test_model_timeout_is_explicit_and_records_request_facts(settings) -> None:
    class Completions:
        async def create(self, **kwargs):
            await asyncio.sleep(60)

    service = SimpleResearchService.__new__(SimpleResearchService)
    service.settings = settings.model_copy(update={"simple_request_timeout_seconds": 0.01})
    service.client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    diagnostics = []

    with pytest.raises(TimeoutError, match="timed out after 0.01 seconds"):
        await service._completion(
            "heavy-model", "reasoning", "medium", [{"role": "user", "content": "test"}],
            diagnostics=diagnostics,
        )

    assert len(diagnostics) == 1
    diagnostic = diagnostics[0]
    assert diagnostic.outcome == "timed_out"
    assert diagnostic.error_type == "TimeoutError"
    assert diagnostic.estimated_prompt_tokens > 0
    assert diagnostic.requested_max_tokens == service.settings.simple_reasoning_llm_max_tokens


def test_model_draft_rejects_opaque_evidence_ids() -> None:
    from app.services.simple_research import SimpleSynthesisDraft

    model_output = json.loads(output(passage_numbers=[1]))
    model_output["triples"][0]["evidence_ids"] = ["ev_model_must_not_control"]

    with pytest.raises(ValidationError):
        SimpleSynthesisDraft.model_validate(model_output)
