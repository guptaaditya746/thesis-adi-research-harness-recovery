"""Bounded working memory for the long-lived Research Director.

This module deliberately manages only conversational working memory.  Immutable
scientific state remains in :class:`ArtifactStore`; a compaction summary therefore
contains artifact references and factual inventory, never copied evidence or RDF.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.tools import RunContext
from pydantic_ai_harness.compaction import ClearToolResults, estimate_token_count
from pydantic_ai_harness.compaction._shared import compact_with_span, find_token_cutoff

from app.config import Settings
from app.harness.deps import HarnessDeps
from app.models.domain import (
    AgentTraceEvent,
    ContextArtifactReference,
    ContextCompactionReceipt,
    DirectorWorkingMemory,
    ScientificStateSnapshot,
    stable_hash,
)


_SCIENTIFIC_KINDS = {
    "evidence_bundle": "evidence_bundle",
    "edc_open_triples": "edc_open_triples",
    "edc_relation_definitions": "edc_relation_definitions",
    "ontology_snapshot": "ontology_snapshot",
    "candidate_canonical_triples": "candidate_canonical_triples",
    "verification_report": "verification_report",
    "verification": "verification_report",
    "canonical_triples": "canonical_triples",
}


class SemanticContextSummary(BaseModel):
    """The only model-authored portion of a Director working-memory receipt."""

    findings: list[str] = Field(default_factory=list, max_length=8)
    unresolved_questions: list[str] = Field(default_factory=list, max_length=8)
    current_intent: str | None = Field(default=None, max_length=400)
    caveats: list[str] = Field(default_factory=list, max_length=5)


SummaryBuilder = Callable[[str, RunContext[HarnessDeps]], Awaitable[SemanticContextSummary]]


@dataclass
class ResearchContextManager(AbstractCapability[HarnessDeps]):
    """Clear bulky tool history, then replace old dialogue with typed working memory.

    The manager triggers only when an explicit operator-declared context window is
    present.  It reserves response tokens and a safety margin before calculating
    the safe input budget, so it acts before the provider has to reject a request.
    """

    settings: Settings
    summary_builder: SummaryBuilder | None = None

    def _budgets(self) -> tuple[int, int, int] | None:
        if not self.settings.enable_context_management or not self.settings.llm_context_window:
            return None
        safe = self.settings.llm_context_window - self.settings.llm_max_tokens - self.settings.context_safety_margin_tokens
        if safe <= 0:
            return None
        warn = int(safe * self.settings.context_warn_fraction)
        target = max(1_024, int(safe * self.settings.context_compact_target_fraction))
        return safe, warn, target

    @staticmethod
    def _bounded(value: object, limit: int = 900) -> str:
        text = str(value).replace("\x00", " ").strip()
        return text if len(text) <= limit else text[: limit - 1] + "…"

    def _render_history(self, messages: list[ModelMessage], *, max_chars: int = 14_000) -> str:
        """Render a strictly bounded operational transcript, never raw long payloads."""
        rows: list[str] = []
        used = 0
        for message in messages:
            parts = getattr(message, "parts", [])
            for part in parts:
                if isinstance(part, UserPromptPart):
                    value = part.content
                    label = "user"
                elif isinstance(part, SystemPromptPart):
                    value = part.content
                    label = "system"
                elif isinstance(part, TextPart):
                    value = part.content
                    label = "assistant"
                elif isinstance(part, ToolCallPart):
                    value = f"{part.tool_name}({self._bounded(part.args_as_json_str(), 500)})"
                    label = "tool-call"
                elif isinstance(part, ToolReturnPart):
                    value = self._bounded(part.content, 500)
                    label = "tool-result"
                else:
                    continue
                row = f"{label}: {self._bounded(value, 900)}"
                if used + len(row) > max_chars:
                    rows.append("[older transcript clipped; use ArtifactStore references below]")
                    return "\n".join(rows)
                rows.append(row)
                used += len(row)
        return "\n".join(rows)

    def _host_working_memory(self, deps: HarnessDeps, semantic: SemanticContextSummary) -> DirectorWorkingMemory:
        items = deps.artifacts.list()
        by_kind: dict[str, list[dict[str, Any]]] = {}
        for item in items:
            by_kind.setdefault(str(item["kind"]), []).append(item)

        state = ScientificStateSnapshot(**{
            field: any(bool(by_kind.get(kind)) for kind, target in _SCIENTIFIC_KINDS.items() if target == field)
            for field in ScientificStateSnapshot.model_fields
        })
        # Keep references small but sufficient for the Director to inspect/reuse
        # immutable state.  The latest two of each kind cover retries without
        # pretending they are an exhaustive provenance graph.
        refs: list[ContextArtifactReference] = []
        preferred_kinds = [
            "retrieval_candidate_bundle", "evidence_bundle", "edc_open_triples",
            "edc_relation_definitions", "ontology_proposal", "ontology_proposal_acceptance",
            "ontology_snapshot", "candidate_canonical_triples", "verification_report",
            "verification", "canonical_triples",
        ]
        for kind in preferred_kinds:
            kind_items = by_kind.get(kind, [])
            for item in kind_items[-2:]:
                refs.append(ContextArtifactReference(
                    id=str(item["id"]), kind=kind,
                    parent_artifact_ids=list(item.get("parent_artifact_ids", [])),
                ))
        return DirectorWorkingMemory(
            research_question=deps.registry.get(deps.run_id).question,
            target_document_id=deps.target_document_id,
            scientific_state=state,
            artifact_references=refs[:32],
            findings=semantic.findings[:8],
            unresolved_questions=semantic.unresolved_questions[:8],
            current_intent=semantic.current_intent,
            caveats=semantic.caveats[:5],
        )

    def goal_compass(self, deps: HarnessDeps) -> str:
        """A short factual orientation aid; it deliberately never prescribes an action."""
        memory = self._host_working_memory(deps, SemanticContextSummary())
        state = memory.scientific_state.model_dump(mode="json")
        refs = [f"{ref.kind}:{ref.id}" for ref in memory.artifact_references]
        return (
            "<goal-compass>\n"
            "Terminal scientific target: accepted canonical triples grounded in exact evidence, canonicalized against "
            "a run-local ontology snapshot, and independently verified. Prose or intermediate artifacts are not completion.\n"
            f"Factual ArtifactStore state: {state}\n"
            f"Available immutable references: {refs}\n"
            "These are facts only. Decide the next research action autonomously.\n"
            "</goal-compass>"
        )

    async def _summarize(self, transcript: str, ctx: RunContext[HarnessDeps]) -> tuple[SemanticContextSummary, bool]:
        prompt = (
            "Summarize this older operational transcript for a research director. "
            "Return only short findings, unresolved questions, current intent, and caveats. "
            "Do not invent scientific evidence, artifact IDs, ontology terms, or next-step instructions. "
            "ArtifactStore is authoritative; this is only working memory.\n\n"
            f"<transcript>\n{transcript}\n</transcript>"
        )
        try:
            # Hold one request for the Director itself. A summary is an aid, not a
            # reason to spend the final request budget and strand the run.
            if ctx.usage.requests >= self.settings.max_agent_steps - 1:
                raise RuntimeError("insufficient remaining request budget for semantic compaction")
            # Import lazily: tests and zero-cost compaction do not need to pay for
            # the full Agent construction/import path.
            from pydantic_ai import Agent
            from app.agents.model_factory import build_named_model

            model = build_named_model(self.settings, self.settings.context_summary_model) if self.settings.context_summary_model else ctx.model
            agent = Agent(
                model,
                output_type=SemanticContextSummary,
                instructions="Create compact, factual working memory. Never provide chain-of-thought.",
                retries=0,
                model_settings={"temperature": 0.0, "max_tokens": self.settings.context_summary_max_tokens},
            )
            result = await agent.run(prompt, usage=ctx.usage)
            return result.output, False
        except Exception as exc:
            return SemanticContextSummary(caveats=[f"Semantic compaction unavailable: {type(exc).__name__}. Inspect ArtifactStore for authoritative state."]), True

    @staticmethod
    def _first_user_request(messages: list[ModelMessage]) -> ModelRequest | None:
        for message in messages:
            if isinstance(message, ModelRequest) and any(isinstance(part, UserPromptPart) for part in message.parts):
                return message
        return None

    async def _persist_receipt(
        self,
        ctx: RunContext[HarnessDeps],
        receipt: ContextCompactionReceipt,
    ) -> None:
        ref = ctx.deps.artifacts.put(
            "context_compaction_receipt", receipt,
            label="Director context compaction receipt",
            creator_agent_id="research_context_manager",
        )
        await ctx.deps.registry.emit(AgentTraceEvent(
            run_id=ctx.deps.run_id,
            type="system",
            actor="research_context_manager",
            action="context_compacted",
            title="Director context compacted",
            status="info",
            artifact_id=ref.id,
            summary=f"{receipt.strategy}: {receipt.input_tokens_before} → {receipt.input_tokens_after} estimated tokens.",
            payload={"strategy": receipt.strategy, "tokens_before": receipt.input_tokens_before,
                     "tokens_after": receipt.input_tokens_after, "target_tokens": receipt.target_tokens},
        ))

    async def compact(self, messages: list[ModelMessage], ctx: RunContext[HarnessDeps]) -> list[ModelMessage]:
        budgets = self._budgets()
        if budgets is None:
            return messages
        safe, warn, target = budgets
        before_tokens = estimate_token_count(messages)
        if before_tokens < warn:
            return messages

        clearer = ClearToolResults(
            max_tokens=1,
            keep_pairs=self.settings.context_keep_tool_pairs,
            placeholder="[tool result cleared; inspect its ArtifactStore reference or rerun the tool]",
            min_clear_tokens=1,
        )
        cleared = await clearer.compact(messages, ctx)
        cleared_tokens = estimate_token_count(cleared)
        if cleared_tokens <= target:
            memory = self._host_working_memory(ctx.deps, SemanticContextSummary())
            await self._persist_receipt(ctx, ContextCompactionReceipt(
                strategy="clear_tool_results", input_tokens_before=before_tokens, input_tokens_after=cleared_tokens,
                safe_input_budget=safe, target_tokens=target, messages_before=len(messages), messages_after=len(cleared),
                retained_artifact_ids=[ref.id for ref in memory.artifact_references],
            ))
            return cleared

        # Reserve enough space for the new typed working-memory request.  The
        # cutoff is tool-pair-safe, so resumed histories remain valid.
        cutoff = find_token_cutoff(cleared, max(1_024, target - 2_000))
        if cutoff <= 0:
            return cleared
        old, tail = cleared[:cutoff], cleared[cutoff:]
        if self.summary_builder is not None:
            try:
                semantic = await self.summary_builder(self._render_history(old), ctx)
                fallback = False
            except Exception as exc:
                semantic = SemanticContextSummary(caveats=[f"Semantic compaction unavailable: {type(exc).__name__}."])
                fallback = True
        else:
            semantic, fallback = await self._summarize(self._render_history(old), ctx)
        memory = self._host_working_memory(ctx.deps, semantic)
        summary = ModelRequest(parts=[SystemPromptPart(content=(
            "Research working memory after context compaction. This is non-authoritative; "
            "inspect ArtifactStore for scientific payloads.\n" + memory.model_dump_json()
        ))])
        first_user = self._first_user_request(cleared)
        compacted: list[ModelMessage] = [summary]
        if first_user is not None and first_user not in tail:
            compacted.append(first_user)
        compacted.extend(tail)
        after_tokens = estimate_token_count(compacted)
        await self._persist_receipt(ctx, ContextCompactionReceipt(
            strategy="hybrid_typed_summary", input_tokens_before=before_tokens, input_tokens_after=after_tokens,
            safe_input_budget=safe, target_tokens=target, messages_before=len(messages), messages_after=len(compacted),
            retained_artifact_ids=[ref.id for ref in memory.artifact_references],
            summary_sha256=stable_hash(memory), summary_fallback_used=fallback,
        ))
        return compacted

    async def before_model_request(
        self,
        ctx: RunContext[HarnessDeps],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        messages = list(request_context.messages)
        request_context.messages = await compact_with_span(
            ctx,
            strategy="ResearchContextManager",
            messages=messages,
            compact=lambda: self.compact(messages, ctx),
        )
        # The installed Harness version has no SystemReminders capability. Add a
        # small, state-derived request-tail nudge on a cadence instead of bloating
        # durable Director instructions or prescribing a workflow.
        if ctx.run_step and ctx.run_step % 3 == 0:
            request_context.messages = [
                *request_context.messages,
                ModelRequest(parts=[UserPromptPart(content=self.goal_compass(ctx.deps))]),
            ]
        return request_context
