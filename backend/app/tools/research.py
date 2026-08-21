from __future__ import annotations

from typing import Any

from pydantic_ai import Agent, RunContext, UsageLimits

from app.agents.factory import AgentFactory
from app.agents.model_factory import build_model
from app.agents.toolsets import SUPPORTED_CAPABILITIES
from app.harness.deps import HarnessDeps
from app.harness.events import TraceEmitter, make_pydantic_event_handler
from app.models.domain import (
    AgentSpec, DecisionRecord, ModelProfile, PartialResearchResult, SpecialistResult,
)


def register_research_tools(agent) -> None:
    """Small meta-capability surface for the long-lived Research Director."""
    def validate_specialist_spec(ctx: RunContext[HarnessDeps], spec: AgentSpec) -> None:
        unknown = sorted(set(spec.required_capabilities) - set(SUPPORTED_CAPABILITIES))
        if unknown:
            raise ValueError(
                "Unknown specialist capabilities: " + ", ".join(unknown)
                + ". Valid capabilities are: " + ", ".join(sorted(SUPPORTED_CAPABILITIES))
                + ". Artifact kinds such as evidence_span are not capabilities."
            )
        missing = []
        for artifact_id in spec.input_artifact_ids:
            try:
                ctx.deps.artifacts.get(artifact_id)
            except KeyError:
                missing.append(artifact_id)
        if missing:
            raise ValueError(
                "AgentSpec.input_artifact_ids must contain persisted ArtifactStore IDs. "
                f"These IDs do not resolve: {missing}. IDs returned by orientation_search_documents "
                "are retrieval candidates, not artifacts; delegate with the documents capability first."
            )


    @agent.tool
    async def available_research_capabilities(ctx: RunContext[HarnessDeps]) -> dict:
        """Inspect specialist capabilities and execution constraints; this is not a workflow stage list."""
        return {
            "specialist_capabilities": sorted(SUPPORTED_CAPABILITIES),
            "meta": [
                "inspect_research_state", "inspect_artifact", "list_artifacts", "record_decision",
                "spawn_specialist", "run_specialist_fanout", "run_dynamic_workflow",
                "orientation_search_documents", "orientation_search_ontology", "persist_partial_result",
            ],
            "budgets": {
                "max_agent_steps": ctx.deps.settings.max_agent_steps,
                "max_subagent_calls": ctx.deps.settings.max_subagent_calls,
                "max_evidence_spans": ctx.deps.settings.max_evidence_spans,
                "max_evidence_chars": ctx.deps.settings.max_evidence_chars,
            },
            "retrieval_policy": "non-embedding lexical/structural only",
            "final_product": "evidence-backed canonicalized triples + provenance/evaluation artifacts; not a persistent KG",
        }

    @agent.tool
    async def inspect_research_state(ctx: RunContext[HarnessDeps]) -> dict:
        ctx.deps.ensure_not_cancelled()
        run = ctx.deps.registry.get(ctx.deps.run_id)
        return run.state.model_dump(mode="json") if run.state else {}

    @agent.tool
    async def list_artifacts(ctx: RunContext[HarnessDeps], kind: str | None = None) -> list[dict]:
        return ctx.deps.artifacts.list(kind)

    @agent.tool
    async def inspect_artifact(ctx: RunContext[HarnessDeps], artifact_id: str) -> Any:
        return ctx.deps.artifacts.get(artifact_id)

    @agent.tool
    async def record_decision(ctx: RunContext[HarnessDeps], decision: DecisionRecord) -> dict:
        ref = ctx.deps.artifacts.put("decision", decision, creator_agent_id="research_director", parent_artifact_ids=decision.artifact_ids)
        await TraceEmitter(ctx.deps.registry, ctx.deps.run_id).emit(
            type="decision", actor="research_director", action=decision.action, title="Research decision",
            summary=decision.reason, artifact_id=ref.id, payload=decision.model_dump(mode="json"),
        )
        return ref.model_dump(mode="json")

    @agent.tool
    async def orientation_search_documents(
        ctx: RunContext[HarnessDeps],
        query: str,
        limit_per_document: int = 5,
    ) -> dict:
        """
        Cheap lexical orientation for the Director.

        Returned evidence candidates are NOT ArtifactStore artifacts.
        Their IDs must not be passed as input_artifact_ids.
        Delegate to a specialist with the `documents` capability to
        retrieve, validate, and persist evidence.
        """
        ctx.deps.ensure_not_cancelled()

        rows = []

        for doc_id in ctx.deps.document_ids:
            rows.extend(
                x.model_dump(mode="json")
                for x in ctx.deps.retrieval.search(
                    ctx.deps.pdf.load(doc_id),
                    query,
                    min(limit_per_document, 10),
                    agent_id="research_director",
                )
            )

        candidates = sorted(
            rows,
            key=lambda x: -(x.get("score") or 0.0),
        )[:30]

        return {
            "candidates": candidates,
            "provenance_status": "retrieval_candidates_only",
            "artifact_ids": [],
            "instruction": (
                "These EvidenceSpan IDs are retrieval candidate IDs, not ArtifactStore IDs. "
                "Do not place them in AgentSpec.input_artifact_ids. "
                "To persist evidence, create a specialist with required_capabilities=['documents']; "
                "the specialist should search/read/expand as needed and call persist_evidence."
            ),
        }

    @agent.tool
    async def orientation_search_ontology(ctx: RunContext[HarnessDeps], terms: list[str], limit: int = 20) -> list[dict]:
        return [x.model_dump(mode="json") for x in ctx.deps.ontology.search(terms, min(limit, 40))]

    @agent.tool
    async def spawn_specialist(ctx: RunContext[HarnessDeps], spec: AgentSpec) -> dict:
        """Create one isolated ephemeral PydanticAI specialist from this runtime-generated AgentSpec."""
        ctx.deps.ensure_not_cancelled()
        validate_specialist_spec(ctx, spec)
        output = await AgentFactory(ctx.deps).run(spec)
        return output.model_dump(mode="json")

    @agent.tool
    async def run_specialist_fanout(ctx: RunContext[HarnessDeps], specs: list[AgentSpec]) -> list[dict]:
        """Dynamically fan out independent specialists; each receives isolated context/toolsets."""
        ctx.deps.ensure_not_cancelled()
        for spec in specs:
            validate_specialist_spec(ctx, spec)
        outputs = await AgentFactory(ctx.deps).run_fanout(specs)
        return [x.model_dump(mode="json") for x in outputs]

    @agent.tool
    async def run_dynamic_workflow(ctx: RunContext[HarnessDeps], specs: list[AgentSpec], orchestration_instruction: str) -> dict:
        if not specs:
            raise ValueError("Dynamic workflow requires at least one runtime-generated AgentSpec")
        ctx.deps.ensure_not_cancelled()
        for spec in specs:
            validate_specialist_spec(ctx, spec)
        factory = AgentFactory(ctx.deps)
        workflow = factory.build_workflow_capability(specs)
        if workflow is None:
            # Configured fallback preserves dynamic specialists without making DynamicWorkflow mandatory.
            return {"mode": "host_fanout", "results": [x.model_dump(mode="json") for x in await factory.run_fanout(specs)]}
        spec_refs = [ctx.deps.artifacts.put(
            "agent_spec", s, label=f"Dynamic workflow specialist: {s.role}", creator_agent_id="research_director",
            parent_artifact_ids=s.input_artifact_ids,
        ).id for s in specs]
        prompt = (
            f"Orchestration objective: {orchestration_instruction}\n\n"
            "Use run_workflow to orchestrate only the provided runtime-created specialists. They are isolated leaves; "
            "fan out or chain them as useful, then return a concise SpecialistResult summarizing their results and artifact IDs."
        )
        workflow_agent = Agent(
            build_model(ctx.deps.settings, ModelProfile.BALANCED), output_type=SpecialistResult,
            name="dynamic_workflow_orchestrator", instructions="Orchestrate the runtime specialist catalog; do not perform domain research yourself.",
            capabilities=[workflow], retries={"tools": ctx.deps.settings.llm_output_retries, "output": ctx.deps.settings.llm_output_retries},
            model_settings={"temperature": ctx.deps.settings.llm_temperature, "max_tokens": ctx.deps.settings.llm_max_tokens},
        )
        result = await workflow_agent.run(
            prompt, event_stream_handler=make_pydantic_event_handler(ctx.deps.registry, ctx.deps.run_id, "dynamic_workflow"),
            usage_limits=UsageLimits(request_limit=max(3, ctx.deps.settings.max_subagent_calls + 3)),
            conversation_id=f"{ctx.deps.run_id}:dynamic_workflow",
        )
        ref = ctx.deps.artifacts.put("dynamic_workflow_result", result.output, creator_agent_id="dynamic_workflow",
                                    parent_artifact_ids=spec_refs)
        return {"mode": "pydantic_ai_harness_dynamic_workflow", "artifact_ref": ref.model_dump(mode="json"),
                "result": result.output.model_dump(mode="json")}

    @agent.tool
    async def persist_partial_result(ctx: RunContext[HarnessDeps], partial: PartialResearchResult) -> dict:
        """Gracefully terminate with typed partial findings instead of hallucinating success."""
        partial_ref = ctx.deps.artifacts.put("partial_result", partial, label="Graceful partial research result", creator_agent_id="research_director")
        triples_ref = ctx.deps.artifacts.put("canonical_triples", [], label="Accepted canonical triples (partial run)",
                                             creator_agent_id="research_director", parent_artifact_ids=[partial_ref.id])
        rejected_ref = ctx.deps.artifacts.put("rejected_triples", [], creator_agent_id="research_director", parent_artifact_ids=[partial_ref.id])
        return {"partial_ref": partial_ref.model_dump(mode="json"), "canonical_triples_ref": triples_ref.model_dump(mode="json"),
                "rejected_triples_ref": rejected_ref.model_dump(mode="json")}
