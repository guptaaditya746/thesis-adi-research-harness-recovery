from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import Any

from pydantic_ai import Agent, ModelRetry, UsageLimits

from app.agents.model_factory import build_model
from app.config import Settings
from app.harness.capabilities import build_director_capabilities
from app.harness.deps import HarnessDeps
from app.harness.events import TraceEmitter, make_pydantic_event_handler
from app.models.domain import (
    CanonicalTriple, ExperimentRun, PartialResearchResult, RejectedTriple, ResearchConclusion, ResearchState,
)
from app.services.edc import EDCService
from app.services.entities import EntityNormalizationService
from app.services.evaluation import EvaluationService
from app.services.ontology import OntologyService
from app.services.pdf import PDFService
from app.services.retrieval import RetrievalService
from app.services.sparql import SPARQLService
from app.services.validation import StructuralValidationService
from app.services.verification import TripleVerificationService
from app.storage.artifacts import ArtifactStore
from app.storage.registry import RunRegistry
from app.telemetry.tracing import Telemetry
from app.tools.extensions import register_configured_extensions
from app.tools.research import register_research_tools


DIRECTOR_INSTRUCTIONS = """
You are the long-lived Research Director of an ontology-guided scientific triple extraction system.
You are goal-oriented, NOT a fixed pipeline. At every turn inspect the current state and decide what uncertainty or gap
matters next. You may use a small direct orientation tool, inspect artifacts, or dynamically create an ephemeral specialist
with an AgentSpec. Create specialist roles/prompts at runtime; do not assume a predefined list of researchers.

Document-access contract:
- A paper ID is a logical identifier, not a filename.
- First call `orientation_search_documents` with terms from the research question; it reads the indexed corpus directly and returns exact evidence candidates.
- The Research Director must not perform scientific analysis through filesystem, shell, or code-execution tools.
- Use `orientation_search_documents` for cheap corpus orientation.
- Delegate evidence collection, EDC, ontology work, and verification to specialists with the required typed capabilities.
- Do not answer from paper title/assumptions. Use retrieved evidence before finalizing.
- Scientific provenance is ArtifactStore-only: never create or treat workspace files as `evidence_span`, `evidence_bundle`, `candidate_canonical_triples`, `canonical_triples`, or verification artifacts.
- A completed result requires this artifact sequence: persisted evidence bundle → candidate canonical triples from EDC → accepted canonical triples from verification.
- `triple_artifact_id` is an ArtifactStore ID returned by verification, never a filesystem path.
- `orientation_search_documents` returns ephemeral retrieval candidates, not ArtifactStore artifacts.
- Never place an `ev_*` ID returned only by orientation search into `AgentSpec.input_artifact_ids`.
- `input_artifact_ids` may contain only IDs that successfully resolve through `inspect_artifact`.
- To turn retrieved text into immutable evidence, delegate to a specialist with the `documents` capability and have it call `persist_evidence`.
- Specialist `required_capabilities` names must come only from `available_research_capabilities.specialist_capabilities`. Artifact kinds such as `evidence_span` are not capabilities.
- `orientation_search_documents` also returns a `candidate_bundle_artifact_id` containing the transient retrieval candidates.
- Retrieval candidates are not accepted scientific evidence. Pass that candidate bundle ID through `AgentSpec.input_artifact_ids` to a documents specialist.
- Never copy retrieved document text into AgentSpec instructions as a substitute for artifact handoff; instructions describe behavior and artifact IDs carry data.
- A documents specialist should inspect the candidate bundle, select relevant spans, validate them against the source, and call `persist_evidence`.
- Do not invoke EDC until an actual `evidence_bundle` artifact exists and is included in the EDC specialist's `input_artifact_ids`.

Scientific invariants:
- Final research output is evidence-backed canonicalized (subject, predicate, object) triples plus provenance/evaluation.
  It is NOT a persistent knowledge graph.
- Retrieval is non-embedding: lexical/FTS/structural/safe shell search only. Never request embeddings/vector databases.
- DBLP RDF schema is the immutable seed. Missing concepts become explicit run-local OntologyProposal artifacts.
- SPARQL is an inspectable semantic information-need/schema pattern, not a way to query facts from raw PDFs.
- EDC runs only after evidence retrieval/persistence and only over immutable EvidenceSpan artifacts.
- Every accepted triple must resolve to exact evidence in the supplied paper corpus. SHACL checks structure, not truth.
- Specialists are isolated, bounded, and should receive narrow tasks/artifact IDs/tool capabilities.
- Use independent verification when facts/canonicalization are uncertain. Reject/mark uncertain rather than hallucinating.
- Recovery is expected: repair structured/tool errors, narrow context, decompose, re-delegate, use an independent critic,
  escalate model profile, change search strategy, or return a typed partial result when reliable completion is impossible.
- Do not expose hidden chain-of-thought. Make the research auditable with task messages, generated specialist prompts,
  DecisionRecord summaries, tool calls/results, retries/recovery actions, and immutable artifacts.

There is no mandatory order. An easy question may only need retrieval -> EDC -> verification; a hard question may revisit
ontology/retrieval/critics several times. Before a completed terminal answer, ensure a `canonical_triples` artifact exists.
If reliable completion is impossible, use `persist_partial_result` and return status=partial rather than inventing success.
"""


class ResearchHarness:
    def __init__(self, settings: Settings, registry: RunRegistry):
        self.settings = settings
        self.registry = registry
        self.pdf = PDFService(settings)
        self.retrieval = RetrievalService(settings)
        self.ontology = OntologyService(settings)
        self.sparql = SPARQLService()
        self.edc = EDCService(settings)
        self.entities = EntityNormalizationService()
        self.verification = TripleVerificationService(settings)
        self.validation = StructuralValidationService()
        self.evaluation = EvaluationService()
        self.telemetry = Telemetry(settings)

    async def execute(self, run_id: str, *, resume: bool = False) -> None:
        run = self.registry.get(run_id)
        self.registry.clear_cancel(run_id)
        run.status = "running"
        run.resumable = True
        run.active_actor = "research_director"
        workspace = self.settings.data_dir / "runs" / run.id / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        self._mount_documents(workspace, run.document_ids)
        artifacts = ArtifactStore(self.settings, run.id)
        emitter = TraceEmitter(self.registry, run.id)
        if run.state is None:
            run.state = ResearchState(
                run_id=run.id, workspace_id=run.workspace_id, document_ids=run.document_ids, question=run.question,
                context_budget=self.settings.max_evidence_chars, tool_budget=self.settings.max_agent_steps,
                status="running",
            )
        else:
            run.state.status = "running"
        self.registry.persist(run)

        deps = HarnessDeps(
            settings=self.settings, run_id=run.id, workspace_id=run.workspace_id, document_ids=run.document_ids,
            artifacts=artifacts, registry=self.registry, pdf=self.pdf, retrieval=self.retrieval,
            ontology=self.ontology, sparql=self.sparql, edc=self.edc, entities=self.entities,
            verification=self.verification, validation=self.validation, evaluation=self.evaluation, telemetry=self.telemetry,
        )
        capabilities, step_store = build_director_capabilities(self.settings, workspace, run.id)
        director = Agent(
            build_model(self.settings, "balanced"), deps_type=HarnessDeps, output_type=ResearchConclusion,
            name="research_director", description="Goal-oriented thesis Research Director with runtime specialist creation.",
            instructions=DIRECTOR_INSTRUCTIONS, capabilities=capabilities,
            retries={"tools": self.settings.llm_output_retries, "output": self.settings.llm_output_retries},
            model_settings={"temperature": self.settings.llm_temperature, "max_tokens": self.settings.llm_max_tokens},
        )
        register_research_tools(director)
        register_configured_extensions(director, self.settings)

        @director.output_validator
        async def completion_gate(ctx, output: ResearchConclusion) -> ResearchConclusion:
            final_meta = ctx.deps.artifacts.find_latest("canonical_triples")
            if not final_meta:
                raise ModelRetry(
                    "Terminal output is not auditable yet. Persist accepted canonical triples with verification, or call "
                    "persist_partial_result to create an explicit empty/partial final triple artifact."
                )
            if output.status == "completed":
                evidence_bundle = ctx.deps.artifacts.find_latest("evidence_bundle")
                candidate_triples = ctx.deps.artifacts.find_latest("candidate_canonical_triples")
                if not evidence_bundle or not candidate_triples:
                    raise ModelRetry("A completed result requires persisted evidence_bundle and candidate_canonical_triples artifacts. Delegate to a specialist with documents, ontology, edc, and verification capabilities.")
                if candidate_triples["id"] not in final_meta.get("parent_artifact_ids", []):
                    raise ModelRetry("Final canonical_triples must be created by verification from candidate_canonical_triples, not by a filesystem write.")
                if evidence_bundle["id"] not in candidate_triples.get("parent_artifact_ids", []):
                    raise ModelRetry("Candidate canonical triples must descend from a persisted evidence_bundle.")
            triples = [CanonicalTriple.model_validate(x) for x in ctx.deps.artifacts.get(final_meta["id"])]
            for triple in triples:
                if not triple.evidence_ids:
                    raise ModelRetry(f"Triple {triple.id} has no evidence IDs and cannot be accepted.")
                if not triple.predicate_uri or triple.canonicalization == "open_relation":
                    raise ModelRetry(f"Triple {triple.id} still has an open/non-canonical predicate; resolve/reject it before finishing.")
                if triple.verification_status != "accepted":
                    raise ModelRetry(f"Triple {triple.id} is not independently accepted; verify or reject it before finishing.")
                for evidence_id in triple.evidence_ids:
                    try:
                        raw = ctx.deps.artifacts.get(evidence_id)
                    except KeyError:
                        raise ModelRetry(f"Evidence {evidence_id} does not resolve to an immutable evidence_span artifact.")
                    from app.models.domain import EvidenceSpan
                    span = EvidenceSpan.model_validate(raw)
                    ok, reason = ctx.deps.retrieval.validate_span(ctx.deps.pdf.load(span.document_id), span)
                    if not ok:
                        raise ModelRetry(f"Evidence integrity failed for {evidence_id}: {reason}")
            if output.triple_artifact_id and output.triple_artifact_id != final_meta["id"]:
                raise ModelRetry(f"Use the final canonical_triples artifact id {final_meta['id']} in triple_artifact_id.")
            output.triple_artifact_id = final_meta["id"]
            return output

        event_handler = make_pydantic_event_handler(self.registry, run.id, "research_director")
        initial_prompt = self._initial_prompt(run, artifacts, resume)
        message_history = None
        if resume and step_store is not None:
            message_history = await self._resume_history(step_store, run.id)

        await emitter.emit(
            type="run", actor="research_harness", action="resume" if resume else "start",
            title="Research run resumed" if resume else "Research run started", status="started",
            summary="The Research Director dynamically selects actions and specialists; no fixed stage workflow is enforced.",
        )
        await emitter.emit(
            type="message", actor="user", action="research_question", title="Research task message",
            payload={"question": run.question, "document_ids": run.document_ids, "director_prompt": initial_prompt},
        )

        try:
            with self.telemetry.span("research.run", {
                "research.run_id": run.id, "research.workspace_id": run.workspace_id,
                "research.question": run.question, "research.document_ids": run.document_ids,
                "research.resume": resume, "research.final_product": "canonicalized_triples",
            }) as root_span:
                trace_id = self.telemetry.trace_id(root_span)
                if trace_id:
                    run.langtrace_trace_id = trace_id
                    self.registry.persist(run)
                result = await asyncio.wait_for(
                    director.run(
                        initial_prompt, deps=deps, event_stream_handler=event_handler,
                        usage_limits=UsageLimits(request_limit=self.settings.max_agent_steps),
                        conversation_id=run.id,
                        message_history=message_history,
                    ),
                    timeout=self.settings.max_run_seconds,
                )
                conclusion = result.output
                root_span.set_attribute("research.completion_status", conclusion.status)

            self._finalize_success(run.id, conclusion, artifacts)
            await emitter.emit(
                type="answer", actor="research_director", action="finalize", title="Research output finalized",
                status="completed", summary=conclusion.answer_summary, artifact_id=conclusion.triple_artifact_id,
                payload={"human_answer": conclusion.answer, "evidence_ids": conclusion.evidence_ids,
                         "unresolved_issues": conclusion.unresolved_issues, "status": conclusion.status},
            )
            await emitter.emit(type="run", actor="research_harness", action="complete", title=f"Research run {conclusion.status}", status="completed")
        except asyncio.CancelledError:
            self._mark_terminal(run.id, "cancelled", "Run task cancelled")
            await emitter.emit(type="run", actor="research_harness", action="cancel", title="Research run cancelled", status="failed")
            raise
        except asyncio.TimeoutError:
            if artifacts.find_latest("canonical_triples"):
                self._finalize_existing_partial(run.id, artifacts, "Run time budget exhausted after producing an auditable partial result.")
                await emitter.emit(type="run", actor="research_harness", action="timeout_partial", title="Run ended with partial result", status="completed")
            else:
                self._mark_terminal(run.id, "failed", "MAX_RUN_SECONDS budget exhausted before an auditable result was persisted")
                await emitter.emit(type="run", actor="research_harness", action="timeout", title="Research run timed out", status="failed")
        except Exception as exc:
            if self.registry.is_cancelled(run.id):
                self._mark_terminal(run.id, "cancelled", str(exc))
                await emitter.emit(type="run", actor="research_harness", action="cancel", title="Research run cancelled", status="failed")
            else:
                self._mark_terminal(run.id, "failed", f"{type(exc).__name__}: {exc}")
                await emitter.emit(type="run", actor="research_harness", action="error", title="Research run failed",
                                   status="failed", summary=f"{type(exc).__name__}: {exc}")

    def _finalize_success(self, run_id: str, conclusion: ResearchConclusion, artifacts: ArtifactStore) -> None:
        run = self.registry.get(run_id)
        final_meta = artifacts.find_latest("canonical_triples")
        run.final_triples = [CanonicalTriple.model_validate(x) for x in artifacts.get(final_meta["id"])] if final_meta else []
        rejected_meta = artifacts.find_latest("rejected_triples")
        if rejected_meta:
            rejected = []
            for item in artifacts.get(rejected_meta["id"]):
                try:
                    rejected.append(RejectedTriple.model_validate(item))
                except Exception:
                    pass
            run.rejected_triples = rejected
        run.status = conclusion.status
        run.resumable = False
        run.active_actor = None
        run.final_answer = conclusion.answer or None
        if run.state:
            run.state.status = conclusion.status
            run.state.canonical_triple_ids = [x.id for x in run.final_triples]
            run.state.evidence_ids = sorted({eid for x in run.final_triples for eid in x.evidence_ids})
        self._evaluate(run, artifacts)
        self.registry.persist(run)

    def _finalize_existing_partial(self, run_id: str, artifacts: ArtifactStore, reason: str) -> None:
        run = self.registry.get(run_id)
        final_meta = artifacts.find_latest("canonical_triples")
        run.final_triples = [CanonicalTriple.model_validate(x) for x in artifacts.get(final_meta["id"])] if final_meta else []
        run.status = "partial"
        run.resumable = False
        run.active_actor = None
        run.error = reason
        if run.state:
            run.state.status = "partial"
        self._evaluate(run, artifacts)
        self.registry.persist(run)

    def _evaluate(self, run: ExperimentRun, artifacts: ArtifactStore) -> None:
        validation_meta = artifacts.find_latest("structural_validation")
        shacl = False
        if validation_meta:
            try:
                shacl = bool(artifacts.get(validation_meta["id"]).get("shacl_conforms"))
            except Exception:
                pass
        events = self.registry.events(run.id)
        run.evaluation = self.evaluation.evaluate(
            run.final_triples, shacl_conforms=shacl, gold=run.gold,
            rejected_count=len(run.rejected_triples),
            tool_actions=sum(x.type == "tool" and x.status == "completed" for x in events),
            subagents_spawned=sum(x.action == "spawn_specialist" for x in events),
            retries=sum(x.type == "recovery" for x in events),
            model_escalations=sum(x.type == "recovery" and x.action == "model_escalation" for x in events),
        )
        artifacts.put("evaluation", run.evaluation, label="Run evaluation", creator_agent_id="research_harness")

    def _mark_terminal(self, run_id: str, status: str, error: str | None = None) -> None:
        run = self.registry.get(run_id)
        run.status = status
        run.active_actor = None
        run.resumable = False
        run.error = error
        if run.state:
            run.state.status = status
        self.registry.persist(run)

    async def _resume_history(self, step_store: Any, conversation_id: str):
        try:
            records = await step_store.list_runs(conversation_id=conversation_id)
            if not records:
                return None
            from pydantic_ai_harness.step_persistence import continue_run
            # Prefer latest complete settled snapshot. If none exists, start with typed filesystem/run state instead.
            return await continue_run(step_store, run_id=records[-1].run_id)
        except Exception:
            return None

    def _mount_documents(self, workspace: Path, document_ids: list[str]) -> None:
        docs_dir = workspace / "papers"
        docs_dir.mkdir(exist_ok=True)
        for did in document_ids:
            doc = self.pdf.load(did)
            target = docs_dir / did
            target.mkdir(exist_ok=True)
            src = Path(doc.workspace)
            for name in ("document.txt", "document.json", "search.sqlite3"):
                p = src / name
                link = target / name
                if p.exists() and not link.exists():
                    try:
                        link.symlink_to(p.resolve())
                    except OSError:
                        if p.is_file():
                            shutil.copy2(p, link)
            pages = target / "pages"
            pages.mkdir(exist_ok=True)
            for p in src.glob("page_*.txt"):
                link = pages / p.name
                if not link.exists():
                    try:
                        link.symlink_to(p.resolve())
                    except OSError:
                        shutil.copy2(p, link)

    def _initial_prompt(self, run: ExperimentRun, artifacts: ArtifactStore, resume: bool) -> str:
        docs = [self.pdf.load(i) for i in run.document_ids]
        inventory = "\n".join(
            f"- {d.id}: {d.metadata.title or d.filename}; {d.page_count} pages; parser={d.parse_engine}; mounted text=papers/{d.id}/document.txt; DOI={d.metadata.doi or '-'}"
            for d in docs
        )
        prior = artifacts.list()
        prior_text = "\n".join(f"- {x['id']} [{x['kind']}] {x.get('label') or ''}" for x in prior[-30:]) or "(none)"
        return (
            f"Research question:\n{run.question}\n\nAvailable papers:\n{inventory}\n\n"
            f"Existing run artifacts {'(resume)' if resume else ''}:\n{prior_text}\n\n"
            "Inspect the research state and choose the next useful action dynamically. Generate narrowly scoped AgentSpecs when "
            "specialization, independence, critique, or decomposition would improve reliability."
        )
