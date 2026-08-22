from __future__ import annotations

import asyncio
import importlib.metadata
import inspect
import platform
import shutil
from pathlib import Path
from typing import Any

from pydantic_ai import Agent, UsageLimits
from pydantic import TypeAdapter

from app.agents.model_factory import build_model
from app.config import Settings
from app.harness.capabilities import build_director_capabilities
from app.harness.deps import HarnessDeps
from app.harness.events import TraceEmitter, make_pydantic_event_handler
from app.models.domain import (
    CanonicalTriple, ExperimentRun, FailureRecord, RejectedTriple, ResearchConclusion, ResearchState,
    RunRuntimeIdentity, stable_hash,
)
from app.harness.completion import ScientificCompletionService
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
- A completed result requires this exact artifact sequence: evidence_bundle → edc_open_triples → edc_relation_definitions + ontology_snapshot → candidate_canonical_triples → independent verification → canonical_triples.
- `triple_artifact_id` is an ArtifactStore ID returned by verification, never a filesystem path.
- `orientation_search_documents` returns bounded previews plus a persisted `candidate_bundle_artifact_id`; the candidates are not accepted evidence.
- Never place an `ev_*` ID returned only by orientation search into `AgentSpec.input_artifact_ids`.
- `input_artifact_ids` may contain only IDs that successfully resolve through `inspect_artifact`.
- To turn retrieved text into immutable evidence, delegate to a specialist with the `documents` capability and have it call `persist_evidence`.
- Specialist `required_capabilities` names must come only from `available_research_capabilities.specialist_capabilities`. Artifact kinds such as `evidence_span` are not capabilities.
- Retrieval candidates are not accepted scientific evidence. Pass that candidate bundle ID through `AgentSpec.input_artifact_ids` to a documents specialist.
- Never copy retrieved document text into AgentSpec instructions as a substitute for artifact handoff; instructions describe behavior and artifact IDs carry data.
- A documents specialist should inspect the candidate bundle, select existing candidate IDs, and call `persist_candidate_evidence`; never reconstruct an EvidenceSpan from a preview.
- Retrieval candidate previews use `candidate_id`/`preview`; `preview` may contain an ellipsis and is never source evidence.
- EDC Extract requires an evidence_bundle. EDC Define requires edc_open_triples. EDC Canonicalize requires the matching open triples, relation definitions, and a run-specific ontology_snapshot.
- Use EDC relation definitions as the primary query for DBLP relation retrieval before Canonicalize. Prefer a justified run-local ontology proposal to an inappropriate DBLP predicate.
- A blocked specialist is expected when scientific state is missing; continue from its persisted artifacts rather than retrying or escalating the model.

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
  escalate model profile, or change search strategy. If resources end before scientific completion, the run fails with diagnostics.
- Do not expose hidden chain-of-thought. Make the research auditable with task messages, generated specialist prompts,
  DecisionRecord summaries, tool calls/results, retries/recovery actions, and immutable artifacts.

There is no mandatory order. An easy question may only need retrieval -> EDC -> verification; a hard question may revisit
ontology/retrieval/critics several times. When you believe research is complete, call `attempt_complete_research` with the
canonical_triples artifact ID. A rejected attempt returns factual state only; choose the next action yourself.
"""


class ResearchHarness:
    def __init__(self, settings: Settings, registry: RunRegistry, telemetry: Telemetry | None = None):
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
        self.completion = ScientificCompletionService()
        self.evaluation = EvaluationService()
        self.telemetry = telemetry or Telemetry(settings)

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
        emitter = TraceEmitter(self.registry, run.id, self.telemetry)
        if run.state is None:
            run.state = ResearchState(
                run_id=run.id, workspace_id=run.workspace_id, document_ids=run.document_ids, question=run.question,
                context_budget=self.settings.max_evidence_chars, tool_budget=self.settings.max_agent_steps,
                status="running",
            )
        else:
            run.state.status = "running"
        self.registry.persist(run)

        target_document_id = run.target_document_id or run.document_ids[0]
        run.target_document_id = target_document_id
        deps = HarnessDeps(
            settings=self.settings, run_id=run.id, workspace_id=run.workspace_id, document_ids=run.document_ids,
            target_document_id=target_document_id,
            artifacts=artifacts, registry=self.registry, pdf=self.pdf, retrieval=self.retrieval,
            ontology=self.ontology, sparql=self.sparql, edc=self.edc, entities=self.entities,
            verification=self.verification, validation=self.validation, evaluation=self.evaluation,
            completion=self.completion, telemetry=self.telemetry,
        )
        capabilities, step_store = build_director_capabilities(self.settings, workspace, run.id)
        director = Agent(
            build_model(self.settings, "balanced"), deps_type=HarnessDeps, output_type=str,
            name="research_director", description="Goal-oriented thesis Research Director with runtime specialist creation.",
            instructions=DIRECTOR_INSTRUCTIONS, capabilities=capabilities,
            retries={"tools": self.settings.llm_output_retries},
            model_settings={"temperature": self.settings.llm_temperature, "max_tokens": self.settings.llm_max_tokens},
        )
        register_research_tools(director)
        register_configured_extensions(director, self.settings)

        initial_prompt = self._initial_prompt(run, artifacts, resume)
        identity_ref = self._persist_runtime_identity(run, artifacts, initial_prompt)
        run.runtime_identity_artifact_ids.append(identity_ref.id)
        self.registry.persist(run)

        event_handler = make_pydantic_event_handler(self.registry, run.id, self.telemetry, "research_director")
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
                attempt = self._latest_accepted_completion(artifacts)
                if attempt is None:
                    self._record_failed_completion(
                        run.id, artifacts,
                        "Director ended without an accepted scientific completion attempt.",
                    )
                    await emitter.emit(type="run", actor="research_harness", action="incomplete_failure",
                                       title="Research run failed", status="failed",
                                       summary="No accepted scientific completion attempt was recorded.")
                    return
                conclusion = self._conclusion_from_completion(attempt, artifacts)
                root_span.set_attribute("research.completion_status", "completed")

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
            self._record_failed_completion(run.id, artifacts, "MAX_RUN_SECONDS budget exhausted before accepted scientific completion.", "TimeoutError")
            await emitter.emit(type="run", actor="research_harness", action="timeout", title="Research run timed out", status="failed")
        except Exception as exc:
            if self.registry.is_cancelled(run.id):
                self._mark_terminal(run.id, "cancelled", str(exc))
                await emitter.emit(type="run", actor="research_harness", action="cancel", title="Research run cancelled", status="failed")
            else:
                self._record_failed_completion(run.id, artifacts, f"{type(exc).__name__}: {exc}", type(exc).__name__)
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

    def _latest_accepted_completion(self, artifacts: ArtifactStore):
        from app.models.domain import CompletionAttempt
        for meta in reversed(artifacts.list("completion_attempt")):
            try:
                attempt = CompletionAttempt.model_validate(artifacts.get(meta["id"]))
            except Exception:
                continue
            if attempt.accepted and attempt.canonical_triples_artifact_id:
                return attempt
        return None

    def _conclusion_from_completion(self, attempt, artifacts: ArtifactStore) -> ResearchConclusion:
        triples = [CanonicalTriple.model_validate(item) for item in artifacts.get(attempt.canonical_triples_artifact_id)]
        rendered = self._render_answer(triples)
        answer_ref = artifacts.put(
            "research_answer", {"answer": rendered, "triple_artifact_id": attempt.canonical_triples_artifact_id},
            creator_agent_id="research_harness", parent_artifact_ids=[attempt.canonical_triples_artifact_id],
            artifact_id=f"answer_{attempt.canonical_triples_artifact_id}",
        )
        return ResearchConclusion(
            answer=rendered,
            answer_summary=f"{len(triples)} accepted canonical triples grounded in the verified research artifact.",
            answer_artifact_id=answer_ref.id,
            evidence_ids=sorted({evidence_id for triple in triples for evidence_id in triple.evidence_ids}),
            completion_reason="Scientific completion was independently validated by the host.",
            triple_artifact_id=attempt.canonical_triples_artifact_id,
        )

    def _record_failed_completion(self, run_id: str, artifacts: ArtifactStore, reason: str, error_type: str | None = None) -> None:
        record = FailureRecord(
            terminal_reason=reason,
            error_type=error_type,
            scientific_state=self.completion.scientific_state(artifacts),
            artifact_ids=[item["id"] for item in artifacts.list()],
        )
        artifacts.put("failure_record", record, label="Strict scientific completion failure", creator_agent_id="research_harness")
        self._mark_terminal(run_id, "failed", reason)
        self._evaluate(self.registry.get(run_id), artifacts)

    @staticmethod
    def _render_answer(triples: list[CanonicalTriple]) -> str:
        if not triples:
            return "No independently verified canonical triples were accepted."
        lines = ["Verified findings:"]
        lines.extend(f"- {t.subject} — {t.predicate} — {t.object}" for t in triples)
        return "\n".join(lines)

    def _evaluate(self, run: ExperimentRun, artifacts: ArtifactStore) -> None:
        validation_meta = artifacts.find_latest("structural_validation")
        shacl = False
        if validation_meta:
            try:
                shacl = bool(artifacts.get(validation_meta["id"]).get("shacl_conforms"))
            except Exception:
                pass
        events = self.registry.events(run.id)
        artifact_kinds = {item["kind"] for item in artifacts.list()}
        rejected_indices = [index for index, event in enumerate(events) if event.action == "completion_attempt_rejected"]
        actions_after_rejection = sum(
            event.type == "tool" and event.status == "started" and event.action != "attempt_complete_research"
            for index, event in enumerate(events)
            if rejected_indices and index > rejected_indices[-1]
        )
        started_actions = [event.action for event in events if event.type == "tool" and event.status == "started"]
        repeated_invalid = sum(left == right for left, right in zip(started_actions, started_actions[1:]))
        error_text = (run.error or "").lower()
        run.evaluation = self.evaluation.evaluate(
            run.final_triples, shacl_conforms=shacl, gold=run.gold,
            rejected_count=len(run.rejected_triples),
            tool_actions=sum(x.type == "tool" and x.status == "completed" for x in events),
            subagents_spawned=sum(x.action == "spawn_specialist" for x in events),
            retries=sum(x.type == "recovery" for x in events),
            model_escalations=sum(x.type == "recovery" and x.action == "model_escalation" for x in events),
            completion_attempts=sum(x.action in {"completion_attempt_accepted", "completion_attempt_rejected"} for x in events),
            rejected_completion_attempts=len(rejected_indices),
            actions_after_rejected_completion=actions_after_rejection,
            edc_reached=bool({"edc_open_triples", "edc_relation_definitions", "edc_canonicalization"} & artifact_kinds),
            ontology_inspected=any(x.action in {"orientation_search_ontology", "search_ontology", "inspect_ontology_term"} for x in events),
            ontology_snapshot_created="ontology_snapshot" in artifact_kinds,
            candidate_triples_produced="candidate_canonical_triples" in artifact_kinds,
            verification_reached="verification" in artifact_kinds,
            canonical_triples_produced="canonical_triples" in artifact_kinds,
            repeated_invalid_actions=repeated_invalid,
            resource_exhausted="budget exhausted" in error_text or "contextwindowexceeded" in error_text,
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

    def _persist_runtime_identity(self, run: ExperimentRun, artifacts: ArtifactStore, initial_prompt: str):
        def version(name: str) -> str | None:
            try:
                return importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                return None

        declared = {
            "runtime_id": self.settings.llm_runtime_id,
            "runtime_revision": self.settings.llm_runtime_revision,
            "context_window": self.settings.llm_context_window,
            "chat_template": self.settings.llm_chat_template,
            "reasoning_mode": self.settings.llm_reasoning_mode,
            "reasoning_parser": self.settings.llm_reasoning_parser,
            "tool_call_parser": self.settings.llm_tool_call_parser,
            "runtime_backend": self.settings.llm_runtime_backend,
            "quantization": self.settings.llm_runtime_quantization,
            "dtype": self.settings.llm_runtime_dtype,
            "tensor_parallel_size": self.settings.llm_runtime_tensor_parallel_size,
        }
        observed = {
            "python": platform.python_version(),
            "pydantic_ai": version("pydantic-ai"),
            "pydantic_ai_harness": version("pydantic-ai-harness"),
            "litellm": version("litellm"),
            "model_by_profile": {profile: self.settings.model_for_profile(profile) for profile in ("fast", "balanced", "reasoning", "critic", "extractor")},
            "temperature": self.settings.llm_temperature,
            "max_tokens": self.settings.llm_max_tokens,
        }
        identity = RunRuntimeIdentity(
            run_id=run.id,
            observed={key: value for key, value in observed.items() if value is not None},
            operator_declared={key: value for key, value in declared.items() if value is not None},
            prompt_sha256=stable_hash({"director_instructions": DIRECTOR_INSTRUCTIONS, "initial_prompt": initial_prompt}),
            tool_schema_sha256=stable_hash({
                "register_research_tools": inspect.getsource(register_research_tools),
                "output_type": "ResearchConclusion",
            }),
            output_schema_sha256=stable_hash(TypeAdapter(ResearchConclusion).json_schema()),
            document_sha256={doc_id: self.pdf.load(doc_id).sha256 for doc_id in run.document_ids},
        )
        return artifacts.put("run_runtime_identity", identity, label="Observed runtime identity", creator_agent_id="research_harness")

    def _initial_prompt(self, run: ExperimentRun, artifacts: ArtifactStore, resume: bool) -> str:
        docs = [self.pdf.load(i) for i in run.document_ids]
        inventory = "\n".join(
            f"- {d.id}: {d.metadata.title or d.filename}; {d.page_count} pages; parser={d.parse_engine}; mounted text=papers/{d.id}/document.txt; DOI={d.metadata.doi or '-'}"
            for d in docs
        )
        prior = artifacts.list()
        prior_text = "\n".join(f"- {x['id']} [{x['kind']}] {x.get('label') or ''}" for x in prior[-30:]) or "(none)"
        target = self.pdf.load(run.target_document_id or run.document_ids[0])
        return (
            f"Research question:\n{run.question}\n\nTarget paper:\n- {target.id}: {target.metadata.title or target.filename}\n\n"
            f"Selected papers (you may explicitly broaden retrieval beyond the target):\n{inventory}\n\n"
            f"Existing run artifacts {'(resume)' if resume else ''}:\n{prior_text}\n\n"
            "Inspect the research state and choose the next useful action dynamically. Generate narrowly scoped AgentSpecs when "
            "specialization, independence, critique, or decomposition would improve reliability."
        )
