from __future__ import annotations

import asyncio
import re
import time
from dataclasses import replace
from pathlib import Path

from pydantic_ai import Agent, UsageLimits
from pydantic_ai.exceptions import UnexpectedModelBehavior

from app.agents.model_factory import build_model, model_metadata
from app.agents.routing import TaskModelRouter
from app.agents.toolsets import build_specialist_toolsets
from app.harness.contracts import (
    ArtifactIntegrityError,
    ArtifactNotFound,
    CorrectableToolInputError,
    InvalidArtifactKind,
    ScientificStateBlocked,
    validate_artifact_schema,
    validate_specialist_spec,
)
from app.harness.deps import HarnessDeps
from app.harness.events import TraceEmitter, make_pydantic_event_handler
from app.models.domain import AgentSpec, ArtifactRef, FanoutExecution, ModelProfile, ModelRoutingDecision, RecoveryAction, SpecialistResult, stable_hash


def effective_tool_call_limit(spec: AgentSpec) -> int:
    """Keep retrieval specialists viable when a model underestimates their budget."""
    minimum = 12 if spec.task_scope in {"document_analysis", "evidence_selection"} else 1
    return max(minimum, spec.max_tool_calls)


class AgentFactory:
    """Constructs isolated ephemeral PydanticAI specialists from model-generated AgentSpecs."""
    def __init__(self, deps: HarnessDeps):
        self.deps = deps
        self.emitter = TraceEmitter(deps.registry, deps.run_id, deps.telemetry)
        self.router = TaskModelRouter()

    async def run(self, spec: AgentSpec, *, parent_event_id: str | None = None) -> SpecialistResult:
        self.deps.ensure_not_cancelled()
        try:
            validate_specialist_spec(self.deps, spec)
        except CorrectableToolInputError as exc:
            return self._blocked_result(spec, exc, status="partial")
        except ScientificStateBlocked as exc:
            return self._blocked_result(spec, exc)
        except (ArtifactNotFound, InvalidArtifactKind, ArtifactIntegrityError) as exc:
            return self._blocked_result(spec, exc)
        if self._same_task_failure_count(spec) >= self.deps.settings.max_specialist_failures_per_task:
            return self._blocked_result(spec, ScientificStateBlocked(
                "Unchanged specialist task already failed with these artifact inputs; change the inputs or strategy.",
            ))
        if self.deps.registry.get(self.deps.run_id).state and self.deps.registry.get(self.deps.run_id).state.delegate_calls >= self.deps.settings.max_subagent_calls:
            raise RuntimeError("MAX_SUBAGENT_CALLS budget exhausted")

        spec = await self._apply_model_route(spec, parent_event_id=parent_event_id)

        spec_ref = self.deps.artifacts.put(
            "agent_spec", spec, label=f"Dynamic specialist: {spec.role}", creator_agent_id=spec.parent_agent_id,
            parent_artifact_ids=spec.input_artifact_ids,
            model_metadata=model_metadata(self.deps.settings, spec.model_profile),
        )
        spawn_event = await self.emitter.emit(
            type="delegate", actor=spec.parent_agent_id, action="spawn_specialist", title=f"Spawned {spec.role}", status="started",
            artifact_id=spec_ref.id, parent_event_id=parent_event_id,
            summary=spec.creation_reason,
            payload={
                "agent_spec": spec.model_dump(mode="json"),
                "generated_prompt": spec.instructions,
                "tools_granted": spec.required_capabilities,
                "input_artifact_ids": spec.input_artifact_ids,
                "model_profile": spec.model_profile,
            },
        )
        run = self.deps.registry.get(self.deps.run_id)
        if run.state:
            run.state.delegate_calls += 1
            run.state.active_agent_ids.append(spec.id)
            self.deps.registry.persist(run)

        prompt = self._task_prompt(spec)
        await self.emitter.emit(
            type="message", actor=spec.parent_agent_id, action="specialist_task_message", title=f"Task sent to {spec.role}",
            parent_event_id=spawn_event.id,
            payload={"recipient_agent_id": spec.id, "task_message": prompt},
        )

        profiles = self._profile_chain(spec.model_profile)
        last_error: Exception | None = None
        for attempt, profile in enumerate(profiles):
            effective_spec = spec.model_copy(update={"model_profile": profile})
            agent = self.build(effective_spec)
            started = time.perf_counter()
            try:
                with self.deps.telemetry.span("agent.specialist", {
                    "research.run_id": self.deps.run_id, "agent.id": spec.id, "agent.role": spec.role,
                    "agent.model_profile": profile.value, "agent.prompt": prompt,
                    "agent.tools": spec.required_capabilities, "agent.retry": attempt,
                }) as span:
                    result = await agent.run(
                        prompt,
                        event_stream_handler=make_pydantic_event_handler(
                            self.deps.registry, self.deps.run_id, self.deps.telemetry, spec.id,
                            parent_event_id=spawn_event.id, lineage=self.deps.trace_lineage, lineage_scope=spec.id,
                        ),
                        usage_limits=UsageLimits(
                            request_limit=max(
                                2,
                                effective_tool_call_limit(spec) + self.deps.settings.llm_output_retries + 2,
                            ),
                            tool_calls_limit=effective_tool_call_limit(spec),
                        ),
                        conversation_id=f"{self.deps.run_id}:{spec.id}",
                    )
                    span.set_attribute("agent.output", str(result.output.model_dump(mode="json")))
                output = self._validated_output(spec, result.output)
                out_ref = self.deps.artifacts.put(
                    "specialist_output", output, label=f"{spec.role} output", creator_agent_id=spec.id,
                    parent_artifact_ids=[spec_ref.id, *spec.input_artifact_ids],
                    model_metadata=model_metadata(self.deps.settings, profile, attempt),
                )
                output.artifact_ids = list(dict.fromkeys([*output.artifact_ids, out_ref.id]))
                self._record_validated_output_state(output.artifact_ids)
                await self.emitter.emit(
                    type="delegate", actor=spec.id, action="specialist_complete", title=f"{spec.role} completed",
                    status="completed", artifact_id=out_ref.id, parent_event_id=spawn_event.id,
                    summary=output.summary,
                    payload={"duration_ms": round((time.perf_counter() - started) * 1000, 2), "output": output.model_dump(mode="json")},
                )
                self._remove_active(spec.id)
                return output
            except Exception as exc:
                last_error = exc
                correctable = self._is_correctable_tool_input_failure(exc)
                invalid_artifact_reference = isinstance(exc, (ArtifactNotFound, InvalidArtifactKind, ArtifactIntegrityError))
                scientific_failure = isinstance(exc, (ScientificStateBlocked, ArtifactNotFound, InvalidArtifactKind, ArtifactIntegrityError))
                generation_budget_failure = self._is_generation_budget_failure(exc)
                # Correctable tool inputs need a reread/correct action from the
                # Director, not another model profile. Keep the recovery record
                # truthful and avoid counting this as a model escalation.
                next_profile = None if (correctable or scientific_failure or generation_budget_failure) else (
                    profiles[attempt + 1] if attempt + 1 < len(profiles) else None
                )
                await self._record_recovery(
                    spec, profile, attempt, exc, next_profile,
                    correctable=correctable, generation_budget_failure=generation_budget_failure,
                    invalid_artifact_reference=invalid_artifact_reference,
                    parent_event_id=spawn_event.id,
                )
                self._record_failure(spec, exc)
                if correctable or scientific_failure or generation_budget_failure:
                    break
                if attempt + 1 < len(profiles):
                    await asyncio.sleep(min(2 ** attempt, 4))
                    continue
                break
        self._remove_active(spec.id)
        valid_outputs = self._created_artifact_ids(spec.id)
        invalid_reference = isinstance(last_error, (ArtifactNotFound, InvalidArtifactKind, ArtifactIntegrityError))
        # Graceful partial specialist result; the Director can change strategy or create a different specialist.
        partial = SpecialistResult(
            task_status="partial", task_scope=spec.task_scope,
            summary=f"Specialist {spec.role} could not complete reliably.",
            unresolved_questions=[spec.objective],
            recommended_next_actions=(
                ["Do not re-delegate this unchanged task; inspect persisted artifacts and correct the handoff IDs."] if invalid_reference
                else ["Narrow the task or evidence and re-delegate", "Use an independent specialist with different tools"]),
        )
        ref = self.deps.artifacts.put(
            "specialist_partial", partial, label=f"Partial: {spec.role}", creator_agent_id=spec.id,
            parent_artifact_ids=[spec_ref.id, *valid_outputs],
        )
        partial.artifact_ids = [*valid_outputs, ref.id]
        await self.emitter.emit(type="delegate", actor=spec.id, action="specialist_partial", title=f"{spec.role} returned partial result",
                                status="failed", summary=str(last_error or "unknown failure"), artifact_id=ref.id,
                                parent_event_id=spawn_event.id)
        return partial

    _EXPECTED_OUTPUT_KIND = {
        "evidence_selection": "evidence_bundle",
        "ontology_snapshot": "ontology_snapshot",
        "edc_extract": "edc_open_triples",
        "edc_define": "edc_relation_definitions",
        "edc_canonicalize": "candidate_canonical_triples",
        "verification": "canonical_triples",
    }
    _NON_HANDOFF_KINDS = {
        "agent_spec", "model_routing_decision", "recovery", "specialist_failure",
        "specialist_output", "specialist_partial", "specialist_blocked", "specialist_input_error",
    }

    def _validated_output(self, spec: AgentSpec, output: SpecialistResult) -> SpecialistResult:
        claimed_ids = list(dict.fromkeys(output.artifact_ids))
        for artifact_id in claimed_ids:
            validate_artifact_schema(self.deps, artifact_id)

        created_ids = self._created_artifact_ids(spec.id)
        artifact_ids = list(dict.fromkeys([*claimed_ids, *created_ids]))
        expected_kind = self._EXPECTED_OUTPUT_KIND.get(spec.task_scope)
        if output.task_status == "completed" and expected_kind and not any(
            self.deps.artifacts.get_metadata(artifact_id).artifact_type == expected_kind
            for artifact_id in artifact_ids
        ):
            raise ArtifactIntegrityError(
                f"Completed {spec.task_scope} specialist returned no persisted {expected_kind} artifact."
            )
        return output.model_copy(update={"artifact_ids": artifact_ids})

    def _created_artifact_ids(self, agent_id: str) -> list[str]:
        return [
            item["id"] for item in self.deps.artifacts.list()
            if item.get("creator_agent_id") == agent_id and item["kind"] not in self._NON_HANDOFF_KINDS
        ]

    def _record_validated_output_state(self, artifact_ids: list[str]) -> None:
        """Mirror validated references into durable operational state.

        ArtifactStore remains authoritative; this state is only a compact index
        for the director and UI.
        """
        run = self.deps.registry.get(self.deps.run_id)
        if run.state is None:
            return
        refs = {ref.id: ref for ref in getattr(run.state, "artifact_refs", [])}
        for artifact_id in artifact_ids:
            meta = self.deps.artifacts.get_metadata(artifact_id)
            refs[artifact_id] = ArtifactRef(id=artifact_id, kind=meta.artifact_type)
            if meta.artifact_type == "ontology_snapshot":
                run.state.ontology_snapshot_id = artifact_id
            if meta.artifact_type in {"verification", "structural_validation"}:
                run.state.verification_artifact_ids = list(dict.fromkeys([
                    *run.state.verification_artifact_ids, artifact_id,
                ]))
        run.state.artifact_refs = list(refs.values())
        self.deps.registry.persist(run)

    @staticmethod
    def _task_fingerprint(spec: AgentSpec) -> str:
        return stable_hash({
            "task_scope": spec.task_scope,
            "required_capabilities": sorted(spec.required_capabilities),
            "input_artifact_ids": sorted(spec.input_artifact_ids),
        })

    def _same_task_failure_count(self, spec: AgentSpec) -> int:
        fingerprint = self._task_fingerprint(spec)
        return sum(
            self.deps.artifacts.get(item["id"]).get("task_fingerprint") == fingerprint
            for item in self.deps.artifacts.list("specialist_failure")
        )

    def _record_failure(self, spec: AgentSpec, exc: Exception) -> None:
        self.deps.artifacts.put(
            "specialist_failure",
            {
                "task_fingerprint": self._task_fingerprint(spec),
                "task_scope": spec.task_scope,
                "input_artifact_ids": sorted(spec.input_artifact_ids),
                "error_type": type(exc).__name__,
                "reason": str(exc),
            },
            creator_agent_id=spec.id,
            parent_artifact_ids=[
                artifact_id for artifact_id in spec.input_artifact_ids
                if any(item["id"] == artifact_id for item in self.deps.artifacts.list())
            ],
        )

    def _blocked_result(self, spec: AgentSpec, exc: Exception, *, status: str = "blocked") -> SpecialistResult:
        requirements = tuple(getattr(exc, "requirements", ()))
        partial = SpecialistResult(
            task_status=status, task_scope=spec.task_scope,
            summary=f"Specialist {spec.role} is blocked by scientific state.",
            unresolved_questions=[spec.objective],
            recommended_next_actions=["Return control to the Research Director and create the missing state."],
            blocked_requirements=list(requirements),
            blocked_reason=str(exc),
        )
        ref = self.deps.artifacts.put(
            "specialist_blocked" if status == "blocked" else "specialist_input_error", partial,
            label=f"{status.title()}: {spec.role}", creator_agent_id=spec.id,
            parent_artifact_ids=spec.input_artifact_ids,
        )
        partial.artifact_ids.append(ref.id)
        return partial

    def build(self, spec: AgentSpec) -> Agent:
        name = self._agent_name(spec)
        workspace = self.deps.settings.data_dir / "runs" / self.deps.run_id / "workspace"
        capabilities = self._harness_capabilities(spec, workspace)
        return Agent(
            build_model(self.deps.settings, spec.model_profile),
            output_type=SpecialistResult,
            name=name,
            description=spec.objective[:400],
            instructions=self._specialist_instructions(spec),
            toolsets=build_specialist_toolsets(self.deps, spec.required_capabilities, spec.id, spec.input_artifact_ids),
            capabilities=capabilities,
            retries={"tools": self.deps.settings.llm_output_retries, "output": self.deps.settings.llm_output_retries},
            model_settings={"temperature": self.deps.settings.llm_temperature, "max_tokens": self.deps.settings.llm_max_tokens},
        )

    async def run_fanout(self, specs: list[AgentSpec], *, parent_event_id: str | None = None) -> list[SpecialistResult]:
        """Host-enforced dynamic fan-out over runtime-generated specialist specs."""
        if not specs:
            return []
        requested_specs = list(specs)
        remaining = self.deps.settings.max_subagent_calls - (self.deps.registry.get(self.deps.run_id).state.delegate_calls if self.deps.registry.get(self.deps.run_id).state else 0)
        specs = specs[:max(0, remaining)]
        started = time.perf_counter()
        outputs = await asyncio.gather(*(self.run(s, parent_event_id=parent_event_id) for s in specs))
        execution = FanoutExecution(
            requested_specialist_ids=[spec.id for spec in requested_specs],
            dispatched_specialist_ids=[spec.id for spec in specs],
            task_scopes=[spec.task_scope for spec in specs],
            elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
            result_statuses=[output.task_status for output in outputs],
            result_artifact_ids=list(dict.fromkeys(
                artifact_id for output in outputs for artifact_id in output.artifact_ids
            )),
        )
        ref = self.deps.artifacts.put(
            "fanout_execution", execution, label="Concurrent specialist fan-out", creator_agent_id="research_director",
            parent_artifact_ids=execution.result_artifact_ids,
        )
        await self.emitter.emit(
            type="delegate", actor="research_director", action="specialist_fanout", title="Specialist fan-out completed",
            status="completed", artifact_id=ref.id, parent_event_id=parent_event_id,
            summary=f"Ran {len(specs)} independent specialists concurrently in {execution.elapsed_ms} ms.",
            payload=execution.model_dump(mode="json"),
        )
        return outputs

    async def _apply_model_route(self, spec: AgentSpec, *, parent_event_id: str | None = None) -> AgentSpec:
        route = getattr(self, "router", TaskModelRouter()).route_specialist(spec.task_scope, spec.model_profile)
        decision = ModelRoutingDecision(
            requested_profile=route.requested_profile,
            effective_profile=route.effective_profile,
            task_scope=spec.task_scope,
            reason=route.reason,
            requested_model=self.deps.settings.model_for_profile(route.requested_profile.value),
            effective_model=self.deps.settings.model_for_profile(route.effective_profile.value),
        )
        ref = self.deps.artifacts.put(
            "model_routing_decision", decision, label=f"Model route: {spec.task_scope}",
            creator_agent_id="research_director", parent_artifact_ids=spec.input_artifact_ids,
        )
        await self.emitter.emit(
            type="decision", actor="research_director", action="model_routing", title="Task model route",
            artifact_id=ref.id, summary=route.reason, parent_event_id=parent_event_id,
            payload=decision.model_dump(mode="json"),
        )
        return spec.model_copy(update={"model_profile": route.effective_profile})

    def _route_spec(self, spec: AgentSpec) -> AgentSpec:
        """Pure route resolution for non-standard entry points such as workflows."""
        route = getattr(self, "router", TaskModelRouter()).route_specialist(spec.task_scope, spec.model_profile)
        return spec.model_copy(update={"model_profile": route.effective_profile})

    def build_workflow_capability(self, specs: list[AgentSpec]):
        """Build Harness DynamicWorkflow from runtime-created specialists when scripted fan-out/chaining is useful."""
        for spec in specs:
            validate_specialist_spec(self.deps, spec)
        if not self.deps.settings.enable_dynamic_workflow:
            return None
        from pydantic_ai_harness.dynamic_workflow import DynamicWorkflow
        agents = [self.build(self._route_spec(s)) for s in specs]
        return DynamicWorkflow(agents=agents, max_agent_calls=min(self.deps.settings.max_subagent_calls, max(1, len(specs) * 3)))

    def _harness_capabilities(self, spec: AgentSpec, workspace: Path) -> list:
        capabilities: list = []
        try:
            from pydantic_ai_harness import FileSystem
            if "filesystem" in spec.required_capabilities:
                capabilities.append(FileSystem(root_dir=workspace, protected_patterns=[]))
        except Exception:
            pass
        if self.deps.settings.enable_agent_shell and "shell" in spec.required_capabilities:
            try:
                from pydantic_ai_harness import Shell
                commands = list(self.deps.settings.shell_allowed_commands)
                if self.deps.settings.enable_agent_bash and "bash" not in commands:
                    commands.append("bash")
                capabilities.append(Shell(
                    cwd=workspace, allowed_commands=commands, denied_commands=[],
                    default_timeout=self.deps.settings.agent_shell_timeout_seconds,
                    max_output_chars=self.deps.settings.max_tool_output_chars,
                    persist_cwd=False, allow_interactive=False,
                    denied_env_patterns=["*API_KEY*", "*TOKEN*", "*SECRET*", "*PASSWORD*", "*CREDENTIAL*"],
                ))
            except Exception:
                pass
        if self.deps.settings.enable_step_persistence:
            try:
                from pydantic_ai_harness.step_persistence import FileStepStore, StepPersistence
                store = FileStepStore(self.deps.settings.data_dir / "harness_steps" / self.deps.run_id / spec.id, max_snapshots_per_run=8)
                capabilities.append(StepPersistence(store=store, agent_name=self._agent_name(spec)))
            except Exception:
                pass
        return capabilities

    def _task_prompt(self, spec: AgentSpec) -> str:
        artifacts = []
        for aid in spec.input_artifact_ids:
            try:
                envelope = self.deps.artifacts.get_envelope(aid)
            except KeyError:
                artifacts.append(f"ARTIFACT {aid}: unavailable; use the typed artifact tool to inspect it")
                continue
            parents = ", ".join(envelope.parent_artifact_ids) or "none"
            artifacts.append(
                f"ARTIFACT {aid}: kind={envelope.artifact_type}; created_by={envelope.creator_agent_id}; "
                f"parents={parents}. Scientific payload is not embedded here; inspect it with the granted typed tools."
            )
        return (
            f"Specialist role: {spec.role}\nObjective: {spec.objective}\n\n"
            f"Generated task instructions:\n{spec.instructions}\n\n"
            f"Input artifact context:\n" + ("\n\n".join(artifacts) if artifacts else "(none; use granted tools)") +
            "\n\nReturn the typed SpecialistResult. State unresolved uncertainty; never invent success."
        )

    @staticmethod
    def _specialist_instructions(spec: AgentSpec) -> str:
        return (
            "You are an ephemeral specialist inside a scientific research harness. You receive isolated narrow context, not the "
            "Research Director's conversation. Use only granted capabilities. Evidence IDs and artifact IDs are scientific provenance. "
            "Artifact IDs are backend-issued: report only exact IDs returned by tools; never invent a placeholder or kind name as an ID. "
            "Do not expose hidden chain-of-thought; provide concise decisions/findings and artifact references. "
            "task_status describes completion of your assigned task_scope only, not completion of the overall research run. "
            f"Role-specific instructions: {spec.instructions}"
        )

    async def _record_recovery(
        self, spec: AgentSpec, profile: ModelProfile, attempt: int, exc: Exception,
        next_profile: ModelProfile | None, *, correctable: bool = False, generation_budget_failure: bool = False,
        invalid_artifact_reference: bool = False,
        parent_event_id: str | None = None,
    ) -> None:
        level = 3
        action_name = (
            "invalid_artifact_reference" if invalid_artifact_reference
            else "generation_token_limit" if generation_budget_failure
            else "correct_tool_input" if correctable else "bounded_retry"
        )
        if next_profile and next_profile != profile:
            level = 8
            action_name = "model_escalation"
        action = RecoveryAction(
            level=level,
            action=action_name,
            trigger=f"{type(exc).__name__}: {exc}",
            agent_id=spec.id,
            from_profile=profile,
            to_profile=next_profile,
            retry_number=attempt,
            success=False,
        )
        ref = self.deps.artifacts.put("recovery", action, creator_agent_id=spec.id)
        await self.emitter.emit(type="recovery", actor=spec.id, action=action.action, title="Recovery action",
                                status="info", summary=action.trigger, artifact_id=ref.id, parent_event_id=parent_event_id,
                                payload=action.model_dump(mode="json"))

    @staticmethod
    def _is_correctable_tool_input_failure(exc: Exception) -> bool:
        """Keep source/input contract failures on the reread/correct path, not model escalation."""
        seen: set[int] = set()
        current: BaseException | None = exc
        markers = ("source-resolution", "source resolution", "candidate_id", "EvidenceSpan", "preview text", "input_artifact_ids")
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if isinstance(current, (ArtifactNotFound, InvalidArtifactKind, CorrectableToolInputError)):
                return True
            if isinstance(current, ValueError) and any(marker.lower() in str(current).lower() for marker in markers):
                return True
            current = current.__cause__ or current.__context__
        return False

    @staticmethod
    def _is_generation_budget_failure(exc: Exception) -> bool:
        """A provider length stop with no usable output cannot improve by profile escalation.

        Pydantic AI raises this after the provider returns ``finish_reason='length'``
        with empty/blank/thinking-only output. In this harness every profile shares
        the same configured ``max_tokens``, so retrying the same task only repeats
        the identical cap. The Director must narrow or redesign the task instead.
        """
        return isinstance(exc, UnexpectedModelBehavior) and "Model token limit (" in str(exc) and "before any response was generated" in str(exc)

    def _remove_active(self, agent_id: str) -> None:
        run = self.deps.registry.get(self.deps.run_id)
        if run.state:
            run.state.active_agent_ids = [x for x in run.state.active_agent_ids if x != agent_id]
            self.deps.registry.persist(run)

    @staticmethod
    def _profile_chain(start: ModelProfile) -> list[ModelProfile]:
        order = [ModelProfile.FAST, ModelProfile.BALANCED, ModelProfile.REASONING, ModelProfile.CRITIC]
        if start == ModelProfile.EXTRACTOR:
            return [ModelProfile.EXTRACTOR, ModelProfile.BALANCED, ModelProfile.REASONING]
        idx = order.index(start) if start in order else 1
        return order[idx: min(len(order), idx + 3)]

    @staticmethod
    def _agent_name(spec: AgentSpec) -> str:
        value = re.sub(r"[^A-Za-z0-9_]", "_", f"{spec.role}_{spec.id}")
        if value and value[0].isdigit():
            value = "a_" + value
        return value[:80]
