from __future__ import annotations

import asyncio
import re
import time
from dataclasses import replace
from pathlib import Path

from pydantic_ai import Agent, UsageLimits

from app.agents.model_factory import build_model, model_metadata
from app.agents.toolsets import build_specialist_toolsets
from app.harness.deps import HarnessDeps
from app.harness.events import TraceEmitter, make_pydantic_event_handler
from app.models.domain import AgentSpec, ModelProfile, RecoveryAction, SpecialistResult


class AgentFactory:
    """Constructs isolated ephemeral PydanticAI specialists from model-generated AgentSpecs."""
    def __init__(self, deps: HarnessDeps):
        self.deps = deps
        self.emitter = TraceEmitter(deps.registry, deps.run_id)

    async def run(self, spec: AgentSpec) -> SpecialistResult:
        self.deps.ensure_not_cancelled()
        if self.deps.registry.get(self.deps.run_id).state and self.deps.registry.get(self.deps.run_id).state.delegate_calls >= self.deps.settings.max_subagent_calls:
            raise RuntimeError("MAX_SUBAGENT_CALLS budget exhausted")

        spec_ref = self.deps.artifacts.put(
            "agent_spec", spec, label=f"Dynamic specialist: {spec.role}", creator_agent_id=spec.parent_agent_id,
            parent_artifact_ids=spec.input_artifact_ids,
            model_metadata=model_metadata(self.deps.settings, spec.model_profile),
        )
        await self.emitter.emit(
            type="delegate", actor=spec.parent_agent_id, action="spawn_specialist", title=f"Spawned {spec.role}", status="started",
            artifact_id=spec_ref.id,
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
                        event_stream_handler=make_pydantic_event_handler(self.deps.registry, self.deps.run_id, spec.id),
                        usage_limits=UsageLimits(
                            request_limit=max(2, spec.max_tool_calls + self.deps.settings.llm_output_retries + 2),
                            tool_calls_limit=max(1, spec.max_tool_calls),
                        ),
                        conversation_id=f"{self.deps.run_id}:{spec.id}",
                    )
                    span.set_attribute("agent.output", str(result.output.model_dump(mode="json")))
                output = result.output
                out_ref = self.deps.artifacts.put(
                    "specialist_output", output, label=f"{spec.role} output", creator_agent_id=spec.id,
                    parent_artifact_ids=[spec_ref.id, *spec.input_artifact_ids],
                    model_metadata=model_metadata(self.deps.settings, profile, attempt),
                )
                output.artifact_ids = list(dict.fromkeys([*output.artifact_ids, out_ref.id]))
                await self.emitter.emit(
                    type="delegate", actor=spec.id, action="specialist_complete", title=f"{spec.role} completed",
                    status="completed", artifact_id=out_ref.id,
                    summary=output.summary,
                    payload={"duration_ms": round((time.perf_counter() - started) * 1000, 2), "output": output.model_dump(mode="json")},
                )
                self._remove_active(spec.id)
                return output
            except Exception as exc:
                last_error = exc
                await self._record_recovery(spec, profile, attempt, exc, profiles[attempt + 1] if attempt + 1 < len(profiles) else None)
                if attempt + 1 < len(profiles):
                    await asyncio.sleep(min(2 ** attempt, 4))
                    continue
                break
        self._remove_active(spec.id)
        # Graceful partial specialist result; the Director can change strategy or create a different specialist.
        partial = SpecialistResult(
            summary=f"Specialist {spec.role} could not complete reliably.",
            unresolved_questions=[spec.objective],
            recommended_next_actions=["Narrow the task or evidence and re-delegate", "Use an independent specialist with different tools"],
        )
        ref = self.deps.artifacts.put(
            "specialist_partial", partial, label=f"Partial: {spec.role}", creator_agent_id=spec.id,
            parent_artifact_ids=[spec_ref.id],
        )
        partial.artifact_ids.append(ref.id)
        await self.emitter.emit(type="delegate", actor=spec.id, action="specialist_partial", title=f"{spec.role} returned partial result",
                                status="failed", summary=str(last_error or "unknown failure"), artifact_id=ref.id)
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
            toolsets=build_specialist_toolsets(self.deps, spec.required_capabilities, spec.id),
            capabilities=capabilities,
            retries={"tools": self.deps.settings.llm_output_retries, "output": self.deps.settings.llm_output_retries},
            model_settings={"temperature": self.deps.settings.llm_temperature, "max_tokens": self.deps.settings.llm_max_tokens},
        )

    async def run_fanout(self, specs: list[AgentSpec]) -> list[SpecialistResult]:
        """Host-enforced dynamic fan-out over runtime-generated specialist specs."""
        if not specs:
            return []
        remaining = self.deps.settings.max_subagent_calls - (self.deps.registry.get(self.deps.run_id).state.delegate_calls if self.deps.registry.get(self.deps.run_id).state else 0)
        specs = specs[:max(0, remaining)]
        return await asyncio.gather(*(self.run(s) for s in specs))

    def build_workflow_capability(self, specs: list[AgentSpec]):
        """Build Harness DynamicWorkflow from runtime-created specialists when scripted fan-out/chaining is useful."""
        if not self.deps.settings.enable_dynamic_workflow:
            return None
        from pydantic_ai_harness.dynamic_workflow import DynamicWorkflow
        agents = [self.build(s) for s in specs]
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
        chars = 0
        for aid in spec.input_artifact_ids:
            try:
                content = self.deps.artifacts.get(aid)
            except KeyError:
                content = {"error": "artifact not found"}
            text = str(content)
            remain = max(0, spec.context_budget - chars)
            if remain <= 0:
                break
            if len(text) > remain:
                text = text[:remain] + f"… [artifact {aid} truncated in prompt; inspect it with artifact tool]"
            artifacts.append(f"ARTIFACT {aid}:\n{text}")
            chars += len(text)
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
            "Do not expose hidden chain-of-thought; provide concise decisions/findings and artifact references. "
            f"Role-specific instructions: {spec.instructions}"
        )

    async def _record_recovery(self, spec: AgentSpec, profile: ModelProfile, attempt: int, exc: Exception, next_profile: ModelProfile | None) -> None:
        level = 8 if next_profile and next_profile != profile else 3
        action = RecoveryAction(
            level=level,
            action="model_escalation" if level == 8 else "bounded_retry",
            trigger=f"{type(exc).__name__}: {exc}",
            agent_id=spec.id,
            from_profile=profile,
            to_profile=next_profile,
            retry_number=attempt,
            success=False,
        )
        ref = self.deps.artifacts.put("recovery", action, creator_agent_id=spec.id)
        await self.emitter.emit(type="recovery", actor=spec.id, action=action.action, title="Recovery action",
                                status="info", summary=action.trigger, artifact_id=ref.id,
                                payload=action.model_dump(mode="json"))

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
