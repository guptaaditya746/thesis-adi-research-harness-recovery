from __future__ import annotations

from pathlib import Path

from app.config import Settings


def build_director_capabilities(settings: Settings, workspace: Path, run_id: str):
    """Generic Harness capabilities only; research sequencing remains model-directed."""
    capabilities: list = []
    step_store = None
    try:
        from pydantic_ai_harness import FileSystem
        capabilities.append(FileSystem(root_dir=workspace, protected_patterns=[]))
    except Exception:
        pass
    if settings.enable_agent_shell:
        try:
            from pydantic_ai_harness import Shell
            commands = list(settings.shell_allowed_commands)
            if settings.enable_agent_bash and "bash" not in commands:
                commands.append("bash")
            capabilities.append(Shell(
                cwd=workspace, allowed_commands=commands, denied_commands=[], default_timeout=settings.agent_shell_timeout_seconds,
                max_output_chars=settings.max_tool_output_chars, persist_cwd=False, allow_interactive=False,
                denied_env_patterns=["*API_KEY*", "*TOKEN*", "*SECRET*", "*PASSWORD*", "*CREDENTIAL*"],
            ))
        except Exception:
            pass
    if settings.enable_step_persistence:
        try:
            from pydantic_ai_harness.step_persistence import FileStepStore, StepPersistence
            step_store = FileStepStore(settings.data_dir / "harness_steps" / run_id / "director", max_snapshots_per_run=12)
            capabilities.append(StepPersistence(store=step_store, agent_name="research_director"))
        except Exception:
            step_store = None
    return capabilities, step_store
