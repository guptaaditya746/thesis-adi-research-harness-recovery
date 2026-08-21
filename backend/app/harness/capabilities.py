from __future__ import annotations

from pathlib import Path

from app.config import Settings


def build_director_capabilities(
    settings: Settings,
    workspace: Path,
    run_id: str,
):
    """
    Build infrastructure capabilities for the Research Director.

    The Director is an orchestrator, not a scientific worker.
    It must not receive direct filesystem, shell, or code-execution
    capabilities. Scientific document access happens through typed
    research tools and bounded specialists.
    """
    capabilities: list = []
    step_store = None

    if settings.enable_step_persistence:
        try:
            from pydantic_ai_harness.step_persistence import (
                FileStepStore,
                StepPersistence,
            )

            step_store = FileStepStore(
                settings.data_dir
                / "harness_steps"
                / run_id
                / "director",
                max_snapshots_per_run=12,
            )

            capabilities.append(
                StepPersistence(
                    store=step_store,
                    agent_name="research_director",
                )
            )

        except Exception:
            step_store = None

    return capabilities, step_store