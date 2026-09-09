from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.harness.context_management import ResearchContextManager


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

    # These are environment protections, not research orchestration. Tool payloads
    # are bounded before they enter durable chat history; ArtifactStore remains the
    # authoritative place for full scientific payloads.
    from pydantic_ai_harness.compaction import ClampOversizedMessages, WarnNearLimits
    from pydantic_ai_harness.tool_output_limits import Band, ToolOutputLimits, Truncate, TruncationStrategy

    capabilities.extend([
        ToolOutputLimits(bands=[Band(
            over=settings.context_tool_result_max_chars,
            action=Truncate(
                max_chars=settings.context_tool_result_max_chars,
                strategy=TruncationStrategy.head_tail,
            ),
        )]),
        ClampOversizedMessages(
            max_part_chars=settings.context_clamp_part_chars,
            keep_head_chars=min(3_000, settings.context_clamp_part_chars // 3),
            keep_tail_chars=min(3_000, settings.context_clamp_part_chars // 3),
        ),
        ResearchContextManager(settings),
        # Warn about request exhaustion. Context itself is proactively handled
        # above, before a provider-side request can be rejected.
        WarnNearLimits(
            max_iterations=settings.max_agent_steps,
            warn_on=["iterations"],
            warning_threshold=settings.context_warn_fraction,
        ),
    ])

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
