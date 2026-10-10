from __future__ import annotations

import logging

from app.config import Settings

logger = logging.getLogger(__name__)


def configure_observability(settings: Settings) -> None:
    if settings.enable_langtrace and settings.langtrace_api_key:
        try:
            from langtrace_python_sdk import langtrace
            # The SDK performs a synchronous GitHub release lookup before setup.
            # Compute nodes can block that request, preventing backend startup.
            langtrace.check_if_sdk_is_outdated = lambda: None
            # The installed SDK currently targets a removed LiteLLM symbol; keep
            # OpenAI/PydanticAI tracing enabled while excluding that patch.
            langtrace.init(
                api_key=settings.langtrace_api_key,
                api_host=settings.langtrace_api_host,
                batch=False,
                # Avoid SDK auto-patching: the installed Langtrace release is
                # incompatible with the current LiteLLM API and can block startup.
                # Research-run spans are emitted explicitly by the harness instead.
                disable_instrumentations={"all_except": []},
            )
            logger.info("Langtrace initialized", extra={"langtrace_api_host": settings.langtrace_api_host})
        except Exception:
            logger.exception("Langtrace initialization failed")
    if settings.enable_logfire:
        try:
            import logfire
            logfire.configure()
            logfire.instrument_pydantic_ai()
        except Exception:
            logger.exception("Logfire initialization failed")
