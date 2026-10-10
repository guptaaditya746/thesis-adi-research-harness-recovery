"""Small plugin hook for project-specific agent tools.

A configured module must expose ``register(agent)`` and should register typed
PydanticAI tools on the provided agent. This keeps thesis experiments out of the
core harness while preserving one observable tool surface.
"""
from __future__ import annotations

import importlib

from app.config import Settings


def register_configured_extensions(agent, settings: Settings) -> list[str]:
    loaded: list[str] = []
    for raw in settings.agent_tool_extensions.split(","):
        module_name = raw.strip()
        if not module_name:
            continue
        module = importlib.import_module(module_name)
        register = getattr(module, "register", None)
        if not callable(register):
            raise RuntimeError(f"Tool extension {module_name!r} must expose register(agent)")
        register(agent)
        loaded.append(module_name)
    return loaded
