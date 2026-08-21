from __future__ import annotations

from typing import Any

from app.models.domain import AgentTraceEvent
from app.storage.registry import RunRegistry
from app.telemetry.tracing import Telemetry, redact


class TraceEmitter:
    """Durably record a visible event and mirror it into OpenTelemetry.

    Hidden framework thinking is never passed to this class. Every payload is redacted
    before both SQLite/SSE persistence and telemetry export.
    """
    def __init__(self, registry: RunRegistry, run_id: str, telemetry: Telemetry):
        self.registry = registry
        self.run_id = run_id
        self.telemetry = telemetry

    async def emit(self, *, type: str, actor: str, action: str, title: str,
                   status: str = "info", summary: str = "", artifact_id: str | None = None,
                   parent_event_id: str | None = None, payload: dict[str, Any] | None = None) -> AgentTraceEvent:
        safe_payload = redact(payload or {}, max_chars=4000)
        safe_summary = str(redact(summary, max_chars=2000))
        evt = AgentTraceEvent(
            run_id=self.run_id, type=type, actor=actor, action=action, title=title,
            status=status, summary=safe_summary, artifact_id=artifact_id,
            parent_event_id=parent_event_id, payload=safe_payload,
        )
        with self.telemetry.span(f"research.event.{type}.{action}", {
            "research.run_id": self.run_id,
            "research.event_id": evt.id,
            "research.event_type": type,
            "research.actor": actor,
            "research.action": action,
            "research.status": status,
            "research.title": title,
            "research.summary": safe_summary,
            "research.artifact_id": artifact_id or "",
            "research.payload": safe_payload,
        }):
            await self.registry.emit(evt)
        return evt


def _model_event_payload(event: Any, name: str) -> dict[str, Any]:
    response = getattr(event, "response", None) or getattr(event, "model_response", None)
    usage = getattr(event, "usage", None) or getattr(response, "usage", None)
    details: dict[str, Any] = {"event_class": name}
    for key, source in (
        ("model", event), ("model_name", event), ("provider", event), ("provider_name", event),
        ("request_id", event), ("finish_reason", event), ("model_name", response),
        ("provider_response_id", response), ("finish_reason", response),
    ):
        if source is not None:
            value = getattr(source, key, None)
            if value is not None:
                details[key] = value
    if usage is not None:
        try:
            details["usage"] = usage.model_dump(mode="json")
        except Exception:
            details["usage"] = str(usage)
    return details


def make_pydantic_event_handler(registry: RunRegistry, run_id: str, telemetry: Telemetry,
                                actor_fallback: str = "agent", model_metadata: dict[str, Any] | None = None):
    """Expose model/tool/retry boundaries without exposing ThinkingPart/hidden chain-of-thought."""
    emitter = TraceEmitter(registry, run_id, telemetry)

    async def handler(ctx, event_stream):
        actor = getattr(getattr(ctx, "agent", None), "name", None) or getattr(ctx, "agent_name", None) or actor_fallback
        async for event in event_stream:
            name = type(event).__name__
            low = name.lower()
            # Never serialize ThinkingPart/ThinkingPartDelta or similarly named internal reasoning events.
            if "thinking" in low:
                continue
            part = getattr(event, "part", None)
            tool_name = getattr(event, "tool_name", None) or getattr(part, "tool_name", None) or getattr(event, "name", None)
            tool_call_id = getattr(event, "tool_call_id", None) or getattr(part, "tool_call_id", None)
            if "toolcall" in low or "functiontoolcall" in low:
                args = getattr(part, "args", None) or getattr(event, "args", None)
                await emitter.emit(
                    type="tool", actor=actor, action=str(tool_name or "tool_call"), title=f"Tool call: {tool_name or 'tool'}",
                    status="started", payload={"event_class": name, "tool_call_id": tool_call_id, "arguments": args},
                )
            elif "toolresult" in low or "functiontoolresult" in low:
                content = getattr(part, "content", None) or getattr(event, "result", None)
                await emitter.emit(
                    type="tool", actor=actor, action=str(tool_name or "tool_result"), title=f"Tool result: {tool_name or 'tool'}",
                    status="completed", payload={"event_class": name, "tool_call_id": tool_call_id, "result": content},
                )
            elif "retry" in low or "validationerror" in low:
                await emitter.emit(
                    type="recovery", actor=actor, action="framework_repair_retry", title="Bounded framework repair",
                    status="info", summary="PydanticAI requested a bounded tool/output repair.",
                    payload={"event_class": name, "details": str(event)},
                )
            elif "modelrequest" in low and "start" in low:
                await emitter.emit(
                    type="model", actor=actor, action="model_request", title="Model request started", status="started",
                    payload={**(model_metadata or {}), **_model_event_payload(event, name)},
                )
            elif "modelrequest" in low and ("end" in low or "complete" in low):
                await emitter.emit(
                    type="model", actor=actor, action="model_request", title="Model request completed", status="completed",
                    payload={**(model_metadata or {}), **_model_event_payload(event, name)},
                )

    return handler
