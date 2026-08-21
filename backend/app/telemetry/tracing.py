from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from app.config import Settings

_SECRET_KEY = re.compile(r"(api[_-]?key|authorization|token|secret|password|credential)", re.IGNORECASE)
_SECRET_VALUE = re.compile(r"(?i)(bearer\s+[A-Za-z0-9._~+\-/=]+|sk-[A-Za-z0-9_-]{8,})")
logger = logging.getLogger(__name__)


def redact(value: Any, key: str | None = None, *, max_chars: int = 4000) -> Any:
    """Redact secrets and bound telemetry payloads before export."""
    if key and _SECRET_KEY.search(key):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {str(k): redact(v, str(k), max_chars=max_chars) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v, max_chars=max_chars) for v in value[:100]]
    if value is None or isinstance(value, (int, float, bool)):
        return value
    text = _SECRET_VALUE.sub("[REDACTED]", str(value))
    if len(text) > max_chars:
        digest = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
        return f"{text[:max_chars]}… [truncated sha256={digest}]"
    return text


class _NoopSpan:
    def set_attribute(self, *_args, **_kwargs):
        return None
    def add_event(self, *_args, **_kwargs):
        return None
    def record_exception(self, *_args, **_kwargs):
        return None
    def set_status(self, *_args, **_kwargs):
        return None
    def get_span_context(self):
        class C:
            trace_id = 0
            span_id = 0
        return C()


class Telemetry:
    """Application-boundary OpenTelemetry wrapper.

    Initialization/export is best-effort by design: a telemetry outage must never fail a research run.
    The application creates one instance and injects it into the harness and event emitters.
    Langtrace is initialized first when configured; explicit OTLP export can be configured in parallel.
    """
    def __init__(self, settings: Settings):
        self.settings = settings
        self.tracer = None
        self.provider = None
        self._owns_provider = False
        self._shutdown = False
        self.initialization_status: dict[str, Any] = {
            "enabled": bool(settings.enable_langtrace and settings.langtrace_api_key),
            "initialized": False,
            "provider": None,
            "processor_count": 0,
            "exporter": None,
            "error": None,
        }
        self._initialize()

    @staticmethod
    def _processor_count(provider: Any) -> int:
        active = getattr(provider, "_active_span_processor", None)
        processors = getattr(active, "_span_processors", None)
        return len(processors) if processors is not None else 0

    @staticmethod
    def _exporter_name(provider: Any) -> str | None:
        active = getattr(provider, "_active_span_processor", None)
        processors = getattr(active, "_span_processors", None) or []
        names: list[str] = []
        for processor in processors:
            exporter = getattr(processor, "_exporter", None)
            if exporter is None:
                exporter = getattr(getattr(processor, "_batch_processor", None), "_exporter", None)
            if exporter is not None:
                names.append(f"{type(exporter).__module__}.{type(exporter).__name__}")
        return ", ".join(names) or None

    def _initialize(self) -> None:
        try:
            from opentelemetry import trace

            if self.settings.enable_langtrace and self.settings.langtrace_api_key:
                from langtrace_python_sdk import langtrace

                # The SDK defaults Sentry error reporting to an enabled network
                # client. Keep telemetry startup bounded unless explicitly opted in.
                os.environ.setdefault("LANGTRACE_ERROR_REPORTING", "False")
                kwargs: dict[str, Any] = {"api_key": self.settings.langtrace_api_key}
                if self.settings.langtrace_api_host:
                    kwargs["api_host"] = self.settings.langtrace_api_host
                if not self.settings.enable_langtrace_auto_instrumentation:
                    kwargs["disable_instrumentations"] = {"all_except": []}
                # Keep this exception visible: telemetry remains optional, but a silent
                # failure makes a valid local trace ID look like remote ingestion.
                langtrace.init(**kwargs)
                self._owns_provider = True

            if self.settings.otel_exporter_otlp_endpoint:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
                from opentelemetry.sdk.resources import Resource
                from opentelemetry.sdk.trace import TracerProvider
                from opentelemetry.sdk.trace.export import BatchSpanProcessor

                provider = trace.get_tracer_provider()
                if provider.__class__.__module__.startswith("opentelemetry.trace"):
                    provider = TracerProvider(resource=Resource.create({"service.name": self.settings.otel_service_name}))
                    headers = None
                    if self.settings.otel_exporter_otlp_headers:
                        headers = dict(
                            part.split("=", 1)
                            for part in self.settings.otel_exporter_otlp_headers.split(",")
                            if "=" in part
                        )
                    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(
                        endpoint=self.settings.otel_exporter_otlp_endpoint, headers=headers
                    )))
                    trace.set_tracer_provider(provider)
                    self._owns_provider = True

            self.provider = trace.get_tracer_provider()
            self.tracer = trace.get_tracer(self.settings.otel_service_name)
            self.initialization_status.update({
                "initialized": self._processor_count(self.provider) > 0,
                "provider": f"{type(self.provider).__module__}.{type(self.provider).__name__}",
                "processor_count": self._processor_count(self.provider),
                "exporter": self._exporter_name(self.provider),
            })
            if not self.initialization_status["initialized"] and self.initialization_status["enabled"]:
                self.initialization_status["error"] = "No span processor attached to the configured telemetry provider"
                logger.error("Telemetry initialized without an exporter/processor; research remains non-fatal")
        except Exception as exc:  # telemetry must not block scientific execution
            self.initialization_status["error"] = f"{type(exc).__name__}: {exc}"
            logger.exception("Telemetry initialization failed; research will continue without remote tracing")
            try:
                from opentelemetry import trace
                self.provider = trace.get_tracer_provider()
                self.tracer = trace.get_tracer(self.settings.otel_service_name)
                self.initialization_status.update({
                    "provider": f"{type(self.provider).__module__}.{type(self.provider).__name__}",
                    "processor_count": self._processor_count(self.provider),
                    "exporter": self._exporter_name(self.provider),
                })
            except Exception:
                self.tracer = None

    def force_flush(self, timeout_millis: int = 5000) -> bool:
        """Flush queued spans without ending the application-owned provider."""
        if self.provider is None:
            return False
        try:
            result = self.provider.force_flush(timeout_millis)
            return result is not False
        except Exception:
            logger.exception("Telemetry force_flush failed")
            return False

    def shutdown(self) -> bool:
        """Shut down the provider once, only from application lifecycle shutdown."""
        if self._shutdown:
            return True
        self._shutdown = True
        if self.provider is None or not self._owns_provider:
            return True
        try:
            result = self.provider.shutdown()
            return result is not False
        except Exception:
            logger.exception("Telemetry shutdown failed")
            return False

    @contextmanager
    def span(self, name: str, attributes: dict[str, Any] | None = None) -> Iterator[Any]:
        if not self.tracer:
            yield _NoopSpan()
            return
        with self.tracer.start_as_current_span(name) as span:
            for key, value in (attributes or {}).items():
                clean = redact(value, key)
                if isinstance(clean, (dict, list)):
                    clean = json.dumps(clean, ensure_ascii=False, default=str)
                try:
                    span.set_attribute(key, clean)
                except Exception:
                    pass
            yield span

    def trace_id(self, span: Any) -> str | None:
        try:
            value = span.get_span_context().trace_id
            return f"{value:032x}" if value else None
        except Exception:
            return None


_application_telemetry: Telemetry | None = None


def register_application_telemetry(telemetry: Telemetry) -> None:
    """Register the process-owned instance used by legacy integrations."""
    global _application_telemetry
    if _application_telemetry is not None and _application_telemetry is not telemetry:
        raise RuntimeError("Application telemetry is already registered")
    _application_telemetry = telemetry


def get_telemetry() -> Telemetry:
    """Return the registered application instance; never initialize a second provider."""
    if _application_telemetry is None:
        raise RuntimeError("Application telemetry has not been registered; inject Telemetry explicitly")
    return _application_telemetry
