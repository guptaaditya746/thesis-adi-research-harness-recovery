from __future__ import annotations

import hashlib
import json
import re
import threading
from contextlib import contextmanager
from functools import lru_cache
from typing import Any, Iterator

from app.config import Settings, get_settings

_SECRET_KEY = re.compile(r"(api[_-]?key|authorization|token|secret|password|credential)", re.I)
_SECRET_VALUE = re.compile(r"(?i)(bearer\s+[A-Za-z0-9._~+\-/=]+|sk-[A-Za-z0-9_-]{8,})")


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
    Langtrace is initialized first when configured; explicit OTLP export can be configured in parallel.
    """
    def __init__(self, settings: Settings):
        self.settings = settings
        self.tracer = None
        self._langtrace_started = False
        self._langtrace_starting = False
        self._langtrace_lock = threading.Lock()
        try:
            from opentelemetry import trace
            if settings.otel_exporter_otlp_endpoint:
                try:
                    from opentelemetry.sdk.resources import Resource
                    from opentelemetry.sdk.trace import TracerProvider
                    from opentelemetry.sdk.trace.export import BatchSpanProcessor
                    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
                    provider = trace.get_tracer_provider()
                    if provider.__class__.__module__.startswith("opentelemetry.trace"):
                        provider = TracerProvider(resource=Resource.create({"service.name": settings.otel_service_name}))
                        headers = None
                        if settings.otel_exporter_otlp_headers:
                            headers = dict(
                                part.split("=", 1) for part in settings.otel_exporter_otlp_headers.split(",") if "=" in part
                            )
                        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(
                            endpoint=settings.otel_exporter_otlp_endpoint, headers=headers
                        )))
                        trace.set_tracer_provider(provider)
                except Exception:
                    pass
            self.tracer = trace.get_tracer(settings.otel_service_name)
        except Exception:
            self.tracer = None

    def _start_langtrace_background(self) -> None:
        if not (self.settings.enable_langtrace and self.settings.langtrace_api_key):
            return
        with self._langtrace_lock:
            if self._langtrace_started or self._langtrace_starting:
                return
            self._langtrace_starting = True
        def initialize() -> None:
            try:
                from langtrace_python_sdk import langtrace
                langtrace.check_if_sdk_is_outdated = lambda: None
                kwargs: dict[str, Any] = {"api_key": self.settings.langtrace_api_key, "batch": False}
                if self.settings.langtrace_api_host:
                    kwargs["api_host"] = self.settings.langtrace_api_host
                langtrace.init(**kwargs)
                self._langtrace_started = True
            except Exception:
                pass
            finally:
                self._langtrace_starting = False
        threading.Thread(target=initialize, name="langtrace-init", daemon=True).start()

    @contextmanager
    def span(self, name: str, attributes: dict[str, Any] | None = None) -> Iterator[Any]:
        self._start_langtrace_background()
        if not self.tracer:
            yield _NoopSpan()
            return
        try:
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
        except Exception:
            # Never allow tracing failures to affect research.
            yield _NoopSpan()

    def trace_id(self, span: Any) -> str | None:
        try:
            value = span.get_span_context().trace_id
            return f"{value:032x}" if value else None
        except Exception:
            return None


@lru_cache
def get_telemetry() -> Telemetry:
    return Telemetry(get_settings())
