from __future__ import annotations

import logging
import sys
import types
from dataclasses import dataclass

import pytest

from app.telemetry.tracing import Telemetry, register_application_telemetry


@dataclass
class _Processor:
    _exporter: object = object()


class _Provider:
    __module__ = "test.telemetry"

    def __init__(self, processors: list[object] | None = None):
        self._active_span_processor = types.SimpleNamespace(_span_processors=processors or [])
        self.flush_calls = 0
        self.shutdown_calls = 0

    def force_flush(self, _timeout: int) -> bool:
        self.flush_calls += 1
        return True

    def shutdown(self) -> bool:
        self.shutdown_calls += 1
        return True

    def get_tracer(self, _name: str):
        return _Tracer()


class _Context:
    trace_id = 123
    span_id = 456


class _Span:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def get_span_context(self):
        return _Context()

    def set_attribute(self, *_args):
        return None


class _Tracer:
    def start_as_current_span(self, _name: str):
        return _Span()


@pytest.fixture
def fake_otel(monkeypatch):
    provider = _Provider([_Processor()])
    from opentelemetry import trace

    monkeypatch.setattr(trace, "get_tracer_provider", lambda: provider)
    monkeypatch.setattr(trace, "get_tracer", lambda _name: provider.get_tracer(_name))
    return provider


def _settings(settings, *, enabled: bool = False):
    return settings.model_copy(update={
        "enable_langtrace": enabled,
        "enable_langtrace_auto_instrumentation": False,
        "langtrace_api_key": "test-key" if enabled else None,
        "langtrace_api_host": "https://trace.invalid/api/trace",
        "otel_exporter_otlp_endpoint": None,
    })


def test_one_telemetry_initialization_attaches_one_processor(fake_otel, settings):
    telemetry = Telemetry(_settings(settings))
    assert telemetry.initialization_status["initialized"] is True
    assert telemetry.initialization_status["processor_count"] == 1
    assert telemetry.initialization_status["exporter"]


def test_successful_langtrace_init_is_called_once_and_attaches_exporter(monkeypatch, fake_otel, settings):
    calls: list[dict] = []
    module = types.ModuleType("langtrace_python_sdk")
    module.langtrace = types.SimpleNamespace(init=lambda **kwargs: calls.append(kwargs))
    monkeypatch.setitem(sys.modules, "langtrace_python_sdk", module)
    telemetry = Telemetry(_settings(settings, enabled=True))
    assert len(calls) == 1
    assert calls[0]["disable_instrumentations"] == {"all_except": []}
    assert telemetry.initialization_status["processor_count"] == 1
    assert telemetry.initialization_status["exporter"]


def test_application_owner_cannot_be_replaced(monkeypatch, fake_otel, settings):
    from app.telemetry import tracing

    monkeypatch.setattr(tracing, "_application_telemetry", None)
    first = Telemetry(_settings(settings))
    register_application_telemetry(first)
    with pytest.raises(RuntimeError, match="already registered"):
        register_application_telemetry(Telemetry(_settings(settings)))


def test_initialization_failure_is_visible_and_non_fatal(monkeypatch, fake_otel, settings, caplog):
    module = types.ModuleType("langtrace_python_sdk")
    module.langtrace = types.SimpleNamespace(init=lambda **_: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setitem(sys.modules, "langtrace_python_sdk", module)
    with caplog.at_level(logging.ERROR):
        telemetry = Telemetry(_settings(settings, enabled=True))
    assert telemetry.tracer is not None
    assert "RuntimeError: boom" in telemetry.initialization_status["error"]
    assert "Telemetry initialization failed" in caplog.text


def test_root_and_child_trace_id_and_persisted_id(fake_otel, settings):
    telemetry = Telemetry(_settings(settings))
    with telemetry.span("root") as root:
        root_id = telemetry.trace_id(root)
        with telemetry.span("child") as child:
            assert telemetry.trace_id(child) == root_id
    assert root_id == "0000000000000000000000000000007b"


def test_force_flush_and_shutdown_are_explicit_lifecycle_calls(fake_otel, settings):
    telemetry = Telemetry(_settings(settings))
    telemetry._owns_provider = True
    assert fake_otel.flush_calls == 0
    assert telemetry.force_flush() is True
    assert fake_otel.flush_calls == 1
    assert telemetry.shutdown() is True
    assert fake_otel.shutdown_calls == 1
    assert telemetry.shutdown() is True
    assert fake_otel.shutdown_calls == 1
