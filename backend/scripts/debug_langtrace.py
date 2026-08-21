"""Verify application-owned manual spans and Langtrace export."""

from __future__ import annotations

import os

from app.config import get_settings
from app.telemetry.tracing import Telemetry


def main() -> int:
    os.environ.setdefault("LANGTRACE_ERROR_REPORTING", "False")
    telemetry = Telemetry(get_settings())
    status = telemetry.initialization_status
    print("telemetry_status:", status)
    print("provider:", status.get("provider"))
    print("processor_count:", status.get("processor_count"))
    print("exporter:", status.get("exporter"))

    with telemetry.span("langtrace.phase1.root") as root:
        root_trace_id = telemetry.trace_id(root)
        with telemetry.span("langtrace.phase1.child") as child:
            child_trace_id = telemetry.trace_id(child)

    print("root_trace_id:", root_trace_id)
    print("child_trace_id:", child_trace_id)
    print("shared_trace_id:", bool(root_trace_id and root_trace_id == child_trace_id))
    flushed = telemetry.force_flush()
    print("force_flush:", flushed)
    if status.get("initialized") and flushed:
        print("export_result: accepted by the configured exporter; verify visibility in Langtrace UI")
    else:
        print("export_result: not proven; inspect telemetry_status.error")
    telemetry.shutdown()
    return 0 if root_trace_id and root_trace_id == child_trace_id else 1


if __name__ == "__main__":
    raise SystemExit(main())
