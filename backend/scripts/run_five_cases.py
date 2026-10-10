#!/usr/bin/env python3
"""Run five representative research questions and write one durable JSON report."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from app.harness.benchmark import QUESTIONS, summarize_run


async def run_cases(args: argparse.Namespace) -> Path:
    # Keep CLI parsing/help and report-unit tests fast; the research runtime has
    # intentionally heavy optional scientific/telemetry imports.
    from app.config import Settings
    from app.harness.runtime import ResearchHarness
    from app.models.domain import ExperimentRun
    from app.services.pdf import PDFService
    from app.storage.artifacts import ArtifactStore
    from app.storage.registry import RunRegistry

    settings = Settings()
    settings.ensure_dirs()
    pdf = PDFService(settings)
    documents = pdf.list(args.workspace_id)
    selected_ids = args.document_id or ([documents[0].id] if documents else [])
    if not selected_ids:
        raise SystemExit(
            "No parsed document is available. Upload one first or pass --document-id DOC_ID."
        )
    for document_id in selected_ids:
        pdf.load(document_id)

    registry = RunRegistry(settings)
    harness = ResearchHarness(settings, registry)
    cases = []
    for question in QUESTIONS:
        run = registry.create(ExperimentRun(
            workspace_id=args.workspace_id,
            document_ids=selected_ids,
            target_document_id=args.target_document_id or selected_ids[0],
            question=question,
        ))
        started = time.perf_counter()
        try:
            await harness.execute(run.id)
        except Exception as exc:  # keep the suite running and preserve the diagnosis
            run = registry.get(run.id)
            run.status = "failed"
            run.error = f"{type(exc).__name__}: {exc}"
            registry.persist(run)
        elapsed = time.perf_counter() - started
        run = registry.get(run.id)
        cases.append(summarize_run(run, ArtifactStore(settings, run.id), elapsed))

    report = {
        "schema_version": "1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "workspace_id": args.workspace_id,
        "document_ids": selected_ids,
        "case_count": len(cases),
        "completed_count": sum(case["status"] == "completed" for case in cases),
        "partial_result_count": sum(case["result_level"] == "unverified_open" for case in cases),
        "cases": cases,
    }
    output = args.output or (
        settings.data_dir / "benchmarks" /
        f"five_case_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    temp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(output)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document-id", action="append", help="Parsed document ID; repeat to use several.")
    parser.add_argument("--target-document-id", help="Primary document; defaults to the first selected document.")
    parser.add_argument("--workspace-id", default="default")
    parser.add_argument("--output", type=Path, help="Report path; defaults under DATA_DIR/benchmarks.")
    return parser.parse_args()


if __name__ == "__main__":
    report_path = asyncio.run(run_cases(parse_args()))
    print(report_path)
