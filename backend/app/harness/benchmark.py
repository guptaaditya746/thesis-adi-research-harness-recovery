"""Reusable definitions for the automated representative-case harness."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.domain import ExperimentRun
    from app.storage.artifacts import ArtifactStore


QUESTIONS = (
    "What research problem does this paper address?",
    "What method or approach does this paper propose?",
    "Which datasets, benchmarks, or experimental settings are used?",
    "What are the main empirical findings reported by the paper?",
    "What limitations, threats to validity, or open problems does the paper identify?",
)


def summarize_run(run: ExperimentRun, artifacts: ArtifactStore, elapsed_seconds: float) -> dict:
    rows = artifacts.list()
    kinds = Counter(row["kind"] for row in rows)
    events = run.events
    triple_meta = artifacts.find_latest("canonical_triples")
    result_level = "verified"
    if triple_meta is None:
        triple_meta = artifacts.find_latest("edc_open_triples")
        result_level = "unverified_open" if triple_meta else "none"
    triples = []
    if triple_meta:
        payload = artifacts.get(triple_meta["id"])
        triples = payload.get("triples", []) if result_level == "unverified_open" else payload
    return {
        "run_id": run.id,
        "question": run.question,
        "status": run.status,
        "error": run.error,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "artifact_count": len(rows),
        "artifact_counts_by_kind": dict(sorted(kinds.items())),
        "result_level": result_level,
        "triple_count": len(triples),
        "triples": triples,
        "triple_artifact_id": triple_meta["id"] if triple_meta else None,
        "provenance": {
            "evidence_persisted": bool(kinds["evidence_bundle"]),
            "ontology_snapshot_persisted": bool(kinds["ontology_snapshot"]),
            "candidate_triples_persisted": bool(kinds["candidate_canonical_triples"]),
            "verification_persisted": bool(kinds["verification"]),
            "canonical_triples_persisted": bool(kinds["canonical_triples"]),
        },
        "subagent_calls": run.state.delegate_calls if run.state else 0,
        "recovery_events": sum(event.type == "recovery" for event in events),
        "invalid_artifact_failures": kinds["specialist_failure"],
        "model_calls": run.state.model_calls if run.state else 0,
        "tool_calls": run.state.tool_calls if run.state else 0,
    }
