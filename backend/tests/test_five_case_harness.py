from types import SimpleNamespace

from app.harness.benchmark import QUESTIONS, summarize_run
from app.models.domain import CandidateTriple, EDCOpenTriples, ExperimentRun
from app.storage.artifacts import ArtifactStore


def test_five_case_harness_has_five_questions_and_reports_provenance(settings):
    assert len(QUESTIONS) == 5
    assert len(set(QUESTIONS)) == 5

    run = ExperimentRun(
        workspace_id="default", document_ids=["doc_1"], question=QUESTIONS[0], status="failed"
    )
    run.state = SimpleNamespace(
        delegate_calls=2, model_calls=3, tool_calls=4
    )
    store = ArtifactStore(settings, run.id)
    store.put("evidence_bundle", {"spans": []})
    store.put("specialist_failure", {"error_type": "ArtifactNotFound"})

    summary = summarize_run(run, store, 1.25)

    assert summary["elapsed_seconds"] == 1.25
    assert summary["provenance"]["evidence_persisted"] is True
    assert summary["provenance"]["canonical_triples_persisted"] is False
    assert summary["invalid_artifact_failures"] == 1
    assert summary["result_level"] == "none"
    assert summary["triple_count"] == 0


def test_five_case_report_returns_open_triples_when_strict_completion_fails(settings):
    run = ExperimentRun(
        workspace_id="default", document_ids=["doc_1"], question=QUESTIONS[0], status="failed"
    )
    store = ArtifactStore(settings, run.id)
    open_result = EDCOpenTriples(
        edc_execution_id="edc_1",
        evidence_bundle_artifact_id="evidence_bundle_1",
        triples=[CandidateTriple(
            subject="Paper", predicate="addresses", object="Research problem", evidence_ids=["ev_1"]
        )],
    )
    ref = store.put("edc_open_triples", open_result)

    summary = summarize_run(run, store, 1.0)

    assert summary["result_level"] == "unverified_open"
    assert summary["triple_count"] == 1
    assert summary["triples"][0]["subject"] == "Paper"
    assert summary["triple_artifact_id"] == ref.id
