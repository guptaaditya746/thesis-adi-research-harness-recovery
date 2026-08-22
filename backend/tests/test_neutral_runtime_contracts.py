from pydantic import TypeAdapter, ValidationError

from app.harness.completion import ScientificCompletionService
from app.models.domain import ResearchConclusion, SpecialistResult
from app.storage.artifacts import ArtifactStore
from app.tools.research import scientific_state_snapshot


def test_completed_conclusion_structurally_requires_canonical_triples():
    adapter = TypeAdapter(ResearchConclusion)
    with __import__("pytest").raises(ValidationError):
        adapter.validate_python({"status": "completed", "completion_reason": "done"})
    completed = adapter.validate_python({
        "status": "completed", "completion_reason": "done", "triple_artifact_id": "canonical_triples_1",
    })
    assert completed.triple_artifact_id == "canonical_triples_1"


def test_partial_conclusion_is_not_a_valid_scientific_result():
    adapter = TypeAdapter(ResearchConclusion)
    with __import__("pytest").raises(ValidationError):
        adapter.validate_python({
            "status": "partial", "completion_reason": "blocked", "partial_result_artifact_id": "partial_result_1",
        })


def test_specialist_completion_is_explicitly_task_scoped():
    result = SpecialistResult(task_scope="evidence_selection", task_status="completed", summary="Selected evidence.")
    assert result.task_scope == "evidence_selection"
    assert result.task_status == "completed"


def test_scientific_state_reads_artifact_store_not_mutable_research_state(settings):
    store = ArtifactStore(settings, "run_state_snapshot")
    evidence = store.put("evidence_bundle", {"spans": []})
    deps = type("Deps", (), {
        "run_id": "run_state_snapshot", "target_document_id": "doc_target",
        "document_ids": ["doc_target"], "artifacts": store,
    })()

    state = scientific_state_snapshot(deps)

    assert state["artifacts"]["evidence_bundle"] == {
        "count": 1, "artifact_ids": [evidence.id], "latest_artifact_id": evidence.id,
    }


def test_invalid_completion_attempt_returns_factual_state_without_raising(settings):
    store = ArtifactStore(settings, "run_completion_attempt")
    store.put("evidence_bundle", {"spans": []})
    deps = type("Deps", (), {"artifacts": store})()

    attempt = ScientificCompletionService().attempt(deps, "not_an_artifact")

    assert attempt.accepted is False
    assert attempt.reason == "scientific_completion_requirements_not_met"
    assert attempt.scientific_state.evidence_bundle is True
    assert attempt.scientific_state.canonical_triples is False
