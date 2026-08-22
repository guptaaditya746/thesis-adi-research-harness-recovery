from pydantic import TypeAdapter, ValidationError

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


def test_partial_conclusion_requires_explicit_partial_artifact():
    adapter = TypeAdapter(ResearchConclusion)
    with __import__("pytest").raises(ValidationError):
        adapter.validate_python({"status": "partial", "completion_reason": "blocked"})
    partial = adapter.validate_python({
        "status": "partial", "completion_reason": "blocked", "partial_result_artifact_id": "partial_result_1",
    })
    assert partial.partial_result_artifact_id == "partial_result_1"


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
