from app.models.domain import ResearchState


def test_research_state_is_explicit():
    state = ResearchState(run_id='r', workspace_id='w', document_ids=['d'], question='q')
    assert state.ontology_ref is None
    assert state.open_tasks == []
    assert state.current_goal.startswith('Produce evidence-supported')
