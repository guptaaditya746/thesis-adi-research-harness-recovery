from app.services.retrieval import RetrievalService


def test_fts_query_is_lexical():
    query = RetrievalService._fts_query("Which datasets are used by the method?")
    assert "datasets" in query
    assert "method" in query
