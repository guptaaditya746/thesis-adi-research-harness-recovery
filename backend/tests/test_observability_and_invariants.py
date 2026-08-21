from pathlib import Path

from app.telemetry.tracing import redact


def test_secret_redaction_and_bounded_payloads():
    data = redact({"LANGTRACE_API_KEY":"secret", "authorization":"Bearer abcdefghijklmnop", "safe":"x"*5000})
    assert data["LANGTRACE_API_KEY"] == "[REDACTED]"
    assert data["authorization"] == "[REDACTED]"
    assert "truncated sha256=" in data["safe"]


def test_source_has_no_persistent_instance_graph_service():
    root = Path(__file__).parents[1] / "app"
    assert not (root / "services" / "graph.py").exists()
    api = (root / "api" / "main.py").read_text(encoding="utf-8")
    tools = (root / "agents" / "toolsets.py").read_text(encoding="utf-8")
    assert '"/api/graph"' not in api
    assert "commit_graph" not in tools
    assert '"/api/runs/{run_id}/triple-map"' in api


def test_no_embedding_or_vector_dependency_in_base_project():
    pyproject = (Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8").lower()
    banned = ["sentence-transformers", "faiss", "chromadb", "pinecone", "qdrant", "weaviate"]
    assert all(name not in pyproject for name in banned)
