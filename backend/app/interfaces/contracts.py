"""LightRAG-inspired service contracts without taking a LightRAG dependency.

The important idea is stable interfaces between document ingestion, graph access,
query/retrieval, workspaces and the UI. Implementations are deliberately swappable.
"""
from __future__ import annotations

from typing import Protocol, Iterable

from app.models.domain import EvidenceSpan, ScientificDocument


class DocumentRepository(Protocol):
    def list(self, workspace_id: str) -> list[ScientificDocument]: ...
    def get(self, document_id: str) -> ScientificDocument: ...
    def delete(self, document_id: str) -> None: ...


class RetrievalBackend(Protocol):
    def search(self, document_id: str, query: str, limit: int = 10) -> list[EvidenceSpan]: ...
    def read_page(self, document_id: str, page: int) -> str: ...


class GraphBackend(Protocol):
    def snapshot(self, workspace_id: str) -> dict: ...
    def search(self, workspace_id: str, text: str, limit: int = 30) -> dict: ...


class WorkspaceBackend(Protocol):
    def ensure(self, workspace_id: str) -> str: ...
    def list(self) -> list[str]: ...
