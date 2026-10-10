from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

from app.config import Settings
from app.models.domain import VerifiedTriple


class GraphService:
    """Small graph interface modeled after LightRAG's swappable graph storage surface."""
    def __init__(self, settings: Settings):
        self.settings = settings

    def _path(self, workspace_id: str) -> Path:
        p = self.settings.data_dir / "workspaces" / workspace_id / "graph.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def _load(self, workspace_id: str) -> nx.MultiDiGraph:
        path = self._path(workspace_id)
        if not path.exists():
            return nx.MultiDiGraph()
        data = json.loads(path.read_text(encoding="utf-8"))
        return nx.node_link_graph(data, directed=True, multigraph=True, edges="edges")

    def _save(self, workspace_id: str, graph: nx.MultiDiGraph) -> None:
        self._path(workspace_id).write_text(
            json.dumps(nx.node_link_data(graph, edges="edges"), indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def upsert_triples(self, workspace_id: str, triples: list[VerifiedTriple], run_id: str) -> None:
        g = self._load(workspace_id)
        for verified in triples:
            t = verified.triple
            g.add_node(t.subject, label=t.subject, kind="entity")
            g.add_node(t.object, label=t.object, kind="entity")
            g.add_edge(t.subject, t.object, key=t.id, label=t.predicate, predicate=t.predicate,
                       predicate_uri=t.predicate_uri, run_id=run_id, status=verified.status,
                       evidence_ids=t.evidence_ids)
        self._save(workspace_id, g)

    def snapshot(self, workspace_id: str) -> dict:
        g = self._load(workspace_id)
        return {
            "nodes": [{"id": str(n), **attrs} for n, attrs in g.nodes(data=True)],
            "edges": [{"id": str(k), "source": str(s), "target": str(t), **attrs}
                      for s, t, k, attrs in g.edges(keys=True, data=True)],
        }

    def search(self, workspace_id: str, text: str, limit: int = 30) -> dict:
        q = text.lower().strip()
        snap = self.snapshot(workspace_id)
        nodes = [n for n in snap["nodes"] if q in str(n.get("label", n["id"])).lower()][:limit]
        ids = {n["id"] for n in nodes}
        edges = [e for e in snap["edges"] if e["source"] in ids or e["target"] in ids][:limit * 3]
        return {"nodes": nodes, "edges": edges}
