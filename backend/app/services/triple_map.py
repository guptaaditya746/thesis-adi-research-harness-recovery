from __future__ import annotations

import hashlib

from app.models.domain import CanonicalTriple


class TripleMapService:
    """Build a temporary visualization snapshot from accepted run triples.

    This is deliberately stateless: it never persists an instance graph. The canonical
    triple artifacts remain the scientific result and source of truth.
    """

    @staticmethod
    def snapshot(triples: list[CanonicalTriple]) -> dict[str, list[dict[str, str]]]:
        nodes: dict[str, dict[str, str]] = {}
        edges: list[dict[str, str]] = []
        for triple in triples:
            sid = TripleMapService._node_id(triple.subject)
            oid = TripleMapService._node_id(triple.object)
            nodes.setdefault(sid, {"id": sid, "label": triple.subject, "kind": triple.subject_type or "entity"})
            nodes.setdefault(oid, {"id": oid, "label": triple.object, "kind": triple.object_type or "entity"})
            edges.append({
                "id": triple.id,
                "source": sid,
                "target": oid,
                "label": triple.predicate,
                "predicate": triple.predicate_uri or triple.predicate,
                "status": triple.verification_status.value,
            })
        return {"nodes": list(nodes.values()), "edges": edges}

    @staticmethod
    def _node_id(label: str) -> str:
        digest = hashlib.sha256(label.strip().casefold().encode("utf-8")).hexdigest()[:16]
        return f"node_{digest}"
