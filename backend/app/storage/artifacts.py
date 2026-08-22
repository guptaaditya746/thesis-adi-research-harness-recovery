from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from app.config import Settings
from app.models.domain import ArtifactEnvelope, ArtifactMetadata, ArtifactRef, ModelCallMetadata, new_id, stable_hash


class ArtifactStore:
    """Immutable, content-addressed run artifact store.

    Files are append-only by artifact id. The index only points to immutable envelopes;
    retrying a side-effect with the same artifact id is idempotent when content matches.
    """
    def __init__(self, settings: Settings, run_id: str):
        self.run_id = run_id
        self.root = settings.data_dir / "runs" / run_id / "artifacts"
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.json"

    def put(
        self,
        kind: str,
        value: Any,
        label: str | None = None,
        artifact_id: str | None = None,
        *,
        creator_agent_id: str = "research_director",
        parent_artifact_ids: list[str] | None = None,
        schema_version: str = "1.0",
        model_metadata: ModelCallMetadata | None = None,
    ) -> ArtifactRef:
        payload = self._jsonable(value)
        content_hash = stable_hash(payload)
        aid = artifact_id or new_id(kind[:18].replace("-", "_"))
        path = self.root / f"{aid}.json"
        envelope = ArtifactEnvelope(
            artifact_id=aid,
            artifact_type=kind,
            run_id=self.run_id,
            creator_agent_id=creator_agent_id,
            parent_artifact_ids=parent_artifact_ids or [],
            schema_version=schema_version,
            content_sha256=content_hash,
            model_metadata=model_metadata,
            content=payload,
        )
        encoded = envelope.model_dump_json(indent=2)
        if path.exists():
            existing = ArtifactEnvelope.model_validate_json(path.read_text(encoding="utf-8"))
            if existing.content_sha256 != content_hash:
                raise FileExistsError(f"Artifact {aid} already exists with different content")
        else:
            self._atomic_write(path, encoded)
        index = self._index()
        index[aid] = {
            "kind": kind,
            "path": str(path),
            "label": label,
            "content_sha256": content_hash,
            "creator_agent_id": creator_agent_id,
            "parent_artifact_ids": parent_artifact_ids or [],
            "schema_version": schema_version,
            "created_at": envelope.created_at.isoformat(),
        }
        self._atomic_write(self.index_path, json.dumps(index, indent=2, ensure_ascii=False))
        return ArtifactRef(id=aid, kind=kind, path=str(path), label=label, content_sha256=content_hash)

    def get(self, artifact_id: str, *, envelope: bool = False) -> Any:
        item = self._index().get(artifact_id)
        if not item:
            raise KeyError(artifact_id)
        env = ArtifactEnvelope.model_validate_json(Path(item["path"]).read_text(encoding="utf-8"))
        return env.model_dump(mode="json") if envelope else env.content

    def get_envelope(self, artifact_id: str) -> ArtifactEnvelope:
        item = self._index().get(artifact_id)
        if not item:
            raise KeyError(artifact_id)
        return ArtifactEnvelope.model_validate_json(Path(item["path"]).read_text(encoding="utf-8"))

    def get_metadata(self, artifact_id: str) -> ArtifactMetadata:
        """Return payload-free metadata for contract checks."""
        item = self._index().get(artifact_id)
        if not item:
            raise KeyError(artifact_id)
        return ArtifactMetadata(
            artifact_id=artifact_id,
            artifact_type=item["kind"],
            run_id=self.run_id,
            creator_agent_id=item.get("creator_agent_id", "unknown"),
            parent_artifact_ids=list(item.get("parent_artifact_ids", [])),
            schema_version=item.get("schema_version", "1.0"),
            content_sha256=item.get("content_sha256", ""),
        )

    def require_kind(self, artifact_id: str, expected_kind: str) -> ArtifactMetadata:
        metadata = self.get_metadata(artifact_id)
        if metadata.artifact_type != expected_kind:
            raise ValueError(
                f"Artifact {artifact_id} has kind {metadata.artifact_type}; expected {expected_kind}."
            )
        return metadata

    def list(self, kind: str | None = None) -> list[dict[str, Any]]:
        items = [{"id": aid, **meta} for aid, meta in self._index().items()]
        if kind:
            items = [x for x in items if x["kind"] == kind]
        return sorted(items, key=lambda x: x.get("created_at", ""))

    def find_latest(self, kind: str) -> dict[str, Any] | None:
        items = self.list(kind)
        return items[-1] if items else None

    def _index(self) -> dict[str, Any]:
        if not self.index_path.exists():
            return {}
        try:
            return json.loads(self.index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def _jsonable(value: Any) -> Any:
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json")
        if isinstance(value, list):
            return [ArtifactStore._jsonable(v) for v in value]
        if isinstance(value, tuple):
            return [ArtifactStore._jsonable(v) for v in value]
        if isinstance(value, dict):
            return {str(k): ArtifactStore._jsonable(v) for k, v in value.items()}
        return value

    @staticmethod
    def _atomic_write(path: Path, data: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp, path)
        finally:
            try:
                os.unlink(temp)
            except FileNotFoundError:
                pass
