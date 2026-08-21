from pathlib import Path
import re

from app.config import Settings


class WorkspaceService:
    def __init__(self, settings: Settings):
        self.settings = settings

    @staticmethod
    def normalize(workspace_id: str) -> str:
        value = re.sub(r"[^a-zA-Z0-9._-]+", "-", workspace_id.strip()).strip(".-")
        if not value:
            value = "default"
        return value[:80]

    def ensure(self, workspace_id: str) -> Path:
        wid = self.normalize(workspace_id)
        path = self.settings.data_dir / "workspaces" / wid
        for sub in ("documents", "runs", "artifacts", "plans"):
            (path / sub).mkdir(parents=True, exist_ok=True)
        return path

    def list(self) -> list[str]:
        base = self.settings.data_dir / "workspaces"
        return sorted(p.name for p in base.iterdir() if p.is_dir())
