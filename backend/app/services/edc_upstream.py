from __future__ import annotations

from pathlib import Path


class UpstreamEDCAdapter:
    """Optional integration point for the original EDC repository.

    Keep this adapter isolated because upstream EDC expects CSV schemas and may load embedding
    models. The default thesis pipeline uses app.services.edc. Add an ablation implementation
    here when you want to reproduce upstream EDC exactly.
    """

    def __init__(self, source_path: str | None):
        self.source_path = Path(source_path).resolve() if source_path else None

    @property
    def available(self) -> bool:
        return bool(self.source_path and self.source_path.exists())
