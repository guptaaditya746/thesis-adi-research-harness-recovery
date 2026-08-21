from __future__ import annotations

import json
from pathlib import Path

from app.models.domain import ScientificDocument


class SidecarWriter:
    """A provenance-first document IR inspired by LightRAG sidecars.

    Each content line carries source positions. The agent can search convenient plain-text
    files, while every evidence item can still be traced to page/section/character spans.
    """

    def write(self, doc: ScientificDocument) -> str:
        workspace = Path(doc.workspace)
        sidecar = workspace / "sidecar"
        sidecar.mkdir(exist_ok=True)
        blocks_path = sidecar / "document.blocks.jsonl"
        meta = {
            "type": "meta",
            "document_id": doc.id,
            "filename": doc.filename,
            "page_count": doc.page_count,
            "parse_engine": doc.parse_engine,
            "schema": "thesis-adi-sidecar/1",
        }
        with blocks_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
            for section in doc.sections:
                f.write(json.dumps({
                    "type": "content",
                    "id": section.id,
                    "heading": section.heading,
                    "level": section.level,
                    "text": section.text,
                    "positions": [{
                        "type": "page_range",
                        "page_start": section.page_start,
                        "page_end": section.page_end,
                    }],
                }, ensure_ascii=False) + "\n")
        manifest = sidecar / "manifest.json"
        manifest.write_text(json.dumps({
            "blocks": str(blocks_path),
            "assets": [],
            "tables": None,
            "figures": None,
            "equations": None,
        }, indent=2), encoding="utf-8")
        return str(manifest)
