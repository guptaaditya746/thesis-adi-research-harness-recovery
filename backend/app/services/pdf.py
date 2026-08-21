from __future__ import annotations

import hashlib
import re
import shutil
import sqlite3
from pathlib import Path

from app.config import Settings
from app.corpus.parsers import ParserRouter
from app.models.domain import ScientificDocument, new_id
from app.services.sidecar import SidecarWriter


class PDFService:
    """Parser-neutral document ingestion facade."""
    def __init__(self, settings: Settings):
        self.settings = settings
        self.sidecar = SidecarWriter()
        self.parsers = ParserRouter(settings)

    def ingest(self, filename: str, content: bytes, workspace_id: str = "default") -> ScientificDocument:
        if not filename.lower().endswith(".pdf"):
            raise ValueError("Base scientific ingestion currently accepts PDF files only")
        if not content.startswith(b"%PDF"):
            raise ValueError("Uploaded file is not a PDF")
        doc_id = new_id("doc")
        workspace = self.settings.data_dir / "documents" / doc_id
        workspace.mkdir(parents=True, exist_ok=False)
        (workspace / "source.pdf").write_bytes(content)
        try:
            doc = self.parsers.parse(filename, content, workspace)
            if doc.id != doc_id:
                doc.id = doc_id
            doc.sha256 = hashlib.sha256(content).hexdigest()
            joined = "\n\n".join(
                f"=== PAGE {page} ===\n{(workspace / f'page_{page:04d}.txt').read_text(encoding='utf-8')}"
                for page in range(1, doc.page_count + 1)
            )
            (workspace / "document.txt").write_text(joined, encoding="utf-8")
            sections_dir = workspace / "sections"
            sections_dir.mkdir(exist_ok=True)
            for section in doc.sections:
                (sections_dir / f"{section.id}.txt").write_text(section.text, encoding="utf-8")
            doc.sidecar_manifest = self.sidecar.write(doc)
            (workspace / "document.json").write_text(doc.model_dump_json(indent=2), encoding="utf-8")
            (workspace / "workspace_id.txt").write_text(workspace_id, encoding="utf-8")
            self._build_fts(workspace, doc)
            return doc
        except Exception:
            shutil.rmtree(workspace, ignore_errors=True)
            raise

    def load(self, document_id: str) -> ScientificDocument:
        path = self.settings.data_dir / "documents" / document_id / "document.json"
        if not path.exists():
            raise KeyError(document_id)
        return ScientificDocument.model_validate_json(path.read_text(encoding="utf-8"))

    def list(self, workspace_id: str = "default") -> list[ScientificDocument]:
        docs: list[ScientificDocument] = []
        for path in (self.settings.data_dir / "documents").glob("*/document.json"):
            try:
                marker = path.parent / "workspace_id.txt"
                if (marker.read_text(encoding="utf-8").strip() if marker.exists() else "default") != workspace_id:
                    continue
                docs.append(ScientificDocument.model_validate_json(path.read_text(encoding="utf-8")))
            except Exception:
                continue
        return sorted(docs, key=lambda x: x.created_at, reverse=True)

    def delete(self, document_id: str) -> None:
        path = self.settings.data_dir / "documents" / document_id
        if path.exists():
            shutil.rmtree(path)

    def _build_fts(self, workspace: Path, doc: ScientificDocument) -> None:
        db = sqlite3.connect(workspace / "search.sqlite3")
        db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS pages USING fts5(page UNINDEXED, text)")
        db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS paragraphs USING fts5(page UNINDEXED, paragraph UNINDEXED, block_id UNINDEXED, section_id UNINDEXED, text)")
        db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS sections USING fts5(section_id UNINDEXED, heading, text)")
        db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS references_fts USING fts5(reference_id UNINDEXED, text)")
        for table in ("pages", "paragraphs", "sections", "references_fts"):
            db.execute(f"DELETE FROM {table}")
        page_rows = []
        for page in range(1, doc.page_count + 1):
            p = workspace / f"page_{page:04d}.txt"
            page_rows.append((page, p.read_text(encoding="utf-8") if p.exists() else ""))
        db.executemany("INSERT INTO pages(page, text) VALUES (?, ?)", page_rows)
        db.executemany(
            "INSERT INTO paragraphs(page, paragraph, block_id, section_id, text) VALUES (?,?,?,?,?)",
            [(b.page, b.paragraph_index or 0, b.id, b.section_id or "", b.text) for b in doc.blocks],
        )
        db.executemany("INSERT INTO sections(section_id, heading, text) VALUES (?,?,?)",
                       [(s.id, s.heading, s.text) for s in doc.sections])
        db.executemany("INSERT INTO references_fts(reference_id, text) VALUES (?,?)",
                       [(r.id, r.raw_text) for r in doc.references])
        db.commit()
        db.close()
