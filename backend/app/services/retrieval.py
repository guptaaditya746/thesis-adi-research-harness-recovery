from __future__ import annotations

import hashlib
import re
import sqlite3
from pathlib import Path

from app.config import Settings
from app.models.domain import DocumentBlock, EvidenceSpan, ScientificDocument


class RetrievalService:
    """Deterministic lexical/structural scientific retrieval. No embeddings are used."""
    def __init__(self, settings: Settings):
        self.settings = settings

    def search(self, doc: ScientificDocument, query: str, limit: int | None = None, *, agent_id: str = "research_director") -> list[EvidenceSpan]:
        limit = min(limit or self.settings.lexical_top_k, 100)
        db = sqlite3.connect(Path(doc.workspace) / "search.sqlite3")
        db.row_factory = sqlite3.Row
        safe_query = self._fts_query(query)
        try:
            rows = db.execute(
                "SELECT page, paragraph, block_id, section_id, text, bm25(paragraphs) AS score FROM paragraphs "
                "WHERE paragraphs MATCH ? ORDER BY score LIMIT ?", (safe_query, limit),
            ).fetchall()
        finally:
            db.close()
        return [self._span(doc, row, query, "sqlite_fts5_bm25", agent_id) for row in rows]

    def search_sections(self, doc: ScientificDocument, query: str, limit: int = 10, *, agent_id: str = "research_director") -> list[EvidenceSpan]:
        q = self._fts_query(query)
        db = sqlite3.connect(Path(doc.workspace) / "search.sqlite3")
        db.row_factory = sqlite3.Row
        try:
            rows = db.execute(
                "SELECT section_id, heading, text, bm25(sections) AS score FROM sections WHERE sections MATCH ? ORDER BY score LIMIT ?",
                (q, min(limit, 50)),
            ).fetchall()
        finally:
            db.close()
        out: list[EvidenceSpan] = []
        for row in rows:
            section = next((x for x in doc.sections if x.id == row["section_id"]), None)
            if not section:
                continue
            text = row["text"]
            out.append(EvidenceSpan(
                document_id=doc.id, page=section.page_start, section_id=section.id, text=text,
                source_text_sha256=hashlib.sha256(text.encode()).hexdigest(), retrieval_method="sqlite_fts5_section",
                retrieval_query=query, discovering_agent_id=agent_id, score=float(-row["score"]),
            ))
        return out

    def search_headings(self, doc: ScientificDocument, query: str, limit: int = 20) -> list[dict]:
        tokens = [t.lower() for t in re.findall(r"\w+", query)]
        scored = []
        for section in doc.sections:
            h = section.heading.lower()
            score = sum(1 for t in tokens if t in h)
            if score:
                scored.append((score, section))
        scored.sort(key=lambda x: (-x[0], x[1].page_start))
        return [s.model_dump(exclude={"text"}) | {"preview": s.text[:260]} for _, s in scored[:limit]]

    def search_references(self, doc: ScientificDocument, query: str, limit: int = 20) -> list[dict]:
        q = self._fts_query(query)
        db = sqlite3.connect(Path(doc.workspace) / "search.sqlite3")
        db.row_factory = sqlite3.Row
        try:
            rows = db.execute(
                "SELECT reference_id, text, bm25(references_fts) AS score FROM references_fts WHERE references_fts MATCH ? ORDER BY score LIMIT ?",
                (q, min(limit, 100)),
            ).fetchall()
        finally:
            db.close()
        return [{"reference_id": r["reference_id"], "text": r["text"], "score": float(-r["score"])} for r in rows]

    def search_entities(self, doc: ScientificDocument, entity: str, limit: int = 20, *, agent_id: str = "research_director") -> list[EvidenceSpan]:
        return self.find_phrase(doc, entity, limit=limit, agent_id=agent_id)

    def list_sections(self, doc: ScientificDocument) -> list[dict]:
        return [s.model_dump(exclude={"text"}) | {"preview": s.text[:260]} for s in doc.sections]

    def inspect_metadata(self, doc: ScientificDocument) -> dict:
        return doc.model_dump(exclude={"sections", "blocks", "references", "source_positions"}) | {
            "metadata": doc.metadata.model_dump(mode="json"),
            "reference_count": len(doc.references), "section_count": len(doc.sections), "block_count": len(doc.blocks),
        }

    def read_page(self, doc: ScientificDocument, page: int) -> str:
        if page < 1 or page > doc.page_count:
            raise ValueError("page out of range")
        return (Path(doc.workspace) / f"page_{page:04d}.txt").read_text(encoding="utf-8")

    def read_section(self, doc: ScientificDocument, section_id: str) -> str:
        path = Path(doc.workspace) / "sections" / f"{section_id}.txt"
        if not path.exists():
            raise ValueError("section not found")
        return path.read_text(encoding="utf-8")

    def read_block(self, doc: ScientificDocument, block_id: str) -> DocumentBlock:
        block = next((x for x in doc.blocks if x.id == block_id), None)
        if not block:
            raise ValueError("block not found")
        return block

    def find_phrase(self, doc: ScientificDocument, phrase: str, limit: int = 20, *, agent_id: str = "research_director") -> list[EvidenceSpan]:
        results = self.search(doc, f'"{phrase}"', limit=limit, agent_id=agent_id)
        return [r.model_copy(update={"retrieval_method": "exact_phrase+fts5"}) for r in results]

    def find_citations(self, doc: ScientificDocument, citation: str, limit: int = 20, *, agent_id: str = "research_director") -> list[EvidenceSpan]:
        return self.search(doc, citation, limit=limit, agent_id=agent_id)

    def expand(self, doc: ScientificDocument, span: EvidenceSpan, paragraphs: int = 2) -> EvidenceSpan:
        blocks = [b for b in doc.blocks if b.page == span.page]
        if not blocks:
            return span
        idx = next((i for i, b in enumerate(blocks) if b.id == span.block_id), None)
        if idx is None:
            idx = max(range(len(blocks)), key=lambda i: self._overlap(blocks[i].text, span.text))
        lo, hi = max(0, idx - paragraphs), min(len(blocks), idx + paragraphs + 1)
        selected = blocks[lo:hi]
        text = "\n\n".join(b.text for b in selected)
        return span.model_copy(update={
            "id": span.id,
            "text": text,
            "block_id": selected[0].id if len(selected) == 1 else None,
            "source_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "retrieval_method": span.retrieval_method + "+context",
        })

    def nearby_blocks(self, doc: ScientificDocument, block_id: str, radius: int = 2) -> list[DocumentBlock]:
        block = self.read_block(doc, block_id)
        page_blocks = [b for b in doc.blocks if b.page == block.page]
        idx = page_blocks.index(block)
        return page_blocks[max(0, idx-radius): min(len(page_blocks), idx+radius+1)]

    def validate_span(self, doc: ScientificDocument, span: EvidenceSpan) -> tuple[bool, str]:
        if span.document_id != doc.id:
            return False, "Evidence document_id does not match document"
        if span.block_id:
            block = next((x for x in doc.blocks if x.id == span.block_id), None)
            if not block:
                return False, "Evidence block_id does not resolve"
            if span.text == block.text:
                return True, "Exact block evidence matches DocumentIR"
            # Context-expanded spans must be a verbatim concatenation of same-page blocks, never model-generated text.
            if all(part.strip() in self.read_page(doc, span.page) for part in span.text.split("\n\n") if part.strip()):
                return True, "Context-expanded evidence resolves verbatim on source page"
            return False, "Evidence text no longer matches source block/page"
        page = self.read_page(doc, span.page)
        if span.text in page or all(part.strip() in page for part in span.text.split("\n\n") if part.strip()):
            return True, "Evidence text resolves verbatim on source page"
        return False, "Evidence text does not resolve on source page"

    @staticmethod
    def _span(doc: ScientificDocument, row: sqlite3.Row, query: str, method: str, agent_id: str) -> EvidenceSpan:
        text = str(row["text"])
        block_id = str(row["block_id"]) if row["block_id"] else None
        block = next((x for x in doc.blocks if x.id == block_id), None)
        return EvidenceSpan(
            document_id=doc.id, page=int(row["page"]), section_id=str(row["section_id"]) or None,
            block_id=block_id, paragraph_index=int(row["paragraph"]), text=text,
            source_text_sha256=hashlib.sha256(text.encode()).hexdigest(),
            start_char=block.start_char if block else None, end_char=block.end_char if block else None,
            retrieval_method=method, retrieval_query=query, discovering_agent_id=agent_id,
            score=float(-row["score"]),
        )

    @staticmethod
    def _fts_query(query: str) -> str:
        q = query.strip()
        if q.startswith('"') and q.endswith('"') and len(q) > 2:
            return q
        terms = re.findall(r"[A-Za-z0-9_\-]{2,}", query)
        return " OR ".join(f'"{t}"' for t in terms[:24]) or '"scientific"'

    @staticmethod
    def _overlap(a: str, b: str) -> int:
        return len(set(re.findall(r"\w+", a.lower())) & set(re.findall(r"\w+", b.lower())))
