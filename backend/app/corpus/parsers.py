from __future__ import annotations

import hashlib
import importlib.metadata
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import fitz
import httpx

from app.config import Settings
from app.models.domain import (
    DocumentBlock, DocumentMetadata, DocumentReference, DocumentSection, ScientificDocument, new_id,
)


class DocumentParser(ABC):
    name: str

    @abstractmethod
    def parse(self, filename: str, content: bytes, workspace: Path) -> ScientificDocument:
        raise NotImplementedError


class PyMuPDFParser(DocumentParser):
    name = "pymupdf"
    heading_re = re.compile(r"^(?:\d+(?:\.\d+)*\s+)?[A-Z][A-Za-z0-9 ,:/()\-]{2,90}$")

    def __init__(self, settings: Settings):
        self.settings = settings

    def parse(self, filename: str, content: bytes, workspace: Path) -> ScientificDocument:
        pdf = fitz.open(stream=content, filetype="pdf")
        if len(pdf) > self.settings.pdf_max_pages:
            raise ValueError(f"PDF has {len(pdf)} pages; max is {self.settings.pdf_max_pages}")
        pages: list[str] = []
        blocks: list[DocumentBlock] = []
        for page_no, page in enumerate(pdf, start=1):
            text = page.get_text("text") or ""
            pages.append(text)
            (workspace / f"page_{page_no:04d}.txt").write_text(text, encoding="utf-8")
            offset = 0
            for pidx, para in enumerate([p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]):
                start = text.find(para, offset)
                end = start + len(para) if start >= 0 else None
                if end is not None:
                    offset = end
                blocks.append(DocumentBlock(
                    id=f"blk_{page_no:04d}_{pidx:04d}", page=page_no, paragraph_index=pidx, text=para,
                    start_char=start if start >= 0 else None, end_char=end,
                ))
        sections = self._infer_sections(pages, blocks)
        self._bind_blocks_to_sections(blocks, sections)
        metadata = self._metadata(pdf, filename, pages)
        refs = self._references(sections)
        version = None
        try:
            version = importlib.metadata.version("PyMuPDF")
        except Exception:
            pass
        return ScientificDocument(
            id=workspace.name,
            filename=filename,
            sha256=hashlib.sha256(content).hexdigest(),
            page_count=len(pages),
            metadata=metadata,
            sections=sections,
            blocks=blocks,
            references=refs,
            workspace=str(workspace),
            parse_engine=self.name,
            parser_version=version,
            parse_warnings=["PyMuPDF fallback parser used; structure is heuristic."],
        )

    def _metadata(self, pdf: fitz.Document, filename: str, pages: list[str]) -> DocumentMetadata:
        raw = pdf.metadata or {}
        title = (raw.get("title") or "").strip() or None
        author = (raw.get("author") or "").strip()
        authors = [x.strip() for x in re.split(r";|,\s+(?=[A-Z])", author) if x.strip()] if author else []
        doi = None
        head = "\n".join(pages[:2])[:15000]
        m = re.search(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", head, re.I)
        if m:
            doi = m.group(0).rstrip(".,;)")
        year = None
        y = re.search(r"\b(19|20)\d{2}\b", head)
        if y:
            year = int(y.group(0))
        if not title:
            candidates = [x.strip() for x in pages[0].splitlines()[:20] if 8 < len(x.strip()) < 240] if pages else []
            title = candidates[0] if candidates else Path(filename).stem
        return DocumentMetadata(title=title, authors=authors, doi=doi, year=year)

    def _infer_sections(self, pages: list[str], blocks: list[DocumentBlock]) -> list[DocumentSection]:
        sections: list[DocumentSection] = []
        current_heading = "Document"
        current_lines: list[str] = []
        start_page = 1
        counter = 0

        def flush(end_page: int) -> None:
            nonlocal counter, current_lines
            text = "\n".join(current_lines).strip()
            if not text:
                return
            counter += 1
            sections.append(DocumentSection(
                id=f"sec_{counter:04d}", heading=current_heading, level=1,
                page_start=start_page, page_end=max(start_page, end_page), text=text,
            ))
            current_lines = []

        for pageno, text in enumerate(pages, start=1):
            for line in text.splitlines():
                candidate = line.strip()
                if self.heading_re.match(candidate) and len(candidate.split()) <= 12 and len("\n".join(current_lines)) > 400:
                    flush(pageno)
                    current_heading = candidate
                    start_page = pageno
                    continue
                current_lines.append(line)
        flush(len(pages))
        if not sections:
            sections.append(DocumentSection(id="sec_0001", heading="Document", level=1, page_start=1,
                                            page_end=max(1, len(pages)), text="\n".join(pages)))
        return sections

    @staticmethod
    def _bind_blocks_to_sections(blocks: list[DocumentBlock], sections: list[DocumentSection]) -> None:
        for block in blocks:
            section = next((s for s in sections if s.page_start <= block.page <= s.page_end), None)
            if section:
                block.section_id = section.id
                section.block_ids.append(block.id)

    @staticmethod
    def _references(sections: list[DocumentSection]) -> list[DocumentReference]:
        ref_section = next((s for s in sections if "reference" in s.heading.lower() or "bibliograph" in s.heading.lower()), None)
        if not ref_section:
            return []
        refs: list[DocumentReference] = []
        for idx, chunk in enumerate(re.split(r"\n(?=\s*(?:\[?\d+\]?\.?\s+))", ref_section.text)):
            text = chunk.strip()
            if len(text) < 15:
                continue
            m = re.search(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", text, re.I)
            refs.append(DocumentReference(id=f"ref_{idx:04d}", raw_text=text, doi=m.group(0).rstrip(".,;)") if m else None))
        return refs


class GROBIDParser(DocumentParser):
    name = "grobid"

    def __init__(self, settings: Settings):
        self.settings = settings

    def parse(self, filename: str, content: bytes, workspace: Path) -> ScientificDocument:
        if not self.settings.grobid_url:
            raise RuntimeError("GROBID_URL is not configured")
        # We deliberately keep the fallback parser as the canonical block/page extractor.
        # GROBID enriches bibliographic/section metadata without leaking TEI structures to agents.
        base = PyMuPDFParser(self.settings).parse(filename, content, workspace)
        url = self.settings.grobid_url.rstrip("/") + "/api/processFulltextDocument"
        with httpx.Client(timeout=60) as client:
            response = client.post(url, files={"input": (filename, content, "application/pdf")})
            response.raise_for_status()
        tei = response.text
        import xml.etree.ElementTree as ET
        root = ET.fromstring(tei)
        ns = {"tei": "http://www.tei-c.org/ns/1.0"}
        title = root.findtext(".//tei:titleStmt/tei:title", namespaces=ns)
        authors = []
        for a in root.findall(".//tei:sourceDesc//tei:author", ns):
            names = [x.strip() for x in a.itertext() if x.strip()]
            if names:
                authors.append(" ".join(names))
        doi = None
        for node in root.findall(".//tei:idno", ns):
            if (node.attrib.get("type") or "").upper() == "DOI" and node.text:
                doi = node.text.strip()
                break
        base.metadata = base.metadata.model_copy(update={
            "title": title.strip() if title else base.metadata.title,
            "authors": authors or base.metadata.authors,
            "doi": doi or base.metadata.doi,
        })
        base.parse_engine = self.name
        base.parse_warnings = []
        (workspace / "grobid.tei.xml").write_text(tei, encoding="utf-8")
        return base


class DoclingParser(DocumentParser):
    name = "docling"

    def __init__(self, settings: Settings):
        self.settings = settings

    def parse(self, filename: str, content: bytes, workspace: Path) -> ScientificDocument:
        if not self.settings.docling_url:
            raise RuntimeError("DOCLING_URL is not configured")
        # HTTP adapter intentionally optional; normalize any response back into DocumentIR.
        with httpx.Client(timeout=120) as client:
            response = client.post(self.settings.docling_url, files={"file": (filename, content, "application/pdf")})
            response.raise_for_status()
        payload: dict[str, Any] = response.json()
        base = PyMuPDFParser(self.settings).parse(filename, content, workspace)
        if payload.get("title"):
            base.metadata.title = str(payload["title"])
        base.parse_engine = self.name
        base.parse_warnings = ["Docling HTTP adapter normalized into the local DocumentIR."]
        (workspace / "docling.json").write_text(response.text, encoding="utf-8")
        return base


class ParserRouter:
    def __init__(self, settings: Settings):
        self.settings = settings

    def parse(self, filename: str, content: bytes, workspace: Path) -> ScientificDocument:
        warnings: list[str] = []
        if self.settings.grobid_url:
            try:
                return GROBIDParser(self.settings).parse(filename, content, workspace)
            except Exception as exc:
                warnings.append(f"GROBID failed: {type(exc).__name__}: {exc}")
        if self.settings.docling_url:
            try:
                doc = DoclingParser(self.settings).parse(filename, content, workspace)
                doc.parse_warnings = warnings + doc.parse_warnings
                return doc
            except Exception as exc:
                warnings.append(f"Docling failed: {type(exc).__name__}: {exc}")
        doc = PyMuPDFParser(self.settings).parse(filename, content, workspace)
        doc.parse_warnings = warnings + doc.parse_warnings
        return doc
