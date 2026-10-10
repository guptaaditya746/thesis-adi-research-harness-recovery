import fitz

from app.services.pdf import PDFService
from app.services.retrieval import RetrievalService


def make_pdf(text: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


def test_lexical_retrieval_resolves_exact_source(settings):
    pdf = PDFService(settings)
    scientific = pdf.ingest("paper.pdf", make_pdf("We evaluate Qwen on the DBLP benchmark and report strong accuracy."))
    retrieval = RetrievalService(settings)
    spans = retrieval.search(scientific, "Qwen DBLP")
    assert spans
    span = spans[0]
    assert span.document_id == scientific.id
    assert span.block_id
    assert "Qwen" in span.text
    ok, reason = retrieval.validate_span(scientific, span)
    assert ok, reason
    assert span.retrieval_method == "sqlite_fts5_bm25"


def test_evidence_context_is_verbatim_not_model_generated(settings):
    pdf = PDFService(settings)
    scientific = pdf.ingest("paper.pdf", make_pdf("Alpha method uses Beta dataset. Gamma is a second sentence."))
    retrieval = RetrievalService(settings)
    span = retrieval.search(scientific, "Alpha Beta")[0]
    expanded = retrieval.expand(scientific, span, paragraphs=2)
    page = retrieval.read_page(scientific, 1)
    assert all(part.strip() in page for part in expanded.text.split("\n\n") if part.strip())
    assert retrieval.validate_span(scientific, expanded)[0]
