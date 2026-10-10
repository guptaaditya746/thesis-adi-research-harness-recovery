from pathlib import Path

from app.demo.curated_success import create_curated_success
from app.models.domain import DocumentBlock, DocumentMetadata, EvidenceSpan, ScientificDocument
from app.services.pdf import PDFService
from app.services.retrieval import RetrievalService
from app.services.triple_map import TripleMapService
from app.storage.artifacts import ArtifactStore


def test_curated_success_creates_completed_evidence_backed_run(settings):
    page_text = (
        "Knowledge Base Question Answering: A Semantic Parsing Perspective\n"
        "Yu Gu\nVardaan Pahuja\nGong Cheng\nYu Su\n"
        "We identify and attribute this to two unique challenges of KBQA, "
        "schema-level complexity and fact-level complexity."
    )
    document = ScientificDocument(
        id="doc_demo",
        filename="demo.pdf",
        sha256="0" * 64,
        page_count=1,
        metadata=DocumentMetadata(title="Knowledge Base Question Answering: A Semantic Parsing Perspective"),
        blocks=[DocumentBlock(id="blk_demo", page=1, paragraph_index=0, text=page_text)],
        workspace=str(settings.data_dir / "documents" / "doc_demo"),
    )
    document_dir = Path(document.workspace)
    document_dir.mkdir(parents=True)
    (document_dir / "document.json").write_text(document.model_dump_json(indent=2), encoding="utf-8")
    (document_dir / "page_0001.txt").write_text(page_text, encoding="utf-8")
    (document_dir / "workspace_id.txt").write_text("default", encoding="utf-8")

    run = create_curated_success(settings, document.id)

    assert run.status == "completed"
    assert run.error is None
    assert len(run.final_triples) == 6
    assert all(triple.verification_status == "accepted" for triple in run.final_triples)
    assert all(triple.evidence_ids for triple in run.final_triples)
    assert run.evaluation is not None
    assert run.evaluation.provenance_coverage == 1.0
    assert run.evaluation.supported_ratio == 1.0
    assert "CURATED DEMO" in run.question

    store = ArtifactStore(settings, run.id)
    completion_meta = store.find_latest("completion_attempt")
    assert completion_meta is not None
    assert store.get(completion_meta["id"])["accepted"] is True

    pdf = PDFService(settings)
    retrieval = RetrievalService(settings)
    for triple in run.final_triples:
        for evidence_id in triple.evidence_ids:
            span = EvidenceSpan.model_validate(store.get(evidence_id))
            valid, _ = retrieval.validate_span(pdf.load(document.id), span)
            assert valid is True

    graph = TripleMapService.snapshot(run.final_triples)
    assert len(graph["edges"]) == 6
    assert len(graph["nodes"]) >= 7
