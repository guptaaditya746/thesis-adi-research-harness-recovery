from pathlib import Path

from app.models.domain import CanonicalTriple, ProvenanceRecord, VerifiedTriple
from app.services.workspaces import WorkspaceService


def test_workspace_normalization():
    assert WorkspaceService.normalize(" Thesis / Run 01 ") == "Thesis-Run-01"


def test_verified_triple_shape():
    triple = CanonicalTriple(subject="A", predicate="uses", object="B", evidence_ids=["ev_1"])
    value = VerifiedTriple(triple=triple, status="supported", explanation="source",
        provenance=ProvenanceRecord(triple_id=triple.id, document_id="doc", evidence_ids=["ev_1"], inference_type="direct", method="test"))
    assert value.triple.evidence_ids == ["ev_1"]
