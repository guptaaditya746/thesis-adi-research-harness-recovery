from app.models.domain import CanonicalTriple, GoldAnnotation, GoldTriple, ProvenanceRecord, VerifiedTriple
from app.services.evaluation import EvaluationService


def test_exact_metrics():
    triple = CanonicalTriple(subject="Paper", predicate="uses dataset", object="DBLP", evidence_ids=["ev1"])
    verified = VerifiedTriple(
        triple=triple,
        status="supported",
        explanation="supported",
        provenance=ProvenanceRecord(
            triple_id=triple.id, document_id="doc1", evidence_ids=["ev1"], inference_type="direct", method="test"
        ),
    )
    gold = GoldAnnotation(question="q", triples=[GoldTriple(subject="Paper", predicate="uses dataset", object="DBLP")])
    result = EvaluationService().evaluate([verified], True, gold)
    assert result.exact_f1 == 1.0
    assert result.provenance_coverage == 1.0
