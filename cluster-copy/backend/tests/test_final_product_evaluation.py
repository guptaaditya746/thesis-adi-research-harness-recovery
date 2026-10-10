from app.models.domain import CanonicalTriple, GoldAnnotation, GoldTriple, VerificationStatus
from app.services.evaluation import EvaluationService
from app.services.triple_map import TripleMapService


def accepted_triple():
    return CanonicalTriple(
        subject="Paper", predicate="authoredBy", object="Ada",
        predicate_uri="https://dblp.org/rdf/schema#authoredBy",
        evidence_ids=["ev1"], verification_status=VerificationStatus.ACCEPTED,
        canonicalization="exact",
    )


def test_evaluation_uses_canonical_triples_directly():
    triple = accepted_triple()
    gold = GoldAnnotation(question="q", triples=[GoldTriple(subject="Paper", predicate="authoredBy", object="Ada")])
    result = EvaluationService().evaluate([triple], shacl_conforms=True, gold=gold)
    assert result.exact_f1 == 1.0
    assert result.provenance_coverage == 1.0
    assert result.supported_ratio == 1.0


def test_triple_map_is_derived_not_persistent():
    snapshot = TripleMapService.snapshot([accepted_triple()])
    assert len(snapshot["nodes"]) == 2
    assert len(snapshot["edges"]) == 1
    assert snapshot["edges"][0]["predicate"].endswith("#authoredBy")
