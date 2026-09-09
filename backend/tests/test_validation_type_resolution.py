from pathlib import Path

from rdflib import Graph, RDF, URIRef

from app.models.domain import CanonicalTriple, SubOntologyArtifact
from app.services.validation import StructuralValidationService


def _snapshot(tmp_path: Path) -> SubOntologyArtifact:
    path = tmp_path / "ontology.ttl"
    path.write_text(
        """@prefix ex: <https://example.org/ontology#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

ex:Publication a owl:Class ; rdfs:label \"Publication\" .
ex:addresses a owl:ObjectProperty ; rdfs:domain ex:Publication .
""",
        encoding="utf-8",
    )
    return SubOntologyArtifact(turtle_path=str(path), selected_elements=[])


def _triple(subject_type: str) -> CanonicalTriple:
    return CanonicalTriple(
        subject="Paper", predicate="addresses", object="Problem",
        predicate_uri="https://example.org/ontology#addresses", subject_type=subject_type,
    )


def _class_index(service: StructuralValidationService, snapshot: SubOntologyArtifact):
    graph = Graph().parse(snapshot.turtle_path, format="turtle")
    return service._class_uris_by_name(graph)


def test_shacl_resolves_unambiguous_snapshot_class_label(tmp_path):
    service = StructuralValidationService()
    snapshot = _snapshot(tmp_path)
    report = service.validate([_triple("Publication")], snapshot)
    graph = service.build_ephemeral_graph(
        [_triple("Publication")], class_uris_by_name=_class_index(service, snapshot),
    )

    assert (URIRef("https://example.org/thesis/data/Paper"), RDF.type, URIRef("https://example.org/ontology#Publication")) in graph
    assert report.shacl_conforms is True


def test_shacl_does_not_mint_uri_for_unknown_type_label(tmp_path):
    service = StructuralValidationService()
    snapshot = _snapshot(tmp_path)
    graph = service.build_ephemeral_graph(
        [_triple("Invented Publication Type")],
        class_uris_by_name=_class_index(service, snapshot),
    )

    subject = URIRef("https://example.org/thesis/data/Paper")
    assert list(graph.objects(subject, RDF.type)) == []
