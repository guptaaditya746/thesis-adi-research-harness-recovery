from __future__ import annotations

from rdflib import BNode, Graph, Literal, Namespace, RDF, RDFS, URIRef
from pyshacl import validate

from app.models.domain import CanonicalTriple, SubOntologyArtifact, ValidationReport

DATA = Namespace("https://example.org/thesis/data/")
SH = Namespace("http://www.w3.org/ns/shacl#")
XSD = Namespace("http://www.w3.org/2001/XMLSchema#")


class StructuralValidationService:
    """Validate triple structure using an ephemeral in-memory RDF representation.

    The graph exists only for RDF/SHACL checking and is never persisted as a research
    product. Factual support remains a separate evidence-verification dimension.
    """

    def build_ephemeral_graph(self, triples: list[CanonicalTriple]) -> Graph:
        g = Graph()
        for triple in triples:
            subject = URIRef(self._entity_uri(triple.subject))
            predicate = URIRef(triple.predicate_uri) if triple.predicate_uri else URIRef(DATA + self._slug(triple.predicate))
            if triple.subject_type and self._looks_uri(triple.subject_type):
                g.add((subject, RDF.type, URIRef(triple.subject_type)))
            if self._object_is_literal(triple):
                obj = Literal(triple.object)
            else:
                obj = URIRef(self._entity_uri(triple.object))
                if triple.object_type and self._looks_uri(triple.object_type):
                    g.add((obj, RDF.type, URIRef(triple.object_type)))
            g.add((subject, predicate, obj))
        return g

    def validate(self, triples: list[CanonicalTriple], ontology: SubOntologyArtifact) -> ValidationReport:
        try:
            data_graph = self.build_ephemeral_graph(triples)
            ontology_graph = Graph().parse(ontology.turtle_path, format="turtle")
            shapes = self._shapes_from_domains_ranges(ontology_graph)
            conforms, _, report_text = validate(
                data_graph=data_graph,
                shacl_graph=shapes,
                ont_graph=ontology_graph,
                inference="rdfs",
                abort_on_first=False,
            )
            return ValidationReport(
                rdf_valid=True,
                shacl_conforms=bool(conforms),
                shacl_report=str(report_text),
                structural_errors=[] if conforms else ["SHACL/domain-range constraints did not fully conform"],
            )
        except Exception as exc:
            return ValidationReport(
                rdf_valid=False,
                shacl_conforms=False,
                shacl_report=str(exc),
                structural_errors=[f"{type(exc).__name__}: {exc}"],
            )

    def _shapes_from_domains_ranges(self, ontology: Graph) -> Graph:
        shapes = Graph()
        for prop, domain in ontology.subject_objects(RDFS.domain):
            shape = URIRef(f"{prop}Shape")
            pshape = BNode()
            shapes.add((shape, RDF.type, SH.NodeShape))
            shapes.add((shape, SH.targetSubjectsOf, prop))
            if isinstance(domain, URIRef):
                shapes.add((shape, SH["class"], domain))
            shapes.add((shape, SH.property, pshape))
            shapes.add((pshape, SH.path, prop))
            range_ = next(iter(ontology.objects(prop, RDFS.range)), None)
            if isinstance(range_, URIRef):
                if str(range_).startswith(str(XSD)) or range_ == RDFS.Literal:
                    shapes.add((pshape, SH.nodeKind, SH.Literal))
                    if str(range_).startswith(str(XSD)):
                        shapes.add((pshape, SH.datatype, range_))
                else:
                    shapes.add((pshape, SH.nodeKind, SH.IRI))
                    shapes.add((pshape, SH["class"], range_))
        return shapes

    @staticmethod
    def _slug(text: str) -> str:
        return "_".join(text.strip().replace("/", " ").split())

    @classmethod
    def _entity_uri(cls, text: str) -> str:
        return str(DATA) + cls._slug(text)

    @staticmethod
    def _looks_uri(value: str) -> bool:
        return value.startswith("http://") or value.startswith("https://")

    @classmethod
    def _object_is_literal(cls, triple: CanonicalTriple) -> bool:
        if triple.object_type and (triple.object_type.startswith(str(XSD)) or triple.object_type == str(RDFS.Literal)):
            return True
        text = triple.object.strip()
        return text.startswith('"') or text.replace(".", "", 1).isdigit()
