from __future__ import annotations

import hashlib
import re
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import quote

from rdflib import Graph, Literal, OWL, RDF, RDFS, URIRef

from app.config import Settings
from app.models.domain import OntologyElement, OntologyProposal, OntologyProposalStatus, SubOntologyArtifact

DBLP = "https://dblp.org/rdf/schema#"
RUN_OVERLAY_BASE = "https://example.org/thesis/run/"
SCHEMA_VERSION = "2024-06-14"


class OntologyService:
    """Immutable DBLP seed + explicit run-local ontology overlays."""
    def __init__(self, settings: Settings):
        self.settings = settings
        self.seed = Graph()
        self.seed.parse(settings.seed_ontology_path, format="turtle")
        self.seed_sha256 = hashlib.sha256(settings.seed_ontology_path.read_bytes()).hexdigest()

    def inventory(self, limit: int = 5000) -> list[OntologyElement]:
        classes = set(self.seed.subjects(RDF.type, OWL.Class)) | set(self.seed.subjects(RDF.type, RDFS.Class))
        props = (
            set(self.seed.subjects(RDF.type, RDF.Property)) |
            set(self.seed.subjects(RDF.type, OWL.ObjectProperty)) |
            set(self.seed.subjects(RDF.type, OWL.DatatypeProperty))
        )
        out = [self._element(uri, "class") for uri in classes] + [self._element(uri, "property") for uri in props]
        return sorted(out, key=lambda x: (x.kind, x.label.lower()))[:limit]

    def search(self, terms: list[str], limit: int = 40, kind: str | None = None) -> list[OntologyElement]:
        needles = [self._norm(t) for t in terms if self._norm(t)]
        scored: list[tuple[float, OntologyElement]] = []
        for item in self.inventory():
            if kind and item.kind != kind:
                continue
            label = self._norm(item.label)
            desc = self._norm(item.description or "")
            uri = self._norm(item.uri)
            exact = sum(3.0 for t in needles if t == label)
            contains = sum(1.5 for t in needles if t in label or t in desc or t in uri)
            fuzzy = max((SequenceMatcher(None, t, label).ratio() for t in needles), default=0.0)
            score = exact + contains + fuzzy
            if score >= 0.35:
                scored.append((score, item))
        scored.sort(key=lambda x: (-x[0], x[1].label.lower()))
        return [x for _, x in scored[:limit]]

    def inspect(self, uri: str) -> OntologyElement:
        ref = URIRef(uri)
        is_prop = any(any(self.seed.triples((ref, RDF.type, t))) for t in (RDF.Property, OWL.ObjectProperty, OWL.DatatypeProperty))
        if not is_prop and not any(self.seed.triples((ref, RDF.type, OWL.Class))) and not any(self.seed.triples((ref, RDF.type, RDFS.Class))):
            raise KeyError(uri)
        return self._element(ref, "property" if is_prop else "class")

    def dependency_closure(self, selected_uris: list[str]) -> set[str]:
        closure = set(selected_uris)
        frontier = [URIRef(x) for x in selected_uris]
        structural = {RDFS.domain, RDFS.range, RDFS.subClassOf, RDFS.subPropertyOf, OWL.equivalentClass, OWL.equivalentProperty, OWL.inverseOf}
        while frontier:
            node = frontier.pop()
            for pred in structural:
                for obj in self.seed.objects(node, pred):
                    if isinstance(obj, URIRef) and str(obj) not in closure:
                        closure.add(str(obj)); frontier.append(obj)
                for subj in self.seed.subjects(pred, node):
                    # Only pull inverse/equivalent terms; avoid exploding all children of broad classes.
                    if pred in {OWL.inverseOf, OWL.equivalentClass, OWL.equivalentProperty} and str(subj) not in closure:
                        closure.add(str(subj)); frontier.append(subj)
        return closure

    def build_snapshot(
        self,
        selected_uris: list[str],
        proposals: list[OntologyProposal],
        path: Path,
        *,
        run_id: str,
    ) -> SubOntologyArtifact:
        graph = Graph()
        closure = self.dependency_closure(selected_uris)
        selected: list[OntologyElement] = []
        for uri_s in sorted(closure):
            uri = URIRef(uri_s)
            for triple in self.seed.triples((uri, None, None)):
                graph.add(triple)
            # Bring labels for external domain/range/dependency terms when present in the seed graph.
            for triple in self.seed.triples((None, None, uri)):
                if triple[1] in {RDFS.domain, RDFS.range, RDFS.subClassOf, RDFS.subPropertyOf, OWL.inverseOf}:
                    graph.add(triple)
            try:
                element = self.inspect(uri_s)
                if uri_s in selected_uris:
                    selected.append(element)
            except KeyError:
                pass
        warnings: list[str] = []
        for proposal in proposals:
            if proposal.status != OntologyProposalStatus.ACCEPTED_FOR_RUN:
                continue
            try:
                self._validate_proposal(proposal, run_id)
                self._apply_proposal(graph, proposal)
            except ValueError as exc:
                warnings.append(str(exc))
        path.parent.mkdir(parents=True, exist_ok=True)
        graph.serialize(path, format="turtle")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return SubOntologyArtifact(
            turtle_path=str(path), seed_schema_version=SCHEMA_VERSION, selected_elements=selected,
            dependency_uris=sorted(closure - set(selected_uris)), proposals=proposals,
            validation_warnings=warnings, content_sha256=digest,
        )

    # Compatibility name for older callers.
    def build_subontology(self, selected_uris: list[str], proposals: list[OntologyProposal], path: Path, run_id: str) -> SubOntologyArtifact:
        return self.build_snapshot(selected_uris, proposals, path, run_id=run_id)

    def graph_from_artifact(self, artifact: SubOntologyArtifact) -> Graph:
        g = Graph()
        g.parse(artifact.turtle_path, format="turtle")
        return g

    def relation_catalog(self, artifact: SubOntologyArtifact) -> list[OntologyElement]:
        graph = self.graph_from_artifact(artifact)
        props = set(graph.subjects(RDF.type, RDF.Property)) | set(graph.subjects(RDF.type, OWL.ObjectProperty)) | set(graph.subjects(RDF.type, OWL.DatatypeProperty))
        return [self._element_from_graph(graph, uri, "property", "seed" if str(uri).startswith(DBLP) else "run_overlay") for uri in sorted(props, key=str)]

    def class_catalog(self, artifact: SubOntologyArtifact) -> list[OntologyElement]:
        graph = self.graph_from_artifact(artifact)
        classes = set(graph.subjects(RDF.type, OWL.Class)) | set(graph.subjects(RDF.type, RDFS.Class))
        return [self._element_from_graph(graph, uri, "class", "seed" if str(uri).startswith(DBLP) else "run_overlay") for uri in sorted(classes, key=str)]

    def make_proposal_uri(self, run_id: str, label: str) -> str:
        slug = quote(re.sub(r"\s+", "_", label.strip()))
        return f"{RUN_OVERLAY_BASE}{quote(run_id)}/ontology#{slug}"

    def _validate_proposal(self, proposal: OntologyProposal, run_id: str) -> None:
        if not self.settings.allow_ontology_extensions:
            raise ValueError(f"Ontology extensions disabled; proposal {proposal.proposal_id} cannot be accepted")
        if proposal.uri.startswith(DBLP):
            raise ValueError("Run-local extensions may not mint or mutate URIs in the DBLP namespace")
        expected_prefix = f"{RUN_OVERLAY_BASE}{run_id}/ontology#"
        if not proposal.uri.startswith(expected_prefix):
            raise ValueError(f"Run-local proposal URI must be inside {expected_prefix}")
        if proposal.type == "property" and not proposal.definition.strip():
            raise ValueError("Property proposal requires a definition")

    @staticmethod
    def _apply_proposal(graph: Graph, proposal: OntologyProposal) -> None:
        uri = URIRef(proposal.uri)
        graph.add((uri, RDF.type, RDF.Property if proposal.type == "property" else OWL.Class))
        graph.add((uri, RDFS.label, Literal(proposal.label)))
        graph.add((uri, RDFS.comment, Literal(proposal.definition)))
        if proposal.domain:
            graph.add((uri, RDFS.domain, URIRef(proposal.domain)))
        if proposal.range:
            graph.add((uri, RDFS.range, URIRef(proposal.range)))

    def _element(self, uri: URIRef, kind: str) -> OntologyElement:
        return self._element_from_graph(self.seed, uri, kind, "seed")

    @staticmethod
    def _element_from_graph(g: Graph, uri: URIRef, kind: str, source: str) -> OntologyElement:
        local = str(uri).rsplit("/", 1)[-1].rsplit("#", 1)[-1]
        label = next(iter(g.objects(uri, RDFS.label)), Literal(local))
        comment = next(iter(g.objects(uri, RDFS.comment)), None)
        domain = next(iter(g.objects(uri, RDFS.domain)), None)
        range_ = next(iter(g.objects(uri, RDFS.range)), None)
        parents = [str(x) for x in g.objects(uri, RDFS.subPropertyOf if kind == "property" else RDFS.subClassOf)]
        equivalent_pred = OWL.equivalentProperty if kind == "property" else OWL.equivalentClass
        equivalents = [str(x) for x in g.objects(uri, equivalent_pred)]
        return OntologyElement(
            uri=str(uri), label=str(label), kind=kind, description=str(comment) if comment else None,
            domain=str(domain) if domain else None, range=str(range_) if range_ else None,
            parents=parents, equivalents=equivalents, source=source,
        )

    @staticmethod
    def _norm(text: str) -> str:
        return " ".join(re.findall(r"[a-z0-9]+", text.lower()))
