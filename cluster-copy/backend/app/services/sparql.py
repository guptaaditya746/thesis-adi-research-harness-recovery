from __future__ import annotations

from rdflib import Graph
from rdflib.plugins.sparql.parser import parseQuery

from app.models.domain import SPARQLArtifact, SubOntologyArtifact


class SPARQLService:
    def validate_and_preview(self, artifact: SPARQLArtifact, ontology: SubOntologyArtifact) -> SPARQLArtifact:
        try:
            parseQuery(artifact.query)
        except Exception as exc:
            return artifact.model_copy(update={"valid": False, "validation_error": str(exc)})
        g = Graph()
        g.parse(ontology.turtle_path, format="turtle")
        try:
            result = g.query(artifact.query)
            rows: list[dict[str, str]] = []
            for row in list(result)[:20]:
                rows.append({str(var): str(value) for var, value in zip(result.vars, row) if value is not None})
            return artifact.model_copy(update={"valid": True, "validation_error": None, "preview_rows": rows})
        except Exception as exc:
            return artifact.model_copy(update={"valid": False, "validation_error": str(exc)})
