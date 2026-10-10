from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings
from app.services.edc import EDCService
from app.services.entities import EntityNormalizationService
from app.services.evaluation import EvaluationService
from app.services.ontology import OntologyService
from app.services.pdf import PDFService
from app.services.retrieval import RetrievalService
from app.services.sparql import SPARQLService
from app.services.validation import StructuralValidationService
from app.services.verification import TripleVerificationService
from app.storage.artifacts import ArtifactStore
from app.storage.registry import RunRegistry
from app.telemetry.tracing import Telemetry


@dataclass
class HarnessDeps:
    settings: Settings
    run_id: str
    workspace_id: str
    document_ids: list[str]
    artifacts: ArtifactStore
    registry: RunRegistry
    pdf: PDFService
    retrieval: RetrievalService
    ontology: OntologyService
    sparql: SPARQLService
    edc: EDCService
    entities: EntityNormalizationService
    verification: TripleVerificationService
    validation: StructuralValidationService
    evaluation: EvaluationService
    telemetry: Telemetry

    def ensure_not_cancelled(self) -> None:
        if self.registry.is_cancelled(self.run_id):
            raise RuntimeError(f"Research run {self.run_id} was cancelled")
