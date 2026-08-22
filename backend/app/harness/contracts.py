from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.models.domain import AgentSpec, ArtifactMetadata

SUPPORTED_CAPABILITIES = frozenset({
    "documents", "ontology", "sparql", "edc", "verification", "entity_normalization",
    "artifacts", "shell", "filesystem",
})


class ArtifactLookup(Protocol):
    def get_metadata(self, artifact_id: str) -> ArtifactMetadata: ...

    def get(self, artifact_id: str, *, envelope: bool = False): ...


class SpecialistDeps(Protocol):
    artifacts: ArtifactLookup


MAX_MODEL_ERROR_CHARS = 512


def bounded_error(message: object, *, limit: int = MAX_MODEL_ERROR_CHARS) -> str:
    text = str(message).replace("\x00", " ").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


class ScientificStateBlocked(Exception):
    """Expected missing scientific state; control returns to the Director."""

    def __init__(self, message: str, *, requirements: tuple[str, ...] = ()) -> None:
        super().__init__(bounded_error(message))
        self.requirements = requirements


class ArtifactNotFound(KeyError):
    """A referenced artifact does not exist in the current run."""


class InvalidArtifactKind(ValueError):
    """A reference resolves, but not to the kind required by the operation."""


class ArtifactIntegrityError(ValueError):
    """Stored payload violates its declared schema; never retry as model input."""


@dataclass(frozen=True)
class CandidateLineage:
    candidate_id: str
    evidence_bundle_id: str
    ontology_snapshot_id: str
    open_triples_id: str
    relation_definitions_id: str


def metadata(deps: SpecialistDeps, artifact_id: str) -> ArtifactMetadata:
    try:
        return deps.artifacts.get_metadata(artifact_id)
    except KeyError as exc:
        raise ArtifactNotFound(bounded_error(f"Artifact {artifact_id} was not found.")) from exc


def require_kind(deps: SpecialistDeps, artifact_id: str, expected: str) -> ArtifactMetadata:
    meta = metadata(deps, artifact_id)
    if meta.artifact_type != expected:
        raise InvalidArtifactKind(
            bounded_error(f"Artifact {artifact_id}: expected {expected}, got {meta.artifact_type}.")
        )
    return meta


def load_payload(deps: SpecialistDeps, artifact_id: str, expected: str, model):
    require_kind(deps, artifact_id, expected)
    try:
        return model.model_validate(deps.artifacts.get(artifact_id))
    except Exception as exc:
        raise ArtifactIntegrityError(
            bounded_error(f"Artifact {artifact_id} declares {expected} but fails its stored schema.")
        ) from exc


def validate_specialist_spec(deps: SpecialistDeps, spec: AgentSpec) -> None:
    """Reject invalid scientific handoffs before a specialist is constructed."""
    unknown = sorted(set(spec.required_capabilities) - set(SUPPORTED_CAPABILITIES))
    if unknown:
        raise ValueError(
            bounded_error(
                "Unknown specialist capabilities: " + ", ".join(unknown)
                + ". Valid capabilities are: " + ", ".join(sorted(SUPPORTED_CAPABILITIES))
                + ". Artifact kinds such as evidence_span are not capabilities."
            )
        )

    for artifact_id in spec.input_artifact_ids:
        metadata(deps, artifact_id)

    if "verification" in spec.required_capabilities and not any(
        metadata(deps, aid).artifact_type == "candidate_canonical_triples" for aid in spec.input_artifact_ids
    ):
        raise ScientificStateBlocked(
            "Verification requires candidate_canonical_triples.", requirements=("candidate_canonical_triples",)
        )


def resolve_candidate_lineage(deps: SpecialistDeps, candidate_id: str) -> CandidateLineage:
    candidate = require_kind(deps, candidate_id, "candidate_canonical_triples")
    parents = {metadata(deps, aid).artifact_type: aid for aid in candidate.parent_artifact_ids}
    required = {"evidence_bundle", "ontology_snapshot", "edc_open_triples", "edc_relation_definitions"}
    missing = sorted(required - set(parents))
    if missing:
        raise ScientificStateBlocked(
            "Candidate canonical triples lack required correlated lineage: " + ", ".join(missing),
            requirements=tuple(missing),
        )
    return CandidateLineage(
        candidate_id=candidate_id,
        evidence_bundle_id=parents["evidence_bundle"],
        ontology_snapshot_id=parents["ontology_snapshot"],
        open_triples_id=parents["edc_open_triples"],
        relation_definitions_id=parents["edc_relation_definitions"],
    )
