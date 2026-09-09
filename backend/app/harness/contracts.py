from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.models.domain import (
    AgentSpec, ArtifactMetadata, CanonicalTriple, EDCCanonicalizationResult,
    EDCOpenTriples, EDCRelationDefinitions, EvidenceBundle, SubOntologyArtifact,
)

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
GENERIC_ARTIFACT_IDS = frozenset({
    "artifact", "artifact_id", "candidate_canonical_triples", "canonical_triples",
    "evidence_bundle", "ontology_snapshot", "default_ontology_snapshot",
})


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


class OntologyArtifactIntegrityError(ArtifactIntegrityError):
    """An accepted ontology proposal was not faithfully represented in its snapshot."""


class CorrectableToolInputError(ValueError):
    """A bounded model-supplied tool argument error; reread/correct, never escalate."""

    def __init__(self, message: str) -> None:
        super().__init__(bounded_error(message))


@dataclass(frozen=True)
class CandidateLineage:
    candidate_id: str
    evidence_bundle_id: str
    ontology_snapshot_id: str
    open_triples_id: str
    relation_definitions_id: str


def metadata(deps: SpecialistDeps, artifact_id: str) -> ArtifactMetadata:
    if artifact_id in GENERIC_ARTIFACT_IDS or artifact_id.startswith(("default_", "placeholder_")):
        raise ArtifactNotFound(
            bounded_error(f"Artifact reference {artifact_id!r} is a generic placeholder, not a backend-issued ID.")
        )
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

def validate_artifact_schema(deps: SpecialistDeps, artifact_id: str):
    meta = metadata(deps, artifact_id)
    if meta.artifact_type == "ontology_snapshot":
        return load_payload(deps, artifact_id, meta.artifact_type, SubOntologyArtifact)
    if meta.artifact_type == "evidence_bundle":
        return load_payload(deps, artifact_id, meta.artifact_type, EvidenceBundle)
    if meta.artifact_type == "edc_open_triples":
        return load_payload(deps, artifact_id, meta.artifact_type, EDCOpenTriples)
    if meta.artifact_type == "edc_relation_definitions":
        return load_payload(deps, artifact_id, meta.artifact_type, EDCRelationDefinitions)
    if meta.artifact_type == "edc_canonicalization":
        return load_payload(deps, artifact_id, meta.artifact_type, EDCCanonicalizationResult)
    if meta.artifact_type in {"candidate_canonical_triples", "canonical_triples"}:
        try:
            values = [CanonicalTriple.model_validate(item) for item in deps.artifacts.get(artifact_id)]
        except Exception as exc:
            raise ArtifactIntegrityError(
                bounded_error(f"Artifact {artifact_id} declares {meta.artifact_type} but fails its stored schema.")
            ) from exc
        return values
    return deps.artifacts.get(artifact_id)


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

    input_metadata = [metadata(deps, artifact_id) for artifact_id in spec.input_artifact_ids]
    input_ids_by_kind: dict[str, list[str]] = {}
    for item in input_metadata:
        input_ids_by_kind.setdefault(item.artifact_type, []).append(item.artifact_id)

    required_kinds: set[str] = set()
    if "edc" in spec.required_capabilities or spec.task_scope.startswith("edc_"):
        required_kinds.update({"evidence_bundle", "ontology_snapshot"})
    if "verification" in spec.required_capabilities or spec.task_scope == "verification":
        required_kinds.add("candidate_canonical_triples")
    missing = sorted(required_kinds - set(input_ids_by_kind))
    if missing:
        raise ScientificStateBlocked(
            f"{spec.task_scope} specialist prerequisites are missing: {', '.join(missing)}. "
            "Persist the artifacts and include their exact IDs in input_artifact_ids before delegating.",
            requirements=tuple(missing),
        )
    ambiguous = sorted(kind for kind in required_kinds if len(input_ids_by_kind[kind]) != 1)
    if ambiguous:
        raise ScientificStateBlocked(
            f"{spec.task_scope} specialist prerequisites are ambiguous: {', '.join(ambiguous)}. "
            "Include exactly one persisted artifact ID for each prerequisite kind.",
            requirements=tuple(ambiguous),
        )

    # Validate prerequisite payloads before a delegation is counted. A stored
    # artifact with the right kind but the wrong schema is not a valid handoff.
    if "evidence_bundle" in required_kinds:
        load_payload(deps, input_ids_by_kind["evidence_bundle"][0], "evidence_bundle", EvidenceBundle)
    if "ontology_snapshot" in required_kinds:
        load_payload(deps, input_ids_by_kind["ontology_snapshot"][0], "ontology_snapshot", SubOntologyArtifact)
    if "candidate_canonical_triples" in required_kinds:
        candidate_id = input_ids_by_kind["candidate_canonical_triples"][0]
        try:
            [CanonicalTriple.model_validate(item) for item in deps.artifacts.get(candidate_id)]
        except Exception as exc:
            raise ArtifactIntegrityError(
                bounded_error(f"Artifact {candidate_id} declares candidate_canonical_triples but fails its stored schema.")
            ) from exc
        resolve_candidate_lineage(deps, candidate_id)

    # Ontology construction can remain agentic, but its declared scope must at
    # least admit one inspect/search action plus batch create, batch accept, and
    # snapshot materialization. This is a feasibility guard, not a workflow.
    if spec.task_scope == "ontology_snapshot" and "ontology" in spec.required_capabilities and spec.max_tool_calls < 4:
        raise CorrectableToolInputError(
            "ontology_snapshot specialists require max_tool_calls >= 4: inspect/search, batch proposal creation, "
            "batch acceptance, and snapshot materialization."
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
    for artifact_id in parents.values():
        validate_artifact_schema(deps, artifact_id)
    return CandidateLineage(
        candidate_id=candidate_id,
        evidence_bundle_id=parents["evidence_bundle"],
        ontology_snapshot_id=parents["ontology_snapshot"],
        open_triples_id=parents["edc_open_triples"],
        relation_definitions_id=parents["edc_relation_definitions"],
    )
