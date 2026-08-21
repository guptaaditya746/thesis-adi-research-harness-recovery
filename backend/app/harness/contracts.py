from __future__ import annotations

from typing import Protocol

from app.models.domain import AgentSpec

SUPPORTED_CAPABILITIES = frozenset({
    "documents", "ontology", "sparql", "edc", "verification", "entity_normalization",
    "artifacts", "shell", "filesystem",
})


class ArtifactLookup(Protocol):
    def get_envelope(self, artifact_id: str): ...


class SpecialistDeps(Protocol):
    artifacts: ArtifactLookup


def validate_specialist_spec(deps: SpecialistDeps, spec: AgentSpec) -> None:
    """Reject invalid scientific handoffs before a specialist is constructed."""
    unknown = sorted(set(spec.required_capabilities) - set(SUPPORTED_CAPABILITIES))
    if unknown:
        raise ValueError(
            "Unknown specialist capabilities: " + ", ".join(unknown)
            + ". Valid capabilities are: " + ", ".join(sorted(SUPPORTED_CAPABILITIES))
            + ". Artifact kinds such as evidence_span are not capabilities."
        )

    kinds: dict[str, str] = {}
    missing: list[str] = []
    for artifact_id in spec.input_artifact_ids:
        try:
            kinds[artifact_id] = deps.artifacts.get_envelope(artifact_id).artifact_type
        except KeyError:
            missing.append(artifact_id)
    if missing:
        raise ValueError(
            "AgentSpec.input_artifact_ids must contain persisted ArtifactStore IDs. "
            f"These IDs do not resolve: {missing}. Retrieval candidates must be handed off "
            "through a retrieval_candidate_bundle artifact."
        )

    required_kinds: dict[str, set[str]] = {}
    if "edc" in spec.required_capabilities:
        required_kinds["edc"] = {"evidence_bundle"}
    if "verification" in spec.required_capabilities:
        required_kinds["verification"] = {"candidate_canonical_triples", "evidence_bundle"}

    for capability, required in required_kinds.items():
        absent = sorted(required - set(kinds.values()))
        if absent:
            raise ValueError(
                f"A {capability} specialist requires artifact kinds: {', '.join(sorted(required))}. "
                f"Missing: {', '.join(absent)}. Persist the required scientific artifacts first."
            )
