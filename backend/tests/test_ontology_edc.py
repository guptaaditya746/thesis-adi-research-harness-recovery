from pathlib import Path

from app.models.domain import CandidateTriple, OntologyElement, OntologyProposal, OntologyProposalStatus
from app.services.edc import EDCService
from app.services.ontology import DBLP, OntologyService


def test_run_overlay_cannot_mutate_dblp_namespace(settings, tmp_path: Path):
    service = OntologyService(settings)
    proposal = OntologyProposal(
        proposed_by="agent", question="q", type="property", uri=DBLP + "inventedRelation",
        label="invented relation", definition="x relates to y", rationale="test",
        status=OntologyProposalStatus.ACCEPTED_FOR_RUN,
    )
    snapshot = service.build_snapshot([], [proposal], tmp_path / "active.ttl", run_id="r1")
    assert snapshot.validation_warnings
    assert "DBLP namespace" in snapshot.validation_warnings[0]


def test_valid_extension_is_run_local(settings, tmp_path: Path):
    service = OntologyService(settings)
    uri = service.make_proposal_uri("r1", "uses method")
    proposal = OntologyProposal(
        proposed_by="agent", question="q", type="property", uri=uri, label="uses method",
        definition="The subject uses the method in the object.", rationale="paper evidence",
        status=OntologyProposalStatus.ACCEPTED_FOR_RUN,
    )
    snapshot = service.build_snapshot([], [proposal], tmp_path / "active.ttl", run_id="r1")
    assert not snapshot.validation_warnings
    assert any(x.uri == uri for x in service.relation_catalog(snapshot))
    assert snapshot.seed_ontology_sha256 == service.seed_sha256
    assert snapshot.snapshot_ttl_sha256 == snapshot.content_sha256
    assert snapshot.snapshot_builder_version


def test_edc_candidate_ranking_is_deterministic_non_embedding(settings):
    service = EDCService(settings)
    triple = CandidateTriple(subject="Paper", predicate="authored by", object="Ada", evidence_ids=["ev1"])
    relations = [
        OntologyElement(uri=DBLP+"authoredBy", label="authoredBy", kind="property", description="The publication is authored by the creator."),
        OntologyElement(uri=DBLP+"doi", label="doi", kind="property", description="A Digital Object Identifier."),
    ]
    ranked = service.rank_candidates(triple, "The subject publication was authored by the object creator.", relations)
    assert ranked[0].uri == DBLP + "authoredBy"
    assert any("label_similarity" in reason for reason in ranked[0].reasons)
