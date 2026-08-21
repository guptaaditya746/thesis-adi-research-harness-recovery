from pathlib import Path

from rdflib import Graph, RDF, URIRef

from app.config import Settings


def test_comma_separated_frontend_origins(tmp_path: Path):
    settings = Settings(_env_file=None, data_dir=tmp_path, frontend_origin="http://localhost:3000,http://example.test:3000")
    assert settings.frontend_origins == ["http://localhost:3000", "http://example.test:3000"]


def test_model_profiles_can_collapse_to_single_gateway(tmp_path: Path):
    settings = Settings(_env_file=None, data_dir=tmp_path, litellm_model="only-model", litellm_fast_model="fast-alias")
    assert settings.model_for_profile("balanced") == "only-model"
    assert settings.model_for_profile("reasoning") == "only-model"
    assert settings.model_for_profile("critic") == "only-model"
    assert settings.model_for_profile("fast") == "fast-alias"


def test_bundled_seed_is_official_dblp_schema(settings):
    graph = Graph().parse(settings.seed_ontology_path, format="turtle")
    ontology = URIRef("https://dblp.org/rdf/schema")
    article = URIRef("https://dblp.org/rdf/schema#Article")
    authored_by = URIRef("https://dblp.org/rdf/schema#authoredBy")
    fabricated = URIRef("https://dblp.org/rdf/schema#usesMethod")
    assert any(graph.triples((ontology, None, None)))
    assert any(graph.triples((article, RDF.type, None)))
    assert any(graph.triples((authored_by, RDF.type, RDF.Property)))
    assert not any(graph.triples((fabricated, None, None)))
    text = settings.seed_ontology_path.read_text(encoding="utf-8")
    assert 'owl:versionInfo "Fri 14 Jun 2024"' in text
