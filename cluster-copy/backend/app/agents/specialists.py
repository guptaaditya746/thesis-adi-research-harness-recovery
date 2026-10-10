from __future__ import annotations

from pydantic_ai import Agent

from app.agents.model_factory import build_model
from app.config import Settings
from app.harness.deps import HarnessDeps
from app.tools.research import register_research_tools
from app.tools.extensions import register_configured_extensions


def _agent(settings: Settings, name: str, description: str, instructions: str, fast: bool = False) -> Agent:
    agent = Agent(
        build_model(settings, fast=fast),
        deps_type=HarnessDeps,
        name=name,
        description=description,
        instructions=instructions,
        model_settings={"temperature": settings.llm_temperature, "max_tokens": settings.llm_max_tokens},
        retries=settings.llm_output_retries,
    )
    register_research_tools(agent)
    register_configured_extensions(agent, settings)
    return agent


def build_specialists(settings: Settings) -> list[Agent]:
    return [
        _agent(settings, "paper_explorer", "Navigates papers and gathers source-grounded evidence.",
               "Use section/page/FTS tools aggressively. Do not use embeddings. Return concise findings with evidence IDs or exact page/section anchors. Never invent source support.", True),
        _agent(settings, "ontology_specialist", "Adapts the DBLP seed into a question-relevant ontology overlay.",
               "Inspect seed terms first. Prefer reuse over extension. When extension is necessary, propose small explicit changes with rationale. Keep the active ontology question-specific."),
        _agent(settings, "sparql_specialist", "Creates and repairs SPARQL against the active ontology.",
               "SPARQL here captures semantic intent over the seed/sub-ontology. Validate referenced terms and repair syntax/namespace errors. It does not query the PDF itself."),
        _agent(settings, "evidence_critic", "Challenges evidence sufficiency and looks for contradictory or missing source material.",
               "Inspect selected evidence, seek missing context and contradictions, and report whether extraction can proceed. Cite source anchors, not hidden reasoning."),
        _agent(settings, "edc_specialist", "Runs and interprets EDC Extract-Define-Canonicalize artifacts.",
               "Use EDC only after evidence is adequate. Keep open relations distinct from canonical ontology relations. Never silently rewrite unsupported relations."),
        _agent(settings, "triple_auditor", "Audits triples, provenance, RDF and SHACL conformance.",
               "Verify each candidate triple against cited evidence and ontology semantics. Reject unsupported triples. Distinguish factual support from structural SHACL validity."),
    ]
