# Similar GitHub Projects

Checked on 2026-09-09. The target used for comparison is this harness's document-to-triple workflow: parse scientific documents, extract evidence-grounded triples, align them to an ontology/schema, validate the result, and retain useful partial output.

## Recommended shortlist

| Rank | Repository | Why it is relevant | Main limitation for this harness |
|---|---|---|---|
| 1 | [docling-project/docling-graph](https://github.com/docling-project/docling-graph) | The closest recent end-to-end implementation: document conversion, chunking, schema-defined extraction, validation, provenance, graph deduplication, and JSON/CSV/Cypher exports. It supports local and remote models through LiteLLM. | Its Pydantic-template graph model is more schema-first than this harness's open extraction followed by ontology alignment. |
| 2 | [screemix/Wikontic](https://github.com/screemix/Wikontic) | A recent ontology-aware pipeline that extracts SPO triples, links and reranks entities/relations, validates against Wikidata's ontology, and preserves final/filtered variants. | It relies on embeddings/vector search and Wikidata-specific infrastructure. |
| 3 | [DataScienceLabFHSWF/KnowledgeGraphBuilder](https://github.com/DataScienceLabFHSWF/KnowledgeGraphBuilder) | Very close research architecture: document ingestion, OWL-guided extraction, text alignment/provenance, deterministic and LLM validation, SHACL, checkpoints, and experiment artifacts. | Young and small project; better as an architecture reference than a dependency to adopt wholesale. |
| 4 | [monarch-initiative/ontogpt](https://github.com/monarch-initiative/ontogpt) | Mature structured extraction with ontology grounding and LiteLLM support. It is especially relevant because this harness already calls OntoGPT. | It is an extraction component, not a complete benchmark/orchestration layer; its prose output must still be normalized defensively. |
| 5 | [google/langextract](https://github.com/google/langextract) | Strong component for structured extraction whose results are grounded to exact source spans. This directly addresses unverifiable triples and loose evidence strings. | It does not provide ontology induction or a complete KG research workflow. |
| 6 | [HKUST-KnowComp/AutoSchemaKG](https://github.com/HKUST-KnowComp/AutoSchemaKG) | Separates triple extraction from automatic schema induction/conceptualization and includes evaluation code. | Broader schema-generation goal and heavier graph/embedding assumptions than this harness needs. |
| 7 | [HeraclesWang/OSKGC](https://github.com/HeraclesWang/OSKGC) | A benchmark specifically for ontology-schema-based KG construction, evaluating triple accuracy and schema conformity. | Benchmark/reference data rather than a production pipeline. |

## Useful adjacent projects

- [Future-House/paper-qa](https://github.com/Future-House/paper-qa): strong scientific-document retrieval, evidence citation, contradiction handling, and agent orchestration. It is useful for question/evidence selection, but is not a triple-extraction system.
- [neo4j-labs/llm-graph-builder](https://github.com/neo4j-labs/llm-graph-builder): mature document-to-Neo4j application with custom schemas and source metadata. Its persistent Neo4j-centered design is unlike the harness's artifact-first design.
- [clear-nus/edc](https://github.com/clear-nus/edc): the clearest academic baseline for Extract-Define-Canonicalize. Its extraction to open triples followed by schema definition and canonicalization closely matches the conceptual workflow, but the repository has not been active recently.
- [OpenSPG/KAG](https://github.com/OpenSPG/KAG): schema-constrained construction, normalization, source-text indexing, and hybrid reasoning at a larger platform scale.

## What to reuse first

1. Adopt Docling Graph's typed stage boundaries, resumable artifacts, provenance fields, and explicit validation reports.
2. Keep candidate, accepted, and rejected triples separately, following Wikontic's staged outputs. Empty accepted output should not erase valid candidates or diagnostics.
3. Ground each accepted triple to an exact source span, following LangExtract and KnowledgeGraphBuilder.
4. Replace the current five-question smoke run as the main quality claim with gold triple and schema-conformity scoring modeled after OSKGC.
5. Treat OntoGPT output as untrusted model output: parse only valid triples, store rejected prose separately, and continue with partial valid results.

## Bottom line

For a fast rescue, do not replace the whole harness. Use **Docling Graph** as the primary architectural reference, **LangExtract** for evidence grounding ideas, **Wikontic/EDC** for the open-to-canonicalized triple lifecycle, and **OSKGC** for meaningful evaluation.
