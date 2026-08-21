# Architecture

## 1. Scientific product and invariants

The research product is a versioned list of accepted `CanonicalTriple` records plus provenance, ontology snapshot, proposals, SPARQL semantic patterns, retrieval trace, EDC artifacts, verification and evaluation. A temporary RDF graph may be created in memory for ontology or SHACL operations, and the frontend may derive a temporary triple map, but no persistent instance graph is written.

Deterministic completion invariants are enforced in `harness/runtime.py`: an accepted triple needs a canonical predicate URI, accepted independent verification and one or more evidence IDs; each evidence ID must resolve to an immutable `evidence_span` artifact and validate against the current source `DocumentIR`.

## 2. Goal-oriented agent harness

`ResearchHarness` creates one PydanticAI Research Director. Its direct tools are intentionally meta-level: inspect state/artifacts, perform limited orientation searches, record a concise `DecisionRecord`, create a specialist, run bounded fan-out/dynamic workflow and persist an explicit partial result.

`AgentSpec` is generated at runtime and records role, objective, generated instructions, requested capabilities, input artifact IDs, output schema, model profile, context/tool budget, parent agent and creation reason. `AgentFactory` creates a fresh isolated PydanticAI agent from the spec. No specialist role list is hard-coded.

Specialist toolsets are composed dynamically from `FunctionToolset` capabilities: documents, ontology, SPARQL, EDC, verification, entity normalization and artifact inspection. Shell/filesystem access is separately bounded by Harness and the run workspace. The Director's full history is never copied to child agents.

Harness DynamicWorkflow is an optional capability for a Director-selected bounded fan-out/chaining episode; it does not define the research lifecycle. Harness StepPersistence stores resumable agent steps; SQLite stores run state and events.

## 3. Models and LiteLLM

All LLM traffic goes through the configured OpenAI-compatible LiteLLM endpoint. Application code uses profile names (`fast`, `balanced`, `reasoning`, `critic`, `extractor`) rather than provider model IDs. Profiles can all resolve to the same single gateway model when only one model is available. Call settings are environment-driven.

Structured output/tool validation retries are bounded. Specialist execution adds a recovery ladder with context/task narrowing guidance and bounded model-profile escalation. Failed specialists return typed partial outputs to the Director rather than fabricating results.

## 4. Scientific corpus and retrieval

`ParserRouter` normalizes GROBID, Docling or PyMuPDF output into `ScientificDocument`/DocumentIR with metadata, sections, blocks, references and source positions. Parser-specific types never enter agent interfaces.

`RetrievalService` uses local SQLite FTS5/BM25 and structural reads. Tools include document/section/heading/reference/entity searches, page/section/block reads and context expansion. Evidence is persisted as immutable `EvidenceSpan` artifacts with document, page, section, block, text/hash, offsets when available, retrieval query/method and discovering agent. Model context limits may shorten previews, but the evidence artifact itself is never truncated.

No vector store or embedding retrieval dependency exists in the base backend.

## 5. Ontology workspace and SPARQL

`OntologyService` loads `seed/dblp_schema_2024-06-14.ttl`, the supplied official DBLP RDF schema. The seed is never modified. A run snapshot contains selected seed terms, dependency closure and only explicitly accepted run-local `OntologyProposal` records. Proposal URIs must be in that run's overlay namespace; attempts to mint in DBLP are rejected.

SPARQL is an inspectable semantic graph pattern generated from the question and active ontology. It can validate/query ontology structure where meaningful; it never implies that raw PDF facts already form an instance RDF store.

## 6. EDC adaptation

EDC is invoked only after an evidence bundle exists. The adaptation preserves:

1. **Extract** — produce open triples tied to valid evidence IDs.
2. **Define** — create textual definitions for each open predicate.
3. **Canonicalize** — rank active ontology relations using normalized lexical similarity, token/definition overlap, domain/range compatibility and hierarchy cues, then ask a bounded verifier to select among the shortlist.

Every decision persists the open predicate, definition, ranked candidates, scores/reasons, selected/rejected URIs and evidence IDs. If no relation is defensible, EDC produces an explicit ontology proposal and leaves the triple open; it does not silently extend the ontology. The upstream EDC embedding retriever remains isolated in `edc_upstream.py` for optional ablation experiments only.

## 7. Verification and validation

`TripleVerificationService` independently evaluates evidence support, ontology compatibility, direction, structural validity, negation/modality and entity normalization. Accepted and rejected triples are persisted separately. `StructuralValidationService` builds an ephemeral RDF graph solely to run RDF/SHACL checks against the active ontology; this is a structural test, not evidence of truth.

## 8. Persistence and reproducibility

`ArtifactStore` writes immutable JSON envelopes containing artifact ID/type, run/creator, parents, timestamp, schema version, content hash, optional model metadata and content. Reusing an artifact ID with different content is rejected.

`RunRegistry` uses SQLite WAL + FULL synchronous mode for durable run/event indexes and also writes human-readable `run.json`. Run-local artifacts, selected ontology, prompts, AgentSpecs, evidence, EDC decisions, verification, rejected/accepted triples, evaluation and trace IDs form the reproducibility record. Previous runs never automatically enter a new run's context.

## 9. Observability and secret policy

`Telemetry` creates manual OpenTelemetry spans at application boundaries and initializes Langtrace when configured. Trace payloads are recursively redacted for key/token/authorization/secret/password fields and bounded with a hash. Trace exporter failure is swallowed by design and cannot fail a research run.

User-visible `AgentTraceEvent` records expose messages, generated specialist prompts, model/tool boundaries, arguments/results, decisions, recovery, persistence and artifact IDs. Event handling explicitly ignores any framework event named as a thinking event; hidden chain-of-thought is neither logged nor displayed.

## 10. UI

The Next.js console provides documents/parsing status, research chat, expandable live research trace, immutable artifact inspector, evidence-to-PDF navigation, active ontology/SPARQL artifacts, accepted triples, evaluation/run history and a temporary run triple map. The UI deliberately avoids a fixed workflow diagram.
