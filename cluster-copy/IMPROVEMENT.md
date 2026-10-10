# Architectural Improvement Plan

## Purpose

This document describes how to evolve the repository from a research prototype into a reliable, auditable system for ontology-guided research over a corpus of scientific papers.

It intentionally avoids machine-specific deployment history. Operational commands belong in the README or deployment runbooks; this document focuses on architecture, correctness, research quality, and engineering priorities.

## Product vision

Given a corpus of scientific PDFs and a natural-language research question, the system should let an orchestrating model:

1. understand the research goal and ambiguity;
2. inspect a complete or clearly scoped DBLP ontology;
3. derive a question-specific sub-ontology;
4. propose justified ontology extensions when DBLP is insufficient;
5. retrieve relevant evidence using agentic, non-embedding RAG tools;
6. delegate bounded work to specialist agents;
7. run Extract–Define–Canonicalize (EDC);
8. verify candidate triples against source evidence and ontology semantics;
9. commit only accepted triples to the research graph;
10. answer the question with traceable evidence, assumptions, and uncertainty.

The model should choose and revise the research path. The application should not depend on a large, hard-coded taxonomy of question types. Deterministic modules should enforce safety, provenance, validation, budgets, and durable state around the model’s decisions.

```text
Question + selected corpus
          │
          ▼
  Research orchestrator
          │
          ├── corpus exploration / agentic retrieval
          ├── DBLP inspection and sub-ontology construction
          ├── specialist delegation and evidence criticism
          ├── EDC extraction and canonicalization
          ├── triple and ontology validation
          └── graph query, aggregation, and answer synthesis
          │
          ▼
Answer + verified triples + evidence ledger + complete trace
```

## Architectural assessment

The central idea is sound. The repository already contains the main concepts: an orchestrator, specialist agents, lexical document tools, ontology adaptation, EDC, evidence-aware verification, graph persistence, artifacts, and a live trace.

The current implementation is still a vertical slice. Several interfaces imply capabilities that the implementation does not yet provide. The most important work is therefore not adding more agents or prompts; it is making the existing research loop correct, durable, observable, and testable.

## Critical problems

### 1. The research run has no single durable owner

Runs are launched as untracked in-process asynchronous tasks. The event queue and active run registry live in process memory, while only snapshots are written to files. A restart, reload, crash, or multi-worker deployment can leave a run permanently marked `running`, lose live events, or start competing writers.

Cancellation currently records a flag but the execution loop does not consistently check it or cancel the underlying task. Configured durability backends are described but not actually attached to execution. The resulting interface promises more durability than the implementation provides.

Required direction:

- Introduce a `RunEngine` module with a small interface: start, resume, cancel, inspect, and stream events.
- Give every run one durable owner and one persisted state machine.
- Persist transitions atomically before publishing events.
- Make cancellation cooperative at every model, tool, delegate, and validation boundary.
- Recover interrupted runs explicitly as resumable or failed; never leave stale `running` records.
- Treat multi-process and concurrent execution as design requirements, not later optimizations.

### 2. Orchestration policy is underspecified

Specialist agents are registered, but delegation is optional. The orchestrator can execute the same retrieval, ontology, EDC, and graph tools directly. Consequently, the presence of subagents does not ensure they are used, and the trace does not explain why work was delegated or retained.

The opposite risk also exists: a narrow question can trigger an expensive sequence of ontology, extraction, verification, and graph operations without a clear sufficiency decision. A hard-coded title shortcut now bypasses the graph workflow, which conflicts with the intended all-graph design.

Required direction:

- Keep planning model-driven, but define explicit delegation and sufficiency policies.
- Require delegation when work is independently verifiable, spans many papers, or benefits from critique.
- Permit direct tool use for bounded inspection without bypassing provenance and graph invariants.
- Remove intent-specific shortcuts that produce answers outside the research graph interfaces.
- Let the orchestrator stop or revise a branch when evidence is sufficient, contradictory, or unavailable.
- Record every delegation decision, objective, input artifact, result, retry, and failure.

### 3. Document parsing is not research-paper aware

PDF ingestion currently extracts plain text and infers headings with simple line heuristics. It does not reliably preserve reading order, title and author metadata, tables, figures, captions, equations, footnotes, references, multi-column layout, or scanned-page OCR. Paragraph positions and section boundaries can therefore be inaccurate.

The sidecar format reserves fields for richer assets, but those fields are not populated. This creates a provenance risk: the system can cite a page while losing the exact structure needed to interpret the claim.

Required direction:

- Define a versioned `DocumentIR` for blocks, sections, tables, figures, references, metadata, and source coordinates.
- Use a research-paper parser adapter and retain a plain-text fallback.
- Add OCR detection and a bounded OCR adapter for scanned papers.
- Create stable chunk identifiers that survive re-indexing when source content is unchanged.
- Store parser version, warnings, confidence, and layout provenance.
- Validate parsing quality before marking a document ready.

### 4. Retrieval is document-local and shallow

Retrieval is currently SQLite FTS over individual document paragraphs. There is no corpus-level candidate selection, field-aware search, citation traversal, metadata index, query expansion feedback loop, or principled fusion of graph and lexical results. FTS terms are generated by simple token extraction, which is fragile for scientific notation, names, abbreviations, and multiword concepts.

Required direction:

- Build hierarchical retrieval: corpus → documents → sections → blocks → evidence context.
- Index structured fields such as title, authors, abstract, headings, captions, references, datasets, methods, and metrics.
- Let the model reformulate searches and inspect zero-result diagnostics.
- Fuse lexical, metadata, citation, and graph retrieval with explicit provenance.
- Measure retrieval recall independently from final answer quality.
- Keep retrieval outputs bounded, deduplicated, and stable enough to become durable evidence.

### 5. The ontology source is not the promised ontology

The repository contains a minimal DBLP seed, not a complete DBLP ontology. Ontology search performs substring matching over an in-memory inventory. Generated terms use a generic namespace, and ontology changes do not have a governed lifecycle, version identity, conflict detection, or strong semantic validation.

Sub-ontologies copy selected triples but may omit required axioms and dependencies. Domain/range validation alone cannot establish conceptual consistency.

Required direction:

- Import and version the intended DBLP ontology or clearly declare the supported subset.
- Add indexed ontology lookup for labels, synonyms, descriptions, hierarchy, domains, and ranges.
- Compute the dependency closure needed by selected terms.
- Represent every extension as a versioned proposal with authoring agent, question, rationale, evidence, and status.
- Validate namespace policy, duplicate terms, cycles, incompatible domains/ranges, and unsupported extensions.
- Separate proposed, accepted, deprecated, and run-local ontology terms.

### 6. SPARQL is disconnected from the research graph

The current SPARQL module validates and previews queries against the sub-ontology graph. It does not execute semantic questions over the extracted instance graph. This means SPARQL presently checks ontology syntax and vocabulary but cannot answer corpus aggregation questions from committed research facts.

Required direction:

- Define one semantic query interface over both ontology and instance data.
- Execute validated queries against the active run/workspace graph, not only the schema graph.
- Bind queries to an ontology version and graph snapshot.
- Return rows with the contributing triple and evidence identifiers.
- Limit query cost and provide repairable validation errors.

### 7. EDC canonicalization is too weak for scientific knowledge graphs

EDC extracts relations and generates relation definitions, but the definitions are not used by canonicalization. Canonical matching relies mainly on normalized labels and a low token-overlap threshold. Semantically different relations can therefore collapse into the same property, while synonymous relations can remain fragmented.

Required direction:

- Use relation definitions, domain/range compatibility, ontology hierarchy, and evidence context during canonicalization.
- Require an explicit explanation and score breakdown for every ontology match.
- Calibrate thresholds on annotated examples rather than choosing them heuristically.
- Preserve open relations when confidence is insufficient.
- Detect relation direction errors, negation, modality, comparison, and temporal qualifiers.
- Version EDC prompts and canonicalization rules so experiments are reproducible.

### 8. Entity identity and aggregation semantics are missing

Graph nodes are currently derived from normalized display strings. This does not resolve author aliases, initials, name collisions, dataset versions, method variants, organizations, or paper identity. Corpus-level questions such as “most successful author” require a defined measure, normalized entities, missing-data rules, ties, and the exact population being aggregated.

Required direction:

- Introduce typed entity records with stable identifiers and alias provenance.
- Separate mention extraction, entity resolution, and graph insertion.
- Represent metrics, units, datasets, time, venues, citations, and authors explicitly.
- Model aggregation definitions as inspectable artifacts chosen by the orchestrator.
- Preserve the denominator, exclusions, ties, assumptions, and supporting evidence for every aggregate.
- Ask for clarification only when competing interpretations materially change the answer; otherwise state the chosen interpretation.

### 9. Verification and provenance can be misleading

The verifier checks triples with an LLM and then applies structural validation, but provenance may assign the first evidence document to every triple even when evidence spans come from multiple documents. Evidence identifiers are accepted from model output without a complete deterministic integrity check. SHACL conformance proves structural compatibility, not factual correctness.

Required direction:

- Validate that every cited evidence identifier exists and supports the correct source document.
- Derive provenance from the cited spans, never from bundle position.
- Reject or quarantine triples with missing, cross-wired, or contradictory evidence.
- Keep factual support, ontology compatibility, RDF validity, and SHACL conformance as separate dimensions.
- Use deterministic checks before LLM auditing and an independent critic for high-impact claims.
- Preserve rejected triples and reasons for research analysis without committing them as facts.

### 10. Graph storage is a prototype implementation

The graph is loaded and rewritten as one JSON file through an in-memory graph library. Writes are not transactional or concurrency-safe, schema migrations are absent, and large graphs will require full-file reads. Workspace isolation is represented by paths rather than enforced at a storage seam.

Required direction:

- Define a `ResearchGraph` interface for snapshot, query, validated commit, lineage, and rollback.
- Use an adapter with transactions, indexes, concurrent access, and schema/version migration.
- Make commits idempotent with stable triple identities.
- Store ontology version, run, verifier result, and evidence lineage on every committed assertion.
- Support graph snapshots so answers remain reproducible after later updates.

### 11. Persistence is non-atomic and unsafe under concurrency

Run files and artifact indexes are rewritten directly. Concurrent writers can lose events or corrupt indexes, and there is no locking, compare-and-swap, transaction, or append-only journal. Artifact paths are stored as absolute implementation details, which reduces portability.

Required direction:

- Place run state, events, and artifact metadata behind one persistence interface.
- Use atomic writes or a transactional database adapter.
- Store relative/content-addressed artifact locations.
- Add schema versions and migrations for every persisted record.
- Make artifact creation immutable and idempotent.
- Test concurrent event emission and recovery from interrupted writes.

### 12. Module seams are mostly hypothetical

Protocol definitions exist for documents, retrieval, graph, and workspaces, but production code constructs concrete implementations directly and does not consistently program through those interfaces. The orchestrator receives a large dependency bag, while the research-tool module mixes retrieval, ontology mutation, EDC, verification, persistence, tracing, and graph writes.

This produces shallow modules: callers must understand too many ordering constraints and error modes.

Required direction:

- Deepen the system around a few meaningful interfaces rather than adding pass-through wrappers.
- Accept dependencies at module construction instead of creating adapters inside run execution.
- Make the interface the test surface.
- Remove unused hypothetical interfaces or provide real production and test adapters.
- Keep internal seams private; expose research outcomes rather than implementation plumbing.

### 13. Observability is incomplete and can affect availability

A smoke span proves exporter connectivity, but a root run span does not explain agentic behavior. Automatic instrumentation is version-sensitive, and synchronous SDK initialization/export can delay or block application startup. Current trace events often lack tool names, inputs, output summaries, model usage, validation details, and parent-child structure.

Required direction:

- Make telemetry asynchronous, bounded, optional, and unable to prevent serving requests.
- Emit a trace hierarchy for run, model call, delegation, tool, artifact, validation, graph commit, and answer.
- Record model alias, request ID, latency, finish reason, token usage, retry number, and structured-output validation error.
- Record redacted/truncated tool inputs, output summaries, durations, and artifact references.
- Correlate application events, persisted artifacts, model-gateway requests, and external traces by run and trace identifiers.
- Provide a non-secret observability health view with last export success/failure.
- Capture prompt/response bodies only behind an explicit privacy and retention setting.

### 14. Failure semantics are not precise enough

Model-output retries, tool retries, timeouts, connection failures, validation failures, cancellation, and empty evidence are distinct outcomes, but they can collapse into a generic missing-answer experience. Useful artifacts may exist even when a later graph step fails.

Required direction:

- Define typed failure codes and identify the failing phase and actor.
- Distinguish retryable infrastructure errors from semantic/validation failures.
- Persist partial progress and make retry resume from the last valid checkpoint.
- Require every terminal run to contain either a final answer or a typed failure report.
- Allow an evidence-grounded partial answer when graph commit fails, while clearly stating that the graph was not updated.
- Never label failed or cancelled runs as completed.

### 15. The frontend hides important system state

The frontend catches and discards several document, graph, and artifact-loading errors. It refreshes artifact lists repeatedly as trace events arrive and presents a flat event list rather than a research narrative. It does not expose durable run history, reconnection, partial recovery, parsing quality, or backend/model availability clearly.

Required direction:

- Render the persisted run state as the source of truth.
- Group trace events by phase and parent-child relationship.
- Show “what happened, why, result, evidence, duration, retries” for each phase.
- Support reconnecting to an active run and inspecting previous runs.
- Surface parsing, retrieval, model, validation, graph, and telemetry health separately.
- Avoid polling entire artifact indexes on every event; use explicit artifact events or incremental updates.
- Give failures actionable retry/resume controls.

### 16. Security and workspace isolation are incomplete

The application has no meaningful authentication or authorization layer. Document identifiers and workspace identifiers are accepted by endpoints without a strong ownership model. Upload size, request rate, model budget, and tool access need enforceable limits. Shell capability and artifact inspection increase the impact of mistakes.

Required direction:

- Define workspace identity and authorize every document, run, artifact, evidence, and graph access.
- Validate identifiers and prevent path traversal or cross-workspace access.
- Enforce upload size/type limits, quotas, model budgets, and rate limits.
- Keep secrets out of prompts, traces, artifacts, logs, and packaged archives.
- Sandbox shell/tool execution and audit every invocation.
- Define retention and deletion behavior for source papers and derived artifacts.

### 17. Dependency and capability management is fragile

Fast-moving agent, harness, model, and observability packages are specified with broad lower bounds. Observability is both a core and optional dependency with inconsistent constraints. Optional capabilities can silently disappear when imports or constructor signatures change.

Required direction:

- Pin and test a compatible dependency set for reproducible experiments.
- Keep optional integrations optional and isolate them behind adapters.
- Fail visibly when a requested capability is unavailable; do not silently degrade.
- Add startup capability reporting with versions and enabled features.
- Exercise dependency upgrades through compatibility tests before adoption.

### 18. Evaluation and automated testing are insufficient

The repository has evaluation code, but it is not integrated into the run lifecycle. Gold annotations accepted by the request model are not carried into execution. There are effectively no repository-level tests covering the research loop or frontend behavior.

Required direction:

- Integrate evaluation artifacts into completed runs when gold data is supplied.
- Separate retrieval, ontology, extraction, canonicalization, verification, graph, answer, and system reliability metrics.
- Build a small versioned benchmark corpus with annotated questions, evidence, ontology decisions, and triples.
- Add deterministic tests at the deep module interfaces and contract tests for external adapters.
- Include success, ambiguity, no evidence, contradiction, malformed model output, tool repair, timeout, cancellation, restart, and concurrent-run scenarios.
- Track prompt/model/config versions with every evaluation result.

## Target module architecture

The redesign should concentrate complexity behind a small set of deep modules.

### `Corpus`

Interface responsibilities: ingest, index, inspect parsing state, retrieve evidence, and resolve source positions. Parser and index adapters remain internal.

### `OntologyWorkspace`

Interface responsibilities: derive a sub-ontology, propose extensions, validate changes, resolve terms, and publish a versioned ontology snapshot.

### `EvidenceLedger`

Interface responsibilities: register immutable evidence, expand context, validate citations, and resolve evidence back to source documents and coordinates.

### `TriplePipeline`

Interface responsibilities: extract, define, canonicalize, verify, and return accepted/rejected assertions with provenance. EDC and verifier agents are implementation details behind this interface.

### `ResearchGraph`

Interface responsibilities: query a versioned graph, validate and commit assertions transactionally, compute evidence-preserving aggregates, and reproduce prior snapshots.

### `ResearchEngine`

Interface responsibilities: start/resume/cancel a research goal and return a durable outcome. It owns orchestration, specialist delegation, budgets, checkpoints, and failure recovery while depending on the modules above.

### `RunStore`

Interface responsibilities: atomically persist run state, append events, store immutable artifacts, acquire run ownership, and stream/replay events.

### `Telemetry`

Interface responsibilities: create correlated spans and metrics without exposing the research engine to exporter-specific SDK behavior. A no-op adapter must always be available.

The frontend and transport layer should depend on `ResearchEngine` and `RunStore` outcomes, not on internal agent or storage details.

## System invariants

These rules should be enforced deterministically:

1. Every factual answer claim references one or more valid evidence identifiers.
2. Every evidence identifier resolves to a document and source position.
3. Every committed triple has verified evidence, an ontology version, and a producing run.
4. Ontology extensions are explicit, versioned, justified, and distinguishable from DBLP terms.
5. A run transition is persisted before it is streamed.
6. Every terminal run has either an answer or a typed failure report.
7. Cancellation and budgets are enforced at every expensive boundary.
8. Retrying a completed tool or graph commit is idempotent.
9. Telemetry failure never changes the research result or service availability.
10. An answer can be reproduced from its corpus snapshot, ontology snapshot, graph snapshot, configuration, artifacts, and model metadata.

## Intended agent model

The orchestrator remains responsible for the research strategy. Specialist roles provide bounded expertise:

- paper exploration and cross-document evidence gathering;
- DBLP term selection and ontology extension proposals;
- SPARQL generation and repair;
- evidence sufficiency and contradiction review;
- EDC extraction and canonicalization;
- triple, provenance, RDF, and SHACL auditing.

Specialists should exchange artifact references and explicit task briefs, not inherit an uncontrolled conversation transcript. The orchestrator must be able to revise the ontology, retrieve more evidence, reject a specialist result, or revisit an earlier hypothesis.

## Delivery roadmap

### Priority 0 — Make failures diagnosable

- Typed terminal failures and correct frontend status.
- Full correlated trace hierarchy with model/tool retry details.
- Capability and dependency version reporting.
- Regression tests for the currently observed output, tool, token, and connection failures.

Exit criterion: every failed run identifies the exact phase, actor, error class, retry state, and last durable artifact.

### Priority 1 — Make runs durable

- Durable `RunEngine` and `RunStore` interfaces.
- Atomic events/artifacts, real cancellation, restart recovery, and idempotent checkpoints.
- Concurrent-run and multi-process tests.

Exit criterion: terminating and restarting the application cannot silently lose or strand a run.

### Priority 2 — Establish trustworthy evidence

- Versioned research-paper `DocumentIR`.
- Hierarchical corpus retrieval and stable evidence identifiers.
- Citation integrity checks and multi-document provenance fixes.
- Parsing and retrieval benchmark suite.

Exit criterion: each answer citation resolves to the correct source region, and retrieval recall is measured.

### Priority 3 — Strengthen ontology, EDC, and graph semantics

- Versioned DBLP ontology source and governed extensions.
- Semantic canonicalization using definitions and constraints.
- Transactional research graph and SPARQL over instance data.
- Entity resolution and evidence-preserving aggregation.

Exit criterion: corpus-level questions produce reproducible aggregates and semantically validated, evidence-grounded triples.

### Priority 4 — Improve agentic depth and research UX

- Explicit delegation/sufficiency policy without hard-coded question routing.
- Resumable specialist work and independent evidence criticism.
- Phase-oriented frontend trace, run history, health, and recovery controls.
- Evaluation dashboards and ablation support.

Exit criterion: users can understand and reproduce how the system moved from question to evidence, ontology, triples, graph, and answer.

## Definition of done for the research prototype

The prototype is ready for a thesis evaluation when it can repeatedly:

- ingest and accurately parse a versioned paper corpus;
- answer single- and multi-document research questions through the graph-grounded agent workflow;
- derive and validate a question-specific DBLP sub-ontology;
- produce accepted and rejected triples with exact evidence provenance;
- execute aggregations over normalized graph entities;
- survive model/tool failures and process restarts without losing the audit trail;
- expose a complete, understandable trace without revealing secrets or hidden chain-of-thought;
- report retrieval, ontology, triple, answer, latency, cost, and reliability metrics on a fixed benchmark.
