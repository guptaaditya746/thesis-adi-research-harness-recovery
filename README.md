# Thesis ADI Research Harness

A Python-first research system for extracting **evidence-backed canonicalized triples** from scientific papers. The final scientific product is a set of `(subject, predicate, object)` triples with exact evidence provenance, canonicalization/verification artifacts and evaluation—not a persistent knowledge graph.

## Core architecture

- **Research Director:** one long-lived PydanticAI agent repeatedly decides what uncertainty to investigate next. There is no mandatory stage order or fixed research DAG.
- **Dynamic specialists:** the Director creates typed `AgentSpec` objects at runtime. `AgentFactory` constructs isolated ephemeral PydanticAI agents with only the requested toolsets, artifacts and budgets. Harness `DynamicWorkflow` is available when the Director explicitly wants bounded fan-out/chaining.
- **Pydantic AI Harness:** FileSystem, restricted Shell, StepPersistence, CodeMode and DynamicWorkflow are used where available instead of rebuilding generic agent infrastructure.
- **Scientific retrieval:** SQLite FTS5/BM25, structured `DocumentIR` navigation and restricted `rg`/`grep`/`find`. No embedding/vector database is required.
- **Ontology:** the bundled official DBLP RDF schema dated 2024-06-14 is immutable seed vocabulary. Every run creates a selected snapshot plus explicit run-local ontology proposals.
- **EDC:** evidence is retrieved and persisted first. Then Extract → Define → Canonicalize operates over `EvidenceSpan` objects. Default relation candidate ranking is deterministic lexical/fuzzy/domain-range/hierarchy matching followed by bounded LLM verification; upstream embedding retrieval is isolated as an optional ablation.
- **Verification:** evidence support, ontology compatibility, subject/object direction, RDF/SHACL structure, negation/modality and entity normalization are kept as separate dimensions. SHACL conformance never proves factual truth.
- **Provenance:** every accepted triple contains evidence IDs that resolve to immutable spans and exact paper/page/section/block locations.
- **Observability:** manual OpenTelemetry spans exist at application boundaries and can be exported to Langtrace. User-visible trace events expose task messages, generated prompts, decisions, tools, arguments, outputs, retries/recovery, model metadata and artifacts—never hidden chain-of-thought.
- **Durability:** SQLite run state + Harness StepPersistence. No Docker, PostgreSQL, Java daemon or sudo access is required.

## Local setup

Prerequisites: Python 3.11+, Node.js/npm for the optional Next.js console, and access to your existing OpenAI-compatible LiteLLM endpoint.

```bash
cp .env.example .env
python -m venv .venv
source .venv/bin/activate
pip install -e './backend[dev]'
make backend
```

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

The backend listens on `http://localhost:8001` and the frontend on `http://localhost:3000` by default. Configure only the LiteLLM gateway credentials the application needs. Provider credentials should remain on the LiteLLM side and are stripped from research shell environments.

### Scientific parser policy

The base system always works with local PyMuPDF. If `GROBID_URL` is configured, GROBID is attempted first for scientific metadata/structure. `DOCLING_URL` is an optional second adapter. Failure of either optional service falls back locally; neither is a runtime requirement.

## Research run semantics

A run starts with the question, selected papers, budgets, current artifacts and unresolved issues. The Director can search evidence, inspect ontology, create a specialist, generate/repair SPARQL semantic patterns, run EDC, criticize/retrieve again, verify/reject candidates or change strategy in any order. An easy question may use only a few of these capabilities.

A completed run must have a `canonical_triples` artifact. Every accepted triple must have a canonical predicate URI, independent accepted verification and evidence IDs that resolve back to source text. If reliable completion is impossible, the run returns a typed **partial** result instead of claiming success.

## Repository map

- `backend/app/agents/` — dynamic `AgentFactory`, model profiles and specialist toolset construction.
- `backend/app/harness/` — Director runtime, Harness capabilities, typed dependencies and observable events.
- `backend/app/corpus/` — parser adapters and `DocumentIR` creation.
- `backend/app/services/retrieval.py` — lexical/structural evidence retrieval.
- `backend/app/services/ontology.py` — immutable DBLP seed + run overlay/snapshots.
- `backend/app/services/edc.py` — evidence-first EDC adaptation.
- `backend/app/services/verification.py` — independent multidimensional triple verification.
- `backend/app/storage/` — immutable artifact store + durable run/event registry.
- `backend/app/telemetry/` — redacted best-effort OpenTelemetry/Langtrace instrumentation.
- `frontend/` — live research console, trace/artifact/evidence/triple/run views.

See `docs/ARCHITECTURE.md` and `docs/RUN_AND_RECOVERY.md` for the detailed design.

## Automated five-case research run

Run five representative questions against the latest parsed document in the
default workspace:

```bash
make benchmark-five
```

Choose documents or an output path without editing the script:

```bash
make benchmark-five ARGS="--document-id doc_123 --output data/benchmarks/paper.json"
```

The runner executes all five cases even if an individual case fails, then writes
a JSON report containing run IDs, statuses, artifact/provenance coverage, latency,
subagent calls, retries, invalid-reference failures, and model/tool usage.

## Deterministic presentation example

Create one successful, evidence-backed run for the existing UI without making a
model call:

```bash
make demo-success ARGS="--document-id doc_448795ae1e2f4823"
```

Then start the backend and frontend and open `http://localhost:3000`. The newest
run in **Experiment run history** contains six accepted triples, exact page-one
evidence, ontology lineage, RDF/SHACL validation and a triple map. The question,
answer and evaluation explicitly label this as a human-curated deterministic
demonstration; it must not be reported as benchmark performance.

## Related projects

1. [Docling Graph](https://github.com/docling-project/docling-graph) — best overall match: documents/PDFs → validated knowledge graph, provenance, schema validation and JSON export.
2. [Wikontic](https://github.com/screemix/Wikontic) — SPO extraction followed by ontology alignment and separate accepted/filtered triples.
3. [KnowledgeGraphBuilder](https://github.com/DataScienceLabFHSWF/KnowledgeGraphBuilder) — ontology-guided extraction, source alignment, SHACL validation and resumable checkpoints.
4. [LangExtract](https://github.com/google/langextract) — grounds every extracted triple to an exact source passage.
5. [OSKGC](https://github.com/HeraclesWang/OSKGC) — a benchmark for measuring triple accuracy and schema conformity.
6. [OntoGPT](https://github.com/monarch-initiative/ontogpt) — an extraction component already used by this system.
7. [EDC](https://github.com/clear-nus/edc) — closely matches the Extract → Define → Canonicalize workflow, but is older.

See [`docs/SIMILAR_GITHUB_PROJECTS.md`](docs/SIMILAR_GITHUB_PROJECTS.md) for the detailed comparison and recommendations.
