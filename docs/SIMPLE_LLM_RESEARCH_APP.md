# Simple LLM Research App

## Purpose

Build a fast experimental baseline for document research.

- Upload scientific PDFs.
- Ask a natural-language question.
- Retrieve relevant passages.
- Generate a concise answer and simple triples from numbered passages.
- Let the host construct all source IDs, locators, and excerpts.
- Finish with at most two LLM calls.

This is an alternative baseline, not a replacement for the full agentic harness.

## Architecture

```text
PDF upload
    ↓
PyMuPDF parsing + SQLite FTS5 index
    ↓
Question → bounded passage retrieval
    ↓
One typed LLM synthesis call through LiteLLM
    ↓
Host RDF mapping + source citation construction
    ↓
Optional single repair call
    ↓
Answer + citations + simple Triple Map
```

## Stack

- FastAPI backend.
- OpenAI Python SDK using `AsyncOpenAI`.
- Existing `LITELLM_API_BASE`, API key, and model aliases.
- `LITELLM_FAST_MODEL` by default.
- Existing PyMuPDF parser and SQLite FTS5 retrieval.
- Existing document storage plus atomic JSON run records.
- Next.js frontend.

Use the OpenAI-compatible chat-completions interface for current LiteLLM compatibility.

## Request flow

1. Parse and index each PDF once.
2. Search only the selected documents.
3. Return a bounded set of passages with document, page, block, and score.
4. Send the question and numbered passages to one structured LLM request.
5. Let the model select atomic predicate keys, entity types, and bounded passage numbers.
6. Map those selections to a compact host ontology and construct citations from parsed source text.
7. If validation fails, perform one bounded repair request.
8. Persist the validated result and render it.

No Director, specialists, recursive tools, DAG, EDC, or ontology construction.

## API

```text
POST /api/simple/documents
GET  /api/simple/documents
POST /api/simple/research
GET  /api/simple/runs/{run_id}
GET  /api/simple/runs/{run_id}/triple-map
```

`POST /api/simple/research` input:

```json
{
  "question": "What limitations do the authors acknowledge?",
  "document_ids": ["doc_..."],
  "model": "research-fast"
}
```

## Output contract

```json
{
  "status": "completed",
  "result": {
    "answer": "Concise evidence-backed answer.",
    "citations": [{
      "evidence_id": "ev_host_generated",
      "document_id": "doc_...",
      "page": 4,
      "quote": "Exact host-selected source excerpt."
    }],
    "triples": [{
      "subject": "Paper",
      "subject_type": "ResearchPaper",
      "predicate": "has limitation",
      "predicate_key": "has_limitation",
      "predicate_uri": "https://example.org/thesis/simple-research#has_limitation",
      "object": "Limitation statement",
      "object_type": "Limitation",
      "evidence_ids": ["ev_..."]
    }],
    "limitations": []
  }
}
```

Allowed terminal states: `completed`, `failed`, `cancelled`.

## Validation

- The model selects bounded passage numbers, never opaque evidence IDs.
- The host creates evidence IDs, document/page locators, and exact excerpts.
- Triple evidence IDs are derived from the selected passages.
- Atomic triples use a compact controlled RDF vocabulary: paper, method, task, dataset, metric, result, claim, limitation, and entity types; predicates such as `proposes_method`, `evaluates_on`, `improves`, and `has_limitation` map to host-owned URIs.
- Invalid or unavailable passage numbers are rejected.
- Validation errors are bounded and payload-free.
- One repair attempt maximum.

## Frontend

- Document sidebar with upload and selection.
- Main question and answer panel.
- Clickable citations with document and page.
- Small Triple Map derived from validated triples.
- Progress: `retrieving → generating → validating → completed`.
- Advanced toggle for raw JSON and timing data.
- Research continues if the user opens another page or panel.

## Performance limits

- Maximum two LLM requests per question.
- Default to `LITELLM_FAST_MODEL`.
- Optional heavy model only through an explicit user setting.
- Full retains its 32,768-token completion ceiling. Reasoning uses an explicit result-first envelope: four passages, 1,200 characters per passage, low effort, an 8,000-token ceiling, and a 1–3-triple schema. This deliberately favors a timely supported result over exhaustive coverage.
- Bound retrieved passages, passage characters, input tokens, and output tokens.
- Reject unchanged retries.
- Cache parsed documents and FTS indexes.
- Persist the resolved execution profile plus retrieval, generation, repair, validation, provider token usage, finish reason, and total latency. The UI shows a compact timing summary; JSON retains the detailed diagnostics.

## Tradeoffs

- No DBLP canonicalization.
- No EDC phase audit trail.
- No run-local ontology proposals.
- No independent multi-agent verification.
- No autonomous research trajectory.
- Triples are validated research summaries, not canonical graph assertions.

The baseline is faster, smaller, and easier to debug. The full harness remains the strict scientific system.

## Build order

- [x] Reuse parser and retrieval modules behind simple service interfaces.
- [x] Add direct `AsyncOpenAI` synthesis endpoint.
- [x] Add host-generated citation and source-excerpt validation.
- [x] Add bounded repair and failure handling.
- [x] Add the simplified frontend route.
- [ ] Measure latency, completion rate, citation accuracy, and triple accuracy.
- [ ] Compare results with the full agentic harness on identical questions.
