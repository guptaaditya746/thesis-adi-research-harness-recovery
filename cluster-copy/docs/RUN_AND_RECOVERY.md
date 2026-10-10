# Run, Durability and Recovery

## Run lifecycle

`queued → running → completed | partial | failed | cancelled`.

A process restart can recover a run that was `running` and marked `resumable`. FastAPI startup schedules `ResearchHarness.execute(..., resume=True)`; the harness loads the typed run/artifact state and asks Harness StepPersistence for the latest continuable message history when available. SQLite/run sidecars remain the authoritative experiment state even if a model-message snapshot is unavailable.

Cancellation is cooperative: the API records a durable cancellation flag and every model/tool/delegation boundary calls `ensure_not_cancelled`. Side-effecting artifact writes are append-only/idempotent by artifact ID.

## Recovery ladder

The implementation distinguishes these recovery classes instead of blindly rerunning an entire experiment:

1. PydanticAI output validation returns the exact structured-output error for bounded repair.
2. Invalid tool arguments return deterministic validation/tool errors for repair.
3. Transient specialist/model failures get bounded retry/backoff.
4. Director/specialists can narrow context by passing fewer artifact IDs/evidence spans.
5. Director can decompose an objective into smaller AgentSpecs.
6. Director can re-delegate with a fresh isolated specialist and different evidence/tools.
7. Director can create an independent critic/verifier.
8. `AgentFactory` can escalate configured model profiles with an observable `RecoveryAction`.
9. Director can change research strategy/search terminology/ontology inspection.
10. If reliable completion is impossible, `persist_partial_result` stores supported findings, unresolved questions, rejected triples, failure reason and recovery recommendation, plus an explicit final triple artifact.

Every retry/escalation/recovery is emitted to the trace and stored as an artifact where applicable.

## Crash boundaries

The durable registry tracks run state separately from append-only artifacts and Harness step snapshots. Tools persist outputs only after deterministic validation and use immutable artifact IDs for safe retry. A crash cannot turn `failed` into `completed`: terminal status is written only by explicit finalization routines, and completion revalidates every accepted evidence reference.

## Running locally

```bash
cp .env.example .env
python -m venv .venv
source .venv/bin/activate
pip install -e './backend[dev]'
make backend
```

No sudo, Docker, database server, Java service or vector database is required. SQLite, FTS5, filesystem artifacts and local PyMuPDF are sufficient. GROBID/Docling and outer durability backends are optional adapters.
