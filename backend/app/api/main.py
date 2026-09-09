from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from app.config import get_settings
from app.harness.events import TraceEmitter
from app.harness.runtime import ResearchHarness
from app.models.domain import ExperimentRun, StartRunRequest
from app.services.durability import describe_backend
from app.services.pdf import PDFService
from app.services.simple_research import SimpleResearchRequest, SimpleResearchService
from app.services.triple_map import TripleMapService
from app.services.workspaces import WorkspaceService
from app.storage.artifacts import ArtifactStore
from app.storage.registry import RunRegistry
from app.telemetry.tracing import Telemetry, register_application_telemetry

settings = get_settings()
registry = RunRegistry(settings)
telemetry = Telemetry(settings)
register_application_telemetry(telemetry)
harness = ResearchHarness(settings, registry, telemetry=telemetry)
pdf = PDFService(settings)
simple_research = SimpleResearchService(settings, pdf)
triple_map = TripleMapService()
workspaces = WorkspaceService(settings)
_tasks: dict[str, asyncio.Task[None]] = {}
_simple_tasks: dict[str, asyncio.Task[None]] = {}


def _schedule(run_id: str, *, resume: bool = False) -> None:
    existing = _tasks.get(run_id)
    if existing and not existing.done():
        return
    task = asyncio.create_task(harness.execute(run_id, resume=resume), name=f"research:{run_id}")
    _tasks[run_id] = task
    task.add_done_callback(lambda _: _tasks.pop(run_id, None))


def _schedule_simple(run_id: str) -> None:
    existing = _simple_tasks.get(run_id)
    if existing and not existing.done():
        return
    task = asyncio.create_task(simple_research.execute(run_id), name=f"simple-research:{run_id}")
    _simple_tasks[run_id] = task
    task.add_done_callback(lambda _: _simple_tasks.pop(run_id, None))


@asynccontextmanager
async def lifespan(_: FastAPI):
    # A process restart leaves durable SQLite/run sidecars intact. Resume only runs that
    # were explicitly marked resumable, using Harness StepPersistence when available.
    for run in registry.recoverable():
        if run.status == "running" and run.resumable:
            _schedule(run.id, resume=True)
    simple_research.store.mark_interrupted_failed()
    try:
        yield
    finally:
        telemetry.force_flush()
        telemetry.shutdown()


app = FastAPI(title="Thesis ADI Research Harness", version="0.3.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.frontend_origins,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)


@app.get("/health")
def health():
    return {
        "ok": True,
        "runtime": "pydantic-ai+harness",
        "retrieval": "lexical-structural-no-embeddings",
        "final_product": "canonicalized_triples",
        "durability": describe_backend(settings),
    }


@app.get("/api/workspaces")
def list_workspaces():
    return workspaces.list()


@app.get("/api/documents")
def list_documents(workspace_id: str = "default"):
    return pdf.list(workspaces.normalize(workspace_id))


@app.post("/api/documents")
async def upload_document(file: UploadFile = File(...), workspace_id: str = Form("default")):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Scientific ingestion currently accepts PDF files.")
    content = await file.read()
    try:
        return pdf.ingest(file.filename, content, workspaces.normalize(workspace_id))
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@app.delete("/api/documents/{document_id}")
def delete_document(document_id: str):
    # Runs keep hashes/artifacts for reproducibility; deleting a document only removes the
    # corpus copy and should therefore be avoided for papers referenced by active runs.
    active = [r.id for r in registry.list() if document_id in r.document_ids and r.status in {"queued", "running"}]
    if active:
        raise HTTPException(409, f"Document is referenced by active run(s): {', '.join(active)}")
    pdf.delete(document_id)
    return {"ok": True}


@app.get("/api/documents/{document_id}")
def get_document(document_id: str):
    try:
        return pdf.load(document_id)
    except Exception as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/api/documents/{document_id}/pdf")
def source_pdf(document_id: str):
    try:
        doc = pdf.load(document_id)
    except KeyError as exc:
        raise HTTPException(404, "Document not found") from exc
    return FileResponse(Path(doc.workspace) / "source.pdf", media_type="application/pdf", filename=doc.filename)


@app.get("/api/documents/{document_id}/page/{page}")
def source_page(document_id: str, page: int):
    try:
        doc = pdf.load(document_id)
        return {"document_id": document_id, "page": page, "text": harness.retrieval.read_page(doc, page)}
    except Exception as exc:
        raise HTTPException(404, str(exc)) from exc


@app.post("/api/runs")
async def create_run(request: StartRunRequest):
    if not request.question.strip():
        raise HTTPException(400, "Research question may not be blank")
    if not request.document_ids:
        raise HTTPException(400, "Select at least one scientific document")
    for did in request.document_ids:
        try:
            pdf.load(did)
        except Exception as exc:
            raise HTTPException(404, f"Document {did}: {exc}") from exc
    target_document_id = request.target_document_id or request.document_ids[0]
    if target_document_id not in request.document_ids:
        raise HTTPException(400, "target_document_id must be one of document_ids")
    run = ExperimentRun(
        workspace_id=workspaces.normalize(request.workspace_id),
        document_ids=request.document_ids,
        target_document_id=target_document_id,
        question=request.question.strip(),
        gold=request.gold,
    )
    registry.create(run)
    _schedule(run.id)
    return run


@app.get("/api/runs")
def list_runs(workspace_id: str | None = None):
    return registry.list(workspaces.normalize(workspace_id) if workspace_id else None)


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    try:
        return registry.get(run_id)
    except KeyError as exc:
        raise HTTPException(404, "Run not found") from exc


@app.post("/api/runs/{run_id}/resume")
async def resume_run(run_id: str):
    try:
        run = registry.get(run_id)
    except KeyError as exc:
        raise HTTPException(404, "Run not found") from exc
    if run.status == "completed":
        raise HTTPException(409, "Completed runs are immutable; start a new experiment instead")
    if _tasks.get(run_id) and not _tasks[run_id].done():
        return {"ok": True, "status": "already_running"}
    run.resumable = True
    registry.persist(run)
    _schedule(run_id, resume=True)
    return {"ok": True, "status": "resuming"}


@app.post("/api/runs/{run_id}/cancel")
async def cancel_run(run_id: str):
    try:
        registry.get(run_id)
    except KeyError as exc:
        raise HTTPException(404, "Run not found") from exc
    registry.cancel(run_id)
    await TraceEmitter(registry, run_id, telemetry).emit(
        type="system", actor="user", action="cancellation_requested", title="Cancellation requested",
        summary="The run will stop cooperatively at the next model/tool/delegation boundary.",
    )
    return {"ok": True, "note": "Cooperative cancellation flag recorded."}


@app.get("/api/runs/{run_id}/events")
async def run_events(run_id: str):
    try:
        registry.get(run_id)
    except KeyError as exc:
        raise HTTPException(404, "Run not found") from exc

    async def stream():
        sent: set[str] = set()
        history, queue = registry.subscribe(run_id)
        try:
            for event in history:
                sent.add(event.id)
                yield f"data: {event.model_dump_json()}\n\n"
            while True:
                current = registry.get(run_id)
                if current.status in {"completed", "failed", "cancelled"} and queue.empty():
                    yield "data: [DONE]\n\n"
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), 15)
                    if event.id not in sent:
                        sent.add(event.id)
                        yield f"data: {event.model_dump_json()}\n\n"
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
        finally:
            registry.unsubscribe(run_id, queue)

    return StreamingResponse(
        stream(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/runs/{run_id}/artifacts")
def list_artifacts(run_id: str):
    try:
        registry.get(run_id)
    except KeyError as exc:
        raise HTTPException(404, "Run not found") from exc
    return ArtifactStore(settings, run_id).list()


@app.get("/api/runs/{run_id}/artifacts/{artifact_id}")
def get_artifact(run_id: str, artifact_id: str):
    try:
        return ArtifactStore(settings, run_id).get(artifact_id, envelope=True)
    except KeyError as exc:
        raise HTTPException(404, "Artifact not found") from exc


@app.get("/api/runs/{run_id}/evidence/{evidence_id}")
def get_evidence(run_id: str, evidence_id: str):
    store = ArtifactStore(settings, run_id)
    try:
        env = store.get_envelope(evidence_id)
    except KeyError as exc:
        raise HTTPException(404, "Evidence span not found") from exc
    if env.artifact_type != "evidence_span":
        raise HTTPException(404, "Artifact is not an evidence span")
    span = env.content
    document_id = span.get("document_id")
    return {
        "span": span,
        "pdf_url": f"/api/documents/{document_id}/pdf#page={span.get('page', 1)}",
        "artifact_id": evidence_id,
    }


@app.get("/api/runs/{run_id}/triple-map")
def run_triple_map(run_id: str):
    try:
        run = registry.get(run_id)
    except KeyError as exc:
        raise HTTPException(404, "Run not found") from exc
    return triple_map.snapshot(run.final_triples)


# The simple baseline deliberately has its own API namespace and persistence.
# It reuses document ingestion but never enters the strict agentic harness.
@app.get("/api/simple/documents")
def list_simple_documents(workspace_id: str = "default"):
    return list_documents(workspace_id)


@app.post("/api/simple/documents")
async def upload_simple_document(file: UploadFile = File(...), workspace_id: str = Form("default")):
    return await upload_document(file, workspace_id)


@app.post("/api/simple/research")
async def create_simple_research(request: SimpleResearchRequest):
    try:
        run = simple_research.create(request)
    except KeyError as exc:
        raise HTTPException(404, f"Document not found: {exc.args[0]}") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    _schedule_simple(run.id)
    return run


@app.get("/api/simple/runs")
def list_simple_runs():
    return simple_research.store.list()


@app.get("/api/simple/runs/{run_id}")
def get_simple_run(run_id: str):
    try:
        return simple_research.store.get(run_id)
    except KeyError as exc:
        raise HTTPException(404, "Simple research run not found") from exc


@app.post("/api/simple/runs/{run_id}/cancel")
async def cancel_simple_run(run_id: str):
    try:
        run = simple_research.cancel(run_id)
    except KeyError as exc:
        raise HTTPException(404, "Simple research run not found") from exc
    task = _simple_tasks.get(run_id)
    if task and not task.done():
        task.cancel()
    return run


@app.get("/api/simple/runs/{run_id}/triple-map")
def simple_run_triple_map(run_id: str):
    try:
        return simple_research.triple_map(run_id)
    except KeyError as exc:
        raise HTTPException(404, "Simple research run not found") from exc
