from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from pathlib import Path

from app.config import Settings
from app.models.domain import AgentTraceEvent, ExperimentRun, utc_now


class RunRegistry:
    """SQLite-backed run/event registry plus in-process SSE queues.

    SQLite is the authoritative durable index; run.json is retained as a human-readable
    sidecar/checkpoint for experiment reproducibility.
    """
    def __init__(self, settings: Settings):
        self.settings = settings
        self.runs: dict[str, ExperimentRun] = {}
        self.queues: dict[str, asyncio.Queue[AgentTraceEvent]] = {}
        self.cancelled: set[str] = set()
        self._lock = threading.RLock()
        self.db_path = self._sqlite_path(settings.database_url)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @staticmethod
    def _sqlite_path(url: str) -> Path:
        prefix = "sqlite:///"
        raw = url[len(prefix):] if url.startswith(prefix) else "./data/research.db"
        return Path(raw).resolve()

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.db_path, timeout=30)
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=FULL")
        return db

    def _init_db(self) -> None:
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, workspace_id TEXT, status TEXT, updated_at TEXT, json TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS events (id TEXT PRIMARY KEY, run_id TEXT NOT NULL, timestamp TEXT NOT NULL, json TEXT NOT NULL)")
            db.execute("CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id, timestamp)")
            db.commit()

    def create(self, run: ExperimentRun) -> ExperimentRun:
        self.runs[run.id] = run
        self.queues[run.id] = asyncio.Queue()
        self.persist(run)
        return run

    def get(self, run_id: str) -> ExperimentRun:
        if run_id in self.runs:
            return self.runs[run_id]
        with self._connect() as db:
            row = db.execute("SELECT json FROM runs WHERE id=?", (run_id,)).fetchone()
        if not row:
            sidecar = self.settings.data_dir / "runs" / run_id / "run.json"
            if not sidecar.exists():
                raise KeyError(run_id)
            run = ExperimentRun.model_validate_json(sidecar.read_text(encoding="utf-8"))
        else:
            run = ExperimentRun.model_validate_json(row[0])
        self.runs[run_id] = run
        self.queues.setdefault(run_id, asyncio.Queue())
        return run

    def list(self, workspace_id: str | None = None) -> list[ExperimentRun]:
        sql = "SELECT json FROM runs"
        params: tuple[str, ...] = ()
        if workspace_id:
            sql += " WHERE workspace_id=?"
            params = (workspace_id,)
        sql += " ORDER BY updated_at DESC"
        with self._connect() as db:
            rows = db.execute(sql, params).fetchall()
        values = [ExperimentRun.model_validate_json(row[0]) for row in rows]
        for run in values:
            self.runs[run.id] = run
            self.queues.setdefault(run.id, asyncio.Queue())
        return values

    async def emit(self, event: AgentTraceEvent) -> None:
        with self._lock:
            run = self.get(event.run_id)
            # Keep only a bounded in-run mirror; the SQLite events table is the durable full trace index.
            run.events.append(event)
            if len(run.events) > 1000:
                run.events = run.events[-1000:]
            run.updated_at = utc_now()
            with self._connect() as db:
                db.execute("INSERT OR IGNORE INTO events(id, run_id, timestamp, json) VALUES (?, ?, ?, ?)",
                           (event.id, event.run_id, event.timestamp.isoformat(), event.model_dump_json()))
                db.commit()
            self.persist(run)
        await self.queues.setdefault(event.run_id, asyncio.Queue()).put(event)

    def events(self, run_id: str) -> list[AgentTraceEvent]:
        with self._connect() as db:
            rows = db.execute("SELECT json FROM events WHERE run_id=? ORDER BY timestamp", (run_id,)).fetchall()
        return [AgentTraceEvent.model_validate_json(x[0]) for x in rows]

    def persist(self, run: ExperimentRun) -> None:
        with self._lock:
            run.updated_at = utc_now()
            payload = run.model_dump_json()
            with self._connect() as db:
                db.execute(
                    "INSERT INTO runs(id, workspace_id, status, updated_at, json) VALUES(?,?,?,?,?) "
                    "ON CONFLICT(id) DO UPDATE SET workspace_id=excluded.workspace_id,status=excluded.status,updated_at=excluded.updated_at,json=excluded.json",
                    (run.id, run.workspace_id, run.status, run.updated_at.isoformat(), payload),
                )
                db.commit()
            path = self.settings.data_dir / "runs" / run.id
            path.mkdir(parents=True, exist_ok=True)
            temp = path / "run.json.tmp"
            temp.write_text(run.model_dump_json(indent=2), encoding="utf-8")
            temp.replace(path / "run.json")
            self.runs[run.id] = run

    def cancel(self, run_id: str) -> None:
        self.cancelled.add(run_id)
        run = self.get(run_id)
        if run.status in {"queued", "running"}:
            run.resumable = False
        self.persist(run)

    def clear_cancel(self, run_id: str) -> None:
        self.cancelled.discard(run_id)

    def is_cancelled(self, run_id: str) -> bool:
        return run_id in self.cancelled

    def recoverable(self) -> list[ExperimentRun]:
        return [r for r in self.list() if r.status == "running" or r.resumable]
