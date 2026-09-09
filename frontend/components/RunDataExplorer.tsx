'use client';

import Link from 'next/link';
import { useEffect, useMemo, useState } from 'react';
import { ChevronDown, ChevronLeft, ChevronRight, CircleDot, Copy, Download, FileJson, FolderTree, RefreshCw, Search, Wifi } from 'lucide-react';
import { getArtifact, getRun, listArtifacts, listRuns } from '@/lib/api';
import type { ArtifactEnvelope, ArtifactMeta, RunInfo } from '@/lib/types';

type Selection = { type: 'run' } | { type: 'artifact'; id: string };
type JsonValue = null | boolean | number | string | JsonValue[] | { [key: string]: JsonValue };
const pretty = (value: unknown) => JSON.stringify(value, null, 2);
const fileName = (item: Selection, artifact?: ArtifactMeta) => item.type === 'run' ? 'run.json' : `${artifact?.id ?? item.id}.json`;
const isContainer = (value: JsonValue): value is JsonValue[] | Record<string, JsonValue> => typeof value === 'object' && value !== null;
const short = (value: JsonValue[] | Record<string, JsonValue>) => Array.isArray(value) ? `Array(${value.length})` : `{${Object.keys(value).length}}`;

function JsonLeaf({ value }: { value: JsonValue }) {
  if (value === null) return <span className="jsonNull">null</span>;
  if (typeof value === 'string') return <span className="jsonString">&quot;{value}&quot;</span>;
  if (typeof value === 'number') return <span className="jsonNumber">{value}</span>;
  return <span className="jsonBoolean">{String(value)}</span>;
}

function JsonNode({ name, value, depth, query, path }: { name?: string; value: JsonValue; depth: number; query: string; path: string }) {
  const [open, setOpen] = useState(depth < 2);
  const needle = query.trim().toLowerCase();
  const text = typeof value === 'string' ? value : isContainer(value) ? short(value) : String(value);
  const directMatch = !needle || `${path} ${text}`.toLowerCase().includes(needle);
  const entries = isContainer(value) ? (Array.isArray(value) ? value.map((item, index) => [String(index), item] as const) : Object.entries(value)) : [];
  const shown = entries.filter(([key, child]) => !needle || `${path}.${key} ${typeof child === 'string' ? child : isContainer(child) ? JSON.stringify(child) : String(child)}`.toLowerCase().includes(needle));
  if (needle && !directMatch && shown.length === 0) return null;
  if (!isContainer(value)) return <div className="jsonRow" style={{ paddingLeft: `${depth * 18 + 12}px` }}><span className="jsonKey">{name}</span><span className="jsonColon">:</span><JsonLeaf value={value} /></div>;
  return <div className="jsonGroup"><button className="jsonToggle" onClick={() => setOpen((current) => !current)} style={{ paddingLeft: `${depth * 18 + 7}px` }}>{open ? <ChevronDown size={13} /> : <ChevronRight size={13} />}{name && <span className="jsonKey">{name}</span>}<span className="jsonCount">{short(value)}</span></button>{open && <div className="jsonChildren">{shown.map(([key, child]) => <JsonNode key={key} name={Array.isArray(value) ? `[${key}]` : key} value={child as JsonValue} depth={depth + 1} query={query} path={`${path}.${key}`} />)}{needle && !shown.length && <div className="jsonNoMatch">No matching child fields</div>}</div>}</div>;
}

function JsonTree({ value, query }: { value: unknown; query: string }) { return <div className="jsonTree"><JsonNode value={value as JsonValue} depth={0} query={query} path="$" /></div>; }

export function RunDataExplorer() {
  const [runs, setRuns] = useState<RunInfo[]>([]); const [runId, setRunId] = useState<string>(); const [run, setRun] = useState<RunInfo>(); const [artifacts, setArtifacts] = useState<ArtifactMeta[]>([]);
  const [selection, setSelection] = useState<Selection>({ type: 'run' }); const [artifact, setArtifact] = useState<ArtifactEnvelope>(); const [filter, setFilter] = useState(''); const [find, setFind] = useState(''); const [error, setError] = useState<string>(); const [loading, setLoading] = useState(false); const [autoRefresh, setAutoRefresh] = useState(false);
  async function loadRun(id: string) { setLoading(true); setError(undefined); setRunId(id); setSelection({ type: 'run' }); setArtifact(undefined); try { const [nextRun, nextArtifacts] = await Promise.all([getRun(id), listArtifacts(id)]); setRun(nextRun); setArtifacts(nextArtifacts); } catch (cause) { setError(cause instanceof Error ? cause.message : 'Unable to read this run.'); } finally { setLoading(false); } }
  async function refreshRuns(preserveSelection = false) { setLoading(true); try { const nextRuns = await listRuns(); setRuns(nextRuns); const id = runId ?? nextRuns[0]?.id; if (id) { const [nextRun, nextArtifacts] = await Promise.all([getRun(id), listArtifacts(id)]); setRunId(id); setRun(nextRun); setArtifacts(nextArtifacts); if (preserveSelection && selection.type === 'artifact') setArtifact(await getArtifact(id, selection.id)); } } catch (cause) { setError(cause instanceof Error ? cause.message : 'Unable to list runs.'); } finally { setLoading(false); } }
  useEffect(() => { refreshRuns().catch(() => {}); }, []);
  useEffect(() => { if (!autoRefresh) return; const timer = window.setInterval(() => { refreshRuns(true).catch(() => {}); }, 10000); return () => window.clearInterval(timer); }, [autoRefresh, runId, selection]);
  async function selectArtifact(id: string) { if (!runId) return; setLoading(true); setSelection({ type: 'artifact', id }); try { setArtifact(await getArtifact(runId, id)); } catch (cause) { setError(cause instanceof Error ? cause.message : 'Unable to read this artifact.'); } finally { setLoading(false); } }
  const visibleArtifacts = useMemo(() => { const needle = filter.trim().toLowerCase(); return needle ? artifacts.filter((item) => `${item.id} ${item.kind} ${item.label ?? ''}`.toLowerCase().includes(needle)) : artifacts; }, [artifacts, filter]);
  const value = selection.type === 'run' ? run : artifact; const json = value ? pretty(value) : ''; const matchCount = find.trim() ? json.toLowerCase().split(find.trim().toLowerCase()).length - 1 : 0; const selectedArtifact = selection.type === 'artifact' ? artifacts.find((item) => item.id === selection.id) : undefined;
  function download() { if (!json) return; const url = URL.createObjectURL(new Blob([json], { type: 'application/json' })); const anchor = document.createElement('a'); anchor.href = url; anchor.download = fileName(selection, selectedArtifact); anchor.click(); URL.revokeObjectURL(url); }
  async function copy() { if (json) await navigator.clipboard.writeText(json); }
  return <main className="dataExplorer"><header className="dataExplorerHeader"><div><Link href="/" className="backLink"><ChevronLeft size={15} />Research console</Link><h1>Run data explorer</h1><p>Read-only persisted run state and immutable ArtifactStore artifacts.</p></div><div className="dataHeaderActions"><button className={`liveButton ${autoRefresh ? 'on' : ''}`} onClick={() => setAutoRefresh((current) => !current)}><Wifi size={14} />Live read {autoRefresh ? 'on' : 'off'}</button><button className="dataButton" onClick={() => refreshRuns(true)} disabled={loading}><RefreshCw size={14} />Refresh</button></div></header>
    {run && <div className="runOverview"><div><span>RUN</span><b>{run.id}</b></div><div><span>STATUS</span><b className={`runStatus ${run.status}`}>{run.status}</b></div><div><span>EVENTS</span><b>{run.events.length}</b></div><div><span>ARTIFACTS</span><b>{artifacts.length}</b></div><div className="runQuestion"><span>QUESTION</span><b>{run.question}</b></div><div className="readOnlyNote"><CircleDot size={13} />Read-only: experiment keeps running</div></div>}
    <div className="dataExplorerGrid"><aside className="dataRuns"><div className="dataPaneHead"><FolderTree size={15} /><b>Runs</b><span>{runs.length}</span></div><div className="dataRunList">{runs.map((item) => <button key={item.id} className={item.id === runId ? 'selected' : ''} onClick={() => loadRun(item.id)}><b>{item.id}</b><span>{item.status} · {new Date(item.updated_at).toLocaleString()}</span></button>)}{!runs.length && <p>No persisted runs found.</p>}</div></aside>
      <aside className="dataFiles"><div className="dataPaneHead"><FileJson size={15} /><b>JSON files</b><span>{artifacts.length + (run ? 1 : 0)}</span></div><label className="dataSearch"><Search size={14} /><input value={filter} onChange={(event) => setFilter(event.target.value)} placeholder="Filter artifacts" /></label>{run && <button className={`dataFile ${selection.type === 'run' ? 'selected' : ''}`} onClick={() => { setSelection({ type: 'run' }); setArtifact(undefined); }}><b>run.json</b><span>State, event trace, final output</span></button>}<div className="dataFileList">{visibleArtifacts.map((item) => <button key={item.id} className={`dataFile ${selection.type === 'artifact' && selection.id === item.id ? 'selected' : ''}`} onClick={() => selectArtifact(item.id)}><b>{item.id}.json</b><span><i>{item.kind}</i>{item.label ? ` · ${item.label}` : ''}</span></button>)}</div></aside>
      <section className="dataJson"><div className="dataJsonHead"><div><FileJson size={15} /><b>{value ? fileName(selection, selectedArtifact) : 'Select a run'}</b>{loading && <span>loading…</span>}</div><div className="dataActions"><label className="dataSearch find"><Search size={13} /><input value={find} onChange={(event) => setFind(event.target.value)} placeholder="Find key or value" />{find && <em>{matchCount}</em>}</label><button className="dataButton" onClick={copy} disabled={!json}><Copy size={13} />Copy</button><button className="dataButton" onClick={download} disabled={!json}><Download size={13} />Download</button></div></div>{value && <div className="jsonMeta">{selection.type === 'run' ? <><span>Run record</span><code>{run?.workspace_id}</code><span>updated {run && new Date(run.updated_at).toLocaleString()}</span></> : <><span>{artifact?.artifact_type}</span><code>sha256 {artifact?.content_sha256.slice(0, 16)}…</code><span>parents {artifact?.parent_artifact_ids.length ?? 0}</span></>}</div>}{error && <div className="dataError">{error}</div>}{value ? <JsonTree value={value} query={find} /> : <div className="emptyState">Choose a run, then select <code>run.json</code> or an artifact.</div>}</section>
    </div></main>;
}
