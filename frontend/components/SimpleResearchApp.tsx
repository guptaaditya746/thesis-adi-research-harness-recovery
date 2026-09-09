'use client';

import dynamic from 'next/dynamic';
import Link from 'next/link';
import { ChangeEvent, FormEvent, useEffect, useMemo, useState } from 'react';
import {
  ArrowLeft,
  BookOpen,
  Check,
  ChevronRight,
  Clock3,
  FilePlus2,
  LoaderCircle,
  Network,
  Play,
  Square,
} from 'lucide-react';

import {
  API_BASE,
  cancelSimpleRun,
  getSimpleRun,
  getSimpleTripleMap,
  listDocuments,
  listSimpleRuns,
  startSimpleResearch,
  uploadPdf,
} from '@/lib/api';
import type { DocumentInfo, SimpleResearchRun, TripleMapSnapshot } from '@/lib/types';

const TripleMapViewer = dynamic(
  () => import('./TripleMapViewer').then((module) => module.TripleMapViewer),
  { ssr: false },
);

const stages = ['retrieving', 'generating', 'validating', 'completed'] as const;
const storageKey = 'simple-research-active-run';

export function SimpleResearchApp() {
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [question, setQuestion] = useState('');
  const [model, setModel] = useState<'fast' | 'default' | 'reasoning'>('fast');
  const [reasoningEffort, setReasoningEffort] = useState<'low' | 'medium' | 'high'>('low');
  const [run, setRun] = useState<SimpleResearchRun>();
  const [runs, setRuns] = useState<SimpleResearchRun[]>([]);
  const [map, setMap] = useState<TripleMapSnapshot>({ nodes: [], edges: [] });
  const [error, setError] = useState<string>();
  const active = run?.status === 'queued' || run?.status === 'running';
  const firstRequest = run?.model_request_diagnostics?.[0];

  async function loadDocuments() {
    setDocuments(await listDocuments());
  }

  async function refreshRun(id: string) {
    const current = await getSimpleRun(id);
    setRun(current);
    if (current.status === 'completed') {
      setMap(await getSimpleTripleMap(id));
    }
    if (!['queued', 'running'].includes(current.status)) {
      setRuns(await listSimpleRuns());
    }
  }

  useEffect(() => {
    loadDocuments().catch((cause) => setError(String(cause)));
    listSimpleRuns().then(setRuns).catch(() => undefined);
    const saved = window.localStorage.getItem(storageKey);
    if (saved) refreshRun(saved).catch(() => window.localStorage.removeItem(storageKey));
  }, []);

  useEffect(() => {
    if (!run || !['queued', 'running'].includes(run.status)) return;
    const timer = window.setInterval(() => refreshRun(run.id).catch((cause) => setError(String(cause))), 900);
    return () => window.clearInterval(timer);
  }, [run?.id, run?.status]);

  const currentStage = useMemo(() => {
    if (!run) return -1;
    if (run.stage === 'repairing') return 2;
    return stages.indexOf(run.stage as (typeof stages)[number]);
  }, [run]);

  function toggleDocument(id: string) {
    setSelected((current) => current.includes(id) ? current.filter((item) => item !== id) : [...current, id]);
  }

  async function upload(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const document = await uploadPdf(file);
      setDocuments((current) => [document, ...current]);
      setSelected((current) => [...current, document.id]);
    } catch (cause) {
      setError(String(cause));
    } finally {
      event.target.value = '';
    }
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (!question.trim() || !selected.length || active) return;
    setError(undefined);
    setMap({ nodes: [], edges: [] });
    try {
      const created = await startSimpleResearch(question.trim(), selected, model, model === 'reasoning' ? reasoningEffort : undefined);
      window.localStorage.setItem(storageKey, created.id);
      setRun(created);
      setRuns((current) => [created, ...current.filter((item) => item.id !== created.id)]);
    } catch (cause) {
      setError(String(cause));
    }
  }

  async function cancel() {
    if (!run) return;
    setRun(await cancelSimpleRun(run.id));
  }

  return (
    <main className="simplePage">
      <header className="simpleHeader">
        <div>
          <Link href="/" className="simpleBack"><ArrowLeft size={15} /> Agentic console</Link>
          <h1>Paper Lens</h1>
          <p>Fast, direct evidence-backed answers. No agents, ontology, or workflow engine.</p>
        </div>
        <span className="simpleMode">OpenAI SDK · 1 call + optional repair</span>
      </header>

      <div className="simpleLayout">
        <aside className="simpleDocuments">
          <div className="simpleSectionTitle"><BookOpen size={16} /><b>Papers</b><span>{selected.length} selected</span></div>
          <label className="simpleUpload"><FilePlus2 size={15} /> Upload PDF<input type="file" accept="application/pdf" onChange={upload} /></label>
          <div className="simpleDocumentList">
            {documents.map((document) => (
              <button key={document.id} className={selected.includes(document.id) ? 'selected' : ''} onClick={() => toggleDocument(document.id)}>
                <span className="simpleCheck">{selected.includes(document.id) && <Check size={12} />}</span>
                <span><b>{document.filename}</b><small>{document.page_count} pages</small></span>
              </button>
            ))}
            {!documents.length && <p className="simpleHint">Upload a paper to begin.</p>}
          </div>
          <div className="simpleHistory">
            <div className="simpleSectionTitle"><Clock3 size={15} /><b>Recent runs</b></div>
            {runs.slice(0, 8).map((item) => (
              <button key={item.id} onClick={() => { window.localStorage.setItem(storageKey, item.id); refreshRun(item.id); }}>
                <span>{item.question}</span><small>{item.status} · {item.model_calls} call{item.model_calls === 1 ? '' : 's'}</small>
              </button>
            ))}
          </div>
        </aside>

        <section className="simpleWorkspace">
          <form className="simpleQuestion" onSubmit={submit}>
            <label>Research question</label>
            <textarea value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="What problem does this paper address, and what evidence supports it?" />
            <footer>
              <div className="simpleModelControls"><label>Model<select value={model} onChange={(event) => setModel(event.target.value as 'fast' | 'default' | 'reasoning')}><option value="fast">Fast model</option><option value="default">Full model</option><option value="reasoning">Reasoning model</option></select></label>{model === 'reasoning' && <label>Reasoning effort<select value={reasoningEffort} onChange={(event) => setReasoningEffort(event.target.value as 'low' | 'medium' | 'high')}><option value="low">Low</option><option value="medium">Medium</option><option value="high">High</option></select></label>}</div>
              {active ? <button type="button" className="simpleStop" onClick={cancel}><Square size={13} /> Cancel</button> : <button disabled={!question.trim() || !selected.length}><Play size={14} /> Research</button>}
            </footer>
          </form>

          {run && (
            <div className="simpleProgress">
              <div className="simpleRunMeta"><span>{run.id}</span><b className={run.status}>{run.status}</b></div>
              <div className="simpleStages">
                {stages.map((stage, index) => <div key={stage} className={index <= currentStage ? 'reached' : ''}><span>{active && index === currentStage ? <LoaderCircle size={13} className="spin" /> : index < currentStage || run.status === 'completed' ? <Check size={13} /> : index + 1}</span><b>{stage}</b>{index < stages.length - 1 && <ChevronRight size={13} />}</div>)}
              </div>
            </div>
          )}

          {error && <div className="simpleError">{error}</div>}
          {run?.error && <div className="simpleError">{run.error}</div>}

          {run && (run.evidence.length > 0 || firstRequest) && (
            <section className="simpleDiagnostics" aria-label="Run timing and request diagnostics">
              <span><b>Evidence</b>{run.evidence.length} passages</span>
              {run.timings_ms.retrieval !== undefined && <span><b>Retrieval</b>{Math.round(run.timings_ms.retrieval / 1000)}s</span>}
              {run.timings_ms.generation !== undefined && <span><b>Generation</b>{Math.round(run.timings_ms.generation / 1000)}s</span>}
              {run.timings_ms.repair !== undefined && <span><b>Repair</b>{Math.round(run.timings_ms.repair / 1000)}s</span>}
              {firstRequest && <span><b>Request</b>~{firstRequest.estimated_prompt_tokens.toLocaleString()} in · {firstRequest.requested_max_tokens.toLocaleString()} out</span>}
              {firstRequest?.provider_completion_tokens != null && <span><b>Provider output</b>{firstRequest.provider_completion_tokens.toLocaleString()} tokens</span>}
              {firstRequest && <span><b>Outcome</b>{firstRequest.outcome}{firstRequest.finish_reason ? ` · ${firstRequest.finish_reason}` : ''}</span>}
            </section>
          )}

          {run?.result ? (
            <div className="simpleResults">
              <article className="simpleAnswer"><span>Answer</span><p>{run.result.answer}</p></article>
              <section>
                <div className="simpleSectionTitle"><BookOpen size={16} /><b>Source evidence</b><span>{run.result.citations.length}</span></div>
                <div className="simpleCitationGrid">
                  {run.result.citations.map((citation) => <a key={`${citation.evidence_id}:${citation.quote}`} href={`${API_BASE}/api/documents/${citation.document_id}/pdf#page=${citation.page}`} target="_blank" rel="noreferrer"><small>Page {citation.page} · {citation.evidence_id}</small><blockquote>{citation.quote}</blockquote></a>)}
                </div>
              </section>
              <section>
                <div className="simpleSectionTitle"><Network size={16} /><b>Extracted relationships</b><span>{run.result.triples.length}</span></div>
                <div className="simpleTripleGrid">
                  <div>{run.result.triples.map((triple) => <article key={triple.id} title={triple.predicate_uri}><b><small>{triple.subject_type}</small>{triple.subject}</b><i>{triple.predicate}</i><b><small>{triple.object_type}</small>{triple.object}</b></article>)}</div>
                  <div className="simpleMap"><TripleMapViewer snapshot={map} /></div>
                </div>
              </section>
              {!!run.result.limitations.length && <section className="simpleLimitations"><b>Limitations</b><ul>{run.result.limitations.map((item) => <li key={item}>{item}</li>)}</ul></section>}
              <footer className="simpleTiming">Completed with {run.model_calls} model call{run.model_calls === 1 ? '' : 's'} in {Math.round((run.timings_ms.total ?? 0) / 1000)}s</footer>
              <details className="simpleRaw"><summary>Raw validated run JSON</summary><pre>{JSON.stringify(run, null, 2)}</pre></details>
            </div>
          ) : !run ? (
            <div className="simpleEmpty"><Network size={28} /><b>Ask one clear question.</b><p>The app retrieves bounded passages, asks one model to select relevant passage numbers, then creates every citation and relationship link on the host.</p></div>
          ) : null}
        </section>
      </div>
    </main>
  );
}
