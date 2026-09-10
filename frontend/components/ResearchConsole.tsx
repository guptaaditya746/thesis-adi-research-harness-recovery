'use client';

import { FormEvent, useEffect, useMemo, useState } from 'react';
import dynamic from 'next/dynamic';
import Link from 'next/link';
import { useChat } from '@ai-sdk/react';
import { DefaultChatTransport } from 'ai';
import {
  Database,
  Gauge,
  FileStack,
  History,
  Network,
  Search,
  Send,
  Settings2,
  Sparkles,
} from 'lucide-react';

import { AgentTrace } from './AgentTrace';
import { ArtifactInspector } from './ArtifactInspector';
import { DocumentManager } from './DocumentManager';
import { EDCProgressPanel, ScientificStateStrip } from './ScientificProgress';

import {
  deleteDocument,
  getArtifact,
  getRun,
  getTripleMap,
  listArtifacts,
  listDocuments,
  listRuns,
  uploadPdf,
} from '@/lib/api';

import type {
  ArtifactEnvelope,
  ArtifactMeta,
  DocumentInfo,
  ResearchUIMessage,
  RunInfo,
  TraceEvent,
  TripleMapSnapshot,
} from '@/lib/types';

const TripleMapViewer = dynamic(
  () =>
    import('./TripleMapViewer').then((module) => module.TripleMapViewer),
  {
    ssr: false,
  },
);

type View =
  | 'research'
  | 'documents'
  | 'triple-map'
  | 'artifacts'
  | 'runs';

export function ResearchConsole() {
  const workspaceId = 'default';

  const [docs, setDocs] = useState<DocumentInfo[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [targetDocumentId, setTargetDocumentId] = useState<string>();
  const [view, setView] = useState<View>('research');

  const [input, setInput] = useState('');
  const [trace, setTrace] = useState<TraceEvent[]>([]);
  const [runId, setRunId] = useState<string>();

  const [run, setRun] = useState<RunInfo>();
  const [runs, setRuns] = useState<RunInfo[]>([]);

  const [map, setMap] = useState<TripleMapSnapshot>({
    nodes: [],
    edges: [],
  });

  const [artifacts, setArtifacts] = useState<ArtifactMeta[]>([]);
  const [artifact, setArtifact] = useState<ArtifactEnvelope>();
  const [edcArtifacts, setEdcArtifacts] = useState<ArtifactEnvelope[]>([]);
  const [tripleMapMode, setTripleMapMode] = useState<'accepted' | 'edc'>('accepted');

  const transport = useMemo(
    () =>
      new DefaultChatTransport<ResearchUIMessage>({
        api: '/api/chat',
        prepareSendMessagesRequest: ({
          messages,
        }: {
          messages: ResearchUIMessage[];
        }) => ({
          body: {
            messages,
            documentIds: selected,
            targetDocumentId: targetDocumentId ?? selected[0],
            workspaceId,
          },
        }),
      }),
    [selected, targetDocumentId, workspaceId],
  );

  const {
    messages,
    sendMessage,
    status,
    error,
  } = useChat<ResearchUIMessage>({
    transport,

    onData: (part: any) => {
      if (part.type === 'data-trace') {
        setTrace((old) =>
          old.some((event) => event.id === part.data.id)
            ? old
            : [...old, part.data],
        );
      }

      if (part.type === 'data-run') {
        setRunId(part.data.runId);
      }
    },
  });

  async function refreshDocs() {
    const documents = await listDocuments(workspaceId);
    setDocs(documents);
  }

  async function refreshRun(id: string) {
    const [nextRun, nextMap, nextArtifacts] = await Promise.all([
      getRun(id),
      getTripleMap(id),
      listArtifacts(id),
    ]);
    const progressKinds = new Set([
      'edc_open_triples',
      'edc_relation_definitions',
      'edc_canonicalization',
      'edc_canonicalization_attempt',
      'ontology_snapshot',
      'candidate_canonical_triples',
    ]);
    const latestProgress = [...nextArtifacts]
      .reverse()
      .filter((item, index, rows) => progressKinds.has(item.kind) && rows.findIndex((candidate) => candidate.kind === item.kind) === index)
      .map((item) => item.id);

    setRun(nextRun);
    setMap(nextMap);
    setArtifacts(nextArtifacts);
    setEdcArtifacts(await Promise.all(latestProgress.map((artifactId) => getArtifact(id, artifactId))));
  }

  useEffect(() => {
    refreshDocs().catch(() => {});

    listRuns(workspaceId)
      .then(setRuns)
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (!runId) {
      return;
    }

    refreshRun(runId).catch(() => {});
  }, [runId, trace.length]);

  async function upload(file?: File) {
    if (!file) {
      return;
    }

    const document = await uploadPdf(file, workspaceId);

    setDocs((current) => [document, ...current]);
    setSelected((current) => [...current, document.id]);
    setTargetDocumentId((current) => current ?? document.id);
  }

  async function remove(id: string) {
    await deleteDocument(id);

    setSelected((current) =>
      current.filter((value) => value !== id),
    );
    setTargetDocumentId((current) => current === id ? undefined : current);

    setDocs((current) =>
      current.filter((document) => document.id !== id),
    );
  }

  function toggle(id: string) {
    setSelected((current) => {
      const next = current.includes(id) ? current.filter((value) => value !== id) : [...current, id];
      setTargetDocumentId((target) => next.includes(target ?? '') ? target : next[0]);
      return next;
    });
  }

  async function inspect(id: string) {
    if (!runId) {
      return;
    }

    const nextArtifact = await getArtifact(runId, id);

    setArtifact(nextArtifact);
    setView('artifacts');
  }

  async function openHistorical(nextRun: RunInfo) {
    setRunId(nextRun.id);
    await refreshRun(nextRun.id);
    setView('artifacts');
  }

  async function submit(event: FormEvent) {
    event.preventDefault();

    if (!input.trim() || !selected.length) {
      return;
    }

    const text = input.trim();

    setInput('');
    setTrace([]);
    setRun(undefined);
    setRunId(undefined);
    setMap({
      nodes: [],
      edges: [],
    });
    setArtifacts([]);
    setArtifact(undefined);
    setEdcArtifacts([]);
    setTripleMapMode('accepted');

    await sendMessage({ text });
  }

  return (
    <div className="appShell">
      <aside className="nav">
        <div className="mark">
          <Sparkles size={18} />
        </div>

        <button
          className={view === 'research' ? 'active' : ''}
          onClick={() => setView('research')}
        >
          <Search />
          <span>Research</span>
        </button>

        <button
          className={view === 'documents' ? 'active' : ''}
          onClick={() => setView('documents')}
        >
          <FileStack />
          <span>Documents</span>
        </button>

        <button
          className={view === 'triple-map' ? 'active' : ''}
          onClick={() => setView('triple-map')}
        >
          <Network />
          <span>Triple map</span>
        </button>

        <button
          className={view === 'artifacts' ? 'active' : ''}
          onClick={() => setView('artifacts')}
        >
          <Database />
          <span>Artifacts</span>
        </button>

        <button
          className={view === 'runs' ? 'active' : ''}
          onClick={() => {
            setView('runs');

            listRuns(workspaceId)
              .then(setRuns)
              .catch(() => {});
          }}
        >
          <History />
          <span>Runs</span>
        </button>

        <div className="navGrow" />

        <Link href="/simple" className="navDataLink">
          <Gauge />
          <span>Simple</span>
        </Link>

        <Link href="/run-data" className="navDataLink">
          <Database />
          <span>JSON data</span>
        </Link>

        <button>
          <Settings2 />
          <span>Settings</span>
        </button>
      </aside>

      <main className="console">
        <header>
          <div>
            <b>Scientific Literature Research Harness</b>
            <span>Evidence-grounded literature analysis · fixed and agentic workflows</span>
          </div>

          <div className="runtime">
            <i
              className={
                status === 'streaming' || status === 'submitted'
                  ? 'busy'
                  : ''
              }
            />

            {run?.status ?? (status === 'ready' ? 'ready' : status)}
          </div>
        </header>

        {view === 'research' && (
          <div className="researchGrid">
            <section className="chatPane">
              <ScientificStateStrip artifacts={artifacts} onArtifact={inspect} />
              <div className="contextBar">
                <span>
                  {selected.length} paper
                  {selected.length === 1 ? '' : 's'} selected
                  {targetDocumentId ? ` · target ${targetDocumentId}` : ''}
                  {runId ? ` · run ${runId}` : ''}
                </span>

                {selected.length > 1 && (
                  <label>
                    target{' '}
                    <select value={targetDocumentId ?? selected[0]} onChange={(event) => setTargetDocumentId(event.target.value)}>
                      {selected.map((id) => <option key={id} value={id}>{docs.find((doc) => doc.id === id)?.filename ?? id}</option>)}
                    </select>
                  </label>
                )}

                <button onClick={() => setView('documents')}>
                  manage corpus
                </button>
              </div>

              <div className="messages">
                {messages.length === 0 && (
                  <div className="welcome">
                    <Sparkles size={30} />

                    <h1>Ask the papers, watch the research.</h1>

                    <p>
                      The Director can invent bounded specialists, search
                      evidence lexically, adapt a run-local DBLP ontology
                      overlay, run EDC after retrieval, criticize candidates
                      and recover from small-model failures. The trace exposes
                      messages, prompts, decisions, tools, outputs and
                      retries—not hidden chain-of-thought.
                    </p>
                  </div>
                )}

                {messages.map((message) => (
                  <div
                    key={message.id}
                    className={`message ${message.role}`}
                  >
                    {message.parts
                      .filter((part: any) => part.type === 'text')
                      .map((part: any, index) => (
                        <p key={index}>{part.text}</p>
                      ))}
                  </div>
                ))}

                {run && run.final_triples.length > 0 && (
                  <div className="finalTriples">
                    <div className="inspectKind">
                      <ShieldLabel />
                      Accepted evidence-backed triples
                    </div>

                    {run.final_triples.map((triple) => (
                      <article
                        key={triple.id}
                        className="verifiedCard"
                      >
                        <div>
                          <b>{triple.subject}</b>
                          <i>{triple.predicate}</i>
                          <b>{triple.object}</b>
                        </div>

                        <p>
                          {triple.predicate_uri} ·{' '}
                          {triple.evidence_ids.length} evidence span
                          {triple.evidence_ids.length === 1 ? '' : 's'}
                        </p>
                      </article>
                    ))}
                  </div>
                )}
              </div>

              <form
                className="composer"
                onSubmit={submit}
              >
                <textarea
                  value={input}
                  onChange={(event) =>
                    setInput(event.target.value)
                  }
                  placeholder={
                    selected.length
                      ? 'Ask a research question…'
                      : 'Select at least one paper first'
                  }
                  disabled={!selected.length}
                />

                <button
                  disabled={
                    !selected.length ||
                    !input.trim() ||
                    status !== 'ready'
                  }
                >
                  <Send size={17} />
                </button>
              </form>

              {error && (
                <div className="errorBar">
                  {error.message}
                </div>
              )}
            </section>

            <AgentTrace
              events={trace}
              onArtifact={inspect}
            />
          </div>
        )}

        {view === 'documents' && (
          <div className="singlePane">
            <DocumentManager
              documents={docs}
              selected={selected}
              onToggle={toggle}
              onUpload={upload}
              onDelete={remove}
            />
          </div>
        )}

        {view === 'triple-map' && (
          <div className="singlePane graphPane">
            <div className="paneTitle">
              <div>
                <b>{tripleMapMode === 'accepted' ? 'Accepted triple map' : 'EDC progress'}</b>

                <span>
                  {runId
                    ? `${map.nodes.length} entities · ${map.edges.length} accepted triples · derived from ${runId}`
                    : 'select or complete a run'}
                </span>
              </div>
              <div className="viewTabs" role="tablist" aria-label="Triple map views">
                <button className={tripleMapMode === 'accepted' ? 'active' : ''} onClick={() => setTripleMapMode('accepted')}>Accepted triples</button>
                <button className={tripleMapMode === 'edc' ? 'active' : ''} onClick={() => setTripleMapMode('edc')}>EDC progress</button>
              </div>
            </div>

            {tripleMapMode === 'accepted' ? (
              <TripleMapViewer snapshot={map} />
            ) : (
              <EDCProgressPanel artifacts={edcArtifacts} onArtifact={inspect} />
            )}
          </div>
        )}

        {view === 'artifacts' && (
          <div className="artifactGrid">
            <aside>
              <div className="paneTitle">
                <b>Immutable run artifacts</b>
              </div>

              {artifacts.map((item) => (
                <button
                  key={item.id}
                  onClick={() => inspect(item.id)}
                  className="artifactRow"
                >
                  <span>{item.label ?? item.kind}</span>
                  <small>
                    {item.kind} · {item.id}
                  </small>
                </button>
              ))}
            </aside>

            <section>
              <div className="paneTitle">
                <b>Artifact inspector</b>
                <span>
                  triple → evidence ID → exact source page
                </span>
              </div>

              <ArtifactInspector
                payload={artifact}
                runId={runId}
              />
            </section>
          </div>
        )}

        {view === 'runs' && (
          <div className="singlePane">
            <div className="docManager">
              <div className="sectionHeader">
                <div>
                  <b>Experiment run history</b>
                  <span>
                    completed · failed · cancelled · resumable
                  </span>
                </div>
              </div>

              <div className="runList">
                {runs.map((historyRun) => (
                  <button
                    key={historyRun.id}
                    className="runRow"
                    onClick={() => openHistorical(historyRun)}
                  >
                    <b>{historyRun.question}</b>

                    <span>
                      {historyRun.status} ·{' '}
                      {historyRun.final_triples.length} accepted triples ·{' '}
                      {new Date(
                        historyRun.created_at,
                      ).toLocaleString()}
                    </span>

                    {historyRun.error && (
                      <small>{historyRun.error}</small>
                    )}
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

function ShieldLabel() {
  return <span className="statusTag">verified</span>;
}
