'use client';
import { FormEvent, useEffect, useMemo, useState } from 'react';
import dynamic from 'next/dynamic';
import { useChat } from '@ai-sdk/react';
import { DefaultChatTransport } from 'ai';
import { Database, FileStack, History, Network, Search, Send, Settings2, Sparkles } from 'lucide-react';
import { AgentTrace } from './AgentTrace';
import { ArtifactInspector } from './ArtifactInspector';
import { DocumentManager } from './DocumentManager';
import { getArtifact, getRun, getTripleMap, listArtifacts, listDocuments, listRuns, uploadPdf, deleteDocument } from '@/lib/api';
import type { ArtifactEnvelope, ArtifactMeta, DocumentInfo, ResearchUIMessage, RunInfo, TraceEvent, TripleMapSnapshot } from '@/lib/types';

// Sigma requires WebGL and must never be evaluated by Next.js on the server.
const TripleMapViewer = dynamic(
 () => import('./TripleMapViewer').then((module) => module.TripleMapViewer),
 { ssr: false, loading: () => <div className="emptyState">Loading triple map…</div> },
);

type View='research'|'documents'|'triple-map'|'artifacts'|'runs';
export function ResearchConsole(){
 const workspaceId='default';
 const [docs,setDocs]=useState<DocumentInfo[]>([]); const [selected,setSelected]=useState<string[]>([]); const [view,setView]=useState<View>('research');
 const [input,setInput]=useState(''); const [trace,setTrace]=useState<TraceEvent[]>([]); const [runId,setRunId]=useState<string>();
 const [run,setRun]=useState<RunInfo>(); const [runs,setRuns]=useState<RunInfo[]>([]); const [map,setMap]=useState<TripleMapSnapshot>({nodes:[],edges:[]});
 const [artifacts,setArtifacts]=useState<ArtifactMeta[]>([]); const [artifact,setArtifact]=useState<ArtifactEnvelope>();
 const transport=useMemo(()=>new DefaultChatTransport({api:'/api/chat',prepareSendMessagesRequest:({messages})=>({body:{messages,documentIds:selected,workspaceId}})}),[selected,workspaceId]);
 const {messages,sendMessage,status,error}=useChat<ResearchUIMessage>({transport,onData:(part:any)=>{if(part.type==='data-trace')setTrace(old=>old.some(x=>x.id===part.data.id)?old:[...old,part.data]);if(part.type==='data-run')setRunId(part.data.runId);}});
 async function refreshDocs(){setDocs(await listDocuments(workspaceId));}
 async function refreshRun(id:string){const next=await getRun(id);setRun(next);setMap(await getTripleMap(id));setArtifacts(await listArtifacts(id));}
 useEffect(()=>{refreshDocs().catch(()=>{});listRuns(workspaceId).then(setRuns).catch(()=>{});},[]);
 useEffect(()=>{if(runId)refreshRun(runId).catch(()=>{});},[runId,trace.length]);
 async function upload(file?:File){if(!file)return;const d=await uploadPdf(file,workspaceId);setDocs(x=>[d,...x]);setSelected(x=>[...x,d.id]);}
 async function remove(id:string){await deleteDocument(id);setSelected(x=>x.filter(v=>v!==id));setDocs(x=>x.filter(v=>v.id!==id));}
 function toggle(id:string){setSelected(x=>x.includes(id)?x.filter(v=>v!==id):[...x,id]);}
 async function inspect(id:string){if(!runId)return;setArtifact(await getArtifact(runId,id));setView('artifacts');}
 async function openHistorical(r:RunInfo){setRunId(r.id);setRun(r);setMap(await getTripleMap(r.id));setArtifacts(await listArtifacts(r.id));setView('artifacts');}
 async function submit(e:FormEvent){e.preventDefault();if(!input.trim()||!selected.length)return;const text=input.trim();setInput('');setTrace([]);setRun(undefined);setRunId(undefined);setMap({nodes:[],edges:[]});setArtifacts([]);setArtifact(undefined);await sendMessage({text});}
 return <div className="appShell">
  <aside className="nav"><div className="mark"><Sparkles size={18}/></div><button className={view==='research'?'active':''} onClick={()=>setView('research')}><Search/><span>Research</span></button><button className={view==='documents'?'active':''} onClick={()=>setView('documents')}><FileStack/><span>Documents</span></button><button className={view==='triple-map'?'active':''} onClick={()=>setView('triple-map')}><Network/><span>Triple map</span></button><button className={view==='artifacts'?'active':''} onClick={()=>setView('artifacts')}><Database/><span>Artifacts</span></button><button className={view==='runs'?'active':''} onClick={()=>{setView('runs');listRuns(workspaceId).then(setRuns).catch(()=>{});}}><History/><span>Runs</span></button><div className="navGrow"/><button><Settings2/><span>Settings</span></button></aside>
  <main className="console"><header><div><b>ADI Research Console</b><span>dynamic specialists · lexical evidence retrieval · canonical triples + provenance</span></div><div className="runtime"><i className={status==='streaming'||status==='submitted'?'busy':''}/>{run?.status??(status==='ready'?'ready':status)}</div></header>
  {view==='research'&&<div className="researchGrid"><section className="chatPane"><div className="contextBar"><span>{selected.length} paper{selected.length===1?'':'s'} selected{runId?` · run ${runId}`:''}</span><button onClick={()=>setView('documents')}>manage corpus</button></div><div className="messages">
    {messages.length===0&&<div className="welcome"><Sparkles size={30}/><h1>Ask the papers, watch the research.</h1><p>The Director can invent bounded specialists, search evidence lexically, adapt a run-local DBLP ontology overlay, run EDC after retrieval, criticize candidates and recover from small-model failures. The trace exposes messages, prompts, decisions, tools, outputs and retries—not hidden chain-of-thought.</p></div>}
    {messages.map(m=><div key={m.id} className={`message ${m.role}`}>{m.parts.filter((p:any)=>p.type==='text').map((p:any,i)=><p key={i}>{p.text}</p>)}</div>)}
    {run&&run.final_triples.length>0&&<div className="finalTriples"><div className="inspectKind"><ShieldLabel/>Accepted evidence-backed triples</div>{run.final_triples.map(t=><article key={t.id} className="verifiedCard"><div><b>{t.subject}</b><i>{t.predicate}</i><b>{t.object}</b></div><p>{t.predicate_uri} · {t.evidence_ids.length} evidence span{t.evidence_ids.length===1?'':'s'}</p></article>)}</div>}
  </div><form className="composer" onSubmit={submit}><textarea value={input} onChange={e=>setInput(e.target.value)} placeholder={selected.length?'Ask a research question…':'Select at least one paper first'} disabled={!selected.length}/><button disabled={!selected.length||!input.trim()||status!=='ready'}><Send size={17}/></button></form>{error&&<div className="errorBar">{error.message}</div>}</section><AgentTrace events={trace} onArtifact={inspect}/></div>}
  {view==='documents'&&<div className="singlePane"><DocumentManager documents={docs} selected={selected} onToggle={toggle} onUpload={upload} onDelete={remove}/></div>}
  {view==='triple-map'&&<div className="singlePane graphPane"><div className="paneTitle"><div><b>Temporary run triple map</b><span>{runId?`${map.nodes.length} entities · ${map.edges.length} accepted triples · derived from ${runId}`:'select or complete a run'}</span></div></div><TripleMapViewer snapshot={map}/></div>}
  {view==='artifacts'&&<div className="artifactGrid"><aside><div className="paneTitle"><b>Immutable run artifacts</b></div>{artifacts.map(a=><button key={a.id} onClick={()=>inspect(a.id)} className="artifactRow"><span>{a.label??a.kind}</span><small>{a.kind} · {a.id}</small></button>)}</aside><section><div className="paneTitle"><b>Artifact inspector</b><span>triple → evidence ID → exact source page</span></div><ArtifactInspector payload={artifact} runId={runId}/></section></div>}
  {view==='runs'&&<div className="singlePane"><div className="docManager"><div className="sectionHeader"><div><b>Experiment run history</b><span>completed · partial · failed · cancelled · resumable</span></div></div><div className="runList">{runs.map(r=><button key={r.id} className="runRow" onClick={()=>openHistorical(r)}><b>{r.question}</b><span>{r.status} · {r.final_triples.length} accepted triples · {new Date(r.created_at).toLocaleString()}</span>{r.error&&<small>{r.error}</small>}</button>)}</div></div></div>}
  </main></div>;
}
function ShieldLabel(){return <span className="statusTag">verified</span>}
