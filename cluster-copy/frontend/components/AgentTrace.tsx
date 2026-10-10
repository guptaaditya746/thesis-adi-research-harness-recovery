'use client';
import { Bot, Wrench, GitBranch, FileOutput, Search, ShieldCheck, Database } from 'lucide-react';
import type { TraceEvent } from '@/lib/types';
const icon=(e:TraceEvent)=> e.type==='delegate'?<GitBranch size={15}/>:e.type==='tool'?<Wrench size={15}/>:e.type==='artifact'?<FileOutput size={15}/>:e.type==='evidence'?<Search size={15}/>:e.type==='validation'?<ShieldCheck size={15}/>:e.type==='triple'?<Database size={15}/>:<Bot size={15}/>;
export function AgentTrace({events,onArtifact}:{events:TraceEvent[];onArtifact:(id:string)=>void}){
 return <div className="trace"><div className="traceHead"><b>Live agent trace</b><span>{events.length} actions</span></div><div className="traceList">
 {events.length===0&&<div className="traceEmpty">Ask a question. Agent choices, delegations, searches and artifacts will appear here as they happen.</div>}
 {events.map(e=><button key={e.id} className={`traceItem ${e.status}`} onClick={()=>e.artifact_id&&onArtifact(e.artifact_id)}><div className="traceIcon">{icon(e)}</div><div><div className="traceMeta"><b>{e.actor}</b><span>{e.action}</span></div><strong>{e.title}</strong>{e.summary&&<p>{e.summary}</p>}<small>{new Date(e.timestamp).toLocaleTimeString()}</small></div></button>)}
 </div></div>;
}
