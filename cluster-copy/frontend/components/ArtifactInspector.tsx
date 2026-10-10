'use client';
import { useState } from 'react';
import { Braces, ExternalLink, FileSearch2, Network, ShieldCheck } from 'lucide-react';
import { API_BASE, getEvidence } from '@/lib/api';
import type { ArtifactEnvelope } from '@/lib/types';

type Source = { span:any; pdf_url:string; artifact_id:string };
export function ArtifactInspector({payload,runId}:{payload:ArtifactEnvelope|undefined;runId?:string}){
 const [source,setSource]=useState<Source>();
 async function openEvidence(id:string){ if(!runId)return; setSource(await getEvidence(runId,id)); }
 const x:any=payload?.content;
 const triples=payload?.artifact_type==='canonical_triples'&&Array.isArray(x);
 const rejected=payload?.artifact_type==='rejected_triples'&&Array.isArray(x);
 const evidence=payload?.artifact_type==='evidence_bundle'&&x&&Array.isArray(x.spans);
 const singleEvidence=payload?.artifact_type==='evidence_span'&&x?.text;
 const sparql=x&&typeof x==='object'&&typeof x.query==='string';
 const ontology=x&&typeof x==='object'&&Array.isArray(x.selected_elements);
 const validation=x&&typeof x==='object'&&typeof x.shacl_conforms==='boolean';
 return <div className={`inspector ${source?'withSource':''}`}>
  <div className="inspectMain">
   {!payload&&<div className="emptyState">Select an immutable run artifact or click one in the live trace.</div>}
   {payload&&<div className="artifactEnvelope"><b>{payload.artifact_type}</b><span>{payload.artifact_id} · sha256 {payload.content_sha256.slice(0,12)}… · creator {payload.creator_agent_id}</span></div>}
   {triples&&<div className="verifiedList"><div className="inspectKind"><ShieldCheck size={15}/>Accepted canonical triples</div>{x.map((t:any)=><article key={t.id} className="verifiedCard"><div><span className={`statusTag ${t.verification_status}`}>{t.verification_status}</span><b>{t.subject}</b><i>{t.predicate}</i><b>{t.object}</b></div><p>{t.predicate_uri} · canonicalization: {t.canonicalization}</p><footer>{(t.evidence_ids??[]).map((id:string)=><button key={id} onClick={()=>openEvidence(id)}><FileSearch2 size={12}/>{id}</button>)}</footer></article>)}</div>}
   {rejected&&<div className="verifiedList"><div className="inspectKind"><ShieldCheck size={15}/>Rejected triples retained for evaluation</div>{x.map((r:any)=><article key={r.triple?.id??r.reason} className="verifiedCard"><div><span className="statusTag unsupported">rejected</span><b>{r.triple?.subject}</b><i>{r.triple?.predicate}</i><b>{r.triple?.object}</b></div><p>{r.reason}</p></article>)}</div>}
   {evidence&&<div className="evidenceList"><div className="inspectKind"><FileSearch2 size={15}/>Evidence bundle</div>{x.spans.map((s:any)=><article key={s.id} className="evidenceCard"><header><b>{s.id}</b><span>page {s.page} · {s.section_id??'no section'} · {s.retrieval_method}</span></header><p>{s.text}</p><button onClick={()=>openEvidence(s.id)}><ExternalLink size={12}/>source page</button></article>)}</div>}
   {singleEvidence&&<div className="evidenceList"><div className="inspectKind"><FileSearch2 size={15}/>Immutable evidence span</div><article className="evidenceCard"><header><b>{x.id}</b><span>page {x.page} · {x.block_id??'block unavailable'}</span></header><p>{x.text}</p><button onClick={()=>openEvidence(x.id)}><ExternalLink size={12}/>source page</button></article></div>}
   {sparql&&<div className="codeArtifact"><div className="inspectKind"><Braces size={15}/>SPARQL semantic pattern</div><pre>{x.query}</pre><div className="artifactMeta">valid: {String(x.valid)} · {x.purpose}</div></div>}
   {ontology&&<div className="ontologyArtifact"><div className="inspectKind"><Network size={15}/>Active run ontology snapshot</div><div className="ontologyList">{x.selected_elements.map((o:any)=><div key={o.uri}><b>{o.label}</b><span>{o.kind} · {o.source}</span><p>{o.description}</p></div>)}</div></div>}
   {validation&&<div className="codeArtifact"><div className="inspectKind"><ShieldCheck size={15}/>Structural validation (not factual truth)</div><div className="validationGrid"><b>RDF valid</b><span>{String(x.rdf_valid)}</span><b>SHACL conforms</b><span>{String(x.shacl_conforms)}</span></div><pre>{x.shacl_report}</pre></div>}
   {payload&&!triples&&!rejected&&!evidence&&!singleEvidence&&!sparql&&!ontology&&!validation&&<pre className="rawArtifact">{JSON.stringify(x,null,2)}</pre>}
  </div>
  {source&&<aside className="sourcePeek"><div className="sourceHead"><div><b>Source evidence</b><span>{source.span.id} · page {source.span.page}</span></div><button onClick={()=>setSource(undefined)}>×</button></div><div className="sourceText">{source.span.text}</div><iframe src={`${API_BASE}${source.pdf_url}`} title="Evidence source PDF"/></aside>}
 </div>;
}
