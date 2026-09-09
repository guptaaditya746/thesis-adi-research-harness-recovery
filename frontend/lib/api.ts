import { ArtifactEnvelope, ArtifactMeta, DocumentInfo, RunInfo, SimpleResearchRun, TripleMapSnapshot } from './types';
// Browser requests stay same-origin. Next proxies this path to the configured
// backend, so deployments do not depend on the backend's CORS configuration.
export const API_BASE = '/backend';

async function checked<T>(r:Response):Promise<T>{if(!r.ok)throw new Error(await r.text());return r.json() as Promise<T>;}
export async function listDocuments(workspaceId='default'): Promise<DocumentInfo[]> { return checked(await fetch(`${API_BASE}/api/documents?workspace_id=${encodeURIComponent(workspaceId)}`,{cache:'no-store'})); }
export async function uploadPdf(file:File, workspaceId='default'): Promise<DocumentInfo> { const b=new FormData(); b.append('file',file); b.append('workspace_id',workspaceId); return checked(await fetch(`${API_BASE}/api/documents`,{method:'POST',body:b})); }
export async function deleteDocument(id:string){ const r=await fetch(`${API_BASE}/api/documents/${id}`,{method:'DELETE'}); if(!r.ok)throw new Error(await r.text()); }
export async function getRun(id:string):Promise<RunInfo>{return checked(await fetch(`${API_BASE}/api/runs/${id}`,{cache:'no-store'}));}
export async function listRuns(workspaceId='default'):Promise<RunInfo[]>{return checked(await fetch(`${API_BASE}/api/runs?workspace_id=${encodeURIComponent(workspaceId)}`,{cache:'no-store'}));}
export async function resumeRun(id:string){return checked<{ok:boolean;status:string}>(await fetch(`${API_BASE}/api/runs/${id}/resume`,{method:'POST'}));}
export async function cancelRun(id:string){return checked<{ok:boolean;note:string}>(await fetch(`${API_BASE}/api/runs/${id}/cancel`,{method:'POST'}));}
export async function listArtifacts(runId:string):Promise<ArtifactMeta[]>{return checked(await fetch(`${API_BASE}/api/runs/${runId}/artifacts`,{cache:'no-store'}));}
export async function getArtifact(runId:string,id:string):Promise<ArtifactEnvelope>{return checked(await fetch(`${API_BASE}/api/runs/${runId}/artifacts/${id}`,{cache:'no-store'}));}
export async function getTripleMap(runId:string):Promise<TripleMapSnapshot>{return checked(await fetch(`${API_BASE}/api/runs/${runId}/triple-map`,{cache:'no-store'}));}
export async function getEvidence(runId:string,id:string):Promise<{span:any;pdf_url:string;artifact_id:string}>{return checked(await fetch(`${API_BASE}/api/runs/${runId}/evidence/${id}`,{cache:'no-store'}));}
export async function startSimpleResearch(question:string,documentIds:string[],model:'fast'|'default'|'reasoning'='fast',reasoningEffort?:'low'|'medium'|'high'):Promise<SimpleResearchRun>{return checked(await fetch(`${API_BASE}/api/simple/research`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question,document_ids:documentIds,model,reasoning_effort:reasoningEffort})}));}
export async function getSimpleRun(id:string):Promise<SimpleResearchRun>{return checked(await fetch(`${API_BASE}/api/simple/runs/${id}`,{cache:'no-store'}));}
export async function listSimpleRuns():Promise<SimpleResearchRun[]>{return checked(await fetch(`${API_BASE}/api/simple/runs`,{cache:'no-store'}));}
export async function cancelSimpleRun(id:string):Promise<SimpleResearchRun>{return checked(await fetch(`${API_BASE}/api/simple/runs/${id}/cancel`,{method:'POST'}));}
export async function getSimpleTripleMap(id:string):Promise<TripleMapSnapshot>{return checked(await fetch(`${API_BASE}/api/simple/runs/${id}/triple-map`,{cache:'no-store'}));}
