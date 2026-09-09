import type { UIMessage } from 'ai';

export type DocumentInfo = {
  id:string; filename:string; page_count:number; status:string; created_at:string;
  parse_engine?:string; parse_warnings?:string[];
  metadata?:{title?:string;authors?:string[];doi?:string;year?:number;venue?:string};
  sections?:Array<{id:string;heading:string;page_start:number;page_end:number}>;
};
export type TraceEvent = {
  id:string; run_id:string; type:string; actor:string; action:string; status:string;
  title:string; summary:string; artifact_id?:string; parent_event_id?:string;
  payload:Record<string,unknown>; timestamp:string;
};
export type CanonicalTriple = {
  id:string; subject:string; predicate:string; object:string; predicate_uri?:string;
  subject_type?:string; object_type?:string; evidence_ids:string[];
  canonicalization_decision_id?:string; verification_status:string;
  verifier_artifact_ids:string[]; ontology_snapshot_id?:string;
  canonicalization:string; confidence_optional?:number;
};
export type RunInfo = {
  id:string; workspace_id:string; document_ids:string[]; target_document_id?:string; question:string;
  status:string; active_actor?:string; created_at:string; updated_at:string;
  events:TraceEvent[]; final_answer?:string; final_triples:CanonicalTriple[];
  rejected_triples:Array<{triple:CanonicalTriple;reason:string}>;
  evaluation?:Record<string,unknown>; resumable:boolean; error?:string;
};
export type TripleMapSnapshot = {
  nodes:Array<{id:string;label?:string;kind?:string}>;
  edges:Array<{id:string;source:string;target:string;label?:string;predicate?:string;status?:string}>;
};
export type ArtifactMeta = {
  id:string; kind:string; path:string; label?:string; content_sha256?:string;
  creator_agent_id?:string; parent_artifact_ids?:string[]; schema_version?:string; created_at?:string;
};
export type ArtifactEnvelope = {
  artifact_id:string; artifact_type:string; run_id:string; creator_agent_id:string;
  parent_artifact_ids:string[]; created_at:string; schema_version:string;
  content_sha256:string; content:unknown;
};
export type ResearchDataParts = {
  trace: TraceEvent;
  run: {runId:string; status:string};
  artifact: ArtifactMeta;
  answer: {runId:string; answer:string};
};
export type ResearchUIMessage = UIMessage<never, ResearchDataParts>;

export type SimpleCitation = {
  evidence_id:string; document_id:string; page:number; quote:string;
};
export type SimpleTriple = {
  id:string; subject:string; subject_type:string; predicate:string; predicate_key:string;
  predicate_uri:string; object:string; object_type:string; evidence_ids:string[];
};
export type SimpleResearchResult = {
  answer:string; citations:SimpleCitation[]; triples:SimpleTriple[]; limitations:string[];
};
export type SimpleResearchRun = {
  id:string; question:string; document_ids:string[]; model:string; model_mode:'fast'|'default'|'reasoning'|'custom'; reasoning_effort?:'low'|'medium'|'high';
  status:'queued'|'running'|'completed'|'failed'|'cancelled';
  stage:'queued'|'retrieving'|'generating'|'validating'|'repairing'|'completed'|'failed'|'cancelled';
  evidence:Array<{id:string;document_id:string;page:number;text:string;score?:number}>;
  result?:SimpleResearchResult; timings_ms:Record<string,number>; model_calls:number;
  execution_profile?:{retrieval_per_document:number;max_passages:number;passage_max_chars:number;requested_max_tokens:number;reasoning_effort?:'low'|'medium'|'high'};
  model_request_diagnostics?:Array<{stage:'generation'|'repair';model:string;estimated_prompt_tokens:number;requested_max_tokens:number;context_window_tokens?:number;timeout_seconds:number;elapsed_ms?:number;outcome:'pending'|'completed'|'timed_out'|'failed';error_type?:string;finish_reason?:string;provider_prompt_tokens?:number;provider_completion_tokens?:number;visible_output_chars?:number;reasoning_content_chars?:number}>;
  error?:string; created_at:string; updated_at:string;
};
