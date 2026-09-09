'use client';

import { Check, CircleDashed, X } from 'lucide-react';

import type { ArtifactEnvelope, ArtifactMeta } from '@/lib/types';

type Milestone = {
  kind: string;
  label: string;
};

const milestones: Milestone[] = [
  { kind: 'evidence_bundle', label: 'Evidence' },
  { kind: 'edc_open_triples', label: 'Open triples' },
  { kind: 'edc_relation_definitions', label: 'Definitions' },
  { kind: 'ontology_snapshot', label: 'Ontology' },
  { kind: 'candidate_canonical_triples', label: 'Candidates' },
  { kind: 'verification', label: 'Verification' },
  { kind: 'canonical_triples', label: 'Canonical' },
];

function latestByKind(artifacts: ArtifactMeta[], kind: string) {
  return [...artifacts].reverse().find((artifact) => artifact.kind === kind);
}

export function ScientificStateStrip({
  artifacts,
  onArtifact,
}: {
  artifacts: ArtifactMeta[];
  onArtifact: (id: string) => void;
}) {
  return (
    <div className="scientificState" aria-label="Scientific artifact state">
      <span className="scientificStateLabel">Scientific state</span>
      <div className="scientificMilestones">
        {milestones.map((milestone) => {
          const artifact = latestByKind(artifacts, milestone.kind);
          const status = artifact ? 'available' : 'missing';

          return (
            <button
              key={milestone.kind}
              className={`scientificMilestone ${status}`}
              disabled={!artifact}
              onClick={() => artifact && onArtifact(artifact.id)}
              title={artifact ? `${milestone.kind}: ${artifact.id}` : `${milestone.kind} not available`}
            >
              {artifact ? <Check size={12} /> : <CircleDashed size={12} />}
              <span>{milestone.label}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

function count(content: unknown, key: string) {
  if (!content || typeof content !== 'object') return 0;
  const value = (content as Record<string, unknown>)[key];
  return Array.isArray(value) ? value.length : 0;
}

function latestEnvelope(artifacts: ArtifactEnvelope[], kind: string) {
  return [...artifacts].reverse().find((artifact) => artifact.artifact_type === kind);
}

export function EDCProgressPanel({
  artifacts,
  onArtifact,
}: {
  artifacts: ArtifactEnvelope[];
  onArtifact: (id: string) => void;
}) {
  const open = latestEnvelope(artifacts, 'edc_open_triples');
  const definitions = latestEnvelope(artifacts, 'edc_relation_definitions');
  const canonicalization = latestEnvelope(artifacts, 'edc_canonicalization');
  const canonicalizationAttempt = latestEnvelope(artifacts, 'edc_canonicalization_attempt');
  const ontology = latestEnvelope(artifacts, 'ontology_snapshot');
  const candidate = latestEnvelope(artifacts, 'candidate_canonical_triples');

  const rows = [
    { label: 'Open triples', artifact: open, detail: `${count(open?.content, 'triples')} extracted` },
    { label: 'Relation definitions', artifact: definitions, detail: `${count(definitions?.content, 'definitions')} defined` },
    { label: 'Ontology snapshot', artifact: ontology, detail: ontology ? 'materialized' : 'not available' },
    {
      label: 'Canonicalization',
      artifact: canonicalization ?? canonicalizationAttempt,
      detail: canonicalization
        ? `${count(canonicalization.content, 'candidates')} candidates · ${count(canonicalization.content, 'unresolved_open_triple_ids')} unresolved`
        : canonicalizationAttempt
          ? 'blocked before a structured decision'
          : 'not attempted',
    },
    { label: 'Candidate triples', artifact: candidate, detail: candidate ? 'ready for verification' : 'not available' },
  ];

  const unresolved = canonicalization && typeof canonicalization.content === 'object'
    ? ((canonicalization.content as Record<string, unknown>).unresolved_open_triple_ids as unknown[] | undefined) ?? []
    : [];

  return (
    <div className="edcProgress">
      <div className="paneTitle">
        <div>
          <b>EDC progress</b>
          <span>Immutable intermediate artifacts; not accepted scientific output.</span>
        </div>
      </div>

      <div className="edcProgressRows">
        {rows.map((row) => (
          <button
            key={row.label}
            className={`edcProgressRow ${row.artifact ? 'available' : 'missing'}`}
            disabled={!row.artifact}
            onClick={() => row.artifact && onArtifact(row.artifact.artifact_id)}
          >
            {row.artifact ? <Check size={15} /> : <X size={15} />}
            <span>
              <b>{row.label}</b>
              <small>{row.detail}</small>
            </span>
            {row.artifact && <code>{row.artifact.artifact_id}</code>}
          </button>
        ))}
      </div>

      {unresolved.length > 0 && (
        <div className="edcUnresolved">
          <b>{unresolved.length} unresolved open triple{unresolved.length === 1 ? '' : 's'}</b>
          <span>{unresolved.slice(0, 12).join(' · ')}</span>
        </div>
      )}
    </div>
  );
}
