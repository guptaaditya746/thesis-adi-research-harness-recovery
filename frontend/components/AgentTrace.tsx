'use client';

import { useState } from 'react';
import {
  Bot,
  Wrench,
  GitBranch,
  FileOutput,
  Search,
  ShieldCheck,
  Database,
  MessageSquare,
  RotateCcw,
  BrainCircuit,
} from 'lucide-react';
import type { TraceEvent } from '@/lib/types';

const icon = (e: TraceEvent) =>
  e.type === 'delegate' ? (
    <GitBranch size={15} />
  ) : e.type === 'tool' ? (
    <Wrench size={15} />
  ) : e.type === 'artifact' ? (
    <FileOutput size={15} />
  ) : e.type === 'evidence' ? (
    <Search size={15} />
  ) : e.type === 'validation' ? (
    <ShieldCheck size={15} />
  ) : e.type === 'triple' ? (
    <Database size={15} />
  ) : e.type === 'message' ? (
    <MessageSquare size={15} />
  ) : e.type === 'recovery' ? (
    <RotateCcw size={15} />
  ) : e.type === 'decision' || e.type === 'model' ? (
    <BrainCircuit size={15} />
  ) : (
    <Bot size={15} />
  );

export function AgentTrace({
  events,
  onArtifact,
}: {
  events: TraceEvent[];
  onArtifact: (id: string) => void;
}) {
  const [expanded, setExpanded] = useState<string>();

  return (
    <div className="trace">
      <div className="traceHead">
        <b>Live research trace</b>
        <span>{events.length} observable actions</span>
      </div>

      <div className="traceList">
        {events.length === 0 && (
          <div className="traceEmpty">
            Ask a question. Director decisions, generated specialist tasks,
            model/tool boundaries, retries, recovery and artifacts will appear
            here. Hidden chain-of-thought is intentionally excluded.
          </div>
        )}

        {events.map(e => (
          <div
            key={e.id}
            className={`traceItem ${e.status}`}
            role="button"
            tabIndex={0}
            onClick={() =>
              setExpanded(expanded === e.id ? undefined : e.id)
            }
          >
            <div className="traceIcon">{icon(e)}</div>

            <div>
              <div className="traceMeta">
                <b>{e.actor}</b>
                <span>{e.action}</span>
              </div>

              <strong>{e.title}</strong>

              {e.summary && <p>{e.summary}</p>}

              <small>{new Date(e.timestamp).toLocaleTimeString()}</small>

              {e.artifact_id && (
                <button
                  className="inlineArtifact"
                  onClick={ev => {
                    ev.stopPropagation();
                    onArtifact(e.artifact_id!);
                  }}
                >
                  open artifact
                </button>
              )}

              {expanded === e.id &&
                Object.keys(e.payload ?? {}).length > 0 && (
                  <pre className="tracePayload">
                    {JSON.stringify(e.payload, null, 2)}
                  </pre>
                )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}