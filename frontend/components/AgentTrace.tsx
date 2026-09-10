'use client';

import { useMemo, useState } from 'react';
import {
  Bot,
  BrainCircuit,
  ChevronDown,
  ChevronRight,
  Database,
  FileOutput,
  GitBranch,
  MessageSquare,
  RotateCcw,
  Search,
  ShieldCheck,
  Wrench,
} from 'lucide-react';
import type { TraceEvent } from '@/lib/types';

const icon = (e: TraceEvent) =>
  e.type === 'delegate' ? <GitBranch size={15} />
  : e.type === 'tool' ? <Wrench size={15} />
  : e.type === 'artifact' ? <FileOutput size={15} />
  : e.type === 'evidence' ? <Search size={15} />
  : e.type === 'validation' ? <ShieldCheck size={15} />
  : e.type === 'triple' ? <Database size={15} />
  : e.type === 'message' ? <MessageSquare size={15} />
  : e.type === 'recovery' ? <RotateCcw size={15} />
  : e.type === 'decision' || e.type === 'model' ? <BrainCircuit size={15} />
  : <Bot size={15} />;

type TraceTreeEvent = TraceEvent & { children: TraceTreeEvent[] };

function TraceNode({
  event,
  children,
  depth,
  collapsed,
  expanded,
  onToggleBranch,
  onTogglePayload,
  onArtifact,
}: {
  event: TraceTreeEvent;
  children: TraceTreeEvent[];
  depth: number;
  collapsed: Set<string>;
  expanded: string | undefined;
  onToggleBranch: (id: string) => void;
  onTogglePayload: (id: string) => void;
  onArtifact: (id: string) => void;
}) {
  const hasChildren = children.length > 0;
  const isCollapsed = collapsed.has(event.id);

  return (
    <div className="traceTreeNode">
      <div
        className={`traceItem ${event.status}`}
        style={{ marginInlineStart: `${depth * 16}px` }}
        role="treeitem"
        aria-expanded={hasChildren ? !isCollapsed : undefined}
        tabIndex={0}
        onClick={() => onTogglePayload(event.id)}
      >
        <div className="traceIcon">{icon(event)}</div>
        <div>
          <div className="traceMeta">
            {hasChildren ? (
              <button
                className="traceBranch"
                aria-label={isCollapsed ? 'Expand trace branch' : 'Collapse trace branch'}
                onClick={(click) => {
                  click.stopPropagation();
                  onToggleBranch(event.id);
                }}
              >
                {isCollapsed ? <ChevronRight size={13} /> : <ChevronDown size={13} />}
              </button>
            ) : <span className="traceBranchSpacer" />}
            <b>{event.actor}</b>
            <span>{event.action}</span>
          </div>
          <strong>{event.title}</strong>
          {event.summary && <p>{event.summary}</p>}
          <small>{new Date(event.timestamp).toLocaleTimeString()}</small>
          {event.artifact_id && (
            <button
              className="inlineArtifact"
              onClick={(click) => {
                click.stopPropagation();
                onArtifact(event.artifact_id!);
              }}
            >
              open artifact
            </button>
          )}
          {expanded === event.id && Object.keys(event.payload ?? {}).length > 0 && (
            <pre className="tracePayload">{JSON.stringify(event.payload, null, 2)}</pre>
          )}
        </div>
      </div>
      {hasChildren && !isCollapsed && (
        <div role="group">
          {children.map((child) => (
            <TraceNode
              key={child.id}
              event={child}
              children={child.children}
              depth={depth + 1}
              collapsed={collapsed}
              expanded={expanded}
              onToggleBranch={onToggleBranch}
              onTogglePayload={onTogglePayload}
              onArtifact={onArtifact}
            />
          ))}
        </div>
      )}
    </div>
  );
}

export function AgentTrace({ events, onArtifact }: { events: TraceEvent[]; onArtifact: (id: string) => void }) {
  const [expanded, setExpanded] = useState<string>();
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());

  const roots = useMemo(() => {
    const byId = new Map(events.map((event) => [event.id, { ...event, children: [] as TraceTreeEvent[] }]));
    const rootEvents: TraceTreeEvent[] = [];
    for (const event of events) {
      const node = byId.get(event.id)!;
      const parent = event.parent_event_id ? byId.get(event.parent_event_id) : undefined;
      if (parent) parent.children.push(node);
      else rootEvents.push(node);
    }
    return rootEvents;
  }, [events]);

  return (
    <div className="trace">
      <div className="traceHead"><b>Live research trace</b><span className="traceActionCount"><strong>{events.length}</strong><em>observable actions</em></span></div>
      <div className="traceList" role="tree">
        {events.length === 0 && <div className="traceEmpty">Ask a question. Director decisions, generated specialist tasks, model/tool boundaries, retries, recovery and artifacts will appear here. Hidden chain-of-thought is intentionally excluded.</div>}
        {roots.map((event) => (
          <TraceNode
            key={event.id}
            event={event}
            children={event.children}
            depth={0}
            collapsed={collapsed}
            expanded={expanded}
            onToggleBranch={(id) => setCollapsed((old) => {
              const next = new Set(old);
              next.has(id) ? next.delete(id) : next.add(id);
              return next;
            })}
            onTogglePayload={(id) => setExpanded((old) => old === id ? undefined : id)}
            onArtifact={onArtifact}
          />
        ))}
      </div>
    </div>
  );
}
