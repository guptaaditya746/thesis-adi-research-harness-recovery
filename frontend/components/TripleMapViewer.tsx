'use client';

import { useEffect, useMemo } from 'react';
import { MultiDirectedGraph } from 'graphology';
import { SigmaContainer, useLoadGraph } from '@react-sigma/core';
import '@react-sigma/core/lib/style.css';
import type { TripleMapSnapshot } from '@/lib/types';

function Loader({ snapshot }: { snapshot: TripleMapSnapshot }) {
  const load = useLoadGraph();

  const graph = useMemo(() => {
    const g = new MultiDirectedGraph();

    snapshot.nodes.forEach((n, i) =>
      g.addNode(n.id, {
        label: n.label ?? n.id,
        size: 8 + (i % 5),
        x: Math.cos(i) * 10 + (i % 7),
        y: Math.sin(i) * 10 + (i % 11),
      }),
    );

    snapshot.edges.forEach(e => {
      if (g.hasNode(e.source) && g.hasNode(e.target)) {
        g.addEdgeWithKey(e.id, e.source, e.target, {
          label: e.label ?? e.predicate ?? '',
          size: 1.5,
        });
      }
    });

    return g;
  }, [snapshot]);

  useEffect(() => {
    load(graph);
  }, [graph, load]);

  return null;
}

export function TripleMapViewer({
  snapshot,
}: {
  snapshot: TripleMapSnapshot;
}) {
  if (!snapshot.nodes.length) {
    return (
      <div className="emptyState">
        No accepted triples are available for this run. This view is derived on
        demand and is never persisted as an instance graph. Check EDC progress
        for the precise stage that has not completed yet.
      </div>
    );
  }

  return (
    <SigmaContainer
      graph={MultiDirectedGraph}
      className="sigma"
      settings={{ renderEdgeLabels: true, labelDensity: 0.08 }}
    >
      <Loader snapshot={snapshot} />
    </SigmaContainer>
  );
}
