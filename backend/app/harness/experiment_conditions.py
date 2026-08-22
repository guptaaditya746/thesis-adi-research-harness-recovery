"""Dormant definitions for controlled Director-navigation ablations.

They are deliberately not imported by baseline runtime execution.
"""

from __future__ import annotations

from typing import Any


SEMANTIC_CAPABILITY_MANIFEST = {
    "documents": "retrieve, inspect, and persist source-grounded evidence",
    "ontology": "inspect DBLP terms and build a run-local ontology snapshot",
    "edc": "extract open triples, define relations, and canonicalize against an ontology snapshot",
    "verification": "independently assess canonical triple candidates",
}

TERMINAL_OBJECTIVE_CLARIFICATION = (
    "A prose finding is not a completed research result. Completion means accepted canonical triples "
    "and the answer rendered from that immutable artifact."
)


def factual_scientific_state_reminder(state: dict[str, Any]) -> str:
    """Render facts for a future state-aware condition without recommending an action."""
    artifacts = state.get("artifacts", {})
    return "; ".join(
        f"{kind}={entry.get('count', 0)}"
        for kind, entry in sorted(artifacts.items())
    )
