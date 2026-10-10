"""Optional durability adapters.

Only one backend should be enabled per deployment. StepPersistence is the audit/continuation substrate;
DBOS/Prefect/Temporal are optional execution engines for runs that must survive process restarts.
"""
from app.config import Settings


def describe_backend(settings: Settings) -> dict:
    backend = settings.durability_backend.lower()
    return {
        "backend": backend,
        "supported": backend in {"none", "dbos", "prefect", "temporal"},
        "note": "Attach the matching PydanticAI durability capability at deployment time; ordinary local runs remain plain agents." if backend != "none" else "Local process execution with Harness StepPersistence.",
    }
