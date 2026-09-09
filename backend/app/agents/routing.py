"""Centralized, host-enforced model routing for bounded research tasks.

The Research Director still decides *which* task to delegate.  This module only
selects the model profile for task scopes whose computational character is known
to the host, keeping routing policy out of prompts and individual services.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.models.domain import ModelProfile, SpecialistTaskScope


@dataclass(frozen=True)
class TaskModelRoute:
    requested_profile: ModelProfile
    effective_profile: ModelProfile
    task_scope: SpecialistTaskScope | str
    reason: str


class TaskModelRouter:
    """Resolve the smallest suitable profile for known scientific task scopes."""

    _SPECIALIST_PROFILES: dict[SpecialistTaskScope, ModelProfile] = {
        "evidence_selection": ModelProfile.FAST,
        "document_analysis": ModelProfile.FAST,
        "ontology_discovery": ModelProfile.FAST,
        "ontology_snapshot": ModelProfile.BALANCED,
        "edc_extract": ModelProfile.EXTRACTOR,
        "edc_define": ModelProfile.REASONING,
        # Canonicalization is a tiny, structured semantic choice over a
        # deterministic shortlist; use the low-latency LiteLLM alias.
        "edc_canonicalize": ModelProfile.FAST,
        "verification": ModelProfile.CRITIC,
        "critique": ModelProfile.CRITIC,
    }

    _SERVICE_PROFILES: dict[str, ModelProfile] = {
        "edc_extract": ModelProfile.EXTRACTOR,
        "edc_define": ModelProfile.REASONING,
        "edc_canonicalize": ModelProfile.FAST,
        "verification": ModelProfile.CRITIC,
    }

    def route_specialist(self, task_scope: SpecialistTaskScope, requested_profile: ModelProfile) -> TaskModelRoute:
        effective = self._SPECIALIST_PROFILES.get(task_scope, requested_profile)
        reason = (
            f"Host policy routes task_scope={task_scope} to {effective.value}."
            if effective != requested_profile
            else f"No stricter host route exists for task_scope={task_scope}; retained {effective.value}."
        )
        return TaskModelRoute(requested_profile, effective, task_scope, reason)

    def profile_for_service(self, operation: str, fallback: ModelProfile) -> ModelProfile:
        """Resolve model ownership for internal EDC/verification model calls."""
        return self._SERVICE_PROFILES.get(operation, fallback)
