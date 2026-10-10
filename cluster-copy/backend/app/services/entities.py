from __future__ import annotations

import re
from difflib import SequenceMatcher

from app.models.domain import EntityNormalizationDecision, EntityResolutionCandidate


class EntityNormalizationService:
    """Separate relation-independent entity normalization.

    Base implementation is deterministic and local. DOI/ORCID/DBLP/Crossref identifiers can
    be supplied from DocumentIR or future adapters; no external lookup is a runtime dependency.
    """
    def normalize(
        self,
        mention: str,
        *,
        aliases: dict[str, str] | None = None,
        known_entities: list[str] | None = None,
        external_ids: dict[str, str] | None = None,
    ) -> EntityNormalizationDecision:
        aliases = aliases or {}
        known_entities = known_entities or []
        norm = self._norm(mention)
        alias_map = {self._norm(k): v for k, v in aliases.items()}
        if norm in alias_map:
            canonical = alias_map[norm]
            return EntityNormalizationDecision(
                original_mention=mention, canonical_name=canonical, external_ids=external_ids or {},
                aliases=[k for k, v in aliases.items() if v == canonical], method="normalized_alias_exact",
            )
        exact = next((x for x in known_entities if self._norm(x) == norm), None)
        if exact:
            return EntityNormalizationDecision(
                original_mention=mention, canonical_name=exact, external_ids=external_ids or {}, method="normalized_exact"
            )
        candidates = sorted(
            [EntityResolutionCandidate(canonical_name=x, score=SequenceMatcher(None, norm, self._norm(x)).ratio(), source="local") for x in known_entities],
            key=lambda x: -x.score,
        )[:5]
        canonical = candidates[0].canonical_name if candidates and candidates[0].score >= 0.92 else mention.strip()
        return EntityNormalizationDecision(
            original_mention=mention, canonical_name=canonical, external_ids=external_ids or {},
            candidates=candidates, method="fuzzy_local" if canonical != mention.strip() else "preserve_mention",
        )

    @staticmethod
    def _norm(text: str) -> str:
        return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))
