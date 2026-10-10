from __future__ import annotations

import re

from app.models.domain import CanonicalTriple, EvaluationResult, GoldAnnotation, RejectedTriple


class EvaluationService:
    def evaluate(
        self,
        triples: list[CanonicalTriple],
        *,
        shacl_conforms: bool,
        gold: GoldAnnotation | None = None,
        rejected_count: int = 0,
        tool_actions: int = 0,
        subagents_spawned: int = 0,
        retries: int = 0,
        model_escalations: int = 0,
    ) -> EvaluationResult:
        result = EvaluationResult(
            provenance_coverage=(sum(bool(t.evidence_ids) for t in triples) / len(triples)) if triples else 0.0,
            supported_ratio=(sum(t.verification_status == "accepted" for t in triples) / len(triples)) if triples else 0.0,
            shacl_conformance=shacl_conforms,
            tool_actions=tool_actions,
            subagents_spawned=subagents_spawned,
            retries=retries,
            model_escalations=model_escalations,
        )
        if not gold or not gold.triples:
            result.notes.append("No gold triples supplied; extraction P/R/F1 not computed.")
            return result
        pred_exact = {(t.subject, t.predicate, t.object) for t in triples}
        gold_exact = {(t.subject, t.predicate, t.object) for t in gold.triples}
        ep, er, ef = self._prf(pred_exact, gold_exact)
        pred_norm = {tuple(self._norm(x) for x in row) for row in pred_exact}
        gold_norm = {tuple(self._norm(x) for x in row) for row in gold_exact}
        np, nr, nf = self._prf(pred_norm, gold_norm)
        subject_accuracy = self._component_accuracy(pred_norm, gold_norm, 0)
        predicate_accuracy = self._component_accuracy(pred_norm, gold_norm, 1)
        object_accuracy = self._component_accuracy(pred_norm, gold_norm, 2)
        return result.model_copy(update={
            "exact_precision": ep, "exact_recall": er, "exact_f1": ef,
            "normalized_precision": np, "normalized_recall": nr, "normalized_f1": nf,
            "subject_accuracy": subject_accuracy, "predicate_accuracy": predicate_accuracy, "object_accuracy": object_accuracy,
        })

    @staticmethod
    def _component_accuracy(pred: set[tuple[str, str, str]], gold: set[tuple[str, str, str]], idx: int) -> float:
        if not gold:
            return 0.0
        return len({x[idx] for x in pred} & {x[idx] for x in gold}) / len({x[idx] for x in gold})

    @staticmethod
    def _prf(pred: set, gold: set) -> tuple[float, float, float]:
        tp = len(pred & gold)
        p = tp / len(pred) if pred else 0.0
        r = tp / len(gold) if gold else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return p, r, f

    @staticmethod
    def _norm(text: str) -> str:
        return " ".join(re.findall(r"[a-z0-9]+", text.lower()))
