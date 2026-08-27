"""Comparative Baseline Systems for Benchmark Evaluation."""

from typing import List, Dict, Any, Optional
from decimal import Decimal
from datetime import datetime
from finance_ops.core.models import (
    CanonicalTransaction, DecisionLabel, ReasonCode, FinalDecisionRecord
)
from finance_ops.retrieval.similarity import calculate_candidate_similarity, calculate_lexical_similarity


class ExactIdentifierMatcher:
    """Baseline 1: Exact identifier lookup."""
    def match(self, source_tx: CanonicalTransaction, candidates: List[CanonicalTransaction]) -> Dict[str, Any]:
        if not source_tx.invoice_reference:
            return {"decision": DecisionLabel.UNCERTAIN, "reason": ReasonCode.MISSING_SOURCE_RECORD, "matched_id": None}
        
        for c in candidates:
            if c.invoice_reference and c.invoice_reference.upper() == source_tx.invoice_reference.upper():
                return {"decision": DecisionLabel.MATCHED, "reason": ReasonCode.EXACT_IDENTIFIER_MATCH, "matched_id": c.transaction_id}
        return {"decision": DecisionLabel.UNCERTAIN, "reason": ReasonCode.MISSING_SOURCE_RECORD, "matched_id": None}


class DeterministicRuleMatcher:
    """Baseline 2: Strict rule-based matcher (Exact amount & date window)."""
    def match(self, source_tx: CanonicalTransaction, candidates: List[CanonicalTransaction]) -> Dict[str, Any]:
        for c in candidates:
            diff_amt = abs(source_tx.amount - c.amount)
            diff_days = abs((source_tx.transaction_timestamp - c.transaction_timestamp).total_seconds()) / 86400.0
            if diff_amt <= Decimal("0.02") and diff_days <= 7.0:
                return {"decision": DecisionLabel.MATCHED, "reason": ReasonCode.EXACT_IDENTIFIER_MATCH, "matched_id": c.transaction_id}
        return {"decision": DecisionLabel.UNCERTAIN, "reason": ReasonCode.MISSING_SOURCE_RECORD, "matched_id": None}


class FuzzyLexicalMatcher:
    """Baseline 3: Lexical Jaccard thresholding."""
    def match(self, source_tx: CanonicalTransaction, candidates: List[CanonicalTransaction], threshold: float = 0.60) -> Dict[str, Any]:
        best_cand = None
        best_score = 0.0
        for c in candidates:
            sim = calculate_lexical_similarity(source_tx.raw_narrative, c.raw_narrative)
            if sim > best_score:
                best_score = sim
                best_cand = c
        
        if best_cand and best_score >= threshold:
            return {"decision": DecisionLabel.MATCHED, "reason": ReasonCode.FUZZY_ENTITY_MATCH, "matched_id": best_cand.transaction_id}
        return {"decision": DecisionLabel.UNCERTAIN, "reason": ReasonCode.BELOW_CONFIDENCE_THRESHOLD, "matched_id": None}


class Prototype1HybridPipeline:
    """Baseline 8: Prototype-1 static sequential pipeline."""
    def match(self, source_tx: CanonicalTransaction, candidates: List[CanonicalTransaction]) -> Dict[str, Any]:
        if not candidates:
            return {"decision": DecisionLabel.UNCERTAIN, "reason": ReasonCode.MISSING_SOURCE_RECORD, "matched_id": None}

        # Check exact ID first
        for c in candidates:
            if source_tx.invoice_reference and c.invoice_reference and source_tx.invoice_reference.upper() == c.invoice_reference.upper():
                return {"decision": DecisionLabel.MATCHED, "reason": ReasonCode.EXACT_IDENTIFIER_MATCH, "matched_id": c.transaction_id}

        # Check composite score
        best_cand = None
        best_score = 0.0
        for c in candidates:
            sim_dict = calculate_candidate_similarity(source_tx, c)
            score = sim_dict["composite_score"]
            if score > best_score:
                best_score = score
                best_cand = c

        if best_cand and best_score >= 0.75:
            return {"decision": DecisionLabel.MATCHED, "reason": ReasonCode.FUZZY_ENTITY_MATCH, "matched_id": best_cand.transaction_id}
        return {"decision": DecisionLabel.UNCERTAIN, "reason": ReasonCode.BELOW_CONFIDENCE_THRESHOLD, "matched_id": None}
