"""Deterministic Candidate Reranker & Evidence Packet Builder.

Implements research-backed retrieve-rank-reason architecture:
1. Multi-feature deterministic candidate scoring (Amount, Date, Invoice/UTR, Merchant).
2. Top-K filtering (K=3) to prevent context bloat.
3. Structured Evidence Packet generation to replace multi-round search loops with single-turn reasoning.
"""

from typing import List, Dict, Any, Optional, Tuple
from decimal import Decimal
import math
from datetime import datetime

from finance_ops.core.models import CanonicalTransaction


class DeterministicCandidateReranker:
    """Reranks candidate transactions using weighted lexical, numeric, and temporal similarity."""

    def __init__(
        self,
        weight_amount: float = 0.35,
        weight_date: float = 0.25,
        weight_reference: float = 0.25,
        weight_merchant: float = 0.15,
        top_k: int = 3
    ):
        self.w_amount = weight_amount
        self.w_date = weight_date
        self.w_ref = weight_reference
        self.w_merchant = weight_merchant
        self.top_k = top_k

    def calculate_amount_similarity(self, src: CanonicalTransaction, cand: CanonicalTransaction) -> float:
        """Calculates normalized amount proximity with tolerance for minor fees/tax."""
        diff = abs(src.amount - cand.amount)
        if diff == Decimal("0.00"):
            return 1.0
        # Check standard MDR fee band (1.5% to 3.0%)
        ratio = float(cand.amount / src.amount) if src.amount > 0 else 0.0
        if 0.95 <= ratio <= 1.05:
            return float(Decimal("1.0") - (diff / max(src.amount, Decimal("1.00"))))
        # Exponential decay for larger discrepancies
        return max(0.0, 1.0 - (float(diff) / max(float(src.amount), 1.0)))

    def calculate_date_similarity(self, src: CanonicalTransaction, cand: CanonicalTransaction) -> float:
        """Calculates temporal proximity using exponential decay over settlement windows."""
        diff_seconds = abs((src.transaction_timestamp - cand.transaction_timestamp).total_seconds())
        diff_days = diff_seconds / 86400.0
        if diff_days <= 0.25:  # Within 6 hours
            return 1.0
        elif diff_days <= 2.0:  # T+1 / T+2 standard settlement
            return 0.90
        elif diff_days <= 7.0:  # Within 1 week
            return max(0.20, 1.0 - (diff_days / 10.0))
        return 0.0

    def calculate_reference_similarity(self, src: CanonicalTransaction, cand: CanonicalTransaction) -> float:
        """Calculates exact and token-level identifier overlap (UTR, Invoice, Order)."""
        score = 0.0
        # UTR Exact Match
        if src.utr and cand.utr and src.utr == cand.utr:
            return 1.0
        
        # Invoice Reference Match
        if src.invoice_reference and cand.invoice_reference:
            s_inv = src.invoice_reference.upper().strip()
            c_inv = cand.invoice_reference.upper().strip()
            if s_inv == c_inv:
                return 1.0
            if s_inv in c_inv or c_inv in s_inv:
                return 0.85
                
        # Order ID Match
        if src.order_id and cand.order_id:
            if src.order_id == cand.order_id:
                return 0.95
                
        # Narrative substring token overlap
        if src.raw_narrative and cand.raw_narrative:
            s_tokens = set(src.raw_narrative.upper().split())
            c_tokens = set(cand.raw_narrative.upper().split())
            if s_tokens and c_tokens:
                jaccard = len(s_tokens.intersection(c_tokens)) / len(s_tokens.union(c_tokens))
                score = max(score, jaccard)
                
        return score

    def calculate_merchant_similarity(self, src: CanonicalTransaction, cand: CanonicalTransaction) -> float:
        """Calculates entity name/VPA overlap."""
        if src.merchant_id and cand.merchant_id and src.merchant_id == cand.merchant_id:
            return 1.0
        if src.merchant_name_norm and cand.merchant_name_norm:
            if src.merchant_name_norm == cand.merchant_name_norm:
                return 1.0
            if src.merchant_name_norm in cand.merchant_name_norm or cand.merchant_name_norm in src.merchant_name_norm:
                return 0.80
        return 0.0

    def score_candidate(self, src: CanonicalTransaction, cand: CanonicalTransaction) -> Dict[str, Any]:
        """Scores a single candidate across all 4 feature dimensions."""
        amt_sim = self.calculate_amount_similarity(src, cand)
        date_sim = self.calculate_date_similarity(src, cand)
        ref_sim = self.calculate_reference_similarity(src, cand)
        merch_sim = self.calculate_merchant_similarity(src, cand)

        total_score = (
            self.w_amount * amt_sim +
            self.w_date * date_sim +
            self.w_ref * ref_sim +
            self.w_merchant * merch_sim
        )

        return {
            "candidate": cand,
            "candidate_id": cand.transaction_id,
            "composite_score": round(total_score, 4),
            "amount_similarity": round(amt_sim, 4),
            "date_similarity": round(date_sim, 4),
            "reference_similarity": round(ref_sim, 4),
            "merchant_similarity": round(merch_sim, 4),
            "amount_difference": float(abs(src.amount - cand.amount)),
            "days_difference": round(abs((src.transaction_timestamp - cand.transaction_timestamp).total_seconds()) / 86400.0, 2)
        }

    def rerank(self, src: CanonicalTransaction, candidates: List[CanonicalTransaction]) -> List[Dict[str, Any]]:
        """Reranks candidate list and returns top-K with full feature breakdown."""
        if not candidates:
            return []
        
        scored = [self.score_candidate(src, c) for c in candidates]
        scored.sort(key=lambda x: x["composite_score"], reverse=True)
        return scored[:self.top_k]


class EvidencePacketBuilder:
    """Constructs a structured evidence packet for single-turn LLM reasoning."""

    @staticmethod
    def build_packet(src: CanonicalTransaction, ranked_candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Creates a standardized evidence packet containing source and ranked candidates."""
        candidates_summary = []
        for idx, item in enumerate(ranked_candidates):
            cand: CanonicalTransaction = item["candidate"]
            candidates_summary.append({
                "rank": idx + 1,
                "candidate_id": cand.transaction_id,
                "source_system": cand.source_system.value,
                "amount": float(cand.amount),
                "timestamp": cand.transaction_timestamp.isoformat(),
                "utr": cand.utr,
                "invoice_reference": cand.invoice_reference,
                "merchant_name": cand.merchant_name,
                "payment_method": cand.payment_method.value if cand.payment_method else None,
                "narrative": cand.raw_narrative,
                "feature_scores": {
                    "composite_score": item["composite_score"],
                    "amount_similarity": item["amount_similarity"],
                    "date_similarity": item["date_similarity"],
                    "reference_similarity": item["reference_similarity"],
                    "merchant_similarity": item["merchant_similarity"],
                    "amount_delta": item["amount_difference"],
                    "days_delta": item["days_difference"]
                }
            })

        return {
            "source_transaction": {
                "transaction_id": src.transaction_id,
                "source_system": src.source_system.value,
                "amount": float(src.amount),
                "timestamp": src.transaction_timestamp.isoformat(),
                "utr": src.utr,
                "invoice_reference": src.invoice_reference,
                "order_id": src.order_id,
                "merchant_name": src.merchant_name,
                "payment_method": src.payment_method.value if src.payment_method else None,
                "narrative": src.raw_narrative
            },
            "candidate_count": len(ranked_candidates),
            "top_candidates": candidates_summary
        }
