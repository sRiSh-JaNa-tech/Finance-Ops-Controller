"""Investigation Agent Stopping Criteria and Convergence Policy."""

from typing import Dict, List, Optional, Tuple
from finance_ops.core.models import DecisionLabel, ReasonCode, AgentRecommendation, CanonicalTransaction
from finance_ops.evidence.bundle import EvidenceBundle
from finance_ops.decision.calibration import AsymmetricDecisionPolicy


class InvestigationStoppingPolicy:
    """Evaluates whether the agent has collected conclusive evidence or should terminate."""

    def __init__(self, max_tool_budget: int = 5):
        self.max_tool_budget = max_tool_budget
        self.policy = AsymmetricDecisionPolicy(cost_false_match=500.0, cost_human_review=10.0, cost_missed_match=50.0)

    def evaluate_state(
        self,
        bundle: EvidenceBundle,
        tool_call_count: int,
        tested_hypotheses: List[Dict]
    ) -> Tuple[bool, Optional[DecisionLabel], Optional[ReasonCode], float]:
        """
        Evaluates current investigation state.
        Returns: (should_stop, decision, reason_code, confidence)
        """
        # 1. Check budget exhaustion
        if tool_call_count >= self.max_tool_budget:
            return True, DecisionLabel.UNCERTAIN, ReasonCode.INVESTIGATION_BUDGET_EXHAUSTED, 0.50

        # 2. Check zero candidates (Missing record)
        if not bundle.candidate_transactions:
            return True, DecisionLabel.UNCERTAIN, ReasonCode.MISSING_SOURCE_RECORD, 0.40

        # 3. Check for refund / reversal exceptions
        if bundle.source_transaction.is_refund or bundle.source_transaction.is_reversal:
            return True, DecisionLabel.EXCEPTION, ReasonCode.INVALID_REVERSAL, 0.90

        # 4. Check for matching invoice reference candidates
        matching_ref_candidates = [
            c for c in bundle.candidate_transactions 
            if bundle.source_transaction.invoice_reference and c.invoice_reference and 
            bundle.source_transaction.invoice_reference.upper() == c.invoice_reference.upper()
        ]
        if len(matching_ref_candidates) == 1:
            cand = matching_ref_candidates[0]
            date_diff_days = abs((bundle.source_transaction.transaction_timestamp - cand.transaction_timestamp).total_seconds()) / 86400.0
            if date_diff_days > 4.0:
                return True, DecisionLabel.MATCHED, ReasonCode.TIMING_ALIGNED_MATCH, 0.96
            return True, DecisionLabel.MATCHED, ReasonCode.EXACT_IDENTIFIER_MATCH, 0.98
        elif len(matching_ref_candidates) >= 2:
            return True, DecisionLabel.EXCEPTION, ReasonCode.DUPLICATE_TRANSACTION, 0.95

        # 5. Check for confirmed hypothesis
        for hyp in tested_hypotheses:
            if hyp.get("status") == "HYPOTHESIS_CONFIRMED":
                htype = hyp.get("hypothesis")
                if htype == "SPLIT_PAYMENT":
                    return True, DecisionLabel.MATCHED, ReasonCode.SPLIT_PAYMENT_MATCH, 0.95
                elif htype == "FEE_ADJUSTMENT":
                    return True, DecisionLabel.MATCHED, ReasonCode.FEE_ADJUSTED_MATCH, 0.95

        # 6. Check for ambiguous candidates (same amount, unlinked references)
        if len(bundle.candidate_transactions) >= 2:
            c1, c2 = bundle.candidate_transactions[0], bundle.candidate_transactions[1]
            if c1.amount == c2.amount == bundle.source_transaction.amount:
                return True, DecisionLabel.UNCERTAIN, ReasonCode.AMBIGUOUS_CANDIDATES, 0.50

        # 7. Check for hard contradictions
        if bundle.detected_contradictions:
            return True, DecisionLabel.EXCEPTION, ReasonCode.UNRESOLVED_CONTRADICTION, 0.90

        # Inconclusive - continue investigating if budget remains
        return False, None, None, 0.0
