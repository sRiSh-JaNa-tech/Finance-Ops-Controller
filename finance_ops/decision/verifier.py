"""Deterministic Policy Verifier for Hard Accounting Invariants in Prototype 3.

Enforces zero-hallucination policies, integer paise amount conservation, and
mandatory passing rule audit requirements.
"""

import uuid
from typing import Dict, List, Tuple, Optional
from decimal import Decimal

from finance_ops.core.models import (
    AgentRecommendation, FinalDecisionRecord, DecisionLabel, ReasonCode, CanonicalTransaction
)
from finance_ops.ingestion.storage import FinancialDataRepository
from finance_ops.rules.engine import DeterministicRuleEngine


class DeterministicPolicyVerifier:
    """
    Authoritative verifier that inspects agent recommendations against hard financial invariants.
    Cannot be overridden by language model generation.
    """

    def __init__(
        self,
        repository: FinancialDataRepository,
        rule_engine: Optional[DeterministicRuleEngine] = None
    ):
        self.repo = repository
        self.rule_engine = rule_engine or DeterministicRuleEngine()

    def verify_and_finalize(
        self,
        recommendation: AgentRecommendation,
        source_tx: CanonicalTransaction,
        calibrated_confidence: float,
        auto_match_threshold: float = 0.85
    ) -> FinalDecisionRecord:
        """
        Verifies recommendation against hard accounting rules and produces final audited record.
        """
        decision = recommendation.recommended_decision
        reason = recommendation.primary_reason
        verifier_notes = []
        is_verified = True

        # Rule 1: MATCHED requires existing target records
        if decision == DecisionLabel.MATCHED:
            if not recommendation.matched_record_ids:
                decision = DecisionLabel.UNCERTAIN
                reason = ReasonCode.INSUFFICIENT_PROVENANCE
                verifier_notes.append("Veto: MATCHED declared without target record IDs")
                is_verified = False
            else:
                matched_txs = [
                    self.repo.get_transaction(rid)
                    for rid in recommendation.matched_record_ids
                    if self.repo.get_transaction(rid) and rid != source_tx.transaction_id
                ]

                if not matched_txs:
                    matched_txs = [
                        self.repo.get_transaction(rid)
                        for rid in recommendation.matched_record_ids
                        if self.repo.get_transaction(rid)
                    ]

                if not matched_txs:
                    decision = DecisionLabel.UNCERTAIN
                    reason = ReasonCode.MISSING_SOURCE_RECORD
                    verifier_notes.append("Veto: Target records missing from repository")
                    is_verified = False
                else:
                    # Run deterministic rule engine verification
                    report = self.rule_engine.evaluate_pair(source_tx, matched_txs)
                    
                    has_passing_ac = any(r in report.passed_rules for r in ["AC-1", "AC-2", "AC-3", "AC-4", "AC-5"])
                    has_hard_fail = (any(r in report.failed_rules for r in ["AC-1", "AC-4", "AB-1"]) and not has_passing_ac) or (len(matched_txs) == 1 and abs(source_tx.amount - matched_txs[0].amount) > Decimal("0.50") and not has_passing_ac)

                    if has_hard_fail:
                        decision = DecisionLabel.EXCEPTION
                        reason = ReasonCode.AMOUNT_MISMATCH
                        verifier_notes.append(f"Veto: Hard rule failure in evaluation ({report.failed_rules})")
                        is_verified = False
                    elif not has_passing_ac:
                        decision = DecisionLabel.UNCERTAIN
                        reason = ReasonCode.UNRESOLVED_CONTRADICTION
                        verifier_notes.append("Veto: No Amount Conservation (AC) rule passed")
                        is_verified = False

        # Rule 2: Risk-calibrated threshold check
        is_automated = False
        requires_human = False

        # Phase 3: Context-Adaptive Risk Thresholds (Instance-Specific Abstention)
        if reason == ReasonCode.EXACT_IDENTIFIER_MATCH:
            dynamic_threshold = max(0.90, auto_match_threshold)
        elif reason in [ReasonCode.FEE_ADJUSTED_MATCH, ReasonCode.SPLIT_PAYMENT_MATCH]:
            dynamic_threshold = max(0.95, auto_match_threshold)
        elif reason == ReasonCode.FUZZY_ENTITY_MATCH:
            dynamic_threshold = 0.99 # Fuzzy matches need high certainty
        else:
            dynamic_threshold = max(0.92, auto_match_threshold)

        if decision == DecisionLabel.MATCHED:
            if calibrated_confidence >= dynamic_threshold and is_verified:
                is_automated = True
                requires_human = False
            else:
                decision = DecisionLabel.UNCERTAIN
                reason = ReasonCode.BELOW_CONFIDENCE_THRESHOLD
                is_automated = False
                requires_human = True
                is_verified = False
                verifier_notes.append(f"Escalated to Human Review: confidence {calibrated_confidence:.2f} < context threshold {dynamic_threshold:.2f}")
        elif decision == DecisionLabel.EXCEPTION:
            is_automated = True
            requires_human = False
        else:  # UNCERTAIN
            is_automated = False
            requires_human = True

        status = "VERIFIED_VALID" if is_verified else "VETOED_DOWNGRADED"

        return FinalDecisionRecord(
            decision_id=f"DEC_{uuid.uuid4().hex[:10]}",
            case_id=recommendation.case_id,
            decision=decision,
            reason=reason,
            calibrated_confidence=calibrated_confidence,
            is_automated=is_automated,
            requires_human_review=requires_human,
            matched_pairs=[{"source": source_tx.transaction_id, "target": tid} for tid in recommendation.matched_record_ids if tid != source_tx.transaction_id],
            source_record_ids=[source_tx.transaction_id],
            cited_evidence_ids=recommendation.cited_evidence_ids,
            rules_passed=recommendation.rules_passed,
            rules_failed=recommendation.rules_failed,
            rules_warned=recommendation.rules_warned,
            leakage_risk=recommendation.leakage_risk,
            tool_calls_count=recommendation.tool_calls_performed,
            verifier_status=status,
            verifier_notes=verifier_notes,
            explanation=recommendation.explanation_narrative
        )
