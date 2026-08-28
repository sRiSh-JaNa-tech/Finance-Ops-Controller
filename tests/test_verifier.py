"""Tests for Phase 8 Policy Verifier, Calibration, and Audit Exporter."""

from decimal import Decimal
from datetime import datetime, timezone
from finance_ops.core.models import (
    CanonicalTransaction, SourceSystem, DecisionLabel, ReasonCode, AgentRecommendation
)
from finance_ops.ingestion.storage import FinancialDataRepository
from finance_ops.decision.verifier import DeterministicPolicyVerifier
from finance_ops.decision.calibration import (
    ConfidenceCalibrator, calculate_brier_score, calculate_expected_calibration_error
)


def test_verifier_arithmetic_mismatch_veto():
    repo = FinancialDataRepository()
    t = datetime(2026, 3, 1, 10, 0, tzinfo=timezone.utc)
    
    src = CanonicalTransaction(transaction_id="TX_SRC", source_system=SourceSystem.GATEWAY, source_record_id="a", amount=Decimal("100.00"), currency="USD", transaction_timestamp=t)
    cand = CanonicalTransaction(transaction_id="TX_CAND", source_system=SourceSystem.BANK, source_record_id="b", amount=Decimal("800.00"), currency="USD", transaction_timestamp=t)
    
    repo.store_canonical_transaction(src)
    repo.store_canonical_transaction(cand)

    rec = AgentRecommendation(
        case_id="C_1",
        recommended_decision=DecisionLabel.MATCHED,
        primary_reason=ReasonCode.EXACT_IDENTIFIER_MATCH,
        matched_record_ids=["TX_CAND"],
        confidence_score=0.95
    )

    verifier = DeterministicPolicyVerifier(repo)
    final = verifier.verify_and_finalize(rec, src, calibrated_confidence=0.95)

    # Verifier MUST veto arithmetic mismatch from MATCHED to EXCEPTION or UNCERTAIN
    assert final.decision != DecisionLabel.MATCHED
    assert final.verifier_status == "VETOED_DOWNGRADED"


def test_calibration_and_ece():
    calibrator = ConfidenceCalibrator()
    prob = calibrator.calibrate(0.9)
    assert 0.0 < prob < 1.0

    brier = calculate_brier_score([0.9, 0.1], [1, 0])
    assert brier < 0.05

    ece, bin_data = calculate_expected_calibration_error([0.9, 0.8, 0.2], [1, 1, 0])
    assert ece >= 0.0
    assert isinstance(bin_data, list)


def test_verifier_missing_target_record():
    repo = FinancialDataRepository()
    t = datetime(2026, 3, 1, 10, 0, tzinfo=timezone.utc)
    
    src = CanonicalTransaction(transaction_id="TX_SRC", source_system=SourceSystem.GATEWAY, source_record_id="a", amount=Decimal("100.00"), currency="USD", transaction_timestamp=t)
    repo.store_canonical_transaction(src)

    rec = AgentRecommendation(
        case_id="C_1",
        recommended_decision=DecisionLabel.MATCHED,
        primary_reason=ReasonCode.EXACT_IDENTIFIER_MATCH,
        matched_record_ids=["TX_MISSING"],
        confidence_score=0.95
    )

    verifier = DeterministicPolicyVerifier(repo)
    final = verifier.verify_and_finalize(rec, src, calibrated_confidence=0.95)

    assert final.decision == DecisionLabel.UNCERTAIN
    assert final.reason == ReasonCode.MISSING_SOURCE_RECORD
    assert final.verifier_status == "VETOED_DOWNGRADED"
    assert "Veto: Target records missing from repository" in final.verifier_notes[0]


def test_verifier_extreme_amount_discrepancy():
    repo = FinancialDataRepository()
    t = datetime(2026, 3, 1, 10, 0, tzinfo=timezone.utc)
    
    src = CanonicalTransaction(transaction_id="TX_SRC", source_system=SourceSystem.GATEWAY, source_record_id="a", amount=Decimal("100.00"), currency="USD", transaction_timestamp=t)
    cand = CanonicalTransaction(transaction_id="TX_CAND", source_system=SourceSystem.BANK, source_record_id="b", amount=Decimal("1000000.00"), currency="USD", transaction_timestamp=t)
    
    repo.store_canonical_transaction(src)
    repo.store_canonical_transaction(cand)

    rec = AgentRecommendation(
        case_id="C_1",
        recommended_decision=DecisionLabel.MATCHED,
        primary_reason=ReasonCode.EXACT_IDENTIFIER_MATCH,
        matched_record_ids=["TX_CAND"],
        confidence_score=0.99
    )

    verifier = DeterministicPolicyVerifier(repo)
    final = verifier.verify_and_finalize(rec, src, calibrated_confidence=0.99)

    assert final.decision == DecisionLabel.EXCEPTION, f"Decision was {final.decision}, reason: {final.reason}, notes: {final.verifier_notes}"
    assert final.reason == ReasonCode.AMOUNT_MISMATCH
    assert final.verifier_status == "VETOED_DOWNGRADED"


def test_verifier_date_window_escalation():
    # If confidence is below the dynamic risk threshold due to extreme amounts, it should escalate.
    repo = FinancialDataRepository()
    t1 = datetime(2026, 3, 1, 10, 0, tzinfo=timezone.utc)
    
    # 200,000 INR (20 million paise) triggers high amount risk multiplier
    src = CanonicalTransaction(transaction_id="TX_SRC", source_system=SourceSystem.GATEWAY, source_record_id="a", amount=Decimal("200000.00"), currency="INR", transaction_timestamp=t1)
    cand = CanonicalTransaction(transaction_id="TX_CAND", source_system=SourceSystem.BANK, source_record_id="b", amount=Decimal("200000.00"), currency="INR", transaction_timestamp=t1)
    
    repo.store_canonical_transaction(src)
    repo.store_canonical_transaction(cand)

    rec = AgentRecommendation(
        case_id="C_1",
        recommended_decision=DecisionLabel.MATCHED,
        primary_reason=ReasonCode.FUZZY_ENTITY_MATCH,
        matched_record_ids=["TX_CAND"],
        confidence_score=0.86  # Might be below the scaled risk threshold for 200k INR
    )

    verifier = DeterministicPolicyVerifier(repo)
    # The default threshold is 0.85. 
    # With amount > 100k, multiplier is 1.05. 
    # FUZZY_ENTITY_MATCH risk factor is 1.1.
    # Total threshold = 0.85 * 1.05 * 1.1 = 0.98175
    # Since 0.86 < 0.98175, it should be escalated to UNCERTAIN.
    
    final = verifier.verify_and_finalize(rec, src, calibrated_confidence=0.86)

    assert final.decision == DecisionLabel.UNCERTAIN
    assert final.reason == ReasonCode.BELOW_CONFIDENCE_THRESHOLD
    assert final.requires_human_review is True

