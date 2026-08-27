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

