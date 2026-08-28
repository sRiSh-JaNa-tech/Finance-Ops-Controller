"""Tests for Retrieve-Rank-Route-Reason Cascade Architecture."""

import pytest
from decimal import Decimal
from datetime import datetime, timezone

from finance_ops.core.models import (
    CanonicalTransaction, SourceSystem, DecisionLabel, ReasonCode
)
from finance_ops.retrieval.reranker import DeterministicCandidateReranker, EvidencePacketBuilder
from finance_ops.agent.cascade_router import (
    ReconciliationDifficultyEstimator, CascadeExecutionTier, CascadeReconciliationPipeline
)
from finance_ops.benchmark.metrics import compute_selective_prediction_curve


@pytest.fixture
def sample_transactions():
    now = datetime(2026, 8, 28, 12, 0, 0, tzinfo=timezone.utc)
    src = CanonicalTransaction(
        transaction_id="TXN_GW_001",
        source_system=SourceSystem.RAZORPAY,
        source_record_id="SRC_GW_001",
        amount=Decimal("1500.00"),
        amount_paise=150000,
        currency="INR",
        transaction_timestamp=now,
        txn_timestamp=int(now.timestamp()),
        invoice_reference="INV-2026-9999",
        utr="400012345678",
        merchant_name_norm="flipkart"
    )
    cand_exact = CanonicalTransaction(
        transaction_id="TXN_BANK_001",
        source_system=SourceSystem.BANK,
        source_record_id="SRC_BANK_001",
        amount=Decimal("1500.00"),
        amount_paise=150000,
        currency="INR",
        transaction_timestamp=now,
        txn_timestamp=int(now.timestamp()),
        invoice_reference="INV-2026-9999",
        utr="400012345678",
        merchant_name_norm="flipkart"
    )
    cand_fee = CanonicalTransaction(
        transaction_id="TXN_BANK_002",
        source_system=SourceSystem.BANK,
        source_record_id="SRC_BANK_002",
        amount=Decimal("1470.00"),
        amount_paise=147000,
        currency="INR",
        transaction_timestamp=now,
        txn_timestamp=int(now.timestamp()),
        invoice_reference="INV-2026-9999",
        utr=None,
        merchant_name_norm="flipkart"
    )
    return src, [cand_exact, cand_fee]


def test_reranker_scoring(sample_transactions):
    src, candidates = sample_transactions
    reranker = DeterministicCandidateReranker(top_k=3)
    ranked = reranker.rerank(src, candidates)

    assert len(ranked) == 2
    assert ranked[0]["candidate_id"] == "TXN_BANK_001"
    assert ranked[0]["composite_score"] > 0.95
    assert ranked[1]["candidate_id"] == "TXN_BANK_002"
    assert ranked[1]["composite_score"] < ranked[0]["composite_score"]


def test_evidence_packet_builder(sample_transactions):
    src, candidates = sample_transactions
    reranker = DeterministicCandidateReranker(top_k=3)
    ranked = reranker.rerank(src, candidates)
    packet = EvidencePacketBuilder.build_packet(src, ranked)

    assert packet["source_transaction"]["transaction_id"] == "TXN_GW_001"
    assert packet["candidate_count"] == 2
    assert packet["top_candidates"][0]["candidate_id"] == "TXN_BANK_001"
    assert "feature_scores" in packet["top_candidates"][0]


def test_difficulty_estimator_routing(sample_transactions):
    src, candidates = sample_transactions
    reranker = DeterministicCandidateReranker(top_k=3)
    estimator = ReconciliationDifficultyEstimator()

    # Exact match candidate -> Tier 1 Fast Path
    ranked_exact = reranker.rerank(src, [candidates[0]])
    tier, diff, reason = estimator.estimate_difficulty(src, ranked_exact)
    assert tier == CascadeExecutionTier.TIER_1_DETERMINISTIC_FAST_PATH
    assert diff < 0.20

    # Fee adjusted candidate -> Tier 2 Single-Turn AI
    ranked_fee = reranker.rerank(src, [candidates[1]])
    tier_fee, diff_fee, _ = estimator.estimate_difficulty(src, ranked_fee)
    assert tier_fee == CascadeExecutionTier.TIER_2_SINGLE_TURN_EVIDENCE
    assert diff_fee >= 0.40


def test_cascade_pipeline_execution(sample_transactions):
    src, candidates = sample_transactions
    from finance_ops.ingestion.storage import FinancialDataRepository
    repo = FinancialDataRepository()
    repo.store_canonical_transaction(src)
    for c in candidates:
        repo.store_canonical_transaction(c)
        
    pipeline = CascadeReconciliationPipeline(repository=repo, mode="offline")
    
    # 1. Clean single candidate -> Tier 1 Fast-Path Auto-Reconciliation
    res_clean = pipeline.process_single_case(src, [candidates[0]], case_id="CASE_CLEAN_001")
    assert res_clean["case_id"] == "CASE_CLEAN_001"
    assert res_clean["decision"] == DecisionLabel.MATCHED
    assert res_clean["tier"] == CascadeExecutionTier.TIER_1_DETERMINISTIC_FAST_PATH.value

    # 2. Competing candidate tie -> Tier 3 Escalation to Human Review (Selective Prediction)
    res_ambiguous = pipeline.process_single_case(src, candidates, case_id="CASE_AMBIG_001", template="S14_CANDIDATE_TIE_AMBIGUITY")
    assert res_ambiguous["case_id"] == "CASE_AMBIG_001"
    assert res_ambiguous["decision"] == DecisionLabel.UNCERTAIN
    assert res_ambiguous["tier"] == CascadeExecutionTier.TIER_3_DEEP_REASONING.value


def test_selective_prediction_curve():
    preds = [
        {"decision": "MATCHED", "confidence_score": 0.99, "matched_record_ids": ["B1"]},
        {"decision": "MATCHED", "confidence_score": 0.75, "matched_record_ids": ["B2"]},
        {"decision": "UNCERTAIN", "confidence_score": 0.40, "matched_record_ids": []}
    ]
    gt = [
        {"expected_decision": "MATCHED", "candidate_tx_ids": ["B1"]},
        {"expected_decision": "EXCEPTION", "candidate_tx_ids": ["B2"]},
        {"expected_decision": "UNCERTAIN", "candidate_tx_ids": []}
    ]
    curve = compute_selective_prediction_curve(preds, gt, thresholds=[0.70, 0.90])
    assert len(curve) == 2
    # At 0.70 threshold: 2 automated (1 TP, 1 FP) -> FMR = 0.50
    assert curve[0]["automation_rate"] > curve[1]["automation_rate"]
    # At 0.90 threshold: only 1 automated (1 TP, 0 FP) -> FMR = 0.00
    assert curve[1]["false_match_rate"] == 0.0
