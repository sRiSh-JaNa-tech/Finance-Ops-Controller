"""Tests for Production Realism: Double-Entry Ledger, Parsers, DLQ, and Cryptographic Provenance."""

import pytest
from decimal import Decimal
from datetime import datetime, timezone

from finance_ops.core.models import (
    CanonicalTransaction, SourceSystem, FinalDecisionRecord, DecisionLabel, ReasonCode
)
from finance_ops.ledger.journal import (
    GeneralLedgerPostingEngine, DoubleEntryJournalEntry, ChartOfAccounts, AccountType
)
from finance_ops.ingestion.parsers import (
    MT940BankStatementParser, GatewaySettlementCSVParser, DeadLetterQueue
)
from finance_ops.audit.provenance import (
    AuditProvenanceEngine, CryptographicAuditSeal
)
from finance_ops.benchmark.throughput_profiler import (
    ThroughputScalingProfiler, ConcurrentReconciliationPipeline
)


def test_double_entry_journal_balance_clean_match():
    src_tx = CanonicalTransaction(
        transaction_id="TXN_SRC_01",
        source_system=SourceSystem.GATEWAY,
        source_record_id="ch_01",
        amount=Decimal("1500.00"),
        amount_paise=150000,
        currency="INR"
    )
    decision = FinalDecisionRecord(
        decision_id="DEC_01",
        case_id="CASE_01",
        decision=DecisionLabel.MATCHED,
        reason=ReasonCode.EXACT_IDENTIFIER_MATCH,
        calibrated_confidence=0.98,
        is_automated=True,
        requires_human_review=False
    )
    entry = GeneralLedgerPostingEngine.create_journal_entry(decision, src_tx, [src_tx])
    assert entry.is_balanced is True
    assert entry.total_debit_paise == 150000
    assert entry.total_credit_paise == 150000


def test_double_entry_journal_balance_fee_deduction():
    src_tx = CanonicalTransaction(
        transaction_id="TXN_INV_02",
        source_system=SourceSystem.ERP,
        source_record_id="inv_02",
        amount=Decimal("10000.00"),
        amount_paise=1000000,
        currency="INR"
    )
    bank_tx = CanonicalTransaction(
        transaction_id="TXN_BNK_02",
        source_system=SourceSystem.BANK,
        source_record_id="stmt_02",
        amount=Decimal("9764.00"),
        amount_paise=976400,
        currency="INR"
    )
    decision = FinalDecisionRecord(
        decision_id="DEC_02",
        case_id="CASE_02",
        decision=DecisionLabel.MATCHED,
        reason=ReasonCode.FEE_ADJUSTED_MATCH,
        calibrated_confidence=0.95,
        is_automated=True,
        requires_human_review=False
    )
    entry = GeneralLedgerPostingEngine.create_journal_entry(decision, src_tx, [bank_tx])
    assert entry.is_balanced is True
    assert entry.total_debit_paise == 1000000
    assert entry.total_credit_paise == 1000000


def test_mt940_parser_and_dlq():
    dlq = DeadLetterQueue()
    raw_mt940 = """
:20:START
:25:123456789
:61:2603010301CR10000,00NTRFNONREF//REF-INV-2026-88
:86:PAYMENT RECEIVED UTR:UTRN99281928 INVOICE:INV-2026-88
:61:CORRUPT_STATEMENT_LINE_WITHOUT_PROPER_FORMAT
:86:GARBAGE DATA
-
"""
    txs = MT940BankStatementParser.parse_statement(raw_mt940, dlq=dlq)
    assert len(txs) == 1
    assert txs[0].amount_paise == 1000000
    assert txs[0].invoice_reference == "INV-2026-88"
    assert len(dlq.get_all()) == 1
    assert dlq.get_all()[0].source_format == "SWIFT_MT940"


def test_csv_gateway_parser():
    dlq = DeadLetterQueue()
    csv_data = """id,gross_amount,fee,tax,description
pay_001,500.00,10.00,1.80,INV-1001-ORD
pay_002,1200.50,24.00,4.32,INV-1002-ORD
invalid_row,,,,,
"""
    txs = GatewaySettlementCSVParser.parse_csv(csv_data, dlq=dlq)
    assert len(txs) == 2
    assert txs[0].amount_paise == 50000
    assert txs[1].amount_paise == 120050
    assert len(dlq.get_all()) == 1


def test_cryptographic_audit_seal():
    src = CanonicalTransaction(
        transaction_id="TXN_SEAL_SRC",
        source_system=SourceSystem.GATEWAY,
        source_record_id="ch_seal",
        amount=Decimal("250.00"),
        amount_paise=25000,
        currency="INR"
    )
    decision = FinalDecisionRecord(
        decision_id="DEC_SEAL_01",
        case_id="CASE_SEAL_01",
        decision=DecisionLabel.MATCHED,
        reason=ReasonCode.EXACT_IDENTIFIER_MATCH,
        calibrated_confidence=0.99,
        is_automated=True,
        requires_human_review=False
    )
    seal = AuditProvenanceEngine.generate_audit_seal(decision, src, [src])
    assert seal.evidence_merkle_root is not None
    assert len(seal.evidence_merkle_root) == 64
    assert AuditProvenanceEngine.verify_seal(seal, decision, src, [src]) is True


def test_throughput_profiler_execution():
    report = ThroughputScalingProfiler.profile_workload(batch_sizes=[20], workers=4)
    assert "scaling_curve" in report
    assert len(report["scaling_curve"]) == 1
    curve = report["scaling_curve"][0]
    assert curve["batch_size"] == 20
    assert curve["throughput_cases_per_sec"] > 0
    assert curve["p95_latency_ms"] >= 0
