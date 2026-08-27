"""Tests for Phase 3 Ingestion, Normalization, and Storage."""

from decimal import Decimal
from finance_ops.core.models import SourceSystem, FinalDecisionRecord, DecisionLabel, ReasonCode
from finance_ops.ingestion.normalizer import (
    normalize_amount, normalize_timestamp, normalize_narrative, normalize_raw_transaction
)
from finance_ops.ingestion.storage import FinancialDataRepository


def test_amount_normalizer_formats():
    assert normalize_amount("$1,234.56")[0] == Decimal("1234.56")
    assert normalize_amount("(500.25)")[0] == Decimal("-500.25")
    assert normalize_amount(" € 99.99 ")[0] == Decimal("99.99")


def test_provenance_preservation():
    raw_payload = {
        "amount": "$450.00",
        "created_utc": "2026-03-12T14:30:00Z",
        "Description": "INVOICE # 9876 PAYMENT",
        "currency": "USD"
    }
    tx = normalize_raw_transaction(
        source_system=SourceSystem.GATEWAY,
        source_record_id="rec_001",
        raw_payload=raw_payload
    )
    assert tx.amount == Decimal("450.00")
    assert "amount" in tx.provenance
    assert tx.provenance["amount"].raw_value == "$450.00"
    assert tx.provenance["amount"].normalized_value == "450.00"


def test_repository_audit_logging():
    repo = FinancialDataRepository()
    raw_payload = {"amount": "100.00", "created_utc": "2026-03-01"}
    tx = normalize_raw_transaction(SourceSystem.BANK, "stmt_1", raw_payload)
    repo.store_canonical_transaction(tx)
    
    assert len(repo.list_transactions()) == 1
    assert len(repo.audit_trail) == 1
    assert repo.audit_trail[0]["action"] == "STORE_CANONICAL"
