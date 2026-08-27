"""Tests for Phase 1 Core Models, Provenance, and Financial Invariants."""

import pytest
from decimal import Decimal
from datetime import datetime, timezone
from finance_ops.core.models import (
    SourceSystem, TransactionType, DecisionLabel, ReasonCode,
    CanonicalTransaction, FieldProvenance
)
from finance_ops.core.provenance import compute_raw_hash, create_field_provenance
from finance_ops.core.invariants import (
    check_amount_conservation, check_split_payment_conservation,
    check_currency_consistency, check_reversal_integrity
)


def test_canonical_transaction_creation():
    tx = CanonicalTransaction(
        transaction_id="TX_1001",
        source_system=SourceSystem.GATEWAY,
        source_record_id="ch_stripe_999",
        amount=Decimal("150.00"),
        fee=Decimal("4.65"),
        tax=Decimal("0.00"),
        net_amount=Decimal("145.35"),
        currency="USD",
        transaction_timestamp=datetime(2026, 3, 15, 10, 30, tzinfo=timezone.utc),
        customer_name="Acme Corp",
        raw_narrative="STRIPE* ACME CORP INV-2026-01",
        normalized_narrative="acme corp inv 2026 01"
    )
    assert tx.amount == Decimal("150.00")
    assert tx.net_amount == Decimal("145.35")
    assert tx.source_system == SourceSystem.GATEWAY


def test_amount_conservation():
    # 150.00 Gross = 145.35 Net + 4.65 Fee
    valid, diff = check_amount_conservation(
        gross_amount=Decimal("150.00"),
        net_amount=Decimal("145.35"),
        fee_amount=Decimal("4.65"),
        tax_amount=Decimal("0.00")
    )
    assert valid is True
    assert diff == Decimal("0.00")

    # Mismatch test
    invalid, diff2 = check_amount_conservation(
        gross_amount=Decimal("150.00"),
        net_amount=Decimal("140.00"),
        fee_amount=Decimal("4.65"),
        tax_amount=Decimal("0.00")
    )
    assert invalid is False
    assert diff2 == Decimal("5.35")


def test_split_payment_conservation():
    valid, diff = check_split_payment_conservation(
        parent_amount=Decimal("1000.00"),
        child_amounts=[Decimal("300.00"), Decimal("450.00"), Decimal("250.00")]
    )
    assert valid is True
    assert diff == Decimal("0.00")


def test_reversal_integrity():
    t1 = datetime(2026, 3, 10, 12, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 3, 12, 14, 0, tzinfo=timezone.utc)
    
    orig = CanonicalTransaction(
        transaction_id="TX_1",
        source_system=SourceSystem.GATEWAY,
        source_record_id="ch_1",
        amount=Decimal("200.00"),
        currency="USD",
        transaction_timestamp=t1
    )
    reversal = CanonicalTransaction(
        transaction_id="TX_2",
        source_system=SourceSystem.GATEWAY,
        source_record_id="re_1",
        amount=Decimal("200.00"),
        currency="USD",
        transaction_timestamp=t2,
        is_reversal=True
    )
    valid, msg = check_reversal_integrity(orig, reversal)
    assert valid is True
