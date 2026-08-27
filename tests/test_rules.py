"""Tests for Phase 5 Deterministic Financial Rules and Constraint Solver."""

from decimal import Decimal
from datetime import datetime, timezone, timedelta
from finance_ops.core.models import CanonicalTransaction, SourceSystem
from finance_ops.rules.engine import ExactAmountRule, FeeAdjustedSettlementRule, DateWindowToleranceRule
from finance_ops.rules.constraint_solver import SplitReconciliationSolver


def test_exact_amount_rule():
    t = datetime(2026, 3, 1, 10, 0, tzinfo=timezone.utc)
    tx1 = CanonicalTransaction(transaction_id="1", source_system=SourceSystem.GATEWAY, source_record_id="a", amount=Decimal("100.00"), currency="USD", transaction_timestamp=t)
    tx2 = CanonicalTransaction(transaction_id="2", source_system=SourceSystem.BANK, source_record_id="b", amount=Decimal("100.00"), currency="USD", transaction_timestamp=t)
    tx3 = CanonicalTransaction(transaction_id="3", source_system=SourceSystem.BANK, source_record_id="c", amount=Decimal("105.00"), currency="USD", transaction_timestamp=t)

    rule = ExactAmountRule()
    assert rule.evaluate(tx1, tx2).passed is True
    assert rule.evaluate(tx1, tx3).passed is False


def test_fee_adjusted_settlement_rule():
    t = datetime(2026, 3, 1, 10, 0, tzinfo=timezone.utc)
    # 100.00 gross -> 2.9% + 0.30 fee = 3.20 fee -> 96.80 net
    tx_gross = CanonicalTransaction(transaction_id="1", source_system=SourceSystem.GATEWAY, source_record_id="a", amount=Decimal("100.00"), currency="USD", transaction_timestamp=t)
    tx_net = CanonicalTransaction(transaction_id="2", source_system=SourceSystem.BANK, source_record_id="b", amount=Decimal("96.80"), currency="USD", transaction_timestamp=t)
    
    rule = FeeAdjustedSettlementRule()
    res = rule.evaluate(tx_gross, tx_net)
    assert res.passed is True


def test_split_reconciliation_solver():
    t = datetime(2026, 3, 1, 10, 0, tzinfo=timezone.utc)
    target = CanonicalTransaction(transaction_id="INV_1", source_system=SourceSystem.ERP, source_record_id="inv1", amount=Decimal("500.00"), currency="USD", transaction_timestamp=t)
    
    c1 = CanonicalTransaction(transaction_id="SPLIT_1", source_system=SourceSystem.GATEWAY, source_record_id="s1", amount=Decimal("200.00"), currency="USD", transaction_timestamp=t)
    c2 = CanonicalTransaction(transaction_id="SPLIT_2", source_system=SourceSystem.GATEWAY, source_record_id="s2", amount=Decimal("300.00"), currency="USD", transaction_timestamp=t + timedelta(days=1))
    c3 = CanonicalTransaction(transaction_id="OTHER", source_system=SourceSystem.GATEWAY, source_record_id="o1", amount=Decimal("999.00"), currency="USD", transaction_timestamp=t)

    solver = SplitReconciliationSolver()
    solution = solver.solve_1_to_n(target, [c1, c2, c3])
    
    assert solution is not None
    assert solution["child_count"] == 2
    assert "SPLIT_1" in solution["matched_transaction_ids"]
    assert "SPLIT_2" in solution["matched_transaction_ids"]
    assert solution["discrepancy"] == 0.0
