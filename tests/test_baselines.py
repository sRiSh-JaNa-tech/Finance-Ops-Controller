"""Tests for Phase 9 Baselines and Ablations."""

from decimal import Decimal
from datetime import datetime, timezone
from finance_ops.core.models import CanonicalTransaction, SourceSystem, DecisionLabel
from finance_ops.baselines.baseline_models import ExactIdentifierMatcher, DeterministicRuleMatcher, Prototype1HybridPipeline
from finance_ops.baselines.ablations import AblationConfig, AblationMode


def test_baselines_execution():
    t = datetime(2026, 3, 1, 10, 0, tzinfo=timezone.utc)
    src = CanonicalTransaction(transaction_id="SRC", source_system=SourceSystem.GATEWAY, source_record_id="1", amount=Decimal("100.00"), currency="USD", transaction_timestamp=t, invoice_reference="INV-100")
    cand = CanonicalTransaction(transaction_id="CAND", source_system=SourceSystem.BANK, source_record_id="2", amount=Decimal("100.00"), currency="USD", transaction_timestamp=t, invoice_reference="INV-100")

    exact = ExactIdentifierMatcher()
    res1 = exact.match(src, [cand])
    assert res1["decision"] == DecisionLabel.MATCHED
    assert res1["matched_id"] == "CAND"

    rules = DeterministicRuleMatcher()
    res2 = rules.match(src, [cand])
    assert res2["decision"] == DecisionLabel.MATCHED

    proto1 = Prototype1HybridPipeline()
    res3 = proto1.match(src, [cand])
    assert res3["decision"] == DecisionLabel.MATCHED


def test_ablation_config_toggle():
    cfg = AblationConfig(mode=AblationMode.NO_GRAPH)
    assert cfg.enable_graph is False
    assert cfg.enable_rules is True
