import pytest
import os
from unittest.mock import MagicMock
from datetime import datetime

from finance_ops.core.models import CanonicalTransaction, SourceSystem, DecisionLabel
from finance_ops.agent.vertex_client import GeminiReconciliationClient
from finance_ops.evidence.tools import InvestigationToolbox

class MockBrokenToolbox(InvestigationToolbox):
    def __init__(self):
        super().__init__(repository=MagicMock())
        self.call_counts = {}

    def inspect_entity_graph(self, merchant_id, transaction_id):
        self.call_counts["inspect_entity_graph"] = self.call_counts.get("inspect_entity_graph", 0) + 1
        raise ConnectionError("Entity Graph Database is currently unreachable (Timeout).")
        
    def retrieve_candidates(self, query_id, amount_tolerance_pct=0.05, date_window_days=5):
        self.call_counts["retrieve_candidates"] = self.call_counts.get("retrieve_candidates", 0) + 1
        return {"candidates": []}

    def run_financial_rules(self, source_txn_ids, target_txn_ids, rule_category="ALL"):
        self.call_counts["run_financial_rules"] = self.call_counts.get("run_financial_rules", 0) + 1
        return {"passed_rules": ["AC-1"], "failed_rules": [], "warned_rules": [], "leakage_risk": 0.0}

def test_agent_tool_recovery():
    if not os.environ.get("GEMINI_API_KEY"):
        pytest.skip("Skipping live LLM recovery test (No API key found)")
        
    client = GeminiReconciliationClient()
    toolbox = MockBrokenToolbox()
    
    src = CanonicalTransaction(
        transaction_id="SRC_101",
        source_record_id="SRC_REC_101",
        source_system=SourceSystem.GATEWAY,
        amount=100.0,
        amount_paise=10000,
        raw_narrative="Payment for services",
        transaction_timestamp=datetime.now(),
        invoice_reference="INV-2023-01",
        utr="UTR888",
        order_id="ORD-001"
    )
    
    tgt = CanonicalTransaction(
        transaction_id="TGT_101",
        source_record_id="TGT_REC_101",
        source_system=SourceSystem.BANK,
        amount=100.0,
        amount_paise=10000,
        raw_narrative="Settlement for INV-2023-01",
        transaction_timestamp=datetime.now(),
        invoice_reference="INV-2023-01",
        utr="UTR888",
        order_id="ORD-001"
    )

    rule_res = {"passed_rules": ["AC-1"], "failed_rules": [], "warned_rules": [], "leakage_risk": 0.0}
    
    rec = client._investigate_with_gemini(
        case_id="CASE_TOOL_REC_01",
        source_tx=src,
        candidates=[tgt],
        toolbox=toolbox,
        rule_res=rule_res
    )

    assert rec is not None
    assert isinstance(rec.recommended_decision, DecisionLabel)
