"""Tests for Phase 6 Evidence Bundle and Investigation Tools."""

from decimal import Decimal
from datetime import datetime, timezone
from finance_ops.core.models import CanonicalTransaction, SourceSystem
from finance_ops.ingestion.storage import FinancialDataRepository
from finance_ops.retrieval.blocking import CandidateBlockingEngine
from finance_ops.retrieval.graph import FinancialEntityGraph
from finance_ops.rules.engine import FinancialRuleEngine
from finance_ops.rules.constraint_solver import SplitReconciliationSolver
from finance_ops.evidence.bundle import EvidenceBundleBuilder
from finance_ops.evidence.tools import InvestigationToolbox


def test_evidence_bundle_builder():
    t = datetime(2026, 3, 1, 10, 0, tzinfo=timezone.utc)
    src = CanonicalTransaction(transaction_id="SRC_1", source_system=SourceSystem.GATEWAY, source_record_id="a", amount=Decimal("100.00"), currency="USD", transaction_timestamp=t, invoice_reference="INV-123")
    cand = CanonicalTransaction(transaction_id="CAND_1", source_system=SourceSystem.BANK, source_record_id="b", amount=Decimal("100.00"), currency="USD", transaction_timestamp=t, invoice_reference="INV-123")
    
    rule_engine = FinancialRuleEngine()
    rule_results = {cand.transaction_id: rule_engine.evaluate_pair(src, cand)}
    
    bundle = EvidenceBundleBuilder.build_bundle(
        case_id="CASE_001",
        source_tx=src,
        candidates=[cand],
        rule_results_map=rule_results,
        graph_neighborhoods_map={}
    )
    
    assert bundle.case_id == "CASE_001"
    assert len(bundle.facts) >= 2
    assert any(f.fact_type == "IDENTIFIER" for f in bundle.facts)


def test_investigation_toolbox_execution():
    repo = FinancialDataRepository()
    blocking = CandidateBlockingEngine()
    graph = FinancialEntityGraph()
    rules = FinancialRuleEngine()
    solver = SplitReconciliationSolver()

    t = datetime(2026, 3, 1, 10, 0, tzinfo=timezone.utc)
    tx1 = CanonicalTransaction(transaction_id="TX_A", source_system=SourceSystem.GATEWAY, source_record_id="a", amount=Decimal("50.00"), currency="USD", transaction_timestamp=t)
    tx2 = CanonicalTransaction(transaction_id="TX_B", source_system=SourceSystem.BANK, source_record_id="b", amount=Decimal("50.00"), currency="USD", transaction_timestamp=t)
    
    repo.store_canonical_transaction(tx1)
    repo.store_canonical_transaction(tx2)
    blocking.index_transactions([tx1, tx2])
    graph.add_transaction_node(tx1)
    graph.add_transaction_node(tx2)

    toolbox = InvestigationToolbox(repo, blocking, graph, rules, solver)
    
    # 1. Test retrieve candidates
    cands = toolbox.retrieve_candidates("TX_A")
    assert cands["status"] == "SUCCESS"
    assert cands["candidate_count"] == 1
    
    # 2. Test run rules
    rule_res = toolbox.run_financial_rules("TX_A", "TX_B")
    assert rule_res["status"] == "SUCCESS"
    assert rule_res["is_financially_compatible"] is True
