"""Tests for Phase 7 Bounded Investigation Agent and Stopping Policies."""

from decimal import Decimal
from datetime import datetime, timezone
from finance_ops.core.models import CanonicalTransaction, SourceSystem, DecisionLabel, ReasonCode
from finance_ops.ingestion.storage import FinancialDataRepository
from finance_ops.retrieval.blocking import CandidateBlockingEngine
from finance_ops.retrieval.graph import FinancialEntityGraph
from finance_ops.rules.engine import FinancialRuleEngine
from finance_ops.rules.constraint_solver import SplitReconciliationSolver
from finance_ops.evidence.tools import InvestigationToolbox
from finance_ops.agent.investigator import BoundedInvestigationAgent


def test_agent_clean_match():
    repo = FinancialDataRepository()
    blocking = CandidateBlockingEngine()
    graph = FinancialEntityGraph()
    rules = FinancialRuleEngine()
    solver = SplitReconciliationSolver()

    t = datetime(2026, 3, 1, 10, 0, tzinfo=timezone.utc)
    gw = CanonicalTransaction(
        transaction_id="GW_100",
        source_system=SourceSystem.GATEWAY,
        source_record_id="ch_1",
        amount=Decimal("100.00"),
        currency="USD",
        transaction_timestamp=t,
        invoice_reference="INV-2026-99"
    )
    bk = CanonicalTransaction(
        transaction_id="BK_200",
        source_system=SourceSystem.BANK,
        source_record_id="stmt_1",
        amount=Decimal("100.00"),
        currency="USD",
        transaction_timestamp=t,
        invoice_reference="INV-2026-99"
    )
    repo.store_canonical_transaction(gw)
    repo.store_canonical_transaction(bk)
    blocking.index_transactions([gw, bk])
    graph.add_transaction_node(gw)
    graph.add_transaction_node(bk)

    toolbox = InvestigationToolbox(repo, blocking, graph, rules, solver)
    agent = BoundedInvestigationAgent(toolbox, max_steps=5)

    rec = agent.investigate_case("CASE_TEST_01", gw)
    assert rec.recommended_decision == DecisionLabel.MATCHED
    assert rec.primary_reason == ReasonCode.EXACT_IDENTIFIER_MATCH
    assert "BK_200" in rec.matched_record_ids
    assert len(rec.cited_evidence_ids) > 0


def test_agent_abstention_on_missing_records():
    repo = FinancialDataRepository()
    blocking = CandidateBlockingEngine()
    graph = FinancialEntityGraph()
    rules = FinancialRuleEngine()
    solver = SplitReconciliationSolver()

    t = datetime(2026, 3, 1, 10, 0, tzinfo=timezone.utc)
    gw = CanonicalTransaction(
        transaction_id="GW_ORPHAN",
        source_system=SourceSystem.GATEWAY,
        source_record_id="ch_orphan",
        amount=Decimal("5000.00"),
        currency="USD",
        transaction_timestamp=t,
        invoice_reference="INV-NONE"
    )
    repo.store_canonical_transaction(gw)
    blocking.index_transactions([gw])

    toolbox = InvestigationToolbox(repo, blocking, graph, rules, solver)
    agent = BoundedInvestigationAgent(toolbox, max_steps=5)

    rec = agent.investigate_case("CASE_ORPHAN", gw)
    assert rec.recommended_decision == DecisionLabel.UNCERTAIN
    assert rec.primary_reason == ReasonCode.MISSING_SOURCE_RECORD
