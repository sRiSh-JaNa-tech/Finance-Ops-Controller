"""Tests for Phase 4 Candidate Retrieval, Blocking, Similarity, and Entity Graph."""

from decimal import Decimal
from datetime import datetime, timezone
from finance_ops.core.models import CanonicalTransaction, SourceSystem
from finance_ops.retrieval.blocking import CandidateBlockingEngine
from finance_ops.retrieval.similarity import calculate_candidate_similarity
from finance_ops.retrieval.graph import FinancialEntityGraph


def test_blocking_candidate_retrieval():
    t1 = datetime(2026, 3, 1, 10, 0, tzinfo=timezone.utc)
    
    tx1 = CanonicalTransaction(
        transaction_id="TX_1",
        source_system=SourceSystem.GATEWAY,
        source_record_id="ch_1",
        amount=Decimal("120.00"),
        currency="USD",
        transaction_timestamp=t1,
        invoice_reference="INV-999"
    )
    tx2 = CanonicalTransaction(
        transaction_id="TX_2",
        source_system=SourceSystem.BANK,
        source_record_id="stmt_1",
        amount=Decimal("120.00"),
        currency="USD",
        transaction_timestamp=t1,
        invoice_reference="INV-999"
    )
    tx3 = CanonicalTransaction(
        transaction_id="TX_3",
        source_system=SourceSystem.BANK,
        source_record_id="stmt_2",
        amount=Decimal("9000.00"),
        currency="USD",
        transaction_timestamp=t1,
        invoice_reference="INV-000"
    )

    engine = CandidateBlockingEngine()
    engine.index_transactions([tx1, tx2, tx3])
    
    candidates = engine.retrieve_candidate_ids(tx1)
    assert "TX_2" in candidates
    assert "TX_3" not in candidates


def test_similarity_features():
    t = datetime(2026, 3, 1, 10, 0, tzinfo=timezone.utc)
    tx1 = CanonicalTransaction(
        transaction_id="TX_1",
        source_system=SourceSystem.GATEWAY,
        source_record_id="ch_1",
        amount=Decimal("250.00"),
        currency="USD",
        transaction_timestamp=t,
        invoice_reference="INV-100",
        raw_narrative="Acme Corp Payment"
    )
    tx2 = CanonicalTransaction(
        transaction_id="TX_2",
        source_system=SourceSystem.BANK,
        source_record_id="stmt_1",
        amount=Decimal("250.00"),
        currency="USD",
        transaction_timestamp=t,
        invoice_reference="INV-100",
        raw_narrative="ACH Acme Corp Inv-100"
    )
    sim = calculate_candidate_similarity(tx1, tx2)
    assert sim["identifier_score"] == 1.0
    assert sim["amount_score"] == 1.0
    assert sim["composite_score"] >= 0.80


def test_graph_neighborhood_expansion():
    t = datetime(2026, 3, 1, 10, 0, tzinfo=timezone.utc)
    tx1 = CanonicalTransaction(
        transaction_id="TX_1",
        source_system=SourceSystem.GATEWAY,
        source_record_id="ch_1",
        amount=Decimal("50.00"),
        currency="USD",
        transaction_timestamp=t,
        invoice_reference="INV-777",
        customer_name="Omega Inc"
    )
    graph = FinancialEntityGraph()
    graph.add_transaction_node(tx1)
    
    res = graph.get_k_hop_neighborhood("TX_1", k=2)
    assert len(res["nodes"]) >= 2
    assert any(n["id"] == "invoice_ref:INV-777" for n in res["nodes"])
