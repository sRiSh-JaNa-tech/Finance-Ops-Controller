"""Financial Entity Graph and Multi-Hop Context Expansion."""

import networkx as nx
from typing import Dict, List, Any, Optional
from finance_ops.core.models import CanonicalTransaction, InvoiceRecord, SettlementBatch


class FinancialEntityGraph:
    """Graph structure connecting Transactions, Parties, Invoices, and Settlement Batches."""

    def __init__(self):
        self._graph = nx.DiGraph()

    def add_transaction_node(self, tx: CanonicalTransaction) -> None:
        """Adds a canonical transaction node and links to party / reference entities."""
        node_id = f"tx:{tx.transaction_id}"
        self._graph.add_node(
            node_id,
            node_type="TRANSACTION",
            amount=float(tx.amount),
            currency=tx.currency,
            source_system=tx.source_system.value,
            timestamp=tx.transaction_timestamp.isoformat(),
            ground_truth_tx_id=tx.ground_truth_tx_id
        )

        # Connect to Customer if present
        if tx.customer_id or tx.customer_name:
            cust_node = f"customer:{tx.customer_id or tx.customer_name}"
            self._graph.add_node(cust_node, node_type="CUSTOMER", name=tx.customer_name)
            self._graph.add_edge(node_id, cust_node, relation="INITIATED_BY")

        # Connect to Merchant if present
        if tx.merchant_id or tx.merchant_name:
            merch_node = f"merchant:{tx.merchant_id or tx.merchant_name}"
            self._graph.add_node(merch_node, node_type="MERCHANT", name=tx.merchant_name)
            self._graph.add_edge(node_id, merch_node, relation="BENEFICIARY")

        # Connect to Invoice reference if present
        if tx.invoice_reference:
            inv_node = f"invoice_ref:{tx.invoice_reference}"
            self._graph.add_node(inv_node, node_type="INVOICE_REF", reference=tx.invoice_reference)
            self._graph.add_edge(node_id, inv_node, relation="REFERENCES_INVOICE")

        # Connect to parent if refund/reversal
        if tx.parent_transaction_id:
            parent_node = f"tx:{tx.parent_transaction_id}"
            self._graph.add_edge(node_id, parent_node, relation="REVERSES_OR_REFUNDS")

    def get_k_hop_neighborhood(self, transaction_id: str, k: int = 2) -> Dict[str, Any]:
        """Extracts the k-hop ego sub-graph around a transaction to provide contextual relational evidence."""
        node_id = f"tx:{transaction_id}"
        if not self._graph.has_node(node_id):
            return {"nodes": [], "edges": [], "connected_transactions": []}

        # Ego subgraph with undirected view for neighborhood expansion
        undirected_view = self._graph.to_undirected(as_view=True)
        subgraph_nodes = nx.single_source_shortest_path_length(undirected_view, node_id, cutoff=k).keys()
        subgraph = self._graph.subgraph(subgraph_nodes)

        nodes_data = [{"id": n, **subgraph.nodes[n]} for n in subgraph.nodes]
        edges_data = [{"source": u, "target": v, **subgraph.edges[u, v]} for u, v in subgraph.edges]
        
        connected_txs = [
            n.replace("tx:", "") for n in subgraph.nodes 
            if n.startswith("tx:") and n != node_id
        ]

        return {
            "root_node": node_id,
            "hop_depth": k,
            "nodes": nodes_data,
            "edges": edges_data,
            "connected_transactions": connected_txs
        }
