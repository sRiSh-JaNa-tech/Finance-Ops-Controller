from typing import Dict, Any, List
from langchain_core.tools import tool
from finance_ops.evidence.tools import InvestigationToolbox

def create_agent_tools(toolbox: InvestigationToolbox):
    
    @tool
    def run_financial_rules(source_txn_ids: List[str], target_txn_ids: List[str], rule_category: str = "ALL") -> Dict[str, Any]:
        """Executes deterministic integer-paise financial rules on specified records. Useful to verify math and timing logic before deciding."""
        return toolbox.run_financial_rules(source_txn_ids, target_txn_ids, rule_category)

    @tool
    def retrieve_candidates(query_id: str, amount_tolerance_pct: float = 0.05, date_window_days: int = 5) -> Dict[str, Any]:
        """Queries the entity index for alternative matching candidates with relaxed search parameters."""
        return toolbox.retrieve_candidates(query_id, amount_tolerance_pct, date_window_days)

    @tool
    def get_related_events(txn_id: str, event_types: List[str] = None) -> Dict[str, Any]:
        """Retrieves related financial events (like refunds, disputes, or settlements) from the ledger for a specific transaction."""
        return toolbox.get_related_events(txn_id, event_types)

    @tool
    def inspect_entity_graph(merchant_id: str, transaction_id: str = None) -> Dict[str, Any]:
        """Queries the graph database for merchant aliases, historical payout patterns, and known affiliations."""
        return toolbox.inspect_entity_graph(merchant_id, transaction_id)

    @tool
    def search_source_history(search_term: str, lookback_days: int = 30) -> Dict[str, Any]:
        """Performs a fuzzy text search across historical source records for specific keywords or references."""
        return toolbox.search_source_history(search_term, lookback_days)

    @tool
    def test_reconciliation_hypothesis(hypothesis_type: str, record_ids: List[str], expected_delta_paise: int = 0) -> Dict[str, Any]:
        """Formally tests a reconciliation hypothesis (e.g., 'FEE_MDR', 'SPLIT_PAYMENT') against the split solver engine."""
        return toolbox.test_reconciliation_hypothesis(hypothesis_type, record_ids, expected_delta_paise)

    return [
        run_financial_rules,
        retrieve_candidates,
        get_related_events,
        inspect_entity_graph,
        search_source_history,
        test_reconciliation_hypothesis
    ]
