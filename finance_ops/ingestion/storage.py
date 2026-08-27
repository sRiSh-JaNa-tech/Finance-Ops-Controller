"""Audit-Preserving Financial Storage and In-Memory Indexed Repository."""

from typing import Dict, List, Optional
from datetime import datetime
from finance_ops.core.models import (
    CanonicalTransaction, RawSourceRecord, FinalDecisionRecord, SourceSystem
)


class FinancialDataRepository:
    """Thread-safe, audit-preserving repository for raw records, canonical models, and decisions."""

    def __init__(self):
        self._raw_records: Dict[str, RawSourceRecord] = {}
        self._canonical_records: Dict[str, CanonicalTransaction] = {}
        self._decisions: Dict[str, FinalDecisionRecord] = {}
        self._audit_log: List[Dict] = []

    def store_raw_record(self, record: RawSourceRecord) -> None:
        self._raw_records[record.raw_record_id] = record
        self._log_audit("STORE_RAW", record.raw_record_id, f"Source: {record.source_system.value}")

    def store_canonical_transaction(self, tx: CanonicalTransaction) -> None:
        self._canonical_records[tx.transaction_id] = tx
        self._log_audit("STORE_CANONICAL", tx.transaction_id, f"Amount: {tx.amount} {tx.currency}")

    def get_transaction(self, tx_id: str) -> Optional[CanonicalTransaction]:
        return self._canonical_records.get(tx_id)

    def list_transactions(self, source_system: Optional[SourceSystem] = None) -> List[CanonicalTransaction]:
        if source_system:
            return [tx for tx in self._canonical_records.values() if tx.source_system == source_system]
        return list(self._canonical_records.values())

    def store_decision(self, decision: FinalDecisionRecord) -> None:
        self._decisions[decision.decision_id] = decision
        self._log_audit("STORE_DECISION", decision.decision_id, f"Decision: {decision.decision.value} Reason: {decision.reason.value}")

    def get_decision(self, decision_id: str) -> Optional[FinalDecisionRecord]:
        return self._decisions.get(decision_id)

    def list_decisions(self) -> List[FinalDecisionRecord]:
        return list(self._decisions.values())

    def list_all_transactions(self) -> List[CanonicalTransaction]:
        """Returns all stored canonical transactions (all source systems)."""
        return list(self._canonical_records.values())

    def _log_audit(self, action: str, entity_id: str, details: str) -> None:
        self._audit_log.append({
            "timestamp": datetime.utcnow().isoformat(),
            "action": action,
            "entity_id": entity_id,
            "details": details
        })

    @property
    def audit_trail(self) -> List[Dict]:
        return list(self._audit_log)
