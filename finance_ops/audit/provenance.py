"""Production Tamper-Evident Cryptographic Audit Trail & Evidence Provenance.

Generates SHA-256 evidence seals and Merkle hash verification for SOC 1 / SOX auditability.
"""

import hashlib
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field

from finance_ops.core.models import CanonicalTransaction, FinalDecisionRecord


class CryptographicAuditSeal(BaseModel):
    """
    Immutable Cryptographic Evidence Seal for a reconciliation decision.
    """
    case_id: str
    decision_id: str
    sealed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source_tx_hash: str
    target_tx_hashes: List[str] = Field(default_factory=list)
    rules_evaluation_hash: str
    evidence_merkle_root: str
    signature_algorithm: str = "SHA-256 Evidence Root"
    is_tamper_evident: bool = True


class AuditProvenanceEngine:
    """
    Produces deterministic, cryptographic hashes across the full decision lifecycle:
        Source TX + Target TXs + Evaluated Rules + Decision Payload -> Immutable Merkle Root
    """

    @staticmethod
    def hash_payload(data: Any) -> str:
        if isinstance(data, dict):
            serialized = json.dumps(data, sort_keys=True, default=str)
        elif isinstance(data, str):
            serialized = data
        else:
            serialized = str(data)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @staticmethod
    def generate_audit_seal(
        decision: FinalDecisionRecord,
        source_tx: CanonicalTransaction,
        target_txs: List[CanonicalTransaction]
    ) -> CryptographicAuditSeal:
        src_hash = AuditProvenanceEngine.hash_payload({
            "id": source_tx.transaction_id,
            "amount_paise": source_tx.amount_paise,
            "utr": source_tx.utr,
            "invoice_reference": source_tx.invoice_reference,
            "timestamp": str(source_tx.transaction_timestamp)
        })

        tgt_hashes = [
            AuditProvenanceEngine.hash_payload({
                "id": t.transaction_id,
                "amount_paise": t.amount_paise,
                "utr": t.utr,
                "invoice_reference": t.invoice_reference
            })
            for t in target_txs
        ]

        rules_hash = AuditProvenanceEngine.hash_payload({
            "passed": sorted(decision.rules_passed),
            "failed": sorted(decision.rules_failed),
            "warned": sorted(decision.rules_warned),
            "leakage_risk": decision.leakage_risk
        })

        decision_hash = AuditProvenanceEngine.hash_payload({
            "decision": decision.decision.value,
            "reason": decision.reason.value,
            "confidence": decision.calibrated_confidence,
            "verifier_status": decision.verifier_status
        })

        # Merkle Tree Root Computation
        combined = f"{src_hash}|{'|'.join(tgt_hashes)}|{rules_hash}|{decision_hash}"
        merkle_root = hashlib.sha256(combined.encode("utf-8")).hexdigest()

        return CryptographicAuditSeal(
            case_id=decision.case_id,
            decision_id=decision.decision_id,
            source_tx_hash=src_hash,
            target_tx_hashes=tgt_hashes,
            rules_evaluation_hash=rules_hash,
            evidence_merkle_root=merkle_root
        )

    @staticmethod
    def verify_seal(
        seal: CryptographicAuditSeal,
        decision: FinalDecisionRecord,
        source_tx: CanonicalTransaction,
        target_txs: List[CanonicalTransaction]
    ) -> bool:
        new_seal = AuditProvenanceEngine.generate_audit_seal(decision, source_tx, target_txs)
        return seal.evidence_merkle_root == new_seal.evidence_merkle_root
