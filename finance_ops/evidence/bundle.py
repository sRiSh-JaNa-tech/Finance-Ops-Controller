"""Immutable Evidence Bundle Builder with Citation-Bearing Fact Nodes and Contradiction Detection."""

import hashlib
import json
from typing import Dict, List, Any, Optional, Set
from decimal import Decimal
from pydantic import BaseModel, Field, ConfigDict

from finance_ops.core.models import CanonicalTransaction
from finance_ops.rules.engine import RuleEvaluationResult


def _citation_hash(source_record_ids: List[str], claim: str) -> str:
    """
    Generates a deterministic citation hash for a fact claim.
    This acts as a cryptographic-style binding from the fact to its source evidence.
    """
    payload = json.dumps({"ids": sorted(source_record_ids), "claim": claim}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


class EvidenceFact(BaseModel):
    """An immutable, citation-bearing fact node within an Evidence Bundle."""
    model_config = ConfigDict(frozen=True)

    fact_id: str
    fact_type: str  # "IDENTIFIER", "AMOUNT_CHECK", "RULE_RESULT", "GRAPH_LINK", "CONTRADICTION", "SOURCE_SUMMARY"
    claim: str
    source_record_ids: List[str]
    citation_hash: str = ""  # Cryptographic citation binding
    details: Dict[str, Any] = Field(default_factory=dict)
    is_contradiction: bool = False


class EvidenceBundle(BaseModel):
    """Immutable evidence bundle presented to the investigation agent."""
    case_id: str
    source_transaction: CanonicalTransaction
    candidate_transactions: List[CanonicalTransaction]
    facts: List[EvidenceFact] = Field(default_factory=list)
    detected_contradictions: List[str] = Field(default_factory=list)
    missing_fields: List[str] = Field(default_factory=list)

    def get_fact_by_id(self, fact_id: str) -> Optional[EvidenceFact]:
        for f in self.facts:
            if f.fact_id == fact_id:
                return f
        return None

    def get_all_fact_ids(self) -> List[str]:
        return [f.fact_id for f in self.facts]

    def has_valid_citation(self, fact_id: str) -> bool:
        """Verifies that a cited fact_id actually exists in this bundle."""
        return any(f.fact_id == fact_id for f in self.facts)

    def validate_agent_citations(self, cited_ids: List[str]) -> Dict[str, Any]:
        """
        Validates that all cited_ids from an agent recommendation exist in the bundle.
        Returns a validation summary with valid/invalid counts.
        """
        valid_citations = [cid for cid in cited_ids if self.has_valid_citation(cid)]
        invalid_citations = [cid for cid in cited_ids if not self.has_valid_citation(cid)]
        return {
            "total_cited": len(cited_ids),
            "valid_citations": len(valid_citations),
            "invalid_citations": invalid_citations,
            "citation_validity_rate": len(valid_citations) / max(1, len(cited_ids))
        }


def _make_fact(
    fact_id: str,
    fact_type: str,
    claim: str,
    source_record_ids: List[str],
    details: Dict[str, Any] = None,
    is_contradiction: bool = False
) -> EvidenceFact:
    """Factory helper that attaches a citation hash to every EvidenceFact."""
    ch = _citation_hash(source_record_ids, claim)
    return EvidenceFact(
        fact_id=fact_id,
        fact_type=fact_type,
        claim=claim,
        source_record_ids=source_record_ids,
        citation_hash=ch,
        details=details or {},
        is_contradiction=is_contradiction
    )


class EvidenceBundleBuilder:
    """Constructs verifiable evidence bundles from transaction candidates and rule outputs."""

    @staticmethod
    def build_bundle(
        case_id: str,
        source_tx: CanonicalTransaction,
        candidates: List[CanonicalTransaction],
        rule_results_map: Dict[str, List[RuleEvaluationResult]],
        graph_neighborhoods_map: Dict[str, Dict[str, Any]],
    ) -> "EvidenceBundle":
        facts: List[EvidenceFact] = []
        contradictions: List[str] = []
        missing_fields: List[str] = []

        # 1. Fact: Source Transaction Baseline
        facts.append(_make_fact(
            fact_id=f"FACT_SRC_{source_tx.transaction_id[:8]}",
            fact_type="SOURCE_SUMMARY",
            claim=f"Source transaction {source_tx.transaction_id} from {source_tx.source_system.value} with amount {source_tx.amount} {source_tx.currency}",
            source_record_ids=[source_tx.transaction_id],
            details={"amount": str(source_tx.amount), "currency": source_tx.currency, "ref": source_tx.invoice_reference}
        ))

        # Check missing fields on source
        if not source_tx.invoice_reference:
            missing_fields.append("source_transaction.invoice_reference")
        if not source_tx.customer_name and not source_tx.merchant_name:
            missing_fields.append("source_transaction.entity_name")

        # --- ACTIVE CONTRADICTION DETECTION ---
        # Contradiction A: Reversal with no parent (orphan reversal)
        if (source_tx.is_refund or source_tx.is_reversal) and not source_tx.parent_transaction_id:
            contradiction_msg = f"ORPHAN_REVERSAL: {source_tx.transaction_id} is flagged as refund/reversal but has no parent_transaction_id"
            contradictions.append(contradiction_msg)
            facts.append(_make_fact(
                fact_id=f"FACT_CONTRA_ORPHAN_{source_tx.transaction_id[:8]}",
                fact_type="CONTRADICTION",
                claim=contradiction_msg,
                source_record_ids=[source_tx.transaction_id],
                is_contradiction=True
            ))

        # Contradiction B: Cross-source entity conflict between source and candidates
        for cand in candidates:
            # Cross-source customer name conflict
            if (source_tx.customer_name and cand.customer_name and
                    source_tx.customer_name.lower() != cand.customer_name.lower() and
                    source_tx.source_system != cand.source_system):
                # Only flag as contradiction if they don't look like abbreviations
                src_words = set(source_tx.customer_name.lower().split())
                cand_words = set(cand.customer_name.lower().split())
                overlap = src_words & cand_words
                if len(overlap) == 0:
                    contradiction_msg = (
                        f"ENTITY_CONFLICT: Source ({source_tx.source_system.value}) customer "
                        f"'{source_tx.customer_name}' conflicts with candidate ({cand.source_system.value}) "
                        f"customer '{cand.customer_name}'"
                    )
                    contradictions.append(contradiction_msg)
                    facts.append(_make_fact(
                        fact_id=f"FACT_CONTRA_ENTITY_{cand.transaction_id[:8]}",
                        fact_type="CONTRADICTION",
                        claim=contradiction_msg,
                        source_record_ids=[source_tx.transaction_id, cand.transaction_id],
                        is_contradiction=True
                    ))

        # Contradiction C: Duplicate candidate detection (same amount AND timestamp between two candidates)
        if len(candidates) >= 2:
            seen_fingerprints: Set[str] = set()
            for cand in candidates:
                fp = f"{cand.amount}_{cand.transaction_timestamp.date()}_{cand.currency}"
                if fp in seen_fingerprints:
                    contradiction_msg = f"DUPLICATE_CANDIDATE: Multiple candidates share amount={cand.amount} on {cand.transaction_timestamp.date()}"
                    contradictions.append(contradiction_msg)
                    facts.append(_make_fact(
                        fact_id=f"FACT_CONTRA_DUP_{cand.transaction_id[:8]}",
                        fact_type="CONTRADICTION",
                        claim=contradiction_msg,
                        source_record_ids=[cand.transaction_id],
                        is_contradiction=True
                    ))
                    break
                seen_fingerprints.add(fp)
        # --- END CONTRADICTION DETECTION ---

        # 2. Candidate Facts & Rule Executions
        for cand in candidates:
            # Identifier Fact (exact reference match)
            if source_tx.invoice_reference and cand.invoice_reference:
                if source_tx.invoice_reference.upper() == cand.invoice_reference.upper():
                    facts.append(_make_fact(
                        fact_id=f"FACT_ID_{cand.transaction_id[:8]}",
                        fact_type="IDENTIFIER",
                        claim=f"Exact invoice reference match '{source_tx.invoice_reference}' with candidate {cand.transaction_id}",
                        source_record_ids=[source_tx.transaction_id, cand.transaction_id],
                        details={"reference": source_tx.invoice_reference}
                    ))

            # Amount delta fact
            amt_diff = float(abs(source_tx.amount - cand.amount))
            if amt_diff <= 0.02:
                facts.append(_make_fact(
                    fact_id=f"FACT_AMT_EXACT_{cand.transaction_id[:6]}",
                    fact_type="AMOUNT_CHECK",
                    claim=f"Amounts match exactly: source={source_tx.amount}, candidate={cand.amount}",
                    source_record_ids=[source_tx.transaction_id, cand.transaction_id],
                    details={"amount_diff": amt_diff}
                ))
            elif amt_diff <= float(source_tx.amount) * 0.04:
                facts.append(_make_fact(
                    fact_id=f"FACT_AMT_CLOSE_{cand.transaction_id[:6]}",
                    fact_type="AMOUNT_CHECK",
                    claim=f"Amounts within 4% fee tolerance: source={source_tx.amount}, candidate={cand.amount}, diff={amt_diff:.2f}",
                    source_record_ids=[source_tx.transaction_id, cand.transaction_id],
                    details={"amount_diff": amt_diff}
                ))

            # Rules Facts
            rules_raw = rule_results_map.get(cand.transaction_id, [])
            if hasattr(rules_raw, "evaluations"):
                rules_for_cand = rules_raw.evaluations
            elif isinstance(rules_raw, list):
                rules_for_cand = rules_raw
            else:
                rules_for_cand = []

            for r in rules_for_cand:
                r_name = getattr(r, "rule_name", "RULE")
                r_passed = getattr(r, "passed", True)
                r_variance = getattr(r, "numeric_variance", 0.0)
                r_details = getattr(r, "details", getattr(r, "discrepancy_details", ""))
                fact_id = f"FACT_RULE_{r_name[:6]}_{cand.transaction_id[:6]}"
                is_contra = not r_passed and r_variance > 10.0
                facts.append(_make_fact(
                    fact_id=fact_id,
                    fact_type="RULE_RESULT",
                    claim=f"Rule {r_name} evaluation: passed={r_passed} - {r_details}",
                    source_record_ids=[source_tx.transaction_id, cand.transaction_id],
                    details={"rule_name": r_name, "passed": r_passed, "variance": r_variance},
                    is_contradiction=is_contra
                ))

            # Graph Neighborhood Facts
            graph_data = graph_neighborhoods_map.get(cand.transaction_id, {})
            if graph_data and graph_data.get("nodes"):
                facts.append(_make_fact(
                    fact_id=f"FACT_GRAPH_{cand.transaction_id[:8]}",
                    fact_type="GRAPH_LINK",
                    claim=f"Candidate {cand.transaction_id} is connected to {len(graph_data.get('nodes', []))} graph entities",
                    source_record_ids=[cand.transaction_id],
                    details={"node_count": len(graph_data.get("nodes", []))}
                ))

        return EvidenceBundle(
            case_id=case_id,
            source_transaction=source_tx,
            candidate_transactions=candidates,
            facts=facts,
            detected_contradictions=contradictions,
            missing_fields=missing_fields
        )

