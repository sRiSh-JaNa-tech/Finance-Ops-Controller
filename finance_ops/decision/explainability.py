"""Audit Report Packaging and Human Review Formatting."""

from typing import Dict, Any, List
from finance_ops.core.models import FinalDecisionRecord, AgentRecommendation, CanonicalTransaction
from finance_ops.evidence.bundle import EvidenceBundle


class AuditReportGenerator:
    """Generates structured, verifiable markdown and JSON audit packages for financial controllers."""

    @staticmethod
    def generate_case_markdown(
        decision_record: FinalDecisionRecord,
        source_tx: CanonicalTransaction,
        bundle: EvidenceBundle
    ) -> str:
        md = []
        md.append(f"# Financial Reconciliation Audit Report: Case {decision_record.case_id}")
        md.append(f"**Decision**: `{decision_record.decision.value}` | **Reason**: `{decision_record.reason.value}`")
        md.append(f"**Calibrated Confidence**: `{decision_record.calibrated_confidence:.2%}` | **Automated**: `{decision_record.is_automated}`")
        md.append(f"**Verifier Status**: `{decision_record.verifier_status}`\n")

        md.append("## 1. Source Transaction")
        md.append(f"- **ID**: `{source_tx.transaction_id}` ({source_tx.source_system.value})")
        md.append(f"- **Amount**: `{source_tx.amount} {source_tx.currency}`")
        md.append(f"- **Date**: `{source_tx.transaction_timestamp.isoformat()}`")
        md.append(f"- **Reference**: `{source_tx.invoice_reference or 'N/A'}`")
        md.append(f"- **Narrative**: `{source_tx.raw_narrative}`\n")

        md.append("## 2. Cited Verifiable Evidence Facts")
        for fact in bundle.facts:
            icon = "❌" if fact.is_contradiction else "✅"
            md.append(f"- {icon} `[{fact.fact_id}]` **{fact.fact_type}**: {fact.claim}")

        if decision_record.verifier_notes:
            md.append("\n## 3. Verifier Notes")
            for note in decision_record.verifier_notes:
                md.append(f"- ⚠️ {note}")

        md.append("\n## 4. Explanation Narrative")
        md.append(decision_record.explanation)

        return "\n".join(md)
