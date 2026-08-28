"""Bounded Investigation Agent — 5-Stage Cognitive State Machine with Gemini Vertex AI Integration.

Grounded in:
- Liu, K. (MIT, 2025). Detecting Errors in Financial Data: A Multi-Agent LLM and Synthetic Data Approach.
- Fu, J. et al. (SIGMOD 2025). In-context Clustering-based Entity Resolution with Large Language Models.
"""

from typing import Dict, List, Any, Optional, Tuple
from decimal import Decimal
from enum import Enum

from finance_ops.core.models import (
    CanonicalTransaction, DecisionLabel, ReasonCode, AgentRecommendation, TransactionType
)
from finance_ops.evidence.bundle import EvidenceBundle, EvidenceBundleBuilder
from finance_ops.evidence.tools import InvestigationToolbox
from finance_ops.agent.stopping_policy import InvestigationStoppingPolicy
from finance_ops.agent.vertex_client import GeminiReconciliationClient


class HypothesisType(str, Enum):
    EXACT_MATCH = "EXACT_MATCH"
    FEE_ADJUSTMENT = "FEE_ADJUSTMENT"
    SPLIT_PAYMENT = "SPLIT_PAYMENT"
    DELAYED_SETTLEMENT = "DELAYED_SETTLEMENT"
    DUPLICATE = "DUPLICATE"
    REVERSAL_PAIR = "REVERSAL_PAIR"
    REVENUE_LEAKAGE = "REVENUE_LEAKAGE"
    MISSING_RECORD = "MISSING_RECORD"
    AMBIGUOUS = "AMBIGUOUS"


class AgentStage(str, Enum):
    OBSERVE = "OBSERVE"
    HYPOTHESIZE = "HYPOTHESIZE"
    SELECT_TOOL = "SELECT_TOOL"
    SYNTHESIZE = "SYNTHESIZE"
    CONVERGE = "CONVERGE"


class BoundedInvestigationAgent:
    """
    5-Stage Cognitive Investigation Agent integrating Google Gemini on Vertex AI.

    Stage 1 — OBSERVE:
        Load initial evidence bundle; categorize transaction type and identify key signals
        (amount scale, reference presence, refund flag, candidate count, leakage score).

    Stage 2 — HYPOTHESIZE:
        Formulate competing hypotheses from signal profile with explicit prior probabilities.

    Stage 3 — SELECT TOOL:
        Issue Vertex AI function call maximizing information gain and entropy reduction.

    Stage 4 — SYNTHESIZE:
        Update evidence state from tool results, refute invalidated hypotheses.

    Stage 5 — CONVERGE:
        Apply stopping policy and output structured AgentRecommendation with verified citation IDs.
    """

    def __init__(
        self,
        toolbox: InvestigationToolbox,
        max_steps: int = 5,
        enable_graph: bool = True,
        enable_rules: bool = True,
        enable_tools: bool = True,
        allow_abstention: bool = True,
        vertex_client: Optional[GeminiReconciliationClient] = None
    ):
        self.toolbox = toolbox
        self.max_steps = max_steps
        self.enable_graph = enable_graph
        self.enable_rules = enable_rules
        self.enable_tools = enable_tools
        self.allow_abstention = allow_abstention
        self.vertex_client = vertex_client or GeminiReconciliationClient()
        self.stopping_policy = InvestigationStoppingPolicy(
            max_tool_budget=max_steps
        )

    def investigate(
        self,
        source_tx: CanonicalTransaction,
        candidates: List[CanonicalTransaction],
        case_id: Optional[str] = None
    ) -> AgentRecommendation:
        """Runs the investigation agent over the candidate space."""
        cid = case_id or f"CASE_{source_tx.transaction_id}"
        
        # Fast-Path 1: Clean Exact Match (Zero Tool Call Fast Path)
        if len(candidates) == 1:
            cand = candidates[0]
            if (
                source_tx.amount_paise == cand.amount_paise
                and source_tx.amount_paise > 0
                and not source_tx.is_refund
                and not source_tx.is_reversal
                and not cand.is_refund
                and not cand.is_reversal
                and (
                    (source_tx.invoice_reference and cand.invoice_reference and source_tx.invoice_reference.upper().strip() == cand.invoice_reference.upper().strip())
                    or (source_tx.utr and cand.utr and source_tx.utr.strip() == cand.utr.strip())
                    or (source_tx.order_id and cand.order_id and source_tx.order_id.upper().strip() == cand.order_id.upper().strip())
                )
            ):
                return AgentRecommendation(
                    case_id=cid,
                    recommended_decision=DecisionLabel.MATCHED,
                    primary_reason=ReasonCode.EXACT_IDENTIFIER_MATCH,
                    cited_evidence_ids=[source_tx.transaction_id, cand.transaction_id],
                    matched_record_ids=[source_tx.transaction_id, cand.transaction_id],
                    confidence_score=0.99,
                    fuzzy_score=1.0,
                    leakage_risk=0.0,
                    rules_passed=["AC-1", "AI-1", "AI-3", "TC-1"],
                    explanation_narrative="Fast-Path Auto-Match: Exact reference and amount conservation verified.",
                    tool_calls_performed=0,
                    tool_call_sequence=[],
                    investigation_hypotheses_tested=["EXACT_MATCH"],
                    human_review_required=False,
                    investigator="deterministic-fast-path"
                )
                
            # Fast-Path 2: Fee-Adjusted Match (MDR)
            fee_diff = source_tx.amount_paise - cand.amount_paise
            if (
                fee_diff > 0 
                and (fee_diff / source_tx.amount_paise) <= 0.035
                and not source_tx.is_refund
                and (
                    (source_tx.invoice_reference and cand.invoice_reference and source_tx.invoice_reference.upper().strip() == cand.invoice_reference.upper().strip())
                )
            ):
                return AgentRecommendation(
                    case_id=cid,
                    recommended_decision=DecisionLabel.MATCHED,
                    primary_reason=ReasonCode.FEE_ADJUSTED_MATCH,
                    cited_evidence_ids=[source_tx.transaction_id, cand.transaction_id],
                    matched_record_ids=[source_tx.transaction_id, cand.transaction_id],
                    confidence_score=0.98,
                    fuzzy_score=0.95,
                    leakage_risk=0.0,
                    rules_passed=["AC-2", "AI-1", "TC-1"],
                    explanation_narrative="Fast-Path Fee Match: Net settlement adjusted for valid MDR fee.",
                    tool_calls_performed=0,
                    tool_call_sequence=[],
                    investigation_hypotheses_tested=["FEE_ADJUSTMENT"],
                    human_review_required=False,
                    investigator="deterministic-fast-path"
                )
                
        # Fast-Path 3: Split Sum Match
        if len(candidates) > 1:
            total_cand_amount = sum(c.amount_paise for c in candidates)
            if (
                source_tx.amount_paise == total_cand_amount
                and not source_tx.is_refund
                and all(not c.is_refund for c in candidates)
                and all(source_tx.invoice_reference and c.invoice_reference and source_tx.invoice_reference.upper().strip() == c.invoice_reference.upper().strip() for c in candidates)
            ):
                cand_ids = [c.transaction_id for c in candidates]
                return AgentRecommendation(
                    case_id=cid,
                    recommended_decision=DecisionLabel.MATCHED,
                    primary_reason=ReasonCode.SPLIT_PAYMENT_MATCH,
                    cited_evidence_ids=[source_tx.transaction_id] + cand_ids,
                    matched_record_ids=[source_tx.transaction_id] + cand_ids,
                    confidence_score=0.98,
                    fuzzy_score=1.0,
                    leakage_risk=0.0,
                    rules_passed=["AC-4", "AI-1"],
                    explanation_narrative="Fast-Path Split Match: Candidate tranches perfectly sum to source amount.",
                    tool_calls_performed=0,
                    tool_call_sequence=[],
                    investigation_hypotheses_tested=["SPLIT_SUM"],
                    human_review_required=False,
                    investigator="deterministic-fast-path"
                )

        # Dispatch to Gemini Vertex AI Client / Cognitive State Machine
        recommendation = self.vertex_client.investigate_case(
            case_id=cid,
            source_tx=source_tx,
            candidates=candidates,
            toolbox=self.toolbox,
            max_steps=self.max_steps
        )

        return recommendation

    def investigate_case(
        self,
        case_id: str,
        source_tx: CanonicalTransaction,
        candidates: Optional[List[CanonicalTransaction]] = None
    ) -> AgentRecommendation:
        """Helper method to investigate a case by ID."""
        if candidates is None:
            cand_res = self.toolbox.retrieve_candidates(source_tx.transaction_id)
            cands_list = cand_res.get("candidates", [])
            candidates = [
                self.toolbox.repo.get_transaction(c["transaction_id"])
                for c in cands_list
                if self.toolbox.repo.get_transaction(c["transaction_id"])
            ]
        return self.investigate(source_tx=source_tx, candidates=candidates, case_id=case_id)
