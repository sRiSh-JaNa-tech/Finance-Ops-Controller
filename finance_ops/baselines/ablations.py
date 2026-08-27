"""
Ablation Experiment Harness — Isolates the contribution of each Prototype-2 component.

Ablations defined in the plan:
  1. No Graph Context         — strip graph features from retrieval and evidence
  2. No Embeddings            — restrict to lexical+exact blocking only
  3. No Financial Rules       — mask rule engine outputs
  4. No Additional Tools      — single-pass; no iterative tool calls after first retrieval
  5. Forced Classification    — argmax over {MATCHED, EXCEPTION}, no UNCERTAIN
  6. One-Pass Evidence        — agent replaced by static evidence summarizer (no hypothesis loop)
  7. No Structured Validation — bypass DeterministicPolicyVerifier
  8. Varying Tool Budgets     — sweep K ∈ {1, 2, 3, 5, 10}
"""

from enum import Enum
from typing import Dict, Any, List, Optional
from decimal import Decimal

from finance_ops.core.models import DecisionLabel, ReasonCode, CanonicalTransaction
from finance_ops.evidence.bundle import EvidenceBundleBuilder
from finance_ops.evidence.tools import InvestigationToolbox
from finance_ops.decision.verifier import DeterministicPolicyVerifier
from finance_ops.decision.calibration import ConfidenceCalibrator
from finance_ops.retrieval.blocking import CandidateBlockingEngine
from finance_ops.rules.engine import FinancialRuleEngine


class AblationMode(str, Enum):
    FULL_PROTOTYPE_2 = "FULL_PROTOTYPE_2"
    NO_GRAPH = "NO_GRAPH"
    NO_FINANCIAL_RULES = "NO_FINANCIAL_RULES"
    NO_ADDITIONAL_TOOLS = "NO_ADDITIONAL_TOOLS"
    FORCE_NO_ABSTENTION = "FORCE_NO_ABSTENTION"
    NO_POLICY_VERIFIER = "NO_POLICY_VERIFIER"
    ONE_PASS_SUMMARIZER = "ONE_PASS_SUMMARIZER"
    BUDGET_SWEEP = "BUDGET_SWEEP"


class AblationConfig:
    """Configures which Prototype-2 components are enabled for a given ablation run."""

    def __init__(
        self,
        mode: AblationMode = AblationMode.FULL_PROTOTYPE_2,
        enable_graph: bool = True,
        enable_rules: bool = True,
        enable_tools: bool = True,
        allow_abstention: bool = True,
        enable_verifier: bool = True,
        max_tool_budget: int = 5
    ):
        self.mode = mode
        self.enable_graph = enable_graph if mode != AblationMode.NO_GRAPH else False
        self.enable_rules = enable_rules if mode != AblationMode.NO_FINANCIAL_RULES else False
        self.enable_tools = enable_tools if mode != AblationMode.NO_ADDITIONAL_TOOLS else False
        self.allow_abstention = allow_abstention if mode != AblationMode.FORCE_NO_ABSTENTION else False
        self.enable_verifier = enable_verifier if mode != AblationMode.NO_POLICY_VERIFIER else False
        self.max_tool_budget = max_tool_budget


# ────────────────────────────────────────────────────────────────────────────
# One-Pass Evidence Summarizer (Ablation 6)
# ────────────────────────────────────────────────────────────────────────────

class OnePassEvidenceSummarizer:
    """
    Ablation 6: replaces the iterative agent with a single-pass deterministic
    evidence summarizer. No hypothesis loop, no tool calls beyond initial retrieval.
    Decision is made purely from initial evidence bundle + rule scores.
    """

    def __init__(self, toolbox: InvestigationToolbox):
        self.toolbox = toolbox

    def investigate_case(self, case_id: str, source_tx: CanonicalTransaction):
        """Single-pass: retrieve candidates, run rules, decide immediately."""
        from finance_ops.core.models import AgentRecommendation
        from finance_ops.retrieval.similarity import calculate_candidate_similarity

        cands_resp = self.toolbox.retrieve_candidates(source_tx.transaction_id, max_candidates=5)
        candidates = [
            self.toolbox.repo.get_transaction(c["transaction_id"])
            for c in cands_resp.get("candidates", [])
            if self.toolbox.repo.get_transaction(c["transaction_id"])
        ]

        if not candidates:
            return AgentRecommendation(
                case_id=case_id,
                recommended_decision=DecisionLabel.UNCERTAIN,
                primary_reason=ReasonCode.MISSING_SOURCE_RECORD,
                confidence_score=0.40,
                tool_calls_performed=1
            )

        # One-shot scoring: composite similarity on best candidate
        best_cand = None
        best_score = 0.0
        for c in candidates:
            sim = calculate_candidate_similarity(source_tx, c)
            if sim["composite_score"] > best_score:
                best_score = sim["composite_score"]
                best_cand = c

        # Single-threshold decision
        if best_score >= 0.85:
            return AgentRecommendation(
                case_id=case_id,
                recommended_decision=DecisionLabel.MATCHED,
                primary_reason=ReasonCode.FUZZY_ENTITY_MATCH,
                matched_record_ids=[best_cand.transaction_id],
                confidence_score=best_score,
                tool_calls_performed=1
            )

        return AgentRecommendation(
            case_id=case_id,
            recommended_decision=DecisionLabel.UNCERTAIN,
            primary_reason=ReasonCode.BELOW_CONFIDENCE_THRESHOLD,
            confidence_score=best_score,
            tool_calls_performed=1
        )


# ────────────────────────────────────────────────────────────────────────────
# Ablation Runner
# ────────────────────────────────────────────────────────────────────────────

class AblationRunner:
    """
    Executes all defined ablation experiments on a shared benchmark dataset,
    ensuring identical candidate pools and evidence across all variants.

    Architecture guarantee: all ablations receive the same candidate set as
    the full Prototype-2 so that component removal is isolated.
    """

    # Canonical set of ablation configurations
    ABLATIONS: List[Dict[str, Any]] = [
        {"name": "Proto2_Full",           "mode": AblationMode.FULL_PROTOTYPE_2, "budget": 5},
        {"name": "Ablation1_NoGraph",     "mode": AblationMode.NO_GRAPH,         "budget": 5},
        {"name": "Ablation3_NoRules",     "mode": AblationMode.NO_FINANCIAL_RULES, "budget": 5},
        {"name": "Ablation4_NoTools",     "mode": AblationMode.NO_ADDITIONAL_TOOLS, "budget": 1},
        {"name": "Ablation5_NoAbstain",   "mode": AblationMode.FORCE_NO_ABSTENTION, "budget": 5},
        {"name": "Ablation6_OnePass",     "mode": AblationMode.ONE_PASS_SUMMARIZER, "budget": 1},
        {"name": "Ablation7_NoVerifier",  "mode": AblationMode.NO_POLICY_VERIFIER, "budget": 5},
        {"name": "Ablation8_Budget1",     "mode": AblationMode.BUDGET_SWEEP,     "budget": 1},
        {"name": "Ablation8_Budget2",     "mode": AblationMode.BUDGET_SWEEP,     "budget": 2},
        {"name": "Ablation8_Budget3",     "mode": AblationMode.BUDGET_SWEEP,     "budget": 3},
        {"name": "Ablation8_Budget5",     "mode": AblationMode.BUDGET_SWEEP,     "budget": 5},
        {"name": "Ablation8_Budget10",    "mode": AblationMode.BUDGET_SWEEP,     "budget": 10},
    ]

    def __init__(
        self,
        toolbox: InvestigationToolbox,
        verifier: DeterministicPolicyVerifier,
        calibrator: ConfidenceCalibrator
    ):
        self.toolbox = toolbox
        self.verifier = verifier
        self.calibrator = calibrator

    def _build_agent(self, config: AblationConfig):
        """Constructs an agent variant based on the ablation configuration."""
        if config.mode == AblationMode.ONE_PASS_SUMMARIZER:
            return OnePassEvidenceSummarizer(self.toolbox)

        from finance_ops.agent.investigator import BoundedInvestigationAgent
        return BoundedInvestigationAgent(
            toolbox=self.toolbox,
            max_steps=config.max_tool_budget,
            enable_graph=config.enable_graph,
            enable_rules=config.enable_rules,
            enable_tools=config.enable_tools,
            allow_abstention=config.allow_abstention,
        )

    def run_case(
        self,
        ablation_name: str,
        config: AblationConfig,
        case_id: str,
        source_tx: CanonicalTransaction
    ) -> Dict[str, Any]:
        """Runs a single case under a given ablation configuration."""
        agent = self._build_agent(config)
        rec = agent.investigate_case(case_id, source_tx)

        cal_conf = self.calibrator.calibrate(rec.confidence_score)

        if config.enable_verifier:
            final_dec = self.verifier.verify_and_finalize(rec, source_tx, cal_conf)
        else:
            # Bypass verifier: accept recommendation directly
            from finance_ops.core.models import FinalDecisionRecord
            import uuid
            from datetime import datetime
            final_dec = FinalDecisionRecord(
                decision_id=f"ABL_{uuid.uuid4().hex[:8]}",
                case_id=case_id,
                decision=rec.recommended_decision,
                reason=rec.primary_reason,
                calibrated_confidence=cal_conf,
                is_automated=rec.recommended_decision != DecisionLabel.UNCERTAIN,
                requires_human_review=rec.recommended_decision == DecisionLabel.UNCERTAIN,
                matched_pairs=[{"source": source_tx.transaction_id, "target": tid} for tid in rec.matched_record_ids],
                source_record_ids=[source_tx.transaction_id],
                cited_evidence_ids=rec.cited_evidence_ids,
                verifier_status="VERIFIER_BYPASSED",
                explanation=rec.explanation_narrative
            )

        return {
            "ablation": ablation_name,
            "case_id": case_id,
            "decision": final_dec.decision,
            "reason": final_dec.reason,
            "matched_record_ids": [p["target"] for p in final_dec.matched_pairs],
            "calibrated_confidence": cal_conf,
            "tool_calls": rec.tool_calls_performed,
        }

    def run_all(
        self,
        ground_truth_cases: List[Dict[str, Any]],
        repo_get_fn  # callable: (tx_id: str) -> Optional[CanonicalTransaction]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Runs all ablation configurations over the ground_truth_cases.
        Returns a dict mapping ablation_name → list of case result dicts.
        """
        results: Dict[str, List[Dict[str, Any]]] = {}

        for abl_spec in self.ABLATIONS:
            name = abl_spec["name"]
            config = AblationConfig(
                mode=abl_spec["mode"],
                enable_graph=(abl_spec["mode"] != AblationMode.NO_GRAPH),
                enable_rules=(abl_spec["mode"] != AblationMode.NO_FINANCIAL_RULES),
                enable_tools=(abl_spec["mode"] not in [AblationMode.NO_ADDITIONAL_TOOLS, AblationMode.ONE_PASS_SUMMARIZER]),
                allow_abstention=(abl_spec["mode"] != AblationMode.FORCE_NO_ABSTENTION),
                enable_verifier=(abl_spec["mode"] != AblationMode.NO_POLICY_VERIFIER),
                max_tool_budget=abl_spec["budget"]
            )

            case_results = []
            for gt in ground_truth_cases:
                src_tx = repo_get_fn(gt["source_record_id"])
                if not src_tx:
                    continue
                case_results.append(self.run_case(name, config, gt["case_id"], src_tx))

            results[name] = case_results

        return results
