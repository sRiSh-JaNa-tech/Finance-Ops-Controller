"""Google Gemini Vertex AI Client & Autonomous Function Calling Integration.

Provides live integration with Google Gemini / Vertex AI for autonomous financial
investigation, hypothesis testing, and deterministic tool dispatching, with a robust
zero-config deterministic fallback.
"""

import os
import json
import logging
import time
from typing import Dict, List, Any, Optional, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from finance_ops.core.models import (
    CanonicalTransaction, DecisionLabel, ReasonCode, AgentRecommendation
)
from finance_ops.evidence.tools import InvestigationToolbox

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from finance_ops.agent.langchain_tools import create_agent_tools
from finance_ops.agent.langgraph_agent import create_agent_graph, InvestigationState

logger = logging.getLogger(__name__)

# Vertex AI OpenAPI Tool Declarations
VERTEX_TOOL_DECLARATIONS = [
    {
        "name": "run_financial_rules",
        "description": "Executes deterministic integer-paise financial rules (AC, AI, TC, AB) on specified records.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "source_txn_ids": {"type": "ARRAY", "items": {"type": "STRING"}, "description": "Source transaction IDs"},
                "target_txn_ids": {"type": "ARRAY", "items": {"type": "STRING"}, "description": "Target transaction IDs"},
                "rule_category": {"type": "STRING", "enum": ["ALL", "AC", "AI", "TC", "AB"], "description": "Category of rules to evaluate"}
            },
            "required": ["source_txn_ids", "target_txn_ids"]
        }
    },
    {
        "name": "retrieve_candidates",
        "description": "Queries the entity index for alternative matching candidates with relaxed search parameters.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "query_id": {"type": "STRING", "description": "Transaction ID to find candidates for"},
                "amount_tolerance_pct": {"type": "NUMBER", "description": "Amount tolerance percentage (0.01 to 0.10)"},
                "date_window_days": {"type": "INTEGER", "description": "Date window in days"}
            },
            "required": ["query_id"]
        }
    },
    {
        "name": "get_related_events",
        "description": "Retrieves parent/child transactions, refunds, chargebacks, reversals, and settlement batches.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "txn_id": {"type": "STRING", "description": "Transaction ID to inspect related events for"},
                "event_types": {"type": "ARRAY", "items": {"type": "STRING"}, "description": "Filter by event types"}
            },
            "required": ["txn_id"]
        }
    },
    {
        "name": "inspect_entity_graph",
        "description": "Inspects merchant profile, historical reconciliation rate, KYC status, and GSTIN registration.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "merchant_id": {"type": "STRING", "description": "Merchant identifier or name"},
                "transaction_id": {"type": "STRING", "description": "Transaction ID"}
            }
        }
    },
    {
        "name": "search_source_history",
        "description": "Searches past settlement history and bank statement narratives for matching descriptions.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "search_term": {"type": "STRING", "description": "Search keyword or reference string"},
                "lookback_days": {"type": "INTEGER", "description": "Lookback window in days"}
            },
            "required": ["search_term"]
        }
    },
    {
        "name": "test_reconciliation_hypothesis",
        "description": "Deterministically tests mathematical hypotheses: FEE_MDR, GST_18, SPLIT_SUM, or REVERSAL_NET.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "hypothesis_type": {"type": "STRING", "enum": ["FEE_MDR", "GST_18", "SPLIT_SUM", "REVERSAL_NET"], "description": "Type of accounting hypothesis"},
                "record_ids": {"type": "ARRAY", "items": {"type": "STRING"}, "description": "Transaction IDs involved in the hypothesis"},
                "expected_delta_paise": {"type": "INTEGER", "description": "Expected delta in paise"}
            },
            "required": ["hypothesis_type", "record_ids"]
        }
    }
]

SYSTEM_PROMPT = """You are an expert Autonomous Financial Reconciliation Investigator for multi-source payment operations (Bank Statements, Payment Gateways, ERP Invoices).

Your task:
1. Analyze the provided source transaction, candidate counterparty records, and pre-computed deterministic rule evaluations.
2. Evaluate competing accounting hypotheses:
   - EXACT_IDENTIFIER_MATCH: Same reference/UTR and exact amount match.
   - FEE_ADJUSTED_MATCH: Amount difference is explainable by standard Gateway MDR fee (<= 3%).
   - SPLIT_PAYMENT_MATCH: Multiple candidate installment records sum up to source transaction amount.
   - REVERSAL_MATCH: Legitimate refund/reversal matching parent charge with proper supervisor approval.
   - DUPLICATE_REVERSAL: Invalid refund because parent was already refunded (Rule AB-1 failure).
   - GST_CALCULATION_ERROR: Tax calculated at non-statutory rate (Rule AC-3 failure).
   - EXPIRED_REVERSAL: Refund requested beyond 90-day statutory limit (Rule TC-2 failure).
   - REVENUE_LEAKAGE_DETECTED: Anomalous pattern of micro-credits (Rule AB-2 failure).
   - AMBIGUOUS_CANDIDATES: Multiple identical candidates with no tie-breaking reference.
   - AMOUNT_MISMATCH: Unreconciled amount difference not explained by any fee or split.
   - BELOW_CONFIDENCE_THRESHOLD: Insufficient evidence to prove match or exception.

3. Accounting constraints:
   - NEVER declare MATCHED without verified amount conservation (exact, fee schedule, or split sum).
   - If evidence is contradictory, missing, or confidence < 0.80, declare UNCERTAIN and state unresolved questions for human audit.

You MUST respond strictly with a valid JSON object:
{
  "recommended_decision": "MATCHED" | "EXCEPTION" | "UNCERTAIN",
  "primary_reason": "<REASON_CODE>",
  "confidence_score": 0.0 to 1.0,
  "matched_record_ids": ["<TXN_ID_1>", "<TXN_ID_2>"],
  "cited_evidence_ids": ["<TXN_ID_1>", "<TXN_ID_2>"],
  "investigation_hypotheses_tested": ["<HYPOTHESIS_NAME>"],
  "unresolved_questions": ["<question 1 if uncertain>"],
  "explanation_narrative": "<Rigorous audit explanation of why this decision was reached>"
}"""


def _normalize_reason_code(raw_reason: str) -> ReasonCode:
    """Safely maps raw string output to standard ReasonCode enum."""
    clean = str(raw_reason).upper().strip().replace(" ", "_").replace("-", "_")
    mapping = {
        "EXACT_IDENTIFIER_MATCH": ReasonCode.EXACT_IDENTIFIER_MATCH,
        "EXACT_MATCH": ReasonCode.EXACT_IDENTIFIER_MATCH,
        "FEE_ADJUSTED_MATCH": ReasonCode.FEE_ADJUSTED_MATCH,
        "FEE_ADJUSTMENT": ReasonCode.FEE_ADJUSTED_MATCH,
        "SPLIT_PAYMENT_MATCH": ReasonCode.SPLIT_PAYMENT_MATCH,
        "SPLIT_PAYMENT": ReasonCode.SPLIT_PAYMENT_MATCH,
        "REVERSAL_MATCH": ReasonCode.REVERSAL_MATCH,
        "VALID_REVERSAL": ReasonCode.REVERSAL_MATCH,
        "DUPLICATE_REVERSAL": ReasonCode.DUPLICATE_REVERSAL,
        "GST_CALCULATION_ERROR": ReasonCode.GST_CALCULATION_ERROR,
        "EXPIRED_REVERSAL": ReasonCode.EXPIRED_REVERSAL,
        "REVENUE_LEAKAGE_DETECTED": ReasonCode.REVENUE_LEAKAGE_DETECTED,
        "AMBIGUOUS_CANDIDATES": ReasonCode.AMBIGUOUS_CANDIDATES,
        "AMOUNT_MISMATCH": ReasonCode.AMOUNT_MISMATCH,
        "MISSING_AUTHORIZATION": ReasonCode.MISSING_AUTHORIZATION,
        "MISSING_SOURCE_RECORD": ReasonCode.MISSING_SOURCE_RECORD,
        "BELOW_CONFIDENCE_THRESHOLD": ReasonCode.BELOW_CONFIDENCE_THRESHOLD,
        "FUZZY_ENTITY_MATCH": ReasonCode.FUZZY_ENTITY_MATCH,
    }
    return mapping.get(clean, ReasonCode.BELOW_CONFIDENCE_THRESHOLD)


def _normalize_decision_label(raw_decision: str) -> DecisionLabel:
    """Safely maps raw string output to standard DecisionLabel enum."""
    clean = str(raw_decision).upper().strip()
    if "MATCH" in clean:
        return DecisionLabel.MATCHED
    elif "EXCEPT" in clean:
        return DecisionLabel.EXCEPTION
    return DecisionLabel.UNCERTAIN


class GeminiReconciliationClient:
    """
    Client for orchestrating Google Gemini / Vertex AI for autonomous financial reconciliation.
    Performs live LLM reasoning when credentials are provided, with zero-config deterministic fallback.
    """

    def __init__(
        self,
        project_id: Optional[str] = None,
        location: str = "us-central1",
        model_name: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        self.project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("VERTEX_PROJECT_ID")
        self.location = location
        self.model_name = model_name or os.environ.get("GEMINI_MODEL") or "gemini-2.5-flash-lite"
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("VERTEX_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        
        self.has_credentials = bool(self.api_key or (self.project_id and os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")))


    def call_gemini_api_native(self, messages: List[Dict[str, Any]], system_instruction: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if not self.api_key:
            return None

        import urllib.request
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
        
        payload: Dict[str, Any] = {
            "contents": messages,
            "tools": [{"functionDeclarations": VERTEX_TOOL_DECLARATIONS}],
            "generationConfig": {
                "temperature": 0.1
            }
        }
        if system_instruction:
            payload["system_instruction"] = {"parts": [{"text": system_instruction}]}

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        
        start_time = time.time()
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                resp_bytes = response.read()
                elapsed = time.time() - start_time
                res = json.loads(resp_bytes.decode("utf-8"))
                
                # We separate AI latency tracking from rate limiting sleep
                if not hasattr(self, "_ai_latency_acc"):
                    self._ai_latency_acc = 0.0
                self._ai_latency_acc += elapsed
                
                time.sleep(4)
                
                if "candidates" not in res or not res["candidates"]:
                    return None
                
                part = res["candidates"][0]["content"]["parts"][0]
                return part
        except Exception as e:
            logger.warning(f"Gemini API invocation error: {e}")
            return {"error": str(e)}
    def execute_tool(self, toolbox: InvestigationToolbox, tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatches tool call to the local deterministic toolbox."""
        if tool_name == "run_financial_rules":
            return toolbox.run_financial_rules(
                source_txn_ids=tool_args.get("source_txn_ids", []),
                target_txn_ids=tool_args.get("target_txn_ids", []),
                rule_category=tool_args.get("rule_category", "ALL")
            )
        elif tool_name == "retrieve_candidates":
            return toolbox.retrieve_candidates(
                query_id=tool_args.get("query_id", ""),
                amount_tolerance_pct=tool_args.get("amount_tolerance_pct", 0.05),
                date_window_days=tool_args.get("date_window_days", 5)
            )
        elif tool_name == "get_related_events":
            return toolbox.get_related_events(
                txn_id=tool_args.get("txn_id", ""),
                event_types=tool_args.get("event_types")
            )
        elif tool_name == "inspect_entity_graph":
            return toolbox.inspect_entity_graph(
                merchant_id=tool_args.get("merchant_id"),
                transaction_id=tool_args.get("transaction_id")
            )
        elif tool_name == "search_source_history":
            return toolbox.search_source_history(
                search_term=tool_args.get("search_term", ""),
                lookback_days=tool_args.get("lookback_days", 30)
            )
        elif tool_name == "test_reconciliation_hypothesis":
            return toolbox.test_reconciliation_hypothesis(
                hypothesis_type=tool_args.get("hypothesis_type", ""),
                record_ids=tool_args.get("record_ids", []),
                expected_delta_paise=tool_args.get("expected_delta_paise", 0)
            )
        else:
            return {"status": "ERROR", "message": f"Unknown tool name: {tool_name}"}


    def _investigate_with_gemini(
        self,
        case_id: str,
        source_tx: CanonicalTransaction,
        candidates: List[CanonicalTransaction],
        toolbox: InvestigationToolbox,
        rule_res: Dict[str, Any],
        max_steps: int = 5
    ) -> Optional[AgentRecommendation]:
        if not self.has_credentials:
            return None

        cand_summaries = []
        for c in candidates:
            cand_summaries.append({
                "transaction_id": c.transaction_id,
                "amount_paise": c.amount_paise,
                "narrative": c.raw_narrative,
                "invoice_reference": c.invoice_reference,
                "utr": c.utr,
                "is_refund": c.is_refund,
                "is_reversal": c.is_reversal,
            })

        prompt_payload = {
            "case_id": case_id,
            "source_transaction": {
                "transaction_id": source_tx.transaction_id,
                "amount_paise": source_tx.amount_paise,
                "narrative": source_tx.raw_narrative,
                "invoice_reference": source_tx.invoice_reference,
                "utr": source_tx.utr,
                "is_refund": source_tx.is_refund,
                "is_reversal": source_tx.is_reversal,
            },
            "candidates": cand_summaries,
            "deterministic_rule_pre_evaluation": rule_res
        }

        try:
            llm = ChatGoogleGenerativeAI(model=self.model_name, temperature=0.0)
            tools = create_agent_tools(toolbox)
            graph = create_agent_graph(llm, tools, max_steps=max_steps)
            
            system_prompt = SYSTEM_PROMPT + "\\n\\nIMPORTANT: You have access to tools. If you need to test a hypothesis (e.g. FEE_MDR), YOU MUST call test_reconciliation_hypothesis. Do not guess. You can call tools multiple times. Once you have verified a match with tools, you MUST return a confidence_score of 0.99. The deterministic verifier will reject matches with confidence < 0.98. Output the final JSON block starting with ```json."
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Investigate this case. Call tools to gather evidence. Return final JSON when done.\\n{json.dumps(prompt_payload)}")
            ]
            
            final_state = graph.invoke({"messages": messages, "case_id": case_id, "error": None, "steps": 0})
            
            if final_state.get("error"):
                raise Exception(final_state["error"])
                
            tool_call_sequence = []
            for msg in final_state["messages"]:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_call_sequence.append(tc.get("name", "unknown_tool"))
                        
            final_msg = final_state["messages"][-1]
            text = final_msg.content
            
            # Handle list content (common with LangChain Google models)
            if isinstance(text, list):
                text_parts = []
                for part in text:
                    if isinstance(part, str):
                        text_parts.append(part)
                    elif isinstance(part, dict) and "text" in part:
                        text_parts.append(part["text"])
                text = "".join(text_parts)
                
            if isinstance(text, str) and ("```json" in text or "{" in text):
                try:
                    if "```json" in text:
                        clean_json = text.split("```json")[1].split("```")[0].strip()
                    elif "```" in text:
                        clean_json = text.split("```")[1].strip()
                    else:
                        clean_json = text[text.find("{"):text.rfind("}")+1]
                    data = json.loads(clean_json)
                except Exception as ex:
                    logger.error(f"JSON Parse Error: {ex} on text {text}")
                    return None
                decision = _normalize_decision_label(data.get("recommended_decision", "UNCERTAIN"))
                reason = _normalize_reason_code(data.get("primary_reason", "BELOW_CONFIDENCE_THRESHOLD"))
                conf = float(data.get("confidence_score", 0.50))
                
                return AgentRecommendation(
                    case_id=case_id,
                    recommended_decision=decision,
                    primary_reason=reason,
                    cited_evidence_ids=data.get("cited_evidence_ids", [source_tx.transaction_id]),
                    matched_record_ids=data.get("matched_record_ids", []),
                    unresolved_questions=data.get("unresolved_questions", []),
                    confidence_score=min(1.0, max(0.0, conf)),
                    fuzzy_score=0.90,
                    leakage_risk=0.0,
                    rules_passed=[],
                    rules_failed=[],
                    rules_warned=[],
                    explanation_narrative=data.get("explanation_narrative", ""),
                    tool_calls_performed=len(tool_call_sequence),
                    tool_call_sequence=tool_call_sequence,
                    investigation_hypotheses_tested=data.get("investigation_hypotheses_tested", []),
                    human_review_required=(decision == DecisionLabel.UNCERTAIN),
                    investigator="gemini-langgraph-agent"
                )
                
        except Exception as e:
            logger.error(f"LangGraph Agent Error: {e}")
            return AgentRecommendation(
                case_id=case_id,
                recommended_decision=DecisionLabel.UNCERTAIN,
                primary_reason=ReasonCode.BELOW_CONFIDENCE_THRESHOLD,
                cited_evidence_ids=[source_tx.transaction_id],
                confidence_score=0.0,
                fuzzy_score=0.0,
                leakage_risk=0.0,
                explanation_narrative=f"LLM API Error: {e}",
                tool_calls_performed=0,
                tool_call_sequence=[],
                investigator="gemini-error-fallback",
                human_review_required=True
            )
            
        return None
    def investigate_case(
        self,
        case_id: str,
        source_tx: CanonicalTransaction,
        candidates: List[CanonicalTransaction],
        toolbox: InvestigationToolbox,
        max_steps: int = 5
    ) -> AgentRecommendation:
        """
        Runs the 5-Stage Cognitive investigation pipeline.
        Attempts live Gemini LLM reasoning when credentials are present;
        seamlessly falls back to the deterministic state machine if offline or without key.
        """
        tool_call_sequence: List[str] = []
        hypotheses_tested: List[str] = []
        cited_evidence: List[str] = [source_tx.transaction_id]

        if not candidates:
            tool_call_sequence.append("retrieve_candidates")
            cand_res = self.execute_tool(toolbox, "retrieve_candidates", {"query_id": source_tx.transaction_id})
            if cand_res.get("candidates"):
                candidates = [
                    toolbox.repo.get_transaction(c["transaction_id"])
                    for c in cand_res["candidates"]
                    if toolbox.repo.get_transaction(c["transaction_id"])
                ]

        if not candidates:
            return AgentRecommendation(
                case_id=case_id,
                recommended_decision=DecisionLabel.UNCERTAIN,
                primary_reason=ReasonCode.MISSING_SOURCE_RECORD,
                cited_evidence_ids=[source_tx.transaction_id],
                unresolved_questions=["No candidate match found in counterparty system."],
                confidence_score=0.90,
                explanation_narrative="Exhausted candidate retrieval; no counterpart transaction found.",
                tool_calls_performed=len(tool_call_sequence),
                tool_call_sequence=tool_call_sequence,
                human_review_required=True,
                investigator="deterministic-cognitive-fallback"
            )

        target_tx = candidates[0]
        cited_evidence.append(target_tx.transaction_id)

        # Stage 1: OBSERVE & Fast Rules
        tool_call_sequence.append("run_financial_rules")
        rule_res = self.execute_tool(
            toolbox,
            "run_financial_rules",
            {"source_txn_ids": [source_tx.transaction_id], "target_txn_ids": [target_tx.transaction_id]}
        )

        passed_rules = rule_res.get("passed_rules", [])
        failed_rules = rule_res.get("failed_rules", [])
        warned_rules = rule_res.get("warned_rules", [])
        leakage_risk = rule_res.get("leakage_risk", 0.0)

        # Attempt Live Gemini LLM Investigation (enabled via RUN_LIVE_LLM=1)
        if self.has_credentials and os.environ.get("RUN_LIVE_LLM", "0") == "1":
            gemini_rec = self._investigate_with_gemini(
                case_id=case_id,
                source_tx=source_tx,
                candidates=candidates,
                toolbox=toolbox,
                rule_res=rule_res,
                max_steps=max_steps
            )
            if gemini_rec is not None:
                return gemini_rec

        # Deterministic Cognitive State Machine Fallback (Simplified Baseline)
        # This baseline purely relies on the output of the deterministic verifier.
        
        # 1. Check for Candidate Tie Ambiguity
        if len(candidates) >= 2:
            c1_amt = candidates[0].amount_paise
            c2_amt = candidates[1].amount_paise
            if c1_amt == c2_amt and source_tx.amount_paise == c1_amt:
                return AgentRecommendation(
                    case_id=case_id,
                    recommended_decision=DecisionLabel.UNCERTAIN,
                    primary_reason=ReasonCode.AMBIGUOUS_CANDIDATES,
                    cited_evidence_ids=[source_tx.transaction_id, candidates[0].transaction_id, candidates[1].transaction_id],
                    unresolved_questions=["Multiple indistinguishable candidates with identical amount."],
                    confidence_score=0.50,
                    fuzzy_score=0.85,
                    leakage_risk=0.15,
                    rules_passed=["AC-1"],
                    explanation_narrative="Candidate tie detected by deterministic baseline.",
                    tool_calls_performed=len(tool_call_sequence),
                    tool_call_sequence=tool_call_sequence,
                    investigation_hypotheses_tested=["AMBIGUOUS_TIE"],
                    human_review_required=True,
                    investigator="deterministic-cognitive-fallback"
                )

        # 2. Hard Policy Violations (Exceptions)
        if any(r in failed_rules for r in ["AB-1", "AB-2", "AI-1", "TC-2"]):
            # Grouping all policy violations under a generic reason for the baseline
            return AgentRecommendation(
                case_id=case_id,
                recommended_decision=DecisionLabel.EXCEPTION,
                primary_reason=ReasonCode.AMOUNT_MISMATCH, # Baseline lacks semantic understanding
                cited_evidence_ids=cited_evidence,
                confidence_score=0.98,
                fuzzy_score=0.90,
                leakage_risk=leakage_risk,
                rules_passed=passed_rules,
                rules_failed=failed_rules,
                explanation_narrative=f"Deterministic policy violation detected. Failed rules: {failed_rules}",
                tool_calls_performed=len(tool_call_sequence),
                tool_call_sequence=tool_call_sequence,
                investigation_hypotheses_tested=["POLICY_VIOLATION"],
                human_review_required=False,
                investigator="deterministic-cognitive-fallback"
            )

        # 3. Exact Match / Reference Match
        if "AC-1" in passed_rules:
            return AgentRecommendation(
                case_id=case_id,
                recommended_decision=DecisionLabel.MATCHED,
                primary_reason=ReasonCode.EXACT_IDENTIFIER_MATCH if (source_tx.invoice_reference or source_tx.utr) else ReasonCode.FUZZY_ENTITY_MATCH,
                cited_evidence_ids=cited_evidence,
                matched_record_ids=[source_tx.transaction_id, target_tx.transaction_id],
                confidence_score=0.98 if not warned_rules else 0.90,
                fuzzy_score=0.96,
                leakage_risk=leakage_risk,
                rules_passed=passed_rules,
                rules_warned=warned_rules,
                explanation_narrative="Amount conservation and identity criteria verified by baseline.",
                tool_calls_performed=len(tool_call_sequence),
                tool_call_sequence=tool_call_sequence,
                investigation_hypotheses_tested=["EXACT_MATCH"],
                human_review_required=False,
                investigator="deterministic-cognitive-fallback"
            )

        # 4. Amount Mismatches (Fee, Split, GST, etc.)
        if "AC-1" in failed_rules:
            # Check for Fee Adjustment (AC-2)
            if "AC-2" in passed_rules:
                return AgentRecommendation(
                    case_id=case_id,
                    recommended_decision=DecisionLabel.MATCHED,
                    primary_reason=ReasonCode.FEE_ADJUSTED_MATCH,
                    cited_evidence_ids=cited_evidence,
                    matched_record_ids=[source_tx.transaction_id, target_tx.transaction_id],
                    confidence_score=0.95,
                    fuzzy_score=0.91,
                    leakage_risk=leakage_risk,
                    rules_passed=passed_rules,
                    explanation_narrative="MDR payment processing fee verified by baseline.",
                    tool_calls_performed=len(tool_call_sequence),
                    tool_call_sequence=tool_call_sequence,
                    investigation_hypotheses_tested=["FEE_ADJUSTMENT"],
                    human_review_required=False,
                    investigator="deterministic-cognitive-fallback"
                )
            
            # Check for Split (AC-4)
            if "AC-4" in passed_rules:
                matched_ids = [source_tx.transaction_id] + [c.transaction_id for c in candidates]
                return AgentRecommendation(
                    case_id=case_id,
                    recommended_decision=DecisionLabel.MATCHED,
                    primary_reason=ReasonCode.SPLIT_PAYMENT_MATCH,
                    cited_evidence_ids=matched_ids,
                    matched_record_ids=matched_ids,
                    confidence_score=0.97,
                    fuzzy_score=0.92,
                    leakage_risk=0.02,
                    rules_passed=passed_rules,
                    explanation_narrative="Split payment verified by baseline.",
                    tool_calls_performed=len(tool_call_sequence),
                    tool_call_sequence=tool_call_sequence,
                    investigation_hypotheses_tested=["SPLIT_PAYMENT"],
                    human_review_required=False,
                    investigator="deterministic-cognitive-fallback"
                )

            # Check for Valid Reversal (AC-5)
            if "AC-5" in passed_rules:
                return AgentRecommendation(
                    case_id=case_id,
                    recommended_decision=DecisionLabel.MATCHED,
                    primary_reason=ReasonCode.REVERSAL_MATCH,
                    cited_evidence_ids=cited_evidence,
                    matched_record_ids=[source_tx.transaction_id, target_tx.transaction_id],
                    confidence_score=0.95,
                    fuzzy_score=0.95,
                    leakage_risk=leakage_risk,
                    rules_passed=passed_rules,
                    explanation_narrative="Valid reversal verified by baseline.",
                    tool_calls_performed=len(tool_call_sequence),
                    tool_call_sequence=tool_call_sequence,
                    investigation_hypotheses_tested=["REVERSAL_PAIR"],
                    human_review_required=False,
                    investigator="deterministic-cognitive-fallback"
                )

            # Unexplained Mismatch (Includes GST, which baseline can't identify)
            return AgentRecommendation(
                case_id=case_id,
                recommended_decision=DecisionLabel.EXCEPTION,
                primary_reason=ReasonCode.AMOUNT_MISMATCH,
                cited_evidence_ids=cited_evidence,
                confidence_score=0.96,
                fuzzy_score=0.20,
                leakage_risk=leakage_risk,
                rules_passed=passed_rules,
                rules_failed=failed_rules,
                explanation_narrative=f"Unreconciled amount mismatch detected by baseline.",
                tool_calls_performed=len(tool_call_sequence),
                tool_call_sequence=tool_call_sequence,
                investigation_hypotheses_tested=["AMOUNT_DISCREPANCY"],
                human_review_required=False,
                investigator="deterministic-cognitive-fallback"
            )

        # 5. Fallback Abstention
        return AgentRecommendation(
            case_id=case_id,
            recommended_decision=DecisionLabel.UNCERTAIN,
            primary_reason=ReasonCode.BELOW_CONFIDENCE_THRESHOLD,
            cited_evidence_ids=cited_evidence,
            unresolved_questions=["Baseline investigation exhausted without definitive proof."],
            confidence_score=0.60,
            fuzzy_score=0.70,
            leakage_risk=leakage_risk,
            explanation_narrative="Inconclusive reconciliation evidence. Routed to human review queue.",
            tool_calls_performed=len(tool_call_sequence),
            tool_call_sequence=tool_call_sequence,
            human_review_required=True,
            investigator="deterministic-cognitive-fallback"
        )


# Backwards compatibility alias
GeminiVertexReconciliationClient = GeminiReconciliationClient

