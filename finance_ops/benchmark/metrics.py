"""
Statistical Metrics, Cost-Weighted Objective, Scenario Breakdown, Bootstrap CIs.

Core Metrics:
  - Precision, Recall, F1, False-Match Rate
  - Cause-Diagnosis Accuracy (multi-class reason code accuracy)
  - Automation Rate (fraction of cases auto-processed)
  - Cost-Weighted Utility
  - False Match Amount (FMA)
  - Amount-weighted accuracy
  - AI Value and Escalation rates
  - Cost tracking via MODEL_PRICING
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from finance_ops.core.models import DecisionLabel

MODEL_PRICING = {
    "gemini-2.5-flash-lite": {"input": 0.075 / 1e6, "output": 0.30 / 1e6},
    "gemini-1.5-flash-lite": {"input": 0.075 / 1e6, "output": 0.30 / 1e6},
    "gemini-1.5-pro": {"input": 1.25 / 1e6, "output": 5.00 / 1e6},
}

class FinancialReconciliationMetrics:
    def __init__(
        self,
        benefit_per_correct_match: float = 25.0,
        cost_false_match: float = 500.0,
        cost_missed_match: float = 50.0,
        cost_human_review: float = 10.0,
        cost_unexplained_exception: float = 50.0
    ):
        self.benefit_tp = benefit_per_correct_match
        self.cost_fp = cost_false_match
        self.cost_fn = cost_missed_match
        self.cost_u = cost_human_review
        self.cost_e = cost_unexplained_exception

    def evaluate_predictions(
        self,
        predictions: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]],
        model_name: str = "gemini-2.5-flash-lite"
    ) -> Dict[str, Any]:
        match_tp = match_fp = match_fn = 0
        triage_tp = triage_fp = triage_fn = 0
        uncertain_count = 0
        correct_reason_count = total_evaluable_reasons = 0
        unexplained_exceptions = 0
        
        false_match_amount = 0.0
        total_amount = 0.0
        correct_amount = 0.0
        
        ai_escalations = 0
        ai_resolutions = 0
        ai_tokens_input = 0
        ai_tokens_output = 0

        latencies_ms: List[float] = []
        llm_investigated = 0
        deterministic_fast_path = 0

        scenario_stats: Dict[str, Dict[str, int]] = {}

        # Multiclass matrix [Expected][Predicted]
        confusion_matrix = {
            DecisionLabel.MATCHED.value: {DecisionLabel.MATCHED.value: 0, DecisionLabel.EXCEPTION.value: 0, DecisionLabel.UNCERTAIN.value: 0},
            DecisionLabel.EXCEPTION.value: {DecisionLabel.MATCHED.value: 0, DecisionLabel.EXCEPTION.value: 0, DecisionLabel.UNCERTAIN.value: 0},
            DecisionLabel.UNCERTAIN.value: {DecisionLabel.MATCHED.value: 0, DecisionLabel.EXCEPTION.value: 0, DecisionLabel.UNCERTAIN.value: 0},
        }
        
        # for bootstrapping F1 over cases
        case_records = []

        for pred, gt in zip(predictions, ground_truth):
            p_dec = pred["decision"]
            g_dec = gt["expected_decision"]
            if hasattr(p_dec, "value"): p_dec = p_dec.value
            if hasattr(g_dec, "value"): g_dec = g_dec.value

            p_reason = pred.get("reason")
            g_reason = gt.get("expected_reason")
            template = gt.get("template", "UNKNOWN")
            
            amount = pred.get("amount", 0.0)
            if hasattr(amount, "__float__"): amount = float(amount)
            total_amount += amount
            
            if "latency_ms" in pred:
                latencies_ms.append(pred["latency_ms"])
            
            usage = pred.get("usage_metadata", {})
            in_t = usage.get("input_tokens", 0)
            out_t = usage.get("output_tokens", 0)
                
            inv_str = str(pred.get("investigator", ""))
            is_ai = bool(inv_str and ("gemini" in inv_str or "ai" in inv_str) and "fallback" not in inv_str and "fast-path" not in inv_str)
            if is_ai:
                llm_investigated += 1
                ai_escalations += 1
                ai_tokens_input += in_t
                ai_tokens_output += out_t
            else:
                deterministic_fast_path += 1

            if template not in scenario_stats:
                scenario_stats[template] = {"match_tp": 0, "match_fp": 0, "match_fn": 0, "uncertain": 0, "total": 0}
            scenario_stats[template]["total"] += 1

            try:
                confusion_matrix[g_dec][p_dec] += 1
            except KeyError:
                pass

            is_correct_case = False

            if p_dec == DecisionLabel.MATCHED.value:
                if g_dec == DecisionLabel.MATCHED.value:
                    p_target = set(pred.get("matched_record_ids", []))
                    g_target = set(gt.get("target_record_ids", gt.get("candidate_tx_ids", [])))
                    if not g_target or (p_target and p_target.intersection(g_target)):
                        match_tp += 1
                        scenario_stats[template]["match_tp"] += 1
                        correct_amount += amount
                        is_correct_case = True
                    else:
                        match_fp += 1
                        scenario_stats[template]["match_fp"] += 1
                        false_match_amount += amount
                else:
                    match_fp += 1
                    scenario_stats[template]["match_fp"] += 1
                    false_match_amount += amount
            elif g_dec == DecisionLabel.MATCHED.value:
                match_fn += 1
                scenario_stats[template]["match_fn"] += 1

            if p_dec == DecisionLabel.EXCEPTION.value:
                if g_dec == DecisionLabel.EXCEPTION.value:
                    triage_tp += 1
                    correct_amount += amount
                    is_correct_case = True
                else:
                    triage_fp += 1
                    if g_dec == DecisionLabel.MATCHED.value:
                        unexplained_exceptions += 1
            elif g_dec == DecisionLabel.EXCEPTION.value:
                triage_fn += 1

            if p_dec == DecisionLabel.UNCERTAIN.value:
                uncertain_count += 1
                scenario_stats[template]["uncertain"] += 1

            if p_reason and g_reason:
                total_evaluable_reasons += 1
                p_val = p_reason.value if hasattr(p_reason, "value") else str(p_reason)
                g_val = g_reason.value if hasattr(g_reason, "value") else str(g_reason)
                if p_val == g_val:
                    correct_reason_count += 1
                    
            if is_ai and is_correct_case:
                ai_resolutions += 1
                
            case_records.append({
                "p_dec": p_dec,
                "g_dec": g_dec,
                "is_correct_case": is_correct_case
            })


        n_total = len(ground_truth)
        
        match_prec = match_tp / (match_tp + match_fp) if (match_tp + match_fp) > 0 else 0.0
        match_rec = match_tp / (match_tp + match_fn) if (match_tp + match_fn) > 0 else 0.0
        match_f1 = 2 * (match_prec * match_rec) / (match_prec + match_rec) if (match_prec + match_rec) > 0 else 0.0
        
        triage_prec = triage_tp / (triage_tp + triage_fp) if (triage_tp + triage_fp) > 0 else 0.0
        triage_rec = triage_tp / (triage_tp + triage_fn) if (triage_tp + triage_fn) > 0 else 0.0
        triage_f1 = 2 * (triage_prec * triage_rec) / (triage_prec + triage_rec) if (triage_prec + triage_rec) > 0 else 0.0

        # Bootstrap F1
        n_resamples = 1000
        boot_f1s = []
        import random
        for _ in range(n_resamples):
            sample = [random.choice(case_records) for _ in range(len(case_records))]
            tp = fp = fn = 0
            for c in sample:
                if c["p_dec"] == DecisionLabel.MATCHED.value:
                    if c["g_dec"] == DecisionLabel.MATCHED.value and c["is_correct_case"]:
                        tp += 1
                    else:
                        fp += 1
                elif c["g_dec"] == DecisionLabel.MATCHED.value:
                    fn += 1
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
            boot_f1s.append(f)
            
        boot_f1s.sort()
        f1_low = boot_f1s[int(n_resamples * 0.025)] if boot_f1s else 0.0
        f1_high = boot_f1s[int(n_resamples * 0.975)] if boot_f1s else 0.0

        fmr = match_fp / (match_tp + match_fp) if (match_tp + match_fp) > 0 else 0.0
        automation_rate = (n_total - uncertain_count) / n_total if n_total > 0 else 0.0
        reason_acc = correct_reason_count / total_evaluable_reasons if total_evaluable_reasons > 0 else 0.0
        
        amount_weighted_acc = correct_amount / total_amount if total_amount > 0 else 0.0
        
        pricing = MODEL_PRICING.get(model_name, MODEL_PRICING["gemini-2.5-flash-lite"])
        total_cost = (ai_tokens_input * pricing["input"]) + (ai_tokens_output * pricing["output"])

        scenario_breakdown = {}
        for tmpl, stats in scenario_stats.items():
            t, f, miss, unc = stats["match_tp"], stats["match_fp"], stats["match_fn"], stats["uncertain"]
            s_prec = t / (t + f) if (t + f) > 0 else 0.0
            s_rec = t / (t + miss) if (t + miss) > 0 else 0.0
            scenario_breakdown[tmpl] = {
                "total": stats["total"],
                "tp": t, "fp": f, "fn": miss, "uncertain": unc,
                "precision": round(s_prec, 4),
                "recall": round(s_rec, 4),
            }

        return {
            "total_cases": n_total,
            "true_positives": match_tp,
            "false_positives": match_fp,
            "false_negatives": match_fn,
            "triage_true_positives": triage_tp,
            "triage_false_positives": triage_fp,
            "triage_false_negatives": triage_fn,
            "uncertain_cases": uncertain_count,
            "precision": float(match_prec),
            "recall": float(match_rec),
            "f1_score": float(match_f1),
            "triage_precision": float(triage_prec),
            "triage_recall": float(triage_rec),
            "triage_f1_score": float(triage_f1),
            "false_match_rate": float(fmr),
            "false_match_amount": float(false_match_amount),
            "amount_weighted_accuracy": float(amount_weighted_acc),
            "automation_rate": float(automation_rate),
            "cause_diagnosis_accuracy": float(reason_acc),
            "scenario_breakdown": scenario_breakdown,
            "confusion_matrix": confusion_matrix,
            "ai_metrics": {
                "escalations": ai_escalations,
                "resolutions": ai_resolutions,
                "resolution_rate": ai_resolutions / ai_escalations if ai_escalations > 0 else 0.0,
                "input_tokens": ai_tokens_input,
                "output_tokens": ai_tokens_output,
                "total_cost": total_cost,
                "cost_per_case": total_cost / n_total if n_total > 0 else 0.0,
                "cost_per_resolution": total_cost / ai_resolutions if ai_resolutions > 0 else 0.0,
                "cost_per_1000": (total_cost / n_total * 1000) if n_total > 0 else 0.0
            },
            "latencies_ms": latencies_ms,
            "p50_latency": float(np.percentile(latencies_ms, 50)) if latencies_ms else 0.0,
            "p90_latency": float(np.percentile(latencies_ms, 90)) if latencies_ms else 0.0,
            "p95_latency": float(np.percentile(latencies_ms, 95)) if latencies_ms else 0.0,
            "p99_latency": float(np.percentile(latencies_ms, 99)) if latencies_ms else 0.0,
            "llm_investigated": llm_investigated,
            "deterministic_fast_path": deterministic_fast_path,
            "f1_ci95": [float(f1_low), float(f1_high)],
        }


def bootstrap_confidence_interval(
    values: List[float],
    n_resamples: int = 10000,
    alpha: float = 0.05
) -> Tuple[float, float, float]:
    """
    Returns (mean, lower_95ci, upper_95ci) using percentile bootstrap over case-level scores.
    """
    if len(values) == 0:
        return 0.0, 0.0, 0.0
    arr = np.array(values)
    mean_val = float(np.mean(arr))
    if len(arr) == 1:
        return mean_val, mean_val, mean_val

    boot_means = [
        np.mean(np.random.choice(arr, size=len(arr), replace=True))
        for _ in range(n_resamples)
    ]
    low = float(np.percentile(boot_means, 100 * (alpha / 2.0)))
    high = float(np.percentile(boot_means, 100 * (1.0 - alpha / 2.0)))
    return mean_val, low, high
