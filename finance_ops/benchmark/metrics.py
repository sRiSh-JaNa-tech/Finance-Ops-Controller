"""
Statistical Metrics, Cost-Weighted Objective, Scenario Breakdown, Bootstrap CIs.

Core Metrics:
  - Precision, Recall, F1, False-Match Rate
  - Cause-Diagnosis Accuracy (multi-class reason code accuracy)
  - Automation Rate (fraction of cases auto-processed)
  - Cost-Weighted Utility: U = B·TP - c_FP·FP - c_FN·FN - c_U·Uncertain - c_E·Unexplained
  - AURC (Area Under Risk-Coverage Curve)
  - Bootstrap 95% Confidence Intervals

Utility parameters (from implementation plan):
  B     = $25    (benefit per correct automated match)
  c_FP  = $500   (cost of a false automatic match)
  c_FN  = $50    (cost of a missed real match)
  c_U   = $10    (cost of routing to human review)
  c_E   = $50    (cost of unexplained exception)
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from finance_ops.core.models import DecisionLabel


# ─────────────────────────────────────────────────────────────────────────────
# Core Metric Engine
# ─────────────────────────────────────────────────────────────────────────────

class FinancialReconciliationMetrics:
    """Research-grade evaluation metrics for financial reconciliation benchmarking."""

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

    def compute_cost_weighted_utility(
        self,
        tp: int,
        fp: int,
        fn: int,
        uncertain_count: int,
        unexplained_exceptions: int = 0
    ) -> float:
        """
        Cost-Weighted Utility:
            U = B·TP - c_FP·FP - c_FN·FN - c_U·Uncertain - c_E·Unexplained
        """
        return float(
            self.benefit_tp * tp
            - self.cost_fp * fp
            - self.cost_fn * fn
            - self.cost_u * uncertain_count
            - self.cost_e * unexplained_exceptions
        )

    def evaluate_predictions(
        self,
        predictions: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Computes all core metrics from prediction/ground-truth pairs.
        Also returns per-scenario breakdown and calibration data.
        """
        tp = fp = fn = tn = uncertain_count = 0
        correct_reason_count = total_evaluable_reasons = 0
        unexplained_exceptions = 0

        # For calibration / AURC
        match_confidences: List[float] = []
        match_labels: List[int] = []

        # Latency and AI Contribution tracking
        latencies_ms: List[float] = []
        llm_investigated = 0
        deterministic_fast_path = 0

        # Per-scenario template tracking
        scenario_stats: Dict[str, Dict[str, int]] = {}

        for pred, gt in zip(predictions, ground_truth):
            p_dec = pred["decision"]
            g_dec = gt["expected_decision"]

            p_reason = pred.get("reason")
            g_reason = gt.get("expected_reason")
            template = gt.get("template", "UNKNOWN")
            
            # Record Latency and AI contribution
            if "latency_ms" in pred:
                latencies_ms.append(pred["latency_ms"])
            if pred.get("investigator") == "gemini-2.5-flash-agent":
                llm_investigated += 1
            elif pred.get("investigator") == "deterministic-cognitive-fallback":
                deterministic_fast_path += 1

            # Initialize scenario bucket
            if template not in scenario_stats:
                scenario_stats[template] = {"tp": 0, "fp": 0, "fn": 0, "uncertain": 0, "total": 0}
            scenario_stats[template]["total"] += 1

            # Collect calibration data for MATCHED predictions
            if p_dec == DecisionLabel.MATCHED:
                conf = pred.get("calibrated_confidence", 0.5)
                match_confidences.append(conf)
                is_correct = 1 if (g_dec == DecisionLabel.MATCHED) else 0
                match_labels.append(is_correct)

            # Count outcomes
            if p_dec == DecisionLabel.MATCHED:
                if g_dec == DecisionLabel.MATCHED:
                    p_target = set(pred.get("matched_record_ids", []))
                    g_target = set(gt.get("target_record_ids", []))
                    if not g_target or (p_target and p_target.intersection(g_target)):
                        tp += 1
                        scenario_stats[template]["tp"] += 1
                    else:
                        fp += 1
                        scenario_stats[template]["fp"] += 1
                else:
                    # False Match (erroneous ledger reconciliation on non-matching pair)
                    fp += 1
                    scenario_stats[template]["fp"] += 1

            elif p_dec == DecisionLabel.EXCEPTION:
                if g_dec == DecisionLabel.EXCEPTION:
                    tp += 1
                    scenario_stats[template]["tp"] += 1
                elif g_dec == DecisionLabel.MATCHED:
                    # Missed a valid match by declaring an exception
                    fn += 1
                    unexplained_exceptions += 1
                    scenario_stats[template]["fn"] += 1
                else:
                    fn += 1
                    scenario_stats[template]["fn"] += 1

            elif p_dec == DecisionLabel.UNCERTAIN:
                uncertain_count += 1
                scenario_stats[template]["uncertain"] += 1
                if g_dec == DecisionLabel.UNCERTAIN:
                    tp += 1
                elif g_dec == DecisionLabel.MATCHED:
                    fn += 1
                    scenario_stats[template]["fn"] += 1

            # Reason code diagnosis
            if p_reason and g_reason:
                total_evaluable_reasons += 1
                p_val = p_reason.value if hasattr(p_reason, "value") else str(p_reason)
                g_val = g_reason.value if hasattr(g_reason, "value") else str(g_reason)
                if p_val == g_val:
                    correct_reason_count += 1

        n_total = len(ground_truth)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        fmr = fp / n_total if n_total > 0 else 0.0
        automation_rate = (n_total - uncertain_count) / n_total if n_total > 0 else 0.0
        reason_acc = correct_reason_count / total_evaluable_reasons if total_evaluable_reasons > 0 else 0.0
        utility = self.compute_cost_weighted_utility(tp, fp, fn, uncertain_count, unexplained_exceptions)

        # Scenario-level precision/recall
        scenario_breakdown = {}
        for tmpl, stats in scenario_stats.items():
            t, f, miss, unc = stats["tp"], stats["fp"], stats["fn"], stats["uncertain"]
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
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "uncertain_cases": uncertain_count,
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "false_match_rate": float(fmr),
            "automation_rate": float(automation_rate),
            "cause_diagnosis_accuracy": float(reason_acc),
            "cost_weighted_utility": float(utility),
            "scenario_breakdown": scenario_breakdown,
            "latencies_ms": latencies_ms,
            "llm_investigated": llm_investigated,
            "deterministic_fast_path": deterministic_fast_path,
            # For AURC computation
            "_match_confidences": match_confidences,
            "_match_labels": match_labels,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap Confidence Intervals
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_confidence_interval(
    values: List[float],
    n_resamples: int = 1000,
    alpha: float = 0.05
) -> Tuple[float, float, float]:
    """
    Returns (mean, lower_95ci, upper_95ci) using percentile bootstrap.
    """
    if not values:
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


# ─────────────────────────────────────────────────────────────────────────────
# Scenario-Level Aggregation
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_scenario_breakdown(
    seed_results: List[Dict[str, Any]]
) -> Dict[str, Dict[str, float]]:
    """
    Aggregates per-scenario precision/recall across multiple seeds.
    Returns mean precision and recall per scenario template.
    """
    from collections import defaultdict
    template_prec: Dict[str, List[float]] = defaultdict(list)
    template_rec: Dict[str, List[float]] = defaultdict(list)

    for seed_result in seed_results:
        for tmpl, stats in seed_result.get("scenario_breakdown", {}).items():
            template_prec[tmpl].append(stats["precision"])
            template_rec[tmpl].append(stats["recall"])

    aggregated = {}
    for tmpl in template_prec:
        p_mean, p_lo, p_hi = bootstrap_confidence_interval(template_prec[tmpl])
        r_mean, r_lo, r_hi = bootstrap_confidence_interval(template_rec[tmpl])
        aggregated[tmpl] = {
            "mean_precision": round(p_mean, 4),
            "prec_ci95": [round(p_lo, 4), round(p_hi, 4)],
            "mean_recall": round(r_mean, 4),
            "rec_ci95": [round(r_lo, 4), round(r_hi, 4)],
        }

    return aggregated


# ─────────────────────────────────────────────────────────────────────────────
# Auditor Reproducibility Checker
# ─────────────────────────────────────────────────────────────────────────────

def check_citation_validity(case_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Checks that claimed evidence IDs and match arithmetic are internally consistent.
    Returns audit compliance percentage.
    """
    total = len(case_results)
    valid_citation_cases = 0

    for case in case_results:
        cited = case.get("cited_evidence_ids", [])
        available_facts = [f.get("fact_id", "") for f in case.get("evidence_facts", [])]
        available_set = set(available_facts)

        if not cited or all(c in available_set for c in cited):
            valid_citation_cases += 1

    citation_compliance = valid_citation_cases / total if total > 0 else 1.0
    return {
        "total_cases": total,
        "citation_compliant_cases": valid_citation_cases,
        "citation_compliance_rate": round(citation_compliance, 4),
        "audit_pass": citation_compliance >= 0.99,
    }
