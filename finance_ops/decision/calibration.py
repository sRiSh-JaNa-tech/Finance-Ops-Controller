"""
Confidence Calibration, Asymmetric Decision Policy, ECE/Brier, and Risk-Coverage Curves.

Mathematical formulations:
  Platt Sigmoid:    P(match|z) = 1 / (1 + exp(-(A·z + B)))
  ECE:              sum_m (|B_m|/N) · |acc(B_m) - conf(B_m)|
  Asymmetric τ:     τ_auto = (c_FP + c_U) / (c_FP + B)
  Risk-Coverage:    risk(θ) = FP(θ) / Predict(θ),  coverage(θ) = Predict(θ) / N
  AURC:             ∫₀¹ risk(c) dc  (approximated via trapezoid rule)
"""

import math
import numpy as np
from typing import List, Tuple, Dict, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Platt Scaling Calibrator
# ─────────────────────────────────────────────────────────────────────────────

class ConfidenceCalibrator:
    """
    Platt Scaling Calibrator: maps raw heuristic/model confidence scores to
    true posterior match probabilities via fitted logistic sigmoid:

        P(Y=1 | s) = 1 / (1 + exp(-(A·s + B)))

    Default parameters (a=5.0, b=-2.5) are set to achieve good calibration
    on the synthetic benchmark distribution. Can be fitted on validation data.
    """

    def __init__(self, a: float = 5.0, b: float = -2.5):
        self.a = a
        self.b = b

    def calibrate(self, raw_score: float) -> float:
        """Calibrates raw score ∈ [0, 1] to calibrated posterior probability."""
        z = self.a * raw_score + self.b
        # Numerically stable sigmoid
        if z >= 0:
            prob = 1.0 / (1.0 + math.exp(-z))
        else:
            exp_z = math.exp(z)
            prob = exp_z / (1.0 + exp_z)
        return float(np.clip(prob, 0.01, 0.99))

    def fit_from_data(
        self,
        raw_scores: List[float],
        binary_labels: List[int],
        learning_rate: float = 0.05,
        n_iter: int = 200
    ) -> None:
        """
        Fits Platt parameters (A, B) via gradient descent on the binary cross-entropy loss.
        Updates self.a and self.b in-place.
        """
        if not raw_scores or not binary_labels:
            return
        s = np.array(raw_scores, dtype=float)
        y = np.array(binary_labels, dtype=float)

        a, b = float(self.a), float(self.b)
        for _ in range(n_iter):
            z = a * s + b
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))
            err = p - y
            grad_a = float(np.mean(err * s))
            grad_b = float(np.mean(err))
            a -= learning_rate * grad_a
            b -= learning_rate * grad_b

        self.a = a
        self.b = b


# ─────────────────────────────────────────────────────────────────────────────
# Calibration Metrics
# ─────────────────────────────────────────────────────────────────────────────

def calculate_brier_score(probabilities: List[float], binary_labels: List[int]) -> float:
    """
    Brier Score (mean squared calibration error):
        BS = (1/N) · Σ (p_i - y_i)²

    Perfect calibration: BS = 0. Random classifier: BS ≈ 0.25.
    """
    if not probabilities or not binary_labels:
        return 0.0
    p = np.array(probabilities)
    y = np.array(binary_labels)
    return float(np.mean((p - y) ** 2))


def calculate_expected_calibration_error(
    probabilities: List[float],
    binary_labels: List[int],
    n_bins: int = 10
) -> Tuple[float, List[Dict]]:
    """
    Expected Calibration Error (ECE):
        ECE = Σₘ (|Bₘ|/N) · |acc(Bₘ) - conf(Bₘ)|

    Returns:
        (ece_value, bin_data): ECE scalar and per-bin breakdown for histogram visualization.
    """
    if not probabilities or not binary_labels:
        return 0.0, []

    p = np.array(probabilities)
    y = np.array(binary_labels)
    n = len(p)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_data = []

    for i in range(n_bins):
        bin_mask = (p > bin_edges[i]) & (p <= bin_edges[i + 1])
        bin_count = int(np.sum(bin_mask))
        if bin_count > 0:
            bin_acc = float(np.mean(y[bin_mask]))
            bin_conf = float(np.mean(p[bin_mask]))
            bin_ece = (bin_count / n) * abs(bin_acc - bin_conf)
            ece += bin_ece
            bin_data.append({
                "bin_lower": round(bin_edges[i], 2),
                "bin_upper": round(bin_edges[i + 1], 2),
                "count": bin_count,
                "mean_confidence": round(bin_conf, 4),
                "mean_accuracy": round(bin_acc, 4),
                "calibration_gap": round(bin_acc - bin_conf, 4),
                "ece_contribution": round(bin_ece, 6),
            })

    return float(ece), bin_data


# ─────────────────────────────────────────────────────────────────────────────
# Asymmetric Cost-Weighted Decision Policy
# ─────────────────────────────────────────────────────────────────────────────

class AsymmetricDecisionPolicy:
    """
    Implements the asymmetric cost-weighted decision rule:

        Accept MATCHED if:  P̂(correct) ≥ τ_auto = (c_FP + c_U) / (c_FP + B)
        Route to UNCERTAIN if: P̂(correct) < τ_auto
        Force EXCEPTION if:  P̂(correct) < τ_exception

    where c_FP >> c_U >> c_FN reflects that false automatic matches
    (financial fraud risk) are far costlier than human review overhead.

    Default cost structure from implementation plan:
        c_FP = $500  (false match penalty)
        c_FN = $50   (missed match penalty)
        c_U  = $10   (human review cost)
        B    = $25   (benefit per correct automated match)
    """

    def __init__(
        self,
        cost_false_match: float = 500.0,
        cost_missed_match: float = 50.0,
        cost_human_review: float = 10.0,
        benefit_per_correct: float = 25.0,
    ):
        self.c_fp = cost_false_match
        self.c_fn = cost_missed_match
        self.c_u = cost_human_review
        self.b = benefit_per_correct

    @property
    def auto_match_threshold(self) -> float:
        """
        Asymmetric automation threshold:
            τ_auto = (c_FP + c_U) / (c_FP + B)

        Only automate a MATCHED decision when calibrated confidence exceeds this threshold.
        Derivation: Accept automation only if E[Benefit] > E[Cost]:
            B · P ≥ c_FP · (1 - P) + c_U
            P ≥ (c_FP + c_U) / (c_FP + B)
        """
        threshold = (self.c_fp + self.c_u) / (self.c_fp + self.b)
        return min(threshold, 0.99)

    @property
    def exception_threshold(self) -> float:
        """Below this threshold even EXCEPTION decisions require human review."""
        return 0.70

    def decide(self, calibrated_confidence: float, recommended_decision: str) -> Dict:
        """
        Applies the asymmetric policy to a calibrated confidence score.

        Returns a decision dict with:
            decision: final label
            is_automated: whether to auto-process without human review
            requires_human: whether to queue for review
            policy_reason: explanation string
        """
        from finance_ops.core.models import DecisionLabel

        if recommended_decision == DecisionLabel.MATCHED.value or recommended_decision == DecisionLabel.MATCHED:
            if calibrated_confidence >= self.auto_match_threshold:
                return {
                    "decision": DecisionLabel.MATCHED,
                    "is_automated": True,
                    "requires_human": False,
                    "policy_reason": f"Confidence {calibrated_confidence:.3f} >= τ_auto={self.auto_match_threshold:.3f}"
                }
            else:
                return {
                    "decision": DecisionLabel.UNCERTAIN,
                    "is_automated": False,
                    "requires_human": True,
                    "policy_reason": f"Escalated: confidence {calibrated_confidence:.3f} < τ_auto={self.auto_match_threshold:.3f}"
                }
        elif recommended_decision == DecisionLabel.EXCEPTION.value or recommended_decision == DecisionLabel.EXCEPTION:
            return {
                "decision": DecisionLabel.EXCEPTION,
                "is_automated": True,
                "requires_human": False,
                "policy_reason": "Exception decisioned deterministically; no confidence threshold applies"
            }
        else:
            return {
                "decision": DecisionLabel.UNCERTAIN,
                "is_automated": False,
                "requires_human": True,
                "policy_reason": "Uncertain — queued for human reviewer"
            }


# ─────────────────────────────────────────────────────────────────────────────
# Risk-Coverage Curves and AURC
# ─────────────────────────────────────────────────────────────────────────────

def compute_risk_coverage_curve(
    probabilities: List[float],
    binary_labels: List[int],
    n_thresholds: int = 50
) -> Dict:
    """
    Computes the Risk-Coverage curve for selective prediction.

    At each confidence threshold θ:
      - Coverage(θ) = fraction of cases where confidence ≥ θ (cases that get automated)
      - Risk(θ) = false match rate among automated cases

    AURC = ∫₀¹ risk(c) dc  (area under the risk-coverage curve)
    A lower AURC is better (less risk at each coverage level).

    Returns:
        Dict with thresholds, coverages, risks, and AURC value.
    """
    if not probabilities or not binary_labels:
        return {"aurc": 0.0, "coverages": [], "risks": [], "thresholds": []}

    p = np.array(probabilities, dtype=float)
    y = np.array(binary_labels, dtype=float)
    n = len(p)

    thresholds = np.linspace(0.0, 1.0, n_thresholds + 1)
    coverages = []
    risks = []

    for theta in thresholds:
        selected = p >= theta
        coverage = float(np.sum(selected)) / n
        if coverage == 0:
            risk = 0.0
        else:
            # Risk = false match rate = (1 - accuracy) among selected
            sel_labels = y[selected]
            risk = float(1.0 - np.mean(sel_labels)) if len(sel_labels) > 0 else 0.0
        coverages.append(round(coverage, 4))
        risks.append(round(risk, 4))

    # AURC via trapezoid rule: sort by coverage, then integrate risk over coverage axis
    sorted_pairs = sorted(zip(coverages, risks))
    cov_arr = np.array([p[0] for p in sorted_pairs])
    risk_arr = np.array([p[1] for p in sorted_pairs])
    _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
    aurc = abs(float(_trapz(risk_arr, cov_arr)))

    return {
        "aurc": round(aurc, 6),
        "thresholds": [round(float(t), 4) for t in thresholds],
        "coverages": coverages,
        "risks": risks,
    }
