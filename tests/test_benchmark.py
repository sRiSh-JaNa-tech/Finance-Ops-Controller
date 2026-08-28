"""Tests for Phase 10 Benchmark Evaluation and Metrics."""

from finance_ops.benchmark.runner import run_benchmark
from finance_ops.benchmark.metrics import FinancialReconciliationMetrics


def test_metric_engine_cost_tracking():
    engine = FinancialReconciliationMetrics()
    metrics = engine.evaluate_predictions(
        predictions=[{"decision": "MATCHED", "reason": "MOCK", "usage_metadata": {"input_tokens": 1000, "output_tokens": 100}, "investigator": "gemini"}],
        ground_truth=[{"expected_decision": "MATCHED", "expected_reason": "MOCK"}],
        model_name="gemini-2.5-flash-lite"
    )
    # 1000 input tokens * $0.075 / 1e6 + 100 output tokens * $0.30 / 1e6 = 0.000075 + 0.000030 = $0.000105
    # Cost per 1000 cases = 0.000105 * 1000 = $0.105
    assert metrics["ai_metrics"]["cost_per_1000"] > 0


def test_benchmark_execution_fast():
    results = run_benchmark(seeds=[42], cases_per_seed=2)
    assert "systems" in results
    assert "Rules + Gemini + Verifier" in results["systems"]
    p4_summary = results["systems"]["Rules + Gemini + Verifier"]
    assert "f1_score" in p4_summary
    assert "amount_weighted_accuracy" in p4_summary
    # Confirm legacy aliases are GONE
    assert "Prototype2_Agent" not in results["systems"]
    assert "Prototype4_GeminiReAct" not in results["systems"]
