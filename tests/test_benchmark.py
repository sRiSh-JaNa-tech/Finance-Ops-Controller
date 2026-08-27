"""Tests for Phase 10 Benchmark Evaluation and Metrics."""

from finance_ops.benchmark.runner import run_benchmark
from finance_ops.benchmark.metrics import FinancialReconciliationMetrics


def test_metric_engine_cost_utility():
    engine = FinancialReconciliationMetrics(benefit_per_correct_match=25.0, cost_false_match=500.0)
    # 10 TPs, 0 FPs, 0 FNs, 2 Uncertain -> 10*25 - 2*10 = 250 - 20 = 230
    u1 = engine.compute_cost_weighted_utility(tp=10, fp=0, fn=0, uncertain_count=2)
    assert u1 == 230.0

    # 10 TPs, 2 FPs -> 250 - 2*500 = 250 - 1000 = -750
    u2 = engine.compute_cost_weighted_utility(tp=10, fp=2, fn=0, uncertain_count=0)
    assert u2 == -750.0


def test_benchmark_execution_fast():
    results = run_benchmark(seeds=[42, 101], cases_per_seed=15)
    assert "summary" in results
    assert "Prototype3_GeminiVertexAgent" in results["summary"]
    p3_summary = results["summary"]["Prototype3_GeminiVertexAgent"]
    assert "match_f1_score" in p3_summary
    assert "cost_weighted_utility" in p3_summary
    # Confirm legacy alias is GONE — Prototype2 is a separate, retired prototype
    assert "Prototype2_Agent" not in results["summary"]
