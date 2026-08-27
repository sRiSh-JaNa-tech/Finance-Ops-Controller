r"""Prototype 3 Benchmark Runner — Multi-System Evaluation Across 15 Scenarios.

Compares:
1. ExactIdentifierMatcher
2. DeterministicRuleMatcher
3. Prototype1HybridPipeline
4. Prototype3_GeminiVertexAgent (Evidence-Grounded Autonomous Investigation Agent)

Reports:
- PyResolveMetrics (Precision, Recall, F1, Reduction Ratio, Pairs Completeness)
- False Match Rate (FMR)
- Cost-Weighted Utility (U = 25*TP - 500*FP - 50*FN - 10*U)
- Tool Efficiency (Avg Tool Calls per Case)
- Per-scenario Accuracy across all 15 Canonical Scenarios
"""

import time
from typing import List, Dict, Any, Optional
from decimal import Decimal
import numpy as np

from finance_ops.core.models import DecisionLabel
from finance_ops.generators.synthetic_data import generate_synthetic_dataset
from finance_ops.generators.fault_injection import ScenarioTemplate
from finance_ops.ingestion.storage import FinancialDataRepository
from finance_ops.retrieval.blocking import MultiPassBlockingEngine
from finance_ops.retrieval.graph import FinancialEntityGraph
from finance_ops.rules.engine import DeterministicRuleEngine
from finance_ops.rules.constraint_solver import SplitReconciliationSolver, reset_reconciliation_registry
from finance_ops.evidence.tools import InvestigationToolbox
from finance_ops.agent.vertex_client import GeminiReconciliationClient
from finance_ops.agent.investigator import BoundedInvestigationAgent
from finance_ops.decision.verifier import DeterministicPolicyVerifier
from finance_ops.decision.calibration import ConfidenceCalibrator
from finance_ops.baselines.baseline_models import ExactIdentifierMatcher, DeterministicRuleMatcher, Prototype1HybridPipeline
from finance_ops.benchmark.metrics import (
    FinancialReconciliationMetrics,
    bootstrap_confidence_interval,
    aggregate_scenario_breakdown,
)


def _build_env(dataset):
    """Builds the shared in-memory infrastructure for one benchmark run."""
    repo = FinancialDataRepository()
    blocking = MultiPassBlockingEngine()
    graph = FinancialEntityGraph()
    rules = DeterministicRuleEngine()
    solver = SplitReconciliationSolver()

    all_txs = dataset.gateway_records + dataset.bank_records

    for r in all_txs:
        repo.store_canonical_transaction(r)
        graph.add_transaction_node(r)

    toolbox = InvestigationToolbox(repo, blocking, graph, rules, solver)
    vertex_client = GeminiReconciliationClient()
    agent = BoundedInvestigationAgent(toolbox, max_steps=5, vertex_client=vertex_client)
    verifier = DeterministicPolicyVerifier(repo, rules)
    calibrator = ConfidenceCalibrator()

    return repo, blocking, toolbox, agent, verifier, calibrator


def run_benchmark(
    seeds: Optional[List[int]] = None,
    cases_per_seed: int = 45,
    run_ablations: bool = False,
) -> Dict[str, Any]:
    """
    Executes repeated-seed evaluation of Prototype-3 against baseline systems.
    """
    if seeds is None:
        seeds = [42, 101, 202]

    metric_engine = FinancialReconciliationMetrics()
    system_names = ["ExactMatcher", "RuleMatcher", "Prototype1_Hybrid", "Prototype3_GeminiVertexAgent"]

    seed_results: Dict[str, List[Dict[str, Any]]] = {sys: [] for sys in system_names}
    blocking_stats_all: List[Dict[str, Any]] = []
    all_honest_exceptions: List[Dict[str, Any]] = []

    for seed in seeds:
        reset_reconciliation_registry()
        dataset = generate_synthetic_dataset(n_cases=cases_per_seed, seed=seed)
        repo, blocking, toolbox, agent, verifier, calibrator = _build_env(dataset)

        exact_matcher = ExactIdentifierMatcher()
        rule_matcher = DeterministicRuleMatcher()
        proto1_matcher = Prototype1HybridPipeline()

        blocking_pairs = blocking.generate_candidate_pairs(dataset.gateway_records, dataset.bank_records)
        b_metrics = blocking.compute_blocking_metrics(dataset.gateway_records, dataset.bank_records, blocking_pairs)
        blocking_stats_all.append(b_metrics)

        candidate_lookup = {}
        for src, tgt, keys in blocking_pairs:
            if src.transaction_id not in candidate_lookup:
                candidate_lookup[src.transaction_id] = []
            candidate_lookup[src.transaction_id].append(tgt)

        honest_exceptions_seed: List[Dict[str, Any]] = []

        total_cases = len(dataset.ground_truth_cases)
        for sys_idx, sys_name in enumerate(system_names, start=1):
            print(f"\n[{sys_idx}/{len(system_names)}] Evaluating {sys_name} ({total_cases} cases)...", flush=True)
            predictions = []

            for case_num, case in enumerate(dataset.ground_truth_cases, start=1):
                src_id = case["source_tx_id"]
                src_tx = repo.get_transaction(src_id)
                if not src_tx:
                    continue

                candidates = candidate_lookup.get(src_id, [])
                if not candidates:
                    candidates = [repo.get_transaction(cid) for cid in case["candidate_tx_ids"] if repo.get_transaction(cid)]

                if sys_name == "ExactMatcher":
                    start_time = time.perf_counter()
                    res = exact_matcher.match(src_tx, candidates)
                    lat = time.perf_counter() - start_time
                    predictions.append({"decision": res["decision"], "reason": res["reason"], "confidence": 1.0, "latency_ms": lat * 1000})
                elif sys_name == "RuleMatcher":
                    start_time = time.perf_counter()
                    res = rule_matcher.match(src_tx, candidates)
                    lat = time.perf_counter() - start_time
                    predictions.append({"decision": res["decision"], "reason": res["reason"], "confidence": 1.0, "latency_ms": lat * 1000})
                elif sys_name == "Prototype1_Hybrid":
                    start_time = time.perf_counter()
                    res = proto1_matcher.match(src_tx, candidates)
                    lat = time.perf_counter() - start_time
                    predictions.append({"decision": res["decision"], "reason": res["reason"], "confidence": 0.85, "latency_ms": lat * 1000})
                elif sys_name == "Prototype3_GeminiVertexAgent":
                    start_time = time.perf_counter()
                    rec = agent.investigate(src_tx, candidates, case_id=case["case_id"])
                    final_dec = verifier.verify_and_finalize(rec, src_tx, rec.confidence_score)
                    lat = time.perf_counter() - start_time
                    
                    predictions.append({
                        "decision": final_dec.decision,
                        "reason": final_dec.reason,
                        "confidence": final_dec.calibrated_confidence,
                        "tool_calls": rec.tool_calls_performed,
                        "leakage_risk": rec.leakage_risk,
                        "investigator": rec.investigator,
                        "latency_ms": lat * 1000
                    })
                    if final_dec.decision in (DecisionLabel.UNCERTAIN, DecisionLabel.EXCEPTION) or final_dec.verifier_status != "VERIFIED_VALID":
                        honest_exceptions_seed.append({
                            "case_id": case["case_id"],
                            "template": case.get("template", "UNKNOWN"),
                            "decision": final_dec.decision.value,
                            "reason": final_dec.reason.value,
                            "calibrated_confidence": round(final_dec.calibrated_confidence, 4),
                            "verifier_status": final_dec.verifier_status,
                            "verifier_notes": final_dec.verifier_notes,
                            "explanation": final_dec.explanation,
                            "source_amount_inr": float(src_tx.amount),
                            "candidate_count": len(candidates),
                        })

                    tmpl = case.get("template", "SCENARIO")
                    is_ai = bool(rec.investigator and ("gemini" in rec.investigator or "ai" in rec.investigator) and "fallback" not in rec.investigator and "fast-path" not in rec.investigator)
                    inv_type = "AI-ReAct" if is_ai else ("FastPath" if (rec.investigator and "fast-path" in rec.investigator) else "RuleFallback")
                    print(
                        f"  -> Case {case_num:03d}/{total_cases:03d} [{tmpl:<30}] => "
                        f"{final_dec.decision.value:<9} | {final_dec.reason.value:<28} "
                        f"({lat*1000:4.0f}ms | {inv_type} | tools: {rec.tool_calls_performed})",
                        flush=True
                    )

            if sys_name != "Prototype3_GeminiVertexAgent":
                total_lat_ms = sum(p["latency_ms"] for p in predictions)
                print(f"  [+] Completed {total_cases} cases in {total_lat_ms/1000:.3f}s", flush=True)

            sys_metrics = metric_engine.evaluate_predictions(predictions, dataset.ground_truth_cases)
            seed_results[sys_name].append(sys_metrics)

        all_honest_exceptions.extend(honest_exceptions_seed)

    summary_dict = {}
    systems_dict = {}

    for sys_name in system_names:
        f1_scores = [r["f1_score"] for r in seed_results[sys_name]]
        precisions = [r["precision"] for r in seed_results[sys_name]]
        recalls = [r["recall"] for r in seed_results[sys_name]]
        triage_f1_scores = [r.get("triage_f1_score", 0.0) for r in seed_results[sys_name]]
        triage_precisions = [r.get("triage_precision", 0.0) for r in seed_results[sys_name]]
        triage_recalls = [r.get("triage_recall", 0.0) for r in seed_results[sys_name]]
        fmrs = [r["false_match_rate"] for r in seed_results[sys_name]]
        utilities = [r["cost_weighted_utility"] for r in seed_results[sys_name]]
        auto_rates = [r["automation_rate"] for r in seed_results[sys_name]]
        cause_accs = [r.get("cause_diagnosis_accuracy", 0.0) for r in seed_results[sys_name]]

        # Latency and AI Contribution tracking
        all_latencies = []
        llm_count = 0
        deterministic_count = 0
        total_cases = 0

        for r in seed_results[sys_name]:
            all_latencies.extend(r.get("latencies_ms", []))
            llm_count += r.get("llm_investigated", 0)
            deterministic_count += r.get("deterministic_fast_path", 0)
            total_cases += r.get("total_cases", cases_per_seed)

        p95_latency = round(float(np.percentile(all_latencies, 95)), 2) if all_latencies else 0.0
        total_time_s = sum(all_latencies) / 1000.0 if all_latencies else 1.0
        throughput = round(total_cases / total_time_s, 2) if total_time_s > 0 else 0.0

        f1_mean = np.mean(f1_scores)
        f1_ci = bootstrap_confidence_interval(np.array(f1_scores))

        metrics_obj = {
            "match_f1_score": round(float(f1_mean), 4),
            "match_precision": round(float(np.mean(precisions)), 4),
            "match_recall": round(float(np.mean(recalls)), 4),
            "triage_f1_score": round(float(np.mean(triage_f1_scores)), 4),
            "triage_precision": round(float(np.mean(triage_precisions)), 4),
            "triage_recall": round(float(np.mean(triage_recalls)), 4),
            "false_match_rate": round(float(np.mean(fmrs)), 4),
            "cause_diagnosis_accuracy": round(float(np.mean(cause_accs)), 4),
            "cost_weighted_utility": round(float(np.mean(utilities)), 2),
            "automation_rate_pct": round(float(np.mean(auto_rates)) * 100, 2),
            "throughput_cases_per_sec": throughput,
            "p95_latency_ms": p95_latency,
            "llm_investigated": llm_count,
            "deterministic_fast_path": deterministic_count,
        }
        systems_dict[sys_name] = metrics_obj
        summary_dict[sys_name] = metrics_obj

    summary_report: Dict[str, Any] = {
        "benchmark_version": "Prototype-3-PROD",
        "total_seeds": len(seeds),
        "cases_per_seed": cases_per_seed,
        "systems": systems_dict,
        "summary": summary_dict,
        "honest_exception_list": all_honest_exceptions,
        "blocking_performance": {
            "avg_reduction_ratio_pct": round(float(np.mean([b["reduction_ratio_pct"] for b in blocking_stats_all])), 2),
            "avg_pairs_completeness_pct": round(float(np.mean([b["pairs_completeness_pct"] for b in blocking_stats_all])), 2),
        }
    }

    return summary_report
