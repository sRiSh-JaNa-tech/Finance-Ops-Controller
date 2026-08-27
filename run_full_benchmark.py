"""Comprehensive Benchmark Runner for Prototype 3.

Runs full evaluation across all 15 benchmark scenarios, compares systems,
and exports benchmark metrics to benchmark_results.json.
"""

import json
from finance_ops.benchmark.runner import run_benchmark


def print_banner(title: str) -> None:
    w = 88
    print("\n" + "=" * w)
    print(f"  {title}")
    print("=" * w)


def run_full_benchmark():
    print_banner("AI FINANCE CONTROLLER — PROTOTYPE 3 FULL RESEARCH BENCHMARK")
    print("[*] Track 04: Run the books and the cash position")
    print("[*] Architecture: Evidence-Grounded Autonomous Investigation Agent")
    print("[*] Gemini Vertex AI Native Function Calling + Deterministic Fallback Engine\n")

    cases_per_batch = 100
    print(f"[METRIC: Throughput] Processing single batch of {cases_per_batch} complex synthetic transactions...")
    
    results = run_benchmark(
        seeds=[42],
        cases_per_seed=cases_per_batch,
        run_ablations=False
    )

    # We use Prototype3 for the main reporting
    p3_stats = results["systems"]["Prototype3_GeminiVertexAgent"]
    rule_stats = results["systems"]["RuleMatcher"]

    # Calculate raw counts from the first seed
    exceptions = results.get("honest_exception_list", [])
    
    # We must calculate TP, FP, FN, UNCERTAIN
    # Since metrics.py doesn't return raw counts in the summary, we can approximate or if metrics engine was modified:
    # We can calculate the match rate and counts from the honest exception list.
    # Actually, the runner summary doesn't expose raw counts directly, but we have recall and precision.
    # We can just output the metrics.
    
    # 8.5/10 Template
    print(f"\nBATCH: {cases_per_batch} cases")
    print("-" * 48)
    
    # Formatting metrics
    match_rate = p3_stats['automation_rate_pct']
    prec = p3_stats['precision'] * 100
    rec = p3_stats['recall'] * 100
    f1 = p3_stats['f1_score'] * 100
    fmr = p3_stats['false_match_rate'] * 100

    print(f"MATCH RATE: {match_rate:.1f}%")
    print(f"PRECISION: {prec:.1f}%, RECALL: {rec:.1f}%, F1: {f1:.1f}%, FALSE MATCH RATE: {fmr:.1f}%")

    print(f"\nTHROUGHPUT:")
    print(f"  {cases_per_batch} cases")
    print(f"  {p3_stats.get('throughput_cases_per_sec', 0)} cases/sec")
    print(f"  p95 latency: {p3_stats.get('p95_latency_ms', 0)} ms")

    baseline_recall = rule_stats['recall'] * 100
    improvement = rec - baseline_recall

    print(f"\nAI CONTRIBUTION:")
    print(f"  LLM-investigated: {p3_stats.get('llm_investigated', 0)}")
    print(f"  deterministic fast-path: {p3_stats.get('deterministic_fast_path', 0)}")
    print(f"  LLM improved recall by: +{improvement:.1f}% vs baseline")

    print(f"\nEXCEPTIONS:")
    shown = 0
    for e in exceptions[:15]:
        shown += 1
        cid = e.get("case_id", "UNKNOWN")
        reason = e.get("reason", "UNKNOWN")
        expl = e.get("explanation", "").strip()
        print(f"  {cid} -> [{reason}] \"{expl}\"")

    if len(exceptions) > 15:
        print(f"  ... and {len(exceptions) - 15} more.")

    # Section 4: Export JSON
    output_path = "benchmark_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Full benchmark metrics successfully saved to {output_path}")


if __name__ == "__main__":
    run_full_benchmark()

