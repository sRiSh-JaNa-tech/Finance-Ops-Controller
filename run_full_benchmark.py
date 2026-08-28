"""Comprehensive Benchmark Runner for Prototype 4.

Runs full evaluation across all 15 benchmark scenarios, compares systems (Ablation),
and exports benchmark metrics to benchmark_results.json.
"""

import json
import os
import logging
import argparse
from finance_ops.ledger.journal import LedgerRepository
from finance_ops.benchmark.runner import run_benchmark

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def print_banner(title: str) -> None:
    w = 88
    print("\n" + "=" * w)
    print(f"  {title}")
    print("=" * w)


def run_full_benchmark():
    parser = argparse.ArgumentParser(description="AI Finance Controller Benchmark")
    parser.add_argument("--mode", type=str, choices=["offline", "live", "full"], default="offline", 
                        help="offline: Deterministic mock | live: 100-case LLM | full: 300-case LLM")
    args = parser.parse_args()
    
    if args.mode == "full":
        seeds = [42, 101, 202]
        eval_mode = "live"
        cases_per_batch = 100
        os.environ["RUN_LIVE_LLM"] = "1"
        print_banner("LIVE GEMINI EVALUATION (300 cases)")
    elif args.mode == "live":
        seeds = [42]
        eval_mode = "live"
        cases_per_batch = 100
        os.environ["RUN_LIVE_LLM"] = "1"
        print_banner("LIVE GEMINI EVALUATION (100 cases)")
    else:
        seeds = [42, 101, 202]
        eval_mode = "offline"
        cases_per_batch = 100
        os.environ["RUN_LIVE_LLM"] = "0"
        print_banner("OFFLINE ARCHITECTURE TEST (No LLM Measurement)")

    print("[*] Architecture: Evidence-Grounded Autonomous Investigation Agent")
    print("[*] Gemini Vertex AI Native Function Calling + Deterministic Fallback Engine\n")

    print(f"[METRIC: Throughput] Processing {len(seeds)} batches of {cases_per_batch} complex synthetic transactions...")
    
    # Initialize ledger repository
    ledger_repo = LedgerRepository()
    
    results = run_benchmark(
        seeds=seeds,
        cases_per_seed=cases_per_batch,
        run_ablations=True,
        mode=eval_mode
    )

    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\n[*] Benchmark complete. Results written to benchmark_results.json")

if __name__ == "__main__":
    run_full_benchmark()
