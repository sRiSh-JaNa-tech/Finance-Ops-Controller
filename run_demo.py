"""CLI Entry point for AI Finance Controller: Run Benchmark or Launch Dashboard."""

import argparse
import sys
from finance_ops.benchmark.runner import run_benchmark
from finance_ops.ui.server import app, initialize_demo_state


def main():
    parser = argparse.ArgumentParser(description="AI Finance Controller CLI")
    parser.add_argument("--mode", choices=["benchmark", "dashboard"], default="benchmark", help="Mode to execute")
    parser.add_argument("--cases", type=int, default=5, help="Number of test cases per seed (default: 5)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 101, 202], help="Random seeds for benchmark")
    parser.add_argument("--port", type=int, default=5000, help="Web dashboard port")

    args = parser.parse_args()

    if args.mode == "benchmark":
        print(f"[+] Running repeated-seed benchmark over seeds {args.seeds} ({args.cases} cases/seed)...")
        results = run_benchmark(seeds=args.seeds, cases_per_seed=args.cases)
        
        print("\n" + "="*85)
        print("AI FINANCE CONTROLLER: BENCHMARK SUMMARY")
        print("="*100)
        print(f"{'System':<24} | {'Match F1':<10} | {'Triage F1':<10} | {'False Match %':<14} | {'Cause Diag %':<14} | {'Cost Utility':<12}")
        print("-" * 100)
        
        for sys_name, m in results["systems"].items():
            match_f1_str = f"{m.get('match_f1_score', 0)*100:.1f}%"
            triage_f1_str = f"{m.get('triage_f1_score', 0)*100:.1f}%"
            fmr_str = f"{m.get('false_match_rate', 0)*100:.1f}%"
            diag_str = f"{m.get('cause_diagnosis_accuracy', 0)*100:.1f}%"
            util_str = f"${m.get('cost_weighted_utility', 0):.2f}"
            print(f"{sys_name:<24} | {match_f1_str:<10} | {triage_f1_str:<10} | {fmr_str:<14} | {diag_str:<14} | {util_str:<12}")
        print("="*100)
        
        if "blocking_performance" in results:
            bp = results["blocking_performance"]
            print("\n[BLOCKING ENGINE METRICS (Phase 1: Learning Blocking Schemes)]")
            print(f"Candidate Reduction Ratio: {bp.get('avg_reduction_ratio_pct', 0):.2f}%")
            print(f"Pairs Completeness (Recall): {bp.get('avg_pairs_completeness_pct', 0):.2f}%")
            print("="*85 + "\n")

    elif args.mode == "dashboard":
        print(f"[+] Mode: CACHED (Dashboard running on cached benchmark results)")
        print(f"[+] Initializing and launching AI Finance Controller Dashboard on http://127.0.0.1:{args.port} ...")
        initialize_demo_state(n_cases=args.cases, seed=args.seeds[0])
        app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
