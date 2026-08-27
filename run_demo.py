"""CLI Entry point for AI Finance Controller: Run Benchmark or Launch Dashboard."""

import argparse
import sys
from finance_ops.benchmark.runner import run_benchmark
from finance_ops.ui.server import app, initialize_demo_state


def main():
    parser = argparse.ArgumentParser(description="AI Finance Controller CLI")
    parser.add_argument("--mode", choices=["benchmark", "dashboard"], default="benchmark", help="Mode to execute")
    parser.add_argument("--cases", type=int, default=30, help="Number of test cases per seed")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 101, 202], help="Random seeds for benchmark")
    parser.add_argument("--port", type=int, default=5000, help="Web dashboard port")

    args = parser.parse_args()

    if args.mode == "benchmark":
        print(f"[+] Running repeated-seed benchmark over seeds {args.seeds} ({args.cases} cases/seed)...")
        results = run_benchmark(seeds=args.seeds, cases_per_seed=args.cases)
        
        print("\n" + "="*85)
        print("AI FINANCE CONTROLLER: BENCHMARK SUMMARY")
        print("="*85)
        print(f"{'System':<24} | {'F1 (95% CI)':<18} | {'False Match %':<14} | {'Cause Diag %':<14} | {'Cost Utility':<12}")
        print("-"*85)
        
        for sys_name, m in results["systems"].items():
            f1_str = f"{m['f1_score_mean']*100:.1f}% [{m['f1_score_ci95'][0]*100:.1f}-{m['f1_score_ci95'][1]*100:.1f}]"
            fmr_str = f"{m['false_match_rate_mean']*100:.1f}%"
            diag_str = f"{m['cause_diagnosis_accuracy_mean']*100:.1f}%"
            util_str = f"${m['cost_weighted_utility_mean']:.2f}"
            print(f"{sys_name:<24} | {f1_str:<18} | {fmr_str:<14} | {diag_str:<14} | {util_str:<12}")
        print("="*85 + "\n")

    elif args.mode == "dashboard":
        print(f"[+] Initializing and launching AI Finance Controller Dashboard on http://127.0.0.1:{args.port} ...")
        initialize_demo_state(n_cases=args.cases, seed=args.seeds[0])
        app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
